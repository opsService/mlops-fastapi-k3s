# mlops-fastapi-app/app/routers/train.py (수정본)
import asyncio
import json
import logging
import os
import typing as tp

# MLflow 클라이언트 임포트
import mlflow
from app.core.config import settings

# Kubernetes 클라이언트 임포트
from app.core.k8s_client import k8s_client

# Task Manager 임포트
from app.core.task_manager import (  # Pydantic 모델도 임포트
    active_ml_tasks,
    send_model_registration_callback,
    send_status_callback,
)
from app.schemas.callbacks.models_callback import (
    RegisterModelCallback,
    UpdateTaskStatusCallback,
)
from app.schemas.common.models_base import ProgressMetrics
from app.schemas.train.train_requests import CreateTrainJobRequest, Hyperparameters
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from ulid import ULID

logger = logging.getLogger(__name__)


def verify_internal_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    """내부 API 키를 검증합니다."""
    if x_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal API Key"
        )
    return True


router = APIRouter(dependencies=[Depends(verify_internal_api_key)])

# --- MLflow 클라이언트 초기화 ---
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient()


# --- 백그라운드 Job 모니터링 함수 ---
async def _monitor_train_job(task_id: str, k8s_job_name: str, mlflow_run_id: str):
    """
    Kubernetes Job과 MLflow Run의 상태를 주기적으로 모니터링하고 Spring Boot에 콜백합니다.
    """
    logger.info(
        f"Starting background monitoring for Task ID: {task_id}, K8s Job: {k8s_job_name}, MLflow Run: {mlflow_run_id}"
    )
    task_info = active_ml_tasks.get(task_id)
    if not task_info:
        logger.warning(
            f"Monitor started for non-existent task_id: {task_id}. Exiting monitor."
        )
        return

    # 첫 상태를 PENDING으로 콜백
    send_status_callback(task_id, "PENDING", mlflow_run_id)

    max_attempts = 120  # 약 10분 (5초 * 120)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        status_message = "RUNNING"
        error_message = None
        current_metrics = {}
        log_snippet = ""
        k8s_job_status = None
        k8s_pods = []

        try:
            # 1. Kubernetes Job 상태 조회
            k8s_job_status = k8s_client.get_job_status(k8s_job_name)
            if k8s_job_status:
                if k8s_job_status.succeeded:
                    status_message = "SUCCEEDED"
                    logger.info(
                        f"Train Job {k8s_job_name} for Task {task_id} SUCCEEDED."
                    )
                elif k8s_job_status.failed:
                    status_message = "FAILED"
                    error_message = f"Kubernetes Job failed. Reason: {k8s_job_status.conditions[0].message if k8s_job_status.conditions else 'Unknown'}"
                    logger.error(
                        f"Train Job {k8s_job_name} for Task {task_id} FAILED: {error_message}"
                    )
                else:
                    status_message = "RUNNING"
                    # 실행 중인 Pod의 로그 스니펫 가져오기
                    k8s_pods = k8s_client.get_pods_for_job(k8s_job_name)
                    if k8s_pods:
                        # 최신 Pod (또는 RUNNING 상태의 Pod) 로그 가져오기
                        running_pods = [
                            p for p in k8s_pods if p.status.phase == "Running"
                        ]
                        pod_to_log = running_pods[0] if running_pods else k8s_pods[0]
                        try:
                            log_snippet = k8s_client.get_pod_logs(
                                pod_to_log.metadata.name, tail_lines=20
                            )
                            if len(log_snippet) > 1000:
                                log_snippet = log_snippet[-1000:] + "\n... (truncated)"
                        except Exception as e:
                            logger.warning(
                                f"Could not get logs for pod {pod_to_log.metadata.name}: {e}"
                            )
                            log_snippet = f"Error fetching logs: {e}"
            else:
                status_message = "UNKNOWN"
                logger.warning(
                    f"K8s Job {k8s_job_name} status not found for Task {task_id}. May be pending or deleted."
                )

            # 2. MLflow Run 메트릭 조회 (RUNNING 상태일 때만 의미 있음)
            if status_message == "RUNNING":
                try:
                    run_data = mlflow_client.get_run(mlflow_run_id).data
                    current_metrics = {k: v for k, v in run_data.metrics.items()}
                except Exception as e:
                    logger.warning(
                        f"Could not fetch MLflow metrics for run {mlflow_run_id}: {e}"
                    )
                    current_metrics["error_fetching_metrics"] = str(e)

        except Exception as e:
            status_message = "FAILED"
            error_message = f"Monitoring internal error: {e}"
            logger.error(
                f"Error during monitoring for Task {task_id}: {e}", exc_info=True
            )

        # Spring Boot에 콜백
        callback_payload = UpdateTaskStatusCallback(
            status=status_message,
            mlflowRunId=mlflow_run_id,
            progress=ProgressMetrics(metrics=current_metrics),
            logSnippet=log_snippet,
            errorMessage=error_message,
        )
        send_status_callback(task_id, status_message, mlflow_run_id, callback_payload)

        # Job 완료 또는 실패 시 모니터링 종료
        if status_message in ["SUCCEEDED", "FAILED", "STOPPED", "UNKNOWN"]:
            break

        await asyncio.sleep(5)

    # 최종 상태 확인 후 정리
    final_status = active_ml_tasks.get(task_id, {}).get("status")
    if final_status == "SUCCEEDED":
        # 학습 완료 모델 정보 등록 콜백 (MLflow Run에서 최종 모델 정보 가져와서 전달)
        try:
            final_run = mlflow_client.get_run(mlflow_run_id)
            final_metrics = {k: v for k, v in final_run.data.metrics.items()}

            # MLflow에 저장된 모델 아티팩트 경로 (MinIO)
            artifact_uri = final_run.info.artifact_uri
            model_artifact_path_in_run = (
                "model"  # train_cpu.py에서 log_model 시 사용한 이름
            )
            model_s3_path = f"{artifact_uri}/{model_artifact_path_in_run}"

            register_payload = RegisterModelCallback(
                taskId=task_id,
                userId=task_info["user_id"],
                modelName=task_info["custom_model_name"],
                modelType="CUSTOM_TRAINED",
                modelFilePath=model_s3_path,
                modelSizeMB=0.0,  # 실제 모델 파일 크기를 MLflow 아티팩트 정보에서 가져오거나 추정
                version="1.0.0",
                hyperparameters=task_info["hyperparameters"],
                performanceMetrics=final_metrics,
                mlflowRunId=mlflow_run_id,
            )
            send_model_registration_callback(register_payload)
        except Exception as e:
            logger.error(
                f"Failed to send model registration callback for Task {task_id}: {e}",
                exc_info=True,
            )
    elif final_status == "FAILED":
        logger.error(f"Task {task_id} FAILED, no model registration callback sent.")

    # 모니터링 완료 후 딕셔너리에서 해당 Task 정보 삭제 (선택 사항, 영구 저장은 DB)
    if task_id in active_ml_tasks:
        del active_ml_tasks[task_id]
        logger.info(
            f"Monitoring ended for Task ID: {task_id}. Removed from active_ml_tasks."
        )


# --- FastAPI 엔드포인트 정의 ---


@router.post("/job", status_code=status.HTTP_201_CREATED)
async def create_train_job(
    request_body: CreateTrainJobRequest, background_tasks: BackgroundTasks
):
    """
    1.2. Train Task Job 생성 요청을 처리합니다.
    새로운 MLflow Run을 시작하고, Kubernetes Job을 생성하여 학습을 시작합니다.
    """
    task_id = request_body.taskId
    custom_model_name = request_body.customModelName

    # 1. MLflow Run 시작 (FastAPI가 Run ID 생성 책임)
    experiment_name = (
        request_body.experimentName
    )  # Task ID 기반의 고유한 Experiment 이름
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"run-{task_id}") as run:
        mlflow_run_id = run.info.run_id
        logger.info(f"New MLflow Run started: {mlflow_run_id} for Task ID: {task_id}")

        # MLflow Run에 Spring Boot의 Task ID와 User ID를 태그로 기록
        mlflow.set_tag("task_id", task_id)
        mlflow.set_tag
        mlflow.set_tag("user_id", request_body.userId)
        mlflow.set_tag("custom_model_name", custom_model_name)
        mlflow.log_params(request_body.hyperparameters.model_dump())

    # Kubernetes Job 이름 생성
    k8s_job_name = f"train-job-{str(ULID()).lower()}"

    # 2. Kubernetes Job 생성 요청
    try:
        k8s_client.create_train_job(
            job_name=k8s_job_name,
            image=request_body.trainerImage,
            train_script_path=request_body.trainScriptPath,
            mlflow_run_id=mlflow_run_id,
            initial_model_path=request_body.initialModelFilePath,
            dataset_path=request_body.datasetPath,
            hyperparameters=request_body.hyperparameters.model_dump(),
            use_gpu=request_body.useGpu,
        )
    except Exception as e:
        mlflow_client.set_terminated(mlflow_run_id, "FAILED")
        logger.error(
            f"Failed to create K8s Train Job for Task {task_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "K8S_JOB_CREATION_FAILED",
                "message": f"FastAPI failed to create Kubernetes Job: {e}",
            },
        )

    # 3. Task 정보 저장 및 백그라운드 모니터링 시작
    active_ml_tasks[task_id] = {
        "k8s_resource_type": "Job",  # Job 타입임을 명시
        "k8s_resource_name": k8s_job_name,
        "mlflow_run_id": mlflow_run_id,
        "status": "PENDING",
        "user_id": request_body.userId,
        "custom_model_name": custom_model_name,
        "hyperparameters": request_body.hyperparameters.model_dump(),
    }
    background_tasks.add_task(_monitor_train_job, task_id, k8s_job_name, mlflow_run_id)

    return {
        "message": "Kubernetes Train Job creation initiated.",
        "mlflowRunId": mlflow_run_id,
    }


# --- 공통 Task 상태 및 로그 조회 엔드포인트는 train.py에 유지 (또는 별도 common 라우터로 분리 가능) ---
@router.get("/{taskId}/status")
async def get_train_task_status(taskId: str):
    """
    2.1. Task 상태 업데이트 콜백에서 전송될 정보와 유사하게,
    현재 학습 Task의 상태 및 진행 상황을 조회합니다.
    """
    task_info = active_ml_tasks.get(taskId)
    if not task_info or task_info.get("k8s_resource_type") != "Job":  # 학습 Task만 조회
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "errorCode": "TASK_NOT_FOUND",
                "message": f"Train Task with ID '{taskId}' not found.",
            },
        )

    k8s_job_name = task_info["k8s_resource_name"]
    mlflow_run_id = task_info["mlflow_run_id"]
    current_status = task_info["status"]

    status_message = current_status
    current_metrics = {}
    error_message = None

    try:
        k8s_job_status = k8s_client.get_job_status(k8s_job_name)
        if k8s_job_status:
            if k8s_job_status.succeeded:
                status_message = "SUCCEEDED"
            elif k8s_job_status.failed:
                status_message = "FAILED"
                error_message = f"Kubernetes Job failed. Reason: {k8s_job_status.conditions[0].message if k8s_job_status.conditions else 'Unknown'}"
            else:
                status_message = "RUNNING"
        else:
            status_message = "PENDING" if current_status == "PENDING" else "UNKNOWN"

        if status_message == "RUNNING" or status_message == "SUCCEEDED":
            try:
                run_data = mlflow_client.get_run(mlflow_run_id).data
                current_metrics = {k: v for k, v in run_data.metrics.items()}
            except Exception as e:
                logger.warning(
                    f"Could not fetch MLflow metrics for run {mlflow_run_id} during status check: {e}"
                )
                current_metrics["error_fetching_metrics"] = str(e)

    except Exception as e:
        logger.error(f"Error getting status for Task {taskId}: {e}", exc_info=True)
        status_message = "ERROR"
        error_message = f"Internal error during status retrieval: {e}"

    response_progress = ProgressMetrics(metrics=current_metrics)
    if status_message in [
        "SUCCEEDED",
        "FAILED",
        "STOPPED",
    ]:  # 최종 상태일 때만 ProgressMetrics를 최종 값으로 설정
        response_progress.epoch = current_metrics.get(
            "current_epoch"
        )  # train_cpu.py에서 로깅한 current_epoch
        response_progress.loss = current_metrics.get(
            "final_test_loss"
        )  # 최종 테스트 손실
        # 기타 최종 메트릭들도 여기에 포함

    return UpdateTaskStatusCallback(
        status=status_message,
        mlflowRunId=mlflow_run_id,
        progress=response_progress,
        errorMessage=error_message,
    )


# --- Task 삭제 엔드포인트 (Train/Inference 공통으로 확장) ---
# 이 엔드포인트는 routers/train.py에 있지만, routers/inference.py에서도 동일한 로직을 호출할 수 있도록
# 별도의 공통 라우터 파일(예: routers/common_tasks.py)로 분리하는 것이 더 깔끔합니다.
# 현재는 train.py에 임시로 두겠습니다.


@router.delete("/{taskId}")
async def delete_task_resources(taskId: str):
    """
    1.3. Task 리소스 중지/삭제 요청을 처리합니다.
    해당 Task와 관련된 Kubernetes Job (학습) 또는 Deployment/Service/Ingress (추론)를 삭제합니다.
    """
    task_info = active_ml_tasks.get(taskId)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "errorCode": "K8S_RESOURCES_NOT_FOUND",
                "message": f"Kubernetes resources for task ID '{taskId}' not found or already deleted.",
            },
        )

    k8s_resource_type = task_info["k8s_resource_type"]
    k8s_resource_name = task_info["k8s_resource_name"]
    mlflow_run_id = task_info.get("mlflow_run_id")  # 학습 Task에만 있을 수 있음

    try:
        if k8s_resource_type == "Job":
            # 학습 Job 삭제
            k8s_client.delete_job(k8s_resource_name)
            logger.info(f"Deleted K8s Job {k8s_resource_name} for Task {taskId}")
            # MLflow Run 상태 업데이트 (옵션)
            if mlflow_run_id:
                mlflow_client.set_terminated(mlflow_run_id, "KILLED")
                logger.info(f"Set MLflow Run {mlflow_run_id} status to KILLED.")

            send_status_callback(
                taskId,
                "STOPPED",
                mlflow_run_id,
                UpdateTaskStatusCallback(
                    status="STOPPED",
                    mlflowRunId=mlflow_run_id,
                    logSnippet=f"Training job {k8s_resource_name} for task {taskId} was stopped.",
                    errorMessage=None,
                ),
            )

        elif k8s_resource_type == "Deployment":
            # 추론 Deployment, Service, Ingress 삭제
            # Ingress가 있다면 먼저 삭제
            ingress_name = (
                f"inference-ingress-{taskId[:8].lower()}"  # 명세에 따라 이름 규칙 필요
            )
            try:
                k8s_client.delete_ingress(ingress_name)
                logger.info(f"Deleted K8s Ingress {ingress_name} for Task {taskId}")
            except Exception as e:
                logger.warning(f"Could not delete Ingress {ingress_name}: {e}")

            # Service 삭제
            service_name = (
                f"inference-service-{taskId[:8].lower()}"  # 명세에 따라 이름 규칙 필요
            )
            try:
                k8s_client.delete_service(service_name)
                logger.info(f"Deleted K8s Service {service_name} for Task {taskId}")
            except Exception as e:
                logger.warning(f"Could not delete Service {service_name}: {e}")

            # Deployment 삭제
            k8s_client.delete_deployment(k8s_resource_name)
            logger.info(f"Deleted K8s Deployment {k8s_resource_name} for Task {taskId}")

            send_status_callback(
                taskId,
                "STOPPED",
                None,
                UpdateTaskStatusCallback(
                    status="STOPPED",
                    logSnippet=f"Inference deployment {k8s_resource_name} for task {taskId} was stopped.",
                    errorMessage=None,
                ),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "errorCode": "UNKNOWN_RESOURCE_TYPE",
                    "message": f"Unknown resource type for task ID '{taskId}': {k8s_resource_type}",
                },
            )

        # 활성 태스크 목록에서 제거
        if taskId in active_ml_tasks:
            del active_ml_tasks[taskId]

        return {
            "message": f"Kubernetes resources for task '{taskId}' deletion initiated."
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            f"Failed to delete K8s resources for Task {taskId}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "K8S_RESOURCE_DELETION_FAILED",
                "message": f"FastAPI failed to delete Kubernetes resources: {e}",
            },
        )


# --- Task 로그 조회 엔드포인트 (Train/Inference 공통) ---
@router.get("/{taskId}/logs")
async def get_task_logs(taskId: str, tail: tp.Optional[int] = None):
    """
    3.1. Task 컨테이너 로그 조회를 처리합니다.
    해당 Task와 관련된 Kubernetes Pod의 로그를 반환합니다.
    """
    task_info = active_ml_tasks.get(taskId)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "errorCode": "CONTAINER_NOT_FOUND",
                "message": f"Container for task ID '{taskId}' not found.",
            },
        )

    k8s_resource_type = task_info["k8s_resource_type"]
    k8s_resource_name = task_info["k8s_resource_name"]

    try:
        pods = []
        if k8s_resource_type == "Job":
            pods = k8s_client.get_pods_for_job(k8s_resource_name)
        elif k8s_resource_type == "Deployment":
            # Deployment의 경우, Deployment가 관리하는 Pod들을 레이블 셀렉터로 찾습니다.
            # Deployment의 metadata.labels를 참조하여 Pods를 찾습니다.
            # 여기서는 Deployment 이름과 같은 레이블을 가진 Pod를 찾는다고 가정합니다.
            pods = k8s_client.core_v1.list_namespaced_pod(
                namespace=k8s_client.namespace,
                label_selector=f"app={k8s_resource_name}",  # Deployment의 app 레이블 사용
            ).items
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "errorCode": "UNKNOWN_RESOURCE_TYPE",
                    "message": f"Cannot retrieve logs for unknown resource type: {k8s_resource_type}",
                },
            )

        if not pods:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "errorCode": "CONTAINER_NOT_FOUND",
                    "message": f"No running pods found for task ID '{taskId}' (Resource: {k8s_resource_name}).",
                },
            )

        target_pod = None
        running_pods = [p for p in pods if p.status.phase == "Running"]
        if running_pods:
            target_pod = running_pods[0]
        else:
            pods.sort(key=lambda p: p.metadata.creation_timestamp, reverse=True)
            target_pod = pods[0] if pods else None

        if not target_pod:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "errorCode": "CONTAINER_NOT_FOUND",
                    "message": f"No suitable pod found for task ID '{taskId}' (Resource: {k8s_resource_name}).",
                },
            )

        logs = k8s_client.get_pod_logs(target_pod.metadata.name, tail_lines=tail)
        return {"taskId": taskId, "logs": logs}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to retrieve logs for Task {taskId}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "K8S_LOG_RETRIEVAL_FAILED",
                "message": f"FastAPI failed to retrieve logs from Kubernetes: {e}",
            },
        )
