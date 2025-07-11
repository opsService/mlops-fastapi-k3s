# mlops-fastapi-app/app/routers/inference.py
import asyncio
import logging
import os
from typing import Dict, List, Optional

import requests

# Kubernetes 클라이언트 임포트
from app.core.k8s_client import k8s_client

# Task Manager 임포트 (여기서 공통 Pydantic 모델도 임포트)
from app.core.task_manager import (
    UpdateTaskStatusCallback,
    active_ml_tasks,
    send_status_callback,
)
from app.schemas.inference.requests_inference import (
    DeployInferenceRequest,
    PredictRequest,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

# 로거 설정
logger = logging.getLogger(__name__)

# FastAPI 내부 인증을 위한 API Key (환경 변수에서 가져옴)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "your-super-secret-internal-api-key")


def get_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    """내부 API 키를 검증합니다."""
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal API Key"
        )
    return True


router = APIRouter()


async def _monitor_inference_deployment(
    task_id: str,
    deployment_name: str,
    mlflow_run_id: Optional[str],
    inference_api_endpoint: str,
):
    logger.info(
        f"추론 태스크 ID: {task_id}, 배포: {deployment_name}에 대한 백그라운드 모니터링 시작"
    )
    task_info = active_ml_tasks.get(task_id)
    if not task_info:
        logger.warning(
            f"존재하지 않는 추론 태스크 ID: {task_id}에 대한 모니터링이 시작되었습니다. 모니터링을 종료합니다."
        )
        return

    service_name_for_monitor = task_info.get("service_name")
    if not service_name_for_monitor:
        logger.error(
            f"태스크 {task_id}: active_ml_tasks에 service_name이 없습니다. 모니터링을 종료합니다."
        )
        await _rollback_resources(task_id, deployment_name, None, None)
        return

    send_status_callback(
        task_id,
        "PENDING",
        mlflow_run_id,
        UpdateTaskStatusCallback(
            status="PENDING",
            mlflowRunId=mlflow_run_id,
            inferenceApiEndpoint=inference_api_endpoint,
        ),
    )

    max_attempts = 120
    retry_delay = 5
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        status_message = "RUNNING"
        error_message = None

        try:
            # Deployment 상태 조회
            logger.debug(
                f"Task {task_id}: Deployment {deployment_name} 상태 조회 시도 (시도 {attempts}/{max_attempts})."
            )
            deploy_status = k8s_client.apps_v1.read_namespaced_deployment_status(
                name=deployment_name, namespace=k8s_client.namespace
            )

            # ⭐ 핵심 수정: deploy_status.status와 deploy_status.spec이 모두 존재하는지 먼저 확인
            if deploy_status.status is not None and deploy_status.spec is not None:
                # ⭐ 추가된 디버그 로깅 (이전과 동일)
                logger.debug(
                    f"Task {task_id}: Deployment {deployment_name} 상태 확인 (시도 {attempts}/{max_attempts})."
                )
                logger.debug(
                    f"  Ready Replicas: {deploy_status.status.ready_replicas}, Available Replicas: {deploy_status.status.available_replicas}, Replicas: {deploy_status.status.replicas}"
                )
                logger.debug(f"  Spec Replicas: {deploy_status.spec.replicas}")
                if deploy_status.status.conditions:
                    for condition in deploy_status.status.conditions:
                        logger.debug(
                            f"  Condition: {condition.type}={condition.status}, Reason: {condition.reason}, Message: {condition.message}"
                        )

                # ready_replicas와 spec.replicas가 None일 경우 0으로 처리하여 비교
                ready_replicas = (
                    deploy_status.status.ready_replicas
                    if deploy_status.status.ready_replicas is not None
                    else 0
                )
                spec_replicas = (
                    deploy_status.spec.replicas
                    if deploy_status.spec.replicas is not None
                    else 0
                )

                if ready_replicas >= spec_replicas:
                    # Deployment가 Ready 상태이고, 추가적으로 헬스 체크를 수행
                    try:
                        health_check_url = f"http://{service_name_for_monitor}.{k8s_client.namespace}.svc.cluster.local:8000/health"
                        logger.info(
                            f"태스크 {task_id}: 헬스 체크 URL: {health_check_url}"
                        )
                        health_response = requests.get(health_check_url, timeout=3)
                        health_response.raise_for_status()
                        if health_response.json().get("status") == "healthy":
                            status_message = "SUCCEEDED"
                            logger.info(
                                f"추론 배포 {deployment_name} (태스크 {task_id}) 성공: Ready 및 Healthy."
                            )

                            # 모델 로드 요청
                            try:
                                load_model_url = f"http://{service_name_for_monitor}.{k8s_client.namespace}.svc.cluster.local:8000/load_model"
                                logger.info(
                                    f"태스크 {task_id}: 추론 서버에 모델 로드 요청 전송 시도: {load_model_url} (run_id: {mlflow_run_id})"
                                )
                                load_model_response = requests.post(
                                    load_model_url,
                                    json={"run_id": mlflow_run_id},
                                    timeout=30,
                                )
                                load_model_response.raise_for_status()
                                logger.info(
                                    f"모델 로드 요청이 추론 서버로 성공적으로 전송되었습니다 (run_id {mlflow_run_id}): {load_model_response.json()}"
                                )
                            except requests.exceptions.RequestException as load_e:
                                logger.error(
                                    f"추론 서버에 모델 로드 요청 실패: {load_e}",
                                    exc_info=True,
                                )
                                status_message = "FAILED"
                                error_message = f"추론 서버가 시작되었지만 모델 로드에 실패했습니다: {load_e}"

                        else:
                            status_message = "RUNNING"
                            logger.info(
                                f"추론 배포 {deployment_name} (태스크 {task_id}) Ready 상태이지만 아직 Healthy하지 않습니다. 헬스 상태: {health_response.json().get('status')}"
                            )
                    except requests.exceptions.RequestException as health_e:
                        status_message = "RUNNING"
                        logger.warning(
                            f"추론 서버 헬스 체크 실패 (배포 {deployment_name}, URL: {health_check_url}): {health_e}",
                            exc_info=True,
                        )
                else:  # is_ready가 False인 경우 (즉, ready_replicas < spec_replicas)
                    status_message = "RUNNING"
                    logger.info(
                        f"추론 배포 {deployment_name} (태스크 {task_id}) 아직 준비되지 않았지만 진행 중. (Ready Replicas: {ready_replicas}, Spec Replicas: {spec_replicas})"
                    )

            # deploy_status.status 또는 deploy_status.spec이 None인 경우 처리
            else:
                logger.debug(
                    "  Deployment status or spec object is None. Still waiting for full status."
                )
                status_message = "RUNNING"  # 아직 상태 정보가 불완전하므로 계속 대기

        except Exception as e:
            status_message = "FAILED"
            error_message = f"배포 모니터링 중 내부 오류: {e}"
            logger.error(
                f"태스크 {task_id}의 배포 모니터링 중 오류: {e}", exc_info=True
            )

        send_status_callback(
            task_id,
            status_message,
            mlflow_run_id,
            UpdateTaskStatusCallback(
                status=status_message,
                mlflowRunId=mlflow_run_id,
                errorMessage=error_message,
                inferenceApiEndpoint=(
                    inference_api_endpoint if status_message == "SUCCEEDED" else None
                ),
            ),
        )

        if status_message in ["SUCCEEDED", "FAILED", "STOPPED", "UNKNOWN"]:
            break

        await asyncio.sleep(retry_delay)

    if task_id in active_ml_tasks and active_ml_tasks[task_id]["status"] in [
        "SUCCEEDED",
        "FAILED",
        "STOPPED",
    ]:
        if active_ml_tasks[task_id]["status"] in ["FAILED", "STOPPED"]:
            await _rollback_resources(
                task_id,
                deployment_name,
                service_name_for_monitor,
                task_info.get("ingress_name"),
            )
            del active_ml_tasks[task_id]
            logger.info(
                f"추론 태스크 ID: {task_id}에 대한 모니터링이 종료되었습니다. active_ml_tasks에서 제거되었습니다."
            )
        else:
            logger.info(
                f"추론 태스크 ID: {task_id} 성공. 예측을 위해 active_ml_tasks에 유지합니다."
            )


async def _rollback_resources(
    task_id: str,
    deployment_name: str,
    service_name: Optional[str],
    ingress_name: Optional[str],
):
    """
    배포 실패 시 생성된 Kubernetes 리소스를 롤백(삭제)합니다.
    """
    logger.info(f"태스크 {task_id}에 대한 리소스 롤백 중...")
    try:
        # Ingress 삭제
        if ingress_name:
            try:
                await asyncio.to_thread(k8s_client.delete_ingress, ingress_name)
                logger.info(
                    f"태스크 {task_id}에 대한 Ingress {ingress_name} 삭제 완료."
                )
            except Exception as e:
                logger.error(
                    f"태스크 {task_id}에 대한 Ingress {ingress_name} 롤백 실패: {e}",
                    exc_info=True,
                )

        # Service 삭제
        if service_name:  # service_name이 None일 수 있으므로 체크
            try:
                await asyncio.to_thread(k8s_client.delete_service, service_name)
                logger.info(f"태스크 {task_id}에 대한 서비스 {service_name} 삭제 완료.")
            except Exception as e:
                logger.error(
                    f"태스크 {task_id}에 대한 서비스 {service_name} 롤백 실패: {e}",
                    exc_info=True,
                )

        # Deployment 삭제
        try:
            await asyncio.to_thread(k8s_client.delete_deployment, deployment_name)
            logger.info(f"태스크 {task_id}에 대한 배포 {deployment_name} 삭제 완료.")
        except Exception as e:
            logger.error(
                f"태스크 {task_id}에 대한 배포 {deployment_name} 롤백 실패: {e}",
                exc_info=True,
            )

        logger.info(f"태스크 {task_id}에 대한 리소스 롤백 성공.")
    except Exception as e:
        logger.error(
            f"태스크 {task_id}에 대한 리소스 롤백 중 예상치 못한 오류: {e}",
            exc_info=True,
        )


@router.post("/deploy", status_code=status.HTTP_201_CREATED)
async def deploy_inference_task(
    request_body: DeployInferenceRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Depends(get_api_key),
):
    """
    새로운 추론 태스크를 배포합니다.
    """
    task_id = request_body.taskId
    mlflow_run_id = request_body.mlflowRunId
    inference_image = request_body.inferenceImage

    logger.info(
        f"추론 태스크 배포 요청 수신: 태스크 ID={task_id}, MLflow Run ID={mlflow_run_id}, 이미지={inference_image}"
    )

    if task_id in active_ml_tasks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"태스크 ID '{task_id}'가 이미 존재합니다.",
        )

    # K8s 리소스 이름 생성 (Task ID의 일부 사용) - 사용자 정의 규칙 반영
    # Task ID가 너무 길 경우 K8s 이름 길이 제한에 걸릴 수 있으므로 8자로 잘라 사용합니다.
    # 바꿔야함
    safe_task_id_part = task_id.replace("_", "-").lower()[:8]
    deployment_name = f"inference-deploy-{safe_task_id_part}"
    service_name = f"inference-service-{safe_task_id_part}"
    ingress_name = (
        f"inference-ingress-{safe_task_id_part}" if request_body.ingressHost else None
    )

    # 추론 API 엔드포인트 구성 (Ingress가 있다면 Ingress URL, 없으면 내부 ClusterIP)
    inference_api_endpoint: str
    if request_body.ingressHost and request_body.ingressPath:
        inference_api_endpoint = (
            f"http://{request_body.ingressHost}{request_body.ingressPath}"
        )
    else:
        # 클러스터 내부에서만 접근 가능한 주소
        inference_api_endpoint = (
            f"http://{service_name}.{k8s_client.namespace}.svc.cluster.local:8000"
        )

    try:
        # 1. Deployment 생성
        logger.info(
            f"태스크 '{task_id}'에 대한 Kubernetes 배포 '{deployment_name}' 생성 중 (이미지: '{inference_image}')"
        )
        await asyncio.to_thread(
            k8s_client.create_inference_deployment,
            deployment_name=deployment_name,
            image=inference_image,
            mlflow_run_id=mlflow_run_id,  # 추론 서버가 로드할 모델의 MLflow Run ID
            model_file_path=request_body.modelFilePath,  # 추론 서버가 로드할 모델 파일 경로
            replicas=1,  # 초기 레플리카 수
            use_gpu=request_body.useGpu,
        )

        # 2. Service 생성
        logger.info(
            f"배포 '{deployment_name}'에 대한 Kubernetes 서비스 '{service_name}' 생성 중"
        )
        await asyncio.to_thread(
            k8s_client.create_inference_service,
            service_name=service_name,
            deployment_name=deployment_name,
        )

        # 3. Ingress 생성 (선택 사항)
        if ingress_name:
            logger.info(
                f"서비스 '{service_name}'에 대한 Kubernetes Ingress '{ingress_name}' 생성 중"
            )
            await asyncio.to_thread(
                k8s_client.create_inference_ingress,
                ingress_name=ingress_name,
                service_name=service_name,
                host=request_body.ingressHost,
                path=request_body.ingressPath,
                service_port=8000,
            )

        # active_ml_tasks에 현재 태스크 정보 저장
        active_ml_tasks[task_id] = {
            "k8s_resource_type": "Deployment",  # Deployment 타입임을 명시
            "k8s_resource_name": deployment_name,
            "model_id": request_body.modelId,
            "mlflow_run_id": mlflow_run_id,
            "status": "PENDING",  # 초기 상태
            "user_id": request_body.userId,
            "inference_api_endpoint": inference_api_endpoint,
            "service_name": service_name,  # _monitor_inference_deployment에서 사용
            "deployment_name": deployment_name,  # _monitor_inference_deployment에서 사용
            "ingress_name": ingress_name,  # 롤백 시 사용
        }

        # 5. 백그라운드에서 Deployment 상태 모니터링 시작
        background_tasks.add_task(
            _monitor_inference_deployment,
            task_id,
            deployment_name,
            mlflow_run_id,
            inference_api_endpoint,
        )

        return {
            "message": "Kubernetes 추론 리소스 생성이 시작되었습니다.",
            "inferenceApiEndpoint": inference_api_endpoint,
        }

    except Exception as e:
        logger.error(f"추론 태스크 {task_id} 배포 실패: {e}", exc_info=True)
        # 배포 실패 시 생성된 리소스 롤백
        await _rollback_resources(task_id, deployment_name, service_name, ingress_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "K8S_DEPLOYMENT_FAILED",
                "message": f"FastAPI가 Kubernetes 리소스 생성에 실패했습니다: {e}",
            },
        )


@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict_with_deployed_model(
    task_id: str, request_body: PredictRequest, x_api_key: str = Depends(get_api_key)
):
    """
    배포된 추론 서버에 데이터를 보내 예측 결과를 받습니다.
    """
    logger.info(f"태스크 ID: {task_id}에 대한 예측 요청 수신")

    task_info = active_ml_tasks.get(task_id)
    if not task_info or task_info.get("k8s_resource_type") != "Deployment":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "errorCode": "INFERENCE_TASK_NOT_FOUND",
                "message": f"태스크 ID '{task_id}'를 가진 추론 태스크를 찾을 수 없거나 배포가 아닙니다.",
            },
        )

    specific_inference_service_name = task_info.get("service_name")
    if not specific_inference_service_name:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "INFERENCE_SERVICE_NAME_MISSING",
                "message": "이 태스크에 대한 특정 추론 서비스 이름을 찾을 수 없습니다.",
            },
        )

    # 특정 추론 서비스의 내부 ClusterIP URL을 구성합니다.
    # 포트는 추론 서버 Dockerfile 및 K8s Service에 정의된 8000번입니다.
    prediction_url = f"http://{specific_inference_service_name}.{k8s_client.namespace}.svc.cluster.local:8000/predict"

    logger.info(
        f"태스크 {task_id}에 대한 예측 요청을 내부 URL로 전송 중: {prediction_url}"
    )

    try:
        response = requests.post(
            prediction_url,
            # ⭐ 핵심 수정: "data" -> "features"로 변경
            json={"features": request_body.data},
            headers={"Content-Type": "application/json"},
            timeout=30,  # 콜드 스타트 또는 모델 로딩 시간을 고려하여 타임아웃 증가
        )
        response.raise_for_status()  # HTTP 오류가 발생하면 예외 발생
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(
            f"태스크 {task_id}에 대한 추론 Pod에서 예측 가져오기 실패 ({prediction_url}): {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "errorCode": "INFERENCE_PREDICTION_FAILED",
                "message": f"추론 Pod에서 예측 가져오기 실패: {e}",
            },
        )


@router.post("/stop", status_code=status.HTTP_200_OK)
async def stop_inference_task(task_id: str, x_api_key: str = Depends(get_api_key)):
    """
    실행 중인 추론 태스크를 중지하고 관련 Kubernetes 리소스를 삭제합니다.
    """
    logger.info(f"추론 태스크 ID: {task_id}에 대한 중지 요청 수신")
    task_info = active_ml_tasks.get(task_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"태스크 ID '{task_id}'를 가진 추론 태스크를 찾을 수 없습니다.",
        )

    deployment_name = task_info.get("deployment_name")
    service_name = task_info.get("service_name")
    ingress_name = task_info.get("ingress_name")

    if not deployment_name or not service_name:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이 태스크에 대한 K8s 리소스 이름을 찾을 수 없습니다.",
        )

    try:
        await _rollback_resources(task_id, deployment_name, service_name, ingress_name)
        active_ml_tasks[task_id]["status"] = "STOPPED"
        del active_ml_tasks[task_id]
        logger.info(f"추론 태스크 {task_id}가 중지되었고 리소스가 삭제되었습니다.")
        return {
            "message": f"추론 태스크 {task_id}가 중지되었고 리소스가 삭제되었습니다."
        }
    except Exception as e:
        logger.error(f"추론 태스크 {task_id} 중지 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추론 태스크 {task_id} 중지 실패: {e}",
        )


@router.get("/status/{task_id}", status_code=status.HTTP_200_OK)
async def get_inference_task_status(
    task_id: str, x_api_key: str = Depends(get_api_key)
):
    """
    특정 추론 태스크의 현재 상태를 조회합니다.
    """
    task_info = active_ml_tasks.get(task_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"태스크 ID '{task_id}'를 가진 추론 태스크를 찾을 수 없습니다.",
        )

    return {
        "taskId": task_id,
        "status": task_info["status"],
        "mlflowRunId": task_info["mlflow_run_id"],
        "inferenceApiEndpoint": task_info["inference_api_endpoint"],
    }
