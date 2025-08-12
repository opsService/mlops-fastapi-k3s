#/app/core/k8s_orchestrator.py
import asyncio
import logging
import typing as tp

from app.core.config import settings
from app.core.k8s_client import k8s_client
from app.core.task_manager import (
    active_ml_tasks,
    send_model_registration_callback,
    send_status_callback,
)
from app.schemas.callbacks.models_callback import (
    RegisterModelCallback,
    UpdateTaskStatusCallback,
)
from app.schemas.common.models_base import ProgressMetrics
from app.schemas.inference.requests_inference import DeployInferenceRequest
from app.schemas.train.train_requests import CreateTrainJobRequest

logger = logging.getLogger(__name__)

class K8sOrchestrator:
    """
    Kubernetes 리소스 생성, 모니터링, 삭제 등 고수준의 오케스트레이션 로직을 담당합니다.
    """
    def __init__(self):
        self.k8s_client = k8s_client # 기존 K8sClient 인스턴스를 재사용

    async def create_and_monitor_training_job(
        self,
        task_id: str,
        request: CreateTrainJobRequest,
        job_name: str,
        container_name: str,
        volume_name: str,
        pvc_name: str,
    ):
        """
        학습 Job을 생성하고 완료될 때까지 모니터링합니다.
        """
        try:
            logger.info(f"학습 Job 생성 시작: {job_name} (Task ID: {task_id})")

            # 1. Kubernetes Job 생성
            k8s_client.create_job(
                job_name=job_name,
                container_name=container_name,
                image=request.trainerImage,
                namespace=settings.K8S_NAMESPACE,
                command=[
                    "python3.11", # Changed from "python"
                    request.trainScriptPath,
                    "--task-id", task_id,
                    "--experiment-name", request.experimentName,
                    "--initial-model-id", request.initialModelId,
                    "--initial-model-file-path", request.initialModelFilePath,
                    "--dataset-path", request.datasetPath,
                    "--num-epoch", str(request.hyperparameters.numEpoch),
                    "--learning-rate", str(request.hyperparameters.learningRate),
                    "--num-batch", str(request.hyperparameters.numBatch),
                    "--custom-model-name", request.customModelName,
                    "--user-id", request.userId,
                    "--mlflow-tracking-uri", settings.MLFLOW_TRACKING_URI,
                    "--mlflow-s3-endpoint-url", settings.MLFLOW_S3_ENDPOINT_URL,
                ],
                env_vars={
                    "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
                    "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
                    "INTERNAL_API_KEY": settings.INTERNAL_API_KEY, # Added this line
                },
                volume_name=volume_name,
                pvc_name=pvc_name,
                use_gpu=request.useGpu,
            )

            active_ml_tasks[task_id]["status"] = "RUNNING"
            await send_status_callback(task_id, "RUNNING")
            logger.info(f"학습 Job {job_name} 생성 완료. 모니터링 시작.")

            # 2. Job 완료/실패 모니터링
            job_status = await self._wait_for_job_completion(task_id, job_name)

            if job_status == "SUCCEEDED":
                logger.info(f"학습 Job {job_name} 성공적으로 완료.")
                active_ml_tasks[task_id]["status"] = "SUCCEEDED"
                await send_status_callback(task_id, "SUCCEEDED")

                # MLflow Run ID 가져오기
                mlflow_run_id = active_ml_tasks[task_id].get("mlflow_run_id")
                if mlflow_run_id:
                    # 모델 등록 콜백 전송 (train.py에서 하던 로직)
                    # 실제 모델 메트릭 등은 학습 Job 내부에서 MLflow로 로깅되어야 함
                    performance_metrics = {} # MLflow에서 메트릭 가져와서 채울 것
                    
                    # MLflow Tracking Server에서 run 정보 가져오기
                    try:
                        client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
                        run = client.get_run(mlflow_run_id)
                        performance_metrics = run.data.metrics
                        logger.info(f"MLflow Run ID {mlflow_run_id}에서 메트릭 로드 성공: {performance_metrics}")
                    except Exception as e:
                        logger.warning(f"MLflow Run ID {mlflow_run_id}의 메트릭을 로드할 수 없습니다: {e}")

                    registration_payload = RegisterModelCallback(
                        taskId=task_id,
                        userId=request.userId,
                        modelName=request.customModelName,
                        modelType="CUSTOM_TRAINED",
                        modelFilePath=f"runs:/{mlflow_run_id}/ml_model", # MLflow 모델 URI
                        hyperparameters=request.hyperparameters.model_dump(),
                        performanceMetrics=performance_metrics,
                        mlflowRunId=mlflow_run_id,
                    )
                    await send_model_registration_callback(registration_payload)
                    
                else:
                    logger.warning(f"Task {task_id}에 대한 MLflow Run ID가 없습니다. 모델 등록 콜백을 건너뜝니다.")


            else: # FAILED, UNKNOWN 등
                logger.error(f"학습 Job {job_name} 실패 또는 예상치 못한 상태: {job_status}")
                active_ml_tasks[task_id]["status"] = "FAILED"
                await send_status_callback(task_id, "FAILED", errorMessage=f"Training job failed with status: {job_status}")

        except Exception as e:
            logger.error(f"학습 Job {job_name} 처리 중 치명적인 오류 발생: {e}", exc_info=True)
            active_ml_tasks[task_id]["status"] = "FAILED"
            await send_status_callback(
                task_id,
                "FAILED",
                errorMessage=f"Critical error during training job orchestration: {e}",
            )
        finally:
            # 리소스 정리 (성공/실패 무관)
            await self._cleanup_training_resources(task_id, job_name, volume_name, pvc_name)
            if task_id in active_ml_tasks:
                del active_ml_tasks[task_id]


    async def _wait_for_job_completion(self, task_id: str, job_name: str) -> str:
        """
        Kubernetes Job의 완료를 기다립니다.
        """
        timeout = 3600  # 1시간 타임아웃
        interval = 10   # 10초마다 체크
        elapsed_time = 0

        while elapsed_time < timeout:
            try:
                job_status = k8s_client.get_job_status(job_name, settings.K8S_NAMESPACE)
                logger.debug(f"Job {job_name} 현재 상태: {job_status}")

                if job_status == "SUCCEEDED":
                    return "SUCCEEDED"
                elif job_status == "FAILED":
                    # Pod 로그 스니펫 가져오기
                    pods = k8s_client.get_pods_for_job(job_name, settings.K8S_NAMESPACE)
                    logs = ""
                    if pods:
                        # 최신 Pod의 로그를 가져오도록 수정
                        pods.sort(key=lambda p: p.metadata.creation_timestamp, reverse=True)
                        target_pod = pods[0] if pods else None
                        if target_pod:
                            logs = k8s_client.get_pod_logs(target_pod.metadata.name, tail_lines=20)
                            logger.error(f"Job {job_name} 실패 로그 스니펫:\n{logs}")
                    
                    await send_status_callback(
                        task_id, "FAILED", errorMessage=f"Job failed in Kubernetes. Check logs for details.", logSnippet=logs
                    )
                    return "FAILED"
                elif job_status == "UNKNOWN" or job_status == "PENDING":
                    # PENDING 상태가 너무 길어지면 에러 처리 고려
                    pass
                
                # 진행 상태 업데이트 콜백 (옵션)
                # 예: Job의 Pod 로그를 주기적으로 스니펫으로 보내는 로직 추가 가능
                # if elapsed_time % 60 == 0: # 1분마다 로그 스니펫 전송
                #     pods = k8s_client.get_pods_for_job(job_name, settings.K8S_NAMESPACE)
                #     if pods:
                #         pods.sort(key=lambda p: p.metadata.creation_timestamp, reverse=True)
                #         target_pod = pods[0] if pods else None
                #         if target_pod:
                #             logs = k8s_client.get_pod_logs(target_pod.metadata.name, tail_lines=10)
                #             await send_status_callback(task_id, "RUNNING", payload=UpdateTaskStatusCallback(logSnippet=logs))


            except Exception as e:
                logger.error(f"Job {job_name} 모니터링 중 오류 발생: {e}", exc_info=True)
                await send_status_callback(
                    task_id, "FAILED", errorMessage=f"Error monitoring job: {e}"
                )
                return "FAILED" # 모니터링 실패도 FAILED로 간주

            await asyncio.sleep(interval)
            elapsed_time += interval
        
        logger.warning(f"Job {job_name}이 {timeout}초 내에 완료되지 않았습니다. 타임아웃 처리.")
        await send_status_callback(
            task_id, "FAILED", errorMessage=f"Training job timed out after {timeout} seconds."
        )
        return "TIMED_OUT"

    async def _cleanup_training_resources(self, task_id: str, job_name: str, volume_name: str, pvc_name: str):
        """
        학습 Job 관련 Kubernetes 리소스를 정리합니다.
        """
        logger.info(f"학습 Job {job_name} 리소스 정리 시작 (Task ID: {task_id})")
        try:
            k8s_client.delete_job(job_name, settings.K8S_NAMESPACE)
            k8s_client.delete_pvc(pvc_name, settings.K8S_NAMESPACE) # PVC도 삭제
            # Persistent Volume은 PVC 삭제 시 자동으로 해제되거나, 동적 프로비저닝에 따라 처리됨
            logger.info(f"학습 Job {job_name} 관련 리소스 정리 완료.")
        except Exception as e:
            logger.error(f"학습 Job {job_name} 리소스 정리 중 오류 발생: {e}", exc_info=True)
            # 리소스 정리는 실패해도 Task는 FAILED/SUCCEEDED 상태로 유지
            pass # 에러를 다시 raise하지 않고 로깅만 합니다.

    async def deploy_and_monitor_inference_server(
        self,
        task_id: str,
        request: DeployInferenceRequest,
        deployment_name: str,
        service_name: str,
        ingress_name: tp.Optional[str]
    ):
        """
        추론 서버 Deployment, Service, Ingress를 생성하고 모니터링합니다.
        """
        try:
            logger.info(f"추론 서버 배포 시작: {deployment_name} (Task ID: {task_id})")

            # 1. Deployment 생성
            k8s_client.create_deployment(
                deployment_name=deployment_name,
                container_name="inference-server-container", # 컨테이너 이름 고정
                image=request.inferenceImage,
                namespace=settings.K8S_NAMESPACE,
                model_uri=f"runs:/{request.mlflowRunId}/ml_model" if request.mlflowRunId else request.modelFilePath,
                # 추론 서버에 MLflow Tracking URI와 S3 Endpoint URL도 전달하여 모델 로딩에 사용
                env_vars={
                    "MLFLOW_TRACKING_URI": settings.MLFLOW_TRACKING_URI,
                    "MLFLOW_S3_ENDPOINT_URL": settings.MLFLOW_S3_ENDPOINT_URL,
                    "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
                    "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
                },
                use_gpu=request.useGpu,
            )

            # 2. Service 생성
            k8s_client.create_service(
                service_name=service_name,
                deployment_name=deployment_name,
                namespace=settings.K8S_NAMESPACE,
                port=8000, # 추론 서버의 노출 포트 (Flask Gunicorn 기본 포트)
            )
            
            # active_ml_tasks에 현재 상태 저장
            active_ml_tasks[task_id]["status"] = "RUNNING"
            active_ml_tasks[task_id]["service_name"] = service_name
            active_ml_tasks[task_id]["deployment_name"] = deployment_name

            # 3. Ingress 생성 (선택 사항)
            if request.ingressHost and request.ingressPath:
                k8s_client.create_ingress(
                    ingress_name=ingress_name,
                    host=request.ingressHost,
                    path=request.ingressPath,
                    service_name=service_name,
                    service_port=8000, # Ingress가 바라볼 서비스 포트
                    namespace=settings.K8S_NAMESPACE
                )
                active_ml_tasks[task_id]["ingress_name"] = ingress_name
                logger.info(f"Ingress {ingress_name} 생성 완료.")
            
            await send_status_callback(task_id, "RUNNING")
            logger.info(f"추론 서버 Deployment {deployment_name} 생성 완료. 모니터링 시작.")

            # 4. Deployment 롤아웃 완료 모니터링 및 헬스 체크
            success = await self._wait_for_deployment_ready(task_id, deployment_name, service_name, ingress_name)

            if success:
                logger.info(f"추론 서버 Deployment {deployment_name} 성공적으로 배포 및 준비 완료.")
                inference_api_endpoint = self._get_inference_api_endpoint(service_name, ingress_name, request.ingressHost, request.ingressPath)
                active_ml_tasks[task_id]["status"] = "SUCCEEDED"
                active_ml_tasks[task_id]["inference_api_endpoint"] = inference_api_endpoint
                await send_status_callback(task_id, "SUCCEEDED", inferenceApiEndpoint=inference_api_endpoint)
            else:
                logger.error(f"추론 서버 Deployment {deployment_name} 배포 실패 또는 준비 시간 초과.")
                active_ml_tasks[task_id]["status"] = "FAILED"
                await send_status_callback(
                    task_id,
                    "FAILED",
                    errorMessage=f"Inference server deployment failed or timed out: {deployment_name}",
                )

        except Exception as e:
            logger.error(f"추론 서버 배포 중 치명적인 오류 발생: {e}", exc_info=True)
            active_ml_tasks[task_id]["status"] = "FAILED"
            await send_status_callback(
                task_id,
                "FAILED",
                errorMessage=f"Critical error during inference server orchestration: {e}",
            )
            # 오류 발생 시 리소스 롤백 시도
            await self._rollback_inference_resources(task_id, deployment_name, service_name, ingress_name)
        finally:
            # 태스크 완료 또는 실패 시 active_ml_tasks에서 제거
            if task_id in active_ml_tasks and active_ml_tasks[task_id]["status"] in ["SUCCEEDED", "FAILED", "STOPPED"]:
                del active_ml_tasks[task_id]


    async def _wait_for_deployment_ready(self, task_id: str, deployment_name: str, service_name: str, ingress_name: Optional[str]) -> bool:
        """
        Kubernetes Deployment가 준비 상태가 될 때까지 기다리고, 헬스 체크 엔드포인트를 확인합니다.
        """
        timeout = 600  # 10분 타임아웃
        interval = 5    # 5초마다 체크
        elapsed_time = 0

        while elapsed_time < timeout:
            try:
                # 1. Deployment 상태 확인 (replicas, readyReplicas)
                deployment_status = k8s_client.get_deployment_status(deployment_name, settings.K8S_NAMESPACE)
                logger.debug(f"Deployment {deployment_name} 현재 상태: {deployment_status}")

                if deployment_status == "READY":
                    # 2. Service Endpoint 확인 및 헬스 체크
                    inference_api_endpoint = self._get_inference_api_endpoint(service_name, ingress_name, active_ml_tasks[task_id].get("ingress_host"), active_ml_tasks[task_id].get("ingress_path"))
                    if not inference_api_endpoint:
                        logger.warning(f"추론 서버 {deployment_name}의 API 엔드포인트를 확인할 수 없습니다. 재시도합니다.")
                        await asyncio.sleep(interval)
                        elapsed_time += interval
                        continue

                    try:
                        async with httpx.AsyncClient() as client:
                            health_url = f"{inference_api_endpoint}/health"
                            logger.info(f"추론 서버 헬스 체크 시도: {health_url}")
                            response = await client.get(health_url, timeout=5)
                            response.raise_for_status()
                            logger.info(f"추론 서버 {deployment_name} 헬스 체크 성공.")
                            return True
                    except requests.exceptions.RequestException as http_e:
                        logger.warning(f"추론 서버 헬스 체크 실패 ({health_url}): {http_e}. 재시도합니다.")
                        # 헬스 체크 실패 시 Deployment 롤백 상태인지 다시 확인 (새 Pod 생성 중일 수 있음)
                        deployment_status_after_health_check = k8s_client.get_deployment_status(deployment_name, settings.K8S_NAMESPACE)
                        if deployment_status_after_health_check != "READY":
                            logger.warning(f"Deployment {deployment_name}가 헬스 체크 실패 후 READY 상태가 아님. 다시 모니터링 시작.")
                        
                elif deployment_status == "FAILED":
                    logger.error(f"Deployment {deployment_name}이 실패 상태입니다. Pod 로그를 확인합니다.")
                    pods = k8s_client.get_pods_for_deployment(deployment_name, settings.K8S_NAMESPACE)
                    logs = ""
                    if pods:
                        pods.sort(key=lambda p: p.metadata.creation_timestamp, reverse=True)
                        target_pod = pods[0] if pods else None
                        if target_pod:
                            logs = k8s_client.get_pod_logs(target_pod.metadata.name, tail_lines=20)
                            logger.error(f"Deployment {deployment_name} 실패 로그 스니펫:\n{logs}")
                    
                    await send_status_callback(
                        task_id, "FAILED", errorMessage=f"Deployment failed in Kubernetes. Check logs for details.", logSnippet=logs
                    )
                    return False

            except Exception as e:
                logger.error(f"Deployment {deployment_name} 모니터링 중 오류 발생: {e}", exc_info=True)
                await send_status_callback(
                    task_id, "FAILED", errorMessage=f"Error monitoring deployment: {e}"
                )
                return False

            await asyncio.sleep(interval)
            elapsed_time += interval
        
        logger.warning(f"Deployment {deployment_name}이 {timeout}초 내에 준비되지 않았습니다. 타임아웃 처리.")
        return False

    def _get_inference_api_endpoint(
        self, service_name: str, ingress_name: tp.Optional[str], ingress_host: tp.Optional[str], ingress_path: tp.Optional[str]
    ) -> str:
        """
        배포된 추론 서버의 최종 API 엔드포인트를 구성합니다.
        Ingress가 있다면 Ingress 주소를, 없다면 Service 주소를 반환합니다.
        """
        if ingress_name and ingress_host and ingress_path:
            # Ingress가 있다면 Ingress의 호스트와 경로를 사용
            # 실제 Ingress의 외부 IP/호스트를 가져오는 로직이 필요할 수 있습니다.
            # 여기서는 요청에서 받은 ingressHost를 직접 사용
            # Ingress의 포트가 80/443이라면 명시하지 않아도 됩니다.
            # Ingress의 scheme (http/https)도 고려해야 합니다. (여기서는 일단 http)
            return f"http://{ingress_host}{ingress_path}"
        else:
            # Ingress가 없다면 ClusterIP Service의 내부 DNS 이름을 사용 (클러스터 내부 통신용)
            # 외부에서 접근하려면 NodePort, LoadBalancer 서비스 타입을 사용해야 함
            return f"http://{service_name}.{settings.K8S_NAMESPACE}.svc.cluster.local:8000"

    async def _rollback_inference_resources(
        self, task_id: str, deployment_name: str, service_name: str, ingress_name: tp.Optional[str]
    ):
        """
        오류 발생 시 추론 리소스를 롤백/삭제합니다.
        """
        logger.info(f"추론 태스크 {task_id} 리소스 롤백 시작: 배포: {deployment_name}, 서비스: {service_name}, 인그레스: {ingress_name}")
        try:
            if deployment_name:
                self.k8s_client.delete_deployment(deployment_name, settings.K8S_NAMESPACE)
            if service_name:
                self.k8s_client.delete_service(service_name, settings.K8S_NAMESPACE)
            if ingress_name:
                self.k8s_client.delete_ingress(ingress_name)
            logger.info(f"추론 태스크 {task_id} 리소스 롤백 완료.")
        except Exception as e:
            logger.error(f"추론 태스크 {task_id} 리소스 롤백 중 오류 발생: {e}", exc_info=True)
            # 롤백 중 오류가 발생해도 로깅만 하고 넘어가야 합니다.
            pass

    async def stop_inference_deployment(self, task_id: str, deployment_name: str, service_name: str, ingress_name: Optional[str]):
        """
        실행 중인 추론 Deployment를 중지하고 관련 리소스를 삭제합니다.
        """
        logger.info(f"추론 태스크 {task_id} 중지 요청: 배포: {deployment_name}, 서비스: {service_name}, 인그레스: {ingress_name}")
        try:
            await self._rollback_inference_resources(task_id, deployment_name, service_name, ingress_name)
            logger.info(f"추론 태스크 {task_id}가 중지되었고 리소스가 삭제되었습니다.")
            return True
        except Exception as e:
            logger.error(f"추론 태스크 {task_id} 중지 실패: {e}", exc_info=True)
            return False

# K8sOrchestrator 인스턴스 생성 (싱글톤 패턴으로 관리)
k8s_orchestrator = K8sOrchestrator()