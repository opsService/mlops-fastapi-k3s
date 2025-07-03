# mlops-fastapi-app/app/core/task_manager.py
from typing import Dict, Any, Optional
import logging
import requests
import os
from pydantic import BaseModel, Field # 콜백 페이로드 모델을 여기서도 임포트해야 함

logger = logging.getLogger(__name__)

# --- Spring Boot 콜백 URL 설정 (환경 변수에서 가져오거나 기본값) ---
SPRING_BOOT_CALLBACK_URL = os.getenv("SPRING_BOOT_CALLBACK_URL", "http://localhost:8080/internal/api/v1/tasks")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "your-super-secret-internal-api-key") # 환경 변수로 설정 권장

# 2.1 Task 상태 업데이트 콜백을 위한 Request Body 모델 (FastAPI -> Spring Boot)
# 이 모델은 train.py와 inference.py 모두에서 사용되므로 여기에 정의합니다.
class ProgressMetrics(BaseModel):
    epoch: Optional[int] = None
    iteration: Optional[int] = None
    loss: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict) # accuracy, f1_score 등

class UpdateTaskStatusCallback(BaseModel):
    status: str # PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED 등
    mlflowRunId: Optional[str] = None # MLflow Run ID (학습 Task에만 해당)
    progress: Optional[ProgressMetrics] = None
    logSnippet: Optional[str] = None
    errorMessage: Optional[str] = None
    inferenceApiEndpoint: Optional[str] = None # Inference Task 성공 시

# 2.2 학습 완료 모델 정보 등록 콜백을 위한 Request Body 모델 (FastAPI -> Spring Boot)
# 이 모델도 여기에 정의합니다.
class RegisterModelCallback(BaseModel):
    taskId: str
    userId: str
    modelName: str
    modelType: str # "CUSTOM_TRAINED", "PRE_TRAINED" 등
    modelFilePath: str
    modelSizeMB: Optional[float] = None
    version: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    performanceMetrics: Optional[Dict[str, float]] = None
    mlflowRunId: Optional[str] = None


# --- 실행 중인 모든 Task 정보를 저장하는 딕셔너리 (임시, 실제로는 DB 사용) ---
# key: taskId (Spring Boot에서 받은 ID)
# value: {
#   "k8s_resource_type": "Job" or "Deployment",
#   "k8s_resource_name": "...",
#   "mlflow_run_id": "...", # 학습 Task에만 해당
#   "status": "...",
#   "user_id": "...",
#   "custom_model_name": "...", # 학습 Task에만 해당
#   "hyperparameters": {...}, # 학습 Task에만 해당
#   "model_id": "...", # 추론 Task에만 해당
#   "inference_api_endpoint": "..." # 추론 Task에만 해당
# }
active_ml_tasks: Dict[str, Dict] = {}


def send_status_callback(task_id: str, status: str, mlflow_run_id: Optional[str] = None, payload: Optional[UpdateTaskStatusCallback] = None):
    """
    Spring Boot Backend에 Task 상태 업데이트 콜백을 보냅니다.
    """
    callback_url = f"{SPRING_BOOT_CALLBACK_URL}/{task_id}/status"
    if payload is None:
        payload = UpdateTaskStatusCallback(status=status, mlflowRunId=mlflow_run_id)

    logger.info(f"Sending status callback for Task {task_id} ({status}) to {callback_url}")
    try:
        response = requests.put(
            callback_url,
            json=payload.model_dump(exclude_unset=True),
            headers={"Content-Type": "application/json", "X-API-KEY": INTERNAL_API_KEY},
            timeout=5 # 5초 타임아웃
        )
        response.raise_for_status()
        logger.info(f"Successfully sent status callback for Task {task_id}: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send status callback for Task {task_id} to {callback_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending status callback for Task {task_id}: {e}")


def send_model_registration_callback(payload: RegisterModelCallback):
    """
    Spring Boot Backend에 학습 완료 모델 정보 등록 콜백을 보냅니다.
    """
    # SPRING_BOOT_CALLBACK_URL이 /internal/api/v1/tasks 로 끝난다고 가정하고 models/register로 변경
    callback_url = f"{SPRING_BOOT_CALLBACK_URL.replace('tasks', 'models')}/register"
    logger.info(f"Sending model registration callback for Task {payload.taskId} to {callback_url}")
    try:
        response = requests.post(
            callback_url,
            json=payload.model_dump(exclude_unset=True),
            headers={"Content-Type": "application/json", "X-API-KEY": INTERNAL_API_KEY},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Successfully sent model registration callback for Task {payload.taskId}: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send model registration callback for Task {payload.taskId} to {callback_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending model registration callback for Task {payload.taskId}: {e}")