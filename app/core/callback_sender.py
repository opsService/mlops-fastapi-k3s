# app/core/callback_sender.py (previously task_manager.py)
import logging
import typing as tp

import requests
from app.core.config import settings
from app.schemas.callbacks.models_callback import (
    RegisterModelCallback,
    UpdateTaskStatusCallback,
)

logger = logging.getLogger(__name__)

def send_status_callback(
    task_id: str,
    status: str,
    mlflow_run_id: tp.Optional[str] = None,
    error_message: tp.Optional[str] = None,
    log_snippet: tp.Optional[str] = None,
    **kwargs
):
    """
    Spring Boot Backend에 Task 상태 업데이트 콜백을 보냅니다.
    """
    callback_url = f"{settings.SPRING_BOOT_CALLBACK_URL}/{task_id}/status"
    payload = UpdateTaskStatusCallback(
        status=status, 
        mlflowRunId=mlflow_run_id, 
        errorMessage=error_message, 
        logSnippet=log_snippet,
        **kwargs
    )

    logger.info(
        f"Sending status callback for Task {task_id} ({status}) to {callback_url}"
    )
    try:
        response = requests.put(
            callback_url,
            json=payload.model_dump(exclude_unset=True),
            headers={"Content-Type": "application/json", "X-API-KEY": settings.INTERNAL_API_KEY},
            timeout=5,
        )
        response.raise_for_status()
        logger.info(
            f"Successfully sent status callback for Task {task_id}: {response.json()}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Failed to send status callback for Task {task_id} to {callback_url}: {e}"
        )


def send_model_registration_callback(payload: RegisterModelCallback):
    """
    Spring Boot Backend에 학습 완료 모델 정보 등록 콜백을 보냅니다.
    """
    callback_url = f"{settings.SPRING_BOOT_CALLBACK_URL.replace('tasks', 'models')}/register"
    logger.info(
        f"Sending model registration callback for Task {payload.taskId} to {callback_url}"
    )
    try:
        response = requests.post(
            callback_url,
            json=payload.model_dump(exclude_unset=True),
            headers={"Content-Type": "application/json", "X-API-KEY": settings.INTERNAL_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        logger.info(
            f"Successfully sent model registration callback for Task {payload.taskId}: {response.json()}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Failed to send model registration callback for Task {payload.taskId} to {callback_url}: {e}"
        )