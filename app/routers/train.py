# app/routers/train.py (Refactored)
import logging
import typing as tp

from app.core.config import settings
from app.core.k8s_orchestrator import k8s_orchestrator
from app.core.profile_manager import profile_manager
from app.schemas.train.train_requests import CreateTrainJobRequest
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)


def verify_internal_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    if x_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])

@router.post("/job", status_code=status.HTTP_202_ACCEPTED)
async def create_train_job(
    request: CreateTrainJobRequest, background_tasks: BackgroundTasks
):
    """Accepts a training job request based on a model profile."""
    try:
        profile = profile_manager.get_profile(request.modelProfile)
        
        # Combine request and profile to create the full request object
        full_request_data = request.model_dump()
        full_request_data["handlerName"] = profile["handlerName"]
        full_request_data["taskType"] = profile["taskType"]
        full_request_data["resources"] = profile["resources"]
        if not request.trainerImage:
            full_request_data["trainerImage"] = profile["trainerImage"]
        
        # Re-validate with the model to include profile data
        final_request = CreateTrainJobRequest(**full_request_data)

        background_tasks.add_task(
            k8s_orchestrator.create_and_monitor_training_job, request=final_request
        )
        return {"message": "Training job accepted", "taskId": request.taskId}

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating training job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{task_id}/status")
def get_task_status(task_id: str):
    """Gets the status of an active task."""
    status = k8s_orchestrator.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@router.get("/{task_id}/logs")
def get_task_logs(task_id: str, tail_lines: int = 100):
    """Gets the logs of a task's pod."""
    try:
        logs = k8s_orchestrator.get_task_logs(task_id, tail_lines)
        return {"taskId": task_id, "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{task_id}", status_code=status.HTTP_202_ACCEPTED)
async def stop_training_task(task_id: str, background_tasks: BackgroundTasks):
    """Stops a running training task and cleans up its resources."""
    try:
        background_tasks.add_task(k8s_orchestrator.stop_training_task, task_id=task_id)
        return {"message": "Task stop request accepted", "taskId": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))