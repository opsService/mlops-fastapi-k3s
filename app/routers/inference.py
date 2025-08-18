# app/routers/inference.py
import logging
import typing as tp

import httpx
from app.core.config import settings
from app.core.k8s_orchestrator import k8s_orchestrator
from app.core.profile_manager import profile_manager
from app.schemas.inference.requests_inference import DeployInferenceRequest
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)

logger = logging.getLogger(__name__)


def verify_internal_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    if x_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])

@router.post("/deploy", status_code=status.HTTP_202_ACCEPTED)
async def deploy_inference_server(
    request: DeployInferenceRequest, background_tasks: BackgroundTasks
):
    """Accepts an inference deployment request and starts it in the background."""
    try:
        profile = profile_manager.get_profile(request.modelProfile)
        background_tasks.add_task(
            k8s_orchestrator.deploy_and_monitor_inference_server, 
            request=request, 
            profile=profile
        )
        return {"message": "Inference deployment job accepted", "taskId": request.taskId}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating inference deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/predict")
async def predict(task_id: str, request: Request):
    """
    Forwards a prediction request to a deployed and running inference server.

    This single endpoint handles multiple content types:
    - **application/json**: For tabular or text data.
      - Body: `{"userId": "user123", "data": {...}}`
    - **multipart/form-data**: For file-based data like images.
      - Form Fields: `userId` and `input_file`.
    """
    try:
        task_info = k8s_orchestrator.get_task_status(task_id)
        if not task_info or task_info.get("k8s_resource_type") != "Deployment":
            raise HTTPException(status_code=404, detail="Inference task not found")

        if task_info.get("status") not in ["SUCCEEDED", "RUNNING"]:
            raise HTTPException(
                status_code=409,
                detail=f"Inference task is not ready. Current status: {task_info.get('status')}",
            )

        endpoint = task_info.get("inference_api_endpoint")
        content_type = request.headers.get("content-type", "").lower()
        
        user_id = None
        httpx_params = {"timeout": 60}

        if "application/json" in content_type:
            json_body = await request.json()
            user_id = json_body.get("userId")
            httpx_params["json"] = json_body.get("data")
        elif "multipart/form-data" in content_type:
            form = await request.form()
            user_id = form.get("userId")
            input_file = form.get("input_file")
            if not input_file or not hasattr(input_file, "read"):
                raise HTTPException(status_code=400, detail="'input_file' is missing or invalid in form-data.")
            
            file_content = await input_file.read()
            httpx_params["files"] = {"input_file": (input_file.filename, file_content, input_file.content_type)}
        else:
            raise HTTPException(status_code=415, detail="Unsupported Content-Type")

        if not user_id:
            raise HTTPException(status_code=400, detail="'userId' is missing in the request.")

        logger.info(f"Forwarding prediction for task {task_id} (user: {user_id}) to {endpoint}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{endpoint}/predict", **httpx_params)
            response.raise_for_status()

        prediction_result = response.json()
        return {"userId": user_id, "prediction": prediction_result.get("prediction")}

    except HTTPException as e:
        raise e # Re-raise FastAPI exceptions
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to inference server at {endpoint}: {e}")
        raise HTTPException(status_code=503, detail=f"Inference server is unavailable: {endpoint}")
    except Exception as e:
        logger.error(f"An error occurred during prediction forwarding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{task_id}/stop", status_code=status.HTTP_202_ACCEPTED)
async def stop_inference_deployment(task_id: str, background_tasks: BackgroundTasks):
    """Stops (pauses) a running inference deployment by scaling replicas to 0."""
    try:
        background_tasks.add_task(k8s_orchestrator.pause_inference_deployment, task_id=task_id)
        return {"message": "Request to stop inference deployment accepted", "taskId": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{task_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def resume_inference_deployment(task_id: str, background_tasks: BackgroundTasks):
    """Resumes a stopped inference deployment by scaling replicas to 1."""
    try:
        background_tasks.add_task(k8s_orchestrator.resume_inference_deployment, task_id=task_id)
        return {"message": "Request to resume inference deployment accepted", "taskId": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{task_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_inference_deployment(task_id: str, background_tasks: BackgroundTasks):
    """Deletes an inference deployment and all its associated resources."""
    try:
        background_tasks.add_task(k8s_orchestrator.delete_inference_deployment, task_id=task_id)
        return {"message": "Request to delete inference deployment accepted", "taskId": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{task_id}/status")
def get_task_status(task_id: str):
    """Gets the current status of an inference task."""
    status = k8s_orchestrator.get_task_status(task_id)
    if not status or status.get("k8s_resource_type") != "Deployment":
        raise HTTPException(status_code=404, detail="Inference task not found")
    return status

@router.get("/{task_id}/logs")
def get_task_logs(task_id: str, tail_lines: int = 100):
    """Gets the logs of an inference task's pod(s)."""
    try:
        logs = k8s_orchestrator.get_task_logs(task_id, tail_lines)
        return {"taskId": task_id, "logs": logs}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
