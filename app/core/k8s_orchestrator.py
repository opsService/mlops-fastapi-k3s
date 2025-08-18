# app/core/k8s_orchestrator.py (Refactored for Inference)
import asyncio
import logging
import typing as tp
from ulid import ULID

import mlflow
from app.core.config import settings
from app.core.k8s_client import k8s_client
from app.core.callback_sender import (
    send_model_registration_callback,
    send_status_callback,
)
from app.schemas.callbacks.models_callback import RegisterModelCallback
from app.schemas.train.train_requests import CreateTrainJobRequest
from app.schemas.inference.requests_inference import DeployInferenceRequest

logger = logging.getLogger(__name__)

active_ml_tasks: tp.Dict[str, tp.Dict[str, tp.Any]] = {}

class K8sOrchestrator:
    def __init__(self):
        self.k8s_client = k8s_client
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.mlflow_client = mlflow.tracking.MlflowClient()

    # --- Training Methods ---
    async def create_and_monitor_training_job(self, request: CreateTrainJobRequest):
        task_id = request.taskId
        job_name = f"train-job-{str(ULID()).lower()}"
        pvc_name = f"train-pvc-{job_name}" # This might need adjustment if PVC is pre-created

        try:
            mlflow.set_experiment(request.experimentName)
            with mlflow.start_run(run_name=f"run-{task_id}") as run:
                mlflow_run_id = run.info.run_id
                logger.info(f"New MLflow Run started: {mlflow_run_id} for Task ID: {task_id}")
                mlflow.set_tag("task_id", task_id)
                mlflow.set_tag("model_profile", request.modelProfile)
                mlflow.set_tag("custom_model_name", request.customModelName)
                mlflow.log_params(request.hyperparameters.model_dump())

            active_ml_tasks[task_id] = {
                "k8s_resource_type": "Job",
                "k8s_job_name": job_name,
                "mlflow_run_id": mlflow_run_id,
                "status": "PENDING",
                "request": request.model_dump(),
            }
            await send_status_callback(task_id, "PENDING", mlflow_run_id)

            command = self._build_train_command(task_id, mlflow_run_id, request)
            self.k8s_client.create_job(
                job_name=job_name,
                container_name=f"train-container-{task_id}",
                image=request.trainerImage,
                namespace=settings.K8S_NAMESPACE,
                command=command,
                resources=request.resources.model_dump() if hasattr(request, 'resources') and request.resources else None,
                pvc_name=pvc_name,
                use_gpu=request.useGpu,
            )
            active_ml_tasks[task_id]["status"] = "RUNNING"
            await send_status_callback(task_id, "RUNNING", mlflow_run_id)
            logger.info(f"Training Job {job_name} created. Starting monitoring.")

            job_status = await self._wait_for_job_completion(task_id, job_name, mlflow_run_id)

            if job_status == "SUCCEEDED":
                await self._handle_successful_job(task_id, mlflow_run_id, request)
            else:
                await self._handle_failed_job(task_id, mlflow_run_id, job_name, job_status)

        except Exception as e:
            logger.error(f"Critical error in training orchestration for task {task_id}: {e}", exc_info=True)
            if task_id in active_ml_tasks:
                active_ml_tasks[task_id]["status"] = "FAILED"
                await send_status_callback(task_id, "FAILED", errorMessage=str(e))
        finally:
            await self._cleanup_training_resources(job_name, pvc_name)
            if task_id in active_ml_tasks and active_ml_tasks[task_id].get("k8s_resource_type") == "Job":
                del active_ml_tasks[task_id]
                logger.info(f"Training task {task_id} removed from active tasks.")

    def _build_train_command(self, task_id: str, mlflow_run_id: str, request: CreateTrainJobRequest) -> tp.List[str]:
        # ... (implementation unchanged)
        return [
            "python3.11", "script/train.py",
            "--task-id", task_id,
            "--mlflow-run-id", mlflow_run_id,
            "--experiment-name", request.experimentName,
            "--initial-model-file-path", request.initialModelFilePath,
            "--dataset-path", request.datasetPath,
            "--handler-name", request.handlerName,
            "--task-type", request.taskType,
            "--num-epoch", str(request.hyperparameters.numEpoch),
            "--learning-rate", str(request.hyperparameters.learningRate),
            "--num-batch", str(request.hyperparameters.numBatch),
            "--custom-model-name", request.customModelName,
            "--mlflow-tracking-uri", settings.MLFLOW_TRACKING_URI,
            "--mlflow-s3-endpoint-url", settings.MLFLOW_S3_ENDPOINT_URL,
        ]

    async def _wait_for_job_completion(self, task_id: str, job_name: str, mlflow_run_id: str) -> str:
        # ... (implementation unchanged)
        timeout = 3600
        interval = 10
        elapsed_time = 0
        while elapsed_time < timeout:
            status = self.k8s_client.get_job_status(job_name)
            if status and status.succeeded:
                return "SUCCEEDED"
            if status and status.failed:
                return "FAILED"
            await asyncio.sleep(interval)
            elapsed_time += interval
        return "TIMED_OUT"

    async def _handle_successful_job(self, task_id: str, mlflow_run_id: str, request: CreateTrainJobRequest):
        # ... (implementation unchanged)
        logger.info(f"Training Job for task {task_id} succeeded.")
        active_ml_tasks[task_id]["status"] = "SUCCEEDED"
        await send_status_callback(task_id, "SUCCEEDED", mlflow_run_id)
        
        run = self.mlflow_client.get_run(mlflow_run_id)
        model_uri = f"runs:/{mlflow_run_id}/ml_model"

        reg_payload = RegisterModelCallback(
            taskId=task_id,
            modelName=request.customModelName,
            modelType="CUSTOM_TRAINED",
            modelFilePath=model_uri,
            hyperparameters=request.hyperparameters.model_dump(),
            performanceMetrics=run.data.metrics,
            mlflowRunId=mlflow_run_id,
        )
        await send_model_registration_callback(reg_payload)

    async def _handle_failed_job(self, task_id: str, mlflow_run_id: str, job_name: str, job_status: str):
        # ... (implementation unchanged)
        logger.error(f"Training Job {job_name} failed with status: {job_status}")
        active_ml_tasks[task_id]["status"] = "FAILED"
        logs = self.get_task_logs(task_id, tail_lines=50)
        await send_status_callback(task_id, "FAILED", mlflow_run_id, errorMessage=f"Job status: {job_status}", logSnippet=logs)

    async def _cleanup_training_resources(self, job_name: str, pvc_name: str):
        # ... (implementation unchanged)
        logger.info(f"Cleaning up resources for job {job_name}")
        try:
            self.k8s_client.delete_job(job_name)
            # self.k8s_client.delete_pvc(pvc_name) # PVC deletion logic needs to be added to k8s_client
        except Exception as e:
            logger.error(f"Error during resource cleanup for job {job_name}: {e}", exc_info=True)

    async def stop_training_task(self, task_id: str):
        task_info = self._get_task_or_raise(task_id, "Job")
        
        job_name = task_info.get("k8s_job_name")
        pvc_name = f"train-pvc-{job_name}"
        mlflow_run_id = task_info.get("mlflow_run_id")

        logger.info(f"Stopping training task {task_id} and cleaning up resources.")
        await self._cleanup_training_resources(job_name, pvc_name)
        if mlflow_run_id:
            self.mlflow_client.set_terminated(mlflow_run_id, "KILLED")
        
        active_ml_tasks[task_id]["status"] = "STOPPED"
        await send_status_callback(task_id, "STOPPED", mlflow_run_id)
        if task_id in active_ml_tasks:
            del active_ml_tasks[task_id]

    # --- Inference Methods ---
    async def deploy_and_monitor_inference_server(self, request: DeployInferenceRequest, profile: tp.Dict[str, tp.Any]):
        # ... (implementation mostly unchanged)
        task_id = request.taskId
        unique_id = str(ULID()).lower()
        deployment_name = f"inference-deploy-{unique_id}"
        service_name = f"inference-service-{unique_id}"
        ingress_name = f"inference-ingress-{unique_id}" if request.ingressHost else None

        try:
            active_ml_tasks[task_id] = {
                "k8s_resource_type": "Deployment",
                "k8s_deployment_name": deployment_name,
                "k8s_service_name": service_name,
                "k8s_ingress_name": ingress_name,
                "status": "PENDING",
                "request": request.model_dump(),
            }
            await send_status_callback(task_id, "PENDING")

            inference_image = profile.get("inferenceImage")
            if not inference_image:
                raise ValueError(f"Profile '{request.modelProfile}' does not have an inferenceImage.")

            self.k8s_client.create_inference_deployment(
                deployment_name=deployment_name,
                image=inference_image,
                mlflow_run_id=request.mlflowRunId,
                model_file_path=request.modelFilePath,
                resources_requests=profile.get("resources", {}).get("requests"),
                resources_limits=profile.get("resources", {}).get("limits"),
                use_gpu=request.useGpu,
            )
            self.k8s_client.create_inference_service(service_name, deployment_name)
            if ingress_name:
                self.k8s_client.create_inference_ingress(ingress_name, service_name, request.ingressHost, request.ingressPath)

            active_ml_tasks[task_id]["status"] = "RUNNING"
            await send_status_callback(task_id, "RUNNING")
            logger.info(f"Inference server {deployment_name} created. Starting monitoring.")

            await asyncio.sleep(30) # Placeholder for readiness check

            active_ml_tasks[task_id]["status"] = "SUCCEEDED"
            endpoint = self._get_inference_api_endpoint(service_name, ingress_name, request.ingressHost, request.ingressPath)
            active_ml_tasks[task_id]["inference_api_endpoint"] = endpoint
            await send_status_callback(task_id, "SUCCEEDED", inferenceApiEndpoint=endpoint)

        except Exception as e:
            logger.error(f"Critical error in inference orchestration for task {task_id}: {e}", exc_info=True)
            if task_id in active_ml_tasks:
                active_ml_tasks[task_id]["status"] = "FAILED"
                await send_status_callback(task_id, "FAILED", errorMessage=str(e))
            await self.delete_inference_deployment(task_id)

    async def pause_inference_deployment(self, task_id: str):
        """Scales down an inference deployment to 0 replicas."""
        task_info = self._get_task_or_raise(task_id, "Deployment")
        deployment_name = task_info.get("k8s_deployment_name")
        
        logger.info(f"Pausing inference deployment for task {task_id} by scaling to 0 replicas.")
        self.k8s_client.scale_deployment(deployment_name, 0)
        
        task_info["status"] = "STOPPED"
        await send_status_callback(task_id, "STOPPED")
        logger.info(f"Inference deployment for task {task_id} paused.")

    async def resume_inference_deployment(self, task_id: str):
        """Scales up an inference deployment to 1 replica."""
        task_info = self._get_task_or_raise(task_id, "Deployment")
        deployment_name = task_info.get("k8s_deployment_name")
        
        logger.info(f"Resuming inference deployment for task {task_id} by scaling to 1 replica.")
        self.k8s_client.scale_deployment(deployment_name, 1)
        
        task_info["status"] = "RUNNING" # Or should wait for readiness
        await send_status_callback(task_id, "RUNNING")
        logger.info(f"Inference deployment for task {task_id} resumed.")

    async def delete_inference_deployment(self, task_id: str):
        """Deletes all resources associated with an inference deployment."""
        task_info = self._get_task_or_raise(task_id, "Deployment")
        
        deployment_name = task_info.get("k8s_deployment_name")
        service_name = task_info.get("k8s_service_name")
        ingress_name = task_info.get("k8s_ingress_name")

        logger.info(f"Deleting all resources for inference task {task_id}.")
        await self._cleanup_inference_resources(deployment_name, service_name, ingress_name)
        
        task_info["status"] = "DELETED"
        await send_status_callback(task_id, "DELETED")
        if task_id in active_ml_tasks:
            del active_ml_tasks[task_id]
            logger.info(f"Inference task {task_id} removed from active tasks.")

    async def _cleanup_inference_resources(self, deployment_name, service_name, ingress_name):
        logger.info(f"Cleaning up resources for inference server {deployment_name}")
        if ingress_name:
            try: self.k8s_client.delete_ingress(ingress_name)
            except Exception as e: logger.warning(f"Could not delete ingress {ingress_name}: {e}")
        if service_name:
            try: self.k8s_client.delete_service(service_name)
            except Exception as e: logger.warning(f"Could not delete service {service_name}: {e}")
        if deployment_name:
            try: self.k8s_client.delete_deployment(deployment_name)
            except Exception as e: logger.warning(f"Could not delete deployment {deployment_name}: {e}")

    def _get_inference_api_endpoint(self, service_name, ingress_name, host, path):
        # ... (implementation unchanged)
        if ingress_name and host and path:
            return f"http://{host}{path}"
        return f"http://{service_name}.{settings.K8S_NAMESPACE}.svc.cluster.local:8000"

    # --- Common Methods ---
    def get_task_status(self, task_id: str) -> tp.Optional[tp.Dict[str, tp.Any]]:
        return active_ml_tasks.get(task_id)

    def get_task_logs(self, task_id: str, tail_lines: int = 100) -> str:
        task_info = self._get_task_or_raise(task_id)
        
        if task_info.get("k8s_resource_type") == "Job":
            resource_name = task_info.get("k8s_job_name")
            pods = self.k8s_client.get_pods_for_job(resource_name)
            if not pods:
                return f"No pods found for job {resource_name}."
        else: # Deployment
            # This part needs implementation: get pods for a deployment
            return "Log retrieval for deployments is not yet implemented."

        pods.sort(key=lambda p: p.metadata.creation_timestamp, reverse=True)
        return self.k8s_client.get_pod_logs(pods[0].metadata.name, tail_lines=tail_lines)

    def _get_task_or_raise(self, task_id: str, expected_type: str = None) -> tp.Dict[str, tp.Any]:
        """Gets task info, raises HTTPException if not found or type mismatch."""
        task_info = self.get_task_status(task_id)
        if not task_info:
            raise ValueError(f"Task with ID '{task_id}' not found.")
        if expected_type and task_info.get("k8s_resource_type") != expected_type:
            raise ValueError(f"Task '{task_id}' is not a '{expected_type}' task.")
        return task_info

k8s_orchestrator = K8sOrchestrator()
