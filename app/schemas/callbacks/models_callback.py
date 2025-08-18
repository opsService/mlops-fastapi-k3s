import typing as tp

from app.schemas.common.models_base import ProgressMetrics
from pydantic import BaseModel, Field


class UpdateTaskStatusCallback(BaseModel):
    status: str  # PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED 등
    mlflowRunId: tp.Optional[str] = None  # MLflow Run ID (학습 Task에만 해당)
    progress: tp.Optional[ProgressMetrics] = None
    logSnippet: tp.Optional[str] = None
    errorMessage: tp.Optional[str] = None
    inferenceApiEndpoint: tp.Optional[str] = None  # Inference Task 성공 시


# 모델 등록 콜백 모델 (기존 task_manager.py에서 이동)
class RegisterModelCallback(BaseModel):
    taskId: str
    modelName: str
    modelType: str  # "CUSTOM_TRAINED", "PRE_TRAINED" 등
    modelFilePath: str
    modelSizeMB: tp.Optional[float] = None
    version: tp.Optional[str] = None
    hyperparameters: tp.Optional[tp.Dict[str, tp.Any]] = None
    performanceMetrics: tp.Optional[tp.Dict[str, float]] = None
    mlflowRunId: tp.Optional[str] = None
