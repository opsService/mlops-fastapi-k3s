import typing as tp

from pydantic import BaseModel, Field


# Task 상태 업데이트 콜백 모델 (기존 task_manager.py에서 이동)
class ProgressMetrics(BaseModel):
    epoch: tp.Optional[int] = None
    iteration: tp.Optional[int] = None
    loss: tp.Optional[float] = None
    metrics: tp.Dict[str, float] = Field(default_factory=dict)  # accuracy, f1_score 등


# Kubernetes 리소스 요청/제한을 위한 모델
class ResourceSpec(BaseModel):
    requests: tp.Optional[tp.Dict[str, str]] = Field(None, description="요청 리소스. 예: {'cpu': '1', 'memory': '2Gi'}")
    limits: tp.Optional[tp.Dict[str, str]] = Field(None, description="최대 제한 리소스. 예: {'cpu': '2', 'memory': '4Gi', 'nvidia.com/gpu': '1'}")