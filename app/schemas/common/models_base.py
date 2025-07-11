import typing as tp

from pydantic import BaseModel, Field


# Task 상태 업데이트 콜백 모델 (기존 task_manager.py에서 이동)
class ProgressMetrics(BaseModel):
    epoch: tp.Optional[int] = None
    iteration: tp.Optional[int] = None
    loss: tp.Optional[float] = None
    metrics: tp.Dict[str, float] = Field(default_factory=dict)  # accuracy, f1_score 등
