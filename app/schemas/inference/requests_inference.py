import typing as tp

from pydantic import BaseModel, Field
from app.schemas.common.models_base import ResourceSpec


# 추론 Job 배포 요청 모델
class InferenceConfig(BaseModel):
    """추론 태스크의 설정 정보를 정의합니다."""

    inputDataType: str
    postProcessingOption: str
    # 필요에 따라 더 많은 추론 설정 추가


class DeployInferenceRequest(BaseModel):
    """추론 태스크 배포 요청의 바디 형식을 정의합니다."""

    taskId: str = Field(..., description="Spring Boot에서 생성한 Task ID")
    modelId: str = Field(..., description="사용자가 선택한 모델 ID")
    modelFilePath: str = Field(..., description="모델 파일 저장 경로 (S3/MinIO)")
    mlflowRunId: tp.Optional[str] = Field(
        None, description="(선택 사항) 커스텀 학습 모델인 경우 MLflow Run ID"
    )
    inferenceConfig: InferenceConfig = Field(
        ..., description="사용자 입력 추론 관련 설정"
    )
    userId: str = Field(..., description="Task를 생성한 사용자 ID")
    useGpu: bool = False  # GPU 사용 여부
    resources: tp.Optional[ResourceSpec] = Field(
        None, description="추론 Job에 할당할 리소스 (CPU, Memory, GPU)"
    )
    # 사용자의 요청에 따라 final-debug-v1 유지.
    # 단, 이 이미지는 반드시 상세 로깅이 포함된 inference_server.py로 빌드되어야 합니다.
    inferenceImage: str = (
        "localhost:5002/heedong/mlflow-inference-server:final-debug-v1"
    )
    ingressHost: tp.Optional[str] = None  # Ingress 사용 시 호스트
    ingressPath: tp.Optional[str] = None  # Ingress 사용 시 경로


class PredictRequest(BaseModel):
    """예측 요청의 바디 형식을 정의합니다."""

    data: tp.List[float]  # 예측할 데이터 (예: [5.0, 3.5, 1.4, 0.2])
