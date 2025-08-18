import typing as tp

from pydantic import BaseModel, Field

# 추론 Job 배포 요청 모델 (리팩토링 후)
class DeployInferenceRequest(BaseModel):
    """추론 태스크 배포 요청의 바디 형식을 정의합니다."""

    taskId: str = Field(..., description="Spring Boot에서 생성한 Task ID")
    # userId: str = Field(..., description="요청을 보낸 사용자의 ID") # 삭제
    modelId: str = Field(..., description="사용자가 선택한 모델 ID")
    modelFilePath: str = Field(..., description="모델 파일 저장 경로 (S3/MinIO)")
    mlflowRunId: tp.Optional[str] = Field(
        None, description="(선택 사항) 커스텀 학습 모델인 경우 MLflow Run ID"
    )
    
    # 프로필 기반으로 변경
    modelProfile: str = Field(..., description="사용할 모델 프로필 이름 (예: resnet18_classification)")

    useGpu: bool = False  # GPU 사용 여부
    ingressHost: tp.Optional[str] = None  # Ingress 사용 시 호스트
    ingressPath: tp.Optional[str] = None  # Ingress 사용 시 경로


class PredictRequest(BaseModel):
    """예측 요청의 바디 형식을 정의합니다."""

    userId: str = Field(..., description="추론을 요청한 사용자의 ID")
    # 다양한 입력 형식(테이블, 이미지 등)을 지원하기 위해 Any로 변경
    data: tp.Any  # 예측할 데이터