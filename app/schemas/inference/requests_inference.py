import typing as tp

from pydantic import BaseModel, Field

# 추론 Job 배포 요청 모델
class DeployInferenceRequest(BaseModel):
    """추론 태스크 배포 요청의 바디 형식을 정의합니다."""

    taskId: str = Field(..., description="Spring Boot에서 생성한 Task ID")
    modelId: str = Field(..., description="사용자가 선택한 모델 ID")
    modelFilePath: str = Field(..., description="모델 파일 저장 경로 (S3/MinIO)")
    mlflowRunId: tp.Optional[str] = Field(
        None, description="(선택 사항) 커스텀 학습 모델인 경우 MLflow Run ID"
    )
    modelProfile: str = Field(..., description="사용할 모델 프로필 이름 (예: resnet18_classification)")
    useGpu: bool = False
    ingressHost: tp.Optional[str] = None
    ingressPath: tp.Optional[str] = None

# 단일 엔드포인트에서 Request 객체를 직접 사용하므로, 이 Pydantic 모델은 더 이상 직접 사용되지 않습니다.
# API의 Body 형식을 설명하기 위한 흔적으로 남겨둡니다.
# class PredictRequest(BaseModel):
#     """예측 요청의 바디 형식을 정의합니다."""
#
#     userId: str = Field(..., description="추론을 요청한 사용자의 ID")
#     # 다양한 입력 형식(테이블, 이미지 등)을 지원하기 위해 Any로 변경
#     data: tp.Any  # 예측할 데이터
