import typing as tp

from pydantic import BaseModel, Field
from app.schemas.common.models_base import ResourceSpec

# ML 모델 학습 관련 하이퍼파라미터
class Hyperparameters(BaseModel):
    numEpoch: int
    learningRate: float
    numBatch: int
    # 필요에 따라 더 많은 하이퍼파라미터 추가

# 학습 Job 생성 요청 모델 (리팩토링 후)
class CreateTrainJobRequest(BaseModel):
    # Spring 서버에서 관리 및 제공하는 필드들
    taskId: str = Field(..., description="Spring Boot에서 생성한 Task ID")
    experimentName: str = Field(..., description="MLflow Experiment 이름")
    initialModelFilePath: tp.Optional[str] = Field(None, description="초기 모델 파일의 S3/MinIO 경로")
    datasetPath: str = Field(..., description="학습 데이터셋의 S3/MinIO 경로")
    hyperparameters: Hyperparameters = Field(..., description="학습 하이퍼파라미터")
    
    # 새롭게 추가된 프로필 필드
    modelProfile: str = Field(..., description="사용할 모델 프로필 이름 (예: resnet18_classification)")

    # 모델 관련 설정
    customModelName: str = Field(..., description="학습 완료 후 저장될 모델 이름")
    
    # 사용자가 프로필의 기본값을 오버라이드하고 싶을 때 선택적으로 제공
    trainerImage: tp.Optional[str] = Field(None, description="프로필의 기본 트레이너 이미지를 오버라이드")

    useGpu: bool = Field(False, description="GPU 사용 여부")

    # --- 내부 로직에서 프로필을 통해 채워지는 필드들 ---
    handlerName: tp.Optional[str] = None
    taskType: tp.Optional[str] = None
    resources: tp.Optional[ResourceSpec] = None

    class Config:
        json_schema_extra = {
            "example": {
                "taskId": "64c3a1b9-8539-4f27-a1e8-2b39b8d2c0f1",
                "experimentName": "vision-resnet-experiment",
                "initialModelFilePath": "s3://models/initial/resnet18.pth",
                "datasetPath": "s3://datasets/cifar10",
                "hyperparameters": {
                    "numEpoch": 10,
                    "learningRate": 0.001,
                    "numBatch": 32,
                },
                "modelProfile": "resnet18_classification",
                "customModelName": "my-trained-resnet",
                "trainerImage": "my-repo/train-vision:override-latest",
                "useGpu": True,
            }
        }