from pydantic import BaseModel, Field


# ML 모델 학습 관련 하이퍼파라미터
class Hyperparameters(BaseModel):
    numEpoch: int
    learningRate: float
    numBatch: int
    # 필요에 따라 더 많은 하이퍼파라미터 추가


# 학습 Job 생성 요청 모델
class CreateTrainJobRequest(BaseModel):
    taskId: str = Field(..., description="Spring Boot에서 생성한 Task ID")
    experimentName: str = Field(..., description="MLflow Experiment 이름")
    initialModelId: str = Field(
        ..., description="초기 모델 ID (Pre-trained/Custom-trained)"
    )
    initialModelFilePath: str = Field(..., description="초기 모델 파일의 S3/MinIO 경로")
    datasetPath: str = Field(..., description="학습 데이터셋의 S3/MinIO 경로")
    hyperparameters: Hyperparameters = Field(..., description="학습 하이퍼파라미터")
    customModelName: str = Field(..., description="학습 완료 후 저장될 모델 이름")
    userId: str = Field(..., description="Task를 생성한 사용자 ID")
    useGpu: bool = False
    trainerImage: str = (
        "localhost:5002/heedong/mlflow-trainer:linear-regression-test-cpu"  # 학습 컨테이너 이미지 (CPU 기본)
    )
    trainScriptPath: str = "/app/train_cpu.py"  # 학습 스크립트 컨테이너 내 경로
