import os


class Settings:
    # FastAPI 내부 인증 API 키
    INTERNAL_API_KEY: str = os.getenv(
        "INTERNAL_API_KEY", "your-super-secret-internal-api-key"
    )

    # MLflow 설정
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000"
    )
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"
    )
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")  # K8s Secret에서 로드됨
    AWS_SECRET_ACCESS_KEY: str = os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )  # K8s Secret에서 로드됨

    # Spring Boot 콜백 URL
    SPRING_BOOT_CALLBACK_URL: str = os.getenv(
        "SPRING_BOOT_CALLBACK_URL", "http://backend-service:8080/internal/api/v1/tasks"
    )

    # Kubernetes 설정
    K8S_NAMESPACE: str = os.getenv("K8S_NAMESPACE", "default")
    # K8s Job 이미지 이름 (기본값 설정 또는 환경 변수로 받기)
    DEFAULT_TRAINER_IMAGE: str = os.getenv(
        "DEFAULT_TRAINER_IMAGE", "localhost:5002/heedong/mlflow-trainer:latest"
    )
    DEFAULT_INFERENCE_IMAGE: str = os.getenv(
        "DEFAULT_INFERENCE_IMAGE",
        "localhost:5002/heedong/mlflow-inference-server:latest",
    )

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # INFO, DEBUG, WARNING, ERROR 등


# 전역 설정 객체
settings = Settings()
