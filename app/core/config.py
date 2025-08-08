# mlops-fastapi-app/app/core/config.py
import os
from functools import lru_cache  # Settings 객체 캐싱을 위해 추가


class Settings:
    # FastAPI 내부 인증 API 키
    # 보안상 기본값을 제거하고, 환경 변수가 필수로 설정되도록 강제
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY")
    if INTERNAL_API_KEY is None:
        raise ValueError("INTERNAL_API_KEY 환경 변수가 설정되지 않았습니다. 보안상 필수입니다.")

    # MLflow 설정
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000"
    )
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"
    )
    # AWS_ACCESS_KEY_ID 및 AWS_SECRET_ACCESS_KEY는 K8s Secret에서 로드되므로,
    # 코드 내 기본값 없이 필수적으로 설정되도록 강제 (MinIO 연동에 필요)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    if AWS_ACCESS_KEY_ID is None or AWS_SECRET_ACCESS_KEY is None:
        raise ValueError(
            "AWS_ACCESS_KEY_ID 또는 AWS_SECRET_ACCESS_KEY 환경 변수가 설정되지 않았습니다."
            " MinIO/S3 연동에 필수입니다. Kubernetes Secret을 통해 주입하세요."
        )

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
        "DEFAULT_INFERENCE_IMAGE", "localhost:5002/heedong/mlflow-inference-server:final-debug-v1"
    )

    # 로깅 레벨 설정 추가 (logging_config.py에서 사용)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Postgres 설정 (main.py에서 사용되던 환경 변수를 이곳으로 이동)
    POSTGRES_HOST: str = os.getenv("POSTGRES_SERVICE_HOST", "postgresql-service")
    POSTGRES_PORT: str = os.getenv("POSTGRES_SERVICE_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "mlflow_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    if POSTGRES_USER is None or POSTGRES_PASSWORD is None:
        raise ValueError(
            "POSTGRES_USER 또는 POSTGRES_PASSWORD 환경 변수가 설정되지 않았습니다."
            " PostgreSQL 연동에 필수입니다. Kubernetes Secret을 통해 주입하세요."
        )

    # MinIO 설정 (main.py에서 사용되던 환경 변수를 이곳으로 이동)
    MINIO_ENDPOINT: str = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"
    ).replace("http://", "") # http:// 제거

# Settings 객체를 싱글톤처럼 사용하여 애플리케이션 전역에서 일관되게 접근
@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()