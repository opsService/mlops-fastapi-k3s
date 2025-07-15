import asyncio
import logging
import os
import typing as tp

# DB, MinIO
import asyncpg  # PostgreSQL
from app.core.config import settings
from app.core.logging_config import setup_logging

# 라우터 임포트
from app.routers import inference, train
from fastapi import FastAPI, HTTPException, status
from minio import Minio  # MinIO

# 로거 설정 (FastAPI 앱의 로거)
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps FastAPI API",
    description="Kubernetes 기반 MLOps 파이프라인을 위한 FastAPI 백엔드 API",
    version="1.0.0",
)

# 라우터들을 FastAPI 앱에 포함시킵니다.
# 각 라우터 파일(train.py, inference.py)에 정의된 엔드포인트들이 이 경로 아래에 추가됩니다.
app.include_router(
    train.router, prefix="/internal/api/v1/k8s/train", tags=["Training Orchestration"]
)
app.include_router(
    inference.router,
    prefix="/internal/api/v1/k8s/inference",
    tags=["Inference Orchestration"],
)


# ⭐ 추가: Health Check 엔드포인트
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> tp.Dict:
    """
    FastAPI 애플리케이션의 헬스 상태를 확인합니다.
    Kubernetes Liveness/Readiness Probe에서 사용됩니다.
    의존 서비스(PostgreSQL, MinIO, MLflow Tracking)의 연결 상태도 확인합니다.
    """
    logger.debug("FastAPI health check requested.")
    health_status = {"status": "healthy", "dependencies": {}}

    # --- 1. PostgreSQL 연결 확인 ---
    try:
        conn = await asyncpg.connect(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
        )
        await conn.close()
        health_status["dependencies"]["postgresql"] = {"status": "connected"}
        logger.debug("PostgreSQL connection successful.")
    except Exception as e:
        health_status["dependencies"]["postgresql"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"PostgreSQL connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status
        )

    # --- 2. MinIO 연결 확인 (버킷 존재 여부까지 확인) ---
    try:
        # MinIO endpoint에서 포트 제거 (MinioClient는 endpoint에 포트를 포함하지 않음)
        minio_host_port = settings.MINIO_ENDPOINT.split(":")
        minio_host = minio_host_port[0]
        minio_port = int(minio_host_port[1]) if len(minio_host_port) > 1 else 9000

        minio_client = Minio(
            f"{minio_host}:{minio_port}",  # 호스트:포트 형식
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,  # K3s 내부에서는 HTTP 사용 (False)
        )
        # 버킷이 존재하는지 확인 (예: mlflow-artifacts 버킷)
        if not minio_client.bucket_exists("mlflow-artifacts"):
            raise Exception(
                "MLflow artifacts bucket does not exist or is not accessible."
            )
        health_status["dependencies"]["minio"] = {"status": "connected"}
        logger.debug("MinIO connection and bucket check successful.")
    except Exception as e:
        health_status["dependencies"]["minio"] = {"status": "failed", "error": str(e)}
        logger.error(f"MinIO connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status
        )

    # --- 3. MLflow Tracking Server 연결 확인 ---
    # 간단한 HTTP GET 요청으로 서버 응답 확인
    try:
        import httpx  # requests 대신 비동기 http 클라이언트 사용 권장

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.MLFLOW_TRACKING_URI}/health", timeout=5
            )  # MLflow는 보통 /health 엔드포인트 제공
            response.raise_for_status()  # 2xx 응답이 아니면 예외 발생
        health_status["dependencies"]["mlflow_tracking"] = {"status": "connected"}
        logger.debug("MLflow Tracking Server connection successful.")
    except Exception as e:
        health_status["dependencies"]["mlflow_tracking"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"MLflow Tracking Server connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status
        )

    return health_status


@app.get("/")
async def root() -> tp.Dict:
    """
    API의 루트 경로입니다. 서비스가 정상적으로 실행 중인지 확인합니다.
    """
    return {"message": "MLOps Workflow API is running!"}


# FastAPI 애플리케이션 시작 시 이벤트
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup event triggered.")
    # 필요한 초기화 로직 (예: DB 연결, 캐시 로드 등)
    pass


# FastAPI 애플리케이션 종료 시 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown event triggered.")
    # 필요한 정리 로직 (예: DB 연결 해제 등)
    pass


# FastAPI가 Spring Boot로 콜백할 때 사용할 엔드포인트는 여기에 정의하지 않습니다.
# 콜백은 FastAPI 내부 로직에서 requests 라이브러리를 사용하여 Spring Boot API를 호출하는 방식입니다.
