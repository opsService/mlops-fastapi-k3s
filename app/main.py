from fastapi import FastAPI, HTTPException, status
import logging

# 라우터 임포트
from app.routers import train, inference

# 로거 설정 (FastAPI 앱의 로거)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps FastAPI API",
    description="Kubernetes 기반 MLOps 파이프라인을 위한 FastAPI 백엔드 API",
    version="1.0.0"
)

# 라우터들을 FastAPI 앱에 포함시킵니다.
# 각 라우터 파일(train.py, inference.py)에 정의된 엔드포인트들이 이 경로 아래에 추가됩니다.
app.include_router(train.router, prefix="/internal/api/v1/k8s/train", tags=["Training Orchestration"])
app.include_router(inference.router, prefix="/internal/api/v1/k8s/inference", tags=["Inference Orchestration"])

# ⭐ 추가: Health Check 엔드포인트
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    FastAPI 애플리케이션의 헬스 상태를 확인합니다.
    Kubernetes Liveness/Readiness Probe에서 사용됩니다.
    """
    logger.debug("FastAPI health check requested.")
    return {"status": "healthy"}

@app.get("/")
async def root():
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