# mlops-fastapi-app/app/main.py
from fastapi import FastAPI
from app.routers import train, inference # 앞으로 만들 라우터 모듈 임포트

app = FastAPI(title="MLOps Workflow API")

# 라우터들을 FastAPI 앱에 포함시킵니다.
# 각 라우터 파일(train.py, inference.py)에 정의된 엔드포인트들이 이 경로 아래에 추가됩니다.
app.include_router(train.router, prefix="/internal/api/v1/k8s/train", tags=["Training Orchestration"])
app.include_router(inference.router, prefix="/internal/api/v1/k8s/inference", tags=["Inference Orchestration"])

@app.get("/")
async def root():
    """
    API의 루트 경로입니다. 서비스가 정상적으로 실행 중인지 확인합니다.
    """
    return {"message": "MLOps Workflow API is running!"}

# FastAPI가 Spring Boot로 콜백할 때 사용할 엔드포인트는 여기에 정의하지 않습니다.
# 콜백은 FastAPI 내부 로직에서 requests 라이브러리를 사용하여 Spring Boot API를 호출하는 방식입니다.