# FastAPI 기반 Kubernetes MLOps 플랫폼

## 1. 프로젝트 개요

본 프로젝트는 FastAPI를 사용하여 모델 학습 및 추론 파이프라인을 동적으로 관리하는 MLOps 플랫폼입니다. Kubernetes 클러스터 위에서 동작하며, MLflow를 통한 실험 관리 및 모델 등록, MinIO를 통한 아티팩트 저장 등 MLOps의 핵심 구성요소를 통합하여 자동화된 머신러닝 워크플로우를 제공합니다.

사용자는 간단한 API 요청만으로 모델 학습 Job을 실행하고, 학습이 완료된 모델을 즉시 추론 API 서버로 배포할 수 있습니다.

## 2. 시스템 아키텍처

본 플랫폼은 다음과 같은 마이크로서비스 아키텍처로 구성됩니다.

**핵심 구성 요소:**

*   **FastAPI MLOps API (`fastapi-mlops-api`)**: 
    *   사용자 요청을 받는 중앙 API 게이트웨이입니다.
    *   학습(Train) 및 배포(Deploy) 요청을 받아 Kubernetes 리소스를 동적으로 생성하고 관리합니다.
    *   추론(Predict) 요청을 받아서 해당 모델이 배포된 추론 서버로 전달하는 프록시(Proxy) 역할을 수행합니다.
*   **Trainer (Kubernetes Job)**:
    *   API 요청 시 동적으로 생성되는 일회성 학습 실행 환경입니다.
    *   지정된 Docker 이미지와 소스 코드를 사용하여 모델 학습을 수행하고, 완료 후 종료됩니다.
    *   학습 과정, 파라미터, 결과 메트릭을 MLflow에 자동으로 기록합니다.
*   **Inference Server (Kubernetes Deployment)**:
    *   학습된 모델을 서빙하는 API 서버입니다.
    *   MLflow에 등록된 모델을 불러와 실시간 추론 서비스를 제공하며, 독립적으로 실행됩니다.
    *   필요에 따라 Scale-out/in이 가능합니다.
*   **MLflow & Backends**:
    *   **MLflow Tracking Server**: 모든 학습 실험의 파라미터, 메트릭, 아티팩트를 추적하고 모델을 등록/관리하는 Model Registry 역할을 수행합니다.
    *   **PostgreSQL**: MLflow가 사용하는 메타데이터(실험 정보, 모델 정보 등)를 저장하는 데이터베이스입니다.
    *   **MinIO**: MLflow가 생성하는 아티팩트(모델 파일, 이미지 등)를 저장하는 S3 호환 오브젝트 스토리지입니다.

## 3. 주요 기능

*   **RESTful API**: FastAPI를 통해 학습, 배포, 추론 파이프라인을 제어하는 표준 API 제공
*   **동적 학습 환경**: 요청에 따라 K8s Job을 동적으로 생성하여 리소스 효율성 극대화
*   **자동화된 실험 관리**: MLflow와 연동하여 모든 학습 과정을 자동으로 기록 및 추적
*   **원클릭 모델 배포**: MLflow에 등록된 모델을 API 호출 한 번으로 추론 서버에 배포
*   **모델 프로필 관리**: `model_profiles.yaml`을 통해 다양한 종류의 모델과 실행 환경(Docker 이미지, 리소스)을 유연하게 관리
*   **GPU 지원**: GPU를 사용하는 학습 및 추론 작업을 지원

## 4. 디렉토리 구조

```
.
├── app/                # FastAPI 애플리케이션 소스 코드
│   ├── core/           # K8s 클라이언트, 오케스트레이터 등 핵심 로직
│   ├── routers/        # API 엔드포인트 (라우터)
│   ├── schemas/        # Pydantic 데이터 모델 (요청/응답)
│   └── main.py         # FastAPI 앱 진입점
├── docker/             # 서비스별 Dockerfile
├── kubernetes/         # K8s 리소스 배포용 YAML 파일
├── models/             # 모델별 데이터 처리 및 학습 핸들러
├── requirements/       # Python 의존성 목록
├── script/             # 학습 스크립트 (train.py)
└── README.md           # 프로젝트 설명서
```

## 5. 설치 및 실행

### 사전 요구사항

*   `kubectl`이 설치된 환경
*   실행 중인 Kubernetes 클러스터 (예: k3s, Minikube, EKS, GKE)
*   (선택) Ingress Controller (예: NGINX Ingress Controller)

### 배포 절차

1.  **Docker 이미지 빌드 및 푸시**
    `docker/` 디렉토리의 Dockerfile들을 사용하여 각 서비스의 이미지를 빌드하고, 사용하는 컨테이너 레지스트리(예: Docker Hub, ECR, Harbor)에 푸시해야 합니다.
    *   `Dockerfile.fastapi`
    *   `Dockerfile.gpu-base`
    *   `Dockerfile.inference`
    *   `Dockerfile.trainer`

    *YAML 파일들(`fastapi-mlops-api.yaml` 등)에 명시된 이미지 주소를 실제 푸시한 이미지 주소로 수정해야 합니다.*

2.  **전체 서비스 배포**
    프로젝트 루트의 스크립트를 사용하여 모든 쿠버네티스 리소스를 배포합니다.

    ```bash
    # 네임스페이스를 지정하여 배포 (예: mlops)
    ./deploy_all_k3s.sh mlops

    # 기본(default) 네임스페이스에 배포
    ./deploy_all_k3s.sh
    ```

3.  **배포 상태 확인**
    ```bash
    ./monitor_all.sh mlops
    ```

## 6. API 사용법

FastAPI 서버가 배포되면 `fastapi-mlops-api-service`를 통해 접근할 수 있습니다. (Port-forwarding 또는 Ingress 필요)

```bash
# Port-forwarding 예시
kubectl port-forward svc/fastapi-mlops-api-service 8000:80 -n mlops
```
이제 `http://localhost:8000/docs` 에서 자동 생성된 API 문서를 확인할 수 있습니다.

--- 

### 1단계: 모델 학습

*   **Endpoint**: `POST /api/v1/train/jobs`
*   **Description**: 새로운 모델 학습 Job을 생성합니다.
*   **Request Body 예시**:
    ```json
    {
      "taskId": "train-reg-01",
      "experimentName": "House Price Prediction",
      "modelProfile": "tabular_regression",
      "customModelName": "predictorHouseValue",
      "initialModelFilePath": "None",
      "datasetPath": "s3://datasets/house-prices.csv",
      "trainerImage": "localhost:5002/mlflow-trainer:latest",
      "useGpu": false,
      "hyperparameters": {
        "numEpoch": 100,
        "learningRate": 0.001,
        "numBatch": 32
      }
    }
    ```

--- 

### 2단계: 모델 배포

학습이 완료되면 MLflow UI에서 `runId`를 확인하여 모델을 배포합니다.

*   **Endpoint**: `POST /api/v1/inference/deployments`
*   **Description**: 학습된 모델을 추론 서버로 배포합니다.
*   **Request Body 예시**:
    ```json
    {
      "taskId": "deploy-reg-01",
      "modelId": "predictor-house-v1",
      "mlflowRunId": "90df8a7a263149d2bb86a71be7442611",
      "modelProfile": "tabular_regression",
      "useGpu": false
    }
    ```
    *   `modelId`는 배포되는 서비스를 식별하는 고유한 이름입니다. 쿠버네티스 리소스 이름에 사용되므로 **소문자, 숫자, 하이픈(-)만 사용**해야 합니다.

--- 

### 3단계: 모델 추론 (예측)

모델 배포 시 사용했던 `taskId`를 사용하여 예측을 요청합니다.

*   **Endpoint**: `POST /api/v1/inference/{task_id}/predict`
*   **Description**: 배포된 모델에 예측을 요청합니다. FastAPI 백엔드가 요청을 받아 해당 추론 서버로 전달(proxy)합니다.
*   **URL의 `{task_id}`**: 2단계 배포 시 사용했던 `taskId` (예: `deploy-reg-01`)를 입력합니다.
*   **Request Body 예시**:
    ```json
    {
      "userId": "user-1234",
      "data": {
      "features": [
            [5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2]
        ]
      }
    }
    ```
    *   **중요**: 요청 Body는 모델이 기대하는 입력 형식에 정확히 맞춰야 합니다. 현재 `tabular_regression` 모델은 `features` 키와 2차원 배열을 기대합니다.

--- 

### 기타 관리 API

*   `GET /train/jobs/{task_id}/status`: 특정 학습 Job의 상태를 조회합니다.
*   `GET /train/jobs/{task_id}/logs`: 특정 학습 Job의 로그를 조회합니다.
*   `DELETE /inference/deployments/{task_id}`: 배포된 추론 서버를 삭제합니다.

## 7. 리소스 정리

배포된 모든 플랫폼 리소스를 삭제하려면 다음 스크립트를 사용합니다.

```bash
./clear_all_k3s.sh mlops
```
