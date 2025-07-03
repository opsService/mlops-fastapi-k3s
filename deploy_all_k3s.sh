#!/bin/bash

# 이 스크립트는 Kubernetes 클러스터에 MLOps 컴포넌트들을 순서대로 배포합니다.
# 각 단계 후에 Pod가 'Running' 상태가 될 때까지 기다립니다.

NAMESPACE="default" # 사용할 네임스페이스

# --- 1. 시크릿 생성 ---
echo "--- 1. Applying secrets ---"
kubectl apply -f kubernetes/secrets.yaml -n $NAMESPACE
kubectl apply -f kubernetes/fastapi-api-secret.yaml -n $NAMESPACE
echo "Secrets applied."

# --- 2. ServiceAccount 생성 ---
echo "--- 2. Applying service account ---"
kubectl apply -f kubernetes/serviceaccount.yaml -n $NAMESPACE
echo "Service Account applied."
echo "Checking service account status..."
kubectl get sa -n $NAMESPACE
echo "Service Account check complete."

# --- 3. PersistentVolume 및 PersistentVolumeClaim 생성 ---
echo "--- 3. Applying PV and PVC ---"
kubectl apply -f kubernetes/pvpvc_deployment.yaml -n $NAMESPACE
echo "PV and PVC applied. Waiting for PVs/PVCs to be bound (this might take a moment)..."
# PV/PVC 바운드 상태 확인 (HostPath는 보통 즉시 Bound 됨)
echo "PV and PVC applied. PVC binding will be implicitly handled by respective deployments."


# --- 4. 핵심 인프라 서비스 배포 (MinIO, PostgreSQL) ---
echo "--- 4. Applying core infrastructure deployments and services (MinIO, PostgreSQL) ---"
kubectl apply -f kubernetes/minio_deplyment.yaml -n $NAMESPACE
kubectl apply -f kubernetes/postgresql_deployment.yaml -n $NAMESPACE # 파일 경로 확인: yaml_pod_deployment/postgresql_deployment.yaml -> kubernetes/postgresql.yaml

echo "Waiting for MinIO and PostgreSQL pods to be ready..."
kubectl wait --for=condition=Ready pod -l app=minio --timeout=300s -n $NAMESPACE || { echo "MinIO pod not ready"; exit 1; }
kubectl wait --for=condition=Ready pod -l app=postgresql --timeout=300s -n $NAMESPACE || { echo "PostgreSQL pod not ready"; exit 1; }
echo "MinIO and PostgreSQL pods are running."

# --- 5. MLflow Tracking Server 배포 ---
echo "--- 5. Applying MLflow Tracking Server deployment and service ---"
kubectl apply -f kubernetes/mlflow_tracking_deployment.yaml -n $NAMESPACE # 파일 경로 확인: fastAPI_backend/kubernetes/mlflow_tracking_deployment.yaml -> kubernetes/mlflow-tracking-server.yaml

echo "Waiting for MLflow Tracking Server pod to be ready..."
kubectl wait --for=condition=Ready pod -l app=mlflow-tracking --timeout=300s -n $NAMESPACE || { echo "MLflow Tracking pod not ready"; exit 1; }
echo "MLflow Tracking Server pod is running."

# --- 6. RBAC 권한 적용 ---
echo "--- 6. Applying RBAC roles and rolebindings ---"
kubectl apply -f kubernetes/rbac.yaml -n $NAMESPACE
echo "RBAC applied."

# --- 7. FastAPI Deployment 및 Service 적용 ---
echo "--- 7. Applying FastAPI deployment and service ---"
kubectl apply -f kubernetes/fastapi-mlops-api.yaml -n $NAMESPACE

echo "Waiting for FastAPI pod to be ready..."
kubectl wait --for=condition=Ready pod -l app=fastapi-mlops-api --timeout=300s -n $NAMESPACE || { echo "FastAPI pod not ready"; exit 1; }
echo "FastAPI pod is running."

# --- 8. MLflow Inference Server Deployment 및 Service 적용 (기본 추론 서버) ---
echo "--- 8. Applying MLflow Inference Server deployment and service ---"
kubectl apply -f kubernetes/mlflow-inference-server.yaml -n $NAMESPACE

echo "Waiting for MLflow Inference Server pod to be ready..."
kubectl wait --for=condition=Ready pod -l app=mlflow-inference-server --timeout=300s -n $NAMESPACE || { echo "MLflow Inference Server pod not ready"; exit 1; }
echo "MLflow Inference Server pod is running."

echo "--- All Kubernetes components deployed successfully! ---"
echo "Final check of all resources:"
kubectl get all -n $NAMESPACE