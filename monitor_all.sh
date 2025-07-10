#!/bin/bash
echo "Starting port-forwarding..."

# 포워딩 백그라운드 실행
kubectl port-forward svc/minio-service 9001:9001 -n default > temp/minio.log 2>&1 &
MINIO_PID=$!

kubectl port-forward svc/mlflow-tracking-service 5000:5000 -n default > temp/mlflow.log 2>&1 &
MLFLOW_PID=$!

kubectl port-forward svc/fastapi-mlops-api-service 8000:8000 -n default > temp/fastapi.log 2>&1 &
FASTAPI_PID=$!

# PID 저장 및 관리
echo "MINIO PID: $MINIO_PID"
echo "MLFLOW PID: $MLFLOW_PID"
echo "FASTAPI PID: $FASTAPI_PID"

# 포워딩 종료 시 함께 종료하도록 트랩 설정
trap "echo 'Stopping...'; kill $MINIO_PID $MLFLOW_PID $FASTAPI_PID" SIGINT

# 무한 대기
wait
