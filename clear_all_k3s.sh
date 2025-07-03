#!/bin/bash

# 이 스크립트는 모든 Kubernetes 리소스를 삭제하고 정의된 순서대로 다시 배포합니다.
# 개발 환경에서 클러스터 상태를 깨끗하게 초기화할 때 유용합니다.

NAMESPACE="default" # 사용할 네임스페이스

echo "--- 1. 모든 Kubernetes 리소스 정리 시작 ---"

# 모든 Deployment 삭제
echo "Deleting all deployments..."
kubectl delete deployment --all -n $NAMESPACE --ignore-not-found

# 모든 Service 삭제 (kubernetes 기본 서비스 제외)
echo "Deleting all services (excluding 'kubernetes')..."
kubectl get svc -n $NAMESPACE -o name | grep -v kubernetes | xargs -r kubectl delete -n $NAMESPACE

# 모든 Ingress 삭제
echo "Deleting all ingresses..."
kubectl delete ingress --all -n $NAMESPACE --ignore-not-found

# 모든 Secret 삭제 (기본 Secret 제외)
echo "Deleting all secrets (excluding default tokens and helm releases)..."
kubectl get secret -n $NAMESPACE -o name | grep -v "default-token" | grep -v "sh.helm.release" | xargs -r kubectl delete -n $NAMESPACE

# 모든 ServiceAccount 삭제 (기본 ServiceAccount 제외)
echo "Deleting all serviceaccounts (excluding 'default')..."
kubectl get sa -n $NAMESPACE -o name | grep -v "default" | xargs -r kubectl delete -n $NAMESPACE

# 모든 Role 삭제
echo "Deleting all roles..."
kubectl delete role --all -n $NAMESPACE --ignore-not-found

# 모든 RoleBinding 삭제
echo "Deleting all rolebindings..."
kubectl delete rolebinding --all -n $NAMESPACE --ignore-not-found

# 모든 Job 삭제
echo "Deleting all jobs..."
kubectl delete job --all -n $NAMESPACE --ignore-not-found

# 모든 Pod 삭제 (혹시 남아있다면)
echo "Deleting all pods..."
kubectl delete pod --all -n $NAMESPACE --ignore-not-found

# PersistentVolumeClaim (PVC) 삭제 - PV는 Retain 정책이므로 수동 삭제 필요
echo "Deleting all PersistentVolumeClaims..."
kubectl delete pvc --all -n $NAMESPACE --ignore-not-found

# PersistentVolume (PV) 삭제 - Retain 정책이므로 수동으로 삭제해야 합니다.
# 주의: hostPath PV를 사용하므로, 실제 데이터는 노드에 남아있습니다.
echo "Deleting all PersistentVolumes..."
kubectl delete pv --all --ignore-not-found

echo "--- 1. 모든 Kubernetes 리소스 정리 완료 ---"
echo "클러스터가 깨끗한지 확인 중..."
kubectl get all -n $NAMESPACE
echo "--- 확인 완료 ---"