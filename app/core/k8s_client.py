# mlops-fastapi-app/app/core/k8s_client.py
from kubernetes import client, config
import os
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class K8sClient:
    def __init__(self):
        """
        Kubernetes 클라이언트를 초기화합니다.
        클러스터 내에서 실행될 경우 InClusterConfig를 사용하고,
        그렇지 않은 경우 (로컬 개발 등) kubeconfig를 사용합니다.
        """
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            # 클러스터 내에서 실행될 때 (Pod 내부)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        else:
            # 클러스터 외부에서 (로컬 개발 환경)
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig for local development.")
            except config.ConfigException:
                logger.error("Could not load kubeconfig. Ensure Kubernetes context is set up correctly.")
                raise # 설정 로드 실패 시 애플리케이션 시작을 중단합니다.

        self.batch_v1 = client.BatchV1Api()   # Job 관련 API
        self.core_v1 = client.CoreV1Api()    # Pod, Service 관련 API
        self.apps_v1 = client.AppsV1Api()    # Deployment 관련 API
        self.networking_v1 = client.NetworkingV1Api() # Ingress 관련 API (필요시)

        # Kubernetes 네임스페이스는 환경 변수에서 가져오거나 기본값으로 'default'를 사용
        self.namespace = os.getenv("K8S_NAMESPACE", "default")
        logger.info(f"Kubernetes client initialized for namespace: {self.namespace}")


    def _get_base_env_vars(self) -> List[client.V1EnvVar]:
        """
        MLflow 및 MinIO 관련 기본 환경 변수를 반환합니다.
        이는 Trainer Job과 Inference Deployment 모두에 사용됩니다.
        """
        return [
            client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000")),
            client.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000")),
            client.V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="minio-secret",
                        key="MINIO_ROOT_USER"
                    )
                )
            ),
            client.V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="minio-secret",
                        key="MINIO_ROOT_PASSWORD"
                    )
                )
            ),
        ]

    def create_train_job(self,
                         job_name: str,
                         image: str,
                         train_script_path: str,
                         mlflow_run_id: str,
                         initial_model_path: Optional[str] = None, # Spring Boot에서 전달받은 초기 모델 경로
                         dataset_path: Optional[str] = None, # Spring Boot에서 전달받은 데이터셋 경로
                         hyperparameters: Optional[Dict[str, str]] = None, # Spring Boot에서 전달받은 하이퍼파라미터
                         resources_requests: Optional[Dict[str, str]] = None,
                         resources_limits: Optional[Dict[str, str]] = None,
                         use_gpu: bool = False # GPU 사용 여부 플래그
                         ):
        """
        ML 학습을 위한 Kubernetes Job을 생성합니다.

        Args:
            job_name (str): 생성할 Kubernetes Job의 이름.
            image (str): 학습 컨테이너에 사용할 Docker 이미지.
            train_script_path (str): 학습 스크립트의 컨테이너 내 경로 (예: "/app/train_cpu.py").
            mlflow_run_id (str): MLflow Run ID (FastAPI에서 생성하여 전달).
            initial_model_path (str, optional): 초기 모델 파일의 S3/MinIO 경로.
            dataset_path (str, optional): 학습 데이터셋의 S3/MinIO 경로.
            hyperparameters (Dict[str, str], optional): 학습 하이퍼파라미터 딕셔너리.
            resources_requests (Dict[str, str], optional): 컨테이너 자원 요청.
            resources_limits (Dict[str, str], optional): 컨테이너 자원 제한.
            use_gpu (bool): GPU를 사용할지 여부.
        """
        logger.info(f"Attempting to create K8s Train Job: {job_name}")

        container_env = self._get_base_env_vars()
        container_env.append(client.V1EnvVar(name="MLFLOW_RUN_ID", value=mlflow_run_id))
        if initial_model_path:
            container_env.append(client.V1EnvVar(name="INITIAL_MODEL_PATH", value=initial_model_path))
        if dataset_path:
            container_env.append(client.V1EnvVar(name="DATASET_PATH", value=dataset_path))
        if hyperparameters:
            # 하이퍼파라미터를 JSON 문자열로 변환하여 환경 변수로 전달 (학습 스크립트에서 파싱)
            container_env.append(client.V1EnvVar(name="HYPERPARAMETERS_JSON", value=json.dumps(hyperparameters)))


        # 자원 요청 및 제한 설정
        resource_req = resources_requests or {"cpu": "250m", "memory": "512Mi"}
        resource_lim = resources_limits or {"cpu": "500m", "memory": "1Gi"}

        if use_gpu:
            resource_req["nvidia.com/gpu"] = "1"
            resource_lim["nvidia.com/gpu"] = "1"

        container = client.V1Container(
            name="mlflow-trainer-container",
            image=image,
            image_pull_policy="IfNotPresent", # 이미지가 존재하면 다시 다운로드 하지 않음
            command=["python"],
            args=[train_script_path],
            env=container_env,
            resources=client.V1ResourceRequirements(
                requests=resource_req,
                limits=resource_lim,
            )
        )

        pod_spec = client.V1PodSpec(
            restart_policy="OnFailure", # Job은 Pod가 실패하면 다시 시작
            containers=[container]
        )
        if use_gpu:
            # GPU를 사용한다면 runtimeClassName을 "nvidia" 등으로 설정 (클러스터 설정에 따라 다를 수 있음)
            # Nvidia Device Plugin이 설치되어 있어야 합니다.
            pod_spec.runtime_class_name = "nvidia"

        template = client.V1PodTemplateSpec(
            spec=pod_spec
        )

        spec = client.V1JobSpec(
            template=template,
            backoff_limit=3 # 실패 시 최대 3번 재시도
        )

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=spec
        )

        try:
            api_response = self.batch_v1.create_namespaced_job(body=job, namespace=self.namespace)
            logger.info(f"Successfully created K8s Job: {job_name}")
            return api_response
        except client.ApiException as e:
            logger.error(f"Error creating K8s Job {job_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error creating Job: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error creating K8s Job {job_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error creating Job: {e}")


    def get_job_status(self, job_name: str):
        """
        특정 Kubernetes Job의 상태를 조회합니다.
        """
        try:
            api_response = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=self.namespace)
            return api_response.status
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Job {job_name} not found in namespace {self.namespace}.")
                return None # Job을 찾을 수 없음
            logger.error(f"Error getting K8s Job status for {job_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error getting Job status: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error getting K8s Job status for {job_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error getting Job status: {e}")

    def delete_job(self, job_name: str):
        """
        특정 Kubernetes Job과 연관된 Pod를 삭제합니다.
        """
        logger.info(f"Attempting to delete K8s Job: {job_name}")
        try:
            # propagation_policy="Foreground"는 Job 삭제 시 Pod도 함께 삭제하도록 합니다.
            # propagation_policy="Background"는 Job만 먼저 삭제되고 Pod는 나중에 GC에 의해 삭제됩니다.
            # propagation_policy="Orphan"은 Job만 삭제되고 Pod는 남습니다.
            api_response = self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=0) # 즉시 삭제
            )
            logger.info(f"Successfully initiated deletion of K8s Job: {job_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Job {job_name} not found for deletion in namespace {self.namespace}.")
                return {"message": f"Job {job_name} not found for deletion."}
            logger.error(f"Error deleting K8s Job {job_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error deleting Job: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error deleting K8s Job {job_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error deleting Job: {e}")


    def get_pods_for_job(self, job_name: str) -> List[client.V1Pod]:
        """
        특정 Job에 의해 생성된 Pod들을 조회합니다.
        """
        try:
            # Job이 생성하는 Pod들은 'job-name' 레이블을 가집니다.
            label_selector = f"job-name={job_name}"
            api_response = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )
            return api_response.items
        except client.ApiException as e:
            logger.error(f"Error getting Pods for Job {job_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error getting Pods: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error getting Pods for Job {job_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error getting Pods: {e}")

    def get_pod_logs(self, pod_name: str, tail_lines: Optional[int] = None) -> str:
        """
        특정 Pod의 로그를 조회합니다.
        """
        try:
            api_response = self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail_lines # None이면 전체 로그, 숫자면 최근 N라인
            )
            return api_response # 로그는 문자열 형태로 반환됩니다.
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {pod_name} not found for log retrieval in namespace {self.namespace}.")
                return f"Pod {pod_name} not found or no logs available."
            logger.error(f"Error getting Pod logs for {pod_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error getting Pod logs: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error getting Pod logs for {pod_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error getting Pod logs: {e}")

    # --- Inference Deployment 관련 메서드 (선택 사항, 추후 구현) ---
    def create_inference_deployment(self,
                                   deployment_name: str,
                                   image: str,
                                   mlflow_run_id: str, # 로드할 모델의 Run ID
                                   model_file_path: Optional[str] = None, # 모델 파일 경로 (추론 서버가 직접 로드할 경우)
                                   replicas: int = 1,
                                   resources_requests: Optional[Dict[str, str]] = None,
                                   resources_limits: Optional[Dict[str, str]] = None,
                                   use_gpu: bool = False,
                                   ingress_host: Optional[str] = None,
                                   ingress_path: Optional[str] = None):
        logger.info(f"Attempting to create K8s Inference Deployment: {deployment_name}")

        container_env = self._get_base_env_vars()
        container_env.append(client.V1EnvVar(name="MLFLOW_INFERENCE_RUN_ID", value=mlflow_run_id))
        if model_file_path:
             container_env.append(client.V1EnvVar(name="MODEL_FILE_PATH", value=model_file_path))

        resource_req = resources_requests or {"cpu": "500m", "memory": "1Gi"}
        resource_lim = resources_limits or {"cpu": "1", "memory": "2Gi"}
        if use_gpu:
            resource_req["nvidia.com/gpu"] = "1"
            resource_lim["nvidia.com/gpu"] = "1"

        container = client.V1Container(
            name="inference-server-container",
            image=image,
            image_pull_policy="IfNotPresent",
            ports=[client.V1ContainerPort(container_port=8000)], # Inference 서버의 포트
            env=container_env,
            resources=client.V1ResourceRequirements(
                requests=resource_req,
                limits=resource_lim,
            )
        )
        pod_spec = client.V1PodSpec(
            containers=[container]
        )
        if use_gpu:
            pod_spec.runtime_class_name = "nvidia"

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
            spec=pod_spec
        )

        spec = client.AppsV1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(match_labels={"app": deployment_name}),
            template=template
        )

        deployment = client.AppsV1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=spec
        )

        try:
            api_response = self.apps_v1.create_namespaced_deployment(body=deployment, namespace=self.namespace)
            logger.info(f"Successfully created K8s Deployment: {deployment_name}")
            return api_response
        except client.ApiException as e:
            logger.error(f"Error creating K8s Deployment {deployment_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error creating Deployment: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error creating K8s Deployment {deployment_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error creating Deployment: {e}")

    def create_inference_service(self, service_name: str, deployment_name: str, port: int = 8000):
        logger.info(f"Attempting to create K8s Service: {service_name} for Deployment {deployment_name}")
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[client.V1ServicePort(protocol="TCP", port=port, target_port=port)],
                type="ClusterIP" # 내부 통신용
            )
        )
        try:
            api_response = self.core_v1.create_namespaced_service(body=service, namespace=self.namespace)
            logger.info(f"Successfully created K8s Service: {service_name}")
            return api_response
        except client.ApiException as e:
            logger.error(f"Error creating K8s Service {service_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error creating Service: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error creating K8s Service {service_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error creating Service: {e}")

    def create_inference_ingress(self, ingress_name: str, service_name: str, host: str, path: str, service_port: int = 8000):
        logger.info(f"Attempting to create K8s Ingress: {ingress_name} for Service {service_name}")
        ingress = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(name=ingress_name),
            spec=client.V1IngressSpec(
                rules=[
                    client.V1IngressRule(
                        host=host,
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path=path,
                                    path_type="Prefix", # 또는 Exact
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=service_name,
                                            port=client.V1ServiceBackendPort(number=service_port)
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )
        try:
            api_response = self.networking_v1.create_namespaced_ingress(body=ingress, namespace=self.namespace)
            logger.info(f"Successfully created K8s Ingress: {ingress_name}")
            return api_response
        except client.ApiException as e:
            logger.error(f"Error creating K8s Ingress {ingress_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error creating Ingress: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error creating K8s Ingress {ingress_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error creating Ingress: {e}")

    def delete_deployment(self, deployment_name: str):
        logger.info(f"Attempting to delete K8s Deployment: {deployment_name}")
        try:
            api_response = self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=0)
            )
            logger.info(f"Successfully initiated deletion of K8s Deployment: {deployment_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Deployment {deployment_name} not found for deletion in namespace {self.namespace}.")
                return {"message": f"Deployment {deployment_name} not found for deletion."}
            logger.error(f"Error deleting K8s Deployment {deployment_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error deleting Deployment: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error deleting K8s Deployment {deployment_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error deleting Deployment: {e}")

    def delete_service(self, service_name: str):
        logger.info(f"Attempting to delete K8s Service: {service_name}")
        try:
            api_response = self.core_v1.delete_namespaced_service(
                name=service_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=0)
            )
            logger.info(f"Successfully initiated deletion of K8s Service: {service_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Service {service_name} not found for deletion in namespace {self.namespace}.")
                return {"message": f"Service {service_name} not found for deletion."}
            logger.error(f"Error deleting K8s Service {service_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error deleting Service: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error deleting K8s Service {service_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error deleting Service: {e}")

    def delete_ingress(self, ingress_name: str):
        logger.info(f"Attempting to delete K8s Ingress: {ingress_name}")
        try:
            api_response = self.networking_v1.delete_namespaced_ingress(
                name=ingress_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground", grace_period_seconds=0)
            )
            logger.info(f"Successfully initiated deletion of K8s Ingress: {ingress_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"Ingress {ingress_name} not found for deletion in namespace {self.namespace}.")
                return {"message": f"Ingress {ingress_name} not found for deletion."}
            logger.error(f"Error deleting K8s Ingress {ingress_name}: {e.body}", exc_info=True)
            raise Exception(f"K8s API Error deleting Ingress: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(f"Unexpected error deleting K8s Ingress {ingress_name}: {e}", exc_info=True)
            raise Exception(f"Unexpected error deleting Ingress: {e}")


# FastAPI 앱에서 사용할 K8sClient 인스턴스를 미리 생성
k8s_client = K8sClient()