# mlops-fastapi-app/app/core/k8s_client.py
import json
import logging
import os
import typing as tp

from kubernetes import client, config

# 로거 설정
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
            logger.info("클러스터 내 Kubernetes 설정 로드 완료.")
        else:
            # 클러스터 외부에서 (로컬 개발 환경)
            try:
                config.load_kube_config()
                logger.info("로컬 개발용 kubeconfig 로드 완료.")
            except config.ConfigException:
                logger.error(
                    "kubeconfig를 로드할 수 없습니다. Kubernetes 컨텍스트가 올바르게 설정되었는지 확인하세요."
                )
                raise  # 설정 로드 실패 시 애플리케이션 시작을 중단합니다.

        self.batch_v1 = client.BatchV1Api()  # Job 관련 API
        self.core_v1 = client.CoreV1Api()  # Pod, Service 관련 API
        self.apps_v1 = client.AppsV1Api()  # Deployment 관련 API
        self.networking_v1 = client.NetworkingV1Api()  # Ingress 관련 API (필요시)

        # Kubernetes 네임스페이스는 환경 변수에서 가져오거나 기본값으로 'default'를 사용
        self.namespace = os.getenv("K8S_NAMESPACE", "default")
        logger.info(
            f"Kubernetes 클라이언트가 네임스페이스 '{self.namespace}'에 대해 초기화되었습니다."
        )

    def _get_base_env_vars(self) -> tp.List[client.V1EnvVar]:
        """
        MLflow 및 MinIO 관련 기본 환경 변수를 반환합니다.
        이는 Trainer Job과 Inference Deployment 모두에 사용됩니다.
        """
        return [
            client.V1EnvVar(
                name="MLFLOW_TRACKING_URI",
                value=os.getenv(
                    "MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000"
                ),
            ),
            client.V1EnvVar(
                name="MLFLOW_S3_ENDPOINT_URL",
                value=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"),
            ),
            client.V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="minio-secret", key="MINIO_ROOT_USER"
                    )
                ),
            ),
            client.V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="minio-secret", key="MINIO_ROOT_PASSWORD"
                    )
                ),
            ),
            client.V1EnvVar(
                name="INTERNAL_API_KEY",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="fastapi-internal-api-key-secret", key="API_KEY"
                    )
                ),
            ),
            client.V1EnvVar(
                name="POSTGRES_USER",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="postgres-secret", key="POSTGRES_USER"
                    )
                ),
            ),
            client.V1EnvVar(
                name="POSTGRES_PASSWORD",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="postgres-secret", key="POSTGRES_PASSWORD"
                    )
                ),
            ),
        ]

    def create_train_job(
        self,
        job_name: str,
        image: str,
        train_script_path: str,  # 빼야할지도
        mlflow_run_id: str,
        initial_model_path: tp.Optional[str] = None,
        dataset_path: tp.Optional[str] = None,
        hyperparameters: tp.Optional[tp.Dict[str, str]] = None,
        resources_requests: tp.Optional[tp.Dict[str, str]] = None,
        resources_limits: tp.Optional[tp.Dict[str, str]] = None,
        use_gpu: bool = False,
    ):
        """
        ML 학습을 위한 Kubernetes Job을 생성합니다.
        """
        logger.info(f"Kubernetes 학습 Job 생성 시도: {job_name}")

        container_env = self._get_base_env_vars()
        container_env.append(client.V1EnvVar(name="MLFLOW_RUN_ID", value=mlflow_run_id))
        if initial_model_path:
            container_env.append(
                client.V1EnvVar(name="INITIAL_MODEL_PATH", value=initial_model_path)
            )
        if dataset_path:
            container_env.append(
                client.V1EnvVar(name="DATASET_PATH", value=dataset_path)
            )
        if hyperparameters:
            container_env.append(
                client.V1EnvVar(
                    name="HYPERPARAMETERS_JSON", value=json.dumps(hyperparameters)
                )
            )

        resource_req = resources_requests or {"cpu": "250m", "memory": "512Mi"}
        resource_lim = resources_limits or {"cpu": "500m", "memory": "1Gi"}

        if use_gpu:
            resource_req["nvidia.com/gpu"] = "1"
            resource_lim["nvidia.com/gpu"] = "1"

        container = client.V1Container(
            name="mlflow-trainer-container",
            image=image,
            image_pull_policy="IfNotPresent",
            command=["python"],
            args=[train_script_path],
            env=container_env,
            resources=client.V1ResourceRequirements(
                requests=resource_req,
                limits=resource_lim,
            ),
        )

        pod_spec = client.V1PodSpec(restart_policy="OnFailure", containers=[container])
        if use_gpu:
            pod_spec.runtime_class_name = "nvidia"

        template = client.V1PodTemplateSpec(spec=pod_spec)

        spec = client.V1JobSpec(template=template, backoff_limit=3)

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=spec,
        )

        try:
            api_response = self.batch_v1.create_namespaced_job(
                body=job, namespace=self.namespace
            )
            logger.info(f"Kubernetes Job 생성 성공: {job_name}")
            return api_response
        except client.ApiException as e:
            logger.error(
                f"Kubernetes Job {job_name} 생성 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 Job 생성: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Job {job_name} 생성 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Job 생성: {e}")

    def get_job_status(self, job_name: str):
        """
        특정 Kubernetes Job의 상태를 조회합니다.
        """
        try:
            api_response = self.batch_v1.read_namespaced_job_status(
                name=job_name, namespace=self.namespace
            )
            return api_response.status
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 Job {job_name}을 찾을 수 없습니다."
                )
                return None
            logger.error(f"Job {job_name} 상태 조회 오류: {e.body}", exc_info=True)
            raise Exception(f"K8s API 오류 Job 상태 조회: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Job {job_name} 상태 조회 중 예상치 못한 오류: {e}", exc_info=True
            )
            raise Exception(f"예상치 못한 오류 Job 상태 조회: {e}")

    def delete_job(self, job_name: str):
        """
        특정 Kubernetes Job과 연관된 Pod를 삭제합니다.
        """
        logger.info(f"Kubernetes Job 삭제 시도: {job_name}")
        try:
            api_response = self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=0
                ),
            )
            logger.info(f"Kubernetes Job 삭제 시작 성공: {job_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 삭제할 Job {job_name}을 찾을 수 없습니다."
                )
                return {"message": f"삭제할 Job {job_name}을 찾을 수 없습니다."}
            logger.error(
                f"Kubernetes Job {job_name} 삭제 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 Job 삭제: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Job {job_name} 삭제 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Job 삭제: {e}")

    def get_pods_for_job(self, job_name: str) -> tp.List[client.V1Pod]:
        """
        특정 Job에 의해 생성된 Pod들을 조회합니다.
        """
        try:
            label_selector = f"job-name={job_name}"
            api_response = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )
            return api_response.items
        except client.ApiException as e:
            logger.error(f"Job {job_name}의 Pod 조회 오류: {e.body}", exc_info=True)
            raise Exception(f"K8s API 오류 Pod 조회: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Job {job_name}의 Pod 조회 중 예상치 못한 오류: {e}", exc_info=True
            )
            raise Exception(f"예상치 못한 오류 Pod 조회: {e}")

    def get_pod_logs(self, pod_name: str, tail_lines: tp.Optional[int] = None) -> str:
        """
        특정 Pod의 로그를 조회합니다.
        """
        try:
            api_response = self.core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace, tail_lines=tail_lines
            )
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 로그를 가져올 Pod {pod_name}을 찾을 수 없습니다."
                )
                return f"Pod {pod_name}을 찾을 수 없거나 로그를 사용할 수 없습니다."
            logger.error(f"Pod {pod_name} 로그 가져오기 오류: {e.body}", exc_info=True)
            raise Exception(f"K8s API 오류 Pod 로그 가져오기: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Pod {pod_name} 로그 가져오기 중 예상치 못한 오류: {e}", exc_info=True
            )
            raise Exception(f"예상치 못한 오류 Pod 로그 가져오기: {e}")

    def create_inference_deployment(
        self,
        deployment_name: str,
        image: str,
        mlflow_run_id: str,
        model_file_path: tp.Optional[str] = None,
        replicas: int = 1,
        resources_requests: tp.Optional[tp.Dict[str, str]] = None,
        resources_limits: tp.Optional[tp.Dict[str, str]] = None,
        use_gpu: bool = False,
        ingress_host: tp.Optional[str] = None,
        ingress_path: tp.Optional[str] = None,
    ):
        logger.info(f"Kubernetes 추론 Deployment 생성 시도: {deployment_name}")

        container_env = self._get_base_env_vars()
        container_env.append(
            client.V1EnvVar(name="MLFLOW_INFERENCE_RUN_ID", value=mlflow_run_id)
        )
        if model_file_path:
            container_env.append(
                client.V1EnvVar(name="MODEL_FILE_PATH", value=model_file_path)
            )

        resource_req = resources_requests or {"cpu": "500m", "memory": "1Gi"}
        resource_lim = resources_limits or {"cpu": "1", "memory": "2Gi"}
        if use_gpu:
            resource_req["nvidia.com/gpu"] = "1"
            resource_lim["nvidia.com/gpu"] = "1"

        container = client.V1Container(
            name="inference-server-container",
            image=image,
            image_pull_policy="Always",  # 항상 최신 이미지를 가져오도록 Always로 변경
            ports=[
                client.V1ContainerPort(container_port=8000)
            ],  # Inference 서버의 포트
            env=container_env,
            resources=client.V1ResourceRequirements(
                requests=resource_req,
                limits=resource_lim,
            ),
            # ⭐ readinessProbe 및 livenessProbe 추가
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/health", port=8000),
                initial_delay_seconds=10,  # 컨테이너 시작 후 10초 대기
                period_seconds=5,  # 5초마다 체크
                timeout_seconds=3,  # 3초 내에 응답 없으면 실패
                failure_threshold=3,  # 3번 실패하면 Unready
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/health", port=8000),
                initial_delay_seconds=30,  # 컨테이너 시작 후 30초 대기
                period_seconds=10,  # 10초마다 체크
                timeout_seconds=5,  # 5초 내에 응답 없으면 실패
                failure_threshold=3,  # 3번 실패하면 컨테이너 재시작
            ),
        )
        pod_spec = client.V1PodSpec(containers=[container])
        if use_gpu:
            pod_spec.runtime_class_name = "nvidia"

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": deployment_name}), spec=pod_spec
        )

        spec = client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(match_labels={"app": deployment_name}),
            template=template,
        )

        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=spec,
        )

        try:
            api_response = self.apps_v1.create_namespaced_deployment(
                body=deployment, namespace=self.namespace
            )
            logger.info(f"Kubernetes Deployment 생성 성공: {deployment_name}")
            return api_response
        except client.ApiException as e:
            logger.error(
                f"Kubernetes Deployment {deployment_name} 생성 오류: {e.body}",
                exc_info=True,
            )
            raise Exception(f"K8s API 오류 Deployment 생성: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Deployment {deployment_name} 생성 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Deployment 생성: {e}")

    def create_inference_service(
        self, service_name: str, deployment_name: str, port: int = 8000
    ):
        logger.info(
            f"Deployment {deployment_name}을 위한 Kubernetes 서비스 {service_name} 생성 시도"
        )
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[
                    client.V1ServicePort(protocol="TCP", port=port, target_port=port)
                ],
                type="ClusterIP",  # 내부 통신용
            ),
        )
        try:
            api_response = self.core_v1.create_namespaced_service(
                body=service, namespace=self.namespace
            )
            logger.info(f"Kubernetes 서비스 생성 성공: {service_name}")
            return api_response
        except client.ApiException as e:
            logger.error(
                f"Kubernetes 서비스 {service_name} 생성 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 서비스 생성: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes 서비스 {service_name} 생성 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 서비스 생성: {e}")

    def create_inference_ingress(
        self,
        ingress_name: str,
        service_name: str,
        host: str,
        path: str,
        service_port: int = 8000,
    ):
        logger.info(
            f"서비스 {service_name}을 위한 Kubernetes Ingress {ingress_name} 생성 시도"
        )
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
                                    path_type="Prefix",  # 또는 Exact
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=service_name,
                                            port=client.V1ServiceBackendPort(
                                                number=service_port
                                            ),
                                        )
                                    ),
                                )
                            ]
                        ),
                    )
                ]
            ),
        )
        try:
            api_response = self.networking_v1.create_namespaced_ingress(
                body=ingress, namespace=self.namespace
            )
            logger.info(f"Kubernetes Ingress 생성 성공: {ingress_name}")
            return api_response
        except client.ApiException as e:
            logger.error(
                f"Kubernetes Ingress {ingress_name} 생성 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 Ingress 생성: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Ingress {ingress_name} 생성 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Ingress 생성: {e}")

    def delete_deployment(self, deployment_name: str):
        logger.info(f"Kubernetes Deployment 삭제 시도: {deployment_name}")
        try:
            api_response = self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=0
                ),
            )
            logger.info(f"Kubernetes Deployment 삭제 시작 성공: {deployment_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 삭제할 Deployment {deployment_name}을 찾을 수 없습니다."
                )
                return {
                    "message": f"삭제할 Deployment {deployment_name}을 찾을 수 없습니다."
                }
            logger.error(
                f"Kubernetes Deployment {deployment_name} 삭제 오류: {e.body}",
                exc_info=True,
            )
            raise Exception(f"K8s API 오류 Deployment 삭제: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Deployment {deployment_name} 삭제 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Deployment 삭제: {e}")

    def delete_service(self, service_name: str):
        logger.info(f"Kubernetes 서비스 삭제 시도: {service_name}")
        try:
            api_response = self.core_v1.delete_namespaced_service(
                name=service_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=0
                ),
            )
            logger.info(f"Kubernetes 서비스 삭제 시작 성공: {service_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 삭제할 서비스 {service_name}을 찾을 수 없습니다."
                )
                return {"message": f"삭제할 서비스 {service_name}을 찾을 수 없습니다."}
            logger.error(
                f"Kubernetes 서비스 {service_name} 삭제 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 서비스 삭제: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes 서비스 {service_name} 삭제 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 서비스 삭제: {e}")

    def delete_ingress(self, ingress_name: str):
        logger.info(f"Kubernetes Ingress 삭제 시도: {ingress_name}")
        try:
            api_response = self.networking_v1.delete_namespaced_ingress(
                name=ingress_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=0
                ),
            )
            logger.info(f"Kubernetes Ingress 삭제 시작 성공: {ingress_name}")
            return api_response
        except client.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"네임스페이스 {self.namespace}에서 삭제할 Ingress {ingress_name}을 찾을 수 없습니다."
                )
                return {"message": f"삭제할 Ingress {ingress_name}을 찾을 수 없습니다."}
            logger.error(
                f"Kubernetes Ingress {ingress_name} 삭제 오류: {e.body}", exc_info=True
            )
            raise Exception(f"K8s API 오류 Ingress 삭제: {e.reason} - {e.body}")
        except Exception as e:
            logger.error(
                f"Kubernetes Ingress {ingress_name} 삭제 중 예상치 못한 오류: {e}",
                exc_info=True,
            )
            raise Exception(f"예상치 못한 오류 Ingress 삭제: {e}")


# FastAPI 앱에서 사용할 K8sClient 인스턴스를 미리 생성
k8s_client = K8sClient()
