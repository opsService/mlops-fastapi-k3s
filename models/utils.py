import importlib.util
import logging
import os
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def _load_handler_from_context(context):
    """Helper to load a handler module from artifacts."""
    config_path = context.artifacts["wrapper_config"]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    handler_name = config["handler_name"]
    # Correct the path to look inside the app directory first
    handler_script_path = Path("/app/models/") / f"{handler_name}.py"
    if not handler_script_path.exists():
        # Fallback for local execution
        handler_script_path = Path("models/") / f"{handler_name}.py"

    spec = importlib.util.spec_from_file_location(handler_name, handler_script_path)
    handler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_module)
    return handler_module, config

def download_from_s3(s3_path: str, local_path: Path) -> Path:
    """S3 경로의 파일 또는 디렉토리를 로컬 경로로 재귀적으로 다운로드합니다."""
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")

    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=s3_endpoint_url)
    bucket_name, s3_key = s3_path.replace("s3://", "").split("/", 1)

    # If s3_key ends with '/', treat it as a directory
    if s3_key.endswith('/'):
        logger.info(f"Path is a directory. Downloading contents from s3://{bucket_name}/{s3_key} to {local_path}...")
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)

        download_root_path = local_path / Path(s3_key).name
        download_root_path.mkdir(parents=True, exist_ok=True)

        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    relative_path = obj['Key'].replace(s3_key, '', 1)
                    if not relative_path:
                        continue
                    
                    local_file_path = download_root_path / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    logger.debug(f"Downloading {obj['Key']} to {local_file_path}")
                    s3.download_file(bucket_name, obj['Key'], str(local_file_path))
        logger.info("S3 directory download completed successfully.")
        return download_root_path
    
    # Otherwise, treat it as a single file
    else:
        target_file_path = local_path / Path(s3_key).name
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Path is a file. Downloading from s3://{bucket_name}/{s3_key} to {target_file_path}...")
        try:
            s3.download_file(bucket_name, s3_key, str(target_file_path))
            logger.info("S3 file download completed successfully.")
            return target_file_path
        except ClientError as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise

def load_handler_module(handler_name: str):
    """핸들러 이름을 기반으로 핸들러 모듈을 동적으로 로드합니다."""
    handler_script_path = Path("/app/models/") / f"{handler_name}.py"
    if not handler_script_path.exists():
        handler_script_path = Path(__file__).parent.parent / "models" / f"{handler_name}.py"
        if not handler_script_path.exists():
            raise FileNotFoundError(f"Handler script not found for handler: {handler_name}")

    logger.info(f"Loading handler module from: {handler_script_path}")
    spec = importlib.util.spec_from_file_location(handler_name, handler_script_path)
    handler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_module)
    
    assert hasattr(handler_module, 'create_data_loaders'), "Handler must have a 'create_data_loaders' function."
    assert hasattr(handler_module, 'create_model'), "Handler must have a 'create_model' function."
    
    return handler_module