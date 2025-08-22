import argparse
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

import boto3
import cloudpickle
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from botocore.exceptions import ClientError

# Import wrapper class here to avoid circular dependency issues at top level
from models.pyfunc_wrappers import ImageClassificationWrapper

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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

# def get_optimizer(optimizer_name: str, parameters, lr: float):
#     return getattr(optim, optimizer_name)(parameters, lr=lr)

# def get_loss_function(loss_name: str):
#     return getattr(nn, loss_name)()

def train_model(args):
    """범용 모델 학습 및 MLflow 로깅 파이프라인 오케스트레이터"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name)
    
    # 오케스트레이터에 의해 시작된 기존 MLflow run에 연결
    with mlflow.start_run(run_id=args.mlflow_run_id) as run:
        # task_id는 이미 오케스트레이터에서 태그로 설정되었지만, 여기서 다시 설정하여 일관성 보장
        mlflow.set_tag("task_id", args.task_id)
        
        # 하이퍼파라미터 로깅 (오케스트레이터에서도 수행하지만, 여기서 실행된 실제 값들을 기록)
        # mlflow_run_id는 로깅할 필요 없으므로 제외
        params_to_log = vars(args).copy()
        params_to_log.pop('mlflow_run_id', None)
        mlflow.log_params(params_to_log)

        device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            logger.info("Downloading dataset...")
            local_data_path = download_from_s3(args.dataset_path, tmpdir_path)
            logger.info(f"Dataset downloaded to {local_data_path}")
            local_model_path = tmpdir_path / "model.pt"

            if args.initial_model_file_path and args.initial_model_file_path.lower() != "none":
                download_from_s3(args.initial_model_file_path, local_model_path)

            logger.info(f"Loading handler: {args.handler_name}")
            handler = load_handler_module(args.handler_name)
            
            logger.info("Creating data loaders...")
            if args.handler_name == "image_classification_handler":
                train_loader, val_loader, data_info, transforms = handler.create_data_loaders(
                    local_data_path, args.num_batch, args=args
                )
            else:
                train_loader, val_loader, data_info = handler.create_data_loaders(
                    local_data_path, args.num_batch, args=args
                )
                transforms = None
            logger.info("Data loaders created successfully.")

            logger.info("Creating model...")
            model = handler.create_model(args=args, **data_info)
            logger.info("Model created successfully.")

            if local_model_path.exists():
                model.load_state_dict(torch.load(local_model_path, map_location=device))
                logger.info(f"Loaded initial model weights from {local_model_path}")

            logger.info(f"Moving model to device: {device}...")
            model.to(device)
            logger.info("Model moved to device successfully.")

            optimizer = handler.create_optimizer(model, args.learning_rate)
            criterion = handler.create_loss_function()
            logger.info("Optimizer and loss function created.")

            # 범용 학습/검증 루프
            logger.info("Starting training loop...")
            for epoch in range(args.num_epoch):
                model.train()
                for batch in train_loader:
                    if isinstance(batch, list):
                        inputs, labels = batch
                        labels = labels.to(device)
                        if isinstance(inputs, dict):
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(device)
                    else:
                        labels = batch.pop("labels").to(device)
                        inputs = {k: v.to(device) for k, v in batch.items()}

                    if args.task_type == 'regression':
                        labels = labels.view(-1, 1).float()

                    optimizer.zero_grad()
                    outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                    logits = getattr(outputs, 'logits', outputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss, total_samples = 0, 0
                eval_metrics = {}
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, list):
                            inputs, labels = batch
                            labels = labels.to(device)
                            if isinstance(inputs, dict):
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                            else:
                                inputs = inputs.to(device)
                        else:
                            labels = batch.pop("labels").to(device)
                            inputs = {k: v.to(device) for k, v in batch.items()}
                        
                        if args.task_type == 'regression':
                            labels = labels.view(-1, 1).float()

                        outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                        logits = getattr(outputs, 'logits', outputs)
                        val_loss += criterion(logits, labels).item() * labels.size(0)
                        
                        if args.task_type == "classification":
                            preds = torch.argmax(logits, dim=-1)
                            correct = (preds == labels).sum().item()
                            eval_metrics["correct"] = eval_metrics.get("correct", 0) + correct
                        elif args.task_type == "regression":
                            mae = nn.functional.l1_loss(logits, labels, reduction='sum').item()
                            eval_metrics["mae"] = eval_metrics.get("mae", 0) + mae

                        total_samples += labels.size(0)

                avg_val_loss = val_loss / total_samples
                log_metrics = {"val_loss": avg_val_loss}
                log_str = f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}"

                if args.task_type == "classification":
                    accuracy = eval_metrics["correct"] / total_samples
                    log_metrics["val_accuracy"] = accuracy
                    log_str += f", Accuracy: {accuracy:.4f}"
                elif args.task_type == "regression":
                    val_mae = eval_metrics["mae"] / total_samples
                    log_metrics["val_mae"] = val_mae
                    log_str += f", Val MAE: {val_mae:.4f}"
                
                logger.info(log_str)
                mlflow.log_metrics(log_metrics, step=epoch)

            # Conditional model logging
            if args.handler_name == "image_classification_handler":
                logger.info("Logging as custom PyFunc model with ImageClassificationWrapper.")
                with tempfile.TemporaryDirectory() as artifact_tmpdir:
                    artifact_path = Path(artifact_tmpdir)
                    weights_path = artifact_path / "model_weights.pth"
                    torch.save(model.state_dict(), weights_path)

                    transforms_path = artifact_path / "transforms.pkl"
                    with open(transforms_path, "wb") as f:
                        cloudpickle.dump(transforms, f)

                    config_path = artifact_path / "wrapper_config.yaml"
                    wrapper_config = {
                        "handler_name": args.handler_name,
                        # TODO: 향후 API 스키마에 modelArchitecture 필드가 추가되면 아래 코드로 대체해야 합니다.
                        # "model_architecture": args.modelArchitecture,
                        
                        # 현재는 핸들러에 하드코딩된 아키텍처 이름을 사용합니다.
                        "model_architecture": "resnet18",
                        "data_info": data_info,
                        "class_names": data_info.get("class_names")
                    }
                    with open(config_path, "w") as f:
                        yaml.dump(wrapper_config, f)

                    artifacts ={
                        "model_weights": str(weights_path),
                        "transforms": str(transforms_path),
                        "wrapper_config": str(config_path)
                    }

                    mlflow.pyfunc.log_model(
                        artifact_path="ml_model",
                        python_model=ImageClassificationWrapper(),
                        artifacts=artifacts,
                        registered_model_name=args.custom_model_name
                    )
            else:
                logger.info("Logging as a standard PyTorch model.")
                mlflow.pytorch.log_model(model, artifact_path="ml_model", registered_model_name=args.custom_model_name)
            
            logger.info(f"Model '{args.custom_model_name}' logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic PyTorch Model Training Orchestrator")
    
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--mlflow-run-id", type=str, required=True, help="MLflow Run ID to resume.")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--custom-model-name", type=str, required=True)
    parser.add_argument("--handler-name", type=str, required=True, help="Name of the handler script in the models directory.")
    parser.add_argument("--task-type", type=str, default="classification", choices=["classification", "regression"])
    # parser.add_argument("--loss-function", type=str, default="CrossEntropyLoss")
    # parser.add_argument("--optimizer-name", type=str, default="Adam")
    parser.add_argument("--initial-model-file-path", type=str, default=None)
    parser.add_argument("--num-epoch", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-batch", type=int, default=32)
    parser.add_argument("--use-gpu", action="store_true")

    args = parser.parse_args()
    train_model(args)