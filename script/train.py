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
    """S3 경로의 파일 또는 디렉토리를 로컬 경로로 다운로드하고, 실제 파일 경로를 반환합니다."""
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")

    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=s3_endpoint_url)
    bucket_name, s3_key = s3_path.replace("s3://", "").split("/", 1)
    
    target_file_path = local_path / Path(s3_key).name
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading from s3://{bucket_name}/{s3_key} to {target_file_path}...")
        s3.download_file(bucket_name, s3_key, str(target_file_path))
        logger.info("S3 download completed successfully.")
        return target_file_path
    except ClientError as e:
        logger.error(f"Failed to download from S3: {e}")
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
    
    with mlflow.start_run(run_name=f"run-{args.task_id}") as run:
        mlflow.log_params(vars(args))
        mlflow.set_tag("task_id", args.task_id)

        device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            local_data_path = download_from_s3(args.dataset_path, tmpdir_path)
            local_model_path = tmpdir_path / "model.pt"

            if args.initial_model_file_path and args.initial_model_file_path.lower() != "none":
                download_from_s3(args.initial_model_file_path, local_model_path)

            handler = load_handler_module(args.handler_name)
            
            if args.handler_name == "image_classification_handler":
                train_loader, val_loader, data_info, transforms = handler.create_data_loaders(
                    local_data_path, args.num_batch, args=args
                )
            else:
                train_loader, val_loader, data_info = handler.create_data_loaders(
                    local_data_path, args.num_batch, args=args
                )
                transforms = None

            model = handler.create_model(args=args, **data_info)

            if local_model_path.exists():
                model.load_state_dict(torch.load(local_model_path, map_location=device))
                logger.info(f"Loaded initial model weights from {local_model_path}")

            model.to(device)
            # optimizer = get_optimizer(args.optimizer_name, model.parameters(), args.learning_rate)
            # criterion = get_loss_function(args.loss_function)
            optimizer = handler.create_optimizer(model, args.learning_rate)
            criterion = handler.create_loss_function()

            # 범용 학습/검증 루프
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
                        "model_name": args.custom_model_name,
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