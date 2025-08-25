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

# Import helper functions from the new utils module
from models.utils import download_from_s3, load_handler_module

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

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
            train_loader, val_loader, data_info = handler.create_data_loaders(
                local_data_path, args.num_batch, args=args
            )
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

            # Delegate model logging to the specific handler
            logger.info(f"Logging model via handler: {args.handler_name}")
            handler.log_model(model=model, args=args, data_info=data_info)
            
            logger.info(f"Model '{args.custom_model_name}' logged to MLflow.")
            
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