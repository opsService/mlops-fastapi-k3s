import tempfile
from pathlib import Path

import datasets
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Import wrapper class here to avoid circular dependency issues at top level
from models.text_classification_wrapper import TextClassificationWrapper
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# This module-level variable will cache the tokenizer name used by the data loader
_cached_tokenizer_name = None

def create_data_loaders(data_path, batch_size, **kwargs):
    """표준 CSV 파일을 읽어 NLP 분류를 위한 데이터 로더를 생성하고, 토크나이저 이름을 캐시합니다."""
    global _cached_tokenizer_name
    # 고정된 토크나이저 사용
    tokenizer_name = "distilbert-base-uncased"
    _cached_tokenizer_name = tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 탭으로 분리된 데이터 파일을 로드합니다. (헤더는 자동 추론)
    df = pd.read_csv(data_path, sep='\t')

    # 필수 컬럼 확인
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data file must contain 'text' and 'label' columns in the header.")

    # 토크나이저 오류 방지를 위한 데이터 정제
    df['text'] = df['text'].fillna('').astype(str)

    # 학습에 필요한 컬럼만 선택
    df = df[['text', 'label']]
    
    raw_dataset = datasets.Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
        
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # 데이터셋 분할
    train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    print(f"HuggingFace data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, {"num_classes": df['label'].nunique()}

def create_model(num_classes, **kwargs):
    """Creates a DistilBERT model for sequence classification."""
    model_name = "distilbert-base-uncased"
    print(f"Creating {model_name} for {num_classes}-class classification.")
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)


def create_optimizer(model: nn.Module, learning_rate: float):
    """Creates an AdamW optimizer for the given model."""
    return optim.AdamW(model.parameters(), lr=learning_rate)

def create_loss_function():
    """Creates a Cross Entropy Loss function for classification."""
    return nn.CrossEntropyLoss()

def log_model(model, args, **kwargs):
    """Logs the model using the custom TextClassificationWrapper."""
    print("Logging as custom PyFunc model with TextClassificationWrapper.")
    with tempfile.TemporaryDirectory() as artifact_tmpdir:
        artifact_path = Path(artifact_tmpdir)
        model_path = artifact_path / "model"
        config_path = artifact_path / "wrapper_config.yaml"

        # Save the trained model
        model.save_pretrained(model_path)
        
        # Save the wrapper config
        wrapper_config = {"tokenizer_name": _cached_tokenizer_name}
        with open(config_path, "w") as f:
            yaml.dump(wrapper_config, f)

        artifacts = {
            "model_path": str(model_path),
            "wrapper_config": str(config_path)
            }

        mlflow.pyfunc.log_model(
            artifact_path="ml_model",
            python_model=TextClassificationWrapper(),
            artifacts=artifacts,
            registered_model_name=args.custom_model_name,
        )
