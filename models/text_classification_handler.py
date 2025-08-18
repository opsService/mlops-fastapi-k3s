import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


def create_data_loaders(data_path, batch_size, **kwargs):
    """텍스트 파일을 읽어 NLP 분류를 위한 데이터 로더를 생성합니다."""
    # 고정된 토크나이저 사용
    tokenizer_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 사용자가 요청한 대로 .txt 파일을 DataFrame으로 로드 (탭 분리 가정)
    df = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'label'])
    raw_dataset = datasets.Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
        
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