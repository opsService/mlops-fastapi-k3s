import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import mlflow
import tempfile
from pathlib import Path
import cloudpickle
import yaml

# This module-level variable will cache the transforms created by the data loader
_cached_transforms = None

def create_data_loaders(data_path, batch_size, **kwargs):
    """ImageFolder를 사용하여 데이터 로더를 생성하고, 사용된 transform을 캐시합니다."""
    global _cached_transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _cached_transforms = transform
    
    full_dataset = torchvision.datasets.ImageFolder(root=str(data_path), transform=transform)
    
    # Use indices for splitting to avoid loading full dataset into memory for stratify
    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42, stratify=targets)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    data_info = {
        "num_classes": len(full_dataset.classes),
        "class_names": full_dataset.classes
    }
    
    print(f"ImageFolder data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, data_info

def create_model(args, num_classes, **kwargs):
    """지정된 이름의 모델을 torchvision에서 로드하고, 필요한 경우 최종 레이어를 수정합니다."""
    # TODO: 향후 다양한 모델 아키텍처를 지원하려면 API 스키마에 modelArchitecture 필드를 추가하고, 
    # 아래 model_name을 해당 값으로 동적으로 설정해야 합니다.
    # model_name = args.modelArchitecture 
    
    # 현재는 resnet18로 아키텍처를 고정합니다.
    model_name = 'resnet18'
    pretrained = not args.initial_model_file_path
    
    if not hasattr(torchvision.models, model_name):
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    model = getattr(torchvision.models, model_name)(pretrained=pretrained)

    if hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            final_layer_index = -1
            while not isinstance(model.classifier[final_layer_index], nn.Linear):
                final_layer_index -= 1
            num_ftrs = model.classifier[final_layer_index].in_features
            model.classifier[final_layer_index] = nn.Linear(num_ftrs, num_classes)
        else:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise TypeError(f"Could not find a final layer to modify for model {model_name}")

    print(f"Model '{model_name}' loaded and final layer adapted for {num_classes} classes.")
    return model

def create_optimizer(model: nn.Module, learning_rate: float):
    """Creates an Adam optimizer for the given model."""
    return optim.Adam(model.parameters(), lr=learning_rate)


def create_loss_function():
    """Creates a Cross Entropy Loss function for classification."""
    return nn.CrossEntropyLoss()

def log_model(model, args, data_info, **kwargs):
    """Logs the model using the custom ImageClassificationWrapper."""
    from models.image_classification_wrapper import ImageClassificationWrapper
    print("Logging as custom PyFunc model with ImageClassificationWrapper.")
    
    with tempfile.TemporaryDirectory() as artifact_tmpdir:
        artifact_path = Path(artifact_tmpdir)
        weights_path = artifact_path / "model_weights.pth"
        torch.save(model.state_dict(), weights_path)

        transforms_path = artifact_path / "transforms.pkl"
        with open(transforms_path, "wb") as f:
            cloudpickle.dump(_cached_transforms, f)

        config_path = artifact_path / "wrapper_config.yaml"
        wrapper_config = {
            "handler_name": args.handler_name,
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