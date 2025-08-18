import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def create_data_loaders(data_path, batch_size, **kwargs):
    """ImageFolder를 사용하여 데이터 로더를 생성합니다."""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # ImageFolder는 data_path 자체가 클래스 폴더들을 담고 있는 루트라고 가정합니다.
    full_dataset = torchvision.datasets.ImageFolder(root=str(data_path), transform=transform)
    
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42, stratify=full_dataset.targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    data_info = {
        "num_classes": len(full_dataset.classes),
        "class_names": full_dataset.classes
    }
    
    print(f"ImageFolder data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, data_info, transform

def create_model(args, num_classes, **kwargs):
    """지정된 이름의 모델을 torchvision에서 로드하고, 필요한 경우 최종 레이어를 수정합니다."""
    model_name = getattr(args, 'modelName', 'resnet18') # API 스키마 변경으로 modelName이 없어졌으므로 args에서 직접 가져오거나 기본값 사용
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