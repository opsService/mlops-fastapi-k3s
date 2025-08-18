import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for regression tasks."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_model(input_dim: int, output_dim: int = 1, **kwargs) -> nn.Module:
    """Factory function to create an instance of the MLP model."""
    print(f"Creating MLP model with input_dim={input_dim}, output_dim={output_dim}")
    return MLP(input_dim=input_dim, output_dim=output_dim)


def create_data_loaders(data_path, batch_size, **kwargs):
    """CSV 파일을 읽어 회귀 또는 분류를 위한 데이터 로더를 생성합니다."""
    df = pd.read_csv(data_path)
    
    # 마지막 열을 타겟으로, 나머지를 피처로 가정합니다.
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"CSV data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    # 모델 생성에 필요한 입력 및 출력 차원 정보 반환
    return train_loader, val_loader, {"input_dim": X.shape[1], "output_dim": 1}


def create_optimizer(model: nn.Module, learning_rate: float):
    """Creates an Adam optimizer for the given model."""
    return optim.Adam(model.parameters(), lr=learning_rate)


def create_loss_function():
    """Creates a Mean Squared Error loss function for regression."""
    return nn.MSELoss()
