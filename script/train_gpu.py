import argparse
import json

# 로거 설정
import logging
import os
import shutil  # 임시 디렉토리 삭제를 위해 추가
import tempfile  # 임시 디렉토리 생성을 위해 추가

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch  # PyTorch 모델 로깅을 위해 필요
import mlflow.sklearn  # sklearn 모델 로깅을 위해 필요 (스케일러는 joblib으로 저장 후 artifact로 로깅할 것임)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from app.core.logging_config import setup_logging
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

setup_logging()
logger = logging.getLogger(__name__)

# --- 1. MLflow Tracking Server 설정 ---
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5000"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")

MLFLOW_S3_ENDPOINT_URL = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"
)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
logger.info(f"MLflow S3 Endpoint URL set to: {MLFLOW_S3_ENDPOINT_URL}")

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- 2. GPU 사용 가능 여부 확인 및 장치 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- 3. 데이터 생성 및 전처리 (선형 회귀 예시) ---
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# --- 4. PyTorch 모델 정의 (간단한 선형 모델) ---
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# --- Pyfunc 모델 래퍼 ---
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        self.scaler_X = None
        self.scaler_y = None

    def load_context(self, context):
        self.scaler_X = joblib.load(context.artifacts["scaler_x_path"])
        self.scaler_y = joblib.load(context.artifacts["scaler_y_path"])
        logger.info("Scalers loaded successfully within pyfunc model.")

    def predict(self, context, model_input):
        scaled_features = self.scaler_X.transform(model_input.values)
        
        # 모델과 동일한 장치로 입력 텐서 이동
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy()
            
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction

# 모델 초기화
model = SimpleLinearRegression().to(DEVICE)

# 손실 함수 및 최적화 도구 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- 5. MLflow Run 시작 및 모델 학습 ---
CUSTOM_MODEL_NAME = "SimpleLinearRegressionModel"
run_id_from_env = os.getenv("MLFLOW_RUN_ID")

if run_id_from_env:
    mlflow.start_run(run_id=run_id_from_env)
    logger.info(f"Attaching to existing MLflow Run with ID: {run_id_from_env}")
else:
    active_run = mlflow.start_run()
    run_id_from_env = active_run.info.run_id
    logger.info(f"Started new MLflow Run with ID: {run_id_from_env}")


# 하이퍼파라미터 로깅
num_epochs = 100
mlflow.log_param("num_epochs", num_epochs)
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("optimizer", "SGD")
mlflow.log_param("criterion", "MSELoss")


train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    epoch_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)

    mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
    mlflow.log_metric("test_loss", epoch_test_loss, step=epoch)

    logger.info(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}"
    )

# --- 6. 모델 및 스케일러 저장 및 MLflow 로깅 (Pyfunc 사용) ---
with tempfile.TemporaryDirectory() as tmpdir:
    scaler_x_path = os.path.join(tmpdir, "scaler_X.pkl")
    scaler_y_path = os.path.join(tmpdir, "scaler_y.pkl")
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    artifacts_to_log = {
        "scaler_x_path": scaler_x_path,
        "scaler_y_path": scaler_y_path,
    }

    input_example = pd.DataFrame(X_train[:5], columns=['feature'])
    
    logger.info("Logging model using mlflow.pyfunc.log_model")
    
    mlflow.pyfunc.log_model(
        artifact_path="pyfunc_model",
        python_model=ModelWrapper(model=model),
        artifacts=artifacts_to_log,
        # code_path=[__file__],
        input_example=input_example,
        registered_model_name=CUSTOM_MODEL_NAME,
    )
    logger.info(f"Model '{CUSTOM_MODEL_NAME}' logged as pyfunc and registered.")


# 손실 곡선 시각화 및 아티팩트 저장
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Training and Test Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plot_path = "loss_curve.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)
logger.info(f"Loss curve saved as artifact: {plot_path}")
plt.close()

# --- 7. 예측 및 결과 시각화 (선택 사항) ---
with torch.no_grad():
    predictions_scaled = model(X_test_tensor).cpu().numpy()
predictions = scaler_y.inverse_transform(predictions_scaled)

y_test_original = scaler_y.inverse_transform(y_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(
    scaler_X.inverse_transform(X_test_scaled)[:, 0],
    y_test_original,
    label="Actual Values",
    alpha=0.6,
)
plt.scatter(
    scaler_X.inverse_transform(X_test_scaled)[:, 0],
    predictions,
    label="Predictions",
    alpha=0.6,
)
plt.title("Actual vs. Predicted Values")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
prediction_plot_path = "predictions.png"
plt.savefig(prediction_plot_path)
mlflow.log_artifact(prediction_plot_path)
logger.info(f"Prediction plot saved as artifact: {prediction_plot_path}")
plt.close()

mlflow.end_run()
logger.info("MLflow Run ended.")
logger.info(f"Final MLflow Run ID: {run_id_from_env}")
