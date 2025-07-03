import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch # PyTorch 모델 로깅을 위해 필요
import mlflow.sklearn # sklearn 모델 로깅을 위해 필요 (스케일러는 joblib으로 저장 후 artifact로 로깅할 것임)
import tempfile # 임시 디렉토리 생성을 위해 추가
import shutil # 임시 디렉토리 삭제를 위해 추가
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# 로거 설정
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. MLflow Tracking Server 설정 ---
# MLflow Tracking Server의 주소를 환경 변수에서 가져옵니다.
# Kubernetes 내부에서 접근한다면 서비스 이름을 사용합니다.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # 기본값은 로컬 테스트용
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")

# MinIO (S3) 엔드포인트 URL 설정 (아티팩트 저장용)
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
logger.info(f"MLflow S3 Endpoint URL set to: {MLFLOW_S3_ENDPOINT_URL}")

# MinIO (S3) 인증 정보 설정
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- 2. GPU 사용 가능 여부 확인 및 장치 설정 ---
# 이 스크립트에서는 CPU로 고정하여 실행
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# --- 3. 데이터 생성 및 전처리 (선형 회귀 예시) ---
# 간단한 선형 회귀 데이터 생성: y = 2*x + 1 + noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10 # 0에서 10 사이의 X 값
y = 2 * X + 1 + np.random.randn(100, 1) * 2 # 노이즈 추가

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일러 초기화 및 데이터 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# PyTorch Tensor로 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(DEVICE)

# DataLoader 설정
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 4. PyTorch 모델 정의 (간단한 선형 모델) ---
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 입력 1, 출력 1

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = SimpleLinearRegression().to(DEVICE)

# 손실 함수 및 최적화 도구 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- 5. MLflow Run 시작 및 모델 학습 ---
CUSTOM_MODEL_NAME = "SimpleLinearRegressionModel"
# run_name = os.getenv("MLFLOW_RUN_NAME", "pytorch_linear_regression_run")
run_id_from_env = os.getenv("MLFLOW_RUN_ID")

# MLflow Run ID가 환경 변수에 제공되면 해당 Run에 연결, 아니면 새 Run 생성
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
    # Training
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

    # Validation
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

    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

# --- 6. 모델 및 스케일러 저장 및 MLflow 로깅 ---
# 스케일러 저장 (기존 방식 유지 - joblib을 통한 명시적 저장 후 artifact로 로깅)
# 이 방식은 MinIO에 scaler/ 디렉토리가 생성되는 것을 확인했으므로 유지합니다.
temp_dir_for_scaler = tempfile.mkdtemp()
scaler_path = os.path.join(temp_dir_for_scaler, "scaler_X.pkl")
joblib.dump(scaler_X, scaler_path)
mlflow.log_artifact(scaler_path, artifact_path="scaler")
logger.info(f"Scaler X saved as artifact: {scaler_path} -> scaler/scaler_X.pkl")

scaler_y_path = os.path.join(temp_dir_for_scaler, "scaler_y.pkl")
joblib.dump(scaler_y, scaler_y_path)
mlflow.log_artifact(scaler_y_path, artifact_path="scaler") # scaler 디렉토리 밑에 저장
logger.info(f"Scaler Y saved as artifact: {scaler_y_path} -> scaler/scaler_y.pkl")
shutil.rmtree(temp_dir_for_scaler) # 임시 디렉토리 정리

# ⭐ PyTorch 모델 로깅 (torch.save로 명시적 저장 후 mlflow.log_artifact) ⭐
# 기존 mlflow.pytorch.log_model() 대신 이 방법을 시도합니다.
# 이렇게 하면 'model/' 디렉토리 대신 'explicit_model/' 디렉토리에 .pth 파일이 직접 저장되는지 확인할 수 있습니다.
temp_dir_for_model = tempfile.mkdtemp()
model_file_path = os.path.join(temp_dir_for_model, "pytorch_model_state_dict.pth")
torch.save(model.state_dict(), model_file_path) # 모델의 state_dict만 저장
mlflow.log_artifact(model_file_path, artifact_path="explicit_model")
logger.info(f"PyTorch model state_dict explicitly saved as artifact: {model_file_path} -> explicit_model/pytorch_model_state_dict.pth")
shutil.rmtree(temp_dir_for_model) # 임시 디렉토리 정리


# # ⭐ 기존 mlflow.pytorch.log_model 호출 (주석 처리 또는 제거) ⭐
# # 이 부분이 문제의 원인일 수 있으므로, 위 명시적 저장 방식 테스트를 위해 주석 처리합니다.
# # 로깅을 위한 시그니처 및 입력 예시 생성
# signature = mlflow.models.infer_signature(X_train, model(X_train_tensor).cpu().numpy())
# input_example = X_train_tensor[0].cpu().numpy()

# try:
#     mlflow.pytorch.log_model(
#         pytorch_model=model,
#         artifact_path="model",
#         signature=signature,
#         input_example=input_example,
#         registered_model_name=CUSTOM_MODEL_NAME # 모델 레지스트리에 등록
#     )
#     logger.info(f"PyTorch model logged and registered as '{CUSTOM_MODEL_NAME}' under artifact_path 'model/'.")
# except Exception as e:
#     logger.error(f"Failed to log PyTorch model using mlflow.pytorch.log_model: {e}", exc_info=True)


# 손실 곡선 시각화 및 아티팩트 저장
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plot_path = "loss_curve.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)
logger.info(f"Loss curve saved as artifact: {plot_path}")
plt.close()

# --- 7. 예측 및 결과 시각화 (선택 사항) ---
# 예측 (스케일링 역변환)
with torch.no_grad():
    predictions_scaled = model(X_test_tensor).cpu().numpy()
predictions = scaler_y.inverse_transform(predictions_scaled)

# 실제 값 (스케일링 역변환)
y_test_original = scaler_y.inverse_transform(y_test_scaled) # 수정된 부분

# 예측 결과 시각화 및 아티팩트 저장
plt.figure(figsize=(10, 6))
# ⭐ X_test_scaled가 1D가 아닐 경우 첫 번째 특성만 사용 ([:, 0] 추가)
plt.scatter(scaler_X.inverse_transform(X_test_scaled)[:, 0], y_test_original, label='Actual Values', alpha=0.6)
plt.scatter(scaler_X.inverse_transform(X_test_scaled)[:, 0], predictions, label='Predictions', alpha=0.6)
plt.title('Actual vs. Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
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