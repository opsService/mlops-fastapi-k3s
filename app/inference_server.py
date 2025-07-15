# mlops-fastapi-app/app/inference_server.py

import logging
import os

import joblib  # 스케일러 로딩용
import mlflow

# import mlflow # mlflow.pyfunc.load_model을 사용하지 않을 것이므로 주석 처리 가능
# import mlflow.pyfunc # 주석 처리 가능
import numpy as np
import torch  # PyTorch 모델 로딩용
import torch.nn as nn  # PyTorch 모델 클래스 정의를 위해 필요
from app.core.logging_config import setup_logging
from flask import Flask, jsonify, request

# 로거 설정
setup_logging()
logger = logging.getLogger("inference_server")

app = Flask(__name__)

# 전역 변수로 모델과 스케일러 저장
model = None
scaler_X = None  # scaler_X로 명확히
scaler_y = None  # scaler_y 추가
loaded_run_id = None


# SimpleLinearRegression 모델 클래스를 다시 정의해야 합니다!
# train_cpu.py에 있는 동일한 모델 정의가 inference_server.py에도 필요합니다.
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 입력 1, 출력 1

    def forward(self, x):
        return self.linear(x)


@app.route("/health", methods=["GET"])
def health():
    """헬스 체크 엔드포인트: 서버가 정상적으로 작동하는지 확인합니다."""
    logger.debug("Health check requested.")
    return jsonify({"status": "healthy"}), 200


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    MLflow에서 모델과 스케일러를 로드하는 엔드포인트입니다.
    """
    global model, scaler_X, scaler_y, loaded_run_id

    print("DEBUG: Received /load_model request - STARTING FUNCTION.")
    logger.debug("Received /load_model request.")

    data = request.get_json()
    run_id = data.get("run_id")

    if not run_id:
        logger.error("MLflow Run ID가 제공되지 않았습니다.")
        print("ERROR: MLflow Run ID is missing.")
        return jsonify({"error": "MLflow Run ID is missing."}), 400

    # 이미 로드된 모델과 동일한 Run ID라면 다시 로드하지 않음
    if loaded_run_id == run_id:
        logger.info(f"Model for Run ID {run_id} is already loaded.")
        return (
            jsonify({"message": f"Model for Run ID {run_id} is already loaded."}),
            200,
        )

    try:
        # MLflow Tracking URI 및 S3 Endpoint URL 설정
        MLFLOW_TRACKING_URI = os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000"
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000"
        )
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

        # ⭐ 수정된 부분: MLflow 아티팩트에서 직접 파일 로드 ⭐
        # 모델 로드 (state_dict만 로드)
        model_uri = f"runs:/{run_id}/explicit_model/pytorch_model_state_dict.pth"
        model_local_path = mlflow.artifacts.download_artifacts(model_uri)
        logger.info(f"Downloaded model artifact from {model_uri} to {model_local_path}")

        model = SimpleLinearRegression()  # 모델 클래스 인스턴스 생성
        model.load_state_dict(
            torch.load(model_local_path, map_location=torch.device("cpu"))
        )  # state_dict 로드
        model.eval()  # 평가 모드 설정
        logger.info(
            f"PyTorch model (state_dict) loaded successfully from {model_local_path}"
        )

        # 스케일러 로드
        scaler_X_uri = f"runs:/{run_id}/scaler/scaler_X.pkl"
        scaler_X_local_path = mlflow.artifacts.download_artifacts(scaler_X_uri)
        scaler_X = joblib.load(scaler_X_local_path)
        logger.info(f"Scaler X loaded successfully from {scaler_X_local_path}")

        scaler_y_uri = f"runs:/{run_id}/scaler/scaler_y.pkl"
        scaler_y_local_path = mlflow.artifacts.download_artifacts(scaler_y_uri)
        scaler_y = joblib.load(scaler_y_local_path)
        logger.info(f"Scaler Y loaded successfully from {scaler_y_local_path}")

        loaded_run_id = run_id
        logger.info(f"Model and scaler for Run ID {run_id} loaded successfully.")
        return (
            jsonify(
                {
                    "message": f"Model and scaler for Run ID {run_id} loaded successfully."
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Failed to load model for Run ID {run_id}: {e}", exc_info=True)
        print(f"ERROR: Failed to load model - {e}")
        return jsonify({"error": f"Failed to load model: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    로드된 모델을 사용하여 예측을 수행하는 엔드포인트입니다.
    """
    global model, scaler_X, scaler_y  # scaler_X, scaler_y로 변경
    if model is None or scaler_X is None or scaler_y is None:  # scaler_y 추가
        logger.error("예측 요청이 들어왔으나 모델 또는 스케일러가 로드되지 않았습니다.")
        return (
            jsonify(
                {"error": "Prediction requested but model or scaler is not loaded."}
            ),
            500,
        )

    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # 입력 데이터 형태 조정
        logger.debug(f"원시 입력 특성: {features}")

        # 스케일링 적용
        scaled_features = scaler_X.transform(features)
        logger.debug(f"스케일링된 특성: {scaled_features}")

        # PyTorch Tensor로 변환 및 예측 수행
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            prediction_scaled = model(input_tensor).cpu().numpy()

        # 예측 역스케일링
        prediction = scaler_y.inverse_transform(prediction_scaled)
        logger.info(f"예측 결과: {prediction.tolist()}")

        return jsonify({"prediction": prediction.tolist()}), 200
    except KeyError:
        logger.error(
            "입력 데이터에 'features' 키가 누락되었습니다. 예상 형식: {'features': [val1, val2, ...]}"
        )
        return (
            jsonify(
                {
                    "error": "Invalid input: 'features' key is missing or data format is incorrect."
                }
            ),
            400,
        )
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Gunicorn 대신 Flask의 내장 서버로 간단히 테스트하려면 아래 주석 해제
    # app.run(host="0.0.0.0", port=8000, debug=True)
    pass
