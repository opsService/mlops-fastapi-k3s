# mlops-fastapi-app/app/inference_server.py

import logging
import os

import mlflow.pyfunc
import pandas as pd
from app.core.logging_config import setup_logging
from flask import Flask, jsonify, request

# 로거 설정
setup_logging()
logger = logging.getLogger("inference_server")

app = Flask(__name__)

# 전역 변수로 모델 저장
model = None
loaded_run_id = None


@app.route("/health", methods=["GET"])
def health():
    """헬스 체크 엔드포인트: 서버가 정상적으로 작동하는지 확인합니다."""
    logger.debug("Health check requested.")
    return jsonify({"status": "healthy"}), 200


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    MLflow Run ID를 사용하여 pyfunc 모델을 로드합니다.
    """
    global model, loaded_run_id

    logger.debug("Received /load_model request.")
    data = request.get_json()
    run_id = data.get("run_id")

    if not run_id:
        logger.error("MLflow Run ID가 제공되지 않았습니다.")
        return jsonify({"error": "MLflow Run ID is missing."}), 400

    if loaded_run_id == run_id:
        logger.info(f"Model for Run ID {run_id} is already loaded.")
        return jsonify({"message": f"Model for Run ID {run_id} is already loaded."}), 200

    try:
        # MLflow 환경 변수 설정 (컨테이너 환경에 따라 필요)
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service:9000")
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # 모델 URI 구성 (pyfunc 모델이 저장된 경로)
        model_uri = f"runs:/{run_id}/pyfunc_model"
        logger.info(f"Loading model from URI: {model_uri}")

        # mlflow.pyfunc.load_model을 사용하여 모델 로드
        model = mlflow.pyfunc.load_model(model_uri)
        
        loaded_run_id = run_id
        logger.info(f"Pyfunc model for Run ID {run_id} loaded successfully.")
        return jsonify({"message": f"Model for Run ID {run_id} loaded successfully."}), 200

    except Exception as e:
        logger.error(f"Failed to load model for Run ID {run_id}: {e}", exc_info=True)
        model = None # 로드 실패 시 모델을 None으로 설정
        return jsonify({"error": f"Failed to load model: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    로드된 pyfunc 모델을 사용하여 예측을 수행합니다.
    """
    global model
    if model is None:
        logger.error("예측 요청이 들어왔으나 모델이 로드되지 않았습니다.")
        return jsonify({"error": "Prediction requested but model is not loaded."}), 500

    try:
        data = request.get_json()
        # 입력 데이터를 pandas DataFrame으로 변환
        features = pd.DataFrame(data["features"], columns=['feature'])
        logger.debug(f"입력 특성 (DataFrame):\n{features}")

        # 모델의 predict 함수 호출
        prediction = model.predict(features)
        
        logger.info(f"예측 결과: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()}), 200
        
    except KeyError:
        logger.error("입력 데이터에 'features' 키가 누락되었습니다. 예상 형식: {'features': [[val1], [val2], ...]})")
        return jsonify({"error": "Invalid input: 'features' key is missing or data format is incorrect."} ), 400
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Gunicorn을 통해 실행되므로 직접 실행 코드는 불필요
    pass

