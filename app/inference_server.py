# mlops-fastapi-app/app/inference_server.py
import logging
import os
import sys

import mlflow.pyfunc
import pandas as pd
from app.core.logging_config import setup_logging
from flask import Flask, jsonify, request

# --- 로깅 및 Flask 앱 설정 ---
setup_logging()
logger = logging.getLogger("inference_server")
app = Flask(__name__)

# --- 전역 변수 ---
model = None
loaded_run_id = None

def load_model_on_startup():
    """
    애플리케이션 시작 시 환경 변수에서 MLflow Run ID를 읽어 모델을 로드합니다.
    """
    global model, loaded_run_id
    
    run_id = os.getenv("MLFLOW_INFERENCE_RUN_ID")
    if not run_id:
        logger.critical("환경 변수 'MLFLOW_INFERENCE_RUN_ID'가 설정되지 않았습니다. 모델을 로드할 수 없습니다.")
        sys.exit(1)

    logger.info(f"서버 시작 시 모델 로드를 시도합니다. Run ID: {run_id}")
    
    try:
        model_uri = f"runs:/{run_id}/ml_model"
        logger.info(f"MLflow URI에서 모델 로딩 중: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        loaded_run_id = run_id
        
        logger.info(f"성공적으로 모델을 로드했습니다. Run ID: {run_id}")

    except Exception as e:
        logger.critical(f"치명적 오류: Run ID {run_id}의 모델을 로드하지 못했습니다: {e}", exc_info=True)
        sys.exit(1)

# --- 애플리케이션 시작 시 모델 로드 ---
load_model_on_startup()


@app.route("/health", methods=["GET"])
def health():
    """
    헬스 체크 엔드포인트. 모델이 성공적으로 로드되었는지도 확인합니다.
    """
    if model is not None and loaded_run_id is not None:
        logger.debug("Health check requested: Server is healthy and model is loaded.")
        return jsonify({"status": "healthy", "run_id": loaded_run_id}), 200
    else:
        logger.error("Health check requested: Server is unhealthy because model is not loaded.")
        return jsonify({"status": "unhealthy", "reason": "Model is not loaded"}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    로드된 pyfunc 모델을 사용하여 예측을 수행합니다.
    입력 데이터의 형식에 따라 테이블 또는 다른 유형으로 자동 처리합니다.
    """
    if model is None:
        logger.error("예측 요청이 들어왔으나 모델이 로드되지 않았습니다.")
        return jsonify({"error": "Prediction requested but model is not loaded."}), 500

    try:
        data = request.get_json()
        input_data = None

        # 입력 데이터 형식에 따른 분기 처리
        if isinstance(data, dict) and "features" in data:
            input_data = pd.DataFrame(data["features"])
            logger.debug(f"Tabular input detected. DataFrame Shape: {input_data.shape}")
        elif isinstance(data, dict) and "columns" in data and "data" in data:
            input_data = pd.DataFrame(data['data'], columns=data['columns'])
            logger.debug(f"Pandas split-orient input detected. DataFrame Shape: {input_data.shape}")
        else:
            # 테이블 형식이 아닌 경우, 데이터를 그대로 전달 (e.g., 이미지, 텍스트)
            input_data = data
            logger.debug("Non-tabular input detected, passing data directly to model.")

        # 모델의 predict 함수 호출
        prediction = model.predict(input_data)
        
        # 결과가 numpy array일 경우 list로 변환
        if hasattr(prediction, 'tolist'):
            prediction_list = prediction.tolist()
        else:
            prediction_list = prediction

        logger.info(f"Prediction successful. Returning {len(prediction_list)} results.")
        return jsonify({"prediction": prediction_list}), 200
        
    except KeyError:
        logger.error("Invalid input format: 'features' or other required key is missing.")
        return jsonify({"error": "Invalid input format."} ), 400
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500