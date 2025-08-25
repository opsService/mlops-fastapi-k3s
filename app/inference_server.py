# mlops-fastapi-app/app/inference_server.py
import base64
import io
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
loaded_model_source = None

def load_model_on_startup():
    """
    애플리케이션 시작 시 환경 변수를 확인하여 모델을 로드합니다.
    MLFLOW_INFERENCE_RUN_ID를 우선적으로 사용하고, 없으면 MODEL_FILE_PATH를 사용합니다.
    """
    global model, loaded_model_source
    
    run_id = os.getenv("MLFLOW_INFERENCE_RUN_ID")
    model_file_path = os.getenv("MODEL_FILE_PATH")
    model_uri = None

    if run_id:
        logger.info(f"MLflow Run ID를 사용하여 모델 로드를 시도합니다: {run_id}")
        model_uri = f"runs:/{run_id}/ml_model"
        loaded_model_source = f"run_id:{run_id}"
    elif model_file_path:
        logger.info(f"Model File Path를 사용하여 모델 로드를 시도합니다: {model_file_path}")
        model_uri = model_file_path
        loaded_model_source = f"file_path:{model_file_path}"
    else:
        logger.critical("모델 소스를 찾을 수 없습니다. 'MLFLOW_INFERENCE_RUN_ID' 또는 'MODEL_FILE_PATH' 환경 변수 중 하나를 설정해야 합니다.")
        sys.exit(1)

    try:
        logger.info(f"MLflow URI에서 모델 로딩 중: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"성공적으로 모델을 로드했습니다. 소스: {loaded_model_source}")

    except Exception as e:
        logger.critical(f"치명적 오류: URI '{model_uri}'에서 모델을 로드하지 못했습니다: {e}", exc_info=True)
        sys.exit(1)

# --- 애플리케이션 시작 시 모델 로드 ---
load_model_on_startup()


@app.route("/health", methods=["GET"])
def health():
    """
    헬스 체크 엔드포인트. 모델이 성공적으로 로드되었는지도 확인합니다.
    """
    if model is not None and loaded_model_source is not None:
        logger.debug("Health check requested: Server is healthy and model is loaded.")
        return jsonify({"status": "healthy", "model_source": loaded_model_source}), 200
    else:
        logger.error("Health check requested: Server is unhealthy because model is not loaded.")
        return jsonify({"status": "unhealthy", "reason": "Model is not loaded"}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    로드된 pyfunc 모델을 사용하여 예측을 수행합니다.
    Content-Type에 따라 JSON 또는 form-data 입력을 자동 처리합니다.
    """
    if model is None:
        logger.error("예측 요청이 들어왔으나 모델이 로드되지 않았습니다.")
        return jsonify({"error": "Prediction requested but model is not loaded."}), 500

    try:
        content_type = request.headers.get("content-type", "").lower()
        input_data = None

        if "application/json" in content_type:
            logger.debug("Received JSON request.")
            json_data = request.get_json()
            logger.debug(f"Parsed json_data from request body: {json_data}")

            # 테이블 데이터 형식 추론
            if isinstance(json_data, dict) and "features" in json_data:
                logger.debug(f"Handling 'features' key for tabular/regression data. Input: {json_data}")
                input_data = pd.DataFrame(json_data["features"])
            elif isinstance(json_data, dict) and "text" in json_data:
                logger.debug(f"Handling 'text' key for text classification. Input: {json_data}")
                input_data = pd.DataFrame(json_data['text'], columns=['text'])
            else:
                # 기타 JSON 기반 입력 (e.g., 텍스트)
                logger.warning(f"No specific data format detected. Passing raw JSON to model. Input: {json_data}")
                input_data = json_data

        elif "multipart/form-data" in content_type:
            logger.debug("Received multipart/form-data request.")
            if 'input_file' not in request.files:
                return jsonify({"error": "Missing 'input_file' in form-data"}), 400
            
            file = request.files['input_file']
            # pyfunc wrapper가 일관된 입력을 받도록 이미지를 base64로 변환
            img_bytes = file.read()
            b64_string = base64.b64encode(img_bytes).decode('utf-8')
            input_data = {"image": b64_string}

        else:
            return jsonify({"error": f"Unsupported Content-Type: {content_type}"}), 415

        # 모델의 predict 함수 호출
        prediction = model.predict(input_data)
        
        # prediction 결과를 JSON으로 변환 가능한 형태로 처리
        if isinstance(prediction, pd.DataFrame):
            # DataFrame을 dictionary의 list로 변환 (e.g., [{'col1': val1, ...}, ...])
            prediction_list = prediction.to_dict(orient="records")
        elif hasattr(prediction, 'tolist'):
            # numpy array 또는 pandas Series를 list로 변환
            prediction_list = prediction.tolist()
        else:
            # 이미 list이거나 다른 기본 타입인 경우
            prediction_list = prediction

        logger.info(f"Prediction successful.")
        return jsonify({"prediction": prediction_list}), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500
