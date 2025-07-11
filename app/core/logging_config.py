import logging.config

from app.core.config import settings  # 설정 값 임포트


def setup_logging():
    log_level = settings.LOG_LEVEL.upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "json": {  # JSON 포맷터 추가 (ELK/Loki 등에서 파싱하기 용이)
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": log_level,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # 표준 출력으로 로그 전송
            },
            "json_handler": {  # JSON 핸들러 추가
                "level": log_level,
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],  # 기본적으로 standard 포맷 사용
                "level": log_level,
                "propagate": False,
            },
            "app": {  # 'app'으로 시작하는 모든 로거
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "mlflow": {
                "handlers": ["default"],
                "level": log_level,  # MLflow 로깅 레벨도 제어
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(logging_config)
    # 특정 로거에 JSON 핸들러를 적용하려면
    # logging.getLogger("your_specific_json_logger").addHandler(logging_config["handlers"]["json_handler"])

    # 로거 객체 생성 시 이름을 잘 부여하여 필터링 가능하게 합니다.
    # logger = logging.getLogger(__name__) # app.routers.train, app.core.k8s_client 등
