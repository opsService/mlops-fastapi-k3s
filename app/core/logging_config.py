import json
import logging
import logging.config
import sys  # sys.stdout 사용을 위해 추가

from app.core.config import settings  # 설정 값 임포트


class JsonFormatter(logging.Formatter):
    """
    JSON 형식으로 로그를 포맷하는 커스텀 포맷터.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # 예외 정보가 있다면 추가
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        # 스택 추적 정보가 있다면 추가
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(log_record, ensure_ascii=False)


def setup_logging():
    # settings.LOG_LEVEL이 유효한 문자열인지 확인하고, 아니면 기본값으로 설정
    log_level = (
        settings.LOG_LEVEL.upper() if isinstance(settings.LOG_LEVEL, str) else "INFO"
    )
    valid_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    if log_level not in valid_levels:
        logging.warning(
            f"유효하지 않은 LOG_LEVEL '{settings.LOG_LEVEL}'이 설정되었습니다. 기본값인 'INFO'로 설정합니다."
        )
        log_level = "INFO"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # 기존 로거를 비활성화하지 않고 유지합니다.
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # JsonFormatter 클래스 참조
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                # "json_ensure_ascii": False, # 필요에 따라 유니코드 문자를 이스케이프하지 않도록 설정
            },
        },
        "handlers": {
            "console_standard": {
                "level": log_level,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_json": {
                "level": log_level,
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # 루트 로거 (기본 로거):
            # Gunicorn 환경에서는 루트 로거에 대한 제어를 Gunicorn에 맡기거나
            # 충돌을 피하기 위해 propagate=False로 설정하고 명시적인 핸들러를 부여하는 것이 일반적입니다.
            "": {
                "handlers": ["console_standard"],  # 기본적으로 표준 포맷 핸들러 사용
                "level": log_level,
                "propagate": False,  # 다른 로거들이 이 로거로 메시지를 전파하지 않도록 합니다.
            },
            # 'app'으로 시작하는 모든 로거:
            # 우리 애플리케이션의 핵심 로거이며, JSON 포맷을 사용하도록 설정합니다.
            "app": {
                "handlers": ["console_json"],  # JSON 핸들러 사용
                "level": log_level,
                "propagate": False,  # 이 로거의 메시지가 루트 로거로 전파되는 것을 막아 이중 로깅을 방지합니다.
            },
            # Uvicorn 로거:
            # FastAPI가 내부적으로 사용하는 Uvicorn 로거에 대한 설정입니다.
            "uvicorn": {
                "handlers": ["console_standard"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console_standard"],
                "level": log_level,
                "propagate": False,
            },
            # MLflow 로거:
            # MLflow 관련 로그에 대한 설정입니다.
            "mlflow": {
                "handlers": ["console_standard"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    try:
        logging.config.dictConfig(logging_config)
        logging.info(f"로깅 설정 완료. 레벨: {log_level}")
    except Exception as e:
        # dictConfig 설정 실패 시, 오류를 표준 에러 출력으로 내보내고 기본 로깅으로 폴백
        print(f"치명적 오류: dictConfig 로깅 설정 실패: {e}", file=sys.stderr)
        # 기본 로깅 설정으로 폴백
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.error(f"dictConfig 실패. 기본 로깅으로 폴백. 오류: {e}")
        # 애플리케이션 시작 실패를 명확히 하기 위해 예외를 다시 발생시킵니다.
        raise
