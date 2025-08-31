"""
ログ設定システム

構造化ログとリクエスト追跡を実装します。
"""

import json
import logging
import logging.config
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import get_settings

# リクエストIDのコンテキスト変数
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredFormatter(logging.Formatter):
    """構造化ログフォーマッター"""

    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをJSON形式でフォーマット"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # リクエストIDを追加
        request_id = request_id_var.get()
        if request_id:
            log_data['request_id'] = request_id

        # 例外情報を追加
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # 追加のコンテキスト情報を追加
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False)


class RequestContextFilter(logging.Filter):
    """リクエストコンテキストフィルター"""

    def filter(self, record: logging.LogRecord) -> bool:
        """ログレコードにリクエストIDを追加"""
        request_id = request_id_var.get()
        if request_id:
            record.request_id = request_id
        return True


def setup_logging(config_path: Optional[Path] = None) -> None:
    """ログ設定を初期化"""
    settings = get_settings()
    
    if config_path is None:
        config_path = settings.log_config_path

    # ログディレクトリを作成
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    try:
        if config_path.exists():
            # YAML設定ファイルから読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        else:
            # デフォルト設定を使用
            _setup_default_logging()
            
        # カスタムフィルターを追加
        for handler in logging.root.handlers:
            handler.addFilter(RequestContextFilter())
            
    except Exception as e:
        print(f"ログ設定エラー: {e}")
        _setup_default_logging()


def _setup_default_logging() -> None:
    """デフォルトログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/api.log', encoding='utf-8')
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """構造化ログ対応のロガーを取得"""
    logger = logging.getLogger(name)
    return logger


def set_request_id(request_id: str) -> None:
    """リクエストIDを設定"""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """現在のリクエストIDを取得"""
    return request_id_var.get()


def generate_request_id() -> str:
    """新しいリクエストIDを生成"""
    return str(uuid.uuid4())


class APILogger:
    """API専用ロガークラス"""

    def __init__(self, name: str):
        self.logger = get_logger(f"api.{name}")

    def info(self, message: str, **kwargs):
        """情報ログ"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告ログ"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """エラーログ"""
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """デバッグログ"""
        self._log(logging.DEBUG, message, **kwargs)

    def log_request(self, method: str, path: str, status_code: int, 
                   response_time: float, **kwargs):
        """リクエストログ"""
        self.info(
            f"{method} {path} - {status_code}",
            extra_data={
                'request_method': method,
                'request_path': path,
                'status_code': status_code,
                'response_time_ms': round(response_time * 1000, 2),
                **kwargs
            }
        )

    def log_prediction(self, model_type: str, success: bool, 
                      response_time: float, **kwargs):
        """予測ログ"""
        level = logging.INFO if success else logging.ERROR
        message = f"予測{model_type} - {'成功' if success else '失敗'}"
        
        self._log(
            level,
            message,
            extra_data={
                'prediction_type': model_type,
                'success': success,
                'response_time_ms': round(response_time * 1000, 2),
                **kwargs
            }
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """エラーログ（詳細情報付き）"""
        self.logger.error(
            f"エラー発生: {str(error)}",
            exc_info=True,
            extra={'extra_data': context or {}}
        )

    def _log(self, level: int, message: str, extra_data: Dict[str, Any] = None):
        """内部ログメソッド"""
        if extra_data:
            self.logger.log(level, message, extra={'extra_data': extra_data})
        else:
            self.logger.log(level, message)


# モジュールレベルのロガーインスタンス
api_logger = APILogger("main")
prediction_logger = APILogger("prediction")
security_logger = APILogger("security")
metrics_logger = APILogger("metrics")