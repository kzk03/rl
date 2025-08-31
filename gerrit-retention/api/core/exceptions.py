"""
カスタム例外クラス

API固有の例外とエラーハンドリングを実装します。
"""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """エラーコード定義"""
    # バリデーションエラー
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # モデル関連エラー
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    
    # システムエラー
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # セキュリティエラー
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # リソースエラー
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"


class APIException(Exception):
    """基本API例外クラス"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """例外を辞書形式に変換"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }


class ValidationError(APIException):
    """バリデーションエラー"""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field_errors:
            error_details["field_errors"] = field_errors
            
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=422,
            details=error_details
        )


class ModelNotAvailableError(APIException):
    """モデル利用不可エラー"""
    
    def __init__(self, model_name: str, reason: str = "モデルが読み込まれていません"):
        super().__init__(
            message=f"モデル '{model_name}' は利用できません: {reason}",
            error_code=ErrorCode.MODEL_NOT_AVAILABLE,
            status_code=503,
            details={"model_name": model_name, "reason": reason}
        )


class ModelLoadError(APIException):
    """モデル読み込みエラー"""
    
    def __init__(self, model_name: str, error_message: str):
        super().__init__(
            message=f"モデル '{model_name}' の読み込みに失敗しました: {error_message}",
            error_code=ErrorCode.MODEL_LOAD_ERROR,
            status_code=500,
            details={"model_name": model_name, "error_message": error_message}
        )


class PredictionError(APIException):
    """予測エラー"""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        prediction_type: Optional[str] = None
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if prediction_type:
            details["prediction_type"] = prediction_type
            
        super().__init__(
            message=message,
            error_code=ErrorCode.PREDICTION_ERROR,
            status_code=500,
            details=details
        )


class AuthenticationError(APIException):
    """認証エラー"""
    
    def __init__(self, message: str = "認証が必要です"):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            status_code=401,
            details={"auth_required": True}
        )


class AuthorizationError(APIException):
    """認可エラー"""
    
    def __init__(self, message: str = "アクセス権限がありません", required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
            
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_ERROR,
            status_code=403,
            details=details
        )


class RateLimitExceededError(APIException):
    """レート制限エラー"""
    
    def __init__(
        self,
        message: str = "リクエスト制限を超過しました",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None
    ):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if limit:
            details["requests_per_minute"] = limit
            
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details=details
        )


class ResourceNotFoundError(APIException):
    """リソース未発見エラー"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} '{resource_id}' が見つかりません",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ServiceUnavailableError(APIException):
    """サービス利用不可エラー"""
    
    def __init__(self, message: str = "サービスが一時的に利用できません", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
            
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=503,
            details=details
        )


class TimeoutError(APIException):
    """タイムアウトエラー"""
    
    def __init__(self, operation: str, timeout_seconds: int):
        super().__init__(
            message=f"操作 '{operation}' がタイムアウトしました ({timeout_seconds}秒)",
            error_code=ErrorCode.TIMEOUT_ERROR,
            status_code=408,
            details={"operation": operation, "timeout_seconds": timeout_seconds}
        )


class InternalError(APIException):
    """内部エラー"""
    
    def __init__(self, message: str = "内部エラーが発生しました", error_id: Optional[str] = None):
        details = {}
        if error_id:
            details["error_id"] = error_id
            
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details
        )