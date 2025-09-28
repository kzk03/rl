"""
セットアップ検証テスト

基本的な設定とアプリケーション起動をテストします。
"""

from pathlib import Path

import pytest

from api.core.config import Settings, get_settings
from api.core.exceptions import APIException, ModelNotAvailableError, ValidationError
from api.core.logging import generate_request_id, get_logger


class TestConfiguration:
    """設定管理のテスト"""

    def test_default_settings(self):
        """デフォルト設定のテスト"""
        settings = Settings()
        
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 8080
        assert settings.defaults.satisfaction_level == 0.7
        assert settings.defaults.confidence_threshold == 0.5

    def test_config_file_loading(self):
        """設定ファイル読み込みのテスト"""
        # 既定（configs/api/ 優先）を使用
        settings = Settings()
        
        # 設定ファイルが存在する場合の検証（新パス）
        if (Path("configs/api") / "api_config.yaml").exists():
            assert settings.model_configs is not None
            assert isinstance(settings.model_configs, dict)

    def test_model_config_access(self):
        """モデル設定アクセスのテスト"""
        settings = Settings()
        
        # モデル設定が存在しない場合
        assert settings.get_model_config("nonexistent") is None
        assert not settings.is_model_enabled("nonexistent")


class TestExceptions:
    """例外クラスのテスト"""

    def test_api_exception(self):
        """基本API例外のテスト"""
        from api.core.exceptions import ErrorCode
        
        exc = APIException(
            message="テストエラー",
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500
        )
        
        assert exc.message == "テストエラー"
        assert exc.error_code == ErrorCode.INTERNAL_ERROR
        assert exc.status_code == 500
        
        exc_dict = exc.to_dict()
        assert exc_dict["error_code"] == "INTERNAL_ERROR"
        assert exc_dict["message"] == "テストエラー"

    def test_validation_error(self):
        """バリデーションエラーのテスト"""
        field_errors = {"email": "無効なメールアドレス"}
        
        exc = ValidationError(
            message="バリデーションエラー",
            field_errors=field_errors
        )
        
        assert exc.status_code == 422
        assert exc.details["field_errors"] == field_errors

    def test_model_not_available_error(self):
        """モデル利用不可エラーのテスト"""
        exc = ModelNotAvailableError("retention", "ファイルが見つかりません")
        
        assert exc.status_code == 503
        assert "retention" in exc.message
        assert exc.details["model_name"] == "retention"


class TestLogging:
    """ログ機能のテスト"""

    def test_logger_creation(self):
        """ロガー作成のテスト"""
        logger = get_logger("test")
        assert logger.name == "test"

    def test_request_id_generation(self):
        """リクエストID生成のテスト"""
        request_id = generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) > 0
        
        # 複数回生成して重複しないことを確認
        request_id2 = generate_request_id()
        assert request_id != request_id2


class TestApplication:
    """アプリケーションのテスト"""

    def test_app_creation(self, test_app):
        """アプリケーション作成のテスト"""
        assert test_app is not None
        assert test_app.title == "Gerrit 開発者定着予測 API"

    def test_root_endpoint(self, test_client):
        """ルートエンドポイントのテスト"""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Gerrit 開発者定着予測 API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"