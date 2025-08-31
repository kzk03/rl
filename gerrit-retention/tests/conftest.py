"""
テスト設定ファイル

pytest の共通設定とフィクスチャを定義します。
"""

from pathlib import Path

import pytest
from api.core.config import Settings
from api.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def test_settings():
    """テスト用設定"""
    return Settings(
        config_dir=Path("config"),
        log_level="DEBUG"
    )


@pytest.fixture
def test_app():
    """テスト用 FastAPI アプリケーション"""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app):
    """テスト用 HTTP クライアント"""
    return TestClient(test_app)


@pytest.fixture
def sample_developer_profile():
    """サンプル開発者プロファイル"""
    return {
        "email": "test@example.com",
        "name": "Test Developer",
        "expertise_level": 0.7,
        "activity_pattern": {"commits_per_week": 10},
        "collaboration_quality": 0.8,
        "stress_level": 0.4,
        "project": "test-project"
    }