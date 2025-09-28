"""
テスト設定ファイル

pytest の共通設定とフィクスチャを定義します。
"""

from pathlib import Path

import pytest

# FastAPI TestClient is optional for non-API tests; avoid hard failure when httpx is missing
try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:  # starlette may require httpx; make this optional
    TestClient = None  # type: ignore



@pytest.fixture
def test_settings():
    """テスト用設定"""
    try:
        from api.core.config import Settings  # type: ignore
    except Exception:
        pytest.skip("API Settings not available (pydantic settings missing)")
    # 既定（configs/api/ 優先）を使用
    return Settings(log_level="DEBUG")


@pytest.fixture
def test_app():
    """テスト用 FastAPI アプリケーション"""
    try:
        from api.main import create_app  # type: ignore
    except Exception:
        pytest.skip("FastAPI app factory not available")
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app):
    """テスト用 HTTP クライアント"""
    if TestClient is None:
        pytest.skip("TestClient not available (httpx not installed)")
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