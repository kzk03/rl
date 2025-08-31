"""
Gerrit APIクライアントの単体テスト
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from gerrit_retention.data_integration.gerrit_client import (
    GerritAPIError,
    GerritAuthenticationError,
    GerritClient,
    GerritRateLimitError,
    create_gerrit_client,
    test_gerrit_connection,
)


class TestGerritClient:
    """GerritClientのテストクラス"""
    
    def setup_method(self):
        """テストメソッドの前処理"""
        self.config = {
            "url": "https://gerrit.example.com",
            "auth": {
                "username": "test_user",
                "password": "test_password"
            },
            "data_extraction": {
                "batch_size": 100,
                "rate_limit_delay": 0.1,
                "max_retries": 2,
                "timeout": 10
            }
        }
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    def test_init(self, mock_config_manager):
        """初期化のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        client = GerritClient(self.config)
        
        assert client.base_url == "https://gerrit.example.com"
        assert client.username == "test_user"
        assert client.password == "test_password"
        assert client.batch_size == 100
        assert client.rate_limit_delay == 0.1
        assert client.max_retries == 2
        assert client.timeout == 10
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    def test_parse_gerrit_response(self, mock_config_manager):
        """Gerritレスポンス解析のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        client = GerritClient(self.config)
        
        # 通常のJSONレスポンス
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        
        result = client._parse_gerrit_response(mock_response)
        assert result == {"key": "value"}
        
        # Gerritプレフィックス付きレスポンス
        mock_response.text = ')]}\'{"key": "value"}'
        
        result = client._parse_gerrit_response(mock_response)
        assert result == {"key": "value"}
        
        # 空のレスポンス
        mock_response.text = ''
        
        result = client._parse_gerrit_response(mock_response)
        assert result == {}
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch('requests.Session.request')
    def test_make_request_success(self, mock_request, mock_config_manager):
        """正常なリクエストのテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "success"}'
        mock_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        response = client._make_request("GET", "/test")
        
        assert response.status_code == 200
        mock_request.assert_called_once()
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch('requests.Session.request')
    def test_make_request_authentication_error(self, mock_request, mock_config_manager):
        """認証エラーのテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # 401エラーレスポンスを設定
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        with pytest.raises(GerritAuthenticationError):
            client._make_request("GET", "/test")
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch('requests.Session.request')
    def test_make_request_rate_limit_error(self, mock_request, mock_config_manager):
        """レート制限エラーのテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # 429エラーレスポンスを設定
        mock_response = Mock()
        mock_response.status_code = 429
        mock_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        with pytest.raises(GerritRateLimitError):
            client._make_request("GET", "/test")
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_test_connection_success(self, mock_make_request, mock_config_manager):
        """接続テスト成功のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.text = ')]}\'{"version": "3.4.0"}'
        mock_make_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        result = client.test_connection()
        
        assert result is True
        mock_make_request.assert_called_once_with("GET", "/config/server/version")
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_test_connection_failure(self, mock_make_request, mock_config_manager):
        """接続テスト失敗のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # エラーを発生させる
        mock_make_request.side_effect = GerritAPIError("Connection failed")
        
        client = GerritClient(self.config)
        
        result = client.test_connection()
        
        assert result is False
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_get_projects(self, mock_make_request, mock_config_manager):
        """プロジェクト一覧取得のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.text = ')]}\'{"project1": {"id": "project1"}, "project2": {"id": "project2"}}'
        mock_make_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        projects = client.get_projects()
        
        assert len(projects) == 2
        assert projects[0]["name"] == "project1"
        assert projects[1]["name"] == "project2"
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_get_changes(self, mock_make_request, mock_config_manager):
        """変更一覧取得のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.text = ')]}\'[{"id": "change1", "project": "test-project"}, {"id": "change2", "project": "test-project"}]'
        mock_make_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        changes = client.get_changes("test-project")
        
        assert len(changes) == 2
        assert changes[0]["id"] == "change1"
        assert changes[1]["id"] == "change2"
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_get_changes_with_date_range(self, mock_make_request, mock_config_manager):
        """日付範囲指定での変更一覧取得のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.text = ')]}\'[{"id": "change1"}]'
        mock_make_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        changes = client.get_changes("test-project", start_date=start_date, end_date=end_date)
        
        assert len(changes) == 1
        
        # リクエストパラメータを確認
        call_args = mock_make_request.call_args
        params = call_args[1]["params"]
        assert "after:2023-01-01" in params["q"]
        assert "before:2023-12-31" in params["q"]
    
    @patch('gerrit_retention.data_integration.gerrit_client.get_config_manager')
    @patch.object(GerritClient, '_make_request')
    def test_get_change_detail(self, mock_make_request, mock_config_manager):
        """変更詳細取得のテスト"""
        mock_config_manager.return_value.get.return_value = {}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.text = ')]}\'{"id": "change1", "subject": "Test change"}'
        mock_make_request.return_value = mock_response
        
        client = GerritClient(self.config)
        
        change_detail = client.get_change_detail("change1")
        
        assert change_detail["id"] == "change1"
        assert change_detail["subject"] == "Test change"
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args[0] == ("GET", "/changes/change1")
        assert "params" in call_args[1]


class TestUtilityFunctions:
    """ユーティリティ関数のテストクラス"""
    
    @patch('gerrit_retention.data_integration.gerrit_client.GerritClient')
    def test_create_gerrit_client(self, mock_gerrit_client):
        """Gerritクライアント作成のテスト"""
        config = {"url": "https://gerrit.example.com"}
        
        create_gerrit_client(config)
        
        mock_gerrit_client.assert_called_once_with(config)
    
    @patch('gerrit_retention.data_integration.gerrit_client.GerritClient')
    def test_test_gerrit_connection(self, mock_gerrit_client):
        """Gerrit接続テストのテスト"""
        # モッククライアントを設定
        mock_client_instance = Mock()
        mock_client_instance.test_connection.return_value = True
        mock_gerrit_client.return_value = mock_client_instance
        
        config = {"url": "https://gerrit.example.com"}
        
        result = test_gerrit_connection(config)
        
        assert result is True
        mock_client_instance.test_connection.assert_called_once()
        mock_client_instance.close.assert_called_once()