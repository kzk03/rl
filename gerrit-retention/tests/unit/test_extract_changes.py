"""
Changes抽出スクリプトの単体テスト
"""

import json

# テスト対象のインポート
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "data_processing" / "gerrit_extraction"))

from extract_changes import ChangesExtractor


class TestChangesExtractor:
    """ChangesExtractorのテストクラス"""
    
    def setup_method(self):
        """テストメソッドの前処理"""
        self.mock_config = {
            "gerrit": {
                "projects": ["test-project-1", "test-project-2"]
            }
        }
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_init(self, mock_create_client, mock_config_manager):
        """初期化のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        assert extractor.gerrit_client == mock_client
        assert extractor.output_dir.name == "gerrit_changes"
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_changes_for_project(self, mock_create_client, mock_config_manager):
        """プロジェクトの変更抽出のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        
        # モッククライアントを設定
        mock_client = Mock()
        mock_changes = [
            {"id": "change1", "project": "test-project"},
            {"id": "change2", "project": "test-project"}
        ]
        mock_client.get_changes_batch.return_value = mock_changes
        mock_client.get_change_detail.side_effect = [
            {"id": "change1", "subject": "Test change 1", "detailed": True},
            {"id": "change2", "subject": "Test change 2", "detailed": True}
        ]
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        result = extractor.extract_changes_for_project("test-project")
        
        assert len(result) == 2
        assert result[0]["id"] == "change1"
        assert result[0]["detailed"] is True
        assert result[1]["id"] == "change2"
        assert result[1]["detailed"] is True
        
        mock_client.get_changes_batch.assert_called_once()
        assert mock_client.get_change_detail.call_count == 2
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_changes_for_project_without_details(self, mock_create_client, mock_config_manager):
        """詳細なしでのプロジェクト変更抽出のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        
        # モッククライアントを設定
        mock_client = Mock()
        mock_changes = [{"id": "change1", "project": "test-project"}]
        mock_client.get_changes_batch.return_value = mock_changes
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        result = extractor.extract_changes_for_project("test-project", include_details=False)
        
        assert len(result) == 1
        assert result[0]["id"] == "change1"
        
        # 詳細取得が呼ばれていないことを確認
        mock_client.get_change_detail.assert_not_called()
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_changes_for_all_projects(self, mock_create_client, mock_config_manager):
        """全プロジェクトの変更抽出のテスト"""
        mock_config_manager.return_value.get.return_value = ["project1", "project2"]
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        
        # モッククライアントを設定
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        # extract_changes_for_projectをモック
        with patch.object(extractor, 'extract_changes_for_project') as mock_extract:
            mock_extract.side_effect = [
                [{"id": "change1", "project": "project1"}],
                [{"id": "change2", "project": "project2"}]
            ]
            
            result = extractor.extract_changes_for_all_projects()
            
            assert len(result) == 2
            assert "project1" in result
            assert "project2" in result
            assert len(result["project1"]) == 1
            assert len(result["project2"]) == 1
            
            assert mock_extract.call_count == 2
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_save_changes(self, mock_create_client, mock_config_manager):
        """変更データ保存のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        mock_create_client.return_value = Mock()
        
        extractor = ChangesExtractor()
        
        # テストデータ
        changes_data = {
            "project1": [{"id": "change1"}],
            "project2": [{"id": "change2"}, {"id": "change3"}]
        }
        
        # 一時ディレクトリを使用
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor.output_dir = Path(temp_dir)
            
            output_path = extractor.save_changes(changes_data, "test_changes")
            
            # ファイルが作成されたことを確認
            assert output_path.exists()
            assert output_path.name.startswith("test_changes_")
            assert output_path.suffix == ".json"
            
            # ファイル内容を確認
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert "metadata" in saved_data
            assert "data" in saved_data
            assert saved_data["metadata"]["total_projects"] == 2
            assert saved_data["metadata"]["total_changes"] == 3
            assert saved_data["data"] == changes_data
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_and_save_with_projects(self, mock_create_client, mock_config_manager):
        """指定プロジェクトでの抽出・保存のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        # extract_changes_for_projectとsave_changesをモック
        with patch.object(extractor, 'extract_changes_for_project') as mock_extract, \
             patch.object(extractor, 'save_changes') as mock_save:
            
            mock_extract.return_value = [{"id": "change1"}]
            mock_save.return_value = Path("/test/output.json")
            
            result = extractor.extract_and_save(projects=["test-project"])
            
            assert result == Path("/test/output.json")
            mock_extract.assert_called_once_with(
                project="test-project",
                start_date=None,
                end_date=None,
                include_details=True
            )
            mock_save.assert_called_once()
            mock_client.close.assert_called_once()
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_and_save_all_projects(self, mock_create_client, mock_config_manager):
        """全プロジェクトでの抽出・保存のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        # extract_changes_for_all_projectsとsave_changesをモック
        with patch.object(extractor, 'extract_changes_for_all_projects') as mock_extract, \
             patch.object(extractor, 'save_changes') as mock_save:
            
            mock_extract.return_value = {"project1": [{"id": "change1"}]}
            mock_save.return_value = Path("/test/output.json")
            
            result = extractor.extract_and_save()
            
            assert result == Path("/test/output.json")
            mock_extract.assert_called_once_with(
                start_date=None,
                end_date=None,
                include_details=True
            )
            mock_save.assert_called_once()
            mock_client.close.assert_called_once()
    
    @patch('extract_changes.get_config_manager')
    @patch('extract_changes.create_gerrit_client')
    def test_extract_with_date_range(self, mock_create_client, mock_config_manager):
        """日付範囲指定での抽出のテスト"""
        mock_config_manager.return_value.get_config.return_value.to_dict.return_value = self.mock_config
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        extractor = ChangesExtractor()
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        with patch.object(extractor, 'extract_changes_for_project') as mock_extract:
            mock_extract.return_value = [{"id": "change1"}]
            
            extractor.extract_changes_for_project(
                "test-project",
                start_date=start_date,
                end_date=end_date
            )
            
            mock_extract.assert_called_once_with(
                "test-project",
                start_date=start_date,
                end_date=end_date
            )