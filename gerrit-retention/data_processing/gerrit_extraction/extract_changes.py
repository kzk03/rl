#!/usr/bin/env python3
"""
Gerrit Changes（変更）抽出スクリプト

Gerrit APIからChanges（変更）データを抽出し、構造化されたデータとして保存します。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.data_integration.gerrit_client import create_gerrit_client
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger, setup_logging

# ログ設定を初期化
setup_logging(level="INFO")
logger = get_logger(__name__)


class ChangesExtractor:
    """Changes（変更）抽出器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Changes抽出器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # Gerritクライアントを作成
        self.gerrit_client = create_gerrit_client()
        
        # 出力設定
        self.output_dir = Path("data/raw/gerrit_changes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Changes抽出器を初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get_config().to_dict()
    
    def extract_changes_for_project(
        self,
        project: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_details: bool = True
    ) -> List[Dict[str, Any]]:
        """
        プロジェクトのChanges（変更）を抽出
        
        Args:
            project: プロジェクト名
            start_date: 開始日時
            end_date: 終了日時
            include_details: 詳細情報を含めるかどうか
            
        Returns:
            List[Dict[str, Any]]: 変更情報のリスト
        """
        logger.info(f"プロジェクト '{project}' のChanges抽出を開始...")
        
        # バッチ処理用のコールバック関数
        def batch_callback(batch_changes: List[Dict[str, Any]], total_count: int):
            logger.info(f"バッチ処理中: {len(batch_changes)}個の変更を処理 (累計: {total_count}個)")
        
        # 変更一覧を取得
        changes = self.gerrit_client.get_changes_batch(
            project=project,
            start_date=start_date,
            end_date=end_date,
            batch_callback=batch_callback
        )
        
        # 詳細情報を取得
        if include_details:
            logger.info("変更の詳細情報を取得中...")
            detailed_changes = []
            
            for i, change in enumerate(changes):
                try:
                    # 詳細情報を取得
                    change_detail = self.gerrit_client.get_change_detail(change["id"])
                    
                    # 基本情報と詳細情報をマージ
                    enhanced_change = {**change, **change_detail}
                    detailed_changes.append(enhanced_change)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"詳細情報取得進捗: {i + 1}/{len(changes)}")
                        
                except Exception as e:
                    logger.warning(f"変更 '{change['id']}' の詳細取得に失敗: {e}")
                    detailed_changes.append(change)  # 基本情報のみ保存
            
            changes = detailed_changes
        
        logger.info(f"プロジェクト '{project}' から {len(changes)}個の変更を抽出しました")
        return changes
    
    def extract_changes_for_all_projects(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_details: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        全プロジェクトのChanges（変更）を抽出
        
        Args:
            start_date: 開始日時
            end_date: 終了日時
            include_details: 詳細情報を含めるかどうか
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: プロジェクト別の変更情報
        """
        logger.info("全プロジェクトのChanges抽出を開始...")
        
        # 設定からプロジェクト一覧を取得
        projects = self.config_manager.get("gerrit.projects", [])
        
        if not projects:
            logger.warning("設定にプロジェクトが指定されていません。Gerritから取得します...")
            try:
                gerrit_projects = self.gerrit_client.get_projects()
                projects = [p["name"] for p in gerrit_projects[:5]]  # 最初の5個のみ
                logger.info(f"Gerritから取得したプロジェクト: {projects}")
            except Exception as e:
                logger.error(f"プロジェクト一覧の取得に失敗: {e}")
                return {}
        
        all_changes = {}
        
        for project in projects:
            try:
                changes = self.extract_changes_for_project(
                    project=project,
                    start_date=start_date,
                    end_date=end_date,
                    include_details=include_details
                )
                all_changes[project] = changes
                
            except Exception as e:
                logger.error(f"プロジェクト '{project}' の抽出に失敗: {e}")
                all_changes[project] = []
        
        total_changes = sum(len(changes) for changes in all_changes.values())
        logger.info(f"全プロジェクトから合計 {total_changes}個の変更を抽出しました")
        
        return all_changes
    
    def save_changes(
        self,
        changes_data: Dict[str, List[Dict[str, Any]]],
        filename_prefix: str = "changes"
    ) -> Path:
        """
        変更データを保存
        
        Args:
            changes_data: 変更データ
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        logger.info(f"変更データを保存中: {output_path}")
        
        # メタデータを追加
        output_data = {
            "metadata": {
                "extraction_timestamp": timestamp,
                "total_projects": len(changes_data),
                "total_changes": sum(len(changes) for changes in changes_data.values()),
                "projects": list(changes_data.keys())
            },
            "data": changes_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"変更データを保存しました: {output_path}")
        return output_path
    
    def extract_and_save(
        self,
        projects: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_details: bool = True,
        filename_prefix: str = "changes"
    ) -> Path:
        """
        変更データを抽出して保存
        
        Args:
            projects: プロジェクト一覧（Noneの場合は全プロジェクト）
            start_date: 開始日時
            end_date: 終了日時
            include_details: 詳細情報を含めるかどうか
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        try:
            if projects:
                # 指定されたプロジェクトのみ抽出
                changes_data = {}
                for project in projects:
                    changes = self.extract_changes_for_project(
                        project=project,
                        start_date=start_date,
                        end_date=end_date,
                        include_details=include_details
                    )
                    changes_data[project] = changes
            else:
                # 全プロジェクト抽出
                changes_data = self.extract_changes_for_all_projects(
                    start_date=start_date,
                    end_date=end_date,
                    include_details=include_details
                )
            
            # データを保存
            output_path = self.save_changes(changes_data, filename_prefix)
            
            return output_path
            
        finally:
            # クライアントを終了
            self.gerrit_client.close()


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerrit Changes抽出スクリプト")
    parser.add_argument(
        "--projects",
        nargs="+",
        help="抽出するプロジェクト名（指定しない場合は全プロジェクト）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="開始日（YYYY-MM-DD形式）"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="終了日（YYYY-MM-DD形式）"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="詳細情報を取得しない"
    )
    parser.add_argument(
        "--output-prefix",
        default="changes",
        help="出力ファイル名のプレフィックス"
    )
    
    args = parser.parse_args()
    
    # 日付を解析
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"無効な開始日形式: {args.start_date}")
            return 1
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"無効な終了日形式: {args.end_date}")
            return 1
    
    # 抽出器を作成
    extractor = ChangesExtractor()
    
    try:
        # 抽出を実行
        output_path = extractor.extract_and_save(
            projects=args.projects,
            start_date=start_date,
            end_date=end_date,
            include_details=not args.no_details,
            filename_prefix=args.output_prefix
        )
        
        logger.info(f"Changes抽出が完了しました: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Changes抽出中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())