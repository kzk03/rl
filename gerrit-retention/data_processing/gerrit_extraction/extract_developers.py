#!/usr/bin/env python3
"""
Gerrit Developers（開発者）抽出スクリプト

Gerrit APIからDevelopers（開発者）データを抽出し、構造化されたデータとして保存します。
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.data_integration.gerrit_client import create_gerrit_client
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger, setup_logging

# ログ設定を初期化
setup_logging(level="INFO")
logger = get_logger(__name__)


class DevelopersExtractor:
    """Developers（開発者）抽出器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Developers抽出器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # Gerritクライアントを作成
        self.gerrit_client = create_gerrit_client()
        
        # 出力設定
        self.output_dir = Path("data/raw/gerrit_developers")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Developers抽出器を初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get_config().to_dict()
    
    def _calculate_developer_stats(
        self,
        changes_data: List[Dict[str, Any]],
        reviews_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        開発者統計を計算
        
        Args:
            changes_data: 変更データ
            reviews_data: レビューデータ
            
        Returns:
            Dict[str, Dict[str, Any]]: 開発者統計
        """
        logger.info("開発者統計を計算中...")
        
        developers = defaultdict(lambda: {
            "email": "",
            "name": "",
            "first_activity": None,
            "last_activity": None,
            "activity_days": set(),
            
            # 変更統計
            "changes_authored": 0,
            "changes_merged": 0,
            "changes_abandoned": 0,
            "lines_added": 0,
            "lines_deleted": 0,
            "files_changed": set(),
            "projects_contributed": set(),
            
            # レビュー統計
            "reviews_given": 0,
            "reviews_received": 0,
            "review_scores_given": defaultdict(int),
            "review_scores_received": defaultdict(int),
            "avg_review_score_given": 0.0,
            "avg_review_score_received": 0.0,
            
            # 協力関係
            "collaborated_with": set(),
            "reviewed_by": set(),
            "reviewed_for": set(),
            
            # 専門性
            "file_extensions": defaultdict(int),
            "directories": defaultdict(int),
            "technical_domains": set(),
        })
        
        # 変更データから統計を計算
        for change in changes_data:
            owner_email = change.get("owner", {}).get("email", "")
            if not owner_email:
                continue
            
            dev = developers[owner_email]
            
            # 基本情報
            dev["email"] = owner_email
            dev["name"] = change.get("owner", {}).get("name", "")
            
            # 活動日時
            created_date = change.get("created", "")
            updated_date = change.get("updated", "")
            
            for date_str in [created_date, updated_date]:
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        activity_date = date_obj.date()
                        
                        dev["activity_days"].add(activity_date)
                        
                        if dev["first_activity"] is None or date_obj < dev["first_activity"]:
                            dev["first_activity"] = date_obj
                        
                        if dev["last_activity"] is None or date_obj > dev["last_activity"]:
                            dev["last_activity"] = date_obj
                            
                    except Exception as e:
                        logger.debug(f"日付解析エラー: {date_str} - {e}")
            
            # 変更統計
            dev["changes_authored"] += 1
            
            status = change.get("status", "")
            if status == "MERGED":
                dev["changes_merged"] += 1
            elif status == "ABANDONED":
                dev["changes_abandoned"] += 1
            
            # プロジェクト
            project = change.get("project", "")
            if project:
                dev["projects_contributed"].add(project)
            
            # ファイル・行数統計
            revisions = change.get("revisions", {})
            if revisions:
                latest_revision = list(revisions.values())[-1]
                files = latest_revision.get("files", {})
                
                for file_path, file_info in files.items():
                    dev["files_changed"].add(file_path)
                    
                    # 拡張子を抽出
                    if "." in file_path:
                        ext = file_path.split(".")[-1].lower()
                        dev["file_extensions"][ext] += 1
                    
                    # ディレクトリを抽出
                    if "/" in file_path:
                        directory = "/".join(file_path.split("/")[:-1])
                        dev["directories"][directory] += 1
                    
                    # 行数を加算
                    dev["lines_added"] += file_info.get("lines_inserted", 0)
                    dev["lines_deleted"] += file_info.get("lines_deleted", 0)
        
        # レビューデータから統計を計算
        for review in reviews_data:
            reviewer_email = review.get("reviewer_email", "")
            change_owner = review.get("change_owner", "")
            
            if not reviewer_email:
                continue
            
            reviewer_dev = developers[reviewer_email]
            
            # 基本情報
            reviewer_dev["email"] = reviewer_email
            reviewer_dev["name"] = review.get("reviewer_name", "")
            
            # レビュー統計
            reviewer_dev["reviews_given"] += 1
            
            # レビュースコア
            code_review_score = review.get("review_score_code_review", 0)
            if code_review_score != 0:
                reviewer_dev["review_scores_given"][code_review_score] += 1
            
            # 協力関係
            if change_owner and change_owner != reviewer_email:
                reviewer_dev["reviewed_for"].add(change_owner)
                
                # 変更作成者側の統計も更新
                if change_owner in developers:
                    owner_dev = developers[change_owner]
                    owner_dev["reviews_received"] += 1
                    owner_dev["reviewed_by"].add(reviewer_email)
                    
                    if code_review_score != 0:
                        owner_dev["review_scores_received"][code_review_score] += 1
        
        # 平均スコアを計算
        for dev_email, dev_data in developers.items():
            # レビュー与えた平均スコア
            given_scores = dev_data["review_scores_given"]
            if given_scores:
                total_score = sum(score * count for score, count in given_scores.items())
                total_count = sum(given_scores.values())
                dev_data["avg_review_score_given"] = total_score / total_count
            
            # レビュー受けた平均スコア
            received_scores = dev_data["review_scores_received"]
            if received_scores:
                total_score = sum(score * count for score, count in received_scores.items())
                total_count = sum(received_scores.values())
                dev_data["avg_review_score_received"] = total_score / total_count
            
            # セットを協力者数に変換
            dev_data["collaboration_count"] = len(dev_data["reviewed_for"]) + len(dev_data["reviewed_by"])
        
        logger.info(f"{len(developers)}人の開発者統計を計算しました")
        return dict(developers)
    
    def _clean_developer_data(self, developers: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        開発者データをクリーンアップ（JSON化可能にする）
        
        Args:
            developers: 開発者データ
            
        Returns:
            Dict[str, Dict[str, Any]]: クリーンアップされた開発者データ
        """
        cleaned_developers = {}
        
        for email, dev_data in developers.items():
            cleaned_dev = {}
            
            for key, value in dev_data.items():
                if isinstance(value, set):
                    # セットをリストに変換
                    cleaned_dev[key] = list(value)
                elif isinstance(value, defaultdict):
                    # defaultdictを通常の辞書に変換
                    cleaned_dev[key] = dict(value)
                elif isinstance(value, datetime):
                    # datetimeを文字列に変換
                    cleaned_dev[key] = value.isoformat()
                else:
                    cleaned_dev[key] = value
            
            # 統計情報を追加
            cleaned_dev["total_activity_days"] = len(dev_data["activity_days"])
            cleaned_dev["unique_files_changed"] = len(dev_data["files_changed"])
            cleaned_dev["unique_projects"] = len(dev_data["projects_contributed"])
            cleaned_dev["top_file_extensions"] = dict(sorted(
                dev_data["file_extensions"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
            cleaned_dev["top_directories"] = dict(sorted(
                dev_data["directories"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
            
            cleaned_developers[email] = cleaned_dev
        
        return cleaned_developers
    
    def extract_developers_from_files(
        self,
        changes_file_path: Path,
        reviews_file_path: Optional[Path] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        既存ファイルから開発者データを抽出
        
        Args:
            changes_file_path: 変更データファイルのパス
            reviews_file_path: レビューデータファイルのパス（オプション）
            
        Returns:
            Dict[str, Dict[str, Any]]: 開発者データ
        """
        logger.info(f"ファイルから開発者データを抽出中: {changes_file_path}")
        
        # 変更データを読み込み
        with open(changes_file_path, 'r', encoding='utf-8') as f:
            changes_data_file = json.load(f)
        
        # 変更データを平坦化
        all_changes = []
        for project, changes in changes_data_file.get("data", {}).items():
            all_changes.extend(changes)
        
        # レビューデータを読み込み
        all_reviews = []
        if reviews_file_path and reviews_file_path.exists():
            logger.info(f"レビューデータを読み込み中: {reviews_file_path}")
            with open(reviews_file_path, 'r', encoding='utf-8') as f:
                reviews_data_file = json.load(f)
            all_reviews = reviews_data_file.get("data", [])
        
        # 開発者統計を計算
        developers = self._calculate_developer_stats(all_changes, all_reviews)
        
        # データをクリーンアップ
        cleaned_developers = self._clean_developer_data(developers)
        
        return cleaned_developers
    
    def save_developers(
        self,
        developers_data: Dict[str, Dict[str, Any]],
        filename_prefix: str = "developers"
    ) -> Path:
        """
        開発者データを保存
        
        Args:
            developers_data: 開発者データ
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        logger.info(f"開発者データを保存中: {output_path}")
        
        # 統計情報を計算
        total_developers = len(developers_data)
        active_developers = sum(1 for dev in developers_data.values() if dev.get("changes_authored", 0) > 0)
        top_contributors = sorted(
            developers_data.items(),
            key=lambda x: x[1].get("changes_authored", 0),
            reverse=True
        )[:10]
        
        # メタデータを追加
        output_data = {
            "metadata": {
                "extraction_timestamp": timestamp,
                "total_developers": total_developers,
                "active_developers": active_developers,
                "top_contributors": [
                    {
                        "email": email,
                        "name": dev.get("name", ""),
                        "changes_authored": dev.get("changes_authored", 0),
                        "reviews_given": dev.get("reviews_given", 0)
                    }
                    for email, dev in top_contributors
                ]
            },
            "data": developers_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"開発者データを保存しました: {output_path}")
        return output_path
    
    def extract_and_save(
        self,
        changes_file_path: Path,
        reviews_file_path: Optional[Path] = None,
        filename_prefix: str = "developers"
    ) -> Path:
        """
        開発者データを抽出して保存
        
        Args:
            changes_file_path: 変更データファイルのパス
            reviews_file_path: レビューデータファイルのパス（オプション）
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        try:
            # 開発者データを抽出
            developers_data = self.extract_developers_from_files(
                changes_file_path=changes_file_path,
                reviews_file_path=reviews_file_path
            )
            
            # データを保存
            output_path = self.save_developers(developers_data, filename_prefix)
            
            return output_path
            
        finally:
            # クライアントを終了
            if hasattr(self, 'gerrit_client'):
                self.gerrit_client.close()


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerrit Developers抽出スクリプト")
    parser.add_argument(
        "--changes-file",
        type=Path,
        required=True,
        help="変更データファイルのパス"
    )
    parser.add_argument(
        "--reviews-file",
        type=Path,
        help="レビューデータファイルのパス（オプション）"
    )
    parser.add_argument(
        "--output-prefix",
        default="developers",
        help="出力ファイル名のプレフィックス"
    )
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not args.changes_file.exists():
        logger.error(f"変更データファイルが見つかりません: {args.changes_file}")
        return 1
    
    if args.reviews_file and not args.reviews_file.exists():
        logger.error(f"レビューデータファイルが見つかりません: {args.reviews_file}")
        return 1
    
    # 抽出器を作成
    extractor = DevelopersExtractor()
    
    try:
        # 抽出を実行
        output_path = extractor.extract_and_save(
            changes_file_path=args.changes_file,
            reviews_file_path=args.reviews_file,
            filename_prefix=args.output_prefix
        )
        
        logger.info(f"Developers抽出が完了しました: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Developers抽出中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())