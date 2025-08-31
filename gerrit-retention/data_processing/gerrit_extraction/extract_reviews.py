#!/usr/bin/env python3
"""
Gerrit Reviews（レビュー）抽出スクリプト

Gerrit APIからReviews（レビュー）データを抽出し、構造化されたデータとして保存します。
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


class ReviewsExtractor:
    """Reviews（レビュー）抽出器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Reviews抽出器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # Gerritクライアントを作成
        self.gerrit_client = create_gerrit_client()
        
        # 出力設定
        self.output_dir = Path("data/raw/gerrit_reviews")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Reviews抽出器を初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get_config().to_dict()
    
    def _extract_review_data_from_change(self, change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        変更からレビューデータを抽出
        
        Args:
            change: 変更データ
            
        Returns:
            List[Dict[str, Any]]: レビューデータのリスト
        """
        reviews = []
        
        # メッセージからレビューを抽出
        messages = change.get("messages", [])
        
        for message in messages:
            # レビュースコアがある場合のみレビューとして扱う
            if "author" in message and message.get("message", "").strip():
                review = {
                    "change_id": change.get("id", ""),
                    "change_number": change.get("_number", 0),
                    "project": change.get("project", ""),
                    "branch": change.get("branch", ""),
                    "change_subject": change.get("subject", ""),
                    "change_owner": change.get("owner", {}).get("email", ""),
                    "change_created": change.get("created", ""),
                    "change_updated": change.get("updated", ""),
                    "change_status": change.get("status", ""),
                    
                    # レビュー情報
                    "review_id": message.get("id", ""),
                    "reviewer_email": message.get("author", {}).get("email", ""),
                    "reviewer_name": message.get("author", {}).get("name", ""),
                    "review_timestamp": message.get("date", ""),
                    "review_message": message.get("message", ""),
                    
                    # レビュースコア（ラベルから抽出）
                    "review_scores": {},
                    "review_score_code_review": 0,
                    "review_score_verified": 0,
                }
                
                # ラベル情報を処理
                labels = change.get("labels", {})
                
                # Code-Reviewスコアを抽出
                if "Code-Review" in labels:
                    code_review_info = labels["Code-Review"]
                    if "all" in code_review_info:
                        for vote in code_review_info["all"]:
                            if (vote.get("email") == review["reviewer_email"] and 
                                "value" in vote):
                                review["review_score_code_review"] = vote["value"]
                                review["review_scores"]["Code-Review"] = vote["value"]
                
                # Verifiedスコアを抽出
                if "Verified" in labels:
                    verified_info = labels["Verified"]
                    if "all" in verified_info:
                        for vote in verified_info["all"]:
                            if (vote.get("email") == review["reviewer_email"] and 
                                "value" in vote):
                                review["review_score_verified"] = vote["value"]
                                review["review_scores"]["Verified"] = vote["value"]
                
                # その他のラベルも処理
                for label_name, label_info in labels.items():
                    if label_name not in ["Code-Review", "Verified"] and "all" in label_info:
                        for vote in label_info["all"]:
                            if (vote.get("email") == review["reviewer_email"] and 
                                "value" in vote):
                                review["review_scores"][label_name] = vote["value"]
                
                # ファイル情報を追加
                revisions = change.get("revisions", {})
                if revisions:
                    # 最新のリビジョンを取得
                    latest_revision = list(revisions.values())[-1]
                    files = latest_revision.get("files", {})
                    
                    review["files_changed"] = list(files.keys())
                    review["files_count"] = len(files)
                    
                    # 変更行数を計算
                    lines_added = 0
                    lines_deleted = 0
                    
                    for file_info in files.values():
                        lines_added += file_info.get("lines_inserted", 0)
                        lines_deleted += file_info.get("lines_deleted", 0)
                    
                    review["lines_added"] = lines_added
                    review["lines_deleted"] = lines_deleted
                    review["lines_total"] = lines_added + lines_deleted
                
                reviews.append(review)
        
        return reviews
    
    def extract_reviews_from_changes_file(self, changes_file_path: Path) -> List[Dict[str, Any]]:
        """
        変更ファイルからレビューを抽出
        
        Args:
            changes_file_path: 変更データファイルのパス
            
        Returns:
            List[Dict[str, Any]]: レビューデータのリスト
        """
        logger.info(f"変更ファイルからレビューを抽出中: {changes_file_path}")
        
        # 変更データを読み込み
        with open(changes_file_path, 'r', encoding='utf-8') as f:
            changes_data = json.load(f)
        
        all_reviews = []
        
        # プロジェクト別に処理
        for project, changes in changes_data.get("data", {}).items():
            logger.info(f"プロジェクト '{project}' のレビューを抽出中...")
            
            project_reviews = []
            
            for change in changes:
                try:
                    reviews = self._extract_review_data_from_change(change)
                    project_reviews.extend(reviews)
                    
                except Exception as e:
                    logger.warning(f"変更 '{change.get('id', 'unknown')}' のレビュー抽出に失敗: {e}")
            
            logger.info(f"プロジェクト '{project}' から {len(project_reviews)}個のレビューを抽出")
            all_reviews.extend(project_reviews)
        
        logger.info(f"合計 {len(all_reviews)}個のレビューを抽出しました")
        return all_reviews
    
    def extract_reviews_for_project(
        self,
        project: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        プロジェクトのレビューを直接抽出
        
        Args:
            project: プロジェクト名
            start_date: 開始日時
            end_date: 終了日時
            
        Returns:
            List[Dict[str, Any]]: レビューデータのリスト
        """
        logger.info(f"プロジェクト '{project}' のレビューを直接抽出中...")
        
        # まず変更一覧を取得
        changes = self.gerrit_client.get_changes_batch(
            project=project,
            start_date=start_date,
            end_date=end_date
        )
        
        all_reviews = []
        
        # 各変更の詳細を取得してレビューを抽出
        for i, change in enumerate(changes):
            try:
                # 詳細情報を取得
                change_detail = self.gerrit_client.get_change_detail(change["id"])
                
                # レビューを抽出
                reviews = self._extract_review_data_from_change(change_detail)
                all_reviews.extend(reviews)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"レビュー抽出進捗: {i + 1}/{len(changes)} (累計レビュー: {len(all_reviews)})")
                    
            except Exception as e:
                logger.warning(f"変更 '{change['id']}' のレビュー抽出に失敗: {e}")
        
        logger.info(f"プロジェクト '{project}' から {len(all_reviews)}個のレビューを抽出しました")
        return all_reviews
    
    def save_reviews(
        self,
        reviews_data: List[Dict[str, Any]],
        filename_prefix: str = "reviews"
    ) -> Path:
        """
        レビューデータを保存
        
        Args:
            reviews_data: レビューデータ
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        logger.info(f"レビューデータを保存中: {output_path}")
        
        # プロジェクト別統計を計算
        project_stats = {}
        reviewer_stats = {}
        
        for review in reviews_data:
            project = review.get("project", "unknown")
            reviewer = review.get("reviewer_email", "unknown")
            
            # プロジェクト統計
            if project not in project_stats:
                project_stats[project] = 0
            project_stats[project] += 1
            
            # レビュワー統計
            if reviewer not in reviewer_stats:
                reviewer_stats[reviewer] = 0
            reviewer_stats[reviewer] += 1
        
        # メタデータを追加
        output_data = {
            "metadata": {
                "extraction_timestamp": timestamp,
                "total_reviews": len(reviews_data),
                "unique_projects": len(project_stats),
                "unique_reviewers": len(reviewer_stats),
                "project_stats": project_stats,
                "top_reviewers": dict(sorted(reviewer_stats.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "data": reviews_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"レビューデータを保存しました: {output_path}")
        return output_path
    
    def extract_and_save(
        self,
        changes_file_path: Optional[Path] = None,
        projects: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filename_prefix: str = "reviews"
    ) -> Path:
        """
        レビューデータを抽出して保存
        
        Args:
            changes_file_path: 変更データファイルのパス（指定時は既存ファイルから抽出）
            projects: プロジェクト一覧（直接抽出時）
            start_date: 開始日時（直接抽出時）
            end_date: 終了日時（直接抽出時）
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            Path: 保存されたファイルのパス
        """
        try:
            if changes_file_path:
                # 既存の変更ファイルからレビューを抽出
                reviews_data = self.extract_reviews_from_changes_file(changes_file_path)
            else:
                # 直接抽出
                reviews_data = []
                
                if not projects:
                    projects = self.config_manager.get("gerrit.projects", [])
                
                for project in projects:
                    project_reviews = self.extract_reviews_for_project(
                        project=project,
                        start_date=start_date,
                        end_date=end_date
                    )
                    reviews_data.extend(project_reviews)
            
            # データを保存
            output_path = self.save_reviews(reviews_data, filename_prefix)
            
            return output_path
            
        finally:
            # クライアントを終了
            if hasattr(self, 'gerrit_client'):
                self.gerrit_client.close()


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerrit Reviews抽出スクリプト")
    parser.add_argument(
        "--changes-file",
        type=Path,
        help="変更データファイルのパス（指定時は既存ファイルから抽出）"
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        help="抽出するプロジェクト名（直接抽出時）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="開始日（YYYY-MM-DD形式、直接抽出時）"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="終了日（YYYY-MM-DD形式、直接抽出時）"
    )
    parser.add_argument(
        "--output-prefix",
        default="reviews",
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
    extractor = ReviewsExtractor()
    
    try:
        # 抽出を実行
        output_path = extractor.extract_and_save(
            changes_file_path=args.changes_file,
            projects=args.projects,
            start_date=start_date,
            end_date=end_date,
            filename_prefix=args.output_prefix
        )
        
        logger.info(f"Reviews抽出が完了しました: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Reviews抽出中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())