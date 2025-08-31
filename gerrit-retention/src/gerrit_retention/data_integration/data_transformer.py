"""
データ変換器

Gerritから抽出した生データを機械学習に適した形式に変換します。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """データ変換器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        データ変換器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        logger.info("データ変換器を初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        from omegaconf import OmegaConf
        config = self.config_manager.get_config()
        return OmegaConf.to_container(config, resolve=True) if config else {}
    
    def load_raw_data(
        self,
        changes_file: Path,
        reviews_file: Optional[Path] = None,
        developers_file: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        生データファイルを読み込み
        
        Args:
            changes_file: 変更データファイル
            reviews_file: レビューデータファイル（オプション）
            developers_file: 開発者データファイル（オプション）
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 変更、レビュー、開発者のDataFrame
        """
        logger.info("生データファイルを読み込み中...")
        
        # 変更データを読み込み
        with open(changes_file, 'r', encoding='utf-8') as f:
            changes_data = json.load(f)
        
        # 変更データを平坦化してDataFrameに変換
        all_changes = []
        for project, changes in changes_data.get("data", {}).items():
            for change in changes:
                change["project"] = project  # プロジェクト情報を確実に設定
                all_changes.append(change)
        
        changes_df = pd.DataFrame(all_changes)
        logger.info(f"変更データを読み込み: {len(changes_df)}件")
        
        # レビューデータを読み込み
        reviews_df = pd.DataFrame()
        if reviews_file and reviews_file.exists():
            with open(reviews_file, 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
            reviews_df = pd.DataFrame(reviews_data.get("data", []))
            logger.info(f"レビューデータを読み込み: {len(reviews_df)}件")
        
        # 開発者データを読み込み
        developers_df = pd.DataFrame()
        if developers_file and developers_file.exists():
            with open(developers_file, 'r', encoding='utf-8') as f:
                developers_data = json.load(f)
            
            # 開発者データを平坦化
            dev_records = []
            for email, dev_info in developers_data.get("data", {}).items():
                dev_info["email"] = email  # メールアドレスを確実に設定
                dev_records.append(dev_info)
            
            developers_df = pd.DataFrame(dev_records)
            logger.info(f"開発者データを読み込み: {len(developers_df)}件")
        
        return changes_df, reviews_df, developers_df
    
    def transform_changes_data(self, changes_df: pd.DataFrame) -> pd.DataFrame:
        """
        変更データを変換
        
        Args:
            changes_df: 変更データのDataFrame
            
        Returns:
            pd.DataFrame: 変換された変更データ
        """
        logger.info("変更データを変換中...")
        
        if changes_df.empty:
            logger.warning("変更データが空です")
            return changes_df
        
        # コピーを作成
        df = changes_df.copy()
        
        # 日時カラムを変換
        date_columns = ['created', 'updated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 数値カラムを変換
        numeric_columns = ['_number', 'insertions', 'deletions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 新しい特徴量を作成
        if 'created' in df.columns and 'updated' in df.columns:
            df['review_duration_hours'] = (
                df['updated'] - df['created']
            ).dt.total_seconds() / 3600
        
        if 'insertions' in df.columns and 'deletions' in df.columns:
            df['total_lines_changed'] = df['insertions'] + df['deletions']
            df['change_ratio'] = df['insertions'] / (df['insertions'] + df['deletions'] + 1)
        
        # ステータスをカテゴリ化
        if 'status' in df.columns:
            df['status_category'] = df['status'].map({
                'NEW': 'open',
                'MERGED': 'merged',
                'ABANDONED': 'abandoned'
            }).fillna('other')
        
        # プロジェクト情報を正規化
        if 'project' in df.columns:
            df['project_normalized'] = df['project'].str.lower().str.replace('[^a-z0-9]', '_', regex=True)
        
        # 作成者情報を抽出
        if 'owner' in df.columns:
            df['owner_email'] = df['owner'].apply(
                lambda x: x.get('email', '') if isinstance(x, dict) else ''
            )
            df['owner_name'] = df['owner'].apply(
                lambda x: x.get('name', '') if isinstance(x, dict) else ''
            )
        
        logger.info(f"変更データの変換完了: {len(df)}件")
        return df
    
    def transform_reviews_data(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        レビューデータを変換
        
        Args:
            reviews_df: レビューデータのDataFrame
            
        Returns:
            pd.DataFrame: 変換されたレビューデータ
        """
        logger.info("レビューデータを変換中...")
        
        if reviews_df.empty:
            logger.warning("レビューデータが空です")
            return reviews_df
        
        # コピーを作成
        df = reviews_df.copy()
        
        # 日時カラムを変換
        date_columns = ['review_timestamp', 'change_created', 'change_updated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 数値カラムを変換
        numeric_columns = [
            'change_number', 'review_score_code_review', 'review_score_verified',
            'files_count', 'lines_added', 'lines_deleted', 'lines_total'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # レビュー応答時間を計算
        if 'change_created' in df.columns and 'review_timestamp' in df.columns:
            df['review_response_time_hours'] = (
                df['review_timestamp'] - df['change_created']
            ).dt.total_seconds() / 3600
        
        # レビュースコアのカテゴリ化
        if 'review_score_code_review' in df.columns:
            df['review_score_category'] = df['review_score_code_review'].map({
                -2: 'strongly_negative',
                -1: 'negative',
                0: 'neutral',
                1: 'positive',
                2: 'strongly_positive'
            }).fillna('neutral')
        
        # レビューの種類を分類
        df['review_type'] = 'comment'
        if 'review_score_code_review' in df.columns:
            df.loc[df['review_score_code_review'] != 0, 'review_type'] = 'score'
        
        # レビューメッセージの長さ
        if 'review_message' in df.columns:
            df['review_message_length'] = df['review_message'].str.len().fillna(0)
            df['has_detailed_comment'] = df['review_message_length'] > 50
        
        logger.info(f"レビューデータの変換完了: {len(df)}件")
        return df
    
    def transform_developers_data(self, developers_df: pd.DataFrame) -> pd.DataFrame:
        """
        開発者データを変換
        
        Args:
            developers_df: 開発者データのDataFrame
            
        Returns:
            pd.DataFrame: 変換された開発者データ
        """
        logger.info("開発者データを変換中...")
        
        if developers_df.empty:
            logger.warning("開発者データが空です")
            return developers_df
        
        # コピーを作成
        df = developers_df.copy()
        
        # 日時カラムを変換
        date_columns = ['first_activity', 'last_activity']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 数値カラムを変換
        numeric_columns = [
            'changes_authored', 'changes_merged', 'changes_abandoned',
            'lines_added', 'lines_deleted', 'reviews_given', 'reviews_received',
            'total_activity_days', 'unique_files_changed', 'unique_projects',
            'collaboration_count'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 活動期間を計算
        if 'first_activity' in df.columns and 'last_activity' in df.columns:
            df['activity_duration_days'] = (
                df['last_activity'] - df['first_activity']
            ).dt.days
        
        # 活動レベルを分類
        if 'changes_authored' in df.columns:
            df['activity_level'] = pd.cut(
                df['changes_authored'],
                bins=[0, 1, 5, 20, float('inf')],
                labels=['inactive', 'low', 'medium', 'high'],
                include_lowest=True
            )
        
        # 専門性レベルを計算
        if 'unique_projects' in df.columns and 'unique_files_changed' in df.columns:
            df['expertise_breadth'] = df['unique_projects'] * df['unique_files_changed']
        
        # レビュー参加率を計算
        if 'reviews_given' in df.columns and 'changes_authored' in df.columns:
            df['review_participation_ratio'] = (
                df['reviews_given'] / (df['changes_authored'] + 1)
            )
        
        # 協力度を計算
        if 'collaboration_count' in df.columns and 'changes_authored' in df.columns:
            df['collaboration_ratio'] = (
                df['collaboration_count'] / (df['changes_authored'] + 1)
            )
        
        logger.info(f"開発者データの変換完了: {len(df)}件")
        return df
    
    def create_developer_change_features(
        self,
        changes_df: pd.DataFrame,
        developers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        開発者と変更を結合した特徴量を作成
        
        Args:
            changes_df: 変更データ
            developers_df: 開発者データ
            
        Returns:
            pd.DataFrame: 結合された特徴量データ
        """
        logger.info("開発者-変更特徴量を作成中...")
        
        if changes_df.empty or developers_df.empty:
            logger.warning("データが空のため、特徴量作成をスキップします")
            return pd.DataFrame()
        
        # 変更データに開発者情報を結合
        merged_df = changes_df.merge(
            developers_df,
            left_on='owner_email',
            right_on='email',
            how='left',
            suffixes=('_change', '_developer')
        )
        
        # 新しい特徴量を作成
        feature_columns = [
            'change_id', 'project', 'owner_email', 'status',
            'total_lines_changed', 'review_duration_hours',
            'changes_authored', 'activity_level', 'expertise_breadth',
            'review_participation_ratio', 'collaboration_ratio'
        ]
        
        # 存在するカラムのみを選択
        available_columns = [col for col in feature_columns if col in merged_df.columns]
        result_df = merged_df[available_columns].copy()
        
        logger.info(f"開発者-変更特徴量を作成: {len(result_df)}件")
        return result_df
    
    def save_transformed_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        output_path: Path,
        format: str = 'parquet'
    ) -> None:
        """
        変換されたデータを保存
        
        Args:
            data: 保存するデータ
            output_path: 出力パス
            format: 保存形式（'parquet', 'csv', 'json'）
        """
        logger.info(f"変換データを保存中: {output_path}")
        
        # 出力ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict):
            # 複数のDataFrameを保存
            for name, df in data.items():
                file_path = output_path.parent / f"{output_path.stem}_{name}.{format}"
                self._save_single_dataframe(df, file_path, format)
        else:
            # 単一のDataFrameを保存
            self._save_single_dataframe(data, output_path, format)
        
        logger.info(f"変換データの保存完了: {output_path}")
    
    def _save_single_dataframe(
        self,
        df: pd.DataFrame,
        file_path: Path,
        format: str
    ) -> None:
        """
        単一のDataFrameを保存
        
        Args:
            df: 保存するDataFrame
            file_path: ファイルパス
            format: 保存形式
        """
        if format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format == 'csv':
            df.to_csv(file_path, index=False, encoding='utf-8')
        elif format == 'json':
            df.to_json(file_path, orient='records', indent=2, force_ascii=False)
        else:
            raise ValueError(f"サポートされていない形式: {format}")
    
    def transform_all_data(
        self,
        changes_file: Path,
        reviews_file: Optional[Path] = None,
        developers_file: Optional[Path] = None,
        output_dir: Path = Path("data/processed")
    ) -> Dict[str, Path]:
        """
        すべてのデータを変換して保存
        
        Args:
            changes_file: 変更データファイル
            reviews_file: レビューデータファイル
            developers_file: 開発者データファイル
            output_dir: 出力ディレクトリ
            
        Returns:
            Dict[str, Path]: 保存されたファイルのパス
        """
        logger.info("全データの変換を開始...")
        
        # 生データを読み込み
        changes_df, reviews_df, developers_df = self.load_raw_data(
            changes_file, reviews_file, developers_file
        )
        
        # データを変換
        transformed_changes = self.transform_changes_data(changes_df)
        transformed_reviews = self.transform_reviews_data(reviews_df)
        transformed_developers = self.transform_developers_data(developers_df)
        
        # 結合特徴量を作成
        developer_change_features = self.create_developer_change_features(
            transformed_changes, transformed_developers
        )
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_paths = {}
        
        if not transformed_changes.empty:
            path = output_dir / f"changes_transformed_{timestamp}.parquet"
            self.save_transformed_data(transformed_changes, path)
            output_paths['changes'] = path
        
        if not transformed_reviews.empty:
            path = output_dir / f"reviews_transformed_{timestamp}.parquet"
            self.save_transformed_data(transformed_reviews, path)
            output_paths['reviews'] = path
        
        if not transformed_developers.empty:
            path = output_dir / f"developers_transformed_{timestamp}.parquet"
            self.save_transformed_data(transformed_developers, path)
            output_paths['developers'] = path
        
        if not developer_change_features.empty:
            path = output_dir / f"developer_change_features_{timestamp}.parquet"
            self.save_transformed_data(developer_change_features, path)
            output_paths['features'] = path
        
        logger.info("全データの変換が完了しました")
        return output_paths


# ユーティリティ関数

def create_data_transformer(config: Optional[Dict[str, Any]] = None) -> DataTransformer:
    """
    データ変換器を作成
    
    Args:
        config: 設定辞書（オプション）
        
    Returns:
        DataTransformer: データ変換器インスタンス
    """
    return DataTransformer(config)