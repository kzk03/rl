#!/usr/bin/env python3
"""
データクリーニングスクリプト

変換されたデータの品質を向上させるためのクリーニング処理を行います。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger, setup_logging

# ログ設定を初期化
setup_logging(level="INFO")
logger = get_logger(__name__)


class DataCleaner:
    """データクリーニング器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        データクリーニング器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # クリーニング設定
        self.min_activity_threshold = self.config.get("data_cleaning", {}).get("min_activity_threshold", 1)
        self.max_outlier_std = self.config.get("data_cleaning", {}).get("max_outlier_std", 3)
        self.bot_patterns = self.config.get("data_cleaning", {}).get("bot_patterns", [
            "bot", "automation", "jenkins", "ci", "build", "deploy"
        ])
        
        logger.info("データクリーニング器を初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get_config().to_dict()
    
    def detect_bot_accounts(self, df: pd.DataFrame, email_column: str = 'email') -> pd.Series:
        """
        ボットアカウントを検出
        
        Args:
            df: データフレーム
            email_column: メールアドレスのカラム名
            
        Returns:
            pd.Series: ボットアカウントのブール値マスク
        """
        logger.info("ボットアカウントを検出中...")
        
        if email_column not in df.columns:
            logger.warning(f"カラム '{email_column}' が見つかりません")
            return pd.Series([False] * len(df), index=df.index)
        
        # ボットパターンに基づく検出
        bot_mask = pd.Series([False] * len(df), index=df.index)
        
        for pattern in self.bot_patterns:
            pattern_mask = df[email_column].str.contains(
                pattern, case=False, na=False, regex=False
            )
            bot_mask |= pattern_mask
            
            if pattern_mask.any():
                bot_count = pattern_mask.sum()
                logger.info(f"パターン '{pattern}' で {bot_count}個のボットアカウントを検出")
        
        # 追加の検出ルール
        # 1. 非常に高い活動量（統計的外れ値）
        if 'changes_authored' in df.columns:
            changes_mean = df['changes_authored'].mean()
            changes_std = df['changes_authored'].std()
            high_activity_threshold = changes_mean + (self.max_outlier_std * changes_std)
            
            high_activity_mask = df['changes_authored'] > high_activity_threshold
            bot_mask |= high_activity_mask
            
            if high_activity_mask.any():
                high_activity_count = high_activity_mask.sum()
                logger.info(f"高活動量で {high_activity_count}個の潜在的ボットを検出")
        
        # 2. 名前がメールアドレスと同じ（自動生成アカウント）
        if 'name' in df.columns:
            auto_name_mask = df['name'] == df[email_column]
            bot_mask |= auto_name_mask
            
            if auto_name_mask.any():
                auto_name_count = auto_name_mask.sum()
                logger.info(f"自動生成名で {auto_name_count}個の潜在的ボットを検出")
        
        total_bots = bot_mask.sum()
        logger.info(f"合計 {total_bots}個のボットアカウントを検出しました")
        
        return bot_mask
    
    def remove_inactive_users(
        self,
        df: pd.DataFrame,
        activity_column: str = 'changes_authored'
    ) -> pd.DataFrame:
        """
        非活動ユーザーを除去
        
        Args:
            df: データフレーム
            activity_column: 活動量を示すカラム名
            
        Returns:
            pd.DataFrame: フィルタリングされたデータフレーム
        """
        logger.info("非活動ユーザーを除去中...")
        
        if activity_column not in df.columns:
            logger.warning(f"カラム '{activity_column}' が見つかりません")
            return df
        
        initial_count = len(df)
        
        # 最小活動量以上のユーザーのみを保持
        active_mask = df[activity_column] >= self.min_activity_threshold
        filtered_df = df[active_mask].copy()
        
        removed_count = initial_count - len(filtered_df)
        logger.info(f"{removed_count}個の非活動ユーザーを除去しました")
        
        return filtered_df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr'
    ) -> pd.DataFrame:
        """
        外れ値を検出
        
        Args:
            df: データフレーム
            columns: 外れ値検出対象のカラム
            method: 検出方法（'iqr', 'zscore'）
            
        Returns:
            pd.DataFrame: 外れ値情報を含むデータフレーム
        """
        logger.info(f"外れ値を検出中（方法: {method}）...")
        
        result_df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                logger.warning(f"カラム '{column}' が見つかりません")
                continue
            
            if method == 'iqr':
                # IQR法
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif method == 'zscore':
                # Z-score法
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outlier_mask = z_scores > self.max_outlier_std
                
            else:
                logger.error(f"サポートされていない外れ値検出方法: {method}")
                continue
            
            outlier_column = f'{column}_outlier'
            result_df[outlier_column] = outlier_mask
            
            outlier_count = outlier_mask.sum()
            logger.info(f"カラム '{column}' で {outlier_count}個の外れ値を検出")
        
        return result_df
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        欠損値を処理
        
        Args:
            df: データフレーム
            strategy: カラム別の処理戦略
            
        Returns:
            pd.DataFrame: 欠損値処理済みデータフレーム
        """
        logger.info("欠損値を処理中...")
        
        if strategy is None:
            strategy = {
                'numeric': 'median',
                'categorical': 'mode',
                'datetime': 'drop'
            }
        
        result_df = df.copy()
        
        # 欠損値の統計を表示
        missing_stats = result_df.isnull().sum()
        missing_columns = missing_stats[missing_stats > 0]
        
        if len(missing_columns) > 0:
            logger.info("欠損値の統計:")
            for column, count in missing_columns.items():
                percentage = (count / len(result_df)) * 100
                logger.info(f"  {column}: {count}個 ({percentage:.2f}%)")
        
        # カラム別に欠損値を処理
        for column in result_df.columns:
            if result_df[column].isnull().sum() == 0:
                continue
            
            dtype = result_df[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                # 数値型の処理
                if strategy.get('numeric') == 'mean':
                    fill_value = result_df[column].mean()
                elif strategy.get('numeric') == 'median':
                    fill_value = result_df[column].median()
                elif strategy.get('numeric') == 'zero':
                    fill_value = 0
                else:
                    continue
                
                result_df[column] = result_df[column].fillna(fill_value)
                
            elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
                # カテゴリ型・文字列型の処理
                if strategy.get('categorical') == 'mode':
                    mode_value = result_df[column].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        result_df[column] = result_df[column].fillna(fill_value)
                elif strategy.get('categorical') == 'unknown':
                    result_df[column] = result_df[column].fillna('unknown')
                
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                # 日時型の処理
                if strategy.get('datetime') == 'drop':
                    # 日時の欠損値がある行を削除
                    result_df = result_df.dropna(subset=[column])
        
        logger.info("欠損値処理が完了しました")
        return result_df
    
    def normalize_text_fields(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        テキストフィールドを正規化
        
        Args:
            df: データフレーム
            text_columns: 正規化対象のテキストカラム
            
        Returns:
            pd.DataFrame: 正規化済みデータフレーム
        """
        logger.info("テキストフィールドを正規化中...")
        
        result_df = df.copy()
        
        for column in text_columns:
            if column not in result_df.columns:
                logger.warning(f"カラム '{column}' が見つかりません")
                continue
            
            # 文字列型に変換
            result_df[column] = result_df[column].astype(str)
            
            # 基本的な正規化
            result_df[column] = (
                result_df[column]
                .str.strip()  # 前後の空白を除去
                .str.lower()  # 小文字に変換
                .replace('nan', '')  # 'nan'文字列を空文字に変換
                .replace('none', '')  # 'none'文字列を空文字に変換
            )
            
            logger.info(f"カラム '{column}' を正規化しました")
        
        return result_df
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データの整合性を検証
        
        Args:
            df: データフレーム
            
        Returns:
            Dict[str, Any]: 検証結果
        """
        logger.info("データの整合性を検証中...")
        
        validation_results = {
            'total_records': len(df),
            'duplicate_records': 0,
            'invalid_dates': 0,
            'negative_values': {},
            'consistency_issues': []
        }
        
        # 重複レコードをチェック
        if len(df) > 0:
            duplicates = df.duplicated().sum()
            validation_results['duplicate_records'] = duplicates
            if duplicates > 0:
                logger.warning(f"{duplicates}個の重複レコードを検出")
        
        # 日付の整合性をチェック
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for column in date_columns:
            invalid_dates = df[column].isnull().sum()
            validation_results['invalid_dates'] += invalid_dates
            
            # 未来の日付をチェック
            future_dates = (df[column] > datetime.now()).sum()
            if future_dates > 0:
                validation_results['consistency_issues'].append(
                    f"カラム '{column}' に {future_dates}個の未来の日付"
                )
        
        # 負の値をチェック（通常は正の値であるべきカラム）
        positive_columns = [
            'changes_authored', 'reviews_given', 'lines_added', 'lines_deleted'
        ]
        for column in positive_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    validation_results['negative_values'][column] = negative_count
                    logger.warning(f"カラム '{column}' に {negative_count}個の負の値")
        
        # 論理的整合性をチェック
        if 'first_activity' in df.columns and 'last_activity' in df.columns:
            invalid_activity_order = (
                df['first_activity'] > df['last_activity']
            ).sum()
            if invalid_activity_order > 0:
                validation_results['consistency_issues'].append(
                    f"{invalid_activity_order}個のレコードで最初の活動日が最後の活動日より後"
                )
        
        logger.info("データ整合性検証が完了しました")
        return validation_results
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        remove_bots: bool = True,
        remove_inactive: bool = True,
        handle_missing: bool = True,
        detect_outliers_flag: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        データフレームを包括的にクリーニング
        
        Args:
            df: クリーニング対象のデータフレーム
            remove_bots: ボットアカウントを除去するか
            remove_inactive: 非活動ユーザーを除去するか
            handle_missing: 欠損値を処理するか
            detect_outliers_flag: 外れ値を検出するか
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: クリーニング済みデータフレームと統計情報
        """
        logger.info("データフレームの包括的クリーニングを開始...")
        
        initial_count = len(df)
        result_df = df.copy()
        cleaning_stats = {
            'initial_records': initial_count,
            'bots_removed': 0,
            'inactive_removed': 0,
            'final_records': 0
        }
        
        # ボットアカウントを除去
        if remove_bots and 'email' in result_df.columns:
            bot_mask = self.detect_bot_accounts(result_df, 'email')
            bots_count = bot_mask.sum()
            result_df = result_df[~bot_mask]
            cleaning_stats['bots_removed'] = bots_count
            logger.info(f"{bots_count}個のボットアカウントを除去")
        
        # 非活動ユーザーを除去
        if remove_inactive:
            before_count = len(result_df)
            result_df = self.remove_inactive_users(result_df)
            inactive_removed = before_count - len(result_df)
            cleaning_stats['inactive_removed'] = inactive_removed
        
        # 欠損値を処理
        if handle_missing:
            result_df = self.handle_missing_values(result_df)
        
        # 外れ値を検出（除去はしない）
        if detect_outliers_flag:
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                result_df = self.detect_outliers(result_df, numeric_columns[:5])  # 最初の5個のみ
        
        # テキストフィールドを正規化
        text_columns = ['email', 'name', 'project']
        available_text_columns = [col for col in text_columns if col in result_df.columns]
        if available_text_columns:
            result_df = self.normalize_text_fields(result_df, available_text_columns)
        
        # データ整合性を検証
        validation_results = self.validate_data_consistency(result_df)
        cleaning_stats.update(validation_results)
        
        cleaning_stats['final_records'] = len(result_df)
        cleaning_stats['records_removed'] = initial_count - len(result_df)
        cleaning_stats['removal_percentage'] = (
            cleaning_stats['records_removed'] / initial_count * 100
        )
        
        logger.info(f"クリーニング完了: {initial_count} → {len(result_df)}件 "
                   f"({cleaning_stats['removal_percentage']:.2f}%削減)")
        
        return result_df, cleaning_stats


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="データクリーニングスクリプト")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="入力データファイル（Parquet形式）"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="出力データファイル（指定しない場合は自動生成）"
    )
    parser.add_argument(
        "--no-remove-bots",
        action="store_true",
        help="ボットアカウントを除去しない"
    )
    parser.add_argument(
        "--no-remove-inactive",
        action="store_true",
        help="非活動ユーザーを除去しない"
    )
    parser.add_argument(
        "--no-handle-missing",
        action="store_true",
        help="欠損値を処理しない"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not args.input_file.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input_file}")
        return 1
    
    # 出力ファイルパスを決定
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.input_file.parent / f"{args.input_file.stem}_cleaned_{timestamp}.parquet"
    
    try:
        # データを読み込み
        logger.info(f"データを読み込み中: {args.input_file}")
        df = pd.read_parquet(args.input_file)
        
        # クリーニング器を作成
        cleaner = DataCleaner()
        
        # クリーニングを実行
        cleaned_df, stats = cleaner.clean_dataframe(
            df,
            remove_bots=not args.no_remove_bots,
            remove_inactive=not args.no_remove_inactive,
            handle_missing=not args.no_handle_missing
        )
        
        # 結果を保存
        logger.info(f"クリーニング済みデータを保存中: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_parquet(output_file, index=False)
        
        # 統計情報を保存
        stats_file = output_file.parent / f"{output_file.stem}_stats.json"
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"データクリーニングが完了しました: {output_file}")
        logger.info(f"統計情報: {stats_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"データクリーニング中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())