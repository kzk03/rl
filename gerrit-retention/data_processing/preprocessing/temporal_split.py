#!/usr/bin/env python3
"""
時系列データ分割スクリプト

時系列データを訓練・検証・テスト用に分割し、データリークを防止します。
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


class TemporalSplitter:
    """時系列データ分割器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        時系列データ分割器を初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # 時系列分割設定
        temporal_config = self.config.get("temporal_consistency", {})
        self.train_end_date = temporal_config.get("train_end_date", "2022-12-31")
        self.test_start_date = temporal_config.get("test_start_date", "2023-01-01")
        self.enable_strict_validation = temporal_config.get("enable_strict_validation", True)
        
        # 日付を解析
        self.train_end_datetime = datetime.strptime(self.train_end_date, "%Y-%m-%d")
        self.test_start_datetime = datetime.strptime(self.test_start_date, "%Y-%m-%d")
        
        # 検証期間を計算（訓練終了から1ヶ月前）
        self.val_start_datetime = self.train_end_datetime - timedelta(days=30)
        
        logger.info("時系列データ分割器を初期化しました")
        logger.info(f"訓練期間: 〜{self.train_end_date}")
        logger.info(f"検証期間: {self.val_start_datetime.strftime('%Y-%m-%d')} 〜 {self.train_end_date}")
        logger.info(f"テスト期間: {self.test_start_date}〜")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get_config().to_dict()
    
    def validate_temporal_consistency(
        self,
        df: pd.DataFrame,
        date_column: str = 'created'
    ) -> Dict[str, Any]:
        """
        時系列データの整合性を検証
        
        Args:
            df: データフレーム
            date_column: 日付カラム名
            
        Returns:
            Dict[str, Any]: 検証結果
        """
        logger.info("時系列データの整合性を検証中...")
        
        validation_results = {
            'total_records': len(df),
            'date_column': date_column,
            'date_range': {},
            'issues': [],
            'is_valid': True
        }
        
        if date_column not in df.columns:
            validation_results['issues'].append(f"日付カラム '{date_column}' が見つかりません")
            validation_results['is_valid'] = False
            return validation_results
        
        # 日付カラムを datetime に変換
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # 無効な日付をチェック
        invalid_dates = df_copy[date_column].isnull().sum()
        if invalid_dates > 0:
            validation_results['issues'].append(f"{invalid_dates}個の無効な日付")
            validation_results['is_valid'] = False
        
        # 日付範囲を計算
        valid_dates = df_copy[date_column].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            validation_results['date_range'] = {
                'min_date': min_date.isoformat(),
                'max_date': max_date.isoformat(),
                'span_days': (max_date - min_date).days
            }
            
            # 未来の日付をチェック
            future_dates = (valid_dates > datetime.now()).sum()
            if future_dates > 0:
                validation_results['issues'].append(f"{future_dates}個の未来の日付")
                validation_results['is_valid'] = False
            
            # 設定された分割日付との整合性をチェック
            if self.enable_strict_validation:
                # テスト期間より前のデータが訓練データに含まれているかチェック
                train_data_in_test_period = (
                    valid_dates >= self.test_start_datetime
                ).sum()
                
                if train_data_in_test_period > 0:
                    validation_results['issues'].append(
                        f"テスト期間（{self.test_start_date}以降）に{train_data_in_test_period}個の訓練データ"
                    )
                    validation_results['is_valid'] = False
        
        logger.info(f"時系列整合性検証完了: {'有効' if validation_results['is_valid'] else '無効'}")
        
        return validation_results
    
    def split_temporal_data(
        self,
        df: pd.DataFrame,
        date_column: str = 'created',
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        時系列データを分割
        
        Args:
            df: データフレーム
            date_column: 日付カラム名
            target_column: ターゲットカラム名（オプション）
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 訓練、検証、テストデータ
        """
        logger.info("時系列データを分割中...")
        
        # 日付カラムを datetime に変換
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # 無効な日付を除去
        valid_mask = df_copy[date_column].notna()
        df_clean = df_copy[valid_mask].copy()
        
        if len(df_clean) == 0:
            logger.error("有効な日付データがありません")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # 時系列順にソート
        df_clean = df_clean.sort_values(date_column)
        
        # データを分割
        train_mask = df_clean[date_column] < self.val_start_datetime
        val_mask = (
            (df_clean[date_column] >= self.val_start_datetime) &
            (df_clean[date_column] <= self.train_end_datetime)
        )
        test_mask = df_clean[date_column] >= self.test_start_datetime
        
        train_df = df_clean[train_mask].copy()
        val_df = df_clean[val_mask].copy()
        test_df = df_clean[test_mask].copy()
        
        # 統計情報をログ出力
        logger.info(f"データ分割結果:")
        logger.info(f"  訓練データ: {len(train_df)}件 "
                   f"({train_df[date_column].min().strftime('%Y-%m-%d')} 〜 "
                   f"{train_df[date_column].max().strftime('%Y-%m-%d') if len(train_df) > 0 else 'N/A'})")
        logger.info(f"  検証データ: {len(val_df)}件 "
                   f"({val_df[date_column].min().strftime('%Y-%m-%d') if len(val_df) > 0 else 'N/A'} 〜 "
                   f"{val_df[date_column].max().strftime('%Y-%m-%d') if len(val_df) > 0 else 'N/A'})")
        logger.info(f"  テストデータ: {len(test_df)}件 "
                   f"({test_df[date_column].min().strftime('%Y-%m-%d') if len(test_df) > 0 else 'N/A'} 〜 "
                   f"{test_df[date_column].max().strftime('%Y-%m-%d') if len(test_df) > 0 else 'N/A'})")
        
        # ターゲット分布を確認
        if target_column and target_column in df_clean.columns:
            self._analyze_target_distribution(train_df, val_df, test_df, target_column)
        
        return train_df, val_df, test_df
    
    def _analyze_target_distribution(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str
    ) -> None:
        """
        ターゲット変数の分布を分析
        
        Args:
            train_df: 訓練データ
            val_df: 検証データ
            test_df: テストデータ
            target_column: ターゲットカラム名
        """
        logger.info(f"ターゲット変数 '{target_column}' の分布を分析中...")
        
        for name, df in [("訓練", train_df), ("検証", val_df), ("テスト", test_df)]:
            if len(df) == 0:
                logger.info(f"  {name}データ: データなし")
                continue
            
            if target_column not in df.columns:
                logger.warning(f"  {name}データ: ターゲットカラムが見つかりません")
                continue
            
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # 数値型の場合
                stats = df[target_column].describe()
                logger.info(f"  {name}データ: 平均={stats['mean']:.3f}, "
                           f"標準偏差={stats['std']:.3f}, "
                           f"範囲=[{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                # カテゴリ型の場合
                value_counts = df[target_column].value_counts()
                total = len(df)
                logger.info(f"  {name}データ:")
                for value, count in value_counts.head(5).items():
                    percentage = (count / total) * 100
                    logger.info(f"    {value}: {count}件 ({percentage:.1f}%)")
    
    def create_temporal_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'created'
    ) -> pd.DataFrame:
        """
        時系列特徴量を作成
        
        Args:
            df: データフレーム
            date_column: 日付カラム名
            
        Returns:
            pd.DataFrame: 時系列特徴量を追加したデータフレーム
        """
        logger.info("時系列特徴量を作成中...")
        
        result_df = df.copy()
        
        if date_column not in result_df.columns:
            logger.warning(f"日付カラム '{date_column}' が見つかりません")
            return result_df
        
        # 日付カラムを datetime に変換
        result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
        
        # 基本的な時系列特徴量
        result_df[f'{date_column}_year'] = result_df[date_column].dt.year
        result_df[f'{date_column}_month'] = result_df[date_column].dt.month
        result_df[f'{date_column}_day'] = result_df[date_column].dt.day
        result_df[f'{date_column}_dayofweek'] = result_df[date_column].dt.dayofweek
        result_df[f'{date_column}_hour'] = result_df[date_column].dt.hour
        
        # 周期的特徴量（サイン・コサイン変換）
        result_df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * result_df[f'{date_column}_month'] / 12)
        result_df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * result_df[f'{date_column}_month'] / 12)
        result_df[f'{date_column}_dayofweek_sin'] = np.sin(2 * np.pi * result_df[f'{date_column}_dayofweek'] / 7)
        result_df[f'{date_column}_dayofweek_cos'] = np.cos(2 * np.pi * result_df[f'{date_column}_dayofweek'] / 7)
        result_df[f'{date_column}_hour_sin'] = np.sin(2 * np.pi * result_df[f'{date_column}_hour'] / 24)
        result_df[f'{date_column}_hour_cos'] = np.cos(2 * np.pi * result_df[f'{date_column}_hour'] / 24)
        
        # 相対的な時系列特徴量
        min_date = result_df[date_column].min()
        max_date = result_df[date_column].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            total_span = (max_date - min_date).total_seconds()
            if total_span > 0:
                result_df[f'{date_column}_relative_position'] = (
                    (result_df[date_column] - min_date).dt.total_seconds() / total_span
                )
        
        # 基準日からの経過日数
        reference_date = datetime(2020, 1, 1)  # 基準日
        result_df[f'{date_column}_days_since_reference'] = (
            result_df[date_column] - reference_date
        ).dt.days
        
        logger.info("時系列特徴量の作成が完了しました")
        return result_df
    
    def save_split_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
        prefix: str = "split"
    ) -> Dict[str, Path]:
        """
        分割されたデータを保存
        
        Args:
            train_df: 訓練データ
            val_df: 検証データ
            test_df: テストデータ
            output_dir: 出力ディレクトリ
            prefix: ファイル名プレフィックス
            
        Returns:
            Dict[str, Path]: 保存されたファイルのパス
        """
        logger.info("分割データを保存中...")
        
        # 出力ディレクトリを作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        # 各データセットを保存
        datasets = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for name, df in datasets.items():
            if len(df) > 0:
                filename = f"{prefix}_{name}_{timestamp}.parquet"
                file_path = output_dir / filename
                df.to_parquet(file_path, index=False)
                saved_paths[name] = file_path
                logger.info(f"{name}データを保存: {file_path} ({len(df)}件)")
            else:
                logger.warning(f"{name}データが空のため、保存をスキップします")
        
        # メタデータを保存
        metadata = {
            'split_timestamp': timestamp,
            'train_end_date': self.train_end_date,
            'test_start_date': self.test_start_date,
            'val_start_date': self.val_start_datetime.strftime('%Y-%m-%d'),
            'dataset_sizes': {name: len(df) for name, df in datasets.items()},
            'saved_files': {name: str(path) for name, path in saved_paths.items()}
        }
        
        metadata_path = output_dir / f"{prefix}_metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        saved_paths['metadata'] = metadata_path
        logger.info(f"メタデータを保存: {metadata_path}")
        
        return saved_paths


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="時系列データ分割スクリプト")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="入力データファイル（Parquet形式）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--date-column",
        default="created",
        help="日付カラム名"
    )
    parser.add_argument(
        "--target-column",
        help="ターゲットカラム名（オプション）"
    )
    parser.add_argument(
        "--prefix",
        default="temporal_split",
        help="出力ファイル名のプレフィックス"
    )
    parser.add_argument(
        "--create-temporal-features",
        action="store_true",
        help="時系列特徴量を作成する"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="検証のみ実行（分割は行わない）"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not args.input_file.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input_file}")
        return 1
    
    try:
        # データを読み込み
        logger.info(f"データを読み込み中: {args.input_file}")
        df = pd.read_parquet(args.input_file)
        
        # 分割器を作成
        splitter = TemporalSplitter()
        
        # 時系列整合性を検証
        validation_results = splitter.validate_temporal_consistency(
            df, args.date_column
        )
        
        if not validation_results['is_valid']:
            logger.error("時系列データの整合性検証に失敗しました:")
            for issue in validation_results['issues']:
                logger.error(f"  - {issue}")
            
            if not args.validate_only:
                logger.error("分割を中止します")
                return 1
        
        if args.validate_only:
            logger.info("検証のみ完了しました")
            return 0
        
        # 時系列特徴量を作成
        if args.create_temporal_features:
            df = splitter.create_temporal_features(df, args.date_column)
        
        # データを分割
        train_df, val_df, test_df = splitter.split_temporal_data(
            df, args.date_column, args.target_column
        )
        
        # 分割データを保存
        saved_paths = splitter.save_split_data(
            train_df, val_df, test_df, args.output_dir, args.prefix
        )
        
        logger.info("時系列データ分割が完了しました:")
        for name, path in saved_paths.items():
            logger.info(f"  {name}: {path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"時系列データ分割中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())