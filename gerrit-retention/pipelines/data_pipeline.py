#!/usr/bin/env python3
"""
Gerrit開発者定着予測システム - データパイプライン

データ抽出、変換、読み込み（ETL）を統合的に管理するパイプライン
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# プロジェクトパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gerrit_retention.data_integration.data_transformer import DataTransformer
from gerrit_retention.data_integration.gerrit_client import GerritClient
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import (
    get_logger,
    performance_context,
    performance_monitor,
)

logger = get_logger(__name__)


class DataPipeline:
    """データパイプラインクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        データパイプラインを初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_manager = get_config_manager(config_path)
        self.gerrit_client = None
        self.data_transformer = None
        
        # パイプライン設定
        self.pipeline_config = self.config_manager.get('data_pipeline', {})
        self.output_dir = Path(self.pipeline_config.get('output_dir', 'data'))
        
        # ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'external').mkdir(exist_ok=True)
        
        self._initialize_components()
        
        logger.info("データパイプラインを初期化しました")
    
    def _initialize_components(self):
        """コンポーネントを初期化"""
        try:
            # Gerritクライアント
            gerrit_config = self.config_manager.get('gerrit', {})
            if gerrit_config:
                self.gerrit_client = GerritClient(gerrit_config)
                logger.info("Gerritクライアントを初期化しました")
            
            # データ変換器
            self.data_transformer = DataTransformer()
            logger.info("データ変換器を初期化しました")
            
        except Exception as e:
            logger.error(f"コンポーネント初期化エラー: {e}")
            raise
    
    @performance_monitor("data_extraction")
    def extract_data(self, 
                    projects: List[str], 
                    start_date: str, 
                    end_date: str,
                    force_refresh: bool = False) -> Dict[str, Any]:
        """
        データを抽出
        
        Args:
            projects: プロジェクトリスト
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            force_refresh: 強制更新フラグ
            
        Returns:
            Dict[str, Any]: 抽出結果
        """
        logger.info(f"データ抽出開始: {len(projects)}プロジェクト, {start_date} - {end_date}")
        
        extraction_results = {}
        
        for project in projects:
            try:
                with performance_context("project_extraction", {"project": project}):
                    # キャッシュファイルをチェック
                    cache_file = self.output_dir / 'raw' / f"{project}_{start_date}_{end_date}.json"
                    
                    if cache_file.exists() and not force_refresh:
                        logger.info(f"キャッシュからデータを読み込み: {project}")
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            project_data = json.load(f)
                    else:
                        # Gerritからデータを抽出
                        logger.info(f"Gerritからデータを抽出: {project}")
                        
                        time_range = (start_date, end_date)
                        
                        # レビューデータを抽出
                        review_data = self.gerrit_client.extract_review_data(project, time_range)
                        
                        # 開発者プロファイルを抽出
                        developer_profiles = self.gerrit_client.extract_developer_profiles(project)
                        
                        project_data = {
                            'project': project,
                            'time_range': time_range,
                            'review_data': review_data,
                            'developer_profiles': developer_profiles,
                            'extracted_at': datetime.now().isoformat()
                        }
                        
                        # キャッシュに保存
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(project_data, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"データ抽出完了: {project} ({len(review_data)}件のレビュー)")
                    
                    extraction_results[project] = project_data
                    
            except Exception as e:
                logger.error(f"プロジェクト {project} のデータ抽出エラー: {e}")
                extraction_results[project] = {'error': str(e)}
        
        logger.info(f"データ抽出完了: {len(extraction_results)}プロジェクト")
        return extraction_results
    
    @performance_monitor("data_transformation")
    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        データを変換
        
        Args:
            raw_data: 生データ
            
        Returns:
            Dict[str, Any]: 変換済みデータ
        """
        logger.info("データ変換開始")
        
        transformed_data = {}
        
        for project, project_data in raw_data.items():
            if 'error' in project_data:
                logger.warning(f"プロジェクト {project} はエラーのためスキップします")
                continue
            
            try:
                with performance_context("project_transformation", {"project": project}):
                    logger.info(f"プロジェクト {project} のデータ変換中...")
                    
                    # レビューデータを変換
                    review_data = project_data.get('review_data', [])
                    if review_data:
                        reviews_df = pd.DataFrame(review_data)
                        transformed_reviews = self.data_transformer.transform_reviews_data(reviews_df)
                    else:
                        transformed_reviews = pd.DataFrame()
                    
                    # 開発者データを変換
                    developer_profiles = project_data.get('developer_profiles', {})
                    if developer_profiles:
                        # 開発者データを平坦化
                        dev_records = []
                        for email, dev_info in developer_profiles.items():
                            dev_info["email"] = email
                            dev_records.append(dev_info)
                        developers_df = pd.DataFrame(dev_records)
                        transformed_developers = self.data_transformer.transform_developers_data(developers_df)
                    else:
                        transformed_developers = pd.DataFrame()
                    
                    # 特徴量を生成
                    if not transformed_reviews.empty and not transformed_developers.empty:
                        features = self.data_transformer.create_developer_change_features(
                            transformed_reviews, transformed_developers
                        )
                    else:
                        features = pd.DataFrame()
                    
                    # DataFrameをJSONシリアライズ可能な形式に変換
                    reviews_data = transformed_reviews.to_dict('records') if not transformed_reviews.empty else []
                    developers_data = transformed_developers.to_dict('records') if not transformed_developers.empty else []
                    features_data = features.to_dict('records') if not features.empty else []
                    
                    transformed_data[project] = {
                        'reviews': reviews_data,
                        'developers': developers_data,
                        'features': features_data,
                        'metadata': {
                            'project': project,
                            'transformed_at': datetime.now().isoformat(),
                            'review_count': len(reviews_data),
                            'developer_count': len(developers_data)
                        }
                    }
                    
                    logger.info(f"プロジェクト {project} の変換完了")
                    
            except Exception as e:
                logger.error(f"プロジェクト {project} のデータ変換エラー: {e}")
                transformed_data[project] = {'error': str(e)}
        
        logger.info(f"データ変換完了: {len(transformed_data)}プロジェクト")
        return transformed_data
    
    @performance_monitor("data_loading")
    def load_data(self, transformed_data: Dict[str, Any]) -> bool:
        """
        データを読み込み（保存）
        
        Args:
            transformed_data: 変換済みデータ
            
        Returns:
            bool: 成功フラグ
        """
        logger.info("データ読み込み開始")
        
        try:
            # プロジェクト別にデータを保存
            for project, project_data in transformed_data.items():
                if 'error' in project_data:
                    continue
                
                project_dir = self.output_dir / 'processed' / project
                project_dir.mkdir(exist_ok=True)
                
                # レビューデータを保存
                reviews_file = project_dir / 'reviews.json'
                with open(reviews_file, 'w', encoding='utf-8') as f:
                    json.dump(project_data['reviews'], f, ensure_ascii=False, indent=2)
                
                # 開発者データを保存
                developers_file = project_dir / 'developers.json'
                with open(developers_file, 'w', encoding='utf-8') as f:
                    json.dump(project_data['developers'], f, ensure_ascii=False, indent=2)
                
                # 特徴量データを保存
                features_file = project_dir / 'features.json'
                with open(features_file, 'w', encoding='utf-8') as f:
                    json.dump(project_data['features'], f, ensure_ascii=False, indent=2)
                
                # メタデータを保存
                metadata_file = project_dir / 'metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(project_data['metadata'], f, ensure_ascii=False, indent=2)
                
                logger.info(f"プロジェクト {project} のデータを保存しました")
            
            # 統合データセットを作成
            self._create_unified_dataset(transformed_data)
            
            logger.info("データ読み込み完了")
            return True
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return False
    
    def _create_unified_dataset(self, transformed_data: Dict[str, Any]):
        """統合データセットを作成"""
        logger.info("統合データセット作成中...")
        
        all_reviews = []
        all_developers = []
        all_features = []
        
        for project, project_data in transformed_data.items():
            if 'error' in project_data:
                continue
            
            # プロジェクト情報を追加
            for review in project_data['reviews']:
                review['project'] = project
                all_reviews.append(review)
            
            for dev_email, developer in project_data['developers'].items():
                developer['project'] = project
                developer['email'] = dev_email
                all_developers.append(developer)
            
            for feature in project_data['features']:
                feature['project'] = project
                all_features.append(feature)
        
        # 統合データを保存
        unified_dir = self.output_dir / 'processed' / 'unified'
        unified_dir.mkdir(exist_ok=True)
        
        with open(unified_dir / 'all_reviews.json', 'w', encoding='utf-8') as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)
        
        with open(unified_dir / 'all_developers.json', 'w', encoding='utf-8') as f:
            json.dump(all_developers, f, ensure_ascii=False, indent=2)
        
        with open(unified_dir / 'all_features.json', 'w', encoding='utf-8') as f:
            json.dump(all_features, f, ensure_ascii=False, indent=2)
        
        # CSV形式でも保存（分析用）
        try:
            if all_reviews:
                pd.DataFrame(all_reviews).to_csv(unified_dir / 'all_reviews.csv', index=False)
            if all_developers:
                pd.DataFrame(all_developers).to_csv(unified_dir / 'all_developers.csv', index=False)
            if all_features:
                pd.DataFrame(all_features).to_csv(unified_dir / 'all_features.csv', index=False)
        except Exception as e:
            logger.warning(f"CSV保存でエラーが発生しました: {e}")
        
        logger.info(f"統合データセット作成完了: {len(all_reviews)}レビュー, {len(all_developers)}開発者")
    
    @performance_monitor("full_pipeline")
    def run_pipeline(self, 
                    projects: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    force_refresh: bool = False) -> bool:
        """
        完全なデータパイプラインを実行
        
        Args:
            projects: プロジェクトリスト（Noneの場合は設定から取得）
            start_date: 開始日（Noneの場合は設定から取得）
            end_date: 終了日（Noneの場合は設定から取得）
            force_refresh: 強制更新フラグ
            
        Returns:
            bool: 成功フラグ
        """
        logger.info("データパイプライン実行開始")
        
        try:
            # パラメータを設定から取得（指定されていない場合）
            if projects is None:
                projects = self.config_manager.get('gerrit.projects', [])
            
            if start_date is None:
                # デフォルトは3ヶ月前から
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            if end_date is None:
                # デフォルトは今日まで
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"パイプライン設定: プロジェクト={projects}, 期間={start_date}〜{end_date}")
            
            # 1. データ抽出
            raw_data = self.extract_data(projects, start_date, end_date, force_refresh)
            
            # 2. データ変換
            transformed_data = self.transform_data(raw_data)
            
            # 3. データ読み込み
            success = self.load_data(transformed_data)
            
            if success:
                logger.info("データパイプライン実行完了")
                
                # パイプライン実行履歴を保存
                self._save_pipeline_history(projects, start_date, end_date, success)
            else:
                logger.error("データパイプライン実行失敗")
            
            return success
            
        except Exception as e:
            logger.error(f"データパイプライン実行エラー: {e}")
            return False
    
    def _save_pipeline_history(self, projects: List[str], start_date: str, end_date: str, success: bool):
        """パイプライン実行履歴を保存"""
        history_file = self.output_dir / 'pipeline_history.json'
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'projects': projects,
            'start_date': start_date,
            'end_date': end_date,
            'success': success
        }
        
        # 既存の履歴を読み込み
        history = []
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 新しいエントリを追加
        history.append(history_entry)
        
        # 履歴の上限を設定（最新100件まで）
        if len(history) > 100:
            history = history[-100:]
        
        # 履歴を保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプラインの状態を取得"""
        status = {
            'initialized': True,
            'gerrit_client_ready': self.gerrit_client is not None,
            'data_transformer_ready': self.data_transformer is not None,
            'output_directory': str(self.output_dir),
            'last_run': None
        }
        
        # 最後の実行履歴を取得
        history_file = self.output_dir / 'pipeline_history.json'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if history:
                    status['last_run'] = history[-1]
        
        return status


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerrit開発者定着予測システム - データパイプライン')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--projects', nargs='+', help='プロジェクトリスト')
    parser.add_argument('--start-date', type=str, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--force-refresh', action='store_true', help='強制更新')
    parser.add_argument('--status', action='store_true', help='パイプライン状態を表示')
    
    args = parser.parse_args()
    
    try:
        # データパイプラインを初期化
        pipeline = DataPipeline(args.config)
        
        if args.status:
            # 状態を表示
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        
        # パイプラインを実行
        success = pipeline.run_pipeline(
            projects=args.projects,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"データパイプライン実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())