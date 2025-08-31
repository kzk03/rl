#!/usr/bin/env python3
"""
Gerrit開発者定着予測システム - フル統合パイプライン実行スクリプト

データ処理から訓練、評価、デプロイまでの全工程を統合的に実行します。
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import (
    get_logger,
    performance_monitor,
    start_system_monitoring,
)

# パイプラインをインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_pipeline import DataPipeline
from pipelines.training_pipeline import TrainingPipeline

logger = get_logger(__name__)


class FullPipeline:
    """フル統合パイプラインクラス"""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "production"):
        """
        フルパイプラインを初期化
        
        Args:
            config_path: 設定ファイルのパス
            environment: 実行環境
        """
        self.config_manager = get_config_manager(config_path, environment)
        self.environment = environment
        
        # パイプラインコンポーネントを初期化
        self.data_pipeline = DataPipeline(config_path)
        self.training_pipeline = TrainingPipeline(config_path)
        
        # 実行設定
        self.pipeline_config = self.config_manager.get('full_pipeline', {})
        
        # 出力ディレクトリの設定
        self.output_dir = Path("outputs/full_pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"フル統合パイプラインを初期化しました（環境: {environment}）")

    @performance_monitor("full_pipeline_execution")
    def run_full_pipeline(self, 
                         skip_data: bool = False,
                         skip_training: bool = False,
                         skip_evaluation: bool = False,
                         projects: Optional[list] = None,
                         models: Optional[list] = None) -> Dict[str, Any]:
        """
        フル統合パイプラインを実行
        
        Args:
            skip_data: データパイプラインをスキップ
            skip_training: 訓練パイプラインをスキップ
            skip_evaluation: 評価をスキップ
            projects: 対象プロジェクトリスト
            models: 訓練対象モデルリスト
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        logger.info("フル統合パイプライン実行開始")
        
        pipeline_result = {
            'start_time': time.time(),
            'environment': self.environment,
            'data_pipeline_result': None,
            'training_pipeline_result': None,
            'evaluation_result': None,
            'success': False,
            'errors': []
        }
        
        try:
            # 1. データパイプライン実行
            if not skip_data:
                logger.info("=== データパイプライン実行 ===")
                
                data_success = self.data_pipeline.run_pipeline(
                    projects=projects,
                    force_refresh=self.pipeline_config.get('force_data_refresh', False)
                )
                
                pipeline_result['data_pipeline_result'] = {
                    'success': data_success,
                    'timestamp': time.time()
                }
                
                if not data_success:
                    error_msg = "データパイプライン実行失敗"
                    logger.error(error_msg)
                    pipeline_result['errors'].append(error_msg)
                    
                    # データパイプライン失敗時の処理
                    if self.pipeline_config.get('stop_on_data_failure', True):
                        logger.error("データパイプライン失敗のため処理を中止します")
                        return pipeline_result
                else:
                    logger.info("データパイプライン実行完了")
            else:
                logger.info("データパイプラインをスキップしました")
            
            # 2. 訓練パイプライン実行
            if not skip_training:
                logger.info("=== 訓練パイプライン実行 ===")
                
                training_result = self.training_pipeline.run_training_pipeline(
                    models=models,
                    backup_existing=self.pipeline_config.get('backup_models', True),
                    evaluate_after_training=not skip_evaluation
                )
                
                pipeline_result['training_pipeline_result'] = training_result
                
                if not training_result['success']:
                    error_msg = "訓練パイプライン実行失敗"
                    logger.error(error_msg)
                    pipeline_result['errors'].append(error_msg)
                    
                    # 訓練失敗時の処理
                    if self.pipeline_config.get('stop_on_training_failure', False):
                        logger.error("訓練パイプライン失敗のため処理を中止します")
                        return pipeline_result
                else:
                    logger.info("訓練パイプライン実行完了")
            else:
                logger.info("訓練パイプラインをスキップしました")
            
            # 3. 追加評価実行（訓練パイプラインで評価をスキップした場合）
            if not skip_evaluation and skip_training:
                logger.info("=== 追加評価実行 ===")
                
                evaluation_result = self.training_pipeline.evaluate_models()
                pipeline_result['evaluation_result'] = evaluation_result
                
                logger.info("追加評価完了")
            
            # 4. 結果の統合判定
            pipeline_result['success'] = self._assess_overall_success(pipeline_result)
            
            # 5. 後処理
            self._post_pipeline_processing(pipeline_result)
            
            pipeline_result['end_time'] = time.time()
            pipeline_result['duration'] = pipeline_result['end_time'] - pipeline_result['start_time']
            
            if pipeline_result['success']:
                logger.info(f"フル統合パイプライン実行完了（実行時間: {pipeline_result['duration']:.2f}秒）")
            else:
                logger.error(f"フル統合パイプライン実行失敗（実行時間: {pipeline_result['duration']:.2f}秒）")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"フル統合パイプライン実行エラー: {e}")
            pipeline_result['errors'].append(str(e))
            pipeline_result['end_time'] = time.time()
            pipeline_result['duration'] = pipeline_result['end_time'] - pipeline_result['start_time']
            return pipeline_result
    
    def _assess_overall_success(self, result: Dict[str, Any]) -> bool:
        """全体的な成功を評価"""
        # データパイプラインの結果をチェック
        data_success = True
        if result['data_pipeline_result']:
            data_success = result['data_pipeline_result']['success']
        
        # 訓練パイプラインの結果をチェック
        training_success = True
        if result['training_pipeline_result']:
            training_success = result['training_pipeline_result']['success']
        
        # 最低限の成功条件
        if self.pipeline_config.get('require_data_success', True) and not data_success:
            return False
        
        if self.pipeline_config.get('require_training_success', True) and not training_success:
            return False
        
        return len(result['errors']) == 0
    
    def _post_pipeline_processing(self, result: Dict[str, Any]):
        """パイプライン後処理"""
        try:
            # 実行履歴を保存
            self._save_pipeline_history(result)
            
            # 通知を送信（設定されている場合）
            if self.config_manager.get('notifications.enabled', False):
                self._send_notifications(result)
            
            # クリーンアップ処理
            if self.pipeline_config.get('cleanup_after_run', False):
                self._cleanup_temporary_files()
            
        except Exception as e:
            logger.warning(f"後処理でエラーが発生しました: {e}")
    
    def _save_pipeline_history(self, result: Dict[str, Any]):
        """パイプライン実行履歴を保存"""
        history_file = self.output_dir / 'pipeline_history.json'
        
        # 既存の履歴を読み込み
        history = []
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 新しいエントリを追加
        history_entry = {
            'timestamp': result['start_time'],
            'environment': self.environment,
            'success': result['success'],
            'duration': result.get('duration', 0),
            'errors': result['errors']
        }
        
        history.append(history_entry)
        
        # 履歴の上限を設定（最新100件まで）
        if len(history) > 100:
            history = history[-100:]
        
        # 履歴を保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def _send_notifications(self, result: Dict[str, Any]):
        """通知を送信"""
        # 実装: Slack、メール、Discord等への通知
        logger.info("通知送信機能は未実装です")
    
    def _cleanup_temporary_files(self):
        """一時ファイルをクリーンアップ"""
        # 実装: 一時ファイルの削除
        logger.info("クリーンアップ機能は未実装です")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプラインの状態を取得"""
        return {
            'environment': self.environment,
            'data_pipeline_status': self.data_pipeline.get_pipeline_status(),
            'training_pipeline_status': self.training_pipeline.get_training_status(),
            'config_valid': self.config_manager.validate_config()
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Gerrit開発者定着予測システム - フル統合パイプライン')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--environment', type=str, default='production', 
                       choices=['development', 'testing', 'production'],
                       help='実行環境')
    parser.add_argument('--skip-data', action='store_true', help='データパイプラインをスキップ')
    parser.add_argument('--skip-training', action='store_true', help='訓練パイプラインをスキップ')
    parser.add_argument('--skip-evaluation', action='store_true', help='評価をスキップ')
    parser.add_argument('--projects', nargs='+', help='対象プロジェクトリスト')
    parser.add_argument('--models', nargs='+', help='訓練対象モデルリスト')
    parser.add_argument('--status', action='store_true', help='パイプライン状態を表示')
    parser.add_argument('--development', action='store_true', help='開発モードで実行')
    parser.add_argument('--monitoring', action='store_true', help='システム監視を開始')
    
    args = parser.parse_args()
    
    # 開発モードの場合は環境を設定
    if args.development:
        args.environment = 'development'
    
    try:
        # システム監視を開始（オプション）
        if args.monitoring:
            start_system_monitoring(interval_seconds=60)
        
        # フルパイプラインを初期化
        pipeline = FullPipeline(args.config, args.environment)
        
        if args.status:
            # 状態を表示
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        
        # フルパイプラインを実行
        result = pipeline.run_full_pipeline(
            skip_data=args.skip_data,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            projects=args.projects,
            models=args.models
        )
        
        # 結果を表示
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        logger.error(f"フル統合パイプライン実行エラー: {e}")
        return 1


if __name__ == "__main__":
    main()