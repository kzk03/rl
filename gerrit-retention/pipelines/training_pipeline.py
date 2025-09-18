#!/usr/bin/env python3
"""
Gerrit開発者定着予測システム - 訓練パイプライン

モデル訓練を統合的に管理するパイプライン
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import (
    get_logger,
    performance_context,
    performance_monitor,
)

logger = get_logger(__name__)


class TrainingPipeline:
    """訓練パイプラインクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        訓練パイプラインを初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_manager = get_config_manager(config_path)
        
        # パイプライン設定
        self.pipeline_config = self.config_manager.get('training_pipeline', {})
        self.models_dir = Path(self.pipeline_config.get('models_dir', 'models'))
        self.data_dir = Path(self.pipeline_config.get('data_dir', 'data/processed'))
        self.logs_dir = Path(self.pipeline_config.get('logs_dir', 'logs/training'))
        
        # ディレクトリを作成
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 訓練ステップの定義
        self.training_steps = [
            {
                'name': 'retention_model',
                'script': 'training/retention_training/train_retention_model.py',
                'config_key': 'retention_prediction',
                'required_data': ['all_developers.json', 'all_features.json'],
                'output_model': 'retention_model.pkl'
            },
            {
                'name': 'stress_model',
                'script': 'training/stress_training/train_stress_model.py',
                'config_key': 'stress_analysis',
                'required_data': ['all_developers.json', 'all_reviews.json'],
                'output_model': 'stress_model.pkl'
            },
            {
                'name': 'rl_agent',
                'script': 'training/rl_training/train_ppo_production.py',
                'config_key': 'rl_environment',
                'required_data': ['all_reviews.json', 'all_developers.json'],
                'output_model': 'ppo_agent.zip'
            }
        ]
        
        logger.info("訓練パイプラインを初期化しました")
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        訓練データの妥当性を検証
        
        Returns:
            Tuple[bool, List[str]]: (検証結果, エラーメッセージリスト)
        """
        logger.info("訓練データの検証開始")
        
        errors = []
        
        # 統合データディレクトリの存在確認
        unified_data_dir = self.data_dir / 'unified'
        if not unified_data_dir.exists():
            errors.append(f"統合データディレクトリが存在しません: {unified_data_dir}")
            return False, errors
        
        # 必要なデータファイルの存在確認
        required_files = ['all_developers.json', 'all_reviews.json', 'all_features.json']
        
        for file_name in required_files:
            file_path = unified_data_dir / file_name
            if not file_path.exists():
                errors.append(f"必要なデータファイルが存在しません: {file_path}")
            else:
                # ファイルサイズをチェック
                file_size = file_path.stat().st_size
                if file_size == 0:
                    errors.append(f"データファイルが空です: {file_path}")
                elif file_size < 100:  # 100バイト未満は異常とみなす
                    errors.append(f"データファイルが小さすぎます: {file_path} ({file_size}バイト)")
        
        # データの内容を検証
        if not errors:
            try:
                # 開発者データの検証
                with open(unified_data_dir / 'all_developers.json', 'r', encoding='utf-8') as f:
                    developers = json.load(f)
                    if not developers:
                        errors.append("開発者データが空です")
                    elif len(developers) < 2:
                        errors.append(f"開発者データが少なすぎます: {len(developers)}人")
                
                # レビューデータの検証
                with open(unified_data_dir / 'all_reviews.json', 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                    if not reviews:
                        errors.append("レビューデータが空です")
                    elif len(reviews) < 2:
                        errors.append(f"レビューデータが少なすぎます: {len(reviews)}件")
                
                # 特徴量データの検証
                with open(unified_data_dir / 'all_features.json', 'r', encoding='utf-8') as f:
                    features = json.load(f)
                    if not features:
                        errors.append("特徴量データが空です")
                
            except json.JSONDecodeError as e:
                errors.append(f"JSONファイルの解析エラー: {e}")
            except Exception as e:
                errors.append(f"データ検証エラー: {e}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("訓練データの検証完了")
        else:
            logger.error(f"訓練データの検証失敗: {errors}")
        
        return is_valid, errors
    
    @performance_monitor("model_training")
    def train_model(self, step: Dict[str, Any]) -> Tuple[bool, str]:
        """
        個別モデルを訓練
        
        Args:
            step: 訓練ステップ設定
            
        Returns:
            Tuple[bool, str]: (成功フラグ, ログメッセージ)
        """
        model_name = step['name']
        script_path = step['script']
        
        logger.info(f"モデル訓練開始: {model_name}")
        
        try:
            with performance_context("individual_training", {"model": model_name}):
                # 訓練スクリプトのパスを確認
                full_script_path = Path(script_path)
                if not full_script_path.exists():
                    return False, f"訓練スクリプトが存在しません: {full_script_path}"
                
                # 必要なデータファイルの存在確認
                unified_data_dir = self.data_dir / 'unified'
                for data_file in step['required_data']:
                    data_path = unified_data_dir / data_file
                    if not data_path.exists():
                        return False, f"必要なデータファイルが存在しません: {data_path}"
                
                # ログファイルを準備
                log_file = self.logs_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                # 環境変数を設定
                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path(__file__).parent.parent / 'src')
                env['MODEL_OUTPUT_DIR'] = str(self.models_dir)
                env['DATA_DIR'] = str(Path(__file__).parent.parent / 'data/processed/unified')  # 正しいパスに修正
                # 設定ファイルのパスも追加
                env['CONFIG_PATH'] = str(Path(__file__).parent.parent / 'configs')
                # ログレベルを設定
                env['LOG_LEVEL'] = 'INFO'
                
                # 訓練スクリプトを実行
                cmd = [sys.executable, str(full_script_path)]
                
                logger.info(f"訓練コマンド実行: {' '.join(cmd)}")
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        cwd=Path(__file__).parent.parent
                    )
                    
                    # リアルタイムでログを出力
                    for line in process.stdout:
                        f.write(line)
                        f.flush()
                        logger.debug(f"[{model_name}] {line.strip()}")
                    
                    process.wait()
                
                # 実行結果をチェック
                if process.returncode == 0:
                    # 出力モデルの存在確認
                    model_file = self.models_dir / step['output_model']
                    if model_file.exists():
                        logger.info(f"モデル訓練成功: {model_name}")
                        return True, f"モデル訓練成功: {model_file}"
                    else:
                        return False, f"出力モデルファイルが見つかりません: {model_file}"
                else:
                    # エラーログの内容を読み取り
                    error_details = ""
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                error_details = f.read()[-500:]  # 最後の500文字
                        except Exception:
                            pass
                    
                    logger.error(f"訓練スクリプトエラー ({model_name}): 終了コード {process.returncode}")
                    if error_details:
                        logger.error(f"エラー詳細: {error_details}")
                    
                    return False, f"訓練スクリプトがエラーで終了: 終了コード {process.returncode}"
                
        except Exception as e:
            logger.error(f"モデル訓練エラー ({model_name}): {e}")
            return False, f"モデル訓練エラー: {e}"
    
    @performance_monitor("model_evaluation")
    def evaluate_models(self) -> Dict[str, Any]:
        """
        訓練済みモデルを評価
        
        Returns:
            Dict[str, Any]: 評価結果
        """
        logger.info("モデル評価開始")
        
        evaluation_results = {}
        
        for step in self.training_steps:
            model_name = step['name']
            model_file = self.models_dir / step['output_model']
            
            if not model_file.exists():
                evaluation_results[model_name] = {'error': 'モデルファイルが存在しません'}
                continue
            
            try:
                # 評価スクリプトを実行
                eval_script = f"evaluation/model_evaluation/{model_name}_eval.py"
                eval_script_path = Path(eval_script)
                
                if eval_script_path.exists():
                    # 評価を実行
                    cmd = [sys.executable, str(eval_script_path), str(model_file)]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent
                    )
                    
                    if result.returncode == 0:
                        # 評価結果を解析
                        try:
                            eval_result = json.loads(result.stdout)
                            evaluation_results[model_name] = eval_result
                        except json.JSONDecodeError:
                            evaluation_results[model_name] = {
                                'status': 'success',
                                'output': result.stdout
                            }
                    else:
                        evaluation_results[model_name] = {
                            'error': f'評価スクリプトエラー: {result.stderr}'
                        }
                else:
                    # 基本的な評価情報
                    evaluation_results[model_name] = {
                        'model_file': str(model_file),
                        'file_size': model_file.stat().st_size,
                        'created_at': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                        'status': 'trained'
                    }
                    
            except Exception as e:
                evaluation_results[model_name] = {'error': str(e)}
        
        logger.info(f"モデル評価完了: {len(evaluation_results)}モデル")
        return evaluation_results
    
    def backup_models(self) -> bool:
        """
        既存モデルをバックアップ
        
        Returns:
            bool: 成功フラグ
        """
        logger.info("モデルバックアップ開始")
        
        try:
            backup_dir = self.models_dir / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 既存モデルをバックアップ
            for step in self.training_steps:
                model_file = self.models_dir / step['output_model']
                if model_file.exists():
                    backup_file = backup_dir / step['output_model']
                    shutil.copy2(model_file, backup_file)
                    logger.info(f"モデルをバックアップ: {model_file} -> {backup_file}")
            
            logger.info(f"モデルバックアップ完了: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"モデルバックアップエラー: {e}")
            return False
    
    @performance_monitor("full_training_pipeline")
    def run_training_pipeline(self, 
                             models: Optional[List[str]] = None,
                             backup_existing: bool = True,
                             evaluate_after_training: bool = True) -> Dict[str, Any]:
        """
        完全な訓練パイプラインを実行
        
        Args:
            models: 訓練するモデルリスト（Noneの場合は全モデル）
            backup_existing: 既存モデルをバックアップするか
            evaluate_after_training: 訓練後に評価を実行するか
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        logger.info("訓練パイプライン実行開始")
        
        pipeline_result = {
            'start_time': datetime.now().isoformat(),
            'data_validation': None,
            'backup_result': None,
            'training_results': {},
            'evaluation_results': {},
            'success': False
        }
        
        try:
            # 1. データ検証
            is_valid, validation_errors = self.validate_data()
            pipeline_result['data_validation'] = {
                'valid': is_valid,
                'errors': validation_errors
            }
            
            if not is_valid:
                logger.error("データ検証失敗のため訓練を中止します")
                return pipeline_result
            
            # 2. 既存モデルのバックアップ
            if backup_existing:
                backup_success = self.backup_models()
                pipeline_result['backup_result'] = backup_success
                
                if not backup_success:
                    logger.warning("モデルバックアップに失敗しましたが、訓練を続行します")
            
            # 3. モデル訓練
            steps_to_run = self.training_steps
            if models:
                steps_to_run = [step for step in self.training_steps if step['name'] in models]
            
            training_success_count = 0
            
            for step in steps_to_run:
                model_name = step['name']
                logger.info(f"モデル訓練開始: {model_name}")
                
                success, message = self.train_model(step)
                
                pipeline_result['training_results'][model_name] = {
                    'success': success,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                
                if success:
                    training_success_count += 1
                    logger.info(f"モデル訓練成功: {model_name}")
                else:
                    logger.error(f"モデル訓練失敗: {model_name} - {message}")
            
            # 4. モデル評価
            if evaluate_after_training and training_success_count > 0:
                evaluation_results = self.evaluate_models()
                pipeline_result['evaluation_results'] = evaluation_results
            
            # 5. 結果の判定
            total_models = len(steps_to_run)
            success_rate = training_success_count / total_models if total_models > 0 else 0
            pipeline_result['success'] = success_rate >= 0.5  # 50%以上成功で成功とみなす
            
            pipeline_result['end_time'] = datetime.now().isoformat()
            pipeline_result['summary'] = {
                'total_models': total_models,
                'successful_models': training_success_count,
                'success_rate': success_rate
            }
            
            if pipeline_result['success']:
                logger.info(f"訓練パイプライン実行完了: {training_success_count}/{total_models}モデル成功")
            else:
                logger.error(f"訓練パイプライン実行失敗: {training_success_count}/{total_models}モデル成功")
            
            # 実行履歴を保存
            self._save_training_history(pipeline_result)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"訓練パイプライン実行エラー: {e}")
            pipeline_result['error'] = str(e)
            pipeline_result['end_time'] = datetime.now().isoformat()
            return pipeline_result
    
    def _save_training_history(self, result: Dict[str, Any]):
        """訓練実行履歴を保存"""
        history_file = self.models_dir / 'training_history.json'
        
        # 既存の履歴を読み込み
        history = []
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 新しいエントリを追加
        history.append(result)
        
        # 履歴の上限を設定（最新50件まで）
        if len(history) > 50:
            history = history[-50:]
        
        # 履歴を保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_training_status(self) -> Dict[str, Any]:
        """訓練パイプラインの状態を取得"""
        status = {
            'models_directory': str(self.models_dir),
            'data_directory': str(self.data_dir),
            'logs_directory': str(self.logs_dir),
            'available_models': [],
            'last_training': None
        }
        
        # 利用可能なモデルを確認
        for step in self.training_steps:
            model_file = self.models_dir / step['output_model']
            model_info = {
                'name': step['name'],
                'file': step['output_model'],
                'exists': model_file.exists()
            }
            
            if model_file.exists():
                stat = model_file.stat()
                model_info.update({
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            status['available_models'].append(model_info)
        
        # 最後の訓練履歴を取得
        history_file = self.models_dir / 'training_history.json'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if history:
                    status['last_training'] = history[-1]
        
        return status


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerrit開発者定着予測システム - 訓練パイプライン')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--models', nargs='+', help='訓練するモデルリスト')
    parser.add_argument('--no-backup', action='store_true', help='既存モデルをバックアップしない')
    parser.add_argument('--no-evaluation', action='store_true', help='訓練後の評価をスキップ')
    parser.add_argument('--status', action='store_true', help='訓練パイプライン状態を表示')
    parser.add_argument('--validate-data', action='store_true', help='データ検証のみ実行')
    
    args = parser.parse_args()
    
    try:
        # 訓練パイプラインを初期化
        pipeline = TrainingPipeline(args.config)
        
        if args.status:
            # 状態を表示
            status = pipeline.get_training_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        
        if args.validate_data:
            # データ検証のみ実行
            is_valid, errors = pipeline.validate_data()
            result = {'valid': is_valid, 'errors': errors}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if is_valid else 1
        
        # 訓練パイプラインを実行
        result = pipeline.run_training_pipeline(
            models=args.models,
            backup_existing=not args.no_backup,
            evaluate_after_training=not args.no_evaluation
        )
        
        # 結果を表示
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        logger.error(f"訓練パイプライン実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())