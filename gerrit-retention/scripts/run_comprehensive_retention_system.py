#!/usr/bin/env python3
"""
包括的継続予測システム実行スクリプト

動的閾値、段階的予測、リアルタイムスコアリングを統合した
継続予測システムの実行・評価を行う。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from gerrit_retention.analysis.reports.comprehensive_retention_evaluation import (
        ComprehensiveRetentionEvaluator,
    )
    from gerrit_retention.prediction.dynamic_threshold_calculator import (
        DynamicThresholdCalculator,
    )
    from gerrit_retention.prediction.realtime_retention_scorer import (
        RealtimeRetentionScorer,
    )
    from gerrit_retention.prediction.staged_retention_predictor import (
        StagedRetentionPredictor,
    )
except ImportError as e:
    print(f"モジュールのインポートに失敗しました: {e}")
    print("パッケージがインストールされていることを確認してください: uv sync")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveRetentionSystem:
    """包括的継続予測システム"""
    
    def __init__(self, config_path: str):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 出力ディレクトリの作成
        self.output_dir = Path("outputs/comprehensive_retention")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログディレクトリの作成
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # システムコンポーネントの初期化
        self.dynamic_threshold_calculator = None
        self.staged_predictor = None
        self.realtime_scorer = None
        self.evaluator = None
        
        logger.info(f"包括的継続予測システムを初期化しました: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 環境変数の展開
            config = self._expand_environment_variables(config)
            
            logger.info(f"設定ファイルを読み込みました: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"設定ファイルの読み込みでエラー: {e}")
            raise
    
    def _expand_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定内の環境変数を展開"""
        
        def expand_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return value
        
        return expand_value(config)
    
    def initialize_components(self) -> None:
        """システムコンポーネントを初期化"""
        
        logger.info("システムコンポーネントを初期化中...")
        
        try:
            # 動的閾値計算器
            self.dynamic_threshold_calculator = DynamicThresholdCalculator(
                self.config.get('dynamic_threshold', {})
            )
            logger.info("動的閾値計算器を初期化しました")
            
            # 段階的予測器
            self.staged_predictor = StagedRetentionPredictor(
                self.config.get('staged_prediction', {})
            )
            logger.info("段階的予測器を初期化しました")
            
            # リアルタイムスコア計算器
            self.realtime_scorer = RealtimeRetentionScorer(
                self.config.get('realtime_scoring', {})
            )
            logger.info("リアルタイムスコア計算器を初期化しました")
            
            # 評価器
            self.evaluator = ComprehensiveRetentionEvaluator(self.config)
            logger.info("包括的評価器を初期化しました")
            
        except Exception as e:
            logger.error(f"コンポーネント初期化でエラー: {e}")
            raise
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データを読み込み"""
        
        logger.info(f"データを読み込み中: {data_path}")
        
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_path.endswith('.yaml') or data_path.endswith('.yml'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"サポートされていないファイル形式: {data_path}")
            
            # データの前処理
            processed_data = self._preprocess_data(data)
            
            logger.info(f"データ読み込み完了: {len(processed_data)}件")
            return processed_data
            
        except Exception as e:
            logger.error(f"データ読み込みでエラー: {e}")
            raise
    
    def _preprocess_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """データの前処理"""
        
        if isinstance(raw_data, list):
            processed_data = []
            
            for item in raw_data:
                if self._validate_data_item(item):
                    processed_item = self._process_data_item(item)
                    if processed_item:
                        processed_data.append(processed_item)
            
            return processed_data
        
        elif isinstance(raw_data, dict):
            # 単一アイテムの場合
            if self._validate_data_item(raw_data):
                processed_item = self._process_data_item(raw_data)
                return [processed_item] if processed_item else []
            else:
                return []
        
        else:
            logger.warning(f"予期しないデータ形式: {type(raw_data)}")
            return []
    
    def _validate_data_item(self, item: Dict[str, Any]) -> bool:
        """データアイテムの検証"""
        
        required_fields = ['developer', 'activity_history']
        
        for field in required_fields:
            if field not in item:
                logger.warning(f"必須フィールドが不足: {field}")
                return False
        
        # 品質フィルタの適用
        quality_filters = self.config.get('data', {}).get('quality_filters', {})
        
        # 最小活動数チェック
        min_activity_count = quality_filters.get('min_activity_count', 0)
        activity_count = len(item.get('activity_history', []))
        if activity_count < min_activity_count:
            return False
        
        # 最小履歴期間チェック
        min_history_days = quality_filters.get('min_history_days', 0)
        if min_history_days > 0:
            activity_history = item.get('activity_history', [])
            if activity_history:
                # 最初と最後の活動日の差を計算
                dates = []
                for activity in activity_history:
                    date_str = activity.get('timestamp', activity.get('date'))
                    if date_str:
                        try:
                            if isinstance(date_str, str):
                                if 'T' in date_str:
                                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                else:
                                    date = datetime.strptime(date_str, '%Y-%m-%d')
                                dates.append(date)
                        except:
                            continue
                
                if len(dates) >= 2:
                    dates.sort()
                    history_days = (dates[-1] - dates[0]).days
                    if history_days < min_history_days:
                        return False
        
        return True
    
    def _process_data_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """データアイテムの処理"""
        
        try:
            processed_item = {
                'developer': item['developer'],
                'activity_history': item['activity_history'],
                'base_date': item.get('base_date', datetime.now())
            }
            
            # base_dateの正規化
            if isinstance(processed_item['base_date'], str):
                processed_item['base_date'] = datetime.fromisoformat(
                    processed_item['base_date'].replace('Z', '+00:00')
                )
            
            return processed_item
            
        except Exception as e:
            logger.warning(f"データアイテム処理でエラー: {e}")
            return None
    
    def run_dynamic_threshold_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """動的閾値分析を実行"""
        
        logger.info("動的閾値分析を開始...")
        
        try:
            # 開発者と活動履歴のペアを作成
            developers_with_history = [
                (item['developer'], item['activity_history'])
                for item in data
            ]
            
            # 動的閾値を一括計算
            threshold_results = self.dynamic_threshold_calculator.batch_calculate_thresholds(
                developers_with_history
            )
            
            # 統計情報を取得
            threshold_stats = self.dynamic_threshold_calculator.get_threshold_statistics(
                threshold_results
            )
            
            # 結果を保存
            output_path = self.output_dir / "dynamic_threshold_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'threshold_results': self._make_json_serializable(threshold_results),
                    'statistics': self._make_json_serializable(threshold_stats),
                    'analysis_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"動的閾値分析完了: {output_path}")
            
            return {
                'threshold_results': threshold_results,
                'statistics': threshold_stats
            }
            
        except Exception as e:
            logger.error(f"動的閾値分析でエラー: {e}")
            raise
    
    def run_staged_prediction_training(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """段階的予測の訓練を実行"""
        
        logger.info("段階的予測の訓練を開始...")
        
        try:
            # 訓練データの準備
            training_data = []
            for item in data:
                training_item = {
                    'developer': item['developer'],
                    'activity_history': item['activity_history'],
                    'base_date': item['base_date']
                }
                training_data.append(training_item)
            
            # モデル訓練
            training_results = self.staged_predictor.fit(training_data)
            
            # モデルを保存
            model_path = self.output_dir / "staged_prediction_models.joblib"
            self.staged_predictor.save_models(str(model_path))
            
            # 結果を保存
            output_path = self.output_dir / "staged_prediction_training.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'training_results': self._make_json_serializable(training_results),
                    'model_path': str(model_path),
                    'training_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"段階的予測訓練完了: {output_path}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"段階的予測訓練でエラー: {e}")
            raise
    
    def run_staged_prediction_inference(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """段階的予測の推論を実行"""
        
        logger.info("段階的予測の推論を開始...")
        
        try:
            # 開発者と活動履歴のペアを作成
            developers_with_history = [
                (item['developer'], item['activity_history'])
                for item in data
            ]
            
            # 段階的予測を一括実行
            prediction_results = self.staged_predictor.batch_predict_staged_retention(
                developers_with_history
            )
            
            # 結果を保存
            output_path = self.output_dir / "staged_prediction_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'prediction_results': self._make_json_serializable(prediction_results),
                    'prediction_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"段階的予測推論完了: {output_path}")
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"段階的予測推論でエラー: {e}")
            raise
    
    def run_realtime_scoring_simulation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """リアルタイムスコアリングのシミュレーション"""
        
        logger.info("リアルタイムスコアリングシミュレーションを開始...")
        
        try:
            simulation_results = {}
            
            for item in data:
                developer = item['developer']
                activity_history = item['activity_history']
                base_date = item['base_date']
                
                developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
                
                # 初期スコアを設定
                initial_score = self.realtime_scorer.initialize_developer_score(
                    developer, activity_history, base_date
                )
                
                # 活動をシミュレート（最近30日間）
                recent_activities = []
                cutoff_date = base_date - datetime.timedelta(days=30)
                
                for activity in activity_history:
                    activity_date = self._parse_activity_date(activity)
                    if activity_date and activity_date >= cutoff_date:
                        recent_activities.append((activity_date, activity))
                
                # 活動を日付順にソート
                recent_activities.sort(key=lambda x: x[0])
                
                # 各活動でスコアを更新
                score_trajectory = [initial_score]
                for activity_date, activity in recent_activities:
                    update_result = self.realtime_scorer.update_score_with_activity(
                        developer_id, activity, activity_date
                    )
                    score_trajectory.append(update_result)
                
                # 現在のスコア情報を取得
                current_score = self.realtime_scorer.get_current_score(developer_id)
                
                # トレンド分析
                trend_analysis = self.realtime_scorer.get_score_trend(developer_id, days=30)
                
                simulation_results[developer_id] = {
                    'initial_score': initial_score,
                    'current_score': current_score,
                    'score_trajectory': self._make_json_serializable(score_trajectory),
                    'trend_analysis': trend_analysis
                }
            
            # リスクダッシュボードを取得
            risk_dashboard = self.realtime_scorer.get_risk_dashboard()
            
            # 結果を保存
            output_path = self.output_dir / "realtime_scoring_simulation.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'simulation_results': self._make_json_serializable(simulation_results),
                    'risk_dashboard': self._make_json_serializable(risk_dashboard),
                    'simulation_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"リアルタイムスコアリングシミュレーション完了: {output_path}")
            
            return {
                'simulation_results': simulation_results,
                'risk_dashboard': risk_dashboard
            }
            
        except Exception as e:
            logger.error(f"リアルタイムスコアリングシミュレーションでエラー: {e}")
            raise
    
    def run_comprehensive_evaluation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """包括的評価を実行"""
        
        logger.info("包括的評価を開始...")
        
        try:
            # 評価を実行
            evaluation_results = self.evaluator.evaluate_all_systems(
                data, str(self.output_dir / "evaluation")
            )
            
            # 結果を保存
            output_path = self.output_dir / "comprehensive_evaluation.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'evaluation_results': self._make_json_serializable(evaluation_results),
                    'evaluation_date': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"包括的評価完了: {output_path}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"包括的評価でエラー: {e}")
            raise
    
    def generate_summary_report(self, 
                              dynamic_threshold_results: Dict[str, Any],
                              staged_prediction_results: Dict[str, Any],
                              realtime_scoring_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> None:
        """サマリーレポートを生成"""
        
        logger.info("サマリーレポートを生成中...")
        
        try:
            report_path = self.output_dir / "comprehensive_system_summary.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 包括的継続予測システム実行サマリー\n\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # システム概要
                f.write("## システム概要\n\n")
                f.write("本システムは以下の3つのコンポーネントを統合した継続予測システムです：\n\n")
                f.write("1. **動的閾値システム**: 開発者の活動パターンに基づく個別閾値計算\n")
                f.write("2. **段階的予測システム**: 複数時間軸での継続確率予測\n")
                f.write("3. **リアルタイムスコアリング**: 活動に基づくリアルタイムスコア更新\n\n")
                
                # 動的閾値結果
                f.write("## 動的閾値分析結果\n\n")
                if 'statistics' in dynamic_threshold_results:
                    stats = dynamic_threshold_results['statistics']
                    f.write(f"- 分析対象開発者数: {stats.get('total_developers', 0)}人\n")
                    
                    threshold_stats = stats.get('threshold_stats', {})
                    f.write(f"- 平均閾値: {threshold_stats.get('mean', 0):.1f}日\n")
                    f.write(f"- 閾値範囲: {threshold_stats.get('min', 0):.0f} - {threshold_stats.get('max', 0):.0f}日\n")
                    
                    type_dist = stats.get('developer_type_distribution', {})
                    f.write("- 開発者タイプ分布:\n")
                    for dev_type, count in type_dist.items():
                        f.write(f"  - {dev_type}: {count}人\n")
                
                f.write("\n")
                
                # 段階的予測結果
                f.write("## 段階的予測結果\n\n")
                if isinstance(staged_prediction_results, dict):
                    f.write(f"- 予測実行開発者数: {len(staged_prediction_results)}人\n")
                    
                    # リスクレベル分布
                    risk_levels = {}
                    for dev_id, prediction in staged_prediction_results.items():
                        if 'overall_risk_level' in prediction:
                            risk_level = prediction['overall_risk_level']
                            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                    
                    f.write("- リスクレベル分布:\n")
                    for risk_level, count in risk_levels.items():
                        f.write(f"  - {risk_level}: {count}人\n")
                
                f.write("\n")
                
                # リアルタイムスコアリング結果
                f.write("## リアルタイムスコアリング結果\n\n")
                if 'risk_dashboard' in realtime_scoring_results:
                    dashboard = realtime_scoring_results['risk_dashboard']
                    f.write(f"- 監視対象開発者数: {dashboard.get('total_developers', 0)}人\n")
                    
                    risk_dist = dashboard.get('risk_distribution', {})
                    f.write("- リスク分布:\n")
                    for risk_level, count in risk_dist.items():
                        f.write(f"  - {risk_level}: {count}人\n")
                    
                    f.write(f"- 最近のアラート数: {len(dashboard.get('recent_alerts', []))}件\n")
                    f.write(f"- 下降トレンド開発者数: {len(dashboard.get('trending_down', []))}人\n")
                    f.write(f"- 上昇トレンド開発者数: {len(dashboard.get('trending_up', []))}人\n")
                
                f.write("\n")
                
                # 評価結果
                f.write("## システム評価結果\n\n")
                if 'improvement_metrics' in evaluation_results:
                    improvement = evaluation_results['improvement_metrics']
                    overall_summary = improvement.get('overall_summary', {})
                    
                    f.write("### ベースラインからの改善率\n\n")
                    f.write("| システム | F1スコア改善 | 精度改善 | 再現率改善 |\n")
                    f.write("|----------|--------------|----------|------------|\n")
                    
                    system_names = {
                        'dynamic_threshold_system': '動的閾値',
                        'staged_prediction_system': '段階的予測',
                        'realtime_scoring_system': 'リアルタイム'
                    }
                    
                    for system, metrics in overall_summary.items():
                        system_name = system_names.get(system, system)
                        f1_improvement = metrics.get('f1_score', {}).get('mean_relative_improvement', 0)
                        accuracy_improvement = metrics.get('accuracy', {}).get('mean_relative_improvement', 0)
                        recall_improvement = metrics.get('recall', {}).get('mean_relative_improvement', 0)
                        
                        f.write(f"| {system_name} | {f1_improvement:.1%} | {accuracy_improvement:.1%} | {recall_improvement:.1%} |\n")
                
                f.write("\n")
                
                # 推奨事項
                f.write("## 推奨事項\n\n")
                f.write("### 短期的な実装\n")
                f.write("1. 動的閾値システムの本格導入\n")
                f.write("2. 高リスク開発者への個別支援\n")
                f.write("3. アラートシステムの構築\n\n")
                
                f.write("### 中長期的な改善\n")
                f.write("1. 段階的予測システムの精度向上\n")
                f.write("2. リアルタイムスコアリングの自動化\n")
                f.write("3. 予測結果に基づく介入戦略の開発\n\n")
                
                # 設定情報
                f.write("## システム設定\n\n")
                f.write(f"- 設定ファイル: {self.config_path}\n")
                f.write(f"- 出力ディレクトリ: {self.output_dir}\n")
                f.write(f"- 評価期間: {self.config.get('evaluation', {}).get('evaluation_periods', [])}\n")
            
            logger.info(f"サマリーレポートを生成しました: {report_path}")
            
        except Exception as e:
            logger.error(f"サマリーレポート生成でエラー: {e}")
            raise
    
    def _parse_activity_date(self, activity: Dict[str, Any]) -> Optional[datetime]:
        """活動の日時を解析"""
        
        try:
            if 'timestamp' in activity:
                date_str = activity['timestamp']
            elif 'date' in activity:
                date_str = activity['date']
            elif 'created' in activity:
                date_str = activity['created']
            else:
                return None
            
            if isinstance(date_str, str):
                if 'T' in date_str:
                    date_str = date_str.replace('Z', '+00:00')
                    return datetime.fromisoformat(date_str)
                else:
                    return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                return date_str
                
        except (ValueError, TypeError):
            return None
    
    def _make_json_serializable(self, obj):
        """JSONシリアライズ可能な形式に変換"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def run_full_pipeline(self, data_path: str) -> None:
        """フルパイプラインを実行"""
        
        logger.info("包括的継続予測システムのフルパイプラインを開始...")
        
        try:
            # 1. コンポーネント初期化
            self.initialize_components()
            
            # 2. データ読み込み
            data = self.load_data(data_path)
            
            # 3. 動的閾値分析
            dynamic_threshold_results = self.run_dynamic_threshold_analysis(data)
            
            # 4. 段階的予測の訓練
            staged_training_results = self.run_staged_prediction_training(data)
            
            # 5. 段階的予測の推論
            staged_prediction_results = self.run_staged_prediction_inference(data)
            
            # 6. リアルタイムスコアリングシミュレーション
            realtime_scoring_results = self.run_realtime_scoring_simulation(data)
            
            # 7. 包括的評価
            evaluation_results = self.run_comprehensive_evaluation(data)
            
            # 8. サマリーレポート生成
            self.generate_summary_report(
                dynamic_threshold_results,
                staged_prediction_results,
                realtime_scoring_results,
                evaluation_results
            )
            
            logger.info("包括的継続予測システムのフルパイプラインが完了しました")
            logger.info(f"結果は以下に保存されました: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"フルパイプライン実行でエラー: {e}")
            raise


def main():
    """メイン関数"""
    
    parser = argparse.ArgumentParser(
        description="包括的継続予測システムの実行"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/comprehensive_retention_config.yaml',
        help='設定ファイルのパス'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='入力データファイルのパス'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'threshold', 'staged', 'realtime', 'evaluation'],
        default='full',
        help='実行モード'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/comprehensive_retention',
        help='出力ディレクトリ'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログを出力'
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # システム初期化
        system = ComprehensiveRetentionSystem(args.config)
        
        # 出力ディレクトリの設定
        if args.output_dir != 'outputs/comprehensive_retention':
            system.output_dir = Path(args.output_dir)
            system.output_dir.mkdir(parents=True, exist_ok=True)
        
        # モード別実行
        if args.mode == 'full':
            system.run_full_pipeline(args.data)
        
        elif args.mode == 'threshold':
            system.initialize_components()
            data = system.load_data(args.data)
            system.run_dynamic_threshold_analysis(data)
        
        elif args.mode == 'staged':
            system.initialize_components()
            data = system.load_data(args.data)
            system.run_staged_prediction_training(data)
            system.run_staged_prediction_inference(data)
        
        elif args.mode == 'realtime':
            system.initialize_components()
            data = system.load_data(args.data)
            system.run_realtime_scoring_simulation(data)
        
        elif args.mode == 'evaluation':
            system.initialize_components()
            data = system.load_data(args.data)
            system.run_comprehensive_evaluation(data)
        
        print(f"\n✅ 実行完了！結果は {system.output_dir} に保存されました。")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()