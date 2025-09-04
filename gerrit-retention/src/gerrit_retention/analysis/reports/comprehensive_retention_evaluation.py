"""
包括的継続予測システム評価

従来の固定閾値システムと新しい動的・段階的・リアルタイムシステムを
比較評価し、改善効果を定量的に分析する。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ...prediction.dynamic_threshold_calculator import DynamicThresholdCalculator
from ...prediction.realtime_retention_scorer import RealtimeRetentionScorer
from ...prediction.staged_retention_predictor import StagedRetentionPredictor

logger = logging.getLogger(__name__)


class ComprehensiveRetentionEvaluator:
    """包括的継続予測システム評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 評価対象システム
        self.dynamic_threshold_calculator = DynamicThresholdCalculator(
            config.get('dynamic_threshold', {})
        )
        self.staged_predictor = StagedRetentionPredictor(
            config.get('staged_prediction', {})
        )
        self.realtime_scorer = RealtimeRetentionScorer(
            config.get('realtime_scoring', {})
        )
        
        # 評価設定
        self.evaluation_periods = config.get('evaluation_periods', [30, 90, 180, 365])
        self.baseline_threshold = config.get('baseline_threshold', 90)  # 従来の固定閾値
        
        # 結果保存
        self.evaluation_results = {}
        
        logger.info("包括的継続予測システム評価器を初期化しました")
    
    def evaluate_all_systems(self, 
                           test_data: List[Dict[str, Any]],
                           output_dir: str = "outputs/evaluation") -> Dict[str, Any]:
        """
        全システムの包括的評価
        
        Args:
            test_data: テストデータ
            output_dir: 出力ディレクトリ
            
        Returns:
            Dict[str, Any]: 評価結果
        """
        logger.info("包括的継続予測システム評価を開始...")
        
        evaluation_results = {
            'baseline_system': {},
            'dynamic_threshold_system': {},
            'staged_prediction_system': {},
            'realtime_scoring_system': {},
            'comparison_analysis': {},
            'improvement_metrics': {}
        }
        
        # 1. ベースラインシステム（固定閾値）の評価
        logger.info("ベースラインシステムを評価中...")
        evaluation_results['baseline_system'] = self._evaluate_baseline_system(test_data)
        
        # 2. 動的閾値システムの評価
        logger.info("動的閾値システムを評価中...")
        evaluation_results['dynamic_threshold_system'] = self._evaluate_dynamic_threshold_system(test_data)
        
        # 3. 段階的予測システムの評価
        logger.info("段階的予測システムを評価中...")
        evaluation_results['staged_prediction_system'] = self._evaluate_staged_prediction_system(test_data)
        
        # 4. リアルタイムスコアリングシステムの評価
        logger.info("リアルタイムスコアリングシステムを評価中...")
        evaluation_results['realtime_scoring_system'] = self._evaluate_realtime_scoring_system(test_data)
        
        # 5. 比較分析
        logger.info("システム間比較分析を実行中...")
        evaluation_results['comparison_analysis'] = self._perform_comparison_analysis(evaluation_results)
        
        # 6. 改善効果の定量化
        logger.info("改善効果を定量化中...")
        evaluation_results['improvement_metrics'] = self._calculate_improvement_metrics(evaluation_results)
        
        # 7. 可視化とレポート生成
        logger.info("評価レポートを生成中...")
        self._generate_evaluation_report(evaluation_results, output_dir)
        
        self.evaluation_results = evaluation_results
        
        logger.info("包括的継続予測システム評価が完了しました")
        
        return evaluation_results
    
    def _evaluate_baseline_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ベースラインシステム（固定閾値）の評価"""
        
        results = {}
        
        for period_days in self.evaluation_periods:
            logger.debug(f"ベースライン評価: {period_days}日期間")
            
            predictions = []
            true_labels = []
            
            for data_point in test_data:
                try:
                    # 固定閾値による予測
                    prediction = self._baseline_predict(data_point, period_days)
                    true_label = self._get_true_label(data_point, period_days)
                    
                    if prediction is not None and true_label is not None:
                        predictions.append(prediction)
                        true_labels.append(true_label)
                        
                except Exception as e:
                    logger.warning(f"ベースライン予測エラー: {e}")
                    continue
            
            if predictions and true_labels:
                metrics = self._calculate_metrics(true_labels, predictions)
                results[f'{period_days}d'] = metrics
            else:
                results[f'{period_days}d'] = {'error': 'insufficient_data'}
        
        return results
    
    def _evaluate_dynamic_threshold_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """動的閾値システムの評価"""
        
        results = {}
        
        for period_days in self.evaluation_periods:
            logger.debug(f"動的閾値評価: {period_days}日期間")
            
            predictions = []
            true_labels = []
            threshold_info = []
            
            for data_point in test_data:
                try:
                    developer = data_point['developer']
                    activity_history = data_point['activity_history']
                    base_date = data_point.get('base_date', datetime.now())
                    
                    # 動的閾値を計算
                    dynamic_threshold = self.dynamic_threshold_calculator.calculate_dynamic_threshold(
                        developer, activity_history, base_date
                    )
                    
                    # 動的閾値による予測
                    prediction = self._dynamic_threshold_predict(
                        data_point, period_days, dynamic_threshold
                    )
                    true_label = self._get_true_label(data_point, period_days)
                    
                    if prediction is not None and true_label is not None:
                        predictions.append(prediction)
                        true_labels.append(true_label)
                        threshold_info.append(dynamic_threshold)
                        
                except Exception as e:
                    logger.warning(f"動的閾値予測エラー: {e}")
                    continue
            
            if predictions and true_labels:
                metrics = self._calculate_metrics(true_labels, predictions)
                
                # 閾値統計を追加
                thresholds = [info['threshold_days'] for info in threshold_info]
                metrics['threshold_stats'] = {
                    'mean': np.mean(thresholds),
                    'std': np.std(thresholds),
                    'min': min(thresholds),
                    'max': max(thresholds)
                }
                
                results[f'{period_days}d'] = metrics
            else:
                results[f'{period_days}d'] = {'error': 'insufficient_data'}
        
        return results
    
    def _evaluate_staged_prediction_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """段階的予測システムの評価"""
        
        results = {}
        
        # 段階的予測器を訓練（テストデータの一部を使用）
        train_size = int(len(test_data) * 0.7)
        train_data = test_data[:train_size]
        eval_data = test_data[train_size:]
        
        try:
            training_results = self.staged_predictor.fit(train_data)
            logger.debug(f"段階的予測器訓練完了: {training_results}")
        except Exception as e:
            logger.error(f"段階的予測器の訓練でエラー: {e}")
            return {'error': 'training_failed'}
        
        for period_days in self.evaluation_periods:
            logger.debug(f"段階的予測評価: {period_days}日期間")
            
            predictions = []
            true_labels = []
            prediction_details = []
            
            for data_point in eval_data:
                try:
                    developer = data_point['developer']
                    activity_history = data_point['activity_history']
                    base_date = data_point.get('base_date', datetime.now())
                    
                    # 段階的予測を実行
                    staged_prediction = self.staged_predictor.predict_staged_retention(
                        developer, activity_history, base_date
                    )
                    
                    # 対応する期間の予測を取得
                    prediction = self._extract_staged_prediction(staged_prediction, period_days)
                    true_label = self._get_true_label(data_point, period_days)
                    
                    if prediction is not None and true_label is not None:
                        predictions.append(prediction)
                        true_labels.append(true_label)
                        prediction_details.append(staged_prediction)
                        
                except Exception as e:
                    logger.warning(f"段階的予測エラー: {e}")
                    continue
            
            if predictions and true_labels:
                metrics = self._calculate_metrics(true_labels, predictions)
                
                # 予測の一貫性統計を追加
                consistency_scores = [
                    detail.get('consistency_score', 0.5) 
                    for detail in prediction_details
                ]
                metrics['consistency_stats'] = {
                    'mean': np.mean(consistency_scores),
                    'std': np.std(consistency_scores)
                }
                
                results[f'{period_days}d'] = metrics
            else:
                results[f'{period_days}d'] = {'error': 'insufficient_data'}
        
        return results
    
    def _evaluate_realtime_scoring_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """リアルタイムスコアリングシステムの評価"""
        
        results = {}
        
        # リアルタイムスコアリングのシミュレーション
        for period_days in self.evaluation_periods:
            logger.debug(f"リアルタイムスコア評価: {period_days}日期間")
            
            predictions = []
            true_labels = []
            score_trajectories = []
            
            for data_point in test_data:
                try:
                    developer = data_point['developer']
                    activity_history = data_point['activity_history']
                    base_date = data_point.get('base_date', datetime.now())
                    
                    # 初期スコアを設定
                    initial_score = self.realtime_scorer.initialize_developer_score(
                        developer, activity_history, base_date
                    )
                    
                    # 期間中の活動をシミュレート
                    score_trajectory = self._simulate_realtime_scoring(
                        data_point, period_days, base_date
                    )
                    
                    # 最終スコアから予測を生成
                    final_score = score_trajectory[-1]['score'] if score_trajectory else 0.5
                    prediction = 1 if final_score > 0.5 else 0
                    true_label = self._get_true_label(data_point, period_days)
                    
                    if prediction is not None and true_label is not None:
                        predictions.append(prediction)
                        true_labels.append(true_label)
                        score_trajectories.append(score_trajectory)
                        
                except Exception as e:
                    logger.warning(f"リアルタイムスコア予測エラー: {e}")
                    continue
            
            if predictions and true_labels:
                metrics = self._calculate_metrics(true_labels, predictions)
                
                # スコア軌跡統計を追加
                if score_trajectories:
                    final_scores = [
                        traj[-1]['score'] if traj else 0.5 
                        for traj in score_trajectories
                    ]
                    metrics['score_stats'] = {
                        'mean_final_score': np.mean(final_scores),
                        'std_final_score': np.std(final_scores)
                    }
                
                results[f'{period_days}d'] = metrics
            else:
                results[f'{period_days}d'] = {'error': 'insufficient_data'}
        
        return results
    
    def _baseline_predict(self, data_point: Dict[str, Any], period_days: int) -> Optional[int]:
        """ベースライン予測（固定閾値）"""
        
        try:
            activity_history = data_point['activity_history']
            base_date = data_point.get('base_date', datetime.now())
            
            # 最後の活動日を取得
            last_activity_date = None
            for activity in activity_history:
                activity_date = self._parse_activity_date(activity)
                if activity_date:
                    if last_activity_date is None or activity_date > last_activity_date:
                        last_activity_date = activity_date
            
            if last_activity_date is None:
                return 0  # 活動履歴なし = 離脱
            
            # 固定閾値による判定
            days_since_last_activity = (base_date - last_activity_date).days
            
            return 1 if days_since_last_activity <= self.baseline_threshold else 0
            
        except Exception as e:
            logger.warning(f"ベースライン予測エラー: {e}")
            return None
    
    def _dynamic_threshold_predict(self, 
                                 data_point: Dict[str, Any], 
                                 period_days: int,
                                 dynamic_threshold: Dict[str, Any]) -> Optional[int]:
        """動的閾値による予測"""
        
        try:
            activity_history = data_point['activity_history']
            base_date = data_point.get('base_date', datetime.now())
            
            # 最後の活動日を取得
            last_activity_date = None
            for activity in activity_history:
                activity_date = self._parse_activity_date(activity)
                if activity_date:
                    if last_activity_date is None or activity_date > last_activity_date:
                        last_activity_date = activity_date
            
            if last_activity_date is None:
                return 0  # 活動履歴なし = 離脱
            
            # 動的閾値による判定
            threshold_days = dynamic_threshold['threshold_days']
            days_since_last_activity = (base_date - last_activity_date).days
            
            return 1 if days_since_last_activity <= threshold_days else 0
            
        except Exception as e:
            logger.warning(f"動的閾値予測エラー: {e}")
            return None
    
    def _extract_staged_prediction(self, 
                                 staged_prediction: Dict[str, Any], 
                                 period_days: int) -> Optional[int]:
        """段階的予測から対応する期間の予測を抽出"""
        
        stage_predictions = staged_prediction.get('stage_predictions', {})
        
        # 最も近い期間の予測を選択
        best_match = None
        min_diff = float('inf')
        
        for stage_name, prediction in stage_predictions.items():
            horizon_days = prediction.get('horizon_days', 0)
            diff = abs(horizon_days - period_days)
            
            if diff < min_diff:
                min_diff = diff
                best_match = prediction
        
        if best_match and 'probability' in best_match:
            return 1 if best_match['probability'] > 0.5 else 0
        
        return None
    
    def _simulate_realtime_scoring(self, 
                                 data_point: Dict[str, Any], 
                                 period_days: int,
                                 base_date: datetime) -> List[Dict[str, Any]]:
        """リアルタイムスコアリングをシミュレート"""
        
        try:
            developer = data_point['developer']
            activity_history = data_point['activity_history']
            developer_id = developer.get('developer_id', 'test_dev')
            
            # 初期化
            self.realtime_scorer.initialize_developer_score(
                developer, activity_history, base_date
            )
            
            # 期間中の活動をシミュレート
            score_trajectory = []
            current_date = base_date
            
            # 期間中の活動を日付順にソート
            period_activities = []
            end_date = base_date + timedelta(days=period_days)
            
            for activity in activity_history:
                activity_date = self._parse_activity_date(activity)
                if activity_date and base_date <= activity_date <= end_date:
                    period_activities.append((activity_date, activity))
            
            period_activities.sort(key=lambda x: x[0])
            
            # 各活動でスコアを更新
            for activity_date, activity in period_activities:
                update_result = self.realtime_scorer.update_score_with_activity(
                    developer_id, activity, activity_date
                )
                
                if 'new_score' in update_result:
                    score_trajectory.append({
                        'date': activity_date,
                        'score': update_result['new_score'],
                        'activity_type': activity.get('type', 'unknown')
                    })
            
            return score_trajectory
            
        except Exception as e:
            logger.warning(f"リアルタイムスコアシミュレーションエラー: {e}")
            return []
    
    def _get_true_label(self, data_point: Dict[str, Any], period_days: int) -> Optional[int]:
        """真のラベル（実際の継続状況）を取得"""
        
        try:
            base_date = data_point.get('base_date', datetime.now())
            target_date = base_date + timedelta(days=period_days)
            
            # target_date以降の活動があるかチェック
            activity_history = data_point.get('activity_history', [])
            
            future_activities = []
            for activity in activity_history:
                activity_date = self._parse_activity_date(activity)
                if activity_date and activity_date >= target_date:
                    future_activities.append(activity)
            
            # 活動があれば継続、なければ離脱
            return 1 if future_activities else 0
            
        except Exception as e:
            logger.warning(f"真ラベル取得エラー: {e}")
            return None
    
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
    
    def _calculate_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict[str, float]:
        """評価メトリクスを計算"""
        
        try:
            metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1_score': f1_score(true_labels, predictions, zero_division=0),
                'sample_count': len(true_labels)
            }
            
            # AUCは確率予測がある場合のみ計算
            if len(set(true_labels)) > 1:  # 両クラスが存在する場合
                try:
                    metrics['auc_roc'] = roc_auc_score(true_labels, predictions)
                except ValueError:
                    metrics['auc_roc'] = 0.5
            else:
                metrics['auc_roc'] = 0.5
            
            return metrics
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            return {'error': str(e)}
    
    def _perform_comparison_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """システム間比較分析"""
        
        comparison_results = {}
        
        systems = ['baseline_system', 'dynamic_threshold_system', 
                  'staged_prediction_system', 'realtime_scoring_system']
        
        for period_days in self.evaluation_periods:
            period_key = f'{period_days}d'
            comparison_results[period_key] = {}
            
            # 各システムのメトリクスを収集
            system_metrics = {}
            for system in systems:
                if (system in evaluation_results and 
                    period_key in evaluation_results[system] and
                    'error' not in evaluation_results[system][period_key]):
                    system_metrics[system] = evaluation_results[system][period_key]
            
            if len(system_metrics) < 2:
                comparison_results[period_key]['error'] = 'insufficient_systems'
                continue
            
            # メトリクス別比較
            metrics_comparison = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                metric_values = {}
                for system, metrics in system_metrics.items():
                    if metric in metrics:
                        metric_values[system] = metrics[metric]
                
                if metric_values:
                    # 最高性能システム
                    best_system = max(metric_values.items(), key=lambda x: x[1])
                    worst_system = min(metric_values.items(), key=lambda x: x[1])
                    
                    metrics_comparison[metric] = {
                        'best_system': best_system[0],
                        'best_value': best_system[1],
                        'worst_system': worst_system[0],
                        'worst_value': worst_system[1],
                        'improvement': best_system[1] - worst_system[1],
                        'all_values': metric_values
                    }
            
            comparison_results[period_key]['metrics_comparison'] = metrics_comparison
            
            # 統計的有意性テスト（簡易版）
            comparison_results[period_key]['statistical_significance'] = self._test_statistical_significance(
                system_metrics
            )
        
        return comparison_results
    
    def _test_statistical_significance(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """統計的有意性の簡易テスト"""
        
        # 実際の実装では、より厳密な統計テストを行う
        # ここでは簡易的な差の大きさによる判定を行う
        
        significance_results = {}
        
        baseline_metrics = system_metrics.get('baseline_system', {})
        
        for system, metrics in system_metrics.items():
            if system == 'baseline_system':
                continue
            
            significance_results[system] = {}
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in baseline_metrics and metric in metrics:
                    baseline_value = baseline_metrics[metric]
                    system_value = metrics[metric]
                    
                    improvement = system_value - baseline_value
                    relative_improvement = improvement / max(baseline_value, 0.001)
                    
                    # 簡易的な有意性判定
                    if abs(relative_improvement) > 0.05:  # 5%以上の改善
                        significance = 'significant'
                    elif abs(relative_improvement) > 0.02:  # 2%以上の改善
                        significance = 'moderate'
                    else:
                        significance = 'not_significant'
                    
                    significance_results[system][metric] = {
                        'improvement': improvement,
                        'relative_improvement': relative_improvement,
                        'significance': significance
                    }
        
        return significance_results
    
    def _calculate_improvement_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """改善効果の定量化"""
        
        improvement_metrics = {}
        
        baseline_results = evaluation_results.get('baseline_system', {})
        
        systems = ['dynamic_threshold_system', 'staged_prediction_system', 'realtime_scoring_system']
        
        for system in systems:
            if system not in evaluation_results:
                continue
            
            system_results = evaluation_results[system]
            improvement_metrics[system] = {}
            
            for period_days in self.evaluation_periods:
                period_key = f'{period_days}d'
                
                if (period_key in baseline_results and period_key in system_results and
                    'error' not in baseline_results[period_key] and 
                    'error' not in system_results[period_key]):
                    
                    baseline_metrics = baseline_results[period_key]
                    system_metrics = system_results[period_key]
                    
                    period_improvements = {}
                    
                    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                        if metric in baseline_metrics and metric in system_metrics:
                            baseline_value = baseline_metrics[metric]
                            system_value = system_metrics[metric]
                            
                            absolute_improvement = system_value - baseline_value
                            relative_improvement = absolute_improvement / max(baseline_value, 0.001)
                            
                            period_improvements[metric] = {
                                'absolute': absolute_improvement,
                                'relative': relative_improvement,
                                'baseline_value': baseline_value,
                                'system_value': system_value
                            }
                    
                    improvement_metrics[system][period_key] = period_improvements
        
        # 全体的な改善サマリー
        overall_summary = {}
        for system in systems:
            if system in improvement_metrics:
                system_improvements = improvement_metrics[system]
                
                # 各メトリクスの平均改善率
                metric_averages = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    relative_improvements = []
                    for period_key, period_data in system_improvements.items():
                        if metric in period_data:
                            relative_improvements.append(period_data[metric]['relative'])
                    
                    if relative_improvements:
                        metric_averages[metric] = {
                            'mean_relative_improvement': np.mean(relative_improvements),
                            'std_relative_improvement': np.std(relative_improvements),
                            'max_relative_improvement': max(relative_improvements),
                            'min_relative_improvement': min(relative_improvements)
                        }
                
                overall_summary[system] = metric_averages
        
        improvement_metrics['overall_summary'] = overall_summary
        
        return improvement_metrics
    
    def _generate_evaluation_report(self, 
                                  evaluation_results: Dict[str, Any], 
                                  output_dir: str) -> None:
        """評価レポートを生成"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. テキストレポート
        report_path = os.path.join(output_dir, 'comprehensive_evaluation_report.md')
        self._generate_text_report(evaluation_results, report_path)
        
        # 2. 可視化
        self._generate_visualizations(evaluation_results, output_dir)
        
        # 3. 詳細データ（JSON）
        import json
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        
        # datetimeオブジェクトを文字列に変換
        serializable_results = self._make_json_serializable(evaluation_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"評価レポートを生成しました: {output_dir}")
    
    def _generate_text_report(self, 
                            evaluation_results: Dict[str, Any], 
                            report_path: str) -> None:
        """テキストレポートを生成"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 包括的継続予測システム評価レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # エグゼクティブサマリー
            f.write("## エグゼクティブサマリー\n\n")
            self._write_executive_summary(f, evaluation_results)
            
            # システム別詳細結果
            f.write("\n## システム別評価結果\n\n")
            
            systems = [
                ('baseline_system', 'ベースラインシステム（固定閾値）'),
                ('dynamic_threshold_system', '動的閾値システム'),
                ('staged_prediction_system', '段階的予測システム'),
                ('realtime_scoring_system', 'リアルタイムスコアリングシステム')
            ]
            
            for system_key, system_name in systems:
                if system_key in evaluation_results:
                    f.write(f"### {system_name}\n\n")
                    self._write_system_results(f, evaluation_results[system_key])
            
            # 比較分析
            f.write("\n## システム間比較分析\n\n")
            if 'comparison_analysis' in evaluation_results:
                self._write_comparison_analysis(f, evaluation_results['comparison_analysis'])
            
            # 改善効果
            f.write("\n## 改善効果の定量化\n\n")
            if 'improvement_metrics' in evaluation_results:
                self._write_improvement_metrics(f, evaluation_results['improvement_metrics'])
            
            # 推奨事項
            f.write("\n## 推奨事項\n\n")
            self._write_recommendations(f, evaluation_results)
    
    def _write_executive_summary(self, f, evaluation_results: Dict[str, Any]) -> None:
        """エグゼクティブサマリーを書く"""
        
        improvement_metrics = evaluation_results.get('improvement_metrics', {})
        overall_summary = improvement_metrics.get('overall_summary', {})
        
        f.write("### 主要な発見事項\n\n")
        
        # 最も改善効果の高いシステムを特定
        best_system = None
        best_improvement = -1
        
        for system, metrics in overall_summary.items():
            if 'f1_score' in metrics:
                f1_improvement = metrics['f1_score'].get('mean_relative_improvement', 0)
                if f1_improvement > best_improvement:
                    best_improvement = f1_improvement
                    best_system = system
        
        if best_system:
            system_names = {
                'dynamic_threshold_system': '動的閾値システム',
                'staged_prediction_system': '段階的予測システム',
                'realtime_scoring_system': 'リアルタイムスコアリングシステム'
            }
            
            f.write(f"- **最高性能システム**: {system_names.get(best_system, best_system)}\n")
            f.write(f"- **F1スコア改善率**: {best_improvement:.1%}\n")
        
        f.write("- **評価期間**: " + ", ".join([f"{p}日" for p in self.evaluation_periods]) + "\n")
        f.write(f"- **ベースライン閾値**: {self.baseline_threshold}日\n\n")
    
    def _write_system_results(self, f, system_results: Dict[str, Any]) -> None:
        """システム結果を書く"""
        
        f.write("| 期間 | 精度 | 適合率 | 再現率 | F1スコア | AUC |\n")
        f.write("|------|------|--------|--------|----------|-----|\n")
        
        for period_key, metrics in system_results.items():
            if 'error' in metrics:
                f.write(f"| {period_key} | エラー | - | - | - | - |\n")
            else:
                f.write(f"| {period_key} | "
                       f"{metrics.get('accuracy', 0):.3f} | "
                       f"{metrics.get('precision', 0):.3f} | "
                       f"{metrics.get('recall', 0):.3f} | "
                       f"{metrics.get('f1_score', 0):.3f} | "
                       f"{metrics.get('auc_roc', 0):.3f} |\n")
        
        f.write("\n")
    
    def _write_comparison_analysis(self, f, comparison_analysis: Dict[str, Any]) -> None:
        """比較分析を書く"""
        
        for period_key, analysis in comparison_analysis.items():
            if 'error' in analysis:
                continue
            
            f.write(f"### {period_key}期間の比較\n\n")
            
            metrics_comparison = analysis.get('metrics_comparison', {})
            
            for metric, comparison in metrics_comparison.items():
                best_system = comparison['best_system']
                best_value = comparison['best_value']
                improvement = comparison['improvement']
                
                f.write(f"**{metric.upper()}**\n")
                f.write(f"- 最高性能: {best_system} ({best_value:.3f})\n")
                f.write(f"- 改善幅: {improvement:.3f}\n\n")
    
    def _write_improvement_metrics(self, f, improvement_metrics: Dict[str, Any]) -> None:
        """改善メトリクスを書く"""
        
        overall_summary = improvement_metrics.get('overall_summary', {})
        
        for system, metrics in overall_summary.items():
            system_names = {
                'dynamic_threshold_system': '動的閾値システム',
                'staged_prediction_system': '段階的予測システム',
                'realtime_scoring_system': 'リアルタイムスコアリングシステム'
            }
            
            f.write(f"### {system_names.get(system, system)}\n\n")
            
            for metric, improvement_data in metrics.items():
                mean_improvement = improvement_data['mean_relative_improvement']
                max_improvement = improvement_data['max_relative_improvement']
                
                f.write(f"**{metric.upper()}**\n")
                f.write(f"- 平均改善率: {mean_improvement:.1%}\n")
                f.write(f"- 最大改善率: {max_improvement:.1%}\n\n")
    
    def _write_recommendations(self, f, evaluation_results: Dict[str, Any]) -> None:
        """推奨事項を書く"""
        
        f.write("### 実装推奨順序\n\n")
        f.write("1. **動的閾値システム**: 実装が比較的簡単で、即座に効果が期待できる\n")
        f.write("2. **段階的予測システム**: より精密な予測が可能、訓練データが必要\n")
        f.write("3. **リアルタイムスコアリング**: 最も高度だが、継続的な監視が可能\n\n")
        
        f.write("### 注意事項\n\n")
        f.write("- 各システムは段階的に導入し、効果を検証しながら進める\n")
        f.write("- 十分な訓練データの確保が重要\n")
        f.write("- 定期的な再評価とモデル更新が必要\n")
    
    def _generate_visualizations(self, 
                               evaluation_results: Dict[str, Any], 
                               output_dir: str) -> None:
        """可視化を生成"""
        
        try:
            # システム間性能比較
            self._plot_system_comparison(evaluation_results, output_dir)
            
            # 改善効果の可視化
            self._plot_improvement_metrics(evaluation_results, output_dir)
            
            # 期間別性能推移
            self._plot_period_performance(evaluation_results, output_dir)
            
        except Exception as e:
            logger.error(f"可視化生成でエラー: {e}")
    
    def _plot_system_comparison(self, 
                              evaluation_results: Dict[str, Any], 
                              output_dir: str) -> None:
        """システム間性能比較のプロット"""
        
        systems = ['baseline_system', 'dynamic_threshold_system', 
                  'staged_prediction_system', 'realtime_scoring_system']
        system_names = ['ベースライン', '動的閾値', '段階的予測', 'リアルタイム']
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 各システムの各期間での性能を収集
            data = []
            for j, system in enumerate(systems):
                if system in evaluation_results:
                    system_results = evaluation_results[system]
                    for period_days in self.evaluation_periods:
                        period_key = f'{period_days}d'
                        if (period_key in system_results and 
                            'error' not in system_results[period_key] and
                            metric in system_results[period_key]):
                            data.append({
                                'System': system_names[j],
                                'Period': f'{period_days}d',
                                'Value': system_results[period_key][metric]
                            })
            
            if data:
                df = pd.DataFrame(data)
                
                # システム別の平均性能
                system_means = df.groupby('System')['Value'].mean()
                
                ax.bar(system_means.index, system_means.values)
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
        
        # 最後のサブプロットを削除
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'system_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_metrics(self, 
                                evaluation_results: Dict[str, Any], 
                                output_dir: str) -> None:
        """改善効果の可視化"""
        
        improvement_metrics = evaluation_results.get('improvement_metrics', {})
        overall_summary = improvement_metrics.get('overall_summary', {})
        
        if not overall_summary:
            return
        
        systems = list(overall_summary.keys())
        system_names = {
            'dynamic_threshold_system': '動的閾値',
            'staged_prediction_system': '段階的予測',
            'realtime_scoring_system': 'リアルタイム'
        }
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, system in enumerate(systems):
            if system in overall_summary:
                improvements = []
                for metric in metrics:
                    if metric in overall_summary[system]:
                        improvement = overall_summary[system][metric]['mean_relative_improvement']
                        improvements.append(improvement * 100)  # パーセント表示
                    else:
                        improvements.append(0)
                
                ax.bar(x + i * width, improvements, width, 
                      label=system_names.get(system, system))
        
        ax.set_xlabel('メトリクス')
        ax.set_ylabel('改善率 (%)')
        ax.set_title('ベースラインからの改善率')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_period_performance(self, 
                               evaluation_results: Dict[str, Any], 
                               output_dir: str) -> None:
        """期間別性能推移のプロット"""
        
        systems = ['baseline_system', 'dynamic_threshold_system', 
                  'staged_prediction_system', 'realtime_scoring_system']
        system_names = ['ベースライン', '動的閾値', '段階的予測', 'リアルタイム']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, system in enumerate(systems):
            if system in evaluation_results:
                system_results = evaluation_results[system]
                
                periods = []
                f1_scores = []
                
                for period_days in self.evaluation_periods:
                    period_key = f'{period_days}d'
                    if (period_key in system_results and 
                        'error' not in system_results[period_key] and
                        'f1_score' in system_results[period_key]):
                        periods.append(period_days)
                        f1_scores.append(system_results[period_key]['f1_score'])
                
                if periods and f1_scores:
                    ax.plot(periods, f1_scores, marker='o', label=system_names[i])
        
        ax.set_xlabel('予測期間 (日)')
        ax.set_ylabel('F1スコア')
        ax.set_title('期間別F1スコア推移')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'period_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _make_json_serializable(self, obj):
        """JSONシリアライズ可能な形式に変換"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


if __name__ == "__main__":
    # テスト用のサンプル設定
    sample_config = {
        'dynamic_threshold': {
            'min_threshold_days': 14,
            'max_threshold_days': 365,
            'default_threshold_days': 90
        },
        'staged_prediction': {
            'prediction_stages': {
                'immediate': 7,
                'short_term': 30,
                'medium_term': 90,
                'long_term': 180
            }
        },
        'realtime_scoring': {
            'score_update_frequency': 'daily',
            'activity_window_days': 30
        },
        'evaluation_periods': [30, 90, 180],
        'baseline_threshold': 90
    }
    
    evaluator = ComprehensiveRetentionEvaluator(sample_config)
    
    print("包括的継続予測システム評価器のテスト完了")
    print(f"評価期間: {evaluator.evaluation_periods}")
    print(f"ベースライン閾値: {evaluator.baseline_threshold}日")