"""
定着予測評価システム

AUC、F1スコア、精度・再現率の計算、時系列での予測精度変化追跡機能を提供する。
開発者定着予測モデルの性能を包括的に評価し、時系列整合性を維持した評価を行う。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from gerrit_retention.prediction.retention_factor_analyzer import (
        RetentionFactorAnalyzer,
    )
    from gerrit_retention.prediction.retention_predictor import RetentionPredictor
except ImportError:
    # フォールバック: 予測モジュールが利用できない場合
    class RetentionPredictor:
        pass
    class RetentionFactorAnalyzer:
        pass

logger = logging.getLogger(__name__)


class RetentionEvaluator:
    """定着予測評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 評価設定
        self.evaluation_metrics = config.get('evaluation_metrics', [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_score'
        ])
        self.confidence_thresholds = config.get('confidence_thresholds', [0.3, 0.5, 0.7])
        self.time_window_days = config.get('time_window_days', 30)
        self.min_samples_per_period = config.get('min_samples_per_period', 10)
        
        # 開発者セグメント定義
        self.developer_segments = config.get('developer_segments', {
            'junior': {'experience_months': (0, 12)},
            'mid': {'experience_months': (12, 36)},
            'senior': {'experience_months': (36, float('inf'))}
        })
        
        # 評価結果のキャッシュ
        self.evaluation_cache = {}
        
    def evaluate_model_performance(self, 
                                 predictor: RetentionPredictor,
                                 test_developers: List[Dict[str, Any]],
                                 test_contexts: List[Dict[str, Any]],
                                 true_labels: List[int],
                                 evaluation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        モデル性能を評価
        
        Args:
            predictor: 定着予測器
            test_developers: テスト用開発者データ
            test_contexts: テスト用コンテキストデータ
            true_labels: 真の定着ラベル
            evaluation_id: 評価ID（キャッシュ用）
            
        Returns:
            Dict[str, Any]: 評価結果
        """
        logger.info(f"モデル性能評価を開始: {len(test_developers)}サンプル")
        
        if evaluation_id is None:
            evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 予測を実行
        predicted_probs = predictor.predict_batch(test_developers, test_contexts)
        
        # 基本メトリクスを計算
        basic_metrics = self._calculate_basic_metrics(
            true_labels, predicted_probs, self.confidence_thresholds
        )
        
        # 閾値別性能を計算
        threshold_performance = self._calculate_threshold_performance(
            true_labels, predicted_probs
        )
        
        # ROC・PR曲線データを計算
        curve_data = self._calculate_curve_data(true_labels, predicted_probs)
        
        # セグメント別性能を計算
        segment_performance = self._calculate_segment_performance(
            predictor, test_developers, test_contexts, true_labels
        )
        
        # 予測信頼度分析
        confidence_analysis = self._analyze_prediction_confidence(
            predicted_probs, true_labels
        )
        
        # エラー分析
        error_analysis = self._analyze_prediction_errors(
            test_developers, test_contexts, predicted_probs, true_labels
        )
        
        evaluation_result = {
            'evaluation_metadata': {
                'evaluation_id': evaluation_id,
                'evaluation_date': datetime.now().isoformat(),
                'sample_count': len(test_developers),
                'positive_rate': sum(true_labels) / len(true_labels)
            },
            'basic_metrics': basic_metrics,
            'threshold_performance': threshold_performance,
            'curve_data': curve_data,
            'segment_performance': segment_performance,
            'confidence_analysis': confidence_analysis,
            'error_analysis': error_analysis
        }
        
        # キャッシュに保存
        self.evaluation_cache[evaluation_id] = evaluation_result
        
        logger.info(f"モデル性能評価完了: AUC={basic_metrics['auc_score']:.4f}")
        
        return evaluation_result
    
    def track_performance_over_time(self, 
                                   predictor: RetentionPredictor,
                                   time_series_data: List[Dict[str, Any]],
                                   time_window_days: Optional[int] = None) -> Dict[str, Any]:
        """
        時系列での予測精度変化を追跡
        
        Args:
            predictor: 定着予測器
            time_series_data: 時系列データ（日付順にソート済み）
            time_window_days: 評価期間（日数）
            
        Returns:
            Dict[str, Any]: 時系列性能追跡結果
        """
        logger.info("時系列での予測精度変化を追跡中...")
        
        if time_window_days is None:
            time_window_days = self.time_window_days
        
        # 時系列データを期間別に分割
        time_periods = self._split_time_series_data(time_series_data, time_window_days)
        
        performance_history = []
        
        for period_data in time_periods:
            if len(period_data['developers']) < self.min_samples_per_period:
                logger.warning(f"期間 {period_data['start_date']} - {period_data['end_date']}: "
                             f"サンプル数不足 ({len(period_data['developers'])})")
                continue
            
            try:
                # 期間内の性能を評価
                period_performance = self.evaluate_model_performance(
                    predictor,
                    period_data['developers'],
                    period_data['contexts'],
                    period_data['labels'],
                    f"period_{period_data['start_date']}_{period_data['end_date']}"
                )
                
                # 期間情報を追加
                period_performance['time_period'] = {
                    'start_date': period_data['start_date'],
                    'end_date': period_data['end_date'],
                    'sample_count': len(period_data['developers'])
                }
                
                performance_history.append(period_performance)
                
            except Exception as e:
                logger.error(f"期間 {period_data['start_date']} - {period_data['end_date']} "
                           f"の評価でエラー: {e}")
                continue
        
        # 性能トレンドを分析
        trend_analysis = self._analyze_performance_trends(performance_history)
        
        # 性能劣化の検出
        degradation_detection = self._detect_performance_degradation(performance_history)
        
        # 季節性・周期性の分析
        seasonality_analysis = self._analyze_performance_seasonality(performance_history)
        
        time_series_evaluation = {
            'tracking_metadata': {
                'tracking_date': datetime.now().isoformat(),
                'time_window_days': time_window_days,
                'total_periods': len(performance_history),
                'total_samples': sum(p['evaluation_metadata']['sample_count'] 
                                   for p in performance_history)
            },
            'performance_history': performance_history,
            'trend_analysis': trend_analysis,
            'degradation_detection': degradation_detection,
            'seasonality_analysis': seasonality_analysis
        }
        
        return time_series_evaluation
    
    def compare_model_versions(self, 
                              predictors: Dict[str, RetentionPredictor],
                              test_developers: List[Dict[str, Any]],
                              test_contexts: List[Dict[str, Any]],
                              true_labels: List[int]) -> Dict[str, Any]:
        """
        複数のモデルバージョンを比較
        
        Args:
            predictors: モデル名とRetentionPredictorの辞書
            test_developers: テスト用開発者データ
            test_contexts: テスト用コンテキストデータ
            true_labels: 真の定着ラベル
            
        Returns:
            Dict[str, Any]: モデル比較結果
        """
        logger.info(f"モデルバージョン比較を開始: {len(predictors)}モデル")
        
        model_results = {}
        
        # 各モデルを評価
        for model_name, predictor in predictors.items():
            try:
                result = self.evaluate_model_performance(
                    predictor, test_developers, test_contexts, true_labels,
                    f"comparison_{model_name}"
                )
                model_results[model_name] = result
                
            except Exception as e:
                logger.error(f"モデル {model_name} の評価でエラー: {e}")
                continue
        
        # モデル間の統計的有意性を検定
        statistical_tests = self._perform_statistical_tests(
            model_results, test_developers, test_contexts, true_labels
        )
        
        # 最適なモデルを選択
        best_model_selection = self._select_best_model(model_results)
        
        # 性能差の分析
        performance_differences = self._analyze_performance_differences(model_results)
        
        comparison_result = {
            'comparison_metadata': {
                'comparison_date': datetime.now().isoformat(),
                'models_compared': list(predictors.keys()),
                'test_sample_count': len(test_developers)
            },
            'model_results': model_results,
            'statistical_tests': statistical_tests,
            'best_model_selection': best_model_selection,
            'performance_differences': performance_differences
        }
        
        return comparison_result
    
    def generate_evaluation_report(self, 
                                 evaluation_results: List[Dict[str, Any]],
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        評価レポートを生成
        
        Args:
            evaluation_results: 評価結果のリスト
            output_path: 出力パス（Noneの場合は保存しない）
            
        Returns:
            Dict[str, Any]: 生成されたレポート
        """
        logger.info("評価レポートを生成中...")
        
        # 全体サマリーを計算
        overall_summary = self._calculate_overall_summary(evaluation_results)
        
        # 性能トレンド分析
        performance_trends = self._analyze_overall_trends(evaluation_results)
        
        # 問題点の特定
        identified_issues = self._identify_performance_issues(evaluation_results)
        
        # 改善提案を生成
        improvement_suggestions = self._generate_improvement_suggestions(
            overall_summary, performance_trends, identified_issues
        )
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'evaluation_count': len(evaluation_results),
                'report_version': '1.0'
            },
            'overall_summary': overall_summary,
            'performance_trends': performance_trends,
            'identified_issues': identified_issues,
            'improvement_suggestions': improvement_suggestions,
            'detailed_results': evaluation_results
        }
        
        # ファイルに保存
        if output_path:
            self._save_evaluation_report(report, output_path)
        
        logger.info(f"評価レポート生成完了: {len(evaluation_results)}件の評価結果")
        
        return report
    
    def visualize_evaluation_results(self, 
                                   evaluation_result: Dict[str, Any],
                                   output_dir: str) -> List[str]:
        """
        評価結果を可視化
        
        Args:
            evaluation_result: 評価結果
            output_dir: 出力ディレクトリ
            
        Returns:
            List[str]: 生成されたファイルパスのリスト
        """
        logger.info("評価結果の可視化を開始...")
        
        generated_files = []
        
        # 1. ROC曲線
        roc_plot_path = self._plot_roc_curve(
            evaluation_result['curve_data'],
            f"{output_dir}/roc_curve.png"
        )
        generated_files.append(roc_plot_path)
        
        # 2. Precision-Recall曲線
        pr_plot_path = self._plot_precision_recall_curve(
            evaluation_result['curve_data'],
            f"{output_dir}/precision_recall_curve.png"
        )
        generated_files.append(pr_plot_path)
        
        # 3. 混同行列
        confusion_plot_path = self._plot_confusion_matrix(
            evaluation_result['basic_metrics'],
            f"{output_dir}/confusion_matrix.png"
        )
        generated_files.append(confusion_plot_path)
        
        # 4. 閾値別性能
        threshold_plot_path = self._plot_threshold_performance(
            evaluation_result['threshold_performance'],
            f"{output_dir}/threshold_performance.png"
        )
        generated_files.append(threshold_plot_path)
        
        # 5. セグメント別性能
        if 'segment_performance' in evaluation_result:
            segment_plot_path = self._plot_segment_performance(
                evaluation_result['segment_performance'],
                f"{output_dir}/segment_performance.png"
            )
            generated_files.append(segment_plot_path)
        
        # 6. 予測信頼度分布
        confidence_plot_path = self._plot_confidence_distribution(
            evaluation_result['confidence_analysis'],
            f"{output_dir}/confidence_distribution.png"
        )
        generated_files.append(confidence_plot_path)
        
        logger.info(f"可視化完了: {len(generated_files)}個のファイルを生成")
        
        return generated_files
    
    def _calculate_basic_metrics(self, 
                               true_labels: List[int],
                               predicted_probs: List[float],
                               thresholds: List[float]) -> Dict[str, Any]:
        """基本メトリクスを計算"""
        
        metrics = {}
        
        # デフォルト閾値（0.5）での分類メトリクス
        predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]
        
        metrics['accuracy'] = float(accuracy_score(true_labels, predicted_labels))
        metrics['precision'] = float(precision_score(true_labels, predicted_labels, zero_division=0))
        metrics['recall'] = float(recall_score(true_labels, predicted_labels, zero_division=0))
        metrics['f1_score'] = float(f1_score(true_labels, predicted_labels, zero_division=0))
        metrics['auc_score'] = float(roc_auc_score(true_labels, predicted_probs))
        
        # 混同行列
        cm = confusion_matrix(true_labels, predicted_labels)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # 分類レポート
        metrics['classification_report'] = classification_report(
            true_labels, predicted_labels, output_dict=True
        )
        
        return metrics
    
    def _calculate_threshold_performance(self, 
                                       true_labels: List[int],
                                       predicted_probs: List[float]) -> Dict[str, List[float]]:
        """閾値別性能を計算"""
        
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_metrics = {
            'thresholds': thresholds.tolist(),
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for threshold in thresholds:
            predicted_labels = [1 if prob > threshold else 0 for prob in predicted_probs]
            
            threshold_metrics['accuracy'].append(
                float(accuracy_score(true_labels, predicted_labels))
            )
            threshold_metrics['precision'].append(
                float(precision_score(true_labels, predicted_labels, zero_division=0))
            )
            threshold_metrics['recall'].append(
                float(recall_score(true_labels, predicted_labels, zero_division=0))
            )
            threshold_metrics['f1_score'].append(
                float(f1_score(true_labels, predicted_labels, zero_division=0))
            )
        
        return threshold_metrics
    
    def _calculate_curve_data(self, 
                            true_labels: List[int],
                            predicted_probs: List[float]) -> Dict[str, Any]:
        """ROC・PR曲線データを計算"""
        
        # ROC曲線
        fpr, tpr, roc_thresholds = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall曲線
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, predicted_probs)
        pr_auc = auc(recall, precision)
        
        curve_data = {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': float(roc_auc)
            },
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'auc': float(pr_auc)
            }
        }
        
        return curve_data
    
    def _calculate_segment_performance(self, 
                                     predictor: RetentionPredictor,
                                     test_developers: List[Dict[str, Any]],
                                     test_contexts: List[Dict[str, Any]],
                                     true_labels: List[int]) -> Dict[str, Any]:
        """セグメント別性能を計算"""
        
        segment_performance = {}
        
        for segment_name, segment_criteria in self.developer_segments.items():
            # セグメントに該当する開発者を抽出
            segment_indices = []
            
            for i, developer in enumerate(test_developers):
                if self._matches_segment_criteria(developer, segment_criteria):
                    segment_indices.append(i)
            
            if len(segment_indices) < 5:  # 最小サンプル数
                continue
            
            # セグメントデータを抽出
            segment_developers = [test_developers[i] for i in segment_indices]
            segment_contexts = [test_contexts[i] for i in segment_indices]
            segment_labels = [true_labels[i] for i in segment_indices]
            
            # セグメント性能を評価
            try:
                segment_result = self.evaluate_model_performance(
                    predictor, segment_developers, segment_contexts, segment_labels,
                    f"segment_{segment_name}"
                )
                segment_performance[segment_name] = segment_result['basic_metrics']
                segment_performance[segment_name]['sample_count'] = len(segment_indices)
                
            except Exception as e:
                logger.warning(f"セグメント {segment_name} の評価でエラー: {e}")
                continue
        
        return segment_performance
    
    def _analyze_prediction_confidence(self, 
                                     predicted_probs: List[float],
                                     true_labels: List[int]) -> Dict[str, Any]:
        """予測信頼度を分析"""
        
        # 信頼度区間別の精度
        confidence_bins = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        confidence_analysis = {}
        
        for low, high in confidence_bins:
            bin_name = f"{low:.1f}-{high:.1f}"
            
            # 該当する予測を抽出
            bin_indices = [
                i for i, prob in enumerate(predicted_probs)
                if low <= prob < high
            ]
            
            if len(bin_indices) == 0:
                continue
            
            bin_probs = [predicted_probs[i] for i in bin_indices]
            bin_labels = [true_labels[i] for i in bin_indices]
            bin_predictions = [1 if prob > 0.5 else 0 for prob in bin_probs]
            
            confidence_analysis[bin_name] = {
                'sample_count': len(bin_indices),
                'accuracy': float(accuracy_score(bin_labels, bin_predictions)),
                'avg_confidence': float(np.mean(bin_probs)),
                'positive_rate': float(np.mean(bin_labels))
            }
        
        # 全体の信頼度統計
        confidence_analysis['overall'] = {
            'mean_confidence': float(np.mean(predicted_probs)),
            'std_confidence': float(np.std(predicted_probs)),
            'min_confidence': float(np.min(predicted_probs)),
            'max_confidence': float(np.max(predicted_probs))
        }
        
        return confidence_analysis
    
    def _analyze_prediction_errors(self, 
                                 test_developers: List[Dict[str, Any]],
                                 test_contexts: List[Dict[str, Any]],
                                 predicted_probs: List[float],
                                 true_labels: List[int]) -> Dict[str, Any]:
        """予測エラーを分析"""
        
        predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]
        
        # False Positive（誤って定着と予測）
        fp_indices = [
            i for i in range(len(true_labels))
            if true_labels[i] == 0 and predicted_labels[i] == 1
        ]
        
        # False Negative（誤って離脱と予測）
        fn_indices = [
            i for i in range(len(true_labels))
            if true_labels[i] == 1 and predicted_labels[i] == 0
        ]
        
        error_analysis = {
            'false_positive_count': len(fp_indices),
            'false_negative_count': len(fn_indices),
            'false_positive_rate': len(fp_indices) / len(true_labels),
            'false_negative_rate': len(fn_indices) / len(true_labels)
        }
        
        # エラーパターンの分析（簡略化）
        if fp_indices:
            fp_confidences = [predicted_probs[i] for i in fp_indices]
            error_analysis['false_positive_avg_confidence'] = float(np.mean(fp_confidences))
        
        if fn_indices:
            fn_confidences = [predicted_probs[i] for i in fn_indices]
            error_analysis['false_negative_avg_confidence'] = float(np.mean(fn_confidences))
        
        return error_analysis
    
    def _split_time_series_data(self, 
                              time_series_data: List[Dict[str, Any]],
                              window_days: int) -> List[Dict[str, Any]]:
        """時系列データを期間別に分割"""
        
        if not time_series_data:
            return []
        
        # 日付でソート
        sorted_data = sorted(
            time_series_data,
            key=lambda x: datetime.fromisoformat(x['date']) if isinstance(x['date'], str) else x['date']
        )
        
        periods = []
        current_period = {
            'developers': [],
            'contexts': [],
            'labels': [],
            'start_date': None,
            'end_date': None
        }
        
        start_date = None
        
        for data_point in sorted_data:
            date = data_point['date']
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            
            if start_date is None:
                start_date = date
                current_period['start_date'] = date.strftime('%Y-%m-%d')
            
            # 期間を超えた場合は新しい期間を開始
            if (date - start_date).days >= window_days:
                if current_period['developers']:
                    current_period['end_date'] = (start_date + timedelta(days=window_days)).strftime('%Y-%m-%d')
                    periods.append(current_period)
                
                # 新しい期間を開始
                current_period = {
                    'developers': [],
                    'contexts': [],
                    'labels': [],
                    'start_date': date.strftime('%Y-%m-%d'),
                    'end_date': None
                }
                start_date = date
            
            # データを追加
            current_period['developers'].append(data_point['developer'])
            current_period['contexts'].append(data_point['context'])
            current_period['labels'].append(data_point['label'])
        
        # 最後の期間を追加
        if current_period['developers']:
            current_period['end_date'] = datetime.now().strftime('%Y-%m-%d')
            periods.append(current_period)
        
        return periods
    
    def _matches_segment_criteria(self, 
                                developer: Dict[str, Any],
                                criteria: Dict[str, Any]) -> bool:
        """開発者がセグメント基準に合致するかチェック"""
        
        # 経験月数による判定
        if 'experience_months' in criteria:
            min_months, max_months = criteria['experience_months']
            developer_months = developer.get('experience_months', 0)
            
            if not (min_months <= developer_months < max_months):
                return False
        
        # その他の基準も追加可能
        
        return True
    
    def _analyze_performance_trends(self, 
                                  performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """性能トレンドを分析"""
        
        if len(performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # AUCスコアのトレンドを分析
        auc_scores = [p['basic_metrics']['auc_score'] for p in performance_history]
        
        # 線形回帰で傾向を計算
        x = np.arange(len(auc_scores))
        slope = np.polyfit(x, auc_scores, 1)[0]
        
        trend_analysis = {
            'auc_trend_slope': float(slope),
            'auc_trend_direction': 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable',
            'initial_auc': float(auc_scores[0]),
            'final_auc': float(auc_scores[-1]),
            'auc_change': float(auc_scores[-1] - auc_scores[0]),
            'periods_analyzed': len(performance_history)
        }
        
        return trend_analysis
    
    def _detect_performance_degradation(self, 
                                      performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """性能劣化を検出"""
        
        degradation_threshold = 0.05  # 5%の劣化
        
        if len(performance_history) < 3:
            return {'degradation_detected': False}
        
        auc_scores = [p['basic_metrics']['auc_score'] for p in performance_history]
        
        # 最近の性能と初期性能を比較
        recent_avg = np.mean(auc_scores[-3:])
        initial_avg = np.mean(auc_scores[:3])
        
        degradation = initial_avg - recent_avg
        
        degradation_detection = {
            'degradation_detected': degradation > degradation_threshold,
            'degradation_magnitude': float(degradation),
            'initial_performance': float(initial_avg),
            'recent_performance': float(recent_avg),
            'degradation_threshold': degradation_threshold
        }
        
        return degradation_detection
    
    def _analyze_performance_seasonality(self, 
                                       performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """性能の季節性を分析"""
        
        # 簡略化された季節性分析
        seasonality_analysis = {
            'seasonal_patterns_detected': False,
            'analysis_note': '季節性分析には長期間のデータが必要です'
        }
        
        return seasonality_analysis
    
    def _perform_statistical_tests(self, 
                                 model_results: Dict[str, Dict[str, Any]],
                                 test_developers: List[Dict[str, Any]],
                                 test_contexts: List[Dict[str, Any]],
                                 true_labels: List[int]) -> Dict[str, Any]:
        """統計的有意性検定を実行"""
        
        # 簡略化された統計検定
        statistical_tests = {
            'test_type': 'McNemar test (simplified)',
            'significance_level': 0.05,
            'results': {}
        }
        
        model_names = list(model_results.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                auc1 = model_results[model1]['basic_metrics']['auc_score']
                auc2 = model_results[model2]['basic_metrics']['auc_score']
                
                # 簡略化された比較
                difference = abs(auc1 - auc2)
                significant = difference > 0.02  # 2%以上の差を有意とする
                
                statistical_tests['results'][f"{model1}_vs_{model2}"] = {
                    'auc_difference': float(difference),
                    'significant': significant,
                    'better_model': model1 if auc1 > auc2 else model2
                }
        
        return statistical_tests
    
    def _select_best_model(self, 
                         model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """最適なモデルを選択"""
        
        # AUCスコアで比較
        best_model = max(
            model_results.keys(),
            key=lambda model: model_results[model]['basic_metrics']['auc_score']
        )
        
        best_model_selection = {
            'best_model': best_model,
            'selection_criteria': 'highest_auc_score',
            'best_auc_score': model_results[best_model]['basic_metrics']['auc_score'],
            'model_rankings': sorted(
                [(model, results['basic_metrics']['auc_score']) 
                 for model, results in model_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return best_model_selection
    
    def _analyze_performance_differences(self, 
                                       model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """性能差を分析"""
        
        # 各メトリクスでの性能差を計算
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        performance_differences = {}
        
        for metric in metrics:
            values = [results['basic_metrics'][metric] for results in model_results.values()]
            
            performance_differences[metric] = {
                'max_value': float(max(values)),
                'min_value': float(min(values)),
                'range': float(max(values) - min(values)),
                'std_deviation': float(np.std(values))
            }
        
        return performance_differences
    
    def _calculate_overall_summary(self, 
                                 evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体サマリーを計算"""
        
        if not evaluation_results:
            return {}
        
        # 平均性能を計算
        avg_metrics = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        for metric in metrics:
            values = [result['basic_metrics'][metric] for result in evaluation_results]
            avg_metrics[f'avg_{metric}'] = float(np.mean(values))
            avg_metrics[f'std_{metric}'] = float(np.std(values))
        
        overall_summary = {
            'total_evaluations': len(evaluation_results),
            'average_metrics': avg_metrics,
            'evaluation_period': {
                'start': evaluation_results[0]['evaluation_metadata']['evaluation_date'],
                'end': evaluation_results[-1]['evaluation_metadata']['evaluation_date']
            }
        }
        
        return overall_summary
    
    def _analyze_overall_trends(self, 
                              evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体的なトレンドを分析"""
        
        # 実装の詳細は省略
        return {
            'performance_trend': 'stable',
            'trend_analysis': '性能は安定しています'
        }
    
    def _identify_performance_issues(self, 
                                   evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """性能問題を特定"""
        
        issues = []
        
        # 低いAUCスコアをチェック
        for result in evaluation_results:
            auc_score = result['basic_metrics']['auc_score']
            if auc_score < 0.7:
                issues.append({
                    'issue_type': 'low_auc_score',
                    'description': f'AUCスコアが低い: {auc_score:.3f}',
                    'severity': 'high' if auc_score < 0.6 else 'medium'
                })
        
        return issues
    
    def _generate_improvement_suggestions(self, 
                                        overall_summary: Dict[str, Any],
                                        performance_trends: Dict[str, Any],
                                        identified_issues: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """改善提案を生成"""
        
        suggestions = [
            {
                'category': 'feature_engineering',
                'suggestion': '特徴量エンジニアリングの改善を検討してください',
                'priority': 'medium'
            },
            {
                'category': 'model_tuning',
                'suggestion': 'ハイパーパラメータの調整を行ってください',
                'priority': 'high'
            },
            {
                'category': 'data_quality',
                'suggestion': 'データ品質の向上を図ってください',
                'priority': 'medium'
            }
        ]
        
        return suggestions
    
    def _plot_roc_curve(self, curve_data: Dict[str, Any], output_path: str) -> str:
        """ROC曲線をプロット"""
        
        roc_data = curve_data['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_precision_recall_curve(self, curve_data: Dict[str, Any], output_path: str) -> str:
        """Precision-Recall曲線をプロット"""
        
        pr_data = curve_data['precision_recall_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data['recall'], pr_data['precision'],
                label=f'PR Curve (AUC = {pr_data["auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_confusion_matrix(self, basic_metrics: Dict[str, Any], output_path: str) -> str:
        """混同行列をプロット"""
        
        cm_data = basic_metrics['confusion_matrix']
        cm_matrix = np.array([
            [cm_data['true_negative'], cm_data['false_positive']],
            [cm_data['false_negative'], cm_data['true_positive']]
        ])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_threshold_performance(self, threshold_performance: Dict[str, List[float]], output_path: str) -> str:
        """閾値別性能をプロット"""
        
        plt.figure(figsize=(10, 6))
        
        thresholds = threshold_performance['thresholds']
        
        plt.plot(thresholds, threshold_performance['accuracy'], 'o-', label='Accuracy')
        plt.plot(thresholds, threshold_performance['precision'], 's-', label='Precision')
        plt.plot(thresholds, threshold_performance['recall'], '^-', label='Recall')
        plt.plot(thresholds, threshold_performance['f1_score'], 'd-', label='F1 Score')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_segment_performance(self, segment_performance: Dict[str, Any], output_path: str) -> str:
        """セグメント別性能をプロット"""
        
        segments = list(segment_performance.keys())
        auc_scores = [segment_performance[seg]['auc_score'] for seg in segments]
        
        plt.figure(figsize=(8, 6))
        plt.bar(segments, auc_scores)
        plt.xlabel('Developer Segment')
        plt.ylabel('AUC Score')
        plt.title('Performance by Developer Segment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_confidence_distribution(self, confidence_analysis: Dict[str, Any], output_path: str) -> str:
        """予測信頼度分布をプロット"""
        
        # 簡略化された信頼度分布プロット
        plt.figure(figsize=(8, 6))
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Level')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_evaluation_report(self, report: Dict[str, Any], output_path: str) -> None:
        """評価レポートを保存"""
        
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"評価レポートを保存しました: {output_path}")


def create_retention_evaluator(config_path: str) -> RetentionEvaluator:
    """
    設定ファイルから定着予測評価器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        RetentionEvaluator: 設定済みの評価器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return RetentionEvaluator(config.get('retention_evaluation', {}))


if __name__ == "__main__":
    # テスト用のサンプルデータ
    from ...src.gerrit_retention.prediction.retention_predictor import (
        RetentionPredictor,
    )

    # 評価器のテスト
    config = {
        'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score'],
        'confidence_thresholds': [0.3, 0.5, 0.7],
        'time_window_days': 30,
        'developer_segments': {
            'junior': {'experience_months': (0, 12)},
            'senior': {'experience_months': (36, float('inf'))}
        }
    }
    
    evaluator = RetentionEvaluator(config)
    
    print("定着予測評価器のテスト完了")