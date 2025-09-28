#!/usr/bin/env python3
"""
A/Bテスト統計分析システム

このモジュールは、A/Bテスト実験の結果を統計的に分析し、
有意性検定、効果サイズ計算、信頼区間推定などを実行する。

要件: 6.3
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.utils.config_manager import ConfigManager
from gerrit_retention.utils.logger import setup_logger

from .experiment_design import ExperimentConfig, ExperimentMetric, ExperimentVariant


class TestResult(Enum):
    """検定結果"""
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StatisticalTestResult:
    """統計検定結果"""
    metric_id: str
    test_type: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    result: TestResult
    power: float
    sample_size: int
    interpretation: str


@dataclass
class VariantPerformance:
    """バリアント性能"""
    variant_id: str
    variant_name: str
    sample_size: int
    metric_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    conversion_rates: Dict[str, float]


@dataclass
class ExperimentAnalysisResult:
    """実験分析結果"""
    experiment_id: str
    analysis_timestamp: datetime
    experiment_duration_days: int
    total_participants: int
    variant_performances: List[VariantPerformance]
    statistical_tests: List[StatisticalTestResult]
    recommendations: List[str]
    overall_conclusion: str
    data_quality_issues: List[str]


class StatisticalAnalyzer:
    """統計分析器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        統計分析器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 分析結果保存ディレクトリ
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # プロット設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def analyze_experiment(
        self,
        experiment_config: ExperimentConfig,
        experiment_data: pd.DataFrame
    ) -> ExperimentAnalysisResult:
        """
        実験結果を分析
        
        Args:
            experiment_config: 実験設定
            experiment_data: 実験データ
            
        Returns:
            分析結果
        """
        self.logger.info(f"実験分析を開始: {experiment_config.experiment_id}")
        
        # データ品質チェック
        data_quality_issues = self._check_data_quality(experiment_data, experiment_config)
        
        # バリアント性能計算
        variant_performances = self._calculate_variant_performances(
            experiment_data, experiment_config
        )
        
        # 統計検定実行
        statistical_tests = self._run_statistical_tests(
            experiment_data, experiment_config, variant_performances
        )
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(
            statistical_tests, variant_performances, experiment_config
        )
        
        # 全体結論
        overall_conclusion = self._generate_overall_conclusion(
            statistical_tests, variant_performances
        )
        
        # 実験期間計算
        experiment_duration = (experiment_config.end_date - experiment_config.start_date).days
        
        # 分析結果作成
        analysis_result = ExperimentAnalysisResult(
            experiment_id=experiment_config.experiment_id,
            analysis_timestamp=datetime.now(),
            experiment_duration_days=experiment_duration,
            total_participants=len(experiment_data),
            variant_performances=variant_performances,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            overall_conclusion=overall_conclusion,
            data_quality_issues=data_quality_issues
        )
        
        self.logger.info(f"実験分析完了: {experiment_config.experiment_id}")
        return analysis_result
        
    def _check_data_quality(
        self,
        data: pd.DataFrame,
        config: ExperimentConfig
    ) -> List[str]:
        """データ品質をチェック"""
        issues = []
        
        # 必要カラムの存在チェック
        required_columns = ['participant_id', 'variant_id', 'timestamp']
        for metric in config.metrics:
            required_columns.append(metric.metric_id)
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"必要なカラムが不足: {missing_columns}")
            
        # サンプルサイズチェック
        if len(data) < config.minimum_sample_size:
            issues.append(f"サンプルサイズが不足: {len(data)} < {config.minimum_sample_size}")
            
        # バリアント別サンプルサイズチェック
        variant_counts = data['variant_id'].value_counts()
        for variant in config.variants:
            count = variant_counts.get(variant.variant_id, 0)
            expected_min = variant.expected_participants * 0.8  # 80%以上
            if count < expected_min:
                issues.append(f"バリアント {variant.variant_id} のサンプルサイズが不足: {count} < {expected_min}")
                
        # 欠損値チェック
        for metric in config.metrics:
            if metric.metric_id in data.columns:
                missing_rate = data[metric.metric_id].isnull().mean()
                if missing_rate > 0.1:  # 10%以上の欠損
                    issues.append(f"メトリクス {metric.metric_id} の欠損率が高い: {missing_rate:.2%}")
                    
        # 外れ値チェック
        for metric in config.metrics:
            if metric.metric_id in data.columns and metric.metric_type == 'continuous':
                values = data[metric.metric_id].dropna()
                if len(values) > 0:
                    q1, q3 = values.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
                    outlier_rate = len(outliers) / len(values)
                    if outlier_rate > 0.05:  # 5%以上の外れ値
                        issues.append(f"メトリクス {metric.metric_id} の外れ値率が高い: {outlier_rate:.2%}")
                        
        return issues
        
    def _calculate_variant_performances(
        self,
        data: pd.DataFrame,
        config: ExperimentConfig
    ) -> List[VariantPerformance]:
        """バリアント性能を計算"""
        performances = []
        
        for variant in config.variants:
            variant_data = data[data['variant_id'] == variant.variant_id]
            
            if len(variant_data) == 0:
                self.logger.warning(f"バリアント {variant.variant_id} にデータがありません")
                continue
                
            metric_values = {}
            confidence_intervals = {}
            conversion_rates = {}
            
            for metric in config.metrics:
                if metric.metric_id not in variant_data.columns:
                    continue
                    
                values = variant_data[metric.metric_id].dropna()
                
                if len(values) == 0:
                    continue
                    
                if metric.metric_type == 'binary':
                    # バイナリメトリクス（変換率）
                    conversion_rate = values.mean()
                    n = len(values)
                    
                    # 信頼区間計算（Wilson score interval）
                    ci_low, ci_high = proportion_confint(
                        conversion_rate * n, n, method='wilson'
                    )
                    
                    metric_values[metric.metric_id] = conversion_rate
                    confidence_intervals[metric.metric_id] = (ci_low, ci_high)
                    conversion_rates[metric.metric_id] = conversion_rate
                    
                elif metric.metric_type in ['continuous', 'count']:
                    # 連続値メトリクス
                    mean_value = values.mean()
                    std_error = values.std() / np.sqrt(len(values))
                    
                    # 95%信頼区間
                    ci_low = mean_value - 1.96 * std_error
                    ci_high = mean_value + 1.96 * std_error
                    
                    metric_values[metric.metric_id] = mean_value
                    confidence_intervals[metric.metric_id] = (ci_low, ci_high)
                    
                elif metric.metric_type == 'rate':
                    # 率メトリクス（0-1の範囲）
                    rate_value = values.mean()
                    n = len(values)
                    
                    # 信頼区間計算
                    ci_low, ci_high = proportion_confint(
                        rate_value * n, n, method='wilson'
                    )
                    
                    metric_values[metric.metric_id] = rate_value
                    confidence_intervals[metric.metric_id] = (ci_low, ci_high)
                    conversion_rates[metric.metric_id] = rate_value
                    
            performance = VariantPerformance(
                variant_id=variant.variant_id,
                variant_name=variant.name,
                sample_size=len(variant_data),
                metric_values=metric_values,
                confidence_intervals=confidence_intervals,
                conversion_rates=conversion_rates
            )
            
            performances.append(performance)
            
        return performances
        
    def _run_statistical_tests(
        self,
        data: pd.DataFrame,
        config: ExperimentConfig,
        performances: List[VariantPerformance]
    ) -> List[StatisticalTestResult]:
        """統計検定を実行"""
        test_results = []
        
        if len(performances) < 2:
            self.logger.warning("統計検定には最低2つのバリアントが必要です")
            return test_results
            
        # 制御群（最初のバリアント）と各処理群を比較
        control_variant = performances[0]
        
        for treatment_variant in performances[1:]:
            for metric in config.metrics:
                if metric.metric_id not in control_variant.metric_values:
                    continue
                if metric.metric_id not in treatment_variant.metric_values:
                    continue
                    
                # データ取得
                control_data = data[
                    (data['variant_id'] == control_variant.variant_id) &
                    (data[metric.metric_id].notna())
                ][metric.metric_id]
                
                treatment_data = data[
                    (data['variant_id'] == treatment_variant.variant_id) &
                    (data[metric.metric_id].notna())
                ][metric.metric_id]
                
                if len(control_data) == 0 or len(treatment_data) == 0:
                    continue
                    
                # メトリクスタイプに応じた検定
                if metric.metric_type in ['binary', 'rate']:
                    test_result = self._run_proportion_test(
                        control_data, treatment_data, metric, 
                        control_variant.variant_name, treatment_variant.variant_name
                    )
                elif metric.metric_type in ['continuous', 'count']:
                    test_result = self._run_continuous_test(
                        control_data, treatment_data, metric,
                        control_variant.variant_name, treatment_variant.variant_name
                    )
                else:
                    continue
                    
                if test_result:
                    test_results.append(test_result)
                    
        return test_results
        
    def _run_proportion_test(
        self,
        control_data: pd.Series,
        treatment_data: pd.Series,
        metric: ExperimentMetric,
        control_name: str,
        treatment_name: str
    ) -> Optional[StatisticalTestResult]:
        """比率検定を実行"""
        try:
            # 成功数とサンプルサイズ
            control_successes = int(control_data.sum())
            control_n = len(control_data)
            treatment_successes = int(treatment_data.sum())
            treatment_n = len(treatment_data)
            
            # Z検定
            counts = np.array([control_successes, treatment_successes])
            nobs = np.array([control_n, treatment_n])
            
            z_stat, p_value = proportions_ztest(counts, nobs)
            
            # 効果サイズ（Cohen's h）
            control_rate = control_successes / control_n
            treatment_rate = treatment_successes / treatment_n
            
            effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
            
            # 信頼区間（差の信頼区間）
            rate_diff = treatment_rate - control_rate
            se_diff = np.sqrt(
                control_rate * (1 - control_rate) / control_n +
                treatment_rate * (1 - treatment_rate) / treatment_n
            )
            ci_low = rate_diff - 1.96 * se_diff
            ci_high = rate_diff + 1.96 * se_diff
            
            # 検出力計算
            power = self._calculate_proportion_power(
                control_rate, treatment_rate, control_n, treatment_n
            )
            
            # 結果判定
            alpha = 0.05
            if p_value < alpha:
                result = TestResult.SIGNIFICANT
            else:
                result = TestResult.NOT_SIGNIFICANT
                
            # 解釈生成
            interpretation = self._generate_proportion_interpretation(
                control_rate, treatment_rate, p_value, effect_size,
                control_name, treatment_name, metric
            )
            
            return StatisticalTestResult(
                metric_id=metric.metric_id,
                test_type="proportion_z_test",
                statistic=z_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_low, ci_high),
                result=result,
                power=power,
                sample_size=control_n + treatment_n,
                interpretation=interpretation
            )
            
        except Exception as e:
            self.logger.error(f"比率検定でエラー: {e}")
            return None
            
    def _run_continuous_test(
        self,
        control_data: pd.Series,
        treatment_data: pd.Series,
        metric: ExperimentMetric,
        control_name: str,
        treatment_name: str
    ) -> Optional[StatisticalTestResult]:
        """連続値検定を実行"""
        try:
            # 正規性検定
            control_normal = stats.shapiro(control_data.sample(min(5000, len(control_data))))[1] > 0.05
            treatment_normal = stats.shapiro(treatment_data.sample(min(5000, len(treatment_data))))[1] > 0.05
            
            # 等分散性検定
            equal_var = stats.levene(control_data, treatment_data)[1] > 0.05
            
            if control_normal and treatment_normal:
                # t検定
                t_stat, p_value = ttest_ind(
                    treatment_data, control_data, equal_var=equal_var
                )
                test_type = "welch_t_test" if not equal_var else "student_t_test"
                
                # 効果サイズ（Cohen's d）
                pooled_std = np.sqrt(
                    ((len(control_data) - 1) * control_data.var() +
                     (len(treatment_data) - 1) * treatment_data.var()) /
                    (len(control_data) + len(treatment_data) - 2)
                )
                effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
                
            else:
                # Mann-Whitney U検定
                u_stat, p_value = mannwhitneyu(
                    treatment_data, control_data, alternative='two-sided'
                )
                t_stat = u_stat
                test_type = "mann_whitney_u_test"
                
                # 効果サイズ（r = Z / sqrt(N)）
                z_score = stats.norm.ppf(1 - p_value / 2)
                effect_size = z_score / np.sqrt(len(control_data) + len(treatment_data))
                
            # 信頼区間（平均差の信頼区間）
            mean_diff = treatment_data.mean() - control_data.mean()
            se_diff = np.sqrt(
                control_data.var() / len(control_data) +
                treatment_data.var() / len(treatment_data)
            )
            ci_low = mean_diff - 1.96 * se_diff
            ci_high = mean_diff + 1.96 * se_diff
            
            # 検出力計算
            power = ttest_power(
                effect_size, len(control_data) + len(treatment_data), 0.05
            )
            
            # 結果判定
            alpha = 0.05
            if p_value < alpha:
                result = TestResult.SIGNIFICANT
            else:
                result = TestResult.NOT_SIGNIFICANT
                
            # 解釈生成
            interpretation = self._generate_continuous_interpretation(
                control_data.mean(), treatment_data.mean(), p_value, effect_size,
                control_name, treatment_name, metric
            )
            
            return StatisticalTestResult(
                metric_id=metric.metric_id,
                test_type=test_type,
                statistic=t_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_low, ci_high),
                result=result,
                power=power,
                sample_size=len(control_data) + len(treatment_data),
                interpretation=interpretation
            )
            
        except Exception as e:
            self.logger.error(f"連続値検定でエラー: {e}")
            return None 
           
    def _calculate_proportion_power(
        self,
        control_rate: float,
        treatment_rate: float,
        control_n: int,
        treatment_n: int,
        alpha: float = 0.05
    ) -> float:
        """比率検定の検出力を計算"""
        try:
            # 効果サイズ計算
            effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
            
            # 検出力計算（近似）
            pooled_rate = (control_rate * control_n + treatment_rate * treatment_n) / (control_n + treatment_n)
            se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_n + 1/treatment_n))
            
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = (abs(treatment_rate - control_rate) - z_alpha * se) / se
            
            power = stats.norm.cdf(z_beta)
            return max(0, min(1, power))
            
        except:
            return 0.5  # デフォルト値
            
    def _generate_proportion_interpretation(
        self,
        control_rate: float,
        treatment_rate: float,
        p_value: float,
        effect_size: float,
        control_name: str,
        treatment_name: str,
        metric: ExperimentMetric
    ) -> str:
        """比率検定の解釈を生成"""
        rate_diff = treatment_rate - control_rate
        relative_change = (rate_diff / control_rate) * 100 if control_rate > 0 else 0
        
        significance = "統計的に有意" if p_value < 0.05 else "統計的に有意でない"
        
        effect_magnitude = "小さい"
        if abs(effect_size) > 0.2:
            effect_magnitude = "中程度"
        if abs(effect_size) > 0.5:
            effect_magnitude = "大きい"
            
        direction = "向上" if rate_diff > 0 else "低下"
        
        interpretation = (
            f"{treatment_name}は{control_name}と比較して、"
            f"{metric.name}が{abs(relative_change):.1f}%{direction}しました "
            f"({control_rate:.3f} → {treatment_rate:.3f})。"
            f"この差は{significance}です (p={p_value:.4f})。"
            f"効果サイズは{effect_magnitude}です (h={effect_size:.3f})。"
        )
        
        return interpretation
        
    def _generate_continuous_interpretation(
        self,
        control_mean: float,
        treatment_mean: float,
        p_value: float,
        effect_size: float,
        control_name: str,
        treatment_name: str,
        metric: ExperimentMetric
    ) -> str:
        """連続値検定の解釈を生成"""
        mean_diff = treatment_mean - control_mean
        relative_change = (mean_diff / control_mean) * 100 if control_mean != 0 else 0
        
        significance = "統計的に有意" if p_value < 0.05 else "統計的に有意でない"
        
        effect_magnitude = "小さい"
        if abs(effect_size) > 0.5:
            effect_magnitude = "中程度"
        if abs(effect_size) > 0.8:
            effect_magnitude = "大きい"
            
        direction = "向上" if mean_diff > 0 else "低下"
        if not metric.higher_is_better:
            direction = "低下" if mean_diff > 0 else "向上"
            
        interpretation = (
            f"{treatment_name}は{control_name}と比較して、"
            f"{metric.name}が{abs(relative_change):.1f}%{direction}しました "
            f"({control_mean:.3f} → {treatment_mean:.3f})。"
            f"この差は{significance}です (p={p_value:.4f})。"
            f"効果サイズは{effect_magnitude}です (d={effect_size:.3f})。"
        )
        
        return interpretation
        
    def _generate_recommendations(
        self,
        test_results: List[StatisticalTestResult],
        performances: List[VariantPerformance],
        config: ExperimentConfig
    ) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        # プライマリメトリクスの結果に基づく推奨
        primary_metric = next((m for m in config.metrics if m.primary), None)
        if primary_metric:
            primary_tests = [t for t in test_results if t.metric_id == primary_metric.metric_id]
            
            if primary_tests:
                best_test = max(primary_tests, key=lambda t: abs(t.effect_size))
                
                if best_test.result == TestResult.SIGNIFICANT:
                    if best_test.effect_size > 0:
                        recommendations.append(
                            f"プライマリメトリクス（{primary_metric.name}）で統計的に有意な改善が確認されました。"
                            f"新しい戦略の導入を推奨します。"
                        )
                    else:
                        recommendations.append(
                            f"プライマリメトリクス（{primary_metric.name}）で統計的に有意な悪化が確認されました。"
                            f"現在の戦略を維持することを推奨します。"
                        )
                else:
                    if best_test.power < 0.8:
                        recommendations.append(
                            f"プライマリメトリクスで有意差は確認されませんでしたが、"
                            f"検出力が不足している可能性があります（power={best_test.power:.2f}）。"
                            f"サンプルサイズを増やして再実験することを推奨します。"
                        )
                    else:
                        recommendations.append(
                            f"プライマリメトリクスで有意差は確認されませんでした。"
                            f"現在の戦略を維持するか、別のアプローチを検討してください。"
                        )
                        
        # セカンダリメトリクスの結果
        secondary_significant = [
            t for t in test_results 
            if not any(m.metric_id == t.metric_id and m.primary for m in config.metrics)
            and t.result == TestResult.SIGNIFICANT
        ]
        
        if secondary_significant:
            recommendations.append(
                f"セカンダリメトリクスで{len(secondary_significant)}個の有意な変化が確認されました。"
                f"詳細な分析を行い、副次効果を評価してください。"
            )
            
        # 効果サイズに基づく推奨
        large_effects = [t for t in test_results if abs(t.effect_size) > 0.5]
        if large_effects:
            recommendations.append(
                f"{len(large_effects)}個のメトリクスで大きな効果サイズが確認されました。"
                f"実用的に意味のある変化である可能性が高いです。"
            )
            
        # サンプルサイズに関する推奨
        total_participants = sum(p.sample_size for p in performances)
        if total_participants < config.minimum_sample_size:
            recommendations.append(
                f"サンプルサイズが目標値を下回っています（{total_participants} < {config.minimum_sample_size}）。"
                f"結果の信頼性を高めるため、実験期間の延長を検討してください。"
            )
            
        # バリアント間のバランスチェック
        sample_sizes = [p.sample_size for p in performances]
        if len(sample_sizes) > 1:
            cv = np.std(sample_sizes) / np.mean(sample_sizes)
            if cv > 0.2:  # 変動係数が20%以上
                recommendations.append(
                    f"バリアント間のサンプルサイズに大きな偏りがあります（CV={cv:.2f}）。"
                    f"割り当てアルゴリズムの見直しを検討してください。"
                )
                
        return recommendations
        
    def _generate_overall_conclusion(
        self,
        test_results: List[StatisticalTestResult],
        performances: List[VariantPerformance]
    ) -> str:
        """全体結論を生成"""
        if not test_results:
            return "統計検定を実行できませんでした。データの品質を確認してください。"
            
        significant_tests = [t for t in test_results if t.result == TestResult.SIGNIFICANT]
        total_tests = len(test_results)
        
        if len(significant_tests) == 0:
            conclusion = (
                f"実行された{total_tests}個の統計検定のうち、統計的に有意な結果は確認されませんでした。"
                f"現在の戦略を維持するか、実験設計の見直しを検討してください。"
            )
        elif len(significant_tests) == total_tests:
            conclusion = (
                f"実行された{total_tests}個の統計検定すべてで統計的に有意な結果が確認されました。"
                f"新しい戦略の効果は明確であり、導入を強く推奨します。"
            )
        else:
            conclusion = (
                f"実行された{total_tests}個の統計検定のうち{len(significant_tests)}個で"
                f"統計的に有意な結果が確認されました。"
                f"メトリクス別の詳細分析を行い、総合的な判断を行ってください。"
            )
            
        return conclusion
        
    def generate_analysis_report(
        self,
        analysis_result: ExperimentAnalysisResult,
        output_path: Optional[str] = None
    ) -> str:
        """分析レポートを生成"""
        if output_path is None:
            output_path = self.results_dir / f"{analysis_result.experiment_id}_analysis_report.html"
        else:
            output_path = Path(output_path)
            
        # HTMLレポート生成
        html_content = self._generate_html_report(analysis_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"分析レポートを生成しました: {output_path}")
        return str(output_path)
        
    def _generate_html_report(self, result: ExperimentAnalysisResult) -> str:
        """HTML分析レポートを生成"""
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A/Bテスト分析レポート - {result.experiment_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metric-card {{ border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .significant {{ border-left: 4px solid #28a745; }}
        .not-significant {{ border-left: 4px solid #6c757d; }}
        .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; }}
        .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .number {{ font-family: 'Courier New', monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>A/Bテスト分析レポート</h1>
        <p><strong>実験ID:</strong> {result.experiment_id}</p>
        <p><strong>分析日時:</strong> {result.analysis_timestamp.strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p><strong>実験期間:</strong> {result.experiment_duration_days}日間</p>
        <p><strong>総参加者数:</strong> {result.total_participants:,}人</p>
    </div>
    
    <div class="section">
        <h2>全体結論</h2>
        <div class="{'success' if '推奨' in result.overall_conclusion else 'warning'}">
            {result.overall_conclusion}
        </div>
    </div>
    
    <div class="section">
        <h2>バリアント性能</h2>
        <table>
            <thead>
                <tr>
                    <th>バリアント</th>
                    <th>サンプルサイズ</th>
                    <th>主要メトリクス</th>
                    <th>信頼区間</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for performance in result.variant_performances:
            # 主要メトリクスを取得
            primary_metric_value = "N/A"
            primary_ci = "N/A"
            
            if performance.metric_values:
                first_metric = list(performance.metric_values.keys())[0]
                primary_metric_value = f"{performance.metric_values[first_metric]:.4f}"
                
                if first_metric in performance.confidence_intervals:
                    ci_low, ci_high = performance.confidence_intervals[first_metric]
                    primary_ci = f"[{ci_low:.4f}, {ci_high:.4f}]"
                    
            html += f"""
                <tr>
                    <td><strong>{performance.variant_name}</strong></td>
                    <td class="number">{performance.sample_size:,}</td>
                    <td class="number">{primary_metric_value}</td>
                    <td class="number">{primary_ci}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>統計検定結果</h2>
"""
        
        for test in result.statistical_tests:
            significance_class = "significant" if test.result == TestResult.SIGNIFICANT else "not-significant"
            
            html += f"""
        <div class="metric-card {significance_class}">
            <h3>{test.metric_id}</h3>
            <p><strong>検定方法:</strong> {test.test_type}</p>
            <p><strong>検定統計量:</strong> <span class="number">{test.statistic:.4f}</span></p>
            <p><strong>p値:</strong> <span class="number">{test.p_value:.6f}</span></p>
            <p><strong>効果サイズ:</strong> <span class="number">{test.effect_size:.4f}</span></p>
            <p><strong>信頼区間:</strong> <span class="number">[{test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f}]</span></p>
            <p><strong>検出力:</strong> <span class="number">{test.power:.3f}</span></p>
            <p><strong>結果:</strong> {test.result.value}</p>
            <p><strong>解釈:</strong> {test.interpretation}</p>
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>推奨事項</h2>
        <ul>
"""
        
        for recommendation in result.recommendations:
            html += f"            <li>{recommendation}</li>\n"
            
        html += """
        </ul>
    </div>
"""
        
        if result.data_quality_issues:
            html += """
    <div class="section">
        <h2>データ品質の問題</h2>
        <div class="warning">
            <ul>
"""
            for issue in result.data_quality_issues:
                html += f"                <li>{issue}</li>\n"
                
            html += """
            </ul>
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html
        
    def create_visualization_plots(
        self,
        analysis_result: ExperimentAnalysisResult,
        experiment_data: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """可視化プロットを作成"""
        if output_dir is None:
            output_dir = self.results_dir / f"{analysis_result.experiment_id}_plots"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # 1. バリアント性能比較プロット
        performance_plot = self._create_performance_comparison_plot(
            analysis_result, output_dir
        )
        if performance_plot:
            plot_files.append(performance_plot)
            
        # 2. 効果サイズプロット
        effect_size_plot = self._create_effect_size_plot(
            analysis_result, output_dir
        )
        if effect_size_plot:
            plot_files.append(effect_size_plot)
            
        # 3. 信頼区間プロット
        confidence_interval_plot = self._create_confidence_interval_plot(
            analysis_result, output_dir
        )
        if confidence_interval_plot:
            plot_files.append(confidence_interval_plot)
            
        # 4. 時系列プロット（データに時間情報がある場合）
        if 'timestamp' in experiment_data.columns:
            time_series_plot = self._create_time_series_plot(
                experiment_data, analysis_result, output_dir
            )
            if time_series_plot:
                plot_files.append(time_series_plot)
                
        self.logger.info(f"{len(plot_files)}個の可視化プロットを作成しました")
        return plot_files
        
    def _create_performance_comparison_plot(
        self,
        result: ExperimentAnalysisResult,
        output_dir: Path
    ) -> Optional[str]:
        """バリアント性能比較プロットを作成"""
        try:
            if not result.variant_performances:
                return None
                
            # 最初のメトリクスを使用
            first_performance = result.variant_performances[0]
            if not first_performance.metric_values:
                return None
                
            metric_id = list(first_performance.metric_values.keys())[0]
            
            # データ準備
            variants = []
            values = []
            ci_lows = []
            ci_highs = []
            
            for performance in result.variant_performances:
                if metric_id in performance.metric_values:
                    variants.append(performance.variant_name)
                    values.append(performance.metric_values[metric_id])
                    
                    if metric_id in performance.confidence_intervals:
                        ci_low, ci_high = performance.confidence_intervals[metric_id]
                        ci_lows.append(ci_low)
                        ci_highs.append(ci_high)
                    else:
                        ci_lows.append(performance.metric_values[metric_id])
                        ci_highs.append(performance.metric_values[metric_id])
                        
            # プロット作成
            plt.figure(figsize=(10, 6))
            
            x_pos = np.arange(len(variants))
            bars = plt.bar(x_pos, values, alpha=0.7, capsize=5)
            
            # エラーバー追加
            errors = [[v - ci_low for v, ci_low in zip(values, ci_lows)],
                     [ci_high - v for v, ci_high in zip(values, ci_highs)]]
            plt.errorbar(x_pos, values, yerr=errors, fmt='none', color='black', capsize=5)
            
            # 色分け（最初のバリアントを制御群として青、他を赤）
            for i, bar in enumerate(bars):
                if i == 0:
                    bar.set_color('steelblue')
                else:
                    bar.set_color('coral')
                    
            plt.xlabel('バリアント')
            plt.ylabel(metric_id)
            plt.title(f'バリアント性能比較: {metric_id}')
            plt.xticks(x_pos, variants, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # 保存
            plot_file = output_dir / f"performance_comparison_{metric_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.error(f"性能比較プロット作成でエラー: {e}")
            return None
            
    def _create_effect_size_plot(
        self,
        result: ExperimentAnalysisResult,
        output_dir: Path
    ) -> Optional[str]:
        """効果サイズプロットを作成"""
        try:
            if not result.statistical_tests:
                return None
                
            # データ準備
            metrics = []
            effect_sizes = []
            p_values = []
            
            for test in result.statistical_tests:
                metrics.append(test.metric_id)
                effect_sizes.append(test.effect_size)
                p_values.append(test.p_value)
                
            # プロット作成
            plt.figure(figsize=(10, 6))
            
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars = plt.barh(metrics, effect_sizes, color=colors, alpha=0.7)
            
            # 効果サイズの基準線
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='小さい効果')
            plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='中程度の効果')
            plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='大きい効果')
            plt.axvline(x=-0.2, color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
            plt.axvline(x=-0.8, color='red', linestyle='--', alpha=0.5)
            
            plt.xlabel('効果サイズ')
            plt.ylabel('メトリクス')
            plt.title('効果サイズ比較')
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # 保存
            plot_file = output_dir / "effect_sizes.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.error(f"効果サイズプロット作成でエラー: {e}")
            return None
            
    def _create_confidence_interval_plot(
        self,
        result: ExperimentAnalysisResult,
        output_dir: Path
    ) -> Optional[str]:
        """信頼区間プロットを作成"""
        try:
            if not result.statistical_tests:
                return None
                
            # データ準備
            metrics = []
            effect_sizes = []
            ci_lows = []
            ci_highs = []
            significant = []
            
            for test in result.statistical_tests:
                metrics.append(test.metric_id)
                effect_sizes.append(test.effect_size)
                ci_lows.append(test.confidence_interval[0])
                ci_highs.append(test.confidence_interval[1])
                significant.append(test.result == TestResult.SIGNIFICANT)
                
            # プロット作成
            plt.figure(figsize=(10, 8))
            
            y_pos = np.arange(len(metrics))
            
            for i, (metric, effect, ci_low, ci_high, sig) in enumerate(
                zip(metrics, effect_sizes, ci_lows, ci_highs, significant)
            ):
                color = 'red' if sig else 'gray'
                plt.errorbar(effect, i, xerr=[[effect - ci_low], [ci_high - effect]], 
                           fmt='o', color=color, capsize=5, capthick=2)
                
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('効果サイズ（95%信頼区間）')
            plt.ylabel('メトリクス')
            plt.title('効果サイズの信頼区間')
            plt.yticks(y_pos, metrics)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # 保存
            plot_file = output_dir / "confidence_intervals.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.error(f"信頼区間プロット作成でエラー: {e}")
            return None
            
    def _create_time_series_plot(
        self,
        data: pd.DataFrame,
        result: ExperimentAnalysisResult,
        output_dir: Path
    ) -> Optional[str]:
        """時系列プロットを作成"""
        try:
            # 時間列を日時型に変換
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # 最初のメトリクスを使用
            if not result.variant_performances:
                return None
                
            first_performance = result.variant_performances[0]
            if not first_performance.metric_values:
                return None
                
            metric_id = list(first_performance.metric_values.keys())[0]
            
            if metric_id not in data.columns:
                return None
                
            # 日別集計
            daily_data = data.groupby(['variant_id', data['timestamp'].dt.date])[metric_id].mean().reset_index()
            daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
            
            # プロット作成
            plt.figure(figsize=(12, 6))
            
            for variant_id in daily_data['variant_id'].unique():
                variant_data = daily_data[daily_data['variant_id'] == variant_id]
                plt.plot(variant_data['timestamp'], variant_data[metric_id], 
                        marker='o', label=variant_id, alpha=0.7)
                
            plt.xlabel('日付')
            plt.ylabel(metric_id)
            plt.title(f'時系列推移: {metric_id}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存
            plot_file = output_dir / f"time_series_{metric_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.error(f"時系列プロット作成でエラー: {e}")
            return None
            
    def save_analysis_results(
        self,
        analysis_result: ExperimentAnalysisResult,
        output_path: Optional[str] = None
    ) -> str:
        """分析結果をJSONファイルに保存"""
        if output_path is None:
            output_path = self.results_dir / f"{analysis_result.experiment_id}_results.json"
        else:
            output_path = Path(output_path)
            
        # 結果を辞書に変換
        result_dict = asdict(analysis_result)
        
        # datetime オブジェクトを文字列に変換
        result_dict['analysis_timestamp'] = analysis_result.analysis_timestamp.isoformat()
        
        # Enum を文字列に変換
        for test in result_dict['statistical_tests']:
            test['result'] = test['result'].value if hasattr(test['result'], 'value') else str(test['result'])
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"分析結果を保存しました: {output_path}")
        return str(output_path)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A/Bテスト統計分析システム')
    parser.add_argument('--experiment-id', type=str, required=True, help='実験ID')
    parser.add_argument('--data', type=str, required=True, help='実験データファイル（CSV）')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--output', type=str, help='出力ディレクトリ')
    parser.add_argument('--create-plots', action='store_true', help='可視化プロットを作成')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 統計分析器初期化
    analyzer = StatisticalAnalyzer(args.config)
    
    # 実験設定読み込み
    from .experiment_design import ExperimentDesigner
    designer = ExperimentDesigner(args.config)
    experiment_config = designer.load_experiment_config(args.experiment_id)
    
    # 実験データ読み込み
    experiment_data = pd.read_csv(args.data)
    
    # 分析実行
    analysis_result = analyzer.analyze_experiment(experiment_config, experiment_data)
    
    # 結果保存
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON結果保存
        json_file = analyzer.save_analysis_results(
            analysis_result, output_dir / f"{args.experiment_id}_results.json"
        )
        
        # HTMLレポート生成
        html_file = analyzer.generate_analysis_report(
            analysis_result, output_dir / f"{args.experiment_id}_report.html"
        )
        
        # 可視化プロット作成
        if args.create_plots:
            plot_files = analyzer.create_visualization_plots(
                analysis_result, experiment_data, output_dir / "plots"
            )
            print(f"可視化プロット: {len(plot_files)}個作成")
            
        print(f"分析完了:")
        print(f"  JSON結果: {json_file}")
        print(f"  HTMLレポート: {html_file}")
        
    else:
        # 結果をコンソール出力
        print(f"\n=== 実験分析結果: {analysis_result.experiment_id} ===")
        print(f"総参加者数: {analysis_result.total_participants:,}")
        print(f"実験期間: {analysis_result.experiment_duration_days}日")
        print(f"\n全体結論:")
        print(analysis_result.overall_conclusion)
        
        print(f"\n統計検定結果:")
        for test in analysis_result.statistical_tests:
            print(f"  {test.metric_id}: {test.result.value} (p={test.p_value:.4f}, effect={test.effect_size:.3f})")
            
        print(f"\n推奨事項:")
        for i, rec in enumerate(analysis_result.recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == '__main__':
    main()