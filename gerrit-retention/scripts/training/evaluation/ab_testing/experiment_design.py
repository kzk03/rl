#!/usr/bin/env python3
"""
A/Bテスト実験設計システム

このモジュールは、異なる推薦戦略の比較実験を設計・実行する。
統計的に有意な結果を得るための実験設計と実行管理を提供する。

要件: 6.3
"""

import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.utils.config_manager import ConfigManager
from gerrit_retention.utils.logger import setup_logger


class ExperimentStatus(Enum):
    """実験ステータス"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationMethod(Enum):
    """割り当て方法"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    STRATIFIED = "stratified"
    BALANCED = "balanced"


@dataclass
class ExperimentVariant:
    """実験バリアント"""
    variant_id: str
    name: str
    description: str
    config: Dict[str, Any]
    allocation_percentage: float
    expected_participants: int
    
    def __post_init__(self):
        """バリデーション"""
        if not 0 < self.allocation_percentage <= 100:
            raise ValueError("allocation_percentage must be between 0 and 100")


@dataclass
class ExperimentMetric:
    """実験メトリクス"""
    metric_id: str
    name: str
    description: str
    metric_type: str  # 'binary', 'continuous', 'count', 'rate'
    primary: bool = False
    higher_is_better: bool = True
    minimum_detectable_effect: float = 0.05
    baseline_value: Optional[float] = None
    
    def __post_init__(self):
        """バリデーション"""
        valid_types = ['binary', 'continuous', 'count', 'rate']
        if self.metric_type not in valid_types:
            raise ValueError(f"metric_type must be one of {valid_types}")


@dataclass
class ExperimentConfig:
    """実験設定"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    allocation_method: AllocationMethod
    start_date: datetime
    end_date: datetime
    minimum_sample_size: int
    significance_level: float = 0.05
    power: float = 0.8
    stratification_keys: Optional[List[str]] = None
    exclusion_criteria: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """バリデーション"""
        # 割り当て比率の合計チェック
        total_allocation = sum(v.allocation_percentage for v in self.variants)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError(f"Total allocation must be 100%, got {total_allocation}%")
            
        # 日付チェック
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
            
        # プライマリメトリクスチェック
        primary_metrics = [m for m in self.metrics if m.primary]
        if len(primary_metrics) != 1:
            raise ValueError("Exactly one primary metric must be specified")


class ExperimentDesigner:
    """実験設計器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        実験設計器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 実験データ保存ディレクトリ
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
    def design_retention_strategy_experiment(
        self,
        experiment_name: str,
        strategies: List[Dict[str, Any]],
        duration_days: int = 30,
        target_participants: int = 1000
    ) -> ExperimentConfig:
        """
        定着戦略比較実験を設計
        
        Args:
            experiment_name: 実験名
            strategies: 比較する戦略のリスト
            duration_days: 実験期間（日数）
            target_participants: 目標参加者数
            
        Returns:
            実験設定
        """
        self.logger.info(f"定着戦略実験を設計中: {experiment_name}")
        
        # 実験ID生成
        experiment_id = self._generate_experiment_id(experiment_name)
        
        # バリアント作成
        variants = []
        allocation_per_variant = 100.0 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            variant = ExperimentVariant(
                variant_id=f"variant_{i}",
                name=strategy.get('name', f'Strategy {i}'),
                description=strategy.get('description', f'Strategy variant {i}'),
                config=strategy,
                allocation_percentage=allocation_per_variant,
                expected_participants=int(target_participants * allocation_per_variant / 100)
            )
            variants.append(variant)
            
        # メトリクス定義
        metrics = [
            ExperimentMetric(
                metric_id="retention_rate",
                name="開発者定着率",
                description="実験期間終了時点での開発者定着率",
                metric_type="rate",
                primary=True,
                higher_is_better=True,
                minimum_detectable_effect=0.05,
                baseline_value=0.7
            ),
            ExperimentMetric(
                metric_id="review_acceptance_rate",
                name="レビュー受諾率",
                description="レビュー依頼に対する受諾率",
                metric_type="rate",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.03,
                baseline_value=0.8
            ),
            ExperimentMetric(
                metric_id="stress_level",
                name="平均ストレスレベル",
                description="開発者の平均ストレスレベル",
                metric_type="continuous",
                primary=False,
                higher_is_better=False,
                minimum_detectable_effect=0.1,
                baseline_value=0.4
            ),
            ExperimentMetric(
                metric_id="task_completion_rate",
                name="タスク完了率",
                description="割り当てられたタスクの完了率",
                metric_type="rate",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.02,
                baseline_value=0.85
            ),
            ExperimentMetric(
                metric_id="collaboration_score",
                name="協力スコア",
                description="開発者間の協力関係スコア",
                metric_type="continuous",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.05,
                baseline_value=0.6
            )
        ]
        
        # 実験期間設定
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        # サンプルサイズ計算
        primary_metric = next(m for m in metrics if m.primary)
        minimum_sample_size = self._calculate_sample_size(
            primary_metric.baseline_value,
            primary_metric.minimum_detectable_effect,
            significance_level=0.05,
            power=0.8
        )
        
        # 実験設定作成
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"開発者定着戦略の比較実験: {len(strategies)}種類の戦略を比較",
            hypothesis="新しい推薦戦略により開発者定着率が向上する",
            variants=variants,
            metrics=metrics,
            allocation_method=AllocationMethod.HASH_BASED,
            start_date=start_date,
            end_date=end_date,
            minimum_sample_size=minimum_sample_size,
            significance_level=0.05,
            power=0.8,
            stratification_keys=["expertise_level", "project_type"],
            exclusion_criteria={
                "min_activity_days": 7,
                "max_stress_level": 0.9
            }
        )
        
        self.logger.info(f"実験設計完了: {experiment_id}")
        return experiment_config
        
    def design_review_strategy_experiment(
        self,
        experiment_name: str,
        review_strategies: List[Dict[str, Any]],
        duration_days: int = 21,
        target_participants: int = 500
    ) -> ExperimentConfig:
        """
        レビュー戦略比較実験を設計
        
        Args:
            experiment_name: 実験名
            review_strategies: 比較するレビュー戦略のリスト
            duration_days: 実験期間（日数）
            target_participants: 目標参加者数
            
        Returns:
            実験設定
        """
        self.logger.info(f"レビュー戦略実験を設計中: {experiment_name}")
        
        experiment_id = self._generate_experiment_id(experiment_name)
        
        # バリアント作成
        variants = []
        allocation_per_variant = 100.0 / len(review_strategies)
        
        for i, strategy in enumerate(review_strategies):
            variant = ExperimentVariant(
                variant_id=f"review_variant_{i}",
                name=strategy.get('name', f'Review Strategy {i}'),
                description=strategy.get('description', f'Review strategy variant {i}'),
                config=strategy,
                allocation_percentage=allocation_per_variant,
                expected_participants=int(target_participants * allocation_per_variant / 100)
            )
            variants.append(variant)
            
        # レビュー特化メトリクス
        metrics = [
            ExperimentMetric(
                metric_id="review_acceptance_rate",
                name="レビュー受諾率",
                description="レビュー依頼に対する受諾率",
                metric_type="rate",
                primary=True,
                higher_is_better=True,
                minimum_detectable_effect=0.05,
                baseline_value=0.75
            ),
            ExperimentMetric(
                metric_id="review_response_time",
                name="レビュー応答時間",
                description="レビュー依頼から応答までの平均時間（時間）",
                metric_type="continuous",
                primary=False,
                higher_is_better=False,
                minimum_detectable_effect=2.0,
                baseline_value=24.0
            ),
            ExperimentMetric(
                metric_id="review_quality_score",
                name="レビュー品質スコア",
                description="レビューの品質評価スコア",
                metric_type="continuous",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.1,
                baseline_value=0.7
            ),
            ExperimentMetric(
                metric_id="reviewer_satisfaction",
                name="レビュワー満足度",
                description="レビュワーの満足度スコア",
                metric_type="continuous",
                primary=False,
                higher_is_better=True,
                minimum_detectable_effect=0.1,
                baseline_value=0.6
            )
        ]
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        primary_metric = next(m for m in metrics if m.primary)
        minimum_sample_size = self._calculate_sample_size(
            primary_metric.baseline_value,
            primary_metric.minimum_detectable_effect,
            significance_level=0.05,
            power=0.8
        )
        
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"レビュー戦略の比較実験: {len(review_strategies)}種類の戦略を比較",
            hypothesis="最適化されたレビュー戦略によりレビュー受諾率が向上する",
            variants=variants,
            metrics=metrics,
            allocation_method=AllocationMethod.STRATIFIED,
            start_date=start_date,
            end_date=end_date,
            minimum_sample_size=minimum_sample_size,
            significance_level=0.05,
            power=0.8,
            stratification_keys=["review_experience", "technical_domain"],
            exclusion_criteria={
                "min_reviews_given": 5,
                "max_current_workload": 0.8
            }
        )
        
        self.logger.info(f"レビュー戦略実験設計完了: {experiment_id}")
        return experiment_config
        
    def save_experiment_config(self, experiment_config: ExperimentConfig) -> str:
        """
        実験設定を保存
        
        Args:
            experiment_config: 実験設定
            
        Returns:
            保存されたファイルのパス
        """
        config_file = self.experiments_dir / f"{experiment_config.experiment_id}_config.yaml"
        
        # 設定をYAML形式で保存
        config_dict = asdict(experiment_config)
        
        # datetime オブジェクトを文字列に変換
        config_dict['start_date'] = experiment_config.start_date.isoformat()
        config_dict['end_date'] = experiment_config.end_date.isoformat()
        
        # Enum を文字列に変換
        config_dict['allocation_method'] = experiment_config.allocation_method.value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
        self.logger.info(f"実験設定を保存しました: {config_file}")
        return str(config_file)
        
    def load_experiment_config(self, experiment_id: str) -> ExperimentConfig:
        """
        実験設定を読み込み
        
        Args:
            experiment_id: 実験ID
            
        Returns:
            実験設定
        """
        config_file = self.experiments_dir / f"{experiment_id}_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"実験設定ファイルが見つかりません: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # datetime オブジェクトに変換
        config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        
        # Enum に変換
        config_dict['allocation_method'] = AllocationMethod(config_dict['allocation_method'])
        
        # バリアントとメトリクスをオブジェクトに変換
        variants = [ExperimentVariant(**v) for v in config_dict['variants']]
        metrics = [ExperimentMetric(**m) for m in config_dict['metrics']]
        
        config_dict['variants'] = variants
        config_dict['metrics'] = metrics
        
        return ExperimentConfig(**config_dict)
        
    def _generate_experiment_id(self, experiment_name: str) -> str:
        """実験IDを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(experiment_name.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{name_hash}"
        
    def _calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        サンプルサイズを計算
        
        Args:
            baseline_rate: ベースライン率
            minimum_detectable_effect: 最小検出可能効果
            significance_level: 有意水準
            power: 検出力
            
        Returns:
            必要なサンプルサイズ
        """
        # 効果サイズ計算
        effect_size = minimum_detectable_effect / np.sqrt(baseline_rate * (1 - baseline_rate))
        
        # Z値計算
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(power)
        
        # サンプルサイズ計算（各群）
        n_per_group = ((z_alpha + z_beta) / effect_size) ** 2
        
        # 総サンプルサイズ（2群比較の場合）
        total_sample_size = int(np.ceil(n_per_group * 2))
        
        # 最小サンプルサイズの制約
        min_sample_size = 100
        return max(total_sample_size, min_sample_size)


class ParticipantAllocator:
    """参加者割り当て器"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        """
        参加者割り当て器を初期化
        
        Args:
            experiment_config: 実験設定
        """
        self.experiment_config = experiment_config
        self.logger = setup_logger(__name__)
        
        # 割り当て履歴
        self.allocation_history = {}
        
    def allocate_participant(
        self,
        participant_id: str,
        participant_attributes: Dict[str, Any]
    ) -> Optional[str]:
        """
        参加者をバリアントに割り当て
        
        Args:
            participant_id: 参加者ID
            participant_attributes: 参加者属性
            
        Returns:
            割り当てられたバリアントID（除外された場合はNone）
        """
        # 除外基準チェック
        if not self._check_inclusion_criteria(participant_attributes):
            self.logger.debug(f"参加者 {participant_id} は除外基準により除外されました")
            return None
            
        # 割り当て方法に応じて処理
        if self.experiment_config.allocation_method == AllocationMethod.HASH_BASED:
            variant_id = self._hash_based_allocation(participant_id)
        elif self.experiment_config.allocation_method == AllocationMethod.STRATIFIED:
            variant_id = self._stratified_allocation(participant_id, participant_attributes)
        elif self.experiment_config.allocation_method == AllocationMethod.BALANCED:
            variant_id = self._balanced_allocation(participant_id)
        else:  # RANDOM
            variant_id = self._random_allocation()
            
        # 割り当て履歴記録
        self.allocation_history[participant_id] = {
            'variant_id': variant_id,
            'timestamp': datetime.now().isoformat(),
            'attributes': participant_attributes
        }
        
        self.logger.debug(f"参加者 {participant_id} をバリアント {variant_id} に割り当てました")
        return variant_id
        
    def _check_inclusion_criteria(self, attributes: Dict[str, Any]) -> bool:
        """包含基準をチェック"""
        if not self.experiment_config.exclusion_criteria:
            return True
            
        for criterion, threshold in self.experiment_config.exclusion_criteria.items():
            if criterion in attributes:
                if isinstance(threshold, (int, float)):
                    if criterion.startswith('min_') and attributes[criterion] < threshold:
                        return False
                    elif criterion.startswith('max_') and attributes[criterion] > threshold:
                        return False
                        
        return True
        
    def _hash_based_allocation(self, participant_id: str) -> str:
        """ハッシュベース割り当て"""
        # 参加者IDのハッシュ値を計算
        hash_value = int(hashlib.md5(participant_id.encode()).hexdigest(), 16)
        hash_percentage = (hash_value % 10000) / 100.0  # 0-100の範囲
        
        # 累積割り当て率で判定
        cumulative_percentage = 0
        for variant in self.experiment_config.variants:
            cumulative_percentage += variant.allocation_percentage
            if hash_percentage < cumulative_percentage:
                return variant.variant_id
                
        # フォールバック（最後のバリアント）
        return self.experiment_config.variants[-1].variant_id
        
    def _stratified_allocation(
        self,
        participant_id: str,
        attributes: Dict[str, Any]
    ) -> str:
        """層化割り当て"""
        # 層化キーに基づいて層を決定
        stratum_key = self._get_stratum_key(attributes)
        
        # 層ごとのハッシュベース割り当て
        combined_key = f"{participant_id}_{stratum_key}"
        return self._hash_based_allocation(combined_key)
        
    def _get_stratum_key(self, attributes: Dict[str, Any]) -> str:
        """層化キーを生成"""
        if not self.experiment_config.stratification_keys:
            return "default"
            
        stratum_values = []
        for key in self.experiment_config.stratification_keys:
            if key in attributes:
                value = attributes[key]
                if isinstance(value, (int, float)):
                    # 数値の場合は範囲に分割
                    if value < 0.3:
                        stratum_values.append(f"{key}_low")
                    elif value < 0.7:
                        stratum_values.append(f"{key}_medium")
                    else:
                        stratum_values.append(f"{key}_high")
                else:
                    stratum_values.append(f"{key}_{value}")
            else:
                stratum_values.append(f"{key}_unknown")
                
        return "_".join(stratum_values)
        
    def _balanced_allocation(self, participant_id: str) -> str:
        """バランス割り当て"""
        # 現在の割り当て数をカウント
        variant_counts = {}
        for variant in self.experiment_config.variants:
            variant_counts[variant.variant_id] = 0
            
        for allocation in self.allocation_history.values():
            variant_id = allocation['variant_id']
            if variant_id in variant_counts:
                variant_counts[variant_id] += 1
                
        # 最も少ないバリアントに割り当て
        min_count = min(variant_counts.values())
        min_variants = [vid for vid, count in variant_counts.items() if count == min_count]
        
        # 同じ数の場合はハッシュベースで選択
        if len(min_variants) == 1:
            return min_variants[0]
        else:
            hash_value = int(hashlib.md5(participant_id.encode()).hexdigest(), 16)
            return min_variants[hash_value % len(min_variants)]
            
    def _random_allocation(self) -> str:
        """ランダム割り当て"""
        random_value = np.random.random() * 100
        
        cumulative_percentage = 0
        for variant in self.experiment_config.variants:
            cumulative_percentage += variant.allocation_percentage
            if random_value < cumulative_percentage:
                return variant.variant_id
                
        return self.experiment_config.variants[-1].variant_id
        
    def get_allocation_summary(self) -> Dict[str, Any]:
        """割り当てサマリーを取得"""
        variant_counts = {}
        for variant in self.experiment_config.variants:
            variant_counts[variant.variant_id] = 0
            
        for allocation in self.allocation_history.values():
            variant_id = allocation['variant_id']
            if variant_id in variant_counts:
                variant_counts[variant_id] += 1
                
        total_participants = len(self.allocation_history)
        
        summary = {
            'total_participants': total_participants,
            'variant_counts': variant_counts,
            'variant_percentages': {
                vid: (count / total_participants * 100) if total_participants > 0 else 0
                for vid, count in variant_counts.items()
            }
        }
        
        return summary


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A/Bテスト実験設計システム')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--experiment-type', choices=['retention', 'review'], 
                       default='retention', help='実験タイプ')
    parser.add_argument('--name', type=str, required=True, help='実験名')
    parser.add_argument('--duration', type=int, default=30, help='実験期間（日数）')
    parser.add_argument('--participants', type=int, default=1000, help='目標参加者数')
    parser.add_argument('--output', type=str, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 実験設計器初期化
    designer = ExperimentDesigner(args.config)
    
    # 実験設計
    if args.experiment_type == 'retention':
        # サンプル定着戦略
        strategies = [
            {
                'name': 'Baseline Strategy',
                'description': 'Current recommendation strategy',
                'stress_weight': 0.3,
                'expertise_weight': 0.4,
                'workload_weight': 0.3
            },
            {
                'name': 'Stress-Optimized Strategy',
                'description': 'Strategy optimized for stress reduction',
                'stress_weight': 0.5,
                'expertise_weight': 0.3,
                'workload_weight': 0.2
            },
            {
                'name': 'Expertise-Focused Strategy',
                'description': 'Strategy focused on expertise matching',
                'stress_weight': 0.2,
                'expertise_weight': 0.6,
                'workload_weight': 0.2
            }
        ]
        
        experiment_config = designer.design_retention_strategy_experiment(
            args.name, strategies, args.duration, args.participants
        )
        
    else:  # review
        # サンプルレビュー戦略
        review_strategies = [
            {
                'name': 'Current Review Strategy',
                'description': 'Current review assignment strategy',
                'similarity_threshold': 0.7,
                'workload_factor': 0.3
            },
            {
                'name': 'High-Similarity Strategy',
                'description': 'Strategy prioritizing high similarity',
                'similarity_threshold': 0.8,
                'workload_factor': 0.2
            }
        ]
        
        experiment_config = designer.design_review_strategy_experiment(
            args.name, review_strategies, args.duration, args.participants
        )
    
    # 設定保存
    config_file = designer.save_experiment_config(experiment_config)
    
    print(f"実験設計完了: {experiment_config.experiment_id}")
    print(f"設定ファイル: {config_file}")
    print(f"実験期間: {experiment_config.start_date} - {experiment_config.end_date}")
    print(f"必要サンプルサイズ: {experiment_config.minimum_sample_size}")


if __name__ == '__main__':
    main()