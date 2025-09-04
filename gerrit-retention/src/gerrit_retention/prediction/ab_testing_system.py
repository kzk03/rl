"""
A/Bテストによる予測戦略比較システム

複数の予測戦略を同時に評価し、最適な戦略を特定する
統計的に有意な比較分析システム
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

class ABTestingSystem:
    """A/Bテストによる予測戦略比較システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.strategies = {}
        self.test_results = {}
        self.statistical_tests = {}
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('ABTestingSystem')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_strategy(self, name: str, strategy_func: callable, description: str = ""):
        """予測戦略の登録"""
        self.strategies[name] = {
            'function': strategy_func,
            'description': description,
            'results': []
        }
        self.logger.info(f"戦略を登録: {name} - {description}")
    
    def create_baseline_strategies(self) -> Dict[str, callable]:
        """ベースライン戦略の作成"""
        strategies = {}
        
        # 戦略1: 活動頻度ベース
        def activity_frequency_strategy(developer_data: Dict[str, Any]) -> float:
            """活動頻度に基づく継続予測"""
            try:
                first_seen = datetime.fromisoformat(
                    developer_data.get('first_seen', '').replace(' ', 'T')
                )
                last_activity = datetime.fromisoformat(
                    developer_data.get('last_activity', '').replace(' ', 'T')
                )
                
                activity_duration = (last_activity - first_seen).days
                total_activity = (developer_data.get('changes_authored', 0) + 
                                developer_data.get('changes_reviewed', 0))
                
                if activity_duration > 0:
                    frequency = total_activity / activity_duration
                    return min(1.0, frequency * 10)  # スケーリング
                else:
                    return 0.5
            except:
                return 0.0
        
        # 戦略2: 最近の活動ベース
        def recent_activity_strategy(developer_data: Dict[str, Any]) -> float:
            """最近の活動に基づく継続予測"""
            try:
                last_activity = datetime.fromisoformat(
                    developer_data.get('last_activity', '').replace(' ', 'T')
                )
                days_since_last = (datetime.now() - last_activity).days
                
                if days_since_last <= 7:
                    return 0.9
                elif days_since_last <= 30:
                    return 0.7
                elif days_since_last <= 90:
                    return 0.4
                else:
                    return 0.1
            except:
                return 0.0
        
        # 戦略3: バランス型
        def balanced_strategy(developer_data: Dict[str, Any]) -> float:
            """バランス型継続予測"""
            activity_score = activity_frequency_strategy(developer_data)
            recent_score = recent_activity_strategy(developer_data)
            
            # プロジェクト多様性の考慮
            project_count = len(developer_data.get('projects', []))
            diversity_bonus = min(0.2, project_count * 0.05)
            
            # レビュースコアの考慮
            review_scores = developer_data.get('review_scores', [])
            if review_scores:
                avg_score = np.mean([abs(s) for s in review_scores])
                review_bonus = min(0.1, avg_score * 0.1)
            else:
                review_bonus = 0.0
            
            final_score = (activity_score * 0.4 + recent_score * 0.6 + 
                          diversity_bonus + review_bonus)
            
            return min(1.0, final_score)
        
        # 戦略4: 保守的戦略
        def conservative_strategy(developer_data: Dict[str, Any]) -> float:
            """保守的継続予測（高い閾値）"""
            base_score = balanced_strategy(developer_data)
            
            # より厳しい判定
            total_activity = (developer_data.get('changes_authored', 0) + 
                            developer_data.get('changes_reviewed', 0))
            
            if total_activity < 20:
                return base_score * 0.5
            elif total_activity < 50:
                return base_score * 0.7
            else:
                return base_score
        
        # 戦略5: 積極的戦略
        def aggressive_strategy(developer_data: Dict[str, Any]) -> float:
            """積極的継続予測（低い閾値）"""
            base_score = balanced_strategy(developer_data)
            
            # より寛容な判定
            if base_score > 0.3:
                return min(1.0, base_score * 1.3)
            else:
                return base_score
        
        strategies['activity_frequency'] = activity_frequency_strategy
        strategies['recent_activity'] = recent_activity_strategy
        strategies['balanced'] = balanced_strategy
        strategies['conservative'] = conservative_strategy
        strategies['aggressive'] = aggressive_strategy
        
        return strategies
    
    def run_ab_test(self, developer_data: List[Dict[str, Any]], 
                    true_labels: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
        """A/Bテストの実行"""
        self.logger.info(f"A/Bテストを開始: {len(self.strategies)}戦略, {n_splits}分割交差検証")
        
        # 結果格納用
        all_results = {name: [] for name in self.strategies.keys()}
        
        # 交差検証の実行
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # ラベルを二値化（継続/離脱）
        binary_labels = (true_labels > 0.5).astype(int)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(developer_data, binary_labels)):
            self.logger.info(f"フォールド {fold + 1}/{n_splits} を実行中...")
            
            test_data = [developer_data[i] for i in test_idx]
            test_labels = true_labels[test_idx]
            test_binary_labels = binary_labels[test_idx]
            
            # 各戦略で予測
            for strategy_name, strategy_info in self.strategies.items():
                strategy_func = strategy_info['function']
                
                # 予測の実行
                predictions = []
                for dev_data in test_data:
                    pred = strategy_func(dev_data)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                binary_predictions = (predictions > 0.5).astype(int)
                
                # メトリクスの計算
                metrics = {
                    'accuracy': accuracy_score(test_binary_labels, binary_predictions),
                    'precision': precision_score(test_binary_labels, binary_predictions, zero_division=0),
                    'recall': recall_score(test_binary_labels, binary_predictions, zero_division=0),
                    'f1': f1_score(test_binary_labels, binary_predictions, zero_division=0),
                    'mse': np.mean((predictions - test_labels) ** 2),
                    'mae': np.mean(np.abs(predictions - test_labels))
                }
                
                all_results[strategy_name].append(metrics)
        
        # 結果の統計処理
        final_results = {}
        for strategy_name, fold_results in all_results.items():
            metrics_summary = {}
            
            for metric_name in fold_results[0].keys():
                values = [result[metric_name] for result in fold_results]
                metrics_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            
            final_results[strategy_name] = metrics_summary
        
        self.test_results = final_results
        return final_results
    
    def perform_statistical_tests(self, metric: str = 'f1') -> Dict[str, Any]:
        """統計的有意性検定の実行"""
        self.logger.info(f"統計的有意性検定を実行: {metric}メトリック")
        
        if not self.test_results:
            raise ValueError("A/Bテストが実行されていません")
        
        strategy_names = list(self.test_results.keys())
        n_strategies = len(strategy_names)
        
        # ペアワイズt検定
        pairwise_tests = {}
        significance_matrix = np.ones((n_strategies, n_strategies))
        
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names):
                if i != j:
                    values1 = self.test_results[strategy1][metric]['values']
                    values2 = self.test_results[strategy2][metric]['values']
                    
                    # 対応のあるt検定
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    
                    pairwise_tests[f"{strategy1}_vs_{strategy2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': (np.mean(values1) - np.mean(values2)) / np.sqrt(
                            (np.var(values1) + np.var(values2)) / 2
                        )
                    }
                    
                    significance_matrix[i, j] = p_value
        
        # ANOVA検定
        all_values = [self.test_results[name][metric]['values'] for name in strategy_names]
        f_stat, anova_p_value = stats.f_oneway(*all_values)
        
        # 最良戦略の特定
        strategy_means = {name: self.test_results[name][metric]['mean'] 
                         for name in strategy_names}
        best_strategy = max(strategy_means, key=strategy_means.get)
        
        statistical_results = {
            'metric': metric,
            'pairwise_tests': pairwise_tests,
            'anova': {
                'f_statistic': f_stat,
                'p_value': anova_p_value,
                'significant': anova_p_value < 0.05
            },
            'best_strategy': best_strategy,
            'strategy_ranking': sorted(strategy_means.items(), key=lambda x: x[1], reverse=True),
            'significance_matrix': significance_matrix.tolist(),
            'strategy_names': strategy_names
        }
        
        self.statistical_tests[metric] = statistical_results
        return statistical_results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """比較レポートの生成"""
        if not self.test_results:
            raise ValueError("A/Bテストが実行されていません")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': self.test_results,
            'statistical_tests': self.statistical_tests,
            'recommendations': []
        }
        
        # サマリーの作成
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'mse', 'mae']
        
        for metric in metrics:
            metric_summary = {}
            
            for strategy_name, results in self.test_results.items():
                metric_summary[strategy_name] = {
                    'mean': results[metric]['mean'],
                    'std': results[metric]['std']
                }
            
            # 最良戦略の特定
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                best_strategy = max(metric_summary, key=lambda x: metric_summary[x]['mean'])
            else:  # MSE, MAE（小さい方が良い）
                best_strategy = min(metric_summary, key=lambda x: metric_summary[x]['mean'])
            
            report['summary'][metric] = {
                'results': metric_summary,
                'best_strategy': best_strategy
            }
        
        # 推奨事項の生成
        recommendations = self._generate_recommendations()
        report['recommendations'] = recommendations
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        if not self.test_results:
            return recommendations
        
        # F1スコアベースの分析
        f1_results = {name: results['f1']['mean'] 
                     for name, results in self.test_results.items()}
        
        best_strategy = max(f1_results, key=f1_results.get)
        worst_strategy = min(f1_results, key=f1_results.get)
        
        best_f1 = f1_results[best_strategy]
        worst_f1 = f1_results[worst_strategy]
        
        recommendations.append(f"最高性能戦略: {best_strategy} (F1: {best_f1:.4f})")
        
        if best_f1 - worst_f1 > 0.1:
            recommendations.append(f"{best_strategy}戦略が他の戦略より明確に優秀です")
            recommendations.append(f"{worst_strategy}戦略は性能が低いため使用を避けてください")
        
        # 統計的有意性の確認
        if 'f1' in self.statistical_tests:
            stat_results = self.statistical_tests['f1']
            if stat_results['anova']['significant']:
                recommendations.append("戦略間に統計的に有意な差があります")
            
            # 最良戦略との比較
            for strategy_name in f1_results.keys():
                if strategy_name != best_strategy:
                    test_key = f"{best_strategy}_vs_{strategy_name}"
                    if test_key in stat_results['pairwise_tests']:
                        test_result = stat_results['pairwise_tests'][test_key]
                        if test_result['significant']:
                            recommendations.append(
                                f"{best_strategy}は{strategy_name}より統計的に有意に優秀です "
                                f"(p={test_result['p_value']:.4f})"
                            )
        
        # 精度と再現率のバランス
        precision_results = {name: results['precision']['mean'] 
                           for name, results in self.test_results.items()}
        recall_results = {name: results['recall']['mean'] 
                         for name, results in self.test_results.items()}
        
        best_precision = max(precision_results, key=precision_results.get)
        best_recall = max(recall_results, key=recall_results.get)
        
        if best_precision != best_recall:
            recommendations.append(
                f"精度重視なら{best_precision}、再現率重視なら{best_recall}を選択してください"
            )
        
        # 安定性の評価
        stability_scores = {}
        for name, results in self.test_results.items():
            # F1スコアの標準偏差で安定性を評価
            stability_scores[name] = results['f1']['std']
        
        most_stable = min(stability_scores, key=stability_scores.get)
        recommendations.append(f"最も安定した戦略: {most_stable} (F1 std: {stability_scores[most_stable]:.4f})")
        
        return recommendations
    
    def create_visualization(self, output_path: str):
        """結果の可視化"""
        if not self.test_results:
            raise ValueError("A/Bテストが実行されていません")
        
        # 図のサイズとスタイル設定
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('A/Bテスト結果比較', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'mse', 'mae']
        strategy_names = list(self.test_results.keys())
        
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # データの準備
            means = [self.test_results[name][metric]['mean'] for name in strategy_names]
            stds = [self.test_results[name][metric]['std'] for name in strategy_names]
            
            # バープロット
            bars = ax.bar(range(len(strategy_names)), means, yerr=stds, 
                         capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # 最良戦略をハイライト
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                best_idx = np.argmax(means)
            else:
                best_idx = np.argmin(means)
            
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('orange')
            
            ax.set_title(f'{metric.upper()}', fontweight='bold')
            ax.set_xticks(range(len(strategy_names)))
            ax.set_xticklabels(strategy_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"{output_path}/ab_test_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可視化結果を保存: {plot_file}")
        return plot_file
    
    def save_results(self, output_path: str) -> str:
        """結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 比較レポートの生成
        report = self.generate_comparison_report()
        
        # JSON serializable に変換
        def make_json_serializable(obj):
            """オブジェクトをJSON serializable に変換"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_report = make_json_serializable(report)
        
        # ファイル保存
        results_file = f"{output_path}/ab_test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"A/Bテスト結果を保存: {results_file}")
        return results_file

def main():
    """メイン実行関数"""
    config = {
        'output_path': 'outputs/ab_testing',
        'n_splits': 5
    }
    
    # 出力ディレクトリの作成
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    
    # システムの初期化
    ab_system = ABTestingSystem(config)
    
    # ベースライン戦略の登録
    baseline_strategies = ab_system.create_baseline_strategies()
    
    for name, func in baseline_strategies.items():
        description = {
            'activity_frequency': '活動頻度ベースの継続予測',
            'recent_activity': '最近の活動ベースの継続予測',
            'balanced': 'バランス型継続予測',
            'conservative': '保守的継続予測（高閾値）',
            'aggressive': '積極的継続予測（低閾値）'
        }.get(name, '')
        
        ab_system.register_strategy(name, func, description)
    
    print("🧪 A/Bテストシステムが初期化されました")
    print(f"登録された戦略: {len(ab_system.strategies)}個")
    
    return ab_system

if __name__ == "__main__":
    main()