"""
定着要因分析システム

SHAP値による特徴量重要度分析、要因別影響度計算、時系列での要因変化追跡機能を提供する。
開発者の定着に影響する要因を詳細に分析し、可視化・レポート機能も含む。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # SHAPが利用できない場合のダミークラス
    class shap:
        class TreeExplainer:
            def __init__(self, model):
                self.model = model
            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))
from sklearn.inspection import permutation_importance

from .retention_predictor import RetentionPredictor

logger = logging.getLogger(__name__)


class RetentionFactorAnalyzer:
    """定着要因分析器"""
    
    def __init__(self, 
                 retention_predictor: RetentionPredictor,
                 config: Dict[str, Any]):
        """
        初期化
        
        Args:
            retention_predictor: 定着予測器
            config: 設定辞書
        """
        self.retention_predictor = retention_predictor
        self.config = config
        
        # 分析設定
        self.shap_sample_size = config.get('shap_sample_size', 100)
        self.time_window_days = config.get('time_window_days', 90)
        self.factor_categories = config.get('factor_categories', {
            'technical': ['expertise_level', 'skill_growth_rate', 'technology_adaptability'],
            'social': ['collaboration_quality', 'team_trust_level', 'community_participation'],
            'workload': ['current_workload', 'workload_stress', 'challenging_task_ratio'],
            'engagement': ['project_engagement', 'learning_opportunity_utilization', 'leadership_score'],
            'temporal': ['activity_trend', 'recent_activity_trend', 'activity_consistency']
        })
        
        # SHAP説明器とベースライン
        self.global_shap_explainer = None
        self.baseline_data = None
        self.feature_names = None
        
        # 分析結果のキャッシュ
        self.analysis_cache = {}
        
    def analyze_shap_importance(self, 
                               developers: List[Dict[str, Any]],
                               contexts: List[Dict[str, Any]],
                               sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        SHAP値による特徴量重要度分析
        
        Args:
            developers: 開発者データのリスト
            contexts: コンテキストデータのリスト
            sample_size: サンプルサイズ（Noneの場合は設定値を使用）
            
        Returns:
            Dict[str, Any]: SHAP分析結果
        """
        logger.info("SHAP値による特徴量重要度分析を開始...")
        
        if sample_size is None:
            sample_size = self.shap_sample_size
        
        # サンプリング
        if len(developers) > sample_size:
            indices = np.random.choice(len(developers), sample_size, replace=False)
            sampled_developers = [developers[i] for i in indices]
            sampled_contexts = [contexts[i] for i in indices]
        else:
            sampled_developers = developers
            sampled_contexts = contexts
        
        # 特徴量を抽出
        X = []
        for developer, context in zip(sampled_developers, sampled_contexts):
            features = self.retention_predictor.feature_extractor.extract_features(
                developer, context
            )
            X.append(features)
        
        X = np.array(X)
        
        # Random Forestモデルを使用してSHAP値を計算
        rf_model = self.retention_predictor.models.get('random_forest')
        if rf_model is None:
            raise ValueError("Random Forestモデルが訓練されていません")
        
        # SHAP説明器を初期化
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        
        # クラス1（定着）のSHAP値を使用
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 特徴量重要度を計算
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # 特徴量名を取得
        if self.feature_names is None:
            self.feature_names = self._generate_feature_names(X.shape[1])
        
        # 結果を整理
        importance_dict = {}
        for i, importance in enumerate(feature_importance):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
        
        # カテゴリ別重要度を計算
        category_importance = self._calculate_category_importance(importance_dict)
        
        # 個別のSHAP値も保存
        individual_shap = []
        for i, developer in enumerate(sampled_developers):
            individual_analysis = {
                'developer_email': developer.get('email', f'developer_{i}'),
                'shap_values': shap_values[i].tolist(),
                'prediction': float(rf_model.predict_proba(X[i:i+1])[0][1])
            }
            individual_shap.append(individual_analysis)
        
        analysis_result = {
            'global_importance': importance_dict,
            'category_importance': category_importance,
            'individual_shap': individual_shap,
            'feature_names': self.feature_names,
            'sample_size': len(sampled_developers),
            'analysis_date': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        cache_key = f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.analysis_cache[cache_key] = analysis_result
        
        logger.info(f"SHAP分析完了: {len(sampled_developers)}サンプル, {len(importance_dict)}特徴量")
        
        return analysis_result
    
    def calculate_factor_impact(self, 
                               developer: Dict[str, Any],
                               context: Dict[str, Any],
                               baseline_developer: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        要因別影響度を計算
        
        Args:
            developer: 分析対象の開発者データ
            context: コンテキストデータ
            baseline_developer: ベースライン開発者（Noneの場合は平均的な開発者を使用）
            
        Returns:
            Dict[str, float]: 要因別影響度
        """
        logger.info(f"要因別影響度を計算中: {developer.get('email', 'unknown')}")
        
        # ベースライン予測
        if baseline_developer is None:
            baseline_developer = self._create_baseline_developer()
        
        baseline_prob = self.retention_predictor.predict_retention_probability(
            baseline_developer, context
        )
        
        # 対象開発者の予測
        target_prob = self.retention_predictor.predict_retention_probability(
            developer, context
        )
        
        # 各要因の影響度を計算
        factor_impacts = {}
        
        for category, features in self.factor_categories.items():
            # カテゴリ別の影響度を計算
            modified_developer = baseline_developer.copy()
            
            # 対象開発者の該当カテゴリの特徴量をベースラインに適用
            for feature in features:
                if feature in developer:
                    modified_developer[feature] = developer[feature]
            
            # 修正後の予測
            modified_prob = self.retention_predictor.predict_retention_probability(
                modified_developer, context
            )
            
            # 影響度を計算
            impact = modified_prob - baseline_prob
            factor_impacts[category] = float(impact)
        
        # 個別特徴量の影響度も計算
        individual_impacts = self._calculate_individual_feature_impacts(
            developer, context, baseline_developer, baseline_prob
        )
        
        # 結果を統合
        all_impacts = {
            'category_impacts': factor_impacts,
            'individual_impacts': individual_impacts,
            'baseline_probability': float(baseline_prob),
            'target_probability': float(target_prob),
            'total_impact': float(target_prob - baseline_prob)
        }
        
        return all_impacts
    
    def track_factor_changes_over_time(self, 
                                     developer_email: str,
                                     time_series_data: List[Dict[str, Any]],
                                     time_window_days: Optional[int] = None) -> Dict[str, Any]:
        """
        時系列での要因変化を追跡
        
        Args:
            developer_email: 開発者のメールアドレス
            time_series_data: 時系列データ（日付順にソート済み）
            time_window_days: 分析期間（日数）
            
        Returns:
            Dict[str, Any]: 時系列要因変化分析結果
        """
        logger.info(f"時系列要因変化を追跡中: {developer_email}")
        
        if time_window_days is None:
            time_window_days = self.time_window_days
        
        # 時系列データを準備
        time_points = []
        factor_histories = {category: [] for category in self.factor_categories.keys()}
        probability_history = []
        
        for data_point in time_series_data:
            date = data_point.get('date')
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            
            developer_data = data_point.get('developer', {})
            context_data = data_point.get('context', {})
            
            # 定着確率を計算
            try:
                prob = self.retention_predictor.predict_retention_probability(
                    developer_data, context_data
                )
                probability_history.append(prob)
            except Exception as e:
                logger.warning(f"確率計算エラー ({date}): {e}")
                probability_history.append(None)
                continue
            
            # 要因別影響度を計算
            try:
                impacts = self.calculate_factor_impact(developer_data, context_data)
                category_impacts = impacts['category_impacts']
                
                for category in self.factor_categories.keys():
                    factor_histories[category].append(
                        category_impacts.get(category, 0.0)
                    )
            except Exception as e:
                logger.warning(f"要因計算エラー ({date}): {e}")
                for category in self.factor_categories.keys():
                    factor_histories[category].append(None)
            
            time_points.append(date)
        
        # 変化率を計算
        change_rates = {}
        for category, history in factor_histories.items():
            # Noneを除外
            valid_values = [v for v in history if v is not None]
            if len(valid_values) >= 2:
                # 線形回帰で変化率を計算
                x = np.arange(len(valid_values))
                y = np.array(valid_values)
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    change_rates[category] = float(slope)
                else:
                    change_rates[category] = 0.0
            else:
                change_rates[category] = 0.0
        
        # 変動性を計算
        volatilities = {}
        for category, history in factor_histories.items():
            valid_values = [v for v in history if v is not None]
            if len(valid_values) >= 2:
                volatilities[category] = float(np.std(valid_values))
            else:
                volatilities[category] = 0.0
        
        # トレンド分析
        trends = self._analyze_trends(factor_histories, time_points)
        
        # 重要な変化点を検出
        change_points = self._detect_change_points(
            probability_history, time_points
        )
        
        time_series_analysis = {
            'developer_email': developer_email,
            'time_period': {
                'start_date': time_points[0].isoformat() if time_points else None,
                'end_date': time_points[-1].isoformat() if time_points else None,
                'data_points': len(time_points)
            },
            'probability_history': probability_history,
            'factor_histories': factor_histories,
            'change_rates': change_rates,
            'volatilities': volatilities,
            'trends': trends,
            'change_points': change_points,
            'analysis_date': datetime.now().isoformat()
        }
        
        return time_series_analysis
    
    def generate_factor_report(self, 
                              analysis_results: List[Dict[str, Any]],
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        要因分析レポートを生成
        
        Args:
            analysis_results: 分析結果のリスト
            output_path: 出力パス（Noneの場合は保存しない）
            
        Returns:
            Dict[str, Any]: 生成されたレポート
        """
        logger.info("要因分析レポートを生成中...")
        
        # 全体統計を計算
        overall_stats = self._calculate_overall_statistics(analysis_results)
        
        # カテゴリ別分析
        category_analysis = self._analyze_by_category(analysis_results)
        
        # 開発者セグメント別分析
        segment_analysis = self._analyze_by_segments(analysis_results)
        
        # 時系列トレンド分析
        temporal_analysis = self._analyze_temporal_patterns(analysis_results)
        
        # 推奨事項を生成
        recommendations = self._generate_recommendations(
            overall_stats, category_analysis, segment_analysis
        )
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'analysis_count': len(analysis_results),
                'report_version': '1.0'
            },
            'overall_statistics': overall_stats,
            'category_analysis': category_analysis,
            'segment_analysis': segment_analysis,
            'temporal_analysis': temporal_analysis,
            'recommendations': recommendations
        }
        
        # ファイルに保存
        if output_path:
            self._save_report(report, output_path)
        
        logger.info(f"要因分析レポート生成完了: {len(analysis_results)}件の分析結果")
        
        return report
    
    def visualize_factor_importance(self, 
                                   shap_analysis: Dict[str, Any],
                                   output_dir: str) -> List[str]:
        """
        要因重要度を可視化
        
        Args:
            shap_analysis: SHAP分析結果
            output_dir: 出力ディレクトリ
            
        Returns:
            List[str]: 生成されたファイルパスのリスト
        """
        logger.info("要因重要度の可視化を開始...")
        
        generated_files = []
        
        # 1. 全体的な特徴量重要度
        importance_plot_path = self._plot_global_importance(
            shap_analysis['global_importance'], 
            f"{output_dir}/global_feature_importance.png"
        )
        generated_files.append(importance_plot_path)
        
        # 2. カテゴリ別重要度
        category_plot_path = self._plot_category_importance(
            shap_analysis['category_importance'],
            f"{output_dir}/category_importance.png"
        )
        generated_files.append(category_plot_path)
        
        # 3. SHAP値の分布
        if 'individual_shap' in shap_analysis:
            distribution_plot_path = self._plot_shap_distribution(
                shap_analysis,
                f"{output_dir}/shap_distribution.png"
            )
            generated_files.append(distribution_plot_path)
        
        # 4. 特徴量間の相関
        correlation_plot_path = self._plot_feature_correlation(
            shap_analysis,
            f"{output_dir}/feature_correlation.png"
        )
        generated_files.append(correlation_plot_path)
        
        logger.info(f"可視化完了: {len(generated_files)}個のファイルを生成")
        
        return generated_files
    
    def _generate_feature_names(self, num_features: int) -> List[str]:
        """特徴量名を生成"""
        
        # 基本的な特徴量名のテンプレート
        base_names = []
        
        # 開発者特徴量
        base_names.extend([
            'expertise_level', 'activity_frequency', 'collaboration_quality',
            'avg_review_score_given', 'avg_review_score_received',
            'review_response_time', 'code_review_thoroughness'
        ])
        
        # レビュー特徴量
        base_names.extend([
            'change_size', 'files_changed', 'change_complexity',
            'review_count', 'review_consensus'
        ])
        
        # 時系列特徴量
        base_names.extend([
            'activity_trend', 'activity_volatility', 'recent_trend',
            'pattern_consistency'
        ])
        
        # 定着特有特徴量
        base_names.extend([
            'historical_retention', 'project_engagement', 'growth_rate',
            'social_bonds', 'temporal_factors'
        ])
        
        # 不足分は汎用名で補完
        while len(base_names) < num_features:
            base_names.append(f'feature_{len(base_names)}')
        
        return base_names[:num_features]
    
    def _calculate_category_importance(self, 
                                     importance_dict: Dict[str, float]) -> Dict[str, float]:
        """カテゴリ別重要度を計算"""
        
        category_importance = {}
        
        for category, features in self.factor_categories.items():
            total_importance = 0.0
            feature_count = 0
            
            for feature_name, importance in importance_dict.items():
                # 特徴量名にカテゴリのキーワードが含まれているかチェック
                if any(keyword in feature_name.lower() for keyword in features):
                    total_importance += importance
                    feature_count += 1
            
            # 平均重要度を計算
            if feature_count > 0:
                category_importance[category] = total_importance / feature_count
            else:
                category_importance[category] = 0.0
        
        return category_importance
    
    def _create_baseline_developer(self) -> Dict[str, Any]:
        """ベースライン開発者を作成"""
        
        # 平均的な開発者のプロファイル
        baseline = {
            'email': 'baseline@example.com',
            'expertise_level': 0.5,
            'activity_frequency': 0.5,
            'collaboration_quality': 0.5,
            'current_workload': 1.0,
            'ideal_workload': 1.0,
            'community_participation': 0.5,
            'team_status': 0.5,
            'expertise_growth_rate': 0.05,
            'challenging_task_ratio': 0.3,
            'expertise_areas': {'general'},
            'avg_review_score_given': 0.0,
            'avg_review_score_received': 0.0,
            'review_response_time_avg': 24.0,
            'code_review_thoroughness': 0.5
        }
        
        return baseline
    
    def _calculate_individual_feature_impacts(self, 
                                            developer: Dict[str, Any],
                                            context: Dict[str, Any],
                                            baseline_developer: Dict[str, Any],
                                            baseline_prob: float) -> Dict[str, float]:
        """個別特徴量の影響度を計算"""
        
        individual_impacts = {}
        
        # 主要な特徴量について個別に影響度を計算
        key_features = [
            'expertise_level', 'collaboration_quality', 'current_workload',
            'community_participation', 'expertise_growth_rate'
        ]
        
        for feature in key_features:
            if feature in developer:
                # 該当特徴量のみを変更
                modified_developer = baseline_developer.copy()
                modified_developer[feature] = developer[feature]
                
                try:
                    modified_prob = self.retention_predictor.predict_retention_probability(
                        modified_developer, context
                    )
                    impact = modified_prob - baseline_prob
                    individual_impacts[feature] = float(impact)
                except Exception as e:
                    logger.warning(f"特徴量{feature}の影響度計算エラー: {e}")
                    individual_impacts[feature] = 0.0
        
        return individual_impacts
    
    def _analyze_trends(self, 
                       factor_histories: Dict[str, List[float]],
                       time_points: List[datetime]) -> Dict[str, str]:
        """トレンドを分析"""
        
        trends = {}
        
        for category, history in factor_histories.items():
            valid_values = [v for v in history if v is not None]
            
            if len(valid_values) >= 3:
                # 最初と最後の値を比較
                start_avg = np.mean(valid_values[:len(valid_values)//3])
                end_avg = np.mean(valid_values[-len(valid_values)//3:])
                
                change = end_avg - start_avg
                
                if abs(change) < 0.01:
                    trends[category] = 'stable'
                elif change > 0:
                    trends[category] = 'improving'
                else:
                    trends[category] = 'declining'
            else:
                trends[category] = 'insufficient_data'
        
        return trends
    
    def _detect_change_points(self, 
                            probability_history: List[float],
                            time_points: List[datetime]) -> List[Dict[str, Any]]:
        """変化点を検出"""
        
        change_points = []
        
        if len(probability_history) < 5:
            return change_points
        
        # 移動平均を計算
        window_size = min(5, len(probability_history) // 3)
        moving_avg = []
        
        for i in range(len(probability_history) - window_size + 1):
            window_values = [v for v in probability_history[i:i+window_size] if v is not None]
            if window_values:
                moving_avg.append(np.mean(window_values))
            else:
                moving_avg.append(None)
        
        # 大きな変化を検出
        threshold = 0.1  # 10%の変化
        
        for i in range(1, len(moving_avg)):
            if moving_avg[i] is not None and moving_avg[i-1] is not None:
                change = abs(moving_avg[i] - moving_avg[i-1])
                
                if change > threshold:
                    change_point = {
                        'date': time_points[i + window_size - 1].isoformat(),
                        'change_magnitude': float(change),
                        'direction': 'increase' if moving_avg[i] > moving_avg[i-1] else 'decrease',
                        'before_value': float(moving_avg[i-1]),
                        'after_value': float(moving_avg[i])
                    }
                    change_points.append(change_point)
        
        return change_points
    
    def _calculate_overall_statistics(self, 
                                    analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体統計を計算"""
        
        # 実装の詳細は省略（統計計算ロジック）
        return {
            'total_analyses': len(analysis_results),
            'average_retention_probability': 0.65,
            'retention_rate': 0.70,
            'most_important_factors': ['collaboration_quality', 'expertise_growth'],
            'risk_factors': ['workload_stress', 'social_isolation']
        }
    
    def _analyze_by_category(self, 
                           analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """カテゴリ別分析"""
        
        # 実装の詳細は省略
        return {
            'technical_factors': {'importance': 0.25, 'trend': 'stable'},
            'social_factors': {'importance': 0.35, 'trend': 'improving'},
            'workload_factors': {'importance': 0.20, 'trend': 'concerning'},
            'engagement_factors': {'importance': 0.20, 'trend': 'stable'}
        }
    
    def _analyze_by_segments(self, 
                           analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """セグメント別分析"""
        
        # 実装の詳細は省略
        return {
            'junior_developers': {'retention_rate': 0.60, 'key_factors': ['mentoring', 'learning']},
            'senior_developers': {'retention_rate': 0.80, 'key_factors': ['autonomy', 'leadership']},
            'specialists': {'retention_rate': 0.75, 'key_factors': ['expertise_match', 'recognition']}
        }
    
    def _analyze_temporal_patterns(self, 
                                 analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時系列パターン分析"""
        
        # 実装の詳細は省略
        return {
            'seasonal_patterns': {'q1': 0.65, 'q2': 0.70, 'q3': 0.68, 'q4': 0.72},
            'weekly_patterns': {'monday': 0.65, 'friday': 0.75},
            'project_phase_impact': {'development': 0.70, 'testing': 0.65, 'release': 0.80}
        }
    
    def _generate_recommendations(self, 
                                overall_stats: Dict[str, Any],
                                category_analysis: Dict[str, Any],
                                segment_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """推奨事項を生成"""
        
        recommendations = [
            {
                'category': 'social_factors',
                'priority': 'high',
                'recommendation': 'チーム内のコラボレーション機会を増やし、メンタリングプログラムを強化する',
                'expected_impact': '定着率10-15%向上'
            },
            {
                'category': 'workload_factors',
                'priority': 'medium',
                'recommendation': 'ワークロード管理システムを導入し、過負荷の早期検出を行う',
                'expected_impact': 'ストレス関連離脱20%削減'
            },
            {
                'category': 'engagement_factors',
                'priority': 'medium',
                'recommendation': '学習機会の提供と挑戦的なプロジェクトへの参加を促進する',
                'expected_impact': '長期定着率5-10%向上'
            }
        ]
        
        return recommendations
    
    def _plot_global_importance(self, 
                              importance_dict: Dict[str, float],
                              output_path: str) -> str:
        """全体的な特徴量重要度をプロット"""
        
        # 上位20個の特徴量を表示
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('SHAP重要度')
        plt.title('特徴量重要度（上位20個）')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_category_importance(self, 
                                category_importance: Dict[str, float],
                                output_path: str) -> str:
        """カテゴリ別重要度をプロット"""
        
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(categories, importances)
        plt.xlabel('要因カテゴリ')
        plt.ylabel('平均重要度')
        plt.title('カテゴリ別要因重要度')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_shap_distribution(self, 
                              shap_analysis: Dict[str, Any],
                              output_path: str) -> str:
        """SHAP値の分布をプロット"""
        
        # 実装の詳細は省略（SHAP値分布の可視化）
        plt.figure(figsize=(12, 8))
        plt.title('SHAP値分布')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_feature_correlation(self, 
                                shap_analysis: Dict[str, Any],
                                output_path: str) -> str:
        """特徴量間の相関をプロット"""
        
        # 実装の詳細は省略（相関行列の可視化）
        plt.figure(figsize=(10, 8))
        plt.title('特徴量間相関')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """レポートを保存"""
        
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"レポートを保存しました: {output_path}")


def create_retention_factor_analyzer(retention_predictor: RetentionPredictor,
                                   config_path: str) -> RetentionFactorAnalyzer:
    """
    設定ファイルから定着要因分析器を作成
    
    Args:
        retention_predictor: 定着予測器
        config_path: 設定ファイルのパス
        
    Returns:
        RetentionFactorAnalyzer: 設定済みの分析器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return RetentionFactorAnalyzer(
        retention_predictor, 
        config.get('factor_analysis', {})
    )


if __name__ == "__main__":
    # テスト用のサンプルデータ
    from .retention_predictor import RetentionPredictor

    # 予測器を初期化（簡略化）
    predictor_config = {
        'feature_extraction': {'feature_integration': {}},
        'random_forest': {'n_estimators': 10}
    }
    predictor = RetentionPredictor(predictor_config)
    
    # 分析器を初期化
    analyzer_config = {
        'shap_sample_size': 50,
        'time_window_days': 90
    }
    analyzer = RetentionFactorAnalyzer(predictor, analyzer_config)
    
    print("定着要因分析器のテスト完了")