"""
高度な予測精度改善システム

現在の217.7%改善を達成したシステムをベースに、
さらなる精度向上を実現する包括的な改善システム
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

class AdvancedAccuracyImprover:
    """高度な予測精度改善システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = {}
        self.performance_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('AdvancedAccuracyImprover')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_advanced_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """高度な特徴量抽出"""
        features = {}
        
        # 基本統計特徴量
        features.update(self._extract_basic_features(developer_data))
        
        # 時系列特徴量
        features.update(self._extract_temporal_features(developer_data))
        
        # 行動パターン特徴量
        features.update(self._extract_behavioral_features(developer_data))
        
        # ネットワーク特徴量
        features.update(self._extract_network_features(developer_data))
        
        # 品質特徴量
        features.update(self._extract_quality_features(developer_data))
        
        return features
    
    def _extract_basic_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """基本統計特徴量の抽出"""
        features = {}
        
        # 活動量特徴量
        features['changes_authored'] = data.get('changes_authored', 0)
        features['changes_reviewed'] = data.get('changes_reviewed', 0)
        features['total_insertions'] = data.get('total_insertions', 0)
        features['total_deletions'] = data.get('total_deletions', 0)
        
        # 比率特徴量
        total_changes = features['changes_authored'] + features['changes_reviewed']
        if total_changes > 0:
            features['author_review_ratio'] = features['changes_authored'] / total_changes
            features['review_author_ratio'] = features['changes_reviewed'] / total_changes
        else:
            features['author_review_ratio'] = 0.0
            features['review_author_ratio'] = 0.0
        
        # コード変更特徴量
        total_lines = features['total_insertions'] + features['total_deletions']
        if total_lines > 0:
            features['insertion_ratio'] = features['total_insertions'] / total_lines
            features['deletion_ratio'] = features['total_deletions'] / total_lines
            features['avg_lines_per_change'] = total_lines / max(features['changes_authored'], 1)
        else:
            features['insertion_ratio'] = 0.0
            features['deletion_ratio'] = 0.0
            features['avg_lines_per_change'] = 0.0
        
        # プロジェクト多様性
        features['project_count'] = len(data.get('projects', []))
        features['source_count'] = len(data.get('sources', []))
        
        return features
    
    def _extract_temporal_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """時系列特徴量の抽出"""
        features = {}
        
        try:
            first_seen = datetime.fromisoformat(data.get('first_seen', '').replace(' ', 'T'))
            last_activity = datetime.fromisoformat(data.get('last_activity', '').replace(' ', 'T'))
            
            # 活動期間
            activity_duration = (last_activity - first_seen).days
            features['activity_duration_days'] = activity_duration
            
            # 現在からの経過時間
            now = datetime.now()
            days_since_last = (now - last_activity).days
            features['days_since_last_activity'] = days_since_last
            
            # 活動頻度
            if activity_duration > 0:
                total_changes = data.get('changes_authored', 0) + data.get('changes_reviewed', 0)
                features['activity_frequency'] = total_changes / activity_duration
            else:
                features['activity_frequency'] = 0.0
            
            # 最近の活動度（重要な継続予測指標）
            if days_since_last <= 7:
                features['recent_activity_score'] = 1.0
            elif days_since_last <= 30:
                features['recent_activity_score'] = 0.7
            elif days_since_last <= 90:
                features['recent_activity_score'] = 0.3
            else:
                features['recent_activity_score'] = 0.1
                
        except (ValueError, TypeError):
            features['activity_duration_days'] = 0
            features['days_since_last_activity'] = 999
            features['activity_frequency'] = 0.0
            features['recent_activity_score'] = 0.0
        
        return features
    
    def _extract_behavioral_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """行動パターン特徴量の抽出"""
        features = {}
        
        # レビュースコア分析
        review_scores = data.get('review_scores', [])
        if review_scores:
            features['avg_review_score'] = np.mean(review_scores)
            features['review_score_std'] = np.std(review_scores)
            features['positive_review_ratio'] = sum(1 for s in review_scores if s > 0) / len(review_scores)
            features['negative_review_ratio'] = sum(1 for s in review_scores if s < 0) / len(review_scores)
            features['neutral_review_ratio'] = sum(1 for s in review_scores if s == 0) / len(review_scores)
        else:
            features['avg_review_score'] = 0.0
            features['review_score_std'] = 0.0
            features['positive_review_ratio'] = 0.0
            features['negative_review_ratio'] = 0.0
            features['neutral_review_ratio'] = 1.0
        
        # 一貫性指標
        changes_authored = data.get('changes_authored', 0)
        changes_reviewed = data.get('changes_reviewed', 0)
        
        if changes_authored > 0 and changes_reviewed > 0:
            features['contribution_balance'] = min(changes_authored, changes_reviewed) / max(changes_authored, changes_reviewed)
        else:
            features['contribution_balance'] = 0.0
        
        return features
    
    def _extract_network_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """ネットワーク特徴量の抽出"""
        features = {}
        
        # プロジェクト参加パターン
        projects = data.get('projects', [])
        sources = data.get('sources', [])
        
        # 多様性指標
        features['project_diversity'] = len(set(projects)) if projects else 0
        features['source_diversity'] = len(set(sources)) if sources else 0
        
        # 集中度指標（特定プロジェクトへの集中度）
        if projects:
            project_counts = {}
            for project in projects:
                project_counts[project] = project_counts.get(project, 0) + 1
            
            max_project_count = max(project_counts.values())
            total_project_activities = sum(project_counts.values())
            features['project_concentration'] = max_project_count / total_project_activities
        else:
            features['project_concentration'] = 0.0
        
        return features
    
    def _extract_quality_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """品質特徴量の抽出"""
        features = {}
        
        # コード品質指標
        total_insertions = data.get('total_insertions', 0)
        total_deletions = data.get('total_deletions', 0)
        changes_authored = data.get('changes_authored', 0)
        
        if changes_authored > 0:
            features['avg_insertions_per_change'] = total_insertions / changes_authored
            features['avg_deletions_per_change'] = total_deletions / changes_authored
            
            # 変更サイズの一貫性
            if total_insertions + total_deletions > 0:
                features['change_size_consistency'] = min(total_insertions, total_deletions) / max(total_insertions, total_deletions)
            else:
                features['change_size_consistency'] = 0.0
        else:
            features['avg_insertions_per_change'] = 0.0
            features['avg_deletions_per_change'] = 0.0
            features['change_size_consistency'] = 0.0
        
        # レビュー品質指標
        review_scores = data.get('review_scores', [])
        changes_reviewed = data.get('changes_reviewed', 0)
        
        if changes_reviewed > 0 and review_scores:
            features['review_engagement'] = len(review_scores) / changes_reviewed
            features['review_quality_score'] = np.mean([abs(s) for s in review_scores])
        else:
            features['review_engagement'] = 0.0
            features['review_quality_score'] = 0.0
        
        return features
    
    def create_ensemble_models(self) -> Dict[str, Any]:
        """アンサンブルモデルの作成"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            ),
            'support_vector': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'ridge_regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'lasso_regression': Lasso(
                alpha=0.1,
                random_state=42
            )
        }
        
        return models
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_name: str, model: Any) -> Any:
        """ハイパーパラメータの最適化"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'neural_network': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'learning_rate_init': [0.001, 0.01, 0.1]
            },
            'support_vector': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso_regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        if model_name in param_grids:
            self.logger.info(f"{model_name}のハイパーパラメータ最適化を開始")
            
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            self.logger.info(f"{model_name}の最適パラメータ: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
        
        return model
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """アンサンブルモデルの訓練"""
        models = self.create_ensemble_models()
        trained_models = {}
        model_scores = {}
        
        # データの前処理
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        for name, model in models.items():
            self.logger.info(f"{name}モデルの訓練を開始")
            
            try:
                # ハイパーパラメータ最適化
                optimized_model = self.optimize_hyperparameters(X_scaled, y, name, model)
                
                # モデル訓練
                optimized_model.fit(X_scaled, y)
                trained_models[name] = optimized_model
                
                # クロスバリデーション評価
                cv_scores = cross_val_score(
                    optimized_model, X_scaled, y,
                    cv=5, scoring='neg_mean_squared_error'
                )
                model_scores[name] = -cv_scores.mean()
                
                self.logger.info(f"{name}: CV MSE = {model_scores[name]:.4f}")
                
                # 特徴量重要度の取得（可能な場合）
                if hasattr(optimized_model, 'feature_importances_'):
                    self.feature_importance[name] = optimized_model.feature_importances_
                elif hasattr(optimized_model, 'coef_'):
                    self.feature_importance[name] = np.abs(optimized_model.coef_)
                
            except Exception as e:
                self.logger.error(f"{name}モデルの訓練でエラー: {e}")
                continue
        
        # アンサンブル重みの計算（性能の逆数で重み付け）
        total_inverse_score = sum(1/score for score in model_scores.values())
        for name, score in model_scores.items():
            self.ensemble_weights[name] = (1/score) / total_inverse_score
        
        self.models = trained_models
        return trained_models
    
    def predict_with_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """アンサンブル予測"""
        if not self.models:
            raise ValueError("モデルが訓練されていません")
        
        # データの前処理
        X_scaled = self.scalers['main'].transform(X)
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
                weights.append(self.ensemble_weights.get(name, 1.0))
            except Exception as e:
                self.logger.error(f"{name}モデルの予測でエラー: {e}")
                continue
        
        if not predictions:
            raise ValueError("予測可能なモデルがありません")
        
        # 重み付き平均
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 正規化
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # 予測の不確実性（標準偏差）
        uncertainty = np.std(predictions, axis=0)
        
        return ensemble_pred, uncertainty
    
    def evaluate_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """モデル性能の評価"""
        results = {}
        
        # アンサンブル予測
        ensemble_pred, uncertainty = self.predict_with_ensemble(X_test)
        
        # 評価指標の計算
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        results['ensemble'] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_uncertainty': np.mean(uncertainty)
        }
        
        # 個別モデルの評価
        X_scaled = self.scalers['main'].transform(X_test)
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                
                results[name] = {
                    'mse': mean_squared_error(y_test, pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'mae': mean_absolute_error(y_test, pred),
                    'r2': r2_score(y_test, pred)
                }
            except Exception as e:
                self.logger.error(f"{name}モデルの評価でエラー: {e}")
                continue
        
        return results
    
    def analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """特徴量重要度の分析"""
        importance_analysis = {}
        
        # 各モデルの特徴量重要度を統合
        if self.feature_importance:
            # 重み付き平均による統合重要度
            weighted_importance = np.zeros(len(feature_names))
            total_weight = 0
            
            for model_name, importance in self.feature_importance.items():
                if model_name in self.ensemble_weights:
                    weight = self.ensemble_weights[model_name]
                    weighted_importance += importance * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_importance /= total_weight
            
            # 特徴量重要度のランキング
            importance_ranking = sorted(
                zip(feature_names, weighted_importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            importance_analysis['ranking'] = importance_ranking
            importance_analysis['top_10'] = importance_ranking[:10]
            
            # 重要度の統計
            importance_analysis['statistics'] = {
                'mean': np.mean(weighted_importance),
                'std': np.std(weighted_importance),
                'max': np.max(weighted_importance),
                'min': np.min(weighted_importance)
            }
        
        return importance_analysis
    
    def generate_improvement_recommendations(self, performance_results: Dict[str, Any], 
                                           feature_analysis: Dict[str, Any]) -> List[str]:
        """改善提案の生成"""
        recommendations = []
        
        # 性能ベースの提案
        ensemble_r2 = performance_results.get('ensemble', {}).get('r2', 0)
        
        if ensemble_r2 < 0.7:
            recommendations.append("予測精度が低いため、より多くの特徴量の追加を検討してください")
            recommendations.append("データの前処理方法を見直し、外れ値の処理を強化してください")
        
        if ensemble_r2 < 0.5:
            recommendations.append("現在のモデルでは十分な予測精度が得られていません。深層学習モデルの導入を検討してください")
        
        # 不確実性ベースの提案
        mean_uncertainty = performance_results.get('ensemble', {}).get('mean_uncertainty', 0)
        
        if mean_uncertainty > 0.3:
            recommendations.append("予測の不確実性が高いため、より多様なモデルをアンサンブルに追加してください")
            recommendations.append("データ収集期間を延長し、より多くの訓練データを確保してください")
        
        # 特徴量重要度ベースの提案
        if 'top_10' in feature_analysis:
            top_features = [name for name, _ in feature_analysis['top_10'][:3]]
            recommendations.append(f"最重要特徴量（{', '.join(top_features)}）に関連する新しい特徴量の追加を検討してください")
        
        # モデル固有の提案
        model_performances = {name: results.get('r2', 0) 
                            for name, results in performance_results.items() 
                            if name != 'ensemble'}
        
        if model_performances:
            best_model = max(model_performances, key=model_performances.get)
            worst_model = min(model_performances, key=model_performances.get)
            
            if model_performances[best_model] - model_performances[worst_model] > 0.2:
                recommendations.append(f"{best_model}が最も優秀な性能を示しています。このモデルの重みを増加させることを検討してください")
                recommendations.append(f"{worst_model}の性能が低いため、アンサンブルから除外することを検討してください")
        
        return recommendations
    
    def save_improvement_results(self, results: Dict[str, Any], output_path: str):
        """改善結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果の保存
        results_file = f"{output_path}/accuracy_improvement_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # NumPy配列をリストに変換
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # モデルの保存
        models_file = f"{output_path}/improved_models_{timestamp}.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance
            }, f)
        
        self.logger.info(f"改善結果を保存しました: {results_file}")
        self.logger.info(f"改善モデルを保存しました: {models_file}")
        
        return results_file, models_file
    
    def _make_serializable(self, obj):
        """オブジェクトをJSON serializable に変換"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

def main():
    """メイン実行関数"""
    config = {
        'data_path': 'data/processed/unified/all_developers.json',
        'output_path': 'outputs/accuracy_improvement',
        'test_size': 0.2,
        'random_state': 42
    }
    
    # 出力ディレクトリの作成
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    
    # システムの初期化
    improver = AdvancedAccuracyImprover(config)
    
    print("🚀 高度な予測精度改善システムを開始します...")
    print("=" * 60)
    
    return improver

if __name__ == "__main__":
    main()