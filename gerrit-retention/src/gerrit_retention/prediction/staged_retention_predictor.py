"""
段階的継続予測システム

複数の時間軸で開発者の継続確率を予測し、時系列での変化を追跡する。
短期（1週間）から長期（1年）まで、段階的な予測を提供する。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .dynamic_threshold_calculator import DynamicThresholdCalculator
from .retention_predictor import RetentionPredictor

logger = logging.getLogger(__name__)


class StagedRetentionPredictor:
    """段階的継続予測器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 予測段階の定義
        self.prediction_stages = config.get('prediction_stages', {
            'immediate': 7,      # 1週間後
            'short_term': 30,    # 1ヶ月後
            'medium_term': 90,   # 3ヶ月後
            'long_term': 180,    # 6ヶ月後
            'extended': 365      # 1年後
        })
        
        # 各段階用のモデル
        self.stage_models = {}
        self.stage_feature_extractors = {}
        
        # 動的閾値計算器
        self.threshold_calculator = DynamicThresholdCalculator(
            config.get('dynamic_threshold', {})
        )
        
        # 時系列予測用の設定
        self.time_series_features = config.get('time_series_features', True)
        self.trend_analysis_window = config.get('trend_analysis_window', 30)  # 30日
        
        # モデル設定
        self.model_config = config.get('model_config', {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        })
        
        self.is_fitted = False
        
        logger.info(f"段階的継続予測器を初期化しました（段階数: {len(self.prediction_stages)}）")
    
    def fit(self, 
            training_data: List[Dict[str, Any]], 
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        段階別モデルを訓練
        
        Args:
            training_data: 訓練データ（開発者情報、活動履歴、継続ラベルを含む）
            validation_split: 検証データの割合
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info("段階的継続予測モデルの訓練を開始...")
        
        training_results = {}
        
        # 各段階のモデルを訓練
        for stage_name, horizon_days in self.prediction_stages.items():
            logger.info(f"{stage_name}段階（{horizon_days}日後）のモデルを訓練中...")
            
            try:
                # 段階固有の特徴量とラベルを準備
                X, y = self._prepare_stage_data(training_data, horizon_days)
                
                if len(X) == 0:
                    logger.warning(f"{stage_name}段階のデータが不足しています")
                    continue
                
                # 時系列分割で訓練・検証データを分割
                tscv = TimeSeriesSplit(n_splits=3)
                
                # モデルを初期化
                model = RandomForestClassifier(**self.model_config)
                
                # 交差検証で性能評価
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    
                    if len(np.unique(y_val)) > 1:  # 両クラスが存在する場合のみAUC計算
                        score = roc_auc_score(y_val, y_pred_proba)
                        cv_scores.append(score)
                
                # 全データでモデルを再訓練
                model.fit(X, y)
                
                # モデルを保存
                self.stage_models[stage_name] = model
                
                # 特徴量名を保存
                feature_names = self._get_feature_names(horizon_days)
                self.stage_feature_extractors[stage_name] = feature_names
                
                # 結果を記録
                training_results[stage_name] = {
                    'horizon_days': horizon_days,
                    'training_samples': len(X),
                    'feature_count': X.shape[1],
                    'cv_auc_mean': np.mean(cv_scores) if cv_scores else 0.0,
                    'cv_auc_std': np.std(cv_scores) if cv_scores else 0.0,
                    'feature_importance': dict(zip(
                        feature_names, 
                        model.feature_importances_
                    ))
                }
                
                logger.info(f"{stage_name}段階の訓練完了 - AUC: {np.mean(cv_scores):.3f}")
                
            except Exception as e:
                logger.error(f"{stage_name}段階の訓練でエラー: {e}")
                training_results[stage_name] = {'error': str(e)}
        
        self.is_fitted = True
        
        logger.info("段階的継続予測モデルの訓練が完了しました")
        
        return training_results
    
    def predict_staged_retention(self, 
                               developer: Dict[str, Any], 
                               activity_history: List[Dict[str, Any]],
                               context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        段階的継続確率を予測
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日
            
        Returns:
            Dict[str, Any]: 段階別継続確率
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません。先にfit()を呼び出してください。")
        
        if context_date is None:
            context_date = datetime.now()
        
        developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
        
        logger.debug(f"段階的継続予測開始: {developer_id}")
        
        # 動的閾値を計算
        dynamic_threshold = self.threshold_calculator.calculate_dynamic_threshold(
            developer, activity_history, context_date
        )
        
        # 各段階の予測
        stage_predictions = {}
        
        for stage_name, horizon_days in self.prediction_stages.items():
            if stage_name not in self.stage_models:
                logger.warning(f"{stage_name}段階のモデルが存在しません")
                stage_predictions[stage_name] = {
                    'probability': 0.5,
                    'confidence': 0.0,
                    'error': 'model_not_available'
                }
                continue
            
            try:
                # 段階固有の特徴量を抽出
                features = self._extract_stage_features(
                    developer, activity_history, horizon_days, context_date
                )
                
                # 予測実行
                model = self.stage_models[stage_name]
                probability = model.predict_proba(features.reshape(1, -1))[0][1]
                
                # 信頼度計算
                confidence = self._calculate_prediction_confidence(
                    model, features, stage_name
                )
                
                # 動的閾値に基づく調整
                adjusted_probability = self._adjust_probability_with_threshold(
                    probability, dynamic_threshold, horizon_days
                )
                
                stage_predictions[stage_name] = {
                    'horizon_days': horizon_days,
                    'probability': float(adjusted_probability),
                    'raw_probability': float(probability),
                    'confidence': float(confidence),
                    'dynamic_threshold_days': dynamic_threshold['threshold_days']
                }
                
            except Exception as e:
                logger.error(f"{stage_name}段階の予測でエラー: {e}")
                stage_predictions[stage_name] = {
                    'probability': 0.5,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        # 予測の一貫性チェック
        consistency_score = self._check_prediction_consistency(stage_predictions)
        
        # トレンド分析
        trend_analysis = self._analyze_retention_trend(stage_predictions)
        
        result = {
            'developer_id': developer_id,
            'prediction_date': context_date.isoformat(),
            'stage_predictions': stage_predictions,
            'dynamic_threshold': dynamic_threshold,
            'consistency_score': consistency_score,
            'trend_analysis': trend_analysis,
            'overall_risk_level': self._calculate_overall_risk_level(stage_predictions)
        }
        
        logger.debug(f"段階的継続予測完了: {developer_id}")
        
        return result
    
    def _prepare_stage_data(self, 
                          training_data: List[Dict[str, Any]], 
                          horizon_days: int) -> Tuple[np.ndarray, np.ndarray]:
        """段階固有の訓練データを準備"""
        
        X_list = []
        y_list = []
        
        for data_point in training_data:
            try:
                developer = data_point['developer']
                activity_history = data_point['activity_history']
                
                # 基準日から horizon_days 後の継続状況を確認
                base_date = data_point.get('base_date')
                if isinstance(base_date, str):
                    base_date = datetime.fromisoformat(base_date)
                
                target_date = base_date + timedelta(days=horizon_days)
                
                # 特徴量を抽出
                features = self._extract_stage_features(
                    developer, activity_history, horizon_days, base_date
                )
                
                # ラベルを取得（target_date時点での継続状況）
                label = self._get_retention_label_at_date(
                    data_point, target_date
                )
                
                if label is not None:
                    X_list.append(features)
                    y_list.append(label)
                
            except Exception as e:
                logger.warning(f"訓練データ準備でエラー: {e}")
                continue
        
        if not X_list:
            return np.array([]), np.array([])
        
        return np.array(X_list), np.array(y_list)
    
    def _extract_stage_features(self, 
                              developer: Dict[str, Any], 
                              activity_history: List[Dict[str, Any]], 
                              horizon_days: int,
                              context_date: datetime) -> np.ndarray:
        """段階固有の特徴量を抽出"""
        
        features = []
        
        # 基本的な開発者特徴量
        features.extend(self._extract_basic_developer_features(developer))
        
        # 活動パターン特徴量
        features.extend(self._extract_activity_pattern_features(
            activity_history, context_date
        ))
        
        # 時系列特徴量
        if self.time_series_features:
            features.extend(self._extract_time_series_features(
                activity_history, context_date, horizon_days
            ))
        
        # 段階固有の特徴量
        features.extend(self._extract_horizon_specific_features(
            developer, activity_history, horizon_days, context_date
        ))
        
        return np.array(features)
    
    def _extract_basic_developer_features(self, developer: Dict[str, Any]) -> List[float]:
        """基本的な開発者特徴量を抽出"""
        
        features = []
        
        # 活動量
        features.append(float(developer.get('changes_authored', 0)))
        features.append(float(developer.get('changes_reviewed', 0)))
        features.append(float(developer.get('total_insertions', 0)))
        features.append(float(developer.get('total_deletions', 0)))
        
        # プロジェクト関与
        projects = developer.get('projects', [])
        features.append(float(len(projects) if isinstance(projects, list) else 0))
        
        # 経験期間
        first_seen = developer.get('first_seen')
        if first_seen:
            try:
                if isinstance(first_seen, str):
                    first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                else:
                    first_date = first_seen
                experience_days = (datetime.now() - first_date).days
                features.append(float(experience_days))
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_activity_pattern_features(self, 
                                         activity_history: List[Dict[str, Any]], 
                                         context_date: datetime) -> List[float]:
        """活動パターン特徴量を抽出"""
        
        features = []
        
        if not activity_history:
            return [0.0] * 8  # デフォルト特徴量数
        
        # 最近の活動頻度（過去30日）
        recent_cutoff = context_date - timedelta(days=30)
        recent_activities = [
            a for a in activity_history 
            if self._parse_activity_date(a) >= recent_cutoff
        ]
        features.append(float(len(recent_activities)))
        
        # 活動の一貫性（標準偏差）
        activity_dates = [self._parse_activity_date(a) for a in activity_history]
        activity_dates = [d for d in activity_dates if d is not None]
        
        if len(activity_dates) > 1:
            activity_dates.sort()
            gaps = [(activity_dates[i] - activity_dates[i-1]).days 
                   for i in range(1, len(activity_dates))]
            features.append(float(np.std(gaps)))
            features.append(float(np.mean(gaps)))
            features.append(float(max(gaps)))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 最後の活動からの経過日数
        if activity_dates:
            last_activity = max(activity_dates)
            days_since_last = (context_date - last_activity).days
            features.append(float(days_since_last))
        else:
            features.append(999.0)  # 大きな値
        
        # 活動の多様性（異なる活動タイプの数）
        activity_types = set()
        for activity in activity_history:
            activity_type = activity.get('type', activity.get('action', 'unknown'))
            activity_types.add(activity_type)
        features.append(float(len(activity_types)))
        
        # 週末活動の割合
        weekend_activities = 0
        for activity in activity_history:
            activity_date = self._parse_activity_date(activity)
            if activity_date and activity_date.weekday() >= 5:  # 土日
                weekend_activities += 1
        
        weekend_ratio = weekend_activities / len(activity_history) if activity_history else 0
        features.append(float(weekend_ratio))
        
        # 夜間活動の割合（18時以降）
        night_activities = 0
        for activity in activity_history:
            activity_date = self._parse_activity_date(activity)
            if activity_date and activity_date.hour >= 18:
                night_activities += 1
        
        night_ratio = night_activities / len(activity_history) if activity_history else 0
        features.append(float(night_ratio))
        
        return features
    
    def _extract_time_series_features(self, 
                                    activity_history: List[Dict[str, Any]], 
                                    context_date: datetime,
                                    horizon_days: int) -> List[float]:
        """時系列特徴量を抽出"""
        
        features = []
        
        # 複数の時間窓での活動トレンド
        windows = [7, 14, 30, 60]  # 1週間、2週間、1ヶ月、2ヶ月
        
        for window_days in windows:
            window_start = context_date - timedelta(days=window_days)
            window_activities = [
                a for a in activity_history 
                if self._parse_activity_date(a) >= window_start
            ]
            features.append(float(len(window_activities)))
        
        # 活動の加速度（最近の変化率）
        if len(windows) >= 2:
            recent_rate = features[-2] / max(windows[-2], 1)  # 最近2週間の日次活動率
            past_rate = features[-1] / max(windows[-1], 1)    # 過去2ヶ月の日次活動率
            acceleration = recent_rate - past_rate
            features.append(float(acceleration))
        else:
            features.append(0.0)
        
        # 季節性指標
        month = context_date.month
        features.append(float(month))
        features.append(float(np.sin(2 * np.pi * month / 12)))  # 月の周期性
        features.append(float(np.cos(2 * np.pi * month / 12)))
        
        # 予測期間の長さ（正規化）
        features.append(float(horizon_days / 365.0))  # 1年で正規化
        
        return features
    
    def _extract_horizon_specific_features(self, 
                                         developer: Dict[str, Any], 
                                         activity_history: List[Dict[str, Any]], 
                                         horizon_days: int,
                                         context_date: datetime) -> List[float]:
        """予測期間固有の特徴量を抽出"""
        
        features = []
        
        # 予測期間に応じた重み付け特徴量
        if horizon_days <= 30:  # 短期予測
            # 最近の活動パターンを重視
            recent_window = min(horizon_days, 14)
            recent_cutoff = context_date - timedelta(days=recent_window)
            recent_activities = [
                a for a in activity_history 
                if self._parse_activity_date(a) >= recent_cutoff
            ]
            features.append(float(len(recent_activities)))
            
            # 最近のコミット頻度
            recent_commits = [
                a for a in recent_activities 
                if a.get('type', '').lower() in ['commit', 'change', 'push']
            ]
            features.append(float(len(recent_commits)))
            
        elif horizon_days <= 90:  # 中期予測
            # 月次パターンを重視
            monthly_activities = []
            for i in range(3):  # 過去3ヶ月
                month_start = context_date - timedelta(days=(i+1)*30)
                month_end = context_date - timedelta(days=i*30)
                month_activities = [
                    a for a in activity_history 
                    if month_start <= self._parse_activity_date(a) < month_end
                ]
                monthly_activities.append(len(month_activities))
            
            features.extend([float(x) for x in monthly_activities])
            
            # 月次活動の安定性
            if monthly_activities:
                features.append(float(np.std(monthly_activities)))
            else:
                features.append(0.0)
            
        else:  # 長期予測
            # 長期トレンドを重視
            quarterly_activities = []
            for i in range(4):  # 過去4四半期
                quarter_start = context_date - timedelta(days=(i+1)*90)
                quarter_end = context_date - timedelta(days=i*90)
                quarter_activities = [
                    a for a in activity_history 
                    if quarter_start <= self._parse_activity_date(a) < quarter_end
                ]
                quarterly_activities.append(len(quarter_activities))
            
            features.extend([float(x) for x in quarterly_activities])
            
            # 長期トレンド（線形回帰の傾き）
            if len(quarterly_activities) > 1:
                x = np.arange(len(quarterly_activities))
                slope = np.polyfit(x, quarterly_activities, 1)[0]
                features.append(float(slope))
            else:
                features.append(0.0)
        
        return features
    
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
                # ISO形式の場合
                if 'T' in date_str:
                    date_str = date_str.replace('Z', '+00:00')
                    return datetime.fromisoformat(date_str)
                else:
                    return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                return date_str
                
        except (ValueError, TypeError):
            return None
    
    def _get_retention_label_at_date(self, 
                                   data_point: Dict[str, Any], 
                                   target_date: datetime) -> Optional[int]:
        """指定日時点での継続ラベルを取得"""
        
        # 実装は具体的なデータ構造に依存
        # ここでは簡単な例を示す
        
        retention_history = data_point.get('retention_history', {})
        
        # target_date以降の活動があるかチェック
        activity_history = data_point.get('activity_history', [])
        
        future_activities = [
            a for a in activity_history 
            if self._parse_activity_date(a) >= target_date
        ]
        
        # 活動があれば継続、なければ離脱
        return 1 if future_activities else 0
    
    def _get_feature_names(self, horizon_days: int) -> List[str]:
        """特徴量名のリストを取得"""
        
        feature_names = []
        
        # 基本特徴量
        feature_names.extend([
            'changes_authored', 'changes_reviewed', 'total_insertions', 
            'total_deletions', 'project_count', 'experience_days'
        ])
        
        # 活動パターン特徴量
        feature_names.extend([
            'recent_activities_30d', 'activity_gap_std', 'activity_gap_mean',
            'max_activity_gap', 'days_since_last_activity', 'activity_type_diversity',
            'weekend_activity_ratio', 'night_activity_ratio'
        ])
        
        # 時系列特徴量
        if self.time_series_features:
            feature_names.extend([
                'activities_7d', 'activities_14d', 'activities_30d', 'activities_60d',
                'activity_acceleration', 'month', 'month_sin', 'month_cos',
                'horizon_normalized'
            ])
        
        # 段階固有特徴量
        if horizon_days <= 30:
            feature_names.extend(['recent_activities_window', 'recent_commits'])
        elif horizon_days <= 90:
            feature_names.extend([
                'month1_activities', 'month2_activities', 'month3_activities',
                'monthly_activity_std'
            ])
        else:
            feature_names.extend([
                'quarter1_activities', 'quarter2_activities', 
                'quarter3_activities', 'quarter4_activities',
                'quarterly_trend_slope'
            ])
        
        return feature_names
    
    def _calculate_prediction_confidence(self, 
                                       model: RandomForestClassifier, 
                                       features: np.ndarray, 
                                       stage_name: str) -> float:
        """予測の信頼度を計算"""
        
        try:
            # 決定木の予測の一致度を信頼度として使用
            tree_predictions = []
            for tree in model.estimators_:
                pred = tree.predict_proba(features.reshape(1, -1))[0][1]
                tree_predictions.append(pred)
            
            # 予測の分散が小さいほど信頼度が高い
            prediction_std = np.std(tree_predictions)
            confidence = max(0.1, 1.0 - prediction_std)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"信頼度計算でエラー: {e}")
            return 0.5
    
    def _adjust_probability_with_threshold(self, 
                                         probability: float, 
                                         dynamic_threshold: Dict[str, Any], 
                                         horizon_days: int) -> float:
        """動的閾値に基づいて確率を調整"""
        
        threshold_days = dynamic_threshold.get('threshold_days', 90)
        confidence = dynamic_threshold.get('confidence', 0.5)
        
        # 閾値の信頼度が高い場合、より強く調整
        adjustment_strength = confidence * 0.2  # 最大20%の調整
        
        if horizon_days > threshold_days:
            # 予測期間が閾値より長い場合、継続確率を下げる
            adjusted_probability = probability * (1 - adjustment_strength)
        else:
            # 予測期間が閾値より短い場合、継続確率を上げる
            adjusted_probability = probability + (1 - probability) * adjustment_strength
        
        return max(0.0, min(adjusted_probability, 1.0))
    
    def _check_prediction_consistency(self, 
                                    stage_predictions: Dict[str, Any]) -> float:
        """予測の一貫性をチェック"""
        
        probabilities = []
        horizons = []
        
        for stage_name, prediction in stage_predictions.items():
            if 'probability' in prediction and 'horizon_days' in prediction:
                probabilities.append(prediction['probability'])
                horizons.append(prediction['horizon_days'])
        
        if len(probabilities) < 2:
            return 1.0  # 比較できない場合は完全一貫とする
        
        # 一般的に、期間が長いほど継続確率は下がるべき
        # この傾向との一致度を計算
        
        # 期間と確率の相関を計算
        correlation = np.corrcoef(horizons, probabilities)[0, 1]
        
        # 負の相関（期間が長いほど確率が低い）が理想的
        if np.isnan(correlation):
            consistency = 0.5
        else:
            # -1（完全な負の相関）に近いほど一貫性が高い
            consistency = max(0.0, -correlation)
        
        return consistency
    
    def _analyze_retention_trend(self, 
                               stage_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """継続確率のトレンドを分析"""
        
        probabilities = []
        horizons = []
        
        for stage_name, prediction in stage_predictions.items():
            if 'probability' in prediction and 'horizon_days' in prediction:
                probabilities.append(prediction['probability'])
                horizons.append(prediction['horizon_days'])
        
        if len(probabilities) < 2:
            return {'trend': 'unknown', 'slope': 0.0, 'r_squared': 0.0}
        
        # 線形回帰でトレンドを分析
        try:
            coeffs = np.polyfit(horizons, probabilities, 1)
            slope = coeffs[0]
            
            # R²を計算
            y_pred = np.polyval(coeffs, horizons)
            ss_res = np.sum((probabilities - y_pred) ** 2)
            ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # トレンドの分類
            if slope < -0.001:
                trend = 'declining'  # 継続確率が時間とともに低下
            elif slope > 0.001:
                trend = 'improving'  # 継続確率が時間とともに向上（異常）
            else:
                trend = 'stable'     # 安定
            
            return {
                'trend': trend,
                'slope': float(slope),
                'r_squared': float(r_squared)
            }
            
        except Exception as e:
            logger.warning(f"トレンド分析でエラー: {e}")
            return {'trend': 'error', 'slope': 0.0, 'r_squared': 0.0}
    
    def _calculate_overall_risk_level(self, 
                                    stage_predictions: Dict[str, Any]) -> str:
        """総合的なリスクレベルを計算"""
        
        probabilities = []
        
        for prediction in stage_predictions.values():
            if 'probability' in prediction:
                probabilities.append(prediction['probability'])
        
        if not probabilities:
            return 'unknown'
        
        # 短期・中期・長期の重み付き平均
        weights = {'immediate': 0.4, 'short_term': 0.3, 'medium_term': 0.2, 'long_term': 0.1}
        
        weighted_prob = 0.0
        total_weight = 0.0
        
        for stage_name, prediction in stage_predictions.items():
            if 'probability' in prediction:
                weight = weights.get(stage_name, 0.1)
                weighted_prob += prediction['probability'] * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_probability = weighted_prob / total_weight
        else:
            avg_probability = np.mean(probabilities)
        
        # リスクレベルの分類
        if avg_probability >= 0.8:
            return 'low'        # 低リスク
        elif avg_probability >= 0.6:
            return 'moderate'   # 中リスク
        elif avg_probability >= 0.4:
            return 'high'       # 高リスク
        else:
            return 'critical'   # 危険
    
    def batch_predict_staged_retention(self, 
                                     developers_with_history: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
                                     context_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """複数開発者の段階的継続予測を一括実行"""
        
        if context_date is None:
            context_date = datetime.now()
        
        logger.info(f"段階的継続予測一括実行開始: {len(developers_with_history)}人")
        
        results = {}
        
        for developer, activity_history in developers_with_history:
            developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
            
            try:
                prediction = self.predict_staged_retention(
                    developer, activity_history, context_date
                )
                results[developer_id] = prediction
                
            except Exception as e:
                logger.error(f"段階的継続予測エラー ({developer_id}): {e}")
                results[developer_id] = {
                    'error': str(e),
                    'stage_predictions': {},
                    'overall_risk_level': 'unknown'
                }
        
        logger.info(f"段階的継続予測一括実行完了: {len(results)}人")
        
        return results
    
    def save_models(self, filepath: str) -> None:
        """段階別モデルを保存"""
        
        model_data = {
            'stage_models': self.stage_models,
            'stage_feature_extractors': self.stage_feature_extractors,
            'prediction_stages': self.prediction_stages,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"段階的継続予測モデルを保存しました: {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """段階別モデルを読み込み"""
        
        model_data = joblib.load(filepath)
        
        self.stage_models = model_data['stage_models']
        self.stage_feature_extractors = model_data['stage_feature_extractors']
        self.prediction_stages = model_data['prediction_stages']
        self.config = model_data.get('config', {})
        self.is_fitted = model_data['is_fitted']
        
        # 動的閾値計算器を再初期化
        self.threshold_calculator = DynamicThresholdCalculator(
            self.config.get('dynamic_threshold', {})
        )
        
        logger.info(f"段階的継続予測モデルを読み込みました: {filepath}")


if __name__ == "__main__":
    # テスト用のサンプル設定
    sample_config = {
        'prediction_stages': {
            'immediate': 7,
            'short_term': 30,
            'medium_term': 90,
            'long_term': 180
        },
        'dynamic_threshold': {
            'min_threshold_days': 14,
            'max_threshold_days': 365,
            'default_threshold_days': 90
        },
        'time_series_features': True,
        'model_config': {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
    }
    
    predictor = StagedRetentionPredictor(sample_config)
    
    print("段階的継続予測器のテスト完了")
    print(f"予測段階: {list(predictor.prediction_stages.keys())}")