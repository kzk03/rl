"""DEPRECATED BASELINE (RandomForest)

本クラスは IRL(MaxEntBinaryIRL) ベース手法への移行に伴いアーカイブ状態です。
今後のデフォルト推論は IRL。`--use-rf` フラグ指定時のみ比較用途で利用。
維持: 重大バグ修正のみ / 新機能追加なし (2025-09 時点)。

概要:
	作業負荷・専門性考慮型 継続率予測モジュール。
	WorkloadExpertiseAnalyzer / AdvancedAccuracyImprover が抽出する高度特徴
	(作業負荷, 専門性マッチ, バーンアウト指標 等) を統合し RandomForest で確率推定。

目的:
	1. 既存大型アンサンブルより軽量な専門性/負荷重視ベースライン
	2. 空プレースホルダの具体実装
	3. 将来別モデル差し替え容易性確保

最小インターフェース:
	- fit(developers, labels)
	- predict_proba(developer)
	- predict_batch(developers)
	- evaluate(developers, labels)
	- explain(developer)

留意点:
	- developer dict は extended_test_data.json 形式を想定
	- last_activity / review_scores が無い場合は activity_history から補完
	- ラベルは外部で生成 (一定期間内活動継続 → 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
	from .advanced_accuracy_improver import AdvancedAccuracyImprover
	from .workload_expertise_analyzer import WorkloadExpertiseAnalyzer
except Exception as e:  # pragma: no cover - フォールバック
	raise ImportError("必要な分析モジュールのインポートに失敗しました: {}".format(e))


logger = logging.getLogger(__name__)


@dataclass
class WorkloadAwareConfig:
	model_params: Dict[str, Any]
	top_feature_count: int = 10


class WorkloadAwarePredictor:
	"""作業負荷・専門性考慮型の継続率予測器"""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		config = config or {}
		self.config = WorkloadAwareConfig(
			model_params=config.get("model", {}).get(
				"random_forest_params",
				{"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 2, "random_state": 42},
			),
			top_feature_count=config.get("explain", {}).get("top_feature_count", 10),
		)

		self.accuracy_improver = AdvancedAccuracyImprover({"output_path": config.get("output_path", "outputs/workload_aware")})
		self.workload_analyzer = WorkloadExpertiseAnalyzer({"output_path": config.get("output_path", "outputs/workload_aware")})

		self.model = RandomForestClassifier(**self.config.model_params)
		self.is_fitted = False
		self.feature_names: List[str] = []

	# ------------------------------------------------------------------
	# 内部ユーティリティ
	# ------------------------------------------------------------------
	def _ensure_minimal_fields(self, dev: Dict[str, Any]) -> Dict[str, Any]:
		"""不足フィールドを activity_history などから補完"""
		dev = dict(dev)  # copy
		activity_history = dev.get("activity_history") or dev.get("activities") or []

		# last_activity 補完
		if not dev.get("last_activity") and activity_history:
			try:
				last_ts = max(a["timestamp"] for a in activity_history if a.get("timestamp"))
				dev["last_activity"] = last_ts
			except Exception:
				pass

		# review_scores 抽出
		if "review_scores" not in dev:
			scores = [a.get("score") for a in activity_history if a.get("type") == "review" and isinstance(a.get("score"), (int, float))]
			if scores:
				dev["review_scores"] = scores

		return dev

	def _extract_feature_vector(self, dev: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
		"""単一開発者の統合特徴量ベクトルと辞書を返す"""
		dev = self._ensure_minimal_fields(dev)

		advanced = self.accuracy_improver.extract_advanced_features(dev)
		workload = self.workload_analyzer.analyze_workload_features(dev)
		expertise = self.workload_analyzer.analyze_expertise_match(dev)
		burnout = self.workload_analyzer.calculate_burnout_risk(dev, workload, expertise)

		merged: Dict[str, float] = {}
		merged.update(advanced)
		merged.update(workload)
		merged.update(expertise)
		merged.update(burnout)

		# 初回学習時に特徴名固定 (ソートで順序安定)
		if not self.feature_names:
			self.feature_names = sorted(merged.keys())
		# 数値化: 非数値カテゴリーはマッピング
		categorical_map = {"minimal":0.0, "low":1.0, "medium":2.0, "high":3.0, "critical":4.0}
		def _to_float(v):
			if isinstance(v, (int, float)):
				return float(v)
			if isinstance(v, str):
				return float(categorical_map.get(v.lower(), 0.0))
			return 0.0
		vec = [_to_float(merged.get(name, 0.0)) for name in self.feature_names]
		return np.array(vec, dtype=float), merged

	# ------------------------------------------------------------------
	# 公開インターフェース
	# ------------------------------------------------------------------
	def fit(self, developers: Sequence[Dict[str, Any]], labels: Sequence[int]) -> None:
		if not developers:
			raise ValueError("developers が空です")
		if len(developers) != len(labels):
			raise ValueError("developers と labels の長さが一致しません")

		feature_vectors = []
		for dev in developers:
			vec, _ = self._extract_feature_vector(dev)
			feature_vectors.append(vec)

		X = np.vstack(feature_vectors)
		y = np.array(labels, dtype=int)

		logger.info(f"[WorkloadAwarePredictor] 訓練開始: samples={X.shape[0]}, features={X.shape[1]}")
		self.model.fit(X, y)
		self.is_fitted = True
		logger.info("[WorkloadAwarePredictor] 訓練完了")

	def predict_proba(self, developer: Dict[str, Any]) -> float:
		if not self.is_fitted:
			raise RuntimeError("モデルが未訓練です。fit() を先に呼んでください")
		vec, _ = self._extract_feature_vector(developer)
		prob = self.model.predict_proba(vec.reshape(1, -1))[0, 1]
		return float(prob)

	def predict_batch(self, developers: Sequence[Dict[str, Any]]) -> List[float]:
		return [self.predict_proba(d) for d in developers]

	def evaluate(self, developers: Sequence[Dict[str, Any]], labels: Sequence[int]) -> Dict[str, float]:
		probs = self.predict_batch(developers)
		preds = [1 if p >= 0.5 else 0 for p in probs]
		metrics = {
			"accuracy": accuracy_score(labels, preds),
			"precision": precision_score(labels, preds, zero_division=0),
			"recall": recall_score(labels, preds, zero_division=0),
			"f1": f1_score(labels, preds, zero_division=0),
		}
		try:
			metrics["auc"] = roc_auc_score(labels, probs)
		except Exception:
			metrics["auc"] = float("nan")
		return metrics

	def explain(self, developer: Dict[str, Any], top_n: Optional[int] = None) -> List[Tuple[str, float]]:
		if not self.is_fitted:
			raise RuntimeError("モデルが未訓練です")
		if not hasattr(self.model, "feature_importances_"):
			return []
		top_n = top_n or self.config.top_feature_count
		importances = self.model.feature_importances_
		pairs = list(zip(self.feature_names, importances))
		pairs.sort(key=lambda x: x[1], reverse=True)
		return pairs[:top_n]


def create_workload_aware_predictor(config: Optional[Dict[str, Any]] = None) -> WorkloadAwarePredictor:
	return WorkloadAwarePredictor(config)


__all__ = ["WorkloadAwarePredictor", "create_workload_aware_predictor"]

