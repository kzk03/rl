#!/usr/bin/env python3
"""簡易 実行スクリプト: 作業負荷・専門性考慮型 継続率モデル

extended_test_data.json を使って WorkloadAwarePredictor を訓練し
基本メトリクスと上位重要特徴量を表示する。

(小規模データなのでスコアは参考値)
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# プロジェクトルートをパスに追加（scripts/ から一段上）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.prediction.workload_aware_predictor import (
    WorkloadAwarePredictor,
)

DATA_PATH = Path("data/extended_test_data.json")


def build_label(dev_entry):
    """簡易ラベル: base_date から60日以内に activity_history で活動があれば 1, 無ければ0"""
    base_date_str = dev_entry.get("base_date") or dev_entry.get("developer", {}).get("base_date")
    if not base_date_str:
        return 1
    try:
        base_dt = datetime.fromisoformat(base_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return 1
    horizon = base_dt + timedelta(days=60)
    # activity_history 内に base_date 以降 horizon までの timestamp があればポジティブ
    for act in dev_entry.get("activity_history", []):
        ts = act.get("timestamp")
        if not ts:
            continue
        try:
            act_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if base_dt <= act_dt <= horizon:
            return 1
    return 0


def flatten_dev(dev_entry):
    dev = dict(dev_entry.get("developer", {}))
    # last_activity を activity_history 末尾から推定
    acts = dev_entry.get("activity_history", [])
    if acts:
        try:
            last_ts = max(a["timestamp"] for a in acts if a.get("timestamp"))
            dev["last_activity"] = last_ts
        except Exception:
            pass
    # review_scores 収集
    scores = [a.get("score") for a in acts if a.get("type") == "review" and isinstance(a.get("score"), (int, float))]
    if scores:
        dev["review_scores"] = scores
    dev["activity_history"] = acts
    return dev


def main():
    print("🚀 WorkloadAwarePredictor デモ開始")
    if not DATA_PATH.exists():
        print(f"❌ データが見つかりません: {DATA_PATH}")
        return 1

    raw = json.loads(DATA_PATH.read_text())
    developers = [flatten_dev(e) for e in raw]
    labels = [build_label(e) for e in raw]

    predictor = WorkloadAwarePredictor()
    predictor.fit(developers, labels)

    metrics = predictor.evaluate(developers, labels)
    print("\n📊 メトリクス:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n🔍 上位重要特徴量:")
    for name, imp in predictor.explain(developers[0]):
        print(f"  {name}: {imp:.4f}")

    # 各開発者の確率
    print("\n🎯 予測確率:")
    for dev, prob, label in zip(developers, predictor.predict_batch(developers), labels):
        print(f"  {dev.get('developer_id')}: prob={prob:.4f} label={label}")
    print("\n✅ 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
