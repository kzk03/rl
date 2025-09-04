#!/usr/bin/env python3
"""
開発者ごとの継続予測確率と精度評価スクリプト

目的:
 1. all_developers.json から開発者特徴を抽出
 2. "現在時点でまだ継続しているか" を擬似ラベル化
 3. WorkloadAwarePredictor で学習 / 予測し確率を出力
 4. 精度指標 (accuracy, precision, recall, f1, auc) を表示
 5. per-developer の確率とラベルを CSV / JSON 保存

ラベリング方針 (擬似):
 - dataset_end_date = データ中の last_activity 最大日
 - inactivity_days = dataset_end_date - last_activity
 - inactivity_days <= active_threshold_days (既定: 60) なら retention=1 (継続)
 - inactivity_days > churn_threshold_days (既定: 120) なら retention=0 (離脱)
 - 中間帯 (60 < days <=120) は曖昧 → optional: 今回は 0.5 の重み付け無視し単純化: 0 とするか除外
   -> デフォルトは 0 とする ( --drop-ambiguous で除外できる余地を後で追加可能 )

注意: 未来データが無い単一スナップショットでの近似なので“真の”継続離脱ではない。
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.prediction.workload_aware_predictor import (
    WorkloadAwarePredictor,
)

DATA_PATH = Path("data/processed/unified/all_developers.json")
OUTPUT_DIR = Path("outputs/retention_probability_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-threshold-days", type=int, default=60)
    ap.add_argument("--churn-threshold-days", type=int, default=120)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--drop-ambiguous", action="store_true", help="60<days<=120 を除外する")
    return ap.parse_args()


def load_developers() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} が存在しません")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 形式: list of developer dict (show_enhanced_examples と同形式を想定)
    return data


def to_datetime(val: str):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def build_labels(developers: List[Dict[str, Any]], active_threshold: int, churn_threshold: int, drop_ambiguous: bool):
    # dataset_end = max last_activity
    last_acts = []
    for d in developers:
        la = to_datetime(d.get("last_activity") or d.get("lastActivity"))
        if la:
            last_acts.append(la)
    if not last_acts:
        raise ValueError("last_activity が一件も取得できません")
    dataset_end = max(last_acts)

    labeled = []
    dropped = 0
    for d in developers:
        la = to_datetime(d.get("last_activity") or d.get("lastActivity"))
        if not la:
            continue
        inactivity_days = (dataset_end - la).days
        if inactivity_days <= active_threshold:
            label = 1
        elif inactivity_days > churn_threshold:
            label = 0
        else:
            if drop_ambiguous:
                dropped += 1
                continue
            label = 0  # 保守的に離脱扱い
        d['_retention_label'] = label
        d['_inactivity_days'] = inactivity_days
        d['_dataset_end'] = dataset_end.isoformat()
        labeled.append(d)
    return labeled, dataset_end, dropped


def flatten_for_predictor(dev: Dict[str, Any]) -> Dict[str, Any]:
    # predictor 用に expected keys をそのまま渡しつつ review_scores など補完
    out = dict(dev)
    # activity_history が無い場合は空
    if 'activity_history' not in out:
        out['activity_history'] = out.get('activities', []) or []
    return out


def main():
    args = parse_args()
    print("🚀 Retention probability evaluation start")
    developers = load_developers()
    labeled, dataset_end, dropped = build_labels(
        developers, args.active_threshold_days, args.churn_threshold_days, args.drop_ambiguous
    )
    print(f"総開発者: {len(developers)} / ラベル付与: {len(labeled)} (除外 {dropped}) dataset_end={dataset_end.date()}")

    # Flatten
    flat_devs = [flatten_for_predictor(d) for d in labeled]
    labels = [d['_retention_label'] for d in labeled]

    # Train/Test split
    X_train, X_test, y_train, y_test, dev_train, dev_test = train_test_split(
        flat_devs, labels, labeled, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )

    predictor = WorkloadAwarePredictor()
    predictor.fit(X_train, y_train)

    # Predict probabilities
    probs_train = predictor.predict_batch(X_train)
    probs_test = predictor.predict_batch(X_test)

    # Metrics
    def metrics(y_true, probs):
        preds = [1 if p >= 0.5 else 0 for p in probs]
        out = {
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
            'f1': f1_score(y_true, preds, zero_division=0),
        }
        try:
            out['auc'] = roc_auc_score(y_true, probs)
        except Exception:
            out['auc'] = float('nan')
        return out

    m_train = metrics(y_train, probs_train)
    m_test = metrics(y_test, probs_test)

    print("\n📊 Train Metrics:")
    for k,v in m_train.items():
        print(f"  {k}: {v:.4f}")
    print("\n📊 Test Metrics:")
    for k,v in m_test.items():
        print(f"  {k}: {v:.4f}")

    # Save per developer results
    results = []
    for dev, prob in zip(dev_test, probs_test):
        results.append({
            'developer_id': dev.get('developer_id'),
            'last_activity': dev.get('last_activity'),
            'inactivity_days': dev.get('_inactivity_days'),
            'label': dev.get('_retention_label'),
            'probability': prob
        })

    (OUTPUT_DIR / 'test_predictions.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    # Simple CSV
    import csv
    with (OUTPUT_DIR / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['developer_id','last_activity','inactivity_days','label','probability'])
        for r in results:
            w.writerow([r['developer_id'], r['last_activity'], r['inactivity_days'], r['label'], f"{r['probability']:.6f}"])

    print(f"\n💾 予測結果を保存しました: {OUTPUT_DIR}/test_predictions.json (and .csv)")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
