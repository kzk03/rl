#!/usr/bin/env python3
"""
Random Forest - 4×4クロス評価（importants準拠）

importantsと同じデータ準備方式で、Random Forestで評価
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# プロジェクトルート
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.nova_review_acceptance.scripts.train_enhanced_irl_importants import (
    prepare_trajectories_importants_style,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 結果ディレクトリ
RESULT_DIR = Path(__file__).parent.parent / "results_random_forest"
RESULT_DIR.mkdir(exist_ok=True)

# データ
DATA_PATH = ROOT / "data" / "review_requests_openstack_multi_5y_detail.csv"

# 訓練期間と評価期間の定義（importantsと同一）
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# 基準日（importantsと同一）
TRAIN_START = pd.to_datetime("2021-01-01")
TRAIN_END = pd.to_datetime("2023-01-01")
EVAL_START = pd.to_datetime("2023-01-01")
EVAL_END = pd.to_datetime("2024-01-01")


def get_month_offset(period_str):
    """期間文字列から月数オフセットを取得"""
    start, end = period_str.split('-')
    start_month = int(start.replace('m', ''))
    end_month = int(end.replace('m', ''))
    return start_month, end_month


def extract_features_from_trajectory(traj):
    """軌跡から特徴量を抽出（developer_infoのみ使用、10次元に統一）"""
    dev_info = traj['developer_info']
    activity_history = traj.get('activity_history', [])
    
    # 10次元特徴量（Enhanced IRLと同等）
    requests_received = dev_info.get('requests_received', 0)
    requests_accepted = dev_info.get('requests_accepted', 0)
    requests_rejected = dev_info.get('requests_rejected', 0)
    acceptance_rate = dev_info.get('acceptance_rate', 0.0)
    
    # 活動履歴から追加特徴量を計算
    activity_count = len(activity_history)
    recent_activities = [a for a in activity_history[-10:]] if len(activity_history) > 0 else []
    recent_accepted = len([a for a in recent_activities if a.get('accepted', False)])
    recent_acceptance_rate = recent_accepted / len(recent_activities) if len(recent_activities) > 0 else 0.0
    
    features = [
        requests_received,
        requests_accepted,
        requests_rejected,
        acceptance_rate,
        dev_info.get('changes_authored', 0),
        dev_info.get('changes_reviewed', 0),
        activity_count,  # 総活動数
        recent_acceptance_rate,  # 直近10件の受諾率
        len(dev_info.get('projects', [])),  # プロジェクト数
        1.0 if requests_received > 0 else 0.0,  # アクティブかどうか
    ]
    
    return features


def train_and_evaluate_rf(train_trajectories, eval_trajectories, output_dir):
    """Random Forestで訓練・評価"""
    
    # 特徴量とラベル抽出（最後のステップのラベルを予測）
    X_train = np.array([extract_features_from_trajectory(t) for t in train_trajectories])
    y_train = np.array([
        t['step_labels'][-1] if len(t['step_labels']) > 0 else 0
        for t in train_trajectories
    ])
    
    X_eval = np.array([extract_features_from_trajectory(t) for t in eval_trajectories])
    y_eval = np.array([
        t['step_labels'][-1] if len(t['step_labels']) > 0 else 0
        for t in eval_trajectories
    ])
    
    # Random Forest訓練
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # 評価
    eval_probs = rf.predict_proba(X_eval)[:, 1] if len(set(y_eval)) > 1 else np.zeros(len(y_eval))
    
    if len(set(y_eval)) > 1:
        auc_roc = roc_auc_score(y_eval, eval_probs)
        
        precision, recall, thresholds = precision_recall_curve(y_eval, eval_probs)
        auc_pr = calc_auc(recall, precision)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        eval_preds = (eval_probs >= optimal_threshold).astype(int)
        precision_val = precision_score(y_eval, eval_preds, zero_division=0)
        recall_val = recall_score(y_eval, eval_preds, zero_division=0)
        f1_val = f1_score(y_eval, eval_preds, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        optimal_threshold = 0.5
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int(y_eval.sum()),
        'negative_count': int((1 - y_eval).sum()),
        'total_count': len(y_eval)
    }
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    logger.info("=" * 80)
    logger.info("Random Forest - 4×4クロス評価（importants準拠）")
    logger.info("=" * 80)
    
    # データ読み込み
    df = pd.read_csv(DATA_PATH)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    results_summary = []
    
    # 4×4評価
    for train_period in train_periods:
        train_start_month, train_end_month = get_month_offset(train_period)
        
        # 訓練データ準備
        train_trajectories = prepare_trajectories_importants_style(
            df, TRAIN_START, TRAIN_END,
            train_start_month, train_end_month,
            min_history_requests=3,
            project="openstack/nova"
        )
        
        for eval_period in eval_periods:
            eval_start_month, eval_end_month = get_month_offset(eval_period)
            
            output_dir = RESULT_DIR / f"train_{train_period}" / f"eval_{eval_period}"
            
            if (output_dir / "metrics.json").exists():
                logger.info(f"✅ スキップ: {train_period} -> {eval_period}")
                continue
            
            logger.info(f"🔄 評価中: {train_period} -> {eval_period}")
            
            # 評価データ準備
            eval_trajectories = prepare_trajectories_importants_style(
                df, EVAL_START, EVAL_END,
                eval_start_month, eval_end_month,
                min_history_requests=3,
                project="openstack/nova"
            )
            
            # 訓練・評価
            metrics = train_and_evaluate_rf(train_trajectories, eval_trajectories, output_dir)
            
            logger.info(f"✅ {train_period} -> {eval_period}: AUC-ROC={metrics['auc_roc']:.4f}")
            
            results_summary.append({
                'train': train_period,
                'eval': eval_period,
                **metrics
            })
    
    # 対角線サマリー
    logger.info("")
    logger.info("=" * 80)
    logger.info("結果サマリー（Random Forest - 対角線評価）")
    logger.info("=" * 80)
    
    diagonal_aucs = []
    for period in train_periods:
        metrics_file = RESULT_DIR / f"train_{period}" / f"eval_{period}" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            logger.info(f"{period}: AUC-ROC={data['auc_roc']:.4f}, AUC-PR={data['auc_pr']:.4f}, F1={data['f1_score']:.4f}")
            diagonal_aucs.append(data['auc_roc'])
    
    if diagonal_aucs:
        logger.info(f"\n平均AUC-ROC: {np.mean(diagonal_aucs):.4f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info(f"結果ディレクトリ: {RESULT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
