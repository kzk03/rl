#!/usr/bin/env python3
"""
既存モデルで評価のみを実行するスクリプト
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, 'scripts/training/irl')
sys.path.insert(0, 'src')

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from train_irl_review_acceptance import (
    extract_evaluation_trajectories,
    extract_review_acceptance_trajectories,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def evaluate_model(
    model_path: Path,
    reviews_data: Path,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    train_future_window_start: int,
    train_future_window_end: int,
    eval_future_window_start: int,
    eval_future_window_end: int,
    output_dir: Path,
    project: str = 'openstack/nova'
):
    """既存モデルで評価を実行"""
    
    # データ読み込み
    df = pd.read_csv(reviews_data)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    # モデル読み込み
    config = {
        'state_dim': 9,
        'action_dim': 4,
        'hidden_dim': 128,
        'sequence': True,
        'seq_len': 0,
        'use_temporal_features': False,
        'learning_rate': 0.001
    }
    irl = RetentionIRLSystem(config=config)
    irl.network.load_state_dict(torch.load(model_path))
    irl.network.eval()
    
    print(f"✅ モデル読み込み完了: {model_path}")
    
    # 評価用軌跡を抽出
    eval_trajectories = extract_evaluation_trajectories(
        df=df,
        cutoff_date=pd.Timestamp(train_end),
        history_window_months=12,
        future_window_start_months=eval_future_window_start,
        future_window_end_months=eval_future_window_end,
        min_history_requests=3,
        project=project
    )
    
    if not eval_trajectories:
        print("❌ 評価用軌跡が抽出できませんでした")
        return
    
    print(f"✅ 評価軌跡抽出完了: {len(eval_trajectories)}人")
    
    # 予測
    predictions = []
    true_labels = []
    
    for traj in eval_trajectories:
        result = irl.predict_continuation_probability_snapshot(
            developer=traj['developer'],  # 'developer_info' ではなく 'developer'
            activity_history=traj['activity_history'],
            context_date=pd.Timestamp(train_end)
        )
        predictions.append(result['continuation_probability'])
        true_labels.append(1.0 if traj['future_acceptance'] else 0.0)
    
    predictions = torch.tensor(predictions)
    true_labels = torch.tensor(true_labels)
    
    # メトリクス計算
    auc_roc = roc_auc_score(true_labels, predictions)
    auc_pr = average_precision_score(true_labels, predictions)
    
    # 最適閾値を計算
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # 閾値で二値化
    predicted_binary = (predictions >= optimal_threshold).float()
    
    # Precision, Recall, F1
    tp = ((predicted_binary == 1) & (true_labels == 1)).sum().item()
    fp = ((predicted_binary == 1) & (true_labels == 0)).sum().item()
    fn = ((predicted_binary == 0) & (true_labels == 1)).sum().item()
    
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    # 結果を保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int((true_labels == 1).sum()),
        'negative_count': int((true_labels == 0).sum()),
        'total_count': int(len(true_labels))
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Predictions CSV
    results_df = pd.DataFrame({
        'reviewer_email': [traj['developer']['developer_email'] for traj in eval_trajectories],
        'predicted_prob': predictions.tolist(),
        'true_label': true_labels.tolist(),
        'predicted_binary': predicted_binary.tolist(),
        'history_request_count': [traj['developer']['requests_received'] for traj in eval_trajectories],
        'history_acceptance_rate': [traj['developer']['acceptance_rate'] for traj in eval_trajectories]
    })
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    print(f"✅ 評価完了: AUC-PR {auc_pr:.4f}")
    print(f"   結果保存: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--reviews', type=str, required=True)
    parser.add_argument('--train-start', type=str, required=True)
    parser.add_argument('--train-end', type=str, required=True)
    parser.add_argument('--eval-start', type=str, required=True)
    parser.add_argument('--eval-end', type=str, required=True)
    parser.add_argument('--train-future-start', type=int, required=True)
    parser.add_argument('--train-future-end', type=int, required=True)
    parser.add_argument('--eval-future-start', type=int, required=True)
    parser.add_argument('--eval-future-end', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--project', type=str, default='openstack/nova')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=Path(args.model),
        reviews_data=Path(args.reviews),
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        train_future_window_start=args.train_future_start,
        train_future_window_end=args.train_future_end,
        eval_future_window_start=args.eval_future_start,
        eval_future_window_end=args.eval_future_end,
        output_dir=Path(args.output),
        project=args.project
    )

