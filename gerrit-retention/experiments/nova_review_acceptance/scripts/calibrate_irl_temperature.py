#!/usr/bin/env python3
"""
Temperature Scalingでモデルの確率出力をキャリブレーション

確率が偏っている問題を解決するため、Validation setでTemperatureを最適化
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, precision_recall_curve

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train_enhanced_irl_importants import prepare_trajectories_importants_style

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def get_logits_and_labels(model_system, trajectories):
    """モデルからlogitsとラベルを取得"""
    logits = []
    labels = []
    
    for traj in trajectories:
        try:
            # IRLシステムから直接logitを取得する必要がある
            # 一旦確率を取得してlogitに逆変換
            result = model_system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            prob = result['continuation_probability']
            
            # sigmoid^-1 で logit に変換
            # prob = 1 / (1 + exp(-logit))
            # logit = log(prob / (1 - prob))
            prob = np.clip(prob, 1e-7, 1 - 1e-7)  # 数値安定性
            logit = np.log(prob / (1 - prob))
            logits.append(logit)
            
            # ラベル
            if len(traj['step_labels']) > 0:
                labels.append(traj['step_labels'][-1])
            else:
                labels.append(0)
        except Exception as e:
            print(f"警告: スキップ - {e}")
            continue
    
    return np.array(logits), np.array(labels)


def apply_temperature_scaling(logits, temperature):
    """Temperature scalingを適用"""
    scaled_logits = logits / temperature
    probs = 1 / (1 + np.exp(-scaled_logits))
    return probs


def negative_log_likelihood(temperature, logits, labels):
    """
    Temperature最適化の目的関数（負の対数尤度）
    
    Args:
        temperature: スカラー値（最適化対象）
        logits: ロジット値の配列
        labels: 真のラベル
    
    Returns:
        負の対数尤度
    """
    # Temperature scalingを適用
    probs = apply_temperature_scaling(logits, temperature)
    
    # Log loss（負の対数尤度）
    loss = log_loss(labels, probs)
    return loss


def optimize_temperature(logits, labels):
    """
    Validation setでTemperatureを最適化
    
    Args:
        logits: Validation setのlogits
        labels: Validation setのラベル
    
    Returns:
        最適なtemperature値
    """
    print("\nTemperature最適化中...")
    
    # 初期値: temperature = 1.0（変換なし）
    initial_temp = 1.0
    
    # 最適化（temperature > 0の制約）
    result = minimize(
        negative_log_likelihood,
        initial_temp,
        args=(logits, labels),
        method='L-BFGS-B',
        bounds=[(0.1, 10.0)],  # temperatureの範囲
        options={'disp': True}
    )
    
    optimal_temperature = result.x[0]
    
    print(f"\n最適Temperature: {optimal_temperature:.4f}")
    print(f"Log Loss (before): {negative_log_likelihood(1.0, logits, labels):.4f}")
    print(f"Log Loss (after): {negative_log_likelihood(optimal_temperature, logits, labels):.4f}")
    
    return optimal_temperature


def analyze_calibration(logits, labels, temperature):
    """キャリブレーション効果を分析"""
    # Before
    probs_before = apply_temperature_scaling(logits, 1.0)
    
    # After
    probs_after = apply_temperature_scaling(logits, temperature)
    
    print(f"\n{'='*70}")
    print("キャリブレーション前後の比較")
    print(f"{'='*70}")
    
    print("\n【確率分布】")
    print(f"  Before (T=1.0):")
    print(f"    平均: {probs_before.mean():.4f}")
    print(f"    中央値: {np.median(probs_before):.4f}")
    print(f"    標準偏差: {probs_before.std():.4f}")
    print(f"    最小値: {probs_before.min():.4f}")
    print(f"    最大値: {probs_before.max():.4f}")
    
    print(f"\n  After (T={temperature:.4f}):")
    print(f"    平均: {probs_after.mean():.4f}")
    print(f"    中央値: {np.median(probs_after):.4f}")
    print(f"    標準偏差: {probs_after.std():.4f}")
    print(f"    最小値: {probs_after.min():.4f}")
    print(f"    最大値: {probs_after.max():.4f}")
    
    # F1最適化の閾値
    print("\n【F1最適閾値】")
    
    # Before
    precision, recall, thresholds = precision_recall_curve(labels, probs_before)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_before = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"  Before: {optimal_threshold_before:.4f}")
    print(f"    F1: {f1_scores[optimal_idx]:.4f}")
    print(f"    Precision: {precision[optimal_idx]:.4f}")
    print(f"    Recall: {recall[optimal_idx]:.4f}")
    
    # After
    precision, recall, thresholds = precision_recall_curve(labels, probs_after)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_after = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n  After: {optimal_threshold_after:.4f}")
    print(f"    F1: {f1_scores[optimal_idx]:.4f}")
    print(f"    Precision: {precision[optimal_idx]:.4f}")
    print(f"    Recall: {recall[optimal_idx]:.4f}")
    
    # 閾値0.5での性能
    print("\n【閾値0.5での性能】")
    
    preds_before = (probs_before >= 0.5).astype(int)
    f1_before = f1_score(labels, preds_before)
    positive_count_before = preds_before.sum()
    
    preds_after = (probs_after >= 0.5).astype(int)
    f1_after = f1_score(labels, preds_after)
    positive_count_after = preds_after.sum()
    
    print(f"  Before: F1={f1_before:.4f}, Positive予測={positive_count_before}/{len(labels)}")
    print(f"  After:  F1={f1_after:.4f}, Positive予測={positive_count_after}/{len(labels)}")
    
    return {
        'temperature': temperature,
        'optimal_threshold_before': float(optimal_threshold_before),
        'optimal_threshold_after': float(optimal_threshold_after),
        'f1_at_0.5_before': float(f1_before),
        'f1_at_0.5_after': float(f1_after)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="レビューデータCSV")
    parser.add_argument("--model", required=True, help="訓練済みモデルパス")
    parser.add_argument("--train-start", required=True, help="訓練開始日")
    parser.add_argument("--train-end", required=True, help="訓練終了日")
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--project", default="openstack/nova")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation分割率")
    
    args = parser.parse_args()
    
    # データ読み込み
    print("データ読み込み中...")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    
    # 訓練データ準備
    print("訓練データ準備中...")
    trajectories = prepare_trajectories_importants_style(
        df, train_start, train_end,
        args.future_window_start, args.future_window_end,
        min_history_requests=3, project=args.project
    )
    
    # Train/Val分割
    n_samples = len(trajectories)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    # シャッフル（再現性のため固定seed）
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_trajectories = [trajectories[i] for i in indices[:n_train]]
    val_trajectories = [trajectories[i] for i in indices[n_train:]]
    
    print(f"\nデータ分割:")
    print(f"  訓練: {n_train}サンプル")
    print(f"  検証: {n_val}サンプル")
    
    # モデル読み込み
    print(f"\nモデル読み込み: {args.model}")
    system = RetentionIRLSystem.load_model(args.model)
    
    # Validation setでlogitsとラベルを取得
    print("\nValidation setで予測...")
    val_logits, val_labels = get_logits_and_labels(system, val_trajectories)
    
    print(f"Validation: {len(val_logits)}サンプル")
    print(f"Positive率: {val_labels.mean():.1%}")
    
    # Temperature最適化
    optimal_temperature = optimize_temperature(val_logits, val_labels)
    
    # 効果分析
    calibration_results = analyze_calibration(val_logits, val_labels, optimal_temperature)
    
    # 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'temperature': float(optimal_temperature),
        'n_train': n_train,
        'n_val': n_val,
        'calibration_metrics': calibration_results
    }
    
    output_file = output_dir / 'temperature_calibration.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 保存完了: {output_file}")
    print(f"\n{'='*70}")
    print(f"推奨設定: Temperature = {optimal_temperature:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
