#!/usr/bin/env python3
"""
訓練データでの予測確率分布を確認するスクリプト
閾値決定の妥当性を検証
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train_enhanced_irl_importants import prepare_trajectories_importants_style

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def analyze_probability_distribution(
    model_path: Path,
    trajectories: list,
    output_dir: Path
):
    """
    訓練データでの予測確率分布を分析
    """
    print(f"モデル読み込み: {model_path}")
    system = RetentionIRLSystem.load_model(str(model_path))
    
    print(f"軌跡数: {len(trajectories)}")
    
    # 予測
    probs = []
    labels = []
    
    for traj in trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            probs.append(result['continuation_probability'])
            
            if len(traj['step_labels']) > 0:
                labels.append(traj['step_labels'][-1])
            else:
                labels.append(0)
        except Exception as e:
            print(f"警告: 予測スキップ - {e}")
            continue
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    print(f"\n{'='*70}")
    print("確率分布の統計")
    print(f"{'='*70}")
    print(f"サンプル数: {len(probs)}")
    print(f"平均: {probs.mean():.4f}")
    print(f"中央値: {np.median(probs):.4f}")
    print(f"標準偏差: {probs.std():.4f}")
    print(f"最小値: {probs.min():.4f}")
    print(f"最大値: {probs.max():.4f}")
    print(f"\nパーセンタイル:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}%: {np.percentile(probs, p):.4f}")
    
    # ラベル別の確率分布
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    
    print(f"\n{'='*70}")
    print("ラベル別の確率")
    print(f"{'='*70}")
    print(f"Positive (n={len(pos_probs)}):")
    print(f"  平均: {pos_probs.mean():.4f}")
    print(f"  中央値: {np.median(pos_probs):.4f}")
    print(f"  標準偏差: {pos_probs.std():.4f}")
    print(f"\nNegative (n={len(neg_probs)}):")
    print(f"  平均: {neg_probs.mean():.4f}")
    print(f"  中央値: {np.median(neg_probs):.4f}")
    print(f"  標準偏差: {neg_probs.std():.4f}")
    
    # F1最大化の閾値
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n{'='*70}")
    print("F1最大化の閾値")
    print(f"{'='*70}")
    print(f"最適閾値: {optimal_threshold:.4f}")
    print(f"F1スコア: {f1_scores[optimal_idx]:.4f}")
    print(f"Precision: {precision[optimal_idx]:.4f}")
    print(f"Recall: {recall[optimal_idx]:.4f}")
    
    # 閾値0.5での性能
    preds_05 = (probs >= 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_05 = f1_score(labels, preds_05)
    precision_05 = precision_score(labels, preds_05, zero_division=0)
    recall_05 = recall_score(labels, preds_05, zero_division=0)
    
    print(f"\n{'='*70}")
    print("閾値0.5での性能")
    print(f"{'='*70}")
    print(f"F1スコア: {f1_05:.4f}")
    print(f"Precision: {precision_05:.4f}")
    print(f"Recall: {recall_05:.4f}")
    print(f"予測Positive数: {preds_05.sum()} / {len(preds_05)}")
    
    # ヒストグラム作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 全体のヒストグラム
    ax = axes[0, 0]
    ax.hist(probs, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='閾値0.5')
    ax.axvline(optimal_threshold, color='green', linestyle='--', label=f'最適閾値{optimal_threshold:.3f}')
    ax.set_xlabel('予測確率')
    ax.set_ylabel('頻度')
    ax.set_title('全体の確率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ラベル別のヒストグラム
    ax = axes[0, 1]
    ax.hist(pos_probs, bins=30, alpha=0.7, label='Positive', edgecolor='black')
    ax.hist(neg_probs, bins=30, alpha=0.7, label='Negative', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='閾値0.5')
    ax.axvline(optimal_threshold, color='green', linestyle='--', label=f'最適閾値{optimal_threshold:.3f}')
    ax.set_xlabel('予測確率')
    ax.set_ylabel('頻度')
    ax.set_title('ラベル別の確率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 累積分布
    ax = axes[1, 0]
    sorted_probs = np.sort(probs)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax.plot(sorted_probs, cumulative, linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', label='閾値0.5')
    ax.axvline(optimal_threshold, color='green', linestyle='--', label=f'最適閾値{optimal_threshold:.3f}')
    ax.set_xlabel('予測確率')
    ax.set_ylabel('累積確率')
    ax.set_title('累積分布関数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall曲線
    ax = axes[1, 1]
    ax.plot(recall, precision, linewidth=2)
    ax.scatter([recall[optimal_idx]], [precision[optimal_idx]], 
               color='green', s=100, zorder=5, label=f'最適閾値{optimal_threshold:.3f}')
    # 閾値0.5の位置を探す
    idx_05 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
    if idx_05 < len(precision) and idx_05 < len(recall):
        ax.scatter([recall[idx_05]], [precision[idx_05]], 
                   color='red', s=100, zorder=5, label='閾値0.5')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall曲線')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'probability_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ プロット保存: {plot_path}")
    
    # CSVで詳細データも保存
    df = pd.DataFrame({
        'probability': probs,
        'label': labels,
        'prediction_threshold_optimal': (probs >= optimal_threshold).astype(int),
        'prediction_threshold_0.5': (probs >= 0.5).astype(int)
    })
    csv_path = output_dir / 'probability_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ データ保存: {csv_path}")


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
    
    args = parser.parse_args()
    
    # データ読み込み
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
    
    # 分析実行
    analyze_probability_distribution(
        Path(args.model),
        trajectories,
        Path(args.output)
    )
    
    print(f"\n{'='*70}")
    print("完了！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
