"""
クロス評価結果を最適閾値で再計算

閾値0.5固定ではなく、F1スコアを最大化する閾値を各モデルで探索
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def find_optimal_threshold(y_true, y_pred_proba):
    """
    F1スコアを最大化する閾値を探索
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        try:
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': f1,
                    'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                }
        except:
            continue
    
    return best_threshold, best_metrics


def recompute_single_evaluation(model_path, trajectories, output_file):
    """
    単一の評価を再計算
    """
    # モデル読み込み
    config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 64,
        'lstm_hidden': 128,
        'sequence': True,
        'seq_len': 20,
        'device': 'cpu'
    }
    
    irl_system = RetentionIRLSystem(config)
    irl_system.load_model(model_path)
    
    # 予測
    y_true = []
    y_pred = []
    
    for traj in trajectories:
        prediction = irl_system.predict_continuation_probability(
            traj['developer'],
            traj['activity_history'],
            traj['context_date']
        )
        
        y_true.append(1 if traj['future_contribution'] else 0)
        y_pred.append(prediction['continuation_probability'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 基本メトリクス（閾値非依存）
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred),
        'test_samples': len(trajectories),
        'positive_samples': int(sum(y_true)),
        'positive_rate': float(np.mean(y_true)),
    }
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        metrics['auc_pr'] = auc(recall_curve, precision_curve)
    except:
        metrics['auc_pr'] = 0.0
    
    # 予測確率の統計
    metrics['pred_stats'] = {
        'min': float(np.min(y_pred)),
        'max': float(np.max(y_pred)),
        'mean': float(np.mean(y_pred)),
        'median': float(np.median(y_pred)),
        'std': float(np.std(y_pred)),
        'above_0.5': float(np.mean(y_pred >= 0.5)),
    }
    
    # 固定閾値0.5での評価
    y_pred_05 = (y_pred >= 0.5).astype(int)
    metrics['threshold_0.5'] = {
        'threshold': 0.5,
        'f1': f1_score(y_true, y_pred_05, zero_division=0),
        'precision': precision_score(y_true, y_pred_05, zero_division=0),
        'recall': recall_score(y_true, y_pred_05, zero_division=0),
    }
    
    # 最適閾値での評価
    optimal_threshold, optimal_metrics = find_optimal_threshold(y_true, y_pred)
    metrics['optimal'] = {
        'threshold': float(optimal_threshold),
        **optimal_metrics
    }
    
    # 正例率に基づく閾値での評価
    positive_rate_threshold = np.percentile(y_pred, (1 - metrics['positive_rate']) * 100)
    y_pred_pr = (y_pred >= positive_rate_threshold).astype(int)
    metrics['positive_rate_threshold'] = {
        'threshold': float(positive_rate_threshold),
        'f1': f1_score(y_true, y_pred_pr, zero_division=0),
        'precision': precision_score(y_true, y_pred_pr, zero_division=0),
        'recall': recall_score(y_true, y_pred_pr, zero_division=0),
    }
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross-eval-dir', type=Path, 
                       default=Path('outputs/cross_evaluation'))
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/cross_evaluation_recomputed'))
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 全組み合わせを処理
    results = []
    
    train_ranges = ['0_3m', '0_6m', '0_9m', '0_12m']
    eval_ranges = ['0_3m', '3_6m', '6_9m', '9_12m']
    
    for train_range in train_ranges:
        model_path = args.cross_eval_dir / f'model_{train_range}' / 'irl_model.pth'
        
        if not model_path.exists():
            print(f"⚠️  モデルが見つかりません: {model_path}")
            continue
        
        for eval_range in eval_ranges:
            eval_dir = args.cross_eval_dir / f'eval_train{train_range}_eval{eval_range}'
            traj_path = eval_dir / 'trajectories.pkl'
            
            if not traj_path.exists():
                print(f"⚠️  軌跡が見つかりません: {traj_path}")
                continue
            
            print(f"\n処理中: train={train_range}, eval={eval_range}")
            
            # 軌跡読み込み
            with open(traj_path, 'rb') as f:
                trajectories = pickle.load(f)
            
            if len(trajectories) == 0:
                print("  データなし")
                continue
            
            # 再計算
            output_file = args.output_dir / f'metrics_train{train_range}_eval{eval_range}.json'
            metrics = recompute_single_evaluation(model_path, trajectories, output_file)
            
            # 結果表示
            print(f"  サンプル数: {metrics['test_samples']}")
            print(f"  正例率: {metrics['positive_rate']:.1%}")
            print(f"  予測確率: {metrics['pred_stats']['mean']:.3f} ± {metrics['pred_stats']['std']:.3f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  AUC-PR: {metrics['auc_pr']:.3f}")
            print(f"  閾値0.5: F1={metrics['threshold_0.5']['f1']:.3f}, "
                  f"P={metrics['threshold_0.5']['precision']:.3f}, "
                  f"R={metrics['threshold_0.5']['recall']:.3f}")
            print(f"  最適閾値({metrics['optimal']['threshold']:.2f}): "
                  f"F1={metrics['optimal']['f1']:.3f}, "
                  f"P={metrics['optimal']['precision']:.3f}, "
                  f"R={metrics['optimal']['recall']:.3f}")
            
            # CSVデータ蓄積
            results.append({
                'train_range': train_range.replace('_', '-'),
                'eval_range': eval_range.replace('_', '-'),
                'auc_roc': metrics['auc_roc'],
                'auc_pr': metrics['auc_pr'],
                'pred_mean': metrics['pred_stats']['mean'],
                'pred_std': metrics['pred_stats']['std'],
                'optimal_threshold': metrics['optimal']['threshold'],
                'optimal_f1': metrics['optimal']['f1'],
                'optimal_precision': metrics['optimal']['precision'],
                'optimal_recall': metrics['optimal']['recall'],
                'threshold_0.5_f1': metrics['threshold_0.5']['f1'],
                'positive_rate': metrics['positive_rate'],
                'test_samples': metrics['test_samples'],
            })
    
    # CSV保存
    df = pd.DataFrame(results)
    csv_path = args.output_dir / 'recomputed_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 結果を保存: {csv_path}")
    
    # サマリー表示
    print("\n" + "=" * 80)
    print("サマリー: 最適閾値での性能")
    print("=" * 80)
    
    pivot_f1 = df.pivot(index='train_range', columns='eval_range', values='optimal_f1')
    print("\nF1スコア（最適閾値）:")
    print(pivot_f1.round(3))
    
    pivot_threshold = df.pivot(index='train_range', columns='eval_range', values='optimal_threshold')
    print("\n最適閾値:")
    print(pivot_threshold.round(3))


if __name__ == '__main__':
    main()

