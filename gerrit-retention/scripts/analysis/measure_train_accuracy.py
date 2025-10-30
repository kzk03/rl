"""訓練データに対する正解率を測定"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

sys.path.insert(0, 'scripts/training/irl')
sys.path.insert(0, 'src')

from train_irl_review_acceptance import extract_review_acceptance_trajectories

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def measure_train_accuracy():
    """訓練データに対する正解率を測定"""
    
    # データ読み込み
    df = pd.read_csv('data/review_requests_openstack_multi_5y_detail.csv')
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    cutoff = pd.Timestamp('2023-01-01')
    train_start = cutoff - pd.DateOffset(months=12)
    train_end = cutoff
    project = 'openstack/nova'
    
    periods = [
        ('0-3m', 0, 3),
        ('3-6m', 3, 6),
        ('6-9m', 6, 9),
        ('9-12m', 9, 12)
    ]
    
    print("## 📊 訓練データに対する正解率")
    print()
    print("| 訓練期間 | 訓練データ数 | 訓練正解率 | 訓練AUC-ROC | 訓練AUC-PR | 評価正解率 | 評価AUC-PR | 過学習度 |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for name, start_months, end_months in periods:
        # モデル読み込み
        model_path = Path(f'outputs/review_acceptance_cross_eval_nova/train_{name}/irl_model.pt')
        if not model_path.exists():
            print(f"| {name} | - | - | - | - | - | - | - |")
            continue
        
        # Config を作成してモデルを初期化（訓練時と同じ設定）
        config = {
            'state_dim': 9,
            'action_dim': 4,
            'hidden_dim': 128,  # 訓練時と同じ
            'sequence': True,
            'seq_len': 0,
            'use_temporal_features': False,
            'learning_rate': 0.001
        }
        irl = RetentionIRLSystem(config=config)
        irl.network.load_state_dict(torch.load(model_path))
        irl.network.eval()
        
        # 訓練データ抽出
        train_trajectories = extract_review_acceptance_trajectories(
            df=df,
            train_start=train_start,
            train_end=train_end,
            future_window_start_months=start_months,
            future_window_end_months=end_months,
            project=project
        )
        
        # 訓練データで予測
        train_predictions = []
        train_labels = []
        
        for traj in train_trajectories:
            result = irl.predict_continuation_probability_snapshot(
                developer=traj['developer_info'],
                activity_history=traj['activity_history'],
                context_date=train_end
            )
            # dict から確率を取得
            prob = result['continuation_probability']
            train_predictions.append(prob)
            train_labels.append(1.0 if traj['future_acceptance'] else 0.0)
        
        train_predictions = torch.tensor(train_predictions)
        train_labels = torch.tensor(train_labels)
        
        # 閾値読み込み
        metrics_file = Path(f'outputs/review_acceptance_cross_eval_nova/train_{name}/metrics.json')
        with open(metrics_file) as f:
            eval_metrics = json.load(f)
        
        threshold = eval_metrics['optimal_threshold']
        
        # 訓練データのメトリクス計算
        train_pred_binary = (train_predictions >= threshold).float()
        train_accuracy = accuracy_score(train_labels, train_pred_binary)
        train_auc_roc = roc_auc_score(train_labels, train_predictions)
        train_auc_pr = average_precision_score(train_labels, train_predictions)
        
        # 評価データのメトリクス
        eval_accuracy = eval_metrics['precision']  # 実際は Precision を使用
        eval_auc_pr = eval_metrics['auc_pr']
        
        # 過学習度（訓練と評価の差）
        overfit_degree = train_accuracy - eval_accuracy
        
        print(f"| {name} | {len(train_trajectories)}人 | {train_accuracy:.3f} | {train_auc_roc:.3f} | {train_auc_pr:.3f} | {eval_accuracy:.3f} | {eval_auc_pr:.3f} | {overfit_degree:+.3f} |")
    
    print()
    print("### **過学習度の判定基準**")
    print("- **< 0.05**: 適切な汎化")
    print("- **0.05 ～ 0.15**: 軽度の過学習")
    print("- **> 0.15**: 深刻な過学習")

if __name__ == '__main__':
    measure_train_accuracy()

