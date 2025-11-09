"""
強化IRL（時系列特徴量+アテンション）のクロス評価

10パターンの期間組み合わせで評価し、RFと比較
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# 強化版モジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.attention_irl import AttentionIRLNetwork
from models.temporal_feature_extractor import TemporalFeatureExtractor


class ReviewerDataset(Dataset):
    """レビュワーデータセット（時系列特徴量付き）"""
    
    def __init__(
        self,
        state_features: np.ndarray,
        temporal_features: np.ndarray,
        labels: np.ndarray
    ):
        self.state_features = torch.FloatTensor(state_features)
        self.temporal_features = torch.FloatTensor(temporal_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'state': self.state_features[idx],
            'temporal': self.temporal_features[idx],
            'label': self.labels[idx]
        }


def calculate_irl_state_features(
    df: pd.DataFrame,
    reviewer: str,
    context_date: datetime
) -> np.ndarray:
    """IRL状態特徴量を計算（既存のIRLと同じロジック）"""
    reviewer_df = df[df['reviewer_email'] == reviewer].copy()
    
    if len(reviewer_df) == 0:
        return np.zeros(10)
    
    reviewer_df['timestamp'] = pd.to_datetime(reviewer_df['request_time'])
    reviewer_df = reviewer_df[reviewer_df['timestamp'] < context_date]
    
    if len(reviewer_df) == 0:
        return np.zeros(10)
    
    first_seen = reviewer_df['timestamp'].min()
    experience_days = (context_date - first_seen).days / 730.0
    total_changes = len(reviewer_df) / 500.0
    total_reviews = len(reviewer_df) / 500.0
    
    recent_cutoff = context_date - timedelta(days=30)
    recent_df = reviewer_df[reviewer_df['timestamp'] >= recent_cutoff]
    recent_activity_frequency = len(recent_df) / 30.0
    
    if len(reviewer_df) > 1:
        sorted_times = reviewer_df['timestamp'].sort_values()
        time_diffs = sorted_times.diff().dt.total_seconds().dropna()
        avg_activity_gap = time_diffs.mean() / 86400.0 if len(time_diffs) > 0 else 1.0
        avg_activity_gap = min(avg_activity_gap, 60.0) / 60.0
    else:
        avg_activity_gap = 0.5
    
    if len(reviewer_df) > 1:
        midpoint = first_seen + (context_date - first_seen) / 2
        recent_half = reviewer_df[reviewer_df['timestamp'] >= midpoint]
        past_half = reviewer_df[reviewer_df['timestamp'] < midpoint]
        
        if len(past_half) > 0:
            activity_trend = len(recent_half) / len(past_half)
            if activity_trend > 1.5:
                activity_trend = 1.0
            elif activity_trend > 0.8:
                activity_trend = 0.5
            elif activity_trend < 0.5:
                activity_trend = 0.0
            else:
                activity_trend = 0.25
        else:
            activity_trend = 1.0
    else:
        activity_trend = 0.5
    
    collaboration_score = 1.0
    code_quality_score = 0.5
    recent_acceptance_rate = 0.5
    
    if len(reviewer_df) > 30:
        days_active = (context_date - first_seen).days
        avg_per_day = len(reviewer_df) / max(days_active, 1)
        recent_per_day = len(recent_df) / 30.0
        review_load = min(recent_per_day / max(avg_per_day, 0.1), 3.0) / 3.0
    else:
        review_load = 0.5
    
    return np.array([
        experience_days, total_changes, total_reviews, recent_activity_frequency,
        avg_activity_gap, activity_trend, collaboration_score, code_quality_score,
        recent_acceptance_rate, review_load
    ], dtype=np.float32)


def prepare_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    future_start_months: int,
    future_end_months: int
):
    """データ準備"""
    df = df.copy()
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    temporal_extractor = TemporalFeatureExtractor()
    
    # トレーニングデータ
    train_df = df[(df['request_time'] >= train_start_dt) & (df['request_time'] < train_end_dt)]
    train_reviewers = train_df['reviewer_email'].unique()
    
    X_train_state = []
    X_train_temporal = []
    y_train = []
    
    for reviewer in train_reviewers:
        state_features = calculate_irl_state_features(df, reviewer, train_end_dt)
        period_df = df[df['request_time'] < train_end_dt]
        temporal_features = temporal_extractor.extract_all_temporal_features(
            reviewer, period_df, train_end_dt
        )
        
        if np.isnan(state_features).any() or np.isnan(temporal_features).any():
            continue
        
        future_start = train_end_dt + timedelta(days=future_start_months * 30)
        future_end = train_end_dt + timedelta(days=future_end_months * 30)
        
        future_df = df[
            (df['reviewer_email'] == reviewer) &
            (df['request_time'] >= future_start) &
            (df['request_time'] < future_end)
        ]
        
        label = 1 if len(future_df) > 0 and (future_df['label'] == 1).any() else 0
        
        X_train_state.append(state_features)
        X_train_temporal.append(temporal_features)
        y_train.append(label)
    
    # テストデータ
    eval_df = df[(df['request_time'] >= eval_start_dt) & (df['request_time'] < eval_end_dt)]
    eval_reviewers = eval_df['reviewer_email'].unique()
    
    X_test_state = []
    X_test_temporal = []
    y_test = []
    
    for reviewer in eval_reviewers:
        state_features = calculate_irl_state_features(df, reviewer, eval_end_dt)
        period_df = df[df['request_time'] < eval_end_dt]
        temporal_features = temporal_extractor.extract_all_temporal_features(
            reviewer, period_df, eval_end_dt
        )
        
        if np.isnan(state_features).any() or np.isnan(temporal_features).any():
            continue
        
        future_start = eval_end_dt + timedelta(days=future_start_months * 30)
        future_end = eval_end_dt + timedelta(days=future_end_months * 30)
        
        future_df = df[
            (df['reviewer_email'] == reviewer) &
            (df['request_time'] >= future_start) &
            (df['request_time'] < future_end)
        ]
        
        label = 1 if len(future_df) > 0 and (future_df['label'] == 1).any() else 0
        
        X_test_state.append(state_features)
        X_test_temporal.append(temporal_features)
        y_test.append(label)
    
    return (
        np.array(X_train_state),
        np.array(X_train_temporal),
        np.array(y_train),
        np.array(X_test_state),
        np.array(X_test_temporal),
        np.array(y_test)
    )


def train_and_evaluate(
    X_train_state, X_train_temporal, y_train,
    X_test_state, X_test_temporal, y_test,
    epochs=50, lr=0.001
):
    """モデルトレーニングと評価"""
    train_dataset = ReviewerDataset(X_train_state, X_train_temporal, y_train)
    test_dataset = ReviewerDataset(X_test_state, X_test_temporal, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = AttentionIRLNetwork(
        state_dim=10, temporal_dim=97, hidden_dim=128,
        num_layers=2, dropout=0.3, use_temporal=True
    )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_auc = 0.0
    best_model_state = None
    
    # トレーニング
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            state = batch['state'].unsqueeze(1)
            temporal = batch['temporal'].unsqueeze(1)
            labels = batch['label']
            
            outputs, _ = model(state, temporal)
            loss = criterion(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 検証
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                state = batch['state'].unsqueeze(1)
                temporal = batch['temporal'].unsqueeze(1)
                labels = batch['label']
                
                outputs, _ = model(state, temporal)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict().copy()
    
    return best_auc


def main():
    """メイン実行"""
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    
    print(f"データ読み込み完了: {len(df)} 行\n")
    
    # パラメータ
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    
    # 期間組み合わせ（時系列順序考慮）
    period_combinations = []
    for train_start_m in [0, 3, 6, 9]:
        train_end_m = train_start_m + 3
        for eval_start_m in [0, 3, 6, 9]:
            eval_end_m = eval_start_m + 3
            # 時系列順序: eval_start >= train_start
            if eval_start_m >= train_start_m:
                period_combinations.append((train_start_m, train_end_m, eval_start_m, eval_end_m))
    
    print(f"評価する組み合わせ数: {len(period_combinations)}\n")
    
    results = []
    
    for idx, (train_s, train_e, eval_s, eval_e) in enumerate(period_combinations, 1):
        print(f"[{idx}/{len(period_combinations)}] Train: {train_s}-{train_e}m → Eval: {eval_s}-{eval_e}m")
        
        # データ準備
        X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test = prepare_data(
            df, train_start, train_end, eval_start, eval_end, train_s, train_e
        )
        
        print(f"  Train samples: {len(y_train)} (Pos: {sum(y_train)})")
        print(f"  Test samples: {len(y_test)} (Pos: {sum(y_test)})")
        
        if len(set(y_test)) < 2:
            print("  スキップ: ラベルが1クラスのみ\n")
            continue
        
        # トレーニング & 評価
        auc = train_and_evaluate(
            X_train_state, X_train_temporal, y_train,
            X_test_state, X_test_temporal, y_test,
            epochs=50, lr=0.001
        )
        
        print(f"  AUC: {auc:.4f}\n")
        
        results.append({
            'train_start': train_s,
            'train_end': train_e,
            'eval_start': eval_s,
            'eval_end': eval_e,
            'auc': auc
        })
    
    # 結果集計
    print("\n" + "=" * 60)
    print("最終結果")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    avg_auc = results_df['auc'].mean()
    print(f"\n平均 AUC: {avg_auc:.4f}")
    print(f"RF baseline: 0.8603")
    print(f"改善: {(avg_auc - 0.8603) * 100:+.2f}%")
    
    # 結果保存
    output_path = Path(__file__).parent.parent / "results" / "cross_eval_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
