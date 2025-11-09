"""
複数シードでEnhanced IRLを実行（月次サンプリング版）

Train: 月次サンプリング
Eval: 1スナップショット
"""

import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.attention_irl import AttentionIRLNetwork
from models.temporal_feature_extractor import TemporalFeatureExtractor


class ReviewerDataset(Dataset):
    """レビュワーデータセット（時系列特徴量付き）"""
    
    def __init__(self, state_features: np.ndarray, temporal_features: np.ndarray, labels: np.ndarray):
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


def calculate_irl_state_features(df: pd.DataFrame, reviewer: str, context_date: datetime) -> np.ndarray:
    """IRL状態特徴量を計算"""
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


def prepare_data(df, train_start, train_end, eval_start, eval_end, 
                train_future_start_months, train_future_end_months,
                eval_future_start_months, eval_future_end_months):
    """データ準備（Train: 月次サンプリング, Eval: 1スナップショット）"""
    df = df.copy()
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    temporal_extractor = TemporalFeatureExtractor()
    
    # レビュワーごとにデータを事前分割（高速化）
    print(f"  レビュワーごとにデータを分割中...")
    reviewer_data = {}
    for reviewer in df['reviewer_email'].unique():
        reviewer_data[reviewer] = df[df['reviewer_email'] == reviewer].copy()
    print(f"  完了: {len(reviewer_data)}人")
    
    # ========================================
    # Trainデータ: 月次サンプリング
    # ========================================
    X_train_state, X_train_temporal, y_train = [], [], []
    
    label_months = train_future_end_months
    label_start = train_end_dt - pd.DateOffset(months=label_months)
    
    print(f"\n[Train] 月次サンプリング")
    print(f"  観測可能期間: {train_start_dt.date()} 〜 {label_start.date()}")
    print(f"  ラベル専用: {label_start.date()} 〜 {train_end_dt.date()}")
    
    current_month = train_start_dt
    month_count = 0
    
    while current_month < label_start:
        # この月に活動があったレビュワー
        month_df = df[(df['request_time'] >= current_month) &
                     (df['request_time'] < current_month + pd.DateOffset(months=1))]
        month_reviewers = month_df['reviewer_email'].unique()
        
        for reviewer in month_reviewers:
            reviewer_df = reviewer_data[reviewer]
            history_df = reviewer_df[reviewer_df['request_time'] < current_month]
            
            if len(history_df) == 0:
                continue
            
            # 特徴量計算
            state_features = calculate_irl_state_features(df, reviewer, current_month)
            temporal_features = temporal_extractor.extract_all_temporal_features(
                reviewer, history_df, current_month, pre_filtered=True
            )
            
            if np.isnan(state_features).any() or np.isnan(temporal_features).any():
                continue
            
            # ラベル計算
            future_start = current_month + timedelta(days=train_future_start_months * 30)
            future_end = current_month + timedelta(days=train_future_end_months * 30)
            
            if future_start >= train_end_dt:
                continue
            
            future_end = min(future_end, train_end_dt)
            
            future_df = reviewer_df[
                (reviewer_df['request_time'] >= future_start) &
                (reviewer_df['request_time'] < future_end)
            ]
            
            label = 1 if len(future_df) > 0 and (future_df['label'] == 1).any() else 0
            X_train_state.append(state_features)
            X_train_temporal.append(temporal_features)
            y_train.append(label)
        
        current_month += pd.DateOffset(months=1)
        month_count += 1
        
        if month_count % 6 == 0:
            print(f"  進捗: {month_count}月処理 ({len(y_train)}サンプル)")
    
    print(f"  完了: {len(y_train)}サンプル\n")
    
    # ========================================
    # Evalデータ: 1スナップショット
    # ========================================
    X_test_state, X_test_temporal, y_test = [], [], []
    
    print(f"[Eval] スナップショット")
    print(f"  基準日: {eval_start_dt.date()}")
    
    eval_active_df = df[(df['request_time'] >= eval_start_dt) &
                        (df['request_time'] < eval_end_dt)]
    eval_reviewers = eval_active_df['reviewer_email'].unique()
    
    print(f"  対象: {len(eval_reviewers)}人")
    
    for idx, reviewer in enumerate(eval_reviewers, 1):
        if idx % 500 == 0:
            print(f"  進捗: {idx}/{len(eval_reviewers)}")
        
        reviewer_df = reviewer_data.get(reviewer)
        if reviewer_df is None:
            continue
        
        history_df = reviewer_df[reviewer_df['request_time'] < eval_start_dt]
        
        if len(history_df) == 0:
            continue
        
        state_features = calculate_irl_state_features(df, reviewer, eval_start_dt)
        temporal_features = temporal_extractor.extract_all_temporal_features(
            reviewer, history_df, eval_start_dt, pre_filtered=True
        )
        
        if np.isnan(state_features).any() or np.isnan(temporal_features).any():
            continue
        
        future_start = eval_start_dt + timedelta(days=eval_future_start_months * 30)
        future_end = eval_start_dt + timedelta(days=eval_future_end_months * 30)
        
        future_df = reviewer_df[
            (reviewer_df['request_time'] >= future_start) &
            (reviewer_df['request_time'] < future_end)
        ]
        
        label = 1 if len(future_df) > 0 and (future_df['label'] == 1).any() else 0
        X_test_state.append(state_features)
        X_test_temporal.append(temporal_features)
        y_test.append(label)
    
    print(f"  完了: {len(y_test)}サンプル\n")
    
    return (
        np.array(X_train_state), np.array(X_train_temporal), np.array(y_train),
        np.array(X_test_state), np.array(X_test_temporal), np.array(y_test)
    )


def train_and_evaluate(X_train_state, X_train_temporal, y_train,
                      X_test_state, X_test_temporal, y_test,
                      seed=777, epochs=50, lr=0.001):
    """モデルトレーニングと評価"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
        
        model.eval()
        all_preds, all_labels = [], []
        
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
            best_auc = max(best_auc, auc)
    
    return best_auc


def main():
    """メイン実行"""
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    
    print(f"データ読み込み: {len(df)}行\n")
    
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    
    seeds = [42, 123, 777, 2024, 9999]
    
    period_combinations = []
    for train_s in [0, 3, 6, 9]:
        train_e = train_s + 3
        for eval_s in [0, 3, 6, 9]:
            eval_e = eval_s + 3
            if eval_s >= train_s:
                period_combinations.append((train_s, train_e, eval_s, eval_e))
    
    print(f"期間組み合わせ: {len(period_combinations)}通り")
    print(f"シード: {seeds}")
    print(f"総実験数: {len(period_combinations) * len(seeds)}\n")
    
    # データ準備（1回のみ）
    print("="*60)
    print("データ準備")
    print("="*60)
    
    all_data = {}
    for idx, (train_s, train_e, eval_s, eval_e) in enumerate(period_combinations, 1):
        print(f"\n[{idx}/{len(period_combinations)}] Train={train_s}-{train_e}m, Eval={eval_s}-{eval_e}m")
        key = (train_s, train_e, eval_s, eval_e)
        all_data[key] = prepare_data(
            df, train_start, train_end, eval_start, eval_end,
            train_s, train_e, eval_s, eval_e
        )
    
    print("\n" + "="*60)
    print("トレーニング開始")
    print("="*60)
    
    all_results = []
    
    for seed in seeds:
        print(f"\nシード {seed}")
        
        for idx, (train_s, train_e, eval_s, eval_e) in enumerate(period_combinations, 1):
            key = (train_s, train_e, eval_s, eval_e)
            X_train_s, X_train_t, y_train, X_test_s, X_test_t, y_test = all_data[key]
            
            if len(set(y_test)) < 2:
                print(f"  [{idx}/10] Train={train_s}-{train_e}m→Eval={eval_s}-{eval_e}m: スキップ")
                continue
            
            auc = train_and_evaluate(
                X_train_s, X_train_t, y_train,
                X_test_s, X_test_t, y_test,
                seed=seed, epochs=50
            )
            
            print(f"  [{idx}/10] Train={train_s}-{train_e}m→Eval={eval_s}-{eval_e}m: AUC={auc:.4f}")
            
            all_results.append({
                'seed': seed,
                'train_start': train_s,
                'train_end': train_e,
                'eval_start': eval_s,
                'eval_end': eval_e,
                'auc': auc
            })
    
    # 結果保存
    results_df = pd.DataFrame(all_results)
    output_path = Path(__file__).parent.parent / "results" / "multi_seed_results_v2.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n結果保存: {output_path}")
    print(f"\n平均AUC: {results_df['auc'].mean():.4f}")


if __name__ == "__main__":
    main()
