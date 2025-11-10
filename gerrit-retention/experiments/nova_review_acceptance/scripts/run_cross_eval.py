"""
Nova レビュー受諾予測 - 4×4クロス評価（importants準拠）

訓練期間: 0-3m, 3-6m, 6-9m, 9-12m
評価期間: 0-3m, 3-6m, 6-9m, 9-12m
総評価数: 4×4 = 16通り
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# Enhanced IRLモデルをインポート
enhanced_irl_path = Path(__file__).parent.parent.parent / "enhanced_irl"
sys.path.insert(0, str(enhanced_irl_path))
from models.attention_irl import AttentionIRLNetwork


class ReviewerDataset(Dataset):
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


def calculate_state_features(df: pd.DataFrame, reviewer: str, context_date: datetime) -> np.ndarray:
    """IRL状態特徴量（10次元）"""
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
    project_count = 1.0
    
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
            else:
                activity_trend = 0.0
        else:
            activity_trend = 0.5
    else:
        activity_trend = 0.5
    
    last_activity = reviewer_df['timestamp'].max()
    days_since_last = (context_date - last_activity).days / 365.0
    
    acceptance_rate = reviewer_df['label'].mean() if 'label' in reviewer_df.columns else 0.0
    
    if len(recent_df) > 0:
        recent_acceptance = recent_df['label'].mean() if 'label' in recent_df.columns else 0.0
    else:
        recent_acceptance = 0.0
    
    return np.array([
        experience_days, total_changes, total_reviews, project_count,
        recent_activity_frequency, avg_activity_gap, activity_trend,
        days_since_last, acceptance_rate, recent_acceptance
    ])


def prepare_period_data(df: pd.DataFrame, train_start: datetime, train_end: datetime,
                        eval_start: datetime, eval_months: int, 
                        future_window_start_months: int, future_window_months: int):
    """
    特定期間の訓練・評価データ準備（importants方式完全準拠）
    
    訓練: 各月末から future_window_start_months ~ future_window_start_months+future_window_months を見る
    評価: 評価期間末から future_window_start_months ~ future_window_start_months+future_window_months を見る
    
    Args:
        train_start: 訓練期間開始
        train_end: 訓練期間終了（全訓練期間で共通: 2023-01-01）
        eval_start: 評価期間開始
        eval_months: 評価期間の長さ（月）
        future_window_start_months: 将来窓の開始オフセット（月）
        future_window_months: 将来窓の幅（月）
    """
    
    # 訓練: 月次集約
    train_trajectories = []
    current = train_start
    
    # 各月末を基準点として軌跡を生成
    while current < train_end:
        month_end = current + pd.DateOffset(months=1)
        if month_end > train_end:
            break
        
        # この月末までの累積活動履歴
        history_df = df[(df['request_time'] >= train_start) & (df['request_time'] < month_end)]
        active_reviewers = history_df['reviewer_email'].unique()
        
        # 将来窓: 月末から future_window_start_months ~ future_window_start_months+future_window_months
        future_start = month_end + pd.DateOffset(months=future_window_start_months)
        future_end = month_end + pd.DateOffset(months=future_window_start_months + future_window_months)
        
        # train_endでクリップ（データリーク防止）
        if future_end > train_end:
            future_end = train_end
        if future_start >= train_end:
            # 将来窓が完全にtrain_endを超える場合はスキップ
            current = month_end
            continue
        
        future_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
        accepted = set(future_df[future_df['label'] == 1]['reviewer_email'].unique())
        
        for reviewer in active_reviewers:
            train_trajectories.append({
                'reviewer': reviewer,
                'cutoff_date': month_end,
                'label': 1 if reviewer in accepted else 0
            })
        
        current = month_end
    
    train_df = pd.DataFrame(train_trajectories)
    
    # 評価
    eval_end = eval_start + pd.DateOffset(months=eval_months)
    eval_period_df = df[(df['request_time'] >= eval_start) & (df['request_time'] < eval_end)]
    eval_reviewers = eval_period_df['reviewer_email'].unique()
    
    # 評価の将来窓: 評価終了から future_window_start_months ~ future_window_start_months+future_window_months
    future_start = eval_end + pd.DateOffset(months=future_window_start_months)
    future_end = eval_end + pd.DateOffset(months=future_window_start_months + future_window_months)
    future_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
    accepted = set(future_df[future_df['label'] == 1]['reviewer_email'].unique())
    
    eval_samples = []
    for reviewer in eval_reviewers:
        eval_samples.append({
            'reviewer': reviewer,
            'cutoff_date': eval_start,
            'label': 1 if reviewer in accepted else 0
        })
    
    eval_df = pd.DataFrame(eval_samples)
    
    return train_df, eval_df


def train_enhanced_irl(X_train_state, y_train, X_eval_state, y_eval, epochs=50):
    """Enhanced IRL学習"""
    
    X_train_temporal = X_train_state[:, np.newaxis, :]
    X_eval_temporal = X_eval_state[:, np.newaxis, :]
    
    train_dataset = ReviewerDataset(X_train_state, X_train_temporal, y_train)
    eval_dataset = ReviewerDataset(X_eval_state, X_eval_temporal, y_eval)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionIRLNetwork(
        state_dim=10, temporal_dim=10, hidden_dim=128,
        num_layers=2, dropout=0.3, use_temporal=True
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            state = batch['state'].to(device).unsqueeze(1)
            temporal = batch['temporal'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(state, temporal)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 評価
        model.eval()
        eval_probs = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                state = batch['state'].to(device).unsqueeze(1)
                temporal = batch['temporal'].to(device)
                labels = batch['label']
                
                outputs, _ = model(state, temporal)
                outputs = outputs.squeeze()
                eval_probs.extend(outputs.cpu().numpy())
                eval_labels.extend(labels.numpy())
        
        if len(set(eval_labels)) > 1:
            eval_auc = roc_auc_score(eval_labels, eval_probs)
            if eval_auc > best_auc:
                best_auc = eval_auc
    
    return best_auc


def train_random_forest(X_train, y_train, X_eval, y_eval):
    """ランダムフォレスト学習"""
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    eval_probs = rf.predict_proba(X_eval)[:, 1]
    
    if len(set(y_eval)) > 1:
        return roc_auc_score(y_eval, eval_probs)
    else:
        return 0.0


def main():
    print("=" * 80)
    print("Nova レビュー受諾予測 - 4×4クロス評価")
    print("=" * 80)
    
    # データ読み込み
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    df = df[df['project'] == 'openstack/nova'].copy()
    
    print(f"\n✅ Nova: {len(df)} レコード, {df['reviewer_email'].nunique()} レビュワー")
    
    # 実験設定（importants準拠）
    # 全訓練期間: 2021-01-01 ~ 2023-01-01（24ヶ月固定）
    # Future Windowは各月末からの相対オフセット（月）
    # クリップでtrain_endを超えないように制御
    
    base_train_start = datetime(2021, 1, 1)
    train_end = datetime(2023, 1, 1)  # 全訓練期間共通
    base_eval_start = datetime(2023, 1, 1)
    
    # 各訓練期間の設定（importantsと同一）
    # fw_start, fw_end: 各月末からの相対オフセット（月）
    train_configs = [
        {'name': '0-3m', 'fw_start': 0, 'fw_end': 3},    # 各月末から0~3ヶ月後
        {'name': '3-6m', 'fw_start': 3, 'fw_end': 6},    # 各月末から3~6ヶ月後
        {'name': '6-9m', 'fw_start': 6, 'fw_end': 9},    # 各月末から6~9ヶ月後
        {'name': '9-12m', 'fw_start': 9, 'fw_end': 12},  # 各月末から9~12ヶ月後
    ]
    
    eval_configs = [
        {'name': '0-3m', 'offset': 0, 'months': 3, 'fw_start': 0, 'fw_end': 3},
        {'name': '3-6m', 'offset': 3, 'months': 3, 'fw_start': 3, 'fw_end': 6},
        {'name': '6-9m', 'offset': 6, 'months': 3, 'fw_start': 6, 'fw_end': 9},
        {'name': '9-12m', 'offset': 9, 'months': 3, 'fw_start': 9, 'fw_end': 12},
    ]
    
    results = {
        'enhanced_irl': {},
        'random_forest': {}
    }
    
    total = len(train_configs) * len(eval_configs)
    count = 0
    
    for train_cfg in train_configs:
        for eval_cfg in eval_configs:
            count += 1
            key = f"{train_cfg['name']}_train_{eval_cfg['name']}_eval"
            
            print(f"\n{'='*80}")
            print(f"[{count}/{total}] {key}")
            print(f"{'='*80}")
            
            # 訓練期間は全て同じ（2021-01-01 ~ 2023-01-01）
            train_start = base_train_start
            eval_start = base_eval_start + pd.DateOffset(months=eval_cfg['offset'])
            
            # データ準備（importants準拠）
            train_df, eval_df = prepare_period_data(
                df, train_start, train_end,
                eval_start, eval_cfg['months'],
                future_window_start_months=train_cfg['fw_start'],
                future_window_months=train_cfg['fw_end'] - train_cfg['fw_start']
            )
            
            print(f"Train: {len(train_df)} 軌跡, 継続率={train_df['label'].mean():.3f}")
            print(f"Eval: {len(eval_df)} サンプル, 継続率={eval_df['label'].mean():.3f}")
            
            if len(train_df) < 10 or len(eval_df) < 5:
                print("⚠️  データ不足のためスキップ")
                results['enhanced_irl'][key] = {'auc': None, 'reason': 'insufficient_data'}
                results['random_forest'][key] = {'auc': None, 'reason': 'insufficient_data'}
                continue
            
            # 特徴量計算
            X_train = np.array([calculate_state_features(df, row['reviewer'], row['cutoff_date'])
                                for _, row in train_df.iterrows()])
            y_train = train_df['label'].values
            
            X_eval = np.array([calculate_state_features(df, row['reviewer'], row['cutoff_date'])
                               for _, row in eval_df.iterrows()])
            y_eval = eval_df['label'].values
            
            # Enhanced IRL
            print("Enhanced IRL学習中...")
            irl_auc = train_enhanced_irl(X_train, y_train, X_eval, y_eval, epochs=50)
            print(f"✅ Enhanced IRL AUC: {irl_auc:.4f}")
            
            results['enhanced_irl'][key] = {
                'auc': float(irl_auc),
                'train_samples': len(train_df),
                'eval_samples': len(eval_df),
                'train_continuation': float(train_df['label'].mean()),
                'eval_continuation': float(eval_df['label'].mean())
            }
            
            # Random Forest
            print("Random Forest学習中...")
            rf_auc = train_random_forest(X_train, y_train, X_eval, y_eval)
            print(f"✅ Random Forest AUC: {rf_auc:.4f}")
            
            results['random_forest'][key] = {
                'auc': float(rf_auc),
                'train_samples': len(train_df),
                'eval_samples': len(eval_df),
                'train_continuation': float(train_df['label'].mean()),
                'eval_continuation': float(eval_df['label'].mean())
            }
    
    # 結果保存
    result_dir = Path(__file__).parent.parent / "results"
    result_dir.mkdir(exist_ok=True)
    
    with open(result_dir / "cross_eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # サマリー
    print("\n" + "=" * 80)
    print("実験サマリー")
    print("=" * 80)
    
    irl_aucs = [v['auc'] for v in results['enhanced_irl'].values() if v['auc'] is not None]
    rf_aucs = [v['auc'] for v in results['random_forest'].values() if v['auc'] is not None]
    
    print(f"\nEnhanced IRL:")
    print(f"  平均AUC: {np.mean(irl_aucs):.4f}")
    print(f"  最良AUC: {np.max(irl_aucs):.4f}")
    print(f"  最悪AUC: {np.min(irl_aucs):.4f}")
    
    print(f"\nRandom Forest:")
    print(f"  平均AUC: {np.mean(rf_aucs):.4f}")
    print(f"  最良AUC: {np.max(rf_aucs):.4f}")
    print(f"  最悪AUC: {np.min(rf_aucs):.4f}")
    
    print(f"\n比較（importants）:")
    print(f"  AttentionなしIRL: 0.801")
    print(f"  Enhanced IRL: {np.mean(irl_aucs):.4f}")
    print(f"  Random Forest: {np.mean(rf_aucs):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
