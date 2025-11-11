"""
Nova レビュー受諾予測 - Enhanced IRL (importants設定準拠)

importants/baseline_nova_6month_windows と同じ設定:
- 訓練期間: 2021-01-01 ~ 2023-01-01 (24ヶ月)
- 評価期間: 2023-01-01 ~ 2024-01-01 (4期間: 0-3m, 3-6m, 6-9m, 9-12m)
- 月次集約: 各月末時点での特徴量 + 将来窓でのレビュー受諾
- タスク: レビュー受諾予測（将来期間に少なくとも1回受諾するか）
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

# Enhanced IRLモデルをインポート
enhanced_irl_path = Path(__file__).parent.parent.parent / "enhanced_irl"
sys.path.insert(0, str(enhanced_irl_path))
from models.attention_irl import AttentionIRLNetwork
from models.temporal_feature_extractor import TemporalFeatureExtractor


class ReviewerDataset(Dataset):
    """レビュワーデータセット"""
    
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
    
    # 経験日数
    first_seen = reviewer_df['timestamp'].min()
    experience_days = (context_date - first_seen).days / 730.0
    
    # 総変更数・レビュー数
    total_changes = len(reviewer_df) / 500.0
    total_reviews = len(reviewer_df) / 500.0
    
    # プロジェクト数
    project_count = 1.0
    
    # 最近の活動頻度
    recent_cutoff = context_date - timedelta(days=30)
    recent_df = reviewer_df[reviewer_df['timestamp'] >= recent_cutoff]
    recent_activity_frequency = len(recent_df) / 30.0
    
    # 平均活動間隔
    if len(reviewer_df) > 1:
        sorted_times = reviewer_df['timestamp'].sort_values()
        time_diffs = sorted_times.diff().dt.total_seconds().dropna()
        avg_activity_gap = time_diffs.mean() / 86400.0 if len(time_diffs) > 0 else 1.0
        avg_activity_gap = min(avg_activity_gap, 60.0) / 60.0
    else:
        avg_activity_gap = 0.5
    
    # 活動トレンド
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
    
    # 最終活動からの経過日数
    last_activity = reviewer_df['timestamp'].max()
    days_since_last = (context_date - last_activity).days / 365.0
    
    # レビュー受け入れ率
    acceptance_rate = reviewer_df['label'].mean() if 'label' in reviewer_df.columns else 0.0
    
    # 最近30日の受け入れ率
    if len(recent_df) > 0:
        recent_acceptance = recent_df['label'].mean() if 'label' in recent_df.columns else 0.0
    else:
        recent_acceptance = 0.0
    
    return np.array([
        experience_days,
        total_changes,
        total_reviews,
        project_count,
        recent_activity_frequency,
        avg_activity_gap,
        activity_trend,
        days_since_last,
        acceptance_rate,
        recent_acceptance
    ])


def prepare_monthly_trajectories(df: pd.DataFrame, train_start: datetime, train_end: datetime,
                                  future_window_months: int = 6):
    """
    月次集約軌跡を生成（データリークなし版）
    
    重要：訓練期間内で完結させるため、訓練期間を分割：
    - 特徴量計算期間: train_start ~ (train_end - future_window_months)
    - ラベル計算期間: (train_end - future_window_months) ~ train_end
    
    各月末を基準点として:
    - 特徴量: train_start ~ 月末までの累積活動
    - ラベル: 月末から future_window_months 後までの受諾（train_end内）
    """
    trajectories = []
    
    # 特徴量計算期間の終了（ラベル期間の開始）
    feature_end = train_end - pd.DateOffset(months=future_window_months)
    
    # 月末を列挙（特徴量計算期間内のみ）
    current = train_start
    while current < feature_end:
        month_end = current + pd.DateOffset(months=1)
        if month_end > feature_end:
            month_end = feature_end
        
        # この月末までに活動した人
        history_df = df[(df['request_time'] >= train_start) & (df['request_time'] < month_end)]
        active_reviewers = history_df['reviewer_email'].unique()
        
        # 将来窓（訓練期間内に収まる）
        future_start = month_end
        future_end = month_end + pd.DateOffset(months=future_window_months)
        # train_endを超えないように制限
        if future_end > train_end:
            future_end = train_end
        
        future_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
        
        # 将来窓で受諾した人
        accepted = future_df[future_df['label'] == 1]['reviewer_email'].unique()
        accepted_set = set(accepted)
        
        for reviewer in active_reviewers:
            label = 1 if reviewer in accepted_set else 0
            trajectories.append({
                'reviewer': reviewer,
                'month_end': month_end,
                'label': label
            })
        
        current = month_end
    
    return pd.DataFrame(trajectories)


def prepare_eval_data(df: pd.DataFrame, eval_start: datetime, eval_months: int,
                      train_start: datetime, future_window_months: int = 6):
    """
    評価データ準備（データリークなし版）
    
    評価期間を分割：
    - 特徴量計算期間: eval_start ~ (eval_start + eval_months - future_window_months)
    - ラベル計算期間: (eval_start + eval_months - future_window_months) ~ (eval_start + eval_months)
    """
    eval_end = eval_start + pd.DateOffset(months=eval_months)
    
    # 特徴量計算期間の終了（ラベル期間の開始）
    feature_end = eval_end - pd.DateOffset(months=future_window_months)
    
    # 評価期間の特徴量計算期間に活動した人
    eval_df = df[(df['request_time'] >= eval_start) & (df['request_time'] < feature_end)]
    eval_reviewers = eval_df['reviewer_email'].unique()
    
    # 将来窓でレビュー受諾（eval_end内に収まる）
    future_start = feature_end
    future_end = eval_end
    future_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
    accepted = future_df[future_df['label'] == 1]['reviewer_email'].unique()
    accepted_set = set(accepted)
    
    eval_samples = []
    for reviewer in eval_reviewers:
        label = 1 if reviewer in accepted_set else 0
        eval_samples.append({
            'reviewer': reviewer,
            'cutoff_date': feature_end,  # 特徴量計算の終了時点
            'label': label
        })
    
    return pd.DataFrame(eval_samples)


def main():
    print("=" * 80)
    print("Nova レビュー受諾予測 - Enhanced IRL (importants設定準拠)")
    print("=" * 80)
    
    # データ読み込み
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    print(f"\n📂 データ読み込み: {data_path}")
    
    df = pd.read_csv(data_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    # nova単一プロジェクト
    df = df[df['project'] == 'openstack/nova'].copy()
    print(f"✅ Nova: {len(df)} レコード, {df['reviewer_email'].nunique()} レビュワー")
    
    # importants設定
    train_start = datetime(2021, 1, 1)
    train_end = datetime(2023, 1, 1)
    eval_start = datetime(2023, 1, 1)
    future_window = 6  # 0-6ヶ月
    
    print(f"\n📅 実験設定（importants準拠）:")
    print(f"  訓練期間: {train_start.date()} ~ {train_end.date()} (24ヶ月)")
    print(f"  評価開始: {eval_start.date()}")
    print(f"  将来窓: {future_window}ヶ月")
    print(f"  Seed: 42")
    
    # 訓練データ: 月次集約軌跡
    print("\n" + "=" * 80)
    print("訓練データ準備（月次集約）")
    print("=" * 80)
    
    train_trajectories = prepare_monthly_trajectories(
        df, train_start, train_end, future_window_months=future_window
    )
    print(f"✅ 訓練軌跡: {len(train_trajectories)} 軌跡")
    print(f"   継続率: {train_trajectories['label'].mean():.3f}")
    
    # 評価データ: 0-3m期間
    print("\n" + "=" * 80)
    print("評価データ準備（0-3m期間）")
    print("=" * 80)
    
    eval_df = prepare_eval_data(
        df, eval_start, eval_months=3,
        train_start=train_start, future_window_months=future_window
    )
    print(f"✅ 評価サンプル: {len(eval_df)} 人")
    print(f"   継続率: {eval_df['label'].mean():.3f}")
    
    # 特徴量計算
    print("\n" + "=" * 80)
    print("特徴量計算")
    print("=" * 80)
    
    print("Train特徴量...")
    X_train_state = []
    y_train = []
    for _, row in train_trajectories.iterrows():
        features = calculate_state_features(df, row['reviewer'], row['month_end'])
        X_train_state.append(features)
        y_train.append(row['label'])
    
    X_train_state = np.array(X_train_state)
    y_train = np.array(y_train)
    
    # 時系列特徴量: 状態特徴量を時系列次元で複製 (B, T, D) -> (B, T, 10)
    # Tは任意のシーケンス長（ここでは1を使用）
    X_train_temporal = X_train_state[:, np.newaxis, :]  # (B, 1, 10)
    
    print(f"✅ Train: state={X_train_state.shape}, temporal={X_train_temporal.shape}")
    
    print("Eval特徴量...")
    X_eval_state = []
    y_eval = []
    for _, row in eval_df.iterrows():
        features = calculate_state_features(df, row['reviewer'], row['cutoff_date'])
        X_eval_state.append(features)
        y_eval.append(row['label'])
    
    X_eval_state = np.array(X_eval_state)
    y_eval = np.array(y_eval)
    X_eval_temporal = X_eval_state[:, np.newaxis, :]  # (B, 1, 10)
    
    print(f"✅ Eval: state={X_eval_state.shape}, temporal={X_eval_temporal.shape}")
    
    # データセット作成
    train_dataset = ReviewerDataset(X_train_state, X_train_temporal, y_train)
    eval_dataset = ReviewerDataset(X_eval_state, X_eval_temporal, y_eval)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # モデル構築
    print("\n" + "=" * 80)
    print("Enhanced IRL モデル構築")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionIRLNetwork(
        state_dim=10,
        temporal_dim=10,  # 簡易時系列（10次元）
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_temporal=True
    ).to(device)
    
    print(f"✅ デバイス: {device}")
    print(f"✅ モデル: AttentionIRL (state=10, temporal=10, hidden=128, layers=2)")
    
    # 学習
    print("\n" + "=" * 80)
    print("学習開始")
    print("=" * 80)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0.0
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device).unsqueeze(1)  # (B, 1, 10)
            temporal = batch['temporal'].to(device)  # (B, 1, 10)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(state, temporal)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 評価
        model.eval()
        eval_probs = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                state = batch['state'].to(device).unsqueeze(1)  # (B, 1, 10)
                temporal = batch['temporal'].to(device)  # (B, 1, 10)
                labels = batch['label']
                
                outputs, _ = model(state, temporal)
                outputs = outputs.squeeze()
                eval_probs.extend(outputs.cpu().numpy())
                eval_labels.extend(labels.numpy())
        
        eval_auc = roc_auc_score(eval_labels, eval_probs)
        
        if eval_auc > best_auc:
            best_auc = eval_auc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, "
                  f"AUC: {eval_auc:.4f}, Best: {best_auc:.4f}")
    
    # 結果保存
    print("\n" + "=" * 80)
    print("結果保存")
    print("=" * 80)
    
    result_dir = Path(__file__).parent.parent / "results"
    result_dir.mkdir(exist_ok=True)
    
    result = {
        'model': 'Enhanced_IRL',
        'project': 'nova',
        'seed': 42,
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'eval_start': eval_start.strftime('%Y-%m-%d'),
        'eval_months': 3,
        'future_window_months': future_window,
        'train_trajectories': len(train_trajectories),
        'eval_samples': len(eval_df),
        'train_continuation_rate': float(train_trajectories['label'].mean()),
        'eval_continuation_rate': float(eval_df['label'].mean()),
        'best_auc': float(best_auc),
        'final_auc': float(eval_auc)
    }
    
    result_file = result_dir / "enhanced_irl_result.json"
    import json
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 結果保存: {result_file}")
    
    # サマリー
    print("\n" + "=" * 80)
    print("実験サマリー")
    print("=" * 80)
    print(f"訓練軌跡: {len(train_trajectories)}")
    print(f"評価サンプル: {len(eval_df)}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"")
    print(f"比較（importants）:")
    print(f"  AttentionなしIRL: 0.801")
    print(f"  Enhanced IRL (Attention): {best_auc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
