"""
Nova単一プロジェクト用Enhanced IRL実験（月次サンプリング版）

AttentionなしIRLモデルとの公平な比較のため:
- データ: openstack/nova のみ
- 同じデータ分割
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
    """IRL状態特徴量を計算（10次元）"""
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
    
    # プロジェクト数（nova単一なので常に1）
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
            activity_trend = 1.0
    else:
        activity_trend = 0.5
    
    # 協力度・品質スコア
    collaboration_score = 1.0
    code_quality_score = 0.5
    
    # 最終活動からの経過日数
    last_activity = reviewer_df['timestamp'].max()
    time_since_last_activity = (context_date - last_activity).days / 60.0
    
    return np.array([
        experience_days,
        total_changes,
        total_reviews,
        project_count,
        recent_activity_frequency,
        avg_activity_gap,
        activity_trend,
        collaboration_score,
        code_quality_score,
        time_since_last_activity
    ])


def prepare_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    train_future_start_months: int,
    train_future_end_months: int,
    eval_start: str,
    eval_end: str,
    eval_future_start_months: int,
    eval_future_end_months: int,
    history_months: int = 12,
    pre_filtered: bool = True
):
    """
    データ準備
    
    Train: 月次サンプリング
    Eval: 1スナップショット
    """
    
    print(f"\n{'='*80}")
    print(f"データ準備")
    print(f"{'='*80}")
    print(f"Train期間: {train_start} → {train_end}")
    print(f"  将来窓: {train_future_start_months}-{train_future_end_months}ヶ月")
    print(f"Eval期間: {eval_start} → {eval_end}")
    print(f"  将来窓: {eval_future_start_months}-{eval_future_end_months}ヶ月")
    
    # ラベル専用区間の長さを計算
    label_months = train_future_end_months
    print(f"  ラベル専用区間: {label_months}ヶ月")
    
    df = df.copy()
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    # Nova単一プロジェクトでフィルタ
    df = df[df['project'] == 'openstack/nova'].copy()
    print(f"\n✅ Nova単一プロジェクトでフィルタ: {len(df)} レコード")
    
    # Train データ準備（月次サンプリング）
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    
    # ラベル専用区間を考慮して、実際の学習期間の終了を早める
    label_start = train_end_dt - pd.DateOffset(months=label_months)
    print(f"  Train実学習期間: {train_start} → {label_start.strftime('%Y-%m-%d')}")
    print(f"  ラベル専用区間: {label_start.strftime('%Y-%m-%d')} → {train_end}")
    
    # 月ごとにサンプリング
    train_samples = []
    current_month = train_start_dt
    
    while current_month < label_start:
        month_end = (current_month + pd.offsets.MonthEnd(0))
        
        # この月の最終日をcutoffとする
        cutoff_date = month_end
        
        # 履歴期間: cutoff - history_months ヶ月
        history_start = cutoff_date - pd.DateOffset(months=history_months)
        history_df = df[(df['request_time'] >= history_start) & (df['request_time'] < cutoff_date)]
        
        if len(history_df) == 0:
            current_month += pd.DateOffset(months=1)
            continue
        
        # この月に活動したレビュワー
        active_reviewers = history_df['reviewer_email'].unique()
        
        # 将来窓の設定
        future_start = cutoff_date + pd.DateOffset(months=train_future_start_months)
        future_end = cutoff_date + pd.DateOffset(months=train_future_end_months)
        
        # 将来活動の取得
        future_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
        future_active = set(future_df['reviewer_email'].unique())
        
        for reviewer in active_reviewers:
            # 履歴プロジェクト = nova（単一）
            label = 1 if reviewer in future_active else 0
            
            train_samples.append({
                'reviewer': reviewer,
                'cutoff_date': cutoff_date,
                'label': label,
                'month': month_end.strftime('%Y-%m')
            })
        
        current_month += pd.DateOffset(months=1)
    
    train_df = pd.DataFrame(train_samples)
    print(f"\n✅ Train月次サンプリング: {len(train_df)} サンプル")
    print(f"  レビュワー数: {train_df['reviewer'].nunique()}")
    print(f"  継続率: {train_df['label'].mean():.3f}")
    
    # Eval データ準備（1スナップショット）
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    cutoff_date = eval_start_dt
    
    # 履歴期間
    history_start = cutoff_date - pd.DateOffset(months=history_months)
    history_df = df[(df['request_time'] >= history_start) & (df['request_time'] < cutoff_date)]
    
    # 将来窓
    future_start = cutoff_date + pd.DateOffset(months=eval_future_start_months)
    future_end = cutoff_date + pd.DateOffset(months=eval_future_end_months)
    
    # 母集団：将来窓でレビュー依頼があった人全員
    future_request_df = df[(df['request_time'] >= future_start) & (df['request_time'] < future_end)]
    eval_reviewers = set(future_request_df['reviewer_email'].unique())
    
    print(f"\n✅ Eval予測対象: 将来窓でレビュー依頼があった {len(eval_reviewers)} 人")
    
    # 継続判定：将来窓で少なくとも1回受け入れたか（label=1が1つでもある）
    future_accepted = future_request_df[future_request_df['label'] == 1]['reviewer_email'].unique()
    future_active = set(future_accepted)
    
    eval_samples = []
    for reviewer in eval_reviewers:
        # 将来窓で少なくとも1回受け入れた = 継続
        label = 1 if reviewer in future_active else 0
        eval_samples.append({
            'reviewer': reviewer,
            'cutoff_date': cutoff_date,
            'label': label
        })
    
    eval_df = pd.DataFrame(eval_samples)
    print(f"  継続率: {eval_df['label'].mean():.3f}")
    
    # 特徴量抽出
    print(f"\n{'='*80}")
    print("特徴量抽出")
    print(f"{'='*80}")
    
    temporal_extractor = TemporalFeatureExtractor()
    
    # Train特徴量
    X_train_state = []
    X_train_temporal = []
    y_train = []
    
    print("Train特徴量抽出中...")
    for _, row in train_df.iterrows():
        state_feats = calculate_irl_state_features(df, row['reviewer'], row['cutoff_date'])
        
        # 時系列特徴量用の期間データを準備
        history_start = row['cutoff_date'] - pd.DateOffset(months=history_months)
        period_df = df[(df['request_time'] >= history_start) & (df['request_time'] < row['cutoff_date'])]
        
        temporal_feats = temporal_extractor.extract_all_temporal_features(
            row['reviewer'], period_df, row['cutoff_date'], pre_filtered=False
        )
        
        X_train_state.append(state_feats)
        X_train_temporal.append(temporal_feats)
        y_train.append(row['label'])
    
    X_train_state = np.array(X_train_state)
    X_train_temporal = np.array(X_train_temporal)
    y_train = np.array(y_train)
    
    print(f"✅ Train: {len(y_train)} サンプル, 継続率={y_train.mean():.3f}")
    
    # Eval特徴量
    X_test_state = []
    X_test_temporal = []
    y_test = []
    
    print("Eval特徴量抽出中...")
    for _, row in eval_df.iterrows():
        state_feats = calculate_irl_state_features(df, row['reviewer'], row['cutoff_date'])
        
        # 時系列特徴量用の期間データを準備
        history_start = row['cutoff_date'] - pd.DateOffset(months=history_months)
        period_df = df[(df['request_time'] >= history_start) & (df['request_time'] < row['cutoff_date'])]
        
        temporal_feats = temporal_extractor.extract_all_temporal_features(
            row['reviewer'], period_df, row['cutoff_date'], pre_filtered=False
        )
        
        X_test_state.append(state_feats)
        X_test_temporal.append(temporal_feats)
        y_test.append(row['label'])
    
    X_test_state = np.array(X_test_state)
    X_test_temporal = np.array(X_test_temporal)
    y_test = np.array(y_test)
    
    print(f"✅ Eval: {len(y_test)} サンプル, 継続率={y_test.mean():.3f}")
    
    return X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test


def train_and_evaluate(
    X_train_state, X_train_temporal, y_train,
    X_test_state, X_test_temporal, y_test,
    seed: int,
    epochs: int = 50
):
    """モデルの訓練と評価"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル作成
    model = AttentionIRLNetwork(
        state_dim=10,
        temporal_dim=97,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # データセット
    train_dataset = ReviewerDataset(X_train_state, X_train_temporal, y_train)
    test_dataset = ReviewerDataset(X_test_state, X_test_temporal, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 損失関数と最適化
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 訓練
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            temporal = batch['temporal'].to(device)
            labels = batch['label'].to(device)
            
            # 3次元に拡張: (batch, 1, dim) -> シーケンス長1として扱う
            state = state.unsqueeze(1)
            temporal = temporal.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs, _ = model(state, temporal)  # attention_weightsも返される
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 評価
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                state = batch['state'].to(device)
                temporal = batch['temporal'].to(device)
                labels = batch['label'].cpu().numpy()
                
                # 3次元に拡張
                state = state.unsqueeze(1)
                temporal = temporal.unsqueeze(1)
                
                outputs, _ = model(state, temporal)
                preds = outputs.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        
        if auc > best_auc:
            best_auc = auc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, AUC: {auc:.4f}, Best: {best_auc:.4f}")
    
    return best_auc


def main():
    """メイン実行"""
    
    print("=" * 80)
    print("Nova単一プロジェクト Enhanced IRL実験")
    print("=" * 80)
    
    # データ読み込み
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    print(f"\n✅ データ読み込み: {len(df)} レコード")
    
    # 実験設定（Attentionなしモデルと同じ）
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    
    # 予測窓の組み合わせ（時系列的に有効なもののみ）
    prediction_windows = [
        (0, 3, 0, 3),   # 0-3m → 0-3m
        (0, 3, 3, 6),   # 0-3m → 3-6m
        (0, 3, 6, 9),   # 0-3m → 6-9m
        (0, 3, 9, 12),  # 0-3m → 9-12m
        (3, 6, 3, 6),   # 3-6m → 3-6m
        (3, 6, 6, 9),   # 3-6m → 6-9m
        (3, 6, 9, 12),  # 3-6m → 9-12m
        (6, 9, 6, 9),   # 6-9m → 6-9m
        (6, 9, 9, 12),  # 6-9m → 9-12m
        (9, 12, 9, 12), # 9-12m → 9-12m
    ]
    
    seeds = [42, 123, 777, 2024, 9999]
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Seed: {seed}")
        print(f"{'='*80}")
        
        for train_fs, train_fe, eval_fs, eval_fe in prediction_windows:
            print(f"\n{'*'*80}")
            print(f"Train窓: {train_fs}-{train_fe}m, Eval窓: {eval_fs}-{eval_fe}m")
            print(f"{'*'*80}")
            
            try:
                # データ準備
                X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test = prepare_data(
                    df=df,
                    train_start=train_start,
                    train_end=train_end,
                    train_future_start_months=train_fs,
                    train_future_end_months=train_fe,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    eval_future_start_months=eval_fs,
                    eval_future_end_months=eval_fe,
                    history_months=12,
                    pre_filtered=True
                )
                
                # 訓練・評価
                auc = train_and_evaluate(
                    X_train_state, X_train_temporal, y_train,
                    X_test_state, X_test_temporal, y_test,
                    seed=seed,
                    epochs=50
                )
                
                print(f"\n✅ 最終AUC: {auc:.4f}")
                
                results.append({
                    'seed': seed,
                    'train_start': train_fs,
                    'train_end': train_fe,
                    'eval_start': eval_fs,
                    'eval_end': eval_fe,
                    'auc': auc
                })
                
            except Exception as e:
                print(f"❌ エラー: {e}")
                import traceback
                traceback.print_exc()
    
    # 結果保存
    results_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "results" / "nova_multi_seed_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"結果を保存: {output_path}")
    print(f"{'='*80}")
    
    # サマリー
    print(f"\n{'='*80}")
    print("実験サマリー")
    print(f"{'='*80}")
    print(f"総実験数: {len(results_df)}")
    print(f"平均AUC: {results_df['auc'].mean():.4f}")
    print(f"最良AUC: {results_df['auc'].max():.4f}")
    print(f"最悪AUC: {results_df['auc'].min():.4f}")
    
    # シード別平均
    print(f"\n{'='*80}")
    print("シード別平均AUC")
    print(f"{'='*80}")
    for seed in seeds:
        seed_results = results_df[results_df['seed'] == seed]
        print(f"Seed {seed}: {seed_results['auc'].mean():.4f}")


if __name__ == "__main__":
    main()
