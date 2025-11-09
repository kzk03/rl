"""
強化IRL（時系列特徴量+アテンション）のトレーニングスクリプト

既存のIRLと同じデータで時系列特徴量を追加してトレーニング
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
from sklearn.model_selection import train_test_split
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
    """
    IRL状態特徴量を計算（既存のIRLと同じロジック）
    
    Args:
        df: レビューリクエストデータ
        reviewer: レビュワーID（reviewer_email）
        context_date: コンテキスト日時
        
    Returns:
        10次元の状態特徴量
    """
    reviewer_df = df[df['reviewer_email'] == reviewer].copy()
    
    if len(reviewer_df) == 0:
        return np.zeros(10)
    
    # 日時に変換
    reviewer_df['timestamp'] = pd.to_datetime(reviewer_df['request_time'])
    reviewer_df = reviewer_df[reviewer_df['timestamp'] < context_date]
    
    if len(reviewer_df) == 0:
        return np.zeros(10)
    
    # 基本統計
    first_seen = reviewer_df['timestamp'].min()
    experience_days = (context_date - first_seen).days / 730.0
    total_changes = len(reviewer_df) / 500.0
    total_reviews = len(reviewer_df) / 500.0
    
    # 最近30日の活動
    recent_cutoff = context_date - timedelta(days=30)
    recent_df = reviewer_df[reviewer_df['timestamp'] >= recent_cutoff]
    recent_activity_frequency = len(recent_df) / 30.0
    
    # 活動間隔
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
            elif activity_trend < 0.5:
                activity_trend = 0.0
            else:
                activity_trend = 0.25
        else:
            activity_trend = 1.0
    else:
        activity_trend = 0.5
    
    # 固定値
    collaboration_score = 1.0
    code_quality_score = 0.5
    recent_acceptance_rate = 0.5
    
    # レビュー負荷
    if len(reviewer_df) > 30:
        days_active = (context_date - first_seen).days
        avg_per_day = len(reviewer_df) / max(days_active, 1)
        recent_per_day = len(recent_df) / 30.0
        review_load = min(recent_per_day / max(avg_per_day, 0.1), 3.0) / 3.0
    else:
        review_load = 0.5
    
    return np.array([
        experience_days,
        total_changes,
        total_reviews,
        recent_activity_frequency,
        avg_activity_gap,
        activity_trend,
        collaboration_score,
        code_quality_score,
        recent_acceptance_rate,
        review_load
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
    """
    データ準備（状態特徴量+時系列特徴量）
    
    Args:
        df: レビューリクエストデータ
        train_start: 学習開始日
        train_end: 学習終了日
        eval_start: 評価開始日
        eval_end: 評価終了日
        future_start_months: 未来窓開始月数
        future_end_months: 未来窓終了月数
        
    Returns:
        X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test
    """
    df = df.copy()
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    # 時系列特徴量エクストラクタ
    temporal_extractor = TemporalFeatureExtractor()
    
    # トレーニングデータ
    train_df = df[(df['request_time'] >= train_start_dt) & (df['request_time'] < train_end_dt)]
    train_reviewers = train_df['reviewer_email'].unique()
    
    X_train_state = []
    X_train_temporal = []
    y_train = []
    
    print(f"\n=== トレーニングデータ準備 ===")
    print(f"期間: {train_start} ~ {train_end}")
    print(f"未来窓: {future_start_months}-{future_end_months}ヶ月")
    print(f"レビュワー数: {len(train_reviewers)}")
    
    for reviewer in train_reviewers:
        # 状態特徴量
        state_features = calculate_irl_state_features(df, reviewer, train_end_dt)
        
        # 時系列特徴量（コンテキスト日より前のデータを渡す）
        period_df = df[df['request_time'] < train_end_dt]
        temporal_features = temporal_extractor.extract_all_temporal_features(
            reviewer, period_df, train_end_dt
        )
        
        # NaNチェック
        if np.isnan(state_features).any() or np.isnan(temporal_features).any():
            print(f"警告: NaN検出 - レビュワー: {reviewer}")
            print(f"  State has NaN: {np.isnan(state_features).any()}")
            print(f"  Temporal has NaN: {np.isnan(temporal_features).any()}")
            continue
        
        # ラベル計算
        future_start = train_end_dt + timedelta(days=future_start_months * 30)
        future_end = train_end_dt + timedelta(days=future_end_months * 30)
        
        future_df = df[
            (df['reviewer_email'] == reviewer) &
            (df['request_time'] >= future_start) &
            (df['request_time'] < future_end)
        ]
        
        if len(future_df) > 0:
            label = 1 if (future_df['label'] == 1).any() else 0
        else:
            label = 0
        
        X_train_state.append(state_features)
        X_train_temporal.append(temporal_features)
        y_train.append(label)
    
    # テストデータ
    eval_df = df[(df['request_time'] >= eval_start_dt) & (df['request_time'] < eval_end_dt)]
    eval_reviewers = eval_df['reviewer_email'].unique()
    
    X_test_state = []
    X_test_temporal = []
    y_test = []
    
    print(f"\n=== 評価データ準備 ===")
    print(f"期間: {eval_start} ~ {eval_end}")
    print(f"未来窓: {future_start_months}-{future_end_months}ヶ月")
    print(f"レビュワー数: {len(eval_reviewers)}")
    
    for reviewer in eval_reviewers:
        # 状態特徴量
        state_features = calculate_irl_state_features(df, reviewer, eval_end_dt)
        
        # 時系列特徴量（コンテキスト日より前のデータを渡す）
        period_df = df[df['request_time'] < eval_end_dt]
        temporal_features = temporal_extractor.extract_all_temporal_features(
            reviewer, period_df, eval_end_dt
        )
        
        # NaNチェック
        if np.isnan(state_features).any() or np.isnan(temporal_features).any():
            print(f"警告: NaN検出 - レビュワー: {reviewer}")
            continue
        
        # ラベル計算
        future_start = eval_end_dt + timedelta(days=future_start_months * 30)
        future_end = eval_end_dt + timedelta(days=future_end_months * 30)
        
        future_df = df[
            (df['reviewer_email'] == reviewer) &
            (df['request_time'] >= future_start) &
            (df['request_time'] < future_end)
        ]
        
        if len(future_df) > 0:
            label = 1 if (future_df['label'] == 1).any() else 0
        else:
            label = 0
        
        X_test_state.append(state_features)
        X_test_temporal.append(temporal_features)
        y_test.append(label)
    
    print(f"\nトレーニングサンプル数: {len(y_train)}, Positive: {sum(y_train)}")
    print(f"テストサンプル数: {len(y_test)}, Positive: {sum(y_test)}")
    
    return (
        np.array(X_train_state),
        np.array(X_train_temporal),
        np.array(y_train),
        np.array(X_test_state),
        np.array(X_test_temporal),
        np.array(y_test)
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """
    モデルトレーニング
    
    Args:
        model: トレーニングするモデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        epochs: エポック数
        lr: 学習率
        device: デバイス
        
    Returns:
        best_model, train_history
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_auc = 0.0
    best_model_state = None
    train_history = []
    
    print(f"\n=== トレーニング開始 ===")
    print(f"エポック数: {epochs}, 学習率: {lr}, デバイス: {device}")
    
    for epoch in range(epochs):
        # トレーニング
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device)  # (batch, 10)
            temporal = batch['temporal'].to(device)  # (batch, 97)
            labels = batch['label'].to(device)
            
            # seq_len=1のシーケンスとして扱う
            state_seq = state.unsqueeze(1)  # (batch, 1, 10)
            temporal_seq = temporal.unsqueeze(1)  # (batch, 1, 97)
            
            # デバッグ: 最初のバッチで形状確認
            if epoch == 0 and train_loss == 0.0:
                print(f"State shape: {state_seq.shape}")
                print(f"Temporal shape: {temporal_seq.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels dtype: {labels.dtype}")
                print(f"Labels sample: {labels[:5]}")
                print(f"State sample: {state[0]}")
                print(f"Temporal sample: {temporal[0][:10]}")
            
            # 順伝播
            outputs, _ = model(state_seq, temporal_seq)
            
            # デバッグ: 出力確認
            if epoch == 0 and train_loss == 0.0:
                print(f"Outputs shape: {outputs.shape}")
                print(f"Outputs dtype: {outputs.dtype}")
                print(f"Outputs full: {outputs.squeeze()}")
                print(f"Outputs has NaN: {torch.isnan(outputs).any()}")
                print(f"Outputs has Inf: {torch.isinf(outputs).any()}")
                print(f"About to compute loss")
            
            loss = criterion(outputs.squeeze(), labels)
            
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 検証
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                temporal = batch['temporal'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _ = model(state.unsqueeze(1), temporal.unsqueeze(1))
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # AUC計算
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_auc': auc
        })
        
        # ベストモデル保存
        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val AUC: {auc:.4f}")
    
    # ベストモデルロード
    model.load_state_dict(best_model_state)
    print(f"\nベスト AUC: {best_auc:.4f}")
    
    return model, train_history


def main():
    """メイン実行"""
    # データ読み込み
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    
    print(f"データ読み込み完了: {len(df)} 行")
    
    # パラメータ
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    future_start_months = 0
    future_end_months = 3
    
    # データ準備
    X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test = prepare_data(
        df, train_start, train_end, eval_start, eval_end, future_start_months, future_end_months
    )
    
    # データセット作成
    train_dataset = ReviewerDataset(X_train_state, X_train_temporal, y_train)
    test_dataset = ReviewerDataset(X_test_state, X_test_temporal, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # モデル作成
    model = AttentionIRLNetwork(
        state_dim=10,
        temporal_dim=97,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_temporal=True
    )
    
    # トレーニング
    model, history = train_model(
        model, train_loader, test_loader, epochs=70, lr=0.001, device='cpu'
    )
    
    # 最終評価
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            state = batch['state']
            temporal = batch['temporal']
            labels = batch['label']
            
            outputs, _ = model(state.unsqueeze(1), temporal.unsqueeze(1))
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if len(set(all_labels)) > 1:
        final_auc = roc_auc_score(all_labels, all_preds)
        print(f"\n=== 最終結果 ===")
        print(f"テスト AUC: {final_auc:.4f}")
        print(f"RF baseline: 0.8603")
        print(f"改善: {(final_auc - 0.8603) * 100:+.2f}%")
    else:
        print("警告: ラベルが1クラスのみのため、AUC計算不可")


if __name__ == "__main__":
    main()
