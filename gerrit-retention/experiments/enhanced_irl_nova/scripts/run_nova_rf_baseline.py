"""
Nova単一プロジェクト用ランダムフォレストベースライン

Enhanced IRL・Attention-less IRL との比較用
データ: openstack/nova のみ
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


def calculate_rf_features(df: pd.DataFrame, reviewer: str, context_date: datetime) -> np.ndarray:
    """RF用特徴量を計算（10次元 - IRL状態特徴量と同じ）"""
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


def prepare_data(df: pd.DataFrame, cutoff_date: datetime,
                 history_months: int = 6,
                 eval_future_start_months: int = 6,
                 eval_future_end_months: int = 9):
    """学習・評価データを準備"""
    
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
    continuation_rate = eval_df['label'].mean()
    
    print(f"✅ Eval: {len(eval_df)} サンプル, 継続率={continuation_rate:.3f}")
    
    # 学習データ：履歴期間内でサンプリング
    train_df = history_df.copy()
    train_reviewers = train_df['reviewer_email'].unique()
    
    # 月次サンプリング
    train_df['year_month'] = pd.to_datetime(train_df['request_time']).dt.to_period('M')
    months = train_df['year_month'].unique()
    
    train_samples = []
    for month in months:
        month_df = train_df[train_df['year_month'] == month]
        month_end = pd.Timestamp(month.to_timestamp()) + pd.DateOffset(months=1)
        
        # その月に活動した人を対象
        month_reviewers = month_df['reviewer_email'].unique()
        
        for reviewer in month_reviewers:
            reviewer_month_df = month_df[month_df['reviewer_email'] == reviewer]
            # その月に受け入れたか
            label = 1 if (reviewer_month_df['label'] == 1).any() else 0
            
            train_samples.append({
                'reviewer': reviewer,
                'cutoff_date': month_end,
                'label': label
            })
    
    train_df = pd.DataFrame(train_samples)
    train_continuation_rate = train_df['label'].mean()
    
    print(f"✅ Train: {len(train_df)} サンプル, 継続率={train_continuation_rate:.3f}")
    
    return train_df, eval_df


def main():
    print("=" * 80)
    print("Nova単一プロジェクト - ランダムフォレストベースライン")
    print("=" * 80)
    
    # データ読み込み
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    print(f"\n📂 データ読み込み: {data_path}")
    
    df = pd.read_csv(data_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    # nova単一プロジェクトのみ
    df = df[df['project'] == 'openstack/nova'].copy()
    print(f"✅ Nova単一プロジェクト: {len(df)} レコード, {df['reviewer_email'].nunique()} ユニークレビュワー")
    
    # 実験設定（Enhanced IRLと同じ）
    seed = 42
    np.random.seed(seed)
    
    # 2023年1月1日を基準にした実験
    cutoff_date = datetime(2023, 1, 1)
    history_months = 6
    eval_future_start = 6
    eval_future_end = 9
    
    print(f"\n📅 実験設定:")
    print(f"  カットオフ日: {cutoff_date.date()}")
    print(f"  履歴期間: {history_months}ヶ月")
    print(f"  評価将来窓: +{eval_future_start}〜+{eval_future_end}ヶ月")
    print(f"  Seed: {seed}")
    
    # データ準備
    print("\n" + "=" * 80)
    print("データ準備")
    print("=" * 80)
    
    train_df, eval_df = prepare_data(
        df, cutoff_date,
        history_months=history_months,
        eval_future_start_months=eval_future_start,
        eval_future_end_months=eval_future_end
    )
    
    # 特徴量計算
    print("\n" + "=" * 80)
    print("特徴量計算")
    print("=" * 80)
    
    print("Train特徴量計算中...")
    X_train = []
    y_train = []
    for _, row in train_df.iterrows():
        features = calculate_rf_features(df, row['reviewer'], row['cutoff_date'])
        X_train.append(features)
        y_train.append(row['label'])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Train: {X_train.shape}")
    
    print("Eval特徴量計算中...")
    X_eval = []
    y_eval = []
    for _, row in eval_df.iterrows():
        features = calculate_rf_features(df, row['reviewer'], row['cutoff_date'])
        X_eval.append(features)
        y_eval.append(row['label'])
    
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
    
    print(f"✅ Eval: {X_eval.shape}")
    
    # ランダムフォレスト学習
    print("\n" + "=" * 80)
    print("ランダムフォレスト学習")
    print("=" * 80)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    print("✅ 学習完了")
    
    # 評価
    print("\n" + "=" * 80)
    print("評価")
    print("=" * 80)
    
    # Train AUC
    train_probs = rf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Train AUC: {train_auc:.4f}")
    
    # Eval AUC
    eval_probs = rf.predict_proba(X_eval)[:, 1]
    eval_auc = roc_auc_score(y_eval, eval_probs)
    print(f"Eval AUC: {eval_auc:.4f}")
    
    # 結果保存
    print("\n" + "=" * 80)
    print("結果保存")
    print("=" * 80)
    
    result_dir = Path(__file__).parent.parent / "results"
    result_dir.mkdir(exist_ok=True)
    
    result = {
        'model': 'RandomForest',
        'project': 'nova',
        'seed': seed,
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'history_months': history_months,
        'eval_future_start': eval_future_start,
        'eval_future_end': eval_future_end,
        'train_samples': len(train_df),
        'eval_samples': len(eval_df),
        'train_continuation_rate': float(train_df['label'].mean()),
        'eval_continuation_rate': float(eval_df['label'].mean()),
        'train_auc': float(train_auc),
        'eval_auc': float(eval_auc),
        'n_estimators': 100,
        'max_depth': 10
    }
    
    result_file = result_dir / "rf_baseline_result.json"
    import json
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 結果保存: {result_file}")
    
    # サマリー
    print("\n" + "=" * 80)
    print("実験サマリー")
    print("=" * 80)
    print(f"モデル: RandomForest")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Eval AUC: {eval_auc:.4f}")
    print(f"")
    print(f"比較:")
    print(f"  Enhanced IRL (Attention): 0.8033")
    print(f"  Attention-less IRL: 0.7536")
    print(f"  RandomForest (baseline): {eval_auc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
