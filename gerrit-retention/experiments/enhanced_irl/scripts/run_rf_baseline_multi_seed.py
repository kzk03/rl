#!/usr/bin/env python3
"""
RFベースライン - Enhanced IRLと同条件での評価

Enhanced IRLと完全に同じ条件で実行:
- Train: current_monthに活動があった人、3-6ヶ月後に継続するか
- Eval: 予測期間にレビュー依頼があった人、その依頼を受け入れるか
- 10組み合わせ × 5シード = 50実験
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "experiments/enhanced_irl/scripts"))

# Enhanced IRLのprepare_data関数をインポート
from run_multi_seed_eval import prepare_data


def train_and_evaluate_rf(X_train, y_train, X_test, y_test, seed=777):
    """RFでトレーニングと評価"""
    # 状態特徴量のみ使用（10次元）
    # Temporal特徴量は使わない（RFはシーケンス不要）
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc


def main():
    print("=" * 80)
    print("RFベースライン - Enhanced IRL同条件評価")
    print("=" * 80)
    print()
    
    # データ読み込み
    data_path = project_root / "data/review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    print(f"データ: {len(df)}行\n")
    
    # 期間設定
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    
    # 期間組み合わせ（Enhanced IRLと同じ）
    future_periods = [(0, 3), (3, 6), (6, 9), (9, 12)]
    combinations = []
    for train_s, train_e in future_periods:
        for eval_s, eval_e in future_periods:
            if eval_s >= train_s:
                combinations.append((train_s, train_e, eval_s, eval_e))
    
    print(f"組み合わせ数: {len(combinations)}")
    
    # シード設定（1つのみ）
    seed = 42
    
    # フェーズ1: データ準備
    print("\n" + "=" * 80)
    print("フェーズ1: データ準備（10組み合わせ）")
    print("=" * 80)
    
    all_data = {}
    for idx, (train_s, train_e, eval_s, eval_e) in enumerate(combinations, 1):
        print(f"\n[{idx}/10] Train={train_s}-{train_e}m, Eval={eval_s}-{eval_e}m")
        
        key = (train_s, train_e, eval_s, eval_e)
        all_data[key] = prepare_data(
            df, train_start, train_end, eval_start, eval_end,
            train_s, train_e, eval_s, eval_e
        )
    
    # フェーズ2: トレーニング
    print("\n" + "=" * 80)
    print(f"フェーズ2: RFトレーニング（Seed={seed}, 10組み合わせ）")
    print("=" * 80)
    
    results = []
    
    for idx, (train_s, train_e, eval_s, eval_e) in enumerate(combinations, 1):
        key = (train_s, train_e, eval_s, eval_e)
        X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test = all_data[key]
        
        # 状態特徴量のみ使用（10次元）
        auc = train_and_evaluate_rf(X_train_state, y_train, X_test_state, y_test, seed=seed)
        
        print(f"  [{idx}/10] Train={train_s}-{train_e}m→Eval={eval_s}-{eval_e}m: AUC={auc:.4f}")
        
        results.append({
            'seed': seed,
            'train_start': train_s,
            'train_end': train_e,
            'eval_start': eval_s,
            'eval_end': eval_e,
            'auc': auc
        })
    
    # 結果保存
    results_df = pd.DataFrame(results)
    output_path = project_root / "experiments/enhanced_irl/results/rf_baseline_multi_seed_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n結果保存: {output_path}")
    
    # 統計出力
    print("\n" + "=" * 80)
    print("統計")
    print("=" * 80)
    
    print(f"\n平均AUC: {results_df['auc'].mean():.4f}")
    
    print("\n" + "=" * 80)
    print("詳細統計")
    print("=" * 80)
    
    mean_auc = results_df['auc'].mean()
    std_auc = results_df['auc'].std()
    print(f"\nSeed {seed}: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # 組み合わせごとの統計
    print("\n組み合わせごとのAUC:")
    for idx, (train_s, train_e, eval_s, eval_e) in enumerate(combinations, 1):
        combo_results = results_df[
            (results_df['train_start'] == train_s) &
            (results_df['train_end'] == train_e) &
            (results_df['eval_start'] == eval_s) &
            (results_df['eval_end'] == eval_e)
        ]
        if len(combo_results) > 0:
            print(f"  {train_s}-{train_e}→{eval_s}-{eval_e}: {combo_results['auc'].values[0]:.4f}")


if __name__ == "__main__":
    main()
