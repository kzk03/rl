#!/usr/bin/env python3
"""
ロジスティック回帰ベースラインによる特徴量分析

ニューラルネットワークの複雑な非線形性を避け、
シンプルなロジスティック回帰で各特徴量の正負の影響を直接分析します。

利点:
- 重みの符号が直接解釈可能（正 = 継続を促進、負 = 離脱を促進）
- ReLU, Dropout, LSTM の影響を受けない
- 統計的有意性を検定可能
- より解釈しやすい
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

# プロジェクトのモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


# 特徴量名の定義
STATE_FEATURE_NAMES = [
    'experience_days', 'experience_normalized', 'total_changes',
    'activity_freq_7d', 'activity_freq_30d', 'activity_freq_90d',
    'lines_changed_7d', 'review_load_7d', 'review_load_30d',
    'unique_collaborators', 'avg_interaction_strength', 'cross_project_ratio',
    'total_projects', 'total_files_touched', 'file_type_diversity',
    'avg_directory_depth', 'specialization_score', 'avg_files_per_change',
    'avg_lines_per_change', 'avg_code_complexity', 'avg_complexity_7d',
    'avg_complexity_30d'
]

EXTENDED_STATE_FEATURES = STATE_FEATURE_NAMES + [
    'code_churn_7d', 'code_churn_30d', 'review_participation_rate',
    'review_response_time', 'avg_review_depth', 'multi_file_change_ratio',
    'collaboration_diversity', 'peak_activity_hour', 'weekend_activity_ratio',
    'consecutive_active_days'
]

ACTION_FEATURE_NAMES = [
    'action_type', 'intensity', 'quality', 'collaboration', 'timestamp_age'
]

EXTENDED_ACTION_FEATURES = ACTION_FEATURE_NAMES + [
    'change_size', 'files_count', 'complexity', 'response_latency'
]


def generate_realistic_data(n_samples: int = 2000, state_dim: int = 32, action_dim: int = 9,
                           seq_len: int = 15, continuation_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    よりリアルな合成データを生成

    Returns:
        states: [n_samples, seq_len, state_dim]
        actions: [n_samples, seq_len, action_dim]
        labels: [n_samples] - 継続ラベル（0 or 1）
    """
    print(f"\nGenerating realistic synthetic data: {n_samples} samples")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}, Seq len: {seq_len}")
    print(f"  Target continuation rate: {continuation_rate:.1%}")

    # 状態特徴の生成
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)

    # 経験日数 (0-3650日、対数正規分布)
    states[:, :, 0] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))

    # 正規化経験 (0-1)
    states[:, :, 1] = states[:, :, 0] / (states[:, :, 0].max() + 1)

    # 総変更数 (0-10000、対数正規分布)
    states[:, :, 2] = np.random.lognormal(mean=4, sigma=2, size=(n_samples, seq_len))

    # 活動頻度 7d, 30d, 90d (0-1の範囲、ベータ分布)
    states[:, :, 3] = np.random.beta(2, 5, size=(n_samples, seq_len))  # 7d
    states[:, :, 4] = np.random.beta(2, 4, size=(n_samples, seq_len))  # 30d
    states[:, :, 5] = np.random.beta(2, 3, size=(n_samples, seq_len))  # 90d

    # コード変更量 (0-10000、対数正規分布)
    states[:, :, 6] = np.random.lognormal(mean=5, sigma=2, size=(n_samples, seq_len))

    # レビュー負荷 (0-100、ガンマ分布)
    states[:, :, 7] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # 7d
    states[:, :, 8] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # 30d

    # コラボレーション指標 (0-1、ベータ分布)
    states[:, :, 9] = np.random.beta(2, 3, size=(n_samples, seq_len))  # unique_collaborators
    states[:, :, 10] = np.random.beta(2, 5, size=(n_samples, seq_len))  # avg_interaction_strength
    states[:, :, 11] = np.random.beta(2, 4, size=(n_samples, seq_len))  # cross_project_ratio

    # プロジェクト/ファイル数 (1-100、対数正規分布)
    states[:, :, 12] = np.random.lognormal(mean=1, sigma=1, size=(n_samples, seq_len))  # total_projects
    states[:, :, 13] = np.random.lognormal(mean=3, sigma=1.5, size=(n_samples, seq_len))  # total_files_touched
    states[:, :, 14] = np.random.beta(3, 3, size=(n_samples, seq_len))  # file_type_diversity

    # コード品質指標 (0-1、ベータ分布)
    states[:, :, 15] = np.random.beta(3, 2, size=(n_samples, seq_len))  # avg_directory_depth
    states[:, :, 16] = np.random.beta(3, 3, size=(n_samples, seq_len))  # specialization_score
    states[:, :, 17] = np.random.lognormal(mean=1, sigma=1, size=(n_samples, seq_len))  # avg_files_per_change
    states[:, :, 18] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # avg_lines_per_change
    states[:, :, 19] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_code_complexity
    states[:, :, 20] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_complexity_7d
    states[:, :, 21] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_complexity_30d

    if state_dim >= 32:
        # 拡張特徴量
        states[:, :, 22] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # code_churn_7d
        states[:, :, 23] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # code_churn_30d
        states[:, :, 24] = np.random.beta(3, 3, size=(n_samples, seq_len))  # review_participation_rate
        states[:, :, 25] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # review_response_time
        states[:, :, 26] = np.random.beta(2, 3, size=(n_samples, seq_len))  # avg_review_depth
        states[:, :, 27] = np.random.beta(3, 3, size=(n_samples, seq_len))  # multi_file_change_ratio
        states[:, :, 28] = np.random.beta(3, 2, size=(n_samples, seq_len))  # collaboration_diversity ★重要
        states[:, :, 29] = np.random.randint(0, 24, size=(n_samples, seq_len)).astype(float)  # peak_activity_hour
        states[:, :, 30] = np.random.beta(2, 5, size=(n_samples, seq_len))  # weekend_activity_ratio
        states[:, :, 31] = np.random.poisson(lam=5, size=(n_samples, seq_len)).astype(float)  # consecutive_active_days

    # 行動特徴の生成
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)

    # action_type (0-1、レビューが80%)
    actions[:, :, 0] = 0.8  # 定数（すべてレビュー）

    # intensity (0-1、ベータ分布)
    actions[:, :, 1] = np.random.beta(3, 2, size=(n_samples, seq_len))

    # quality (0-1、ベータ分布)
    actions[:, :, 2] = np.random.beta(4, 2, size=(n_samples, seq_len))

    # collaboration (0-1、ベータ分布)
    actions[:, :, 3] = np.random.beta(3, 3, size=(n_samples, seq_len))

    # timestamp_age (0-30日、指数分布)
    actions[:, :, 4] = np.random.exponential(scale=5, size=(n_samples, seq_len))

    if action_dim >= 9:
        # 拡張行動特徴
        actions[:, :, 5] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # change_size
        actions[:, :, 6] = np.random.poisson(lam=3, size=(n_samples, seq_len)).astype(float)  # files_count
        actions[:, :, 7] = np.random.beta(3, 3, size=(n_samples, seq_len))  # complexity
        actions[:, :, 8] = np.random.exponential(scale=2, size=(n_samples, seq_len))  # response_latency

    # ラベルの生成（リアルな相関を持たせる）
    # 継続確率を以下の特徴から計算:
    # - collaboration_diversity (正の相関)
    # - timestamp_age (負の相関 - 最近の活動がないと離脱)
    # - intensity (正の相関)
    # - activity_freq_90d (正の相関)

    continuation_prob = np.zeros(n_samples)

    # 最終タイムステップの特徴を使用
    if state_dim >= 32:
        collaboration_diversity = states[:, -1, 28]  # 協働多様性（正の影響）
        continuation_prob += 0.3 * collaboration_diversity

    activity_freq_90d = states[:, -1, 5]  # 90日活動頻度（正の影響）
    continuation_prob += 0.2 * activity_freq_90d

    intensity = actions[:, -1, 1]  # レビュー強度（正の影響）
    continuation_prob += 0.2 * intensity

    timestamp_age = actions[:, -1, 4]  # 活動の新しさ（負の影響 - 古いと離脱）
    continuation_prob -= 0.1 * (timestamp_age / 30.0)

    # ベースラインの継続率を調整
    continuation_prob = continuation_prob - continuation_prob.mean() + continuation_rate

    # [0, 1]に制約
    continuation_prob = np.clip(continuation_prob, 0, 1)

    # ラベル生成
    labels = (np.random.rand(n_samples) < continuation_prob).astype(np.float32)

    actual_rate = labels.mean()
    print(f"  Actual continuation rate: {actual_rate:.1%}")

    return states, actions, labels


def aggregate_sequences(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    時系列データを集約して1サンプル1ベクトルに変換

    集約方法:
    - 平均値
    - 最終値
    - 最大値
    - 最小値
    - トレンド（最終 - 最初）

    Args:
        states: [N, seq_len, state_dim]
        actions: [N, seq_len, action_dim]

    Returns:
        aggregated: [N, feature_dim]
    """
    n_samples, seq_len, state_dim = states.shape
    _, _, action_dim = actions.shape

    features = []

    # 状態特徴の集約
    features.append(states.mean(axis=1))  # 平均
    features.append(states[:, -1, :])     # 最終値
    features.append(states.max(axis=1))   # 最大値
    features.append(states.min(axis=1))   # 最小値
    features.append(states[:, -1, :] - states[:, 0, :])  # トレンド

    # 行動特徴の集約
    features.append(actions.mean(axis=1))  # 平均
    features.append(actions[:, -1, :])     # 最終値
    features.append(actions.max(axis=1))   # 最大値
    features.append(actions.min(axis=1))   # 最小値
    features.append(actions[:, -1, :] - actions[:, 0, :])  # トレンド

    aggregated = np.concatenate(features, axis=1)

    return aggregated


def create_feature_names(state_dim: int, action_dim: int) -> List[str]:
    """集約後の特徴量名を生成"""

    if state_dim == 32:
        state_names = EXTENDED_STATE_FEATURES
    elif state_dim == 22:
        state_names = STATE_FEATURE_NAMES
    else:
        state_names = [f'state_{i}' for i in range(state_dim)]

    if action_dim == 9:
        action_names = EXTENDED_ACTION_FEATURES
    elif action_dim == 5:
        action_names = ACTION_FEATURE_NAMES
    else:
        action_names = [f'action_{i}' for i in range(action_dim)]

    feature_names = []

    # 状態特徴
    for agg_type in ['mean', 'last', 'max', 'min', 'trend']:
        for name in state_names:
            feature_names.append(f'state_{name}_{agg_type}')

    # 行動特徴
    for agg_type in ['mean', 'last', 'max', 'min', 'trend']:
        for name in action_names:
            feature_names.append(f'action_{name}_{agg_type}')

    return feature_names


def train_logistic_regression(X: np.ndarray, y: np.ndarray,
                              feature_names: List[str],
                              C: float = 1.0,
                              max_iter: int = 1000) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """
    ロジスティック回帰を訓練

    Args:
        X: 特徴量 [N, feature_dim]
        y: ラベル [N]
        feature_names: 特徴量名
        C: 正則化の逆数（大きいほど正則化が弱い）
        max_iter: 最大イテレーション数

    Returns:
        model: 訓練済みモデル
        scaler: 標準化スケーラー
        metrics: 評価指標
    """
    print("\n--- Training Logistic Regression ---")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Continuation rate: {y.mean():.1%}")

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ロジスティック回帰訓練
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'  # 不均衡データ対応
    )

    model.fit(X_scaled, y)

    # 予測
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # 評価指標
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'auc_roc': roc_auc_score(y, y_pred_proba),
        'auc_pr': average_precision_score(y, y_pred_proba),
        'f1': f1_score(y, y_pred)
    }

    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    # 交差検証
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"  Cross-val AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, scaler, metrics


def analyze_coefficients(model: LogisticRegression,
                         scaler: StandardScaler,
                         feature_names: List[str],
                         top_k: int = 20) -> pd.DataFrame:
    """
    係数を分析

    Args:
        model: 訓練済みモデル
        scaler: 標準化スケーラー
        feature_names: 特徴量名
        top_k: 表示する上位k個

    Returns:
        df: 係数の詳細データフレーム
    """
    print("\n--- Analyzing Coefficients ---")

    # 係数の取得
    coefs = model.coef_[0]  # [feature_dim]
    intercept = model.intercept_[0]

    print(f"Intercept: {intercept:.4f}")
    print(f"Number of features: {len(coefs)}")

    # 標準化前の特徴量での係数（解釈しやすくするため）
    # coef_original = coef_scaled / std
    std = scaler.scale_
    coefs_original = coefs / std

    # データフレーム作成
    df = pd.DataFrame({
        'feature': feature_names,
        'coef_scaled': coefs,
        'coef_original': coefs_original,
        'abs_coef_scaled': np.abs(coefs),
        'abs_coef_original': np.abs(coefs_original)
    })

    # ソート
    df = df.sort_values('abs_coef_scaled', ascending=False)

    # 統計
    n_positive = (df['coef_scaled'] > 0).sum()
    n_negative = (df['coef_scaled'] < 0).sum()
    n_zero = (np.abs(df['coef_scaled']) < 1e-6).sum()

    print(f"\nCoefficient statistics:")
    print(f"  Positive (increase continuation): {n_positive} ({n_positive/len(df)*100:.1f}%)")
    print(f"  Negative (decrease continuation): {n_negative} ({n_negative/len(df)*100:.1f}%)")
    print(f"  Near-zero: {n_zero} ({n_zero/len(df)*100:.1f}%)")

    print(f"\nTop {top_k} features by absolute coefficient:")
    for idx, row in df.head(top_k).iterrows():
        sign = '+' if row['coef_scaled'] > 0 else '-'
        print(f"  {sign} {row['feature']:50s}  Coef: {row['coef_scaled']:8.4f}  (Original: {row['coef_original']:8.4f})")

    return df


def plot_coefficient_analysis(df: pd.DataFrame, output_dir: Path, top_k: int = 30):
    """係数を可視化"""

    # トップk個を抽出
    df_top = df.head(top_k).copy()
    df_top = df_top.sort_values('coef_scaled', ascending=True)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, top_k * 0.3)))

    # 1. 符号付き係数
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in df_top['coef_scaled']]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['coef_scaled'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_top['feature'], fontsize=9)
    ax1.set_xlabel('Coefficient (Scaled)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Logistic Regression Coefficients (Top {top_k})', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)

    # 凡例
    ax1.text(0.98, 0.02, 'Green: Increase continuation\nRed: Decrease continuation',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 絶対値係数（重要度）
    ax2 = axes[1]
    df_top_abs = df.nlargest(top_k, 'abs_coef_scaled').sort_values('abs_coef_scaled', ascending=True)
    y_pos = np.arange(len(df_top_abs))
    ax2.barh(y_pos, df_top_abs['abs_coef_scaled'], color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_top_abs['feature'], fontsize=9)
    ax2.set_xlabel('|Coefficient| (Importance)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Feature Importance by |Coefficient| (Top {top_k})', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'logistic_coefficients.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCoefficient plot saved: {output_path}")
    plt.close()

    # 3. 正と負の係数の分布
    fig, ax = plt.subplots(figsize=(12, 6))

    positive_df = df[df['coef_scaled'] > 0].nlargest(15, 'coef_scaled')
    negative_df = df[df['coef_scaled'] < 0].nsmallest(15, 'coef_scaled')

    y_pos_pos = np.arange(len(positive_df))
    y_pos_neg = np.arange(len(negative_df)) - len(negative_df) - 1

    ax.barh(y_pos_pos, positive_df['coef_scaled'], color='green', alpha=0.7, label='Positive (Increase continuation)')
    ax.barh(y_pos_neg, negative_df['coef_scaled'], color='red', alpha=0.7, label='Negative (Decrease continuation)')

    all_labels = list(negative_df['feature']) + [''] + list(positive_df['feature'])
    all_pos = list(y_pos_neg) + [0] + list(y_pos_pos)

    ax.set_yticks(all_pos)
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Top Positive and Negative Coefficients', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'logistic_coefficients_split.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Split coefficient plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Logistic regression baseline analysis')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of synthetic samples')
    parser.add_argument('--state-dim', type=int, default=32,
                        help='State feature dimension')
    parser.add_argument('--action-dim', type=int, default=9,
                        help='Action feature dimension')
    parser.add_argument('--seq-len', type=int, default=15,
                        help='Sequence length')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength')
    parser.add_argument('--top-k', type=int, default=30,
                        help='Number of top features to display')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ生成
    states, actions, labels = generate_realistic_data(
        n_samples=args.n_samples,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seq_len=args.seq_len
    )

    # 時系列を集約
    print("\n--- Aggregating sequences ---")
    X = aggregate_sequences(states, actions)
    print(f"Aggregated features: {X.shape[1]}")

    # 特徴量名を生成
    feature_names = create_feature_names(args.state_dim, args.action_dim)
    assert len(feature_names) == X.shape[1], f"Feature name mismatch: {len(feature_names)} != {X.shape[1]}"

    # ロジスティック回帰訓練
    model, scaler, metrics = train_logistic_regression(X, labels, feature_names, C=args.C)

    # 係数分析
    df_coefs = analyze_coefficients(model, scaler, feature_names, top_k=args.top_k)

    # 可視化
    plot_coefficient_analysis(df_coefs, output_dir, top_k=args.top_k)

    # CSV保存
    csv_path = output_dir / 'logistic_coefficients.csv'
    df_coefs.to_csv(csv_path, index=False)
    print(f"\nCoefficients saved: {csv_path}")

    # メトリクス保存
    metrics_path = output_dir / 'logistic_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # レポート作成
    report_path = output_dir / 'logistic_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION BASELINE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Model Configuration:\n")
        f.write(f"  Regularization (C): {args.C}\n")
        f.write(f"  Features: {X.shape[1]}\n")
        f.write(f"  Samples: {X.shape[0]}\n")
        f.write(f"  Continuation rate: {labels.mean():.1%}\n\n")

        f.write("Performance:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  AUC-ROC: {metrics['auc_roc']:.4f}\n")
        f.write(f"  AUC-PR: {metrics['auc_pr']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write(f"TOP {args.top_k} FEATURES (by absolute coefficient)\n")
        f.write("="*80 + "\n\n")

        for idx, row in df_coefs.head(args.top_k).iterrows():
            sign = '+' if row['coef_scaled'] > 0 else '-'
            effect = 'INCREASE continuation' if row['coef_scaled'] > 0 else 'DECREASE continuation'
            f.write(f"{sign} {row['feature']:60s}  Coef: {row['coef_scaled']:8.4f}  ({effect})\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Analysis complete! All outputs saved to: {output_dir}\n")
        f.write("="*80 + "\n")

    print(f"\nReport saved: {report_path}")
    print("\n" + "="*80)
    print("Logistic regression baseline analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
