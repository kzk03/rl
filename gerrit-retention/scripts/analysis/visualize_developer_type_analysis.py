"""
開発者特性別予測精度の可視化スクリプト

実行方法:
uv run python scripts/analysis/visualize_developer_type_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False

# データパス
BASE_DIR = Path("/Users/kazuki-h/rl/gerrit-retention/outputs/review_acceptance_cross_eval_nova")
PREDICTIONS_PATH = BASE_DIR / "train_0-3m" / "eval_6-9m" / "predictions.csv"
OUTPUT_DIR = BASE_DIR / "developer_analysis_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """予測データを読み込む"""
    df = pd.read_csv(PREDICTIONS_PATH)
    return df

def classify_developer(row):
    """開発者を経験レベルと受諾率で分類"""
    request_count = row['history_request_count']
    acceptance_rate = row['history_acceptance_rate']

    # 経験レベル分類
    if request_count < 10:
        exp_level = '初心者\n(<10)'
    elif request_count < 100:
        exp_level = '中級者\n(10-100)'
    elif request_count < 500:
        exp_level = '上級者\n(100-500)'
    else:
        exp_level = 'エキスパート\n(≥500)'

    # 受諾率レベル分類
    if acceptance_rate == 0:
        acc_level = '未承諾\n(0%)'
    elif acceptance_rate < 0.10:
        acc_level = '低受諾\n(1-10%)'
    elif acceptance_rate < 0.25:
        acc_level = '中受諾\n(10-25%)'
    elif acceptance_rate < 0.40:
        acc_level = '高受諾\n(25-40%)'
    else:
        acc_level = '超高受諾\n(>40%)'

    return exp_level, acc_level

def calculate_metrics(df_segment):
    """セグメントのメトリクスを計算"""
    if len(df_segment) == 0:
        return {
            'count': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }

    tp = ((df_segment['predicted_binary'] == 1) & (df_segment['true_label'] == 1)).sum()
    tn = ((df_segment['predicted_binary'] == 0) & (df_segment['true_label'] == 0)).sum()
    fp = ((df_segment['predicted_binary'] == 1) & (df_segment['true_label'] == 0)).sum()
    fn = ((df_segment['predicted_binary'] == 0) & (df_segment['true_label'] == 1)).sum()

    accuracy = (tp + tn) / len(df_segment) if len(df_segment) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'count': len(df_segment),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def create_experience_level_table_and_chart(df):
    """経験レベル別の予測精度表とグラフを作成"""
    print("=== 経験レベル別分析 ===")

    df['exp_level'], _ = zip(*df.apply(classify_developer, axis=1))

    # 表データ作成
    exp_order = ['初心者\n(<10)', '中級者\n(10-100)', '上級者\n(100-500)', 'エキスパート\n(≥500)']
    results = []

    for exp in exp_order:
        df_exp = df[df['exp_level'] == exp]
        metrics = calculate_metrics(df_exp)

        # 正例・負例の内訳
        actual_positive = (df_exp['true_label'] == 1).sum()
        actual_negative = (df_exp['true_label'] == 0).sum()

        results.append({
            '経験レベル': exp.replace('\n', ''),
            '総数': metrics['count'],
            '実際承諾': int(actual_positive),
            '実際拒否': int(actual_negative),
            'TP': metrics['tp'],
            'TN': metrics['tn'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # CSV保存
    results_df.to_csv(OUTPUT_DIR / "experience_level_analysis.csv", index=False, encoding='utf-8-sig')

    # グラフ作成（2×2サブプロット）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('経験レベル別予測精度分析', fontsize=16, fontweight='bold')

    # 1. Accuracy比較
    accuracies = [float(r['Accuracy']) for r in results]
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(exp_order)), accuracies, color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'])
    ax1.set_xticks(range(len(exp_order)))
    ax1.set_xticklabels(exp_order, fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('(A) 経験レベル別 Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.legend()

    # 2. 開発者数の分布
    counts = [r['総数'] for r in results]
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(exp_order)), counts, color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'])
    ax2.set_xticks(range(len(exp_order)))
    ax2.set_xticklabels(exp_order, fontsize=10)
    ax2.set_ylabel('開発者数', fontsize=11)
    ax2.set_title('(B) 経験レベル別 開発者数分布', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val}名',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 予測結果の内訳（積み上げ棒グラフ）
    ax3 = axes[1, 0]
    tps = [r['TP'] for r in results]
    tns = [r['TN'] for r in results]
    fps = [r['FP'] for r in results]
    fns = [r['FN'] for r in results]

    x_pos = np.arange(len(exp_order))
    width = 0.6

    p1 = ax3.bar(x_pos, tps, width, label='TP (正しく承諾予測)', color='#2ecc71')
    p2 = ax3.bar(x_pos, tns, width, bottom=tps, label='TN (正しく拒否予測)', color='#3498db')
    p3 = ax3.bar(x_pos, fps, width, bottom=np.array(tps)+np.array(tns), label='FP (誤って承諾予測)', color='#e74c3c')
    p4 = ax3.bar(x_pos, fns, width, bottom=np.array(tps)+np.array(tns)+np.array(fps), label='FN (誤って拒否予測)', color='#f39c12')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(exp_order, fontsize=10)
    ax3.set_ylabel('開発者数', fontsize=11)
    ax3.set_title('(C) 予測結果の内訳', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Precision vs Recall
    precisions = [float(r['Precision']) for r in results]
    recalls = [float(r['Recall']) for r in results]
    ax4 = axes[1, 1]

    colors_map = {'初心者\n(<10)': '#ff9999', '中級者\n(10-100)': '#ffcc99',
                  '上級者\n(100-500)': '#99ccff', 'エキスパート\n(≥500)': '#99ff99'}

    for i, exp in enumerate(exp_order):
        if precisions[i] > 0 or recalls[i] > 0:  # 0,0の点は表示しない
            ax4.scatter(recalls[i], precisions[i], s=300, alpha=0.6,
                       color=colors_map[exp], edgecolor='black', linewidth=2)
            ax4.text(recalls[i], precisions[i], exp.replace('\n', '\n'),
                    ha='center', va='center', fontsize=8, fontweight='bold')

    ax4.set_xlabel('Recall (再現率)', fontsize=11)
    ax4.set_ylabel('Precision (適合率)', fontsize=11)
    ax4.set_title('(D) Precision vs Recall', fontsize=12, fontweight='bold')
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experience_level_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: {OUTPUT_DIR / 'experience_level_analysis.png'}")
    plt.close()

def create_acceptance_rate_table_and_chart(df):
    """受諾率別の予測精度表とグラフを作成"""
    print("\n=== 受諾率別分析 ===")

    _, df['acc_level'] = zip(*df.apply(classify_developer, axis=1))

    # 表データ作成
    acc_order = ['未承諾\n(0%)', '低受諾\n(1-10%)', '中受諾\n(10-25%)', '高受諾\n(25-40%)', '超高受諾\n(>40%)']
    results = []

    for acc in acc_order:
        df_acc = df[df['acc_level'] == acc]
        metrics = calculate_metrics(df_acc)

        actual_positive = (df_acc['true_label'] == 1).sum()
        actual_negative = (df_acc['true_label'] == 0).sum()
        avg_acc_rate = df_acc['history_acceptance_rate'].mean() if len(df_acc) > 0 else 0

        results.append({
            '受諾率レベル': acc.replace('\n', ''),
            '総数': metrics['count'],
            '平均受諾率': f"{avg_acc_rate:.1%}",
            '実際承諾': int(actual_positive),
            '実際拒否': int(actual_negative),
            'TP': metrics['tp'],
            'TN': metrics['tn'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # CSV保存
    results_df.to_csv(OUTPUT_DIR / "acceptance_rate_analysis.csv", index=False, encoding='utf-8-sig')

    # グラフ作成（2×2サブプロット）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('過去受諾率別予測精度分析', fontsize=16, fontweight='bold')

    # 1. Accuracy比較
    accuracies = [float(r['Accuracy']) for r in results]
    ax1 = axes[0, 0]
    colors = ['#cccccc', '#ff9999', '#99ff99', '#ffcc99', '#ff99cc']
    bars1 = ax1.bar(range(len(acc_order)), accuracies, color=colors)
    ax1.set_xticks(range(len(acc_order)))
    ax1.set_xticklabels(acc_order, fontsize=9)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('(A) 受諾率別 Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, accuracies):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.legend()

    # 2. 開発者数の分布
    counts = [r['総数'] for r in results]
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(acc_order)), counts, color=colors)
    ax2.set_xticks(range(len(acc_order)))
    ax2.set_xticklabels(acc_order, fontsize=9)
    ax2.set_ylabel('開発者数', fontsize=11)
    ax2.set_title('(B) 受諾率別 開発者数分布', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, counts):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val}名',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 予測結果の内訳
    ax3 = axes[1, 0]
    tps = [r['TP'] for r in results]
    tns = [r['TN'] for r in results]
    fps = [r['FP'] for r in results]
    fns = [r['FN'] for r in results]

    x_pos = np.arange(len(acc_order))
    width = 0.6

    p1 = ax3.bar(x_pos, tps, width, label='TP (正しく承諾予測)', color='#2ecc71')
    p2 = ax3.bar(x_pos, tns, width, bottom=tps, label='TN (正しく拒否予測)', color='#3498db')
    p3 = ax3.bar(x_pos, fps, width, bottom=np.array(tps)+np.array(tns), label='FP (誤って承諾予測)', color='#e74c3c')
    p4 = ax3.bar(x_pos, fns, width, bottom=np.array(tps)+np.array(tns)+np.array(fps), label='FN (誤って拒否予測)', color='#f39c12')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(acc_order, fontsize=9)
    ax3.set_ylabel('開発者数', fontsize=11)
    ax3.set_title('(C) 予測結果の内訳', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. 実際の承諾率 vs 予測精度の関係
    ax4 = axes[1, 1]
    avg_rates = [float(r['平均受諾率'].rstrip('%'))/100 for r in results]

    # 各セグメントをプロット
    for i, acc in enumerate(acc_order):
        if counts[i] > 0:
            ax4.scatter(avg_rates[i], accuracies[i], s=counts[i]*30, alpha=0.6,
                       color=colors[i], edgecolor='black', linewidth=2)
            ax4.text(avg_rates[i], accuracies[i], acc.replace('\n', '\n'),
                    ha='center', va='center', fontsize=7, fontweight='bold')

    ax4.set_xlabel('平均過去受諾率', fontsize=11)
    ax4.set_ylabel('予測精度 (Accuracy)', fontsize=11)
    ax4.set_title('(D) 過去受諾率 vs 予測精度\n(バブルサイズ = 開発者数)', fontsize=12, fontweight='bold')
    ax4.set_xlim(-0.05, 0.65)
    ax4.set_ylim(0, 1.05)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "acceptance_rate_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: {OUTPUT_DIR / 'acceptance_rate_analysis.png'}")
    plt.close()

def create_2d_heatmap(df):
    """経験レベル × 受諾率の2次元ヒートマップを作成"""
    print("\n=== 2次元クロス分析（経験 × 受諾率） ===")

    df['exp_level'], df['acc_level'] = zip(*df.apply(classify_developer, axis=1))

    exp_order = ['初心者\n(<10)', '中級者\n(10-100)', '上級者\n(100-500)', 'エキスパート\n(≥500)']
    acc_order = ['未承諾\n(0%)', '低受諾\n(1-10%)', '中受諾\n(10-25%)', '高受諾\n(25-40%)', '超高受諾\n(>40%)']

    # 3つのヒートマップ：開発者数、Accuracy、予測成功者数
    count_matrix = np.zeros((len(exp_order), len(acc_order)))
    accuracy_matrix = np.full((len(exp_order), len(acc_order)), np.nan)
    success_matrix = np.zeros((len(exp_order), len(acc_order)))

    results_list = []

    for i, exp in enumerate(exp_order):
        for j, acc in enumerate(acc_order):
            df_segment = df[(df['exp_level'] == exp) & (df['acc_level'] == acc)]
            metrics = calculate_metrics(df_segment)

            count_matrix[i, j] = metrics['count']
            if metrics['count'] > 0:
                accuracy_matrix[i, j] = metrics['accuracy']
                success_matrix[i, j] = metrics['tp'] + metrics['tn']

                results_list.append({
                    '経験レベル': exp.replace('\n', ''),
                    '受諾率': acc.replace('\n', ''),
                    '開発者数': metrics['count'],
                    'TP': metrics['tp'],
                    'TN': metrics['tn'],
                    'FP': metrics['fp'],
                    'FN': metrics['fn'],
                    'Accuracy': f"{metrics['accuracy']:.3f}"
                })

    # CSV保存
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_DIR / "2d_cross_analysis.csv", index=False, encoding='utf-8-sig')
    print(results_df.to_string(index=False))

    # 3つのヒートマップを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('2次元クロス分析：経験レベル × 過去受諾率', fontsize=16, fontweight='bold')

    exp_labels = [e.replace('\n', '') for e in exp_order]
    acc_labels = [a.replace('\n', '') for a in acc_order]

    # 1. 開発者数
    ax1 = axes[0]
    sns.heatmap(count_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=acc_labels, yticklabels=exp_labels, ax=ax1,
                cbar_kws={'label': '開発者数'}, linewidths=1, linecolor='white')
    ax1.set_title('(A) 開発者数分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('過去受諾率', fontsize=11)
    ax1.set_ylabel('経験レベル', fontsize=11)

    # 2. Accuracy（NaNは灰色）
    ax2 = axes[1]
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=acc_labels, yticklabels=exp_labels, ax=ax2,
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                linewidths=1, linecolor='white', mask=np.isnan(accuracy_matrix))
    ax2.set_title('(B) 予測精度 (Accuracy)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('過去受諾率', fontsize=11)
    ax2.set_ylabel('')

    # 3. 予測成功者数（TP + TN）
    ax3 = axes[2]
    sns.heatmap(success_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=acc_labels, yticklabels=exp_labels, ax=ax3,
                cbar_kws={'label': '予測成功者数'}, linewidths=1, linecolor='white')
    ax3.set_title('(C) 予測成功者数 (TP + TN)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('過去受諾率', fontsize=11)
    ax3.set_ylabel('')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2d_heatmap_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: {OUTPUT_DIR / '2d_heatmap_analysis.png'}")
    plt.close()

def create_feature_importance_transition_chart():
    """特徴量重要度の期間別推移グラフを作成"""
    print("\n=== 特徴量重要度の期間別推移分析 ===")

    # 各期間の特徴量重要度を読み込み
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    feature_data = {'state': {}, 'action': {}}

    for period in periods:
        importance_path = BASE_DIR / f"train_{period}" / "feature_importance" / "gradient_importance.json"
        if importance_path.exists():
            import json
            with open(importance_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for feature, value in data['state_importance'].items():
                    if feature not in feature_data['state']:
                        feature_data['state'][feature] = []
                    feature_data['state'][feature].append(value)
                for feature, value in data['action_importance'].items():
                    if feature not in feature_data['action']:
                        feature_data['action'][feature] = []
                    feature_data['action'][feature].append(value)

    # 2つのグラフを作成（状態特徴量、行動特徴量）
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('特徴量重要度の期間別推移', fontsize=16, fontweight='bold')

    x_pos = np.arange(len(periods))

    # 1. 状態特徴量
    ax1 = axes[0]
    for feature, values in feature_data['state'].items():
        if len(values) == len(periods):
            ax1.plot(x_pos, values, marker='o', linewidth=2, markersize=8, label=feature)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(periods)
    ax1.set_ylabel('重要度（勾配ベース）', fontsize=11)
    ax1.set_title('(A) 状態特徴量の推移', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.grid(alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # 2. 行動特徴量
    ax2 = axes[1]
    for feature, values in feature_data['action'].items():
        if len(values) == len(periods):
            ax2.plot(x_pos, values, marker='s', linewidth=2, markersize=8, label=feature)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(periods)
    ax2.set_xlabel('訓練期間', fontsize=11)
    ax2.set_ylabel('重要度（勾配ベース）', fontsize=11)
    ax2.set_title('(B) 行動特徴量の推移', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.grid(alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_transition.png", dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: {OUTPUT_DIR / 'feature_importance_transition.png'}")
    plt.close()

    # 推移データをCSVで保存
    state_df = pd.DataFrame(feature_data['state'], index=periods)
    state_df.index.name = '期間'
    state_df.to_csv(OUTPUT_DIR / "state_feature_importance_transition.csv", encoding='utf-8-sig')

    action_df = pd.DataFrame(feature_data['action'], index=periods)
    action_df.index.name = '期間'
    action_df.to_csv(OUTPUT_DIR / "action_feature_importance_transition.csv", encoding='utf-8-sig')

    print("\n状態特徴量の推移:")
    print(state_df.to_string())
    print("\n行動特徴量の推移:")
    print(action_df.to_string())

def main():
    """メイン実行"""
    print("=" * 80)
    print("開発者特性別予測精度の可視化分析")
    print("=" * 80)

    df = load_data()
    print(f"\n総開発者数: {len(df)}名")
    print(f"実際承諾: {(df['true_label'] == 1).sum()}名")
    print(f"実際拒否: {(df['true_label'] == 0).sum()}名\n")

    # 各分析を実行
    create_experience_level_table_and_chart(df)
    create_acceptance_rate_table_and_chart(df)
    create_2d_heatmap(df)
    create_feature_importance_transition_chart()

    print("\n" + "=" * 80)
    print(f"✓ すべての分析が完了しました")
    print(f"✓ 出力ディレクトリ: {OUTPUT_DIR}")
    print("=" * 80)

    # 生成されたファイル一覧
    print("\n【生成されたファイル】")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
