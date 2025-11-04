"""
各訓練期間モデルが予測できる開発者の特性を分析

各訓練期間（0-3m, 3-6m, 6-9m, 9-12m）のモデルが、
どのような活動量・受諾率の開発者を正確に予測できているかを可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

BASE_DIR = Path('outputs/review_acceptance_cross_eval_nova')
OUTPUT_DIR = BASE_DIR / 'model_coverage_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_all_predictions():
    """全ての predictions.csv を読み込んで統合"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    all_dfs = []

    for train_period in train_periods:
        for eval_period in eval_periods:
            pred_file = BASE_DIR / f'train_{train_period}' / f'eval_{eval_period}' / 'predictions.csv'

            if pred_file.exists():
                df = pd.read_csv(pred_file)
                df['train_period'] = train_period
                df['eval_period'] = eval_period
                all_dfs.append(df)
            else:
                print(f"⚠ ファイルが見つかりません: {pred_file}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ 予測データを統合しました: {len(combined_df)}件（{len(all_dfs)}ファイル）")

    # prediction_result列を作成
    combined_df['prediction_result'] = ''
    combined_df.loc[(combined_df['predicted_binary'] == 1) & (combined_df['true_label'] == 1), 'prediction_result'] = 'TP'
    combined_df.loc[(combined_df['predicted_binary'] == 0) & (combined_df['true_label'] == 0), 'prediction_result'] = 'TN'
    combined_df.loc[(combined_df['predicted_binary'] == 1) & (combined_df['true_label'] == 0), 'prediction_result'] = 'FP'
    combined_df.loc[(combined_df['predicted_binary'] == 0) & (combined_df['true_label'] == 1), 'prediction_result'] = 'FN'

    return combined_df

def plot_activity_distribution_by_period(df):
    """訓練期間別の活動量分布と予測成功率"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('訓練期間別：活動量と予測成功率の関係', fontsize=16, fontweight='bold')

    for idx, train_period in enumerate(train_periods):
        ax = axes[idx // 2, idx % 2]
        df_train = df[df['train_period'] == train_period]

        # 活動量でビン分け（対数スケール）
        bins = [0, 10, 50, 100, 200, 500, 2000]
        labels = ['<10', '10-50', '50-100', '100-200', '200-500', '≥500']
        df_train['activity_bin'] = pd.cut(df_train['history_request_count'], bins=bins, labels=labels)

        # 各ビンでの予測成功率を計算
        success_rates = []
        counts = []

        for label in labels:
            df_bin = df_train[df_train['activity_bin'] == label]
            if len(df_bin) > 0:
                success_rate = (df_bin['prediction_result'].isin(['TP', 'TN'])).mean()
                success_rates.append(success_rate)
                counts.append(len(df_bin))
            else:
                success_rates.append(0)
                counts.append(0)

        # 棒グラフ（成功率）
        x = np.arange(len(labels))
        bars = ax.bar(x, success_rates, alpha=0.7, color='skyblue', edgecolor='navy')

        # カウントを棒の上に表示
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}件\n{height:.2%}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('活動量（レビュー依頼数）', fontweight='bold')
        ax.set_ylabel('予測成功率 (Accuracy)', fontweight='bold')
        ax.set_title(f'訓練期間: {train_period}（n={len(df_train)}）', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activity_vs_accuracy_by_period.png', dpi=300, bbox_inches='tight')
    print(f"✓ 活動量別成功率を保存: {OUTPUT_DIR / 'activity_vs_accuracy_by_period.png'}")
    plt.close()

def plot_acceptance_distribution_by_period(df):
    """訓練期間別の受諾率分布と予測成功率"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('訓練期間別：受諾率と予測成功率の関係', fontsize=16, fontweight='bold')

    for idx, train_period in enumerate(train_periods):
        ax = axes[idx // 2, idx % 2]
        df_train = df[df['train_period'] == train_period]

        # 受諾率でビン分け
        bins = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 1.01]
        labels = ['0%', '1-5%', '5-10%', '10-20%', '20-30%', '>30%']
        df_train['acceptance_bin'] = pd.cut(df_train['history_acceptance_rate'], bins=bins, labels=labels)

        # 各ビンでの予測成功率を計算
        success_rates = []
        counts = []

        for label in labels:
            df_bin = df_train[df_train['acceptance_bin'] == label]
            if len(df_bin) > 0:
                success_rate = (df_bin['prediction_result'].isin(['TP', 'TN'])).mean()
                success_rates.append(success_rate)
                counts.append(len(df_bin))
            else:
                success_rates.append(0)
                counts.append(0)

        # 棒グラフ（成功率）
        x = np.arange(len(labels))
        colors = ['#cccccc', '#ff9999', '#ffcc99', '#99ff99', '#99ccff', '#ff99cc']
        bars = ax.bar(x, success_rates, alpha=0.7, color=colors, edgecolor='black')

        # カウントを棒の上に表示
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}件\n{height:.2%}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('受諾率', fontweight='bold')
        ax.set_ylabel('予測成功率 (Accuracy)', fontweight='bold')
        ax.set_title(f'訓練期間: {train_period}（n={len(df_train)}）', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'acceptance_vs_accuracy_by_period.png', dpi=300, bbox_inches='tight')
    print(f"✓ 受諾率別成功率を保存: {OUTPUT_DIR / 'acceptance_vs_accuracy_by_period.png'}")
    plt.close()

def plot_2d_heatmap_by_period(df):
    """訓練期間別の2次元ヒートマップ（活動量 × 受諾率）"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('訓練期間別：活動量 × 受諾率の予測成功率ヒートマップ', fontsize=16, fontweight='bold')

    activity_bins = [0, 50, 200, 2000]
    activity_labels = ['<50', '50-200', '≥200']

    acceptance_bins = [0, 0.05, 0.15, 0.30, 1.01]
    acceptance_labels = ['<5%', '5-15%', '15-30%', '>30%']

    for idx, train_period in enumerate(train_periods):
        df_train = df[df['train_period'] == train_period]

        df_train = df_train.copy()
        df_train['activity_bin'] = pd.cut(df_train['history_request_count'],
                                          bins=activity_bins, labels=activity_labels)
        df_train['acceptance_bin'] = pd.cut(df_train['history_acceptance_rate'],
                                            bins=acceptance_bins, labels=acceptance_labels)

        # 成功率マトリクスを作成
        success_matrix = np.full((len(activity_labels), len(acceptance_labels)), np.nan)
        count_matrix = np.zeros((len(activity_labels), len(acceptance_labels)))

        for i, act_label in enumerate(activity_labels):
            for j, acc_label in enumerate(acceptance_labels):
                df_cell = df_train[(df_train['activity_bin'] == act_label) &
                                   (df_train['acceptance_bin'] == acc_label)]
                if len(df_cell) > 0:
                    success_rate = (df_cell['prediction_result'].isin(['TP', 'TN'])).mean()
                    success_matrix[i, j] = success_rate
                    count_matrix[i, j] = len(df_cell)

        # 成功率ヒートマップ
        ax1 = axes[idx // 2, (idx % 2) * 2]
        im1 = ax1.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(np.arange(len(acceptance_labels)))
        ax1.set_yticks(np.arange(len(activity_labels)))
        ax1.set_xticklabels(acceptance_labels, fontsize=9)
        ax1.set_yticklabels(activity_labels, fontsize=9)
        ax1.set_xlabel('受諾率', fontsize=10)
        ax1.set_ylabel('活動量', fontsize=10)
        ax1.set_title(f'{train_period}: 成功率', fontweight='bold', fontsize=11)

        # 数値を追加
        for i in range(len(activity_labels)):
            for j in range(len(acceptance_labels)):
                if not np.isnan(success_matrix[i, j]):
                    text_color = 'white' if success_matrix[i, j] < 0.5 else 'black'
                    ax1.text(j, i, f'{success_matrix[i, j]:.2f}',
                            ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)

        # カウントヒートマップ
        ax2 = axes[idx // 2, (idx % 2) * 2 + 1]
        im2 = ax2.imshow(count_matrix, cmap='Blues', aspect='auto')
        ax2.set_xticks(np.arange(len(acceptance_labels)))
        ax2.set_yticks(np.arange(len(activity_labels)))
        ax2.set_xticklabels(acceptance_labels, fontsize=9)
        ax2.set_yticklabels(activity_labels, fontsize=9)
        ax2.set_xlabel('受諾率', fontsize=10)
        ax2.set_ylabel('活動量', fontsize=10)
        ax2.set_title(f'{train_period}: サンプル数', fontweight='bold', fontsize=11)

        # 数値を追加
        for i in range(len(activity_labels)):
            for j in range(len(acceptance_labels)):
                if count_matrix[i, j] > 0:
                    text_color = 'white' if count_matrix[i, j] > count_matrix.max() / 2 else 'black'
                    ax2.text(j, i, f'{int(count_matrix[i, j])}',
                            ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2d_heatmap_by_period.png', dpi=300, bbox_inches='tight')
    print(f"✓ 2次元ヒートマップを保存: {OUTPUT_DIR / '2d_heatmap_by_period.png'}")
    plt.close()

def create_success_failure_comparison(df):
    """予測成功・失敗の特性比較"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('訓練期間別：予測成功 vs 失敗の特性比較', fontsize=16, fontweight='bold')

    for idx, train_period in enumerate(train_periods):
        ax = axes[idx // 2, idx % 2]
        df_train = df[df['train_period'] == train_period]

        # 成功・失敗で分ける
        df_success = df_train[df_train['prediction_result'].isin(['TP', 'TN'])]
        df_failure = df_train[df_train['prediction_result'].isin(['FP', 'FN'])]

        # 散布図
        ax.scatter(df_success['history_acceptance_rate'],
                  df_success['history_request_count'],
                  alpha=0.5, s=80, color='green', label=f'成功 (n={len(df_success)})',
                  edgecolor='black', linewidth=0.5)

        ax.scatter(df_failure['history_acceptance_rate'],
                  df_failure['history_request_count'],
                  alpha=0.5, s=80, color='red', marker='x', label=f'失敗 (n={len(df_failure)})',
                  linewidth=2)

        # 平均値を表示
        if len(df_success) > 0:
            success_mean_acc = df_success['history_acceptance_rate'].mean()
            success_mean_act = df_success['history_request_count'].mean()
            ax.scatter([success_mean_acc], [success_mean_act], s=300, color='darkgreen',
                      marker='*', edgecolor='black', linewidth=2, zorder=5,
                      label=f'成功平均 ({success_mean_acc:.1%}, {success_mean_act:.0f}件)')

        if len(df_failure) > 0:
            failure_mean_acc = df_failure['history_acceptance_rate'].mean()
            failure_mean_act = df_failure['history_request_count'].mean()
            ax.scatter([failure_mean_acc], [failure_mean_act], s=300, color='darkred',
                      marker='*', edgecolor='black', linewidth=2, zorder=5,
                      label=f'失敗平均 ({failure_mean_acc:.1%}, {failure_mean_act:.0f}件)')

        ax.set_xlabel('受諾率', fontweight='bold', fontsize=11)
        ax.set_ylabel('活動量（レビュー依頼数、対数スケール）', fontweight='bold', fontsize=11)
        ax.set_title(f'訓練期間: {train_period}', fontweight='bold', fontsize=12)
        ax.set_yscale('log')
        ax.set_xlim(-0.05, 0.65)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'success_vs_failure_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ 成功vs失敗の散布図を保存: {OUTPUT_DIR / 'success_vs_failure_scatter.png'}")
    plt.close()

def create_summary_table(df):
    """サマリーテーブルを作成"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    summary_data = []

    for train_period in train_periods:
        df_train = df[df['train_period'] == train_period]
        df_success = df_train[df_train['prediction_result'].isin(['TP', 'TN'])]
        df_failure = df_train[df_train['prediction_result'].isin(['FP', 'FN'])]

        summary_data.append({
            '訓練期間': train_period,
            '総サンプル数': len(df_train),
            '成功数': len(df_success),
            '失敗数': len(df_failure),
            'Accuracy': f"{len(df_success) / len(df_train) if len(df_train) > 0 else 0:.3f}",
            '成功_平均活動量': f"{df_success['history_request_count'].mean():.1f}" if len(df_success) > 0 else "N/A",
            '成功_中央活動量': f"{df_success['history_request_count'].median():.1f}" if len(df_success) > 0 else "N/A",
            '成功_平均受諾率': f"{df_success['history_acceptance_rate'].mean():.1%}" if len(df_success) > 0 else "N/A",
            '失敗_平均活動量': f"{df_failure['history_request_count'].mean():.1f}" if len(df_failure) > 0 else "N/A",
            '失敗_中央活動量': f"{df_failure['history_request_count'].median():.1f}" if len(df_failure) > 0 else "N/A",
            '失敗_平均受諾率': f"{df_failure['history_acceptance_rate'].mean():.1%}" if len(df_failure) > 0 else "N/A",
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'period_comparison_summary.csv', index=False, encoding='utf-8-sig')
    print(f"✓ サマリーテーブルを保存: {OUTPUT_DIR / 'period_comparison_summary.csv'}")

    return summary_df

def main():
    """メイン処理"""
    print("=" * 80)
    print("訓練期間別モデルの予測カバレッジ分析")
    print("=" * 80)

    df = load_all_predictions()

    print("\n【分析開始】")
    print("1. 活動量別の予測成功率を分析中...")
    plot_activity_distribution_by_period(df)

    print("2. 受諾率別の予測成功率を分析中...")
    plot_acceptance_distribution_by_period(df)

    print("3. 2次元ヒートマップを作成中...")
    plot_2d_heatmap_by_period(df)

    print("4. 成功vs失敗の散布図を作成中...")
    create_success_failure_comparison(df)

    print("5. サマリーテーブルを作成中...")
    summary_df = create_summary_table(df)

    print("\n【サマリー】")
    print(summary_df.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"✓ 全ての分析が完了しました！")
    print(f"出力先: {OUTPUT_DIR}/")
    print(f"{'=' * 80}")

    # 主要な発見を表示
    print("\n【主要な発見】")
    for _, row in summary_df.iterrows():
        period = row['訓練期間']
        accuracy = row['Accuracy']
        success_acc = row['成功_平均受諾率']
        failure_acc = row['失敗_平均受諾率']
        print(f"{period}: Accuracy={accuracy}, 成功={success_acc}, 失敗={failure_acc}")

if __name__ == '__main__':
    main()
