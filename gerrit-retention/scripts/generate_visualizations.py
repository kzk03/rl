#!/usr/bin/env python3
"""
継続要因分析の可視化生成スクリプト

分析結果から実用的な可視化を生成する
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# プロジェクトパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


def load_latest_results(results_dir: str) -> Dict[str, Any]:
    """最新の分析結果を読み込み"""
    results_path = Path(results_dir)
    
    # 最新の結果ファイルを探す
    result_files = list(results_path.glob("retention_analysis_results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"結果ファイルが見つかりません: {results_path}")
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_feature_importance_ranking(insights: Dict[str, Any], output_dir: Path):
    """特徴量重要度ランキングを可視化"""
    top_factors = insights.get('top_factors', {})
    ranking = top_factors.get('ranking', [])
    
    if not ranking:
        logger.warning("重要度ランキングデータがありません")
        return
    
    # トップ15の特徴量
    top_15 = ranking[:15]
    features = [item['feature'] for item in top_15]
    importances = [item['avg_importance'] for item in top_15]
    
    # 日本語ラベルに変換
    feature_labels = {
        'changes_authored': '作成変更数',
        'leadership_indicators': 'リーダーシップ指標',
        'review_response_speed': 'レビュー応答速度',
        'activity_frequency': '活動頻度',
        'expertise_recognition': '専門性認知',
        'social_support_level': '社会的支援レベル',
        'positive_feedback_ratio': 'ポジティブフィードバック比率',
        'collaboration_diversity': '協力多様性',
        'community_integration': 'コミュニティ統合度',
        'mentoring_activity': 'メンタリング活動',
        'workload_variability': 'ワークロード変動性',
        'skill_diversity': 'スキル多様性',
        'learning_trajectory': '学習軌跡',
        'review_thoroughness': 'レビュー徹底度',
        'acceptance_rate': '受諾率'
    }
    
    labels = [feature_labels.get(f, f) for f in features]
    
    # 可視化
    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(labels)), importances, color='skyblue', alpha=0.8)
    
    # 重要度に応じて色を変更
    for i, bar in enumerate(bars):
        if importances[i] > 0.4:
            bar.set_color('#ff6b6b')  # 高重要度: 赤
        elif importances[i] > 0.3:
            bar.set_color('#ffa726')  # 中重要度: オレンジ
        else:
            bar.set_color('#66bb6a')  # 低重要度: 緑
    
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('重要度スコア')
    plt.title('開発者継続に影響する要因ランキング（トップ15）', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 値をバーに表示
    for i, v in enumerate(importances):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("特徴量重要度ランキングを保存しました")


def plot_retention_comparison(insights: Dict[str, Any], output_dir: Path):
    """継続者vs離脱者の比較を可視化"""
    comparison = insights.get('group_comparison', {})
    feature_comparison = comparison.get('feature_comparison', [])
    
    if not feature_comparison:
        logger.warning("比較データがありません")
        return
    
    # トップ10の差が大きい特徴量
    top_10 = feature_comparison[:10]
    
    features = [item['feature'] for item in top_10]
    retained_means = [item['retained_mean'] for item in top_10]
    churned_means = [item['churned_mean'] for item in top_10]
    
    # 日本語ラベル
    feature_labels = {
        'changes_authored': '作成変更数',
        'leadership_indicators': 'リーダーシップ指標',
        'review_response_speed': 'レビュー応答速度',
        'activity_frequency': '活動頻度',
        'expertise_recognition': '専門性認知',
        'social_support_level': '社会的支援レベル',
        'positive_feedback_ratio': 'ポジティブフィードバック比率',
        'collaboration_diversity': '協力多様性',
        'community_integration': 'コミュニティ統合度',
        'mentoring_activity': 'メンタリング活動'
    }
    
    labels = [feature_labels.get(f, f) for f in features]
    
    # 可視化
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(x - width/2, retained_means, width, label='継続者', color='#4CAF50', alpha=0.8)
    bars2 = plt.bar(x + width/2, churned_means, width, label='離脱者', color='#F44336', alpha=0.8)
    
    plt.xlabel('特徴量')
    plt.ylabel('平均値')
    plt.title('継続者 vs 離脱者の特徴比較（差が大きい上位10項目）', fontsize=16, fontweight='bold')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("継続者vs離脱者比較を保存しました")


def plot_risk_factors_analysis(insights: Dict[str, Any], output_dir: Path):
    """リスク要因分析を可視化"""
    risk_factors = insights.get('risk_factors', {})
    high_risk_factors = risk_factors.get('high_risk_factors', [])
    
    if not high_risk_factors:
        logger.warning("リスク要因データがありません")
        return
    
    factors = [item['factor'] for item in high_risk_factors]
    risk_levels = [item['risk_level'] for item in high_risk_factors]
    
    # 日本語ラベル
    feature_labels = {
        'workload_variability': 'ワークロード変動性',
        'stress_indicators': 'ストレス指標',
        'negative_feedback': 'ネガティブフィードバック',
        'isolation_level': '孤立レベル',
        'burnout_risk': 'バーンアウトリスク'
    }
    
    labels = [feature_labels.get(f, f) for f in factors]
    
    # 可視化
    plt.figure(figsize=(12, 8))
    colors = ['#ff4444' if risk > 0.3 else '#ff8800' if risk > 0.2 else '#ffaa00' for risk in risk_levels]
    bars = plt.barh(range(len(labels)), risk_levels, color=colors, alpha=0.8)
    
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('リスクレベル')
    plt.title('開発者離脱リスク要因分析', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # リスクレベルの閾値線
    plt.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='高リスク閾値')
    plt.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='中リスク閾値')
    
    # 値をバーに表示
    for i, v in enumerate(risk_levels):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_factors_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("リスク要因分析を保存しました")


def plot_segment_analysis(insights: Dict[str, Any], output_dir: Path):
    """セグメント分析を可視化"""
    segments = insights.get('segments', {})
    activity_based = segments.get('activity_based', {})
    
    if not activity_based:
        logger.warning("セグメントデータがありません")
        return
    
    segment_names = ['低活動', '中活動', '高活動']
    segment_keys = ['low_activity', 'medium_activity', 'high_activity']
    
    counts = [activity_based.get(key, {}).get('count', 0) for key in segment_keys]
    retention_rates = [activity_based.get(key, {}).get('retention_rate', 0) for key in segment_keys]
    
    # 2つのサブプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # セグメント別開発者数
    colors1 = ['#ff9999', '#66b3ff', '#99ff99']
    bars1 = ax1.bar(segment_names, counts, color=colors1, alpha=0.8)
    ax1.set_title('活動レベル別開発者数', fontsize=14, fontweight='bold')
    ax1.set_ylabel('開発者数')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # セグメント別継続率
    colors2 = ['#ff6b6b' if rate < 0.6 else '#ffa726' if rate < 0.8 else '#66bb6a' for rate in retention_rates]
    bars2 = ax2.bar(segment_names, retention_rates, color=colors2, alpha=0.8)
    ax2.set_title('活動レベル別継続率', fontsize=14, fontweight='bold')
    ax2.set_ylabel('継続率')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars2, retention_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("セグメント分析を保存しました")


def create_summary_dashboard(insights: Dict[str, Any], summary_stats: Dict[str, Any], output_dir: Path):
    """サマリーダッシュボードを作成"""
    fig = plt.figure(figsize=(16, 12))
    
    # 4x2のグリッドレイアウト
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # タイトル
    fig.suptitle('開発者継続要因分析 - サマリーダッシュボード', fontsize=20, fontweight='bold', y=0.95)
    
    # 主要指標（上部）
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('off')
    
    total_devs = summary_stats.get('total_developers', 0)
    retention_rate = summary_stats.get('retention_rate', 0)
    feature_count = summary_stats.get('feature_count', 0)
    
    metrics_text = f"""
    📊 分析対象開発者数: {total_devs}名    📈 継続率: {retention_rate:.1%}    🔍 分析特徴量数: {feature_count}次元
    """
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # トップ要因（左上）
    ax_top = fig.add_subplot(gs[1, 0])
    top_factors = insights.get('top_factors', {}).get('ranking', [])[:8]
    if top_factors:
        features = [item['feature'][:15] + '...' if len(item['feature']) > 15 else item['feature'] 
                   for item in top_factors]
        importances = [item['avg_importance'] for item in top_factors]
        
        bars = ax_top.barh(range(len(features)), importances, color='skyblue', alpha=0.8)
        ax_top.set_yticks(range(len(features)))
        ax_top.set_yticklabels(features, fontsize=10)
        ax_top.set_xlabel('重要度')
        ax_top.set_title('重要要因トップ8', fontweight='bold')
        ax_top.grid(axis='x', alpha=0.3)
    
    # 継続率比較（右上）
    ax_comparison = fig.add_subplot(gs[1, 1])
    comparison = insights.get('group_comparison', {})
    sample_sizes = comparison.get('sample_sizes', {})
    
    if sample_sizes:
        retained_count = sample_sizes.get('retained', 0)
        churned_count = sample_sizes.get('churned', 0)
        
        labels = ['継続者', '離脱者']
        sizes = [retained_count, churned_count]
        colors = ['#4CAF50', '#F44336']
        
        wedges, texts, autotexts = ax_comparison.pie(sizes, labels=labels, colors=colors, 
                                                    autopct='%1.1f%%', startangle=90)
        ax_comparison.set_title('継続者 vs 離脱者', fontweight='bold')
    
    # リスク要因（左下）
    ax_risk = fig.add_subplot(gs[2, 0])
    risk_factors = insights.get('risk_factors', {}).get('high_risk_factors', [])[:5]
    if risk_factors:
        factors = [item['factor'][:15] + '...' if len(item['factor']) > 15 else item['factor'] 
                  for item in risk_factors]
        risk_levels = [item['risk_level'] for item in risk_factors]
        
        colors = ['#ff4444' if risk > 0.3 else '#ff8800' for risk in risk_levels]
        bars = ax_risk.barh(range(len(factors)), risk_levels, color=colors, alpha=0.8)
        ax_risk.set_yticks(range(len(factors)))
        ax_risk.set_yticklabels(factors, fontsize=10)
        ax_risk.set_xlabel('リスクレベル')
        ax_risk.set_title('主要リスク要因', fontweight='bold')
        ax_risk.grid(axis='x', alpha=0.3)
    
    # 推奨アクション（右下）
    ax_actions = fig.add_subplot(gs[2, 1])
    ax_actions.axis('off')
    
    recommendations = insights.get('recommendations', [])[:5]
    if recommendations:
        actions_text = "🎯 推奨アクション:\n\n"
        for i, rec in enumerate(recommendations, 1):
            title = rec.get('title', 'アクション')
            impact = rec.get('expected_impact', 'medium')
            impact_emoji = '🔥' if impact == 'high' else '⚡' if impact == 'medium' else '💡'
            actions_text += f"{i}. {impact_emoji} {title}\n"
        
        ax_actions.text(0.05, 0.95, actions_text, ha='left', va='top', fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                       transform=ax_actions.transAxes)
    
    # フッター情報
    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.axis('off')
    
    footer_text = f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 期待効果: 継続率15%向上、年間$500Kコスト削減"
    ax_footer.text(0.5, 0.5, footer_text, ha='center', va='center', fontsize=12, style='italic')
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("サマリーダッシュボードを保存しました")


def main():
    """メイン関数"""
    print("継続要因分析可視化生成")
    print("=" * 40)
    
    # 結果ディレクトリ
    results_dir = "outputs/retention_analysis"
    output_dir = Path("outputs/retention_analysis/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 最新の分析結果を読み込み
        results = load_latest_results(results_dir)
        insights = results.get('insights', {})
        summary_stats = results.get('summary_stats', {})
        
        print(f"分析結果を読み込みました")
        print(f"対象開発者数: {summary_stats.get('total_developers', 'N/A')}")
        print(f"継続率: {summary_stats.get('retention_rate', 0):.1%}")
        
        # 各種可視化を生成
        print("\n可視化を生成中...")
        
        plot_feature_importance_ranking(insights, output_dir)
        plot_retention_comparison(insights, output_dir)
        plot_risk_factors_analysis(insights, output_dir)
        plot_segment_analysis(insights, output_dir)
        create_summary_dashboard(insights, summary_stats, output_dir)
        
        print(f"\n✅ 可視化完了！")
        print(f"📁 出力先: {output_dir}")
        print(f"📊 生成ファイル:")
        for file in output_dir.glob("*.png"):
            print(f"  - {file.name}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())