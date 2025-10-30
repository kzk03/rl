#!/usr/bin/env python3
"""
特徴量重要度の日本語化スクリプト

既存の feature_importance.json を読み込み、
特徴量名を日本語に変換したJSONと可視化を生成
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 特徴量名の定義
STATE_FEATURES_JA = {
    "0": "経験日数",
    "1": "総変更数",
    "2": "総レビュー数",
    "3": "プロジェクト数",
    "4": "最近の活動頻度",
    "5": "平均活動間隔",
    "6": "活動トレンド",
    "7": "協力スコア",
    "8": "コード品質スコア",
    "9": "時間経過（新鮮さ）"
}

ACTION_FEATURES_JA = {
    "0": "行動タイプ",
    "1": "行動の強度",
    "2": "行動の質",
    "3": "協力度",
    "4": "タイムスタンプ"
}


def translate_feature_importance(input_json: Path, output_dir: Path):
    """特徴量重要度を日本語化"""
    
    # JSONを読み込み
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # 日本語版を作成
    translated = {
        "baseline_auc": data["baseline_auc"],
        "n_samples": data["n_samples"],
        "methods": {}
    }
    
    for method_name, method_data in data["methods"].items():
        translated["methods"][method_name] = {}
        
        # State特徴量を変換
        if "state" in method_data:
            translated["methods"][method_name]["state"] = {}
            for feat_id, value in method_data["state"].items():
                ja_name = STATE_FEATURES_JA.get(feat_id, f"State{feat_id}")
                translated["methods"][method_name]["state"][ja_name] = value
        
        # Action特徴量を変換
        if "action" in method_data:
            translated["methods"][method_name]["action"] = {}
            for feat_id, value in method_data["action"].items():
                ja_name = ACTION_FEATURES_JA.get(feat_id, f"Action{feat_id}")
                translated["methods"][method_name]["action"][ja_name] = value
    
    # 日本語版JSONを保存
    output_json = output_dir / "feature_importance_ja.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 日本語版JSON保存: {output_json}")
    
    # 可視化も作成
    visualize_translated(translated, output_dir)
    
    # サマリーも作成
    create_summary(translated, output_dir)


def visualize_translated(data: dict, output_dir: Path):
    """日本語版の可視化を作成"""
    
    for method_name in ["permutation", "gradient"]:
        if method_name not in data["methods"]:
            continue
        
        method_data = data["methods"][method_name]
        
        # State特徴量
        if "state" in method_data:
            state_features = list(method_data["state"].keys())
            state_values = list(method_data["state"].values())
            
            # ソート（重要度順）
            sorted_indices = np.argsort(state_values)[::-1]
            state_features_sorted = [state_features[i] for i in sorted_indices]
            state_values_sorted = [state_values[i] for i in sorted_indices]
            
            # プロット
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['green' if v > 0 else 'red' for v in state_values_sorted]
            ax.barh(state_features_sorted, state_values_sorted, color=colors, alpha=0.7)
            ax.set_xlabel('重要度', fontsize=12)
            ax.set_title(f'State特徴量重要度 ({method_name.capitalize()})', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            output_path = output_dir / f"feature_importance_{method_name}_state_ja.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ State可視化保存: {output_path}")
        
        # Action特徴量
        if "action" in method_data:
            action_features = list(method_data["action"].keys())
            action_values = list(method_data["action"].values())
            
            # ソート（重要度順）
            sorted_indices = np.argsort(action_values)[::-1]
            action_features_sorted = [action_features[i] for i in sorted_indices]
            action_values_sorted = [action_values[i] for i in sorted_indices]
            
            # プロット
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if v > 0 else 'red' for v in action_values_sorted]
            ax.barh(action_features_sorted, action_values_sorted, color=colors, alpha=0.7)
            ax.set_xlabel('重要度', fontsize=12)
            ax.set_title(f'Action特徴量重要度 ({method_name.capitalize()})', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            output_path = output_dir / f"feature_importance_{method_name}_action_ja.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Action可視化保存: {output_path}")


def create_summary(data: dict, output_dir: Path):
    """サマリーテキストを作成"""
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("特徴量重要度サマリー（日本語版）")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"ベースラインAUC-ROC: {data['baseline_auc']:.4f}")
    summary_lines.append(f"サンプル数: {data['n_samples']}")
    summary_lines.append("")
    
    for method_name in ["permutation", "gradient"]:
        if method_name not in data["methods"]:
            continue
        
        method_data = data["methods"][method_name]
        
        summary_lines.append("=" * 80)
        summary_lines.append(f"【{method_name.upper()}】")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # State特徴量
        if "state" in method_data:
            summary_lines.append("◆ State特徴量")
            summary_lines.append("-" * 80)
            
            state_items = sorted(method_data["state"].items(), key=lambda x: abs(x[1]), reverse=True)
            for feat_name, value in state_items:
                sign = "+" if value >= 0 else ""
                summary_lines.append(f"  {feat_name:20s}: {sign}{value:+.6f}")
            summary_lines.append("")
        
        # Action特徴量
        if "action" in method_data:
            summary_lines.append("◆ Action特徴量")
            summary_lines.append("-" * 80)
            
            action_items = sorted(method_data["action"].items(), key=lambda x: abs(x[1]), reverse=True)
            for feat_name, value in action_items:
                sign = "+" if value >= 0 else ""
                summary_lines.append(f"  {feat_name:20s}: {sign}{value:+.6f}")
            summary_lines.append("")
    
    summary_lines.append("=" * 80)
    summary_lines.append("重要な発見")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # 最重要特徴量を抽出
    if "permutation" in data["methods"]:
        perm_state = data["methods"]["permutation"].get("state", {})
        if perm_state:
            top_state = max(perm_state.items(), key=lambda x: abs(x[1]))
            summary_lines.append(f"🏆 最重要State特徴量（Permutation）: {top_state[0]} ({top_state[1]:+.6f})")
        
        perm_action = data["methods"]["permutation"].get("action", {})
        if perm_action:
            top_action = max(perm_action.items(), key=lambda x: abs(x[1]))
            summary_lines.append(f"🏆 最重要Action特徴量（Permutation）: {top_action[0]} ({top_action[1]:+.6f})")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    # サマリー保存
    output_summary = output_dir / "feature_importance_summary_ja.txt"
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    
    print(f"✓ サマリー保存: {output_summary}")
    
    # コンソールにも表示
    print("")
    print("\n".join(summary_lines))


def main():
    if len(sys.argv) < 2:
        print("使用法: python translate_feature_importance.py <feature_importance.json>")
        sys.exit(1)
    
    input_json = Path(sys.argv[1])
    if not input_json.exists():
        print(f"エラー: {input_json} が見つかりません")
        sys.exit(1)
    
    output_dir = input_json.parent
    
    print("=" * 80)
    print("特徴量重要度の日本語化開始")
    print("=" * 80)
    print(f"入力: {input_json}")
    print(f"出力先: {output_dir}")
    print("")
    
    translate_feature_importance(input_json, output_dir)
    
    print("")
    print("=" * 80)
    print("✅ 日本語化完了！")
    print("=" * 80)


if __name__ == "__main__":
    main()

