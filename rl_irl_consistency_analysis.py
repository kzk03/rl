#!/usr/bin/env python3
"""Analyze the developer count consistency between RL and IRL."""

import json
import pickle
from pathlib import Path

import numpy as np
import yaml


def analyze_rl_irl_consistency():
    """Analyze consistency between RL and IRL developer counts."""
    print("=" * 80)
    print("🤖 強化学習(RL) vs 逆強化学習(IRL) 開発者数一致性分析")
    print("=" * 80)
    
    # 1. IRL Expert Trajectories の開発者数
    print("\n1️⃣ IRL Expert Trajectories の開発者分析")
    print("-" * 50)
    
    try:
        with open('/Users/kazuki-h/rl/kazoo/data/expert_trajectories.pkl', 'rb') as f:
            trajectories = pickle.load(f)
        
        if isinstance(trajectories, list) and len(trajectories) > 0:
            trajectory = trajectories[0]
            
            # 開発者を抽出
            developers = set()
            for step in trajectory:
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    developer = action_details.get('developer')
                    if developer:
                        developers.add(developer)
            
            irl_developer_count = len(developers)
            print(f"📊 IRL専門家軌跡の開発者数: {irl_developer_count}人")
            print(f"📊 軌跡のステップ数: {len(trajectory):,}ステップ")
            
            # よく登場する開発者 TOP 10
            dev_counts = {}
            for step in trajectory:
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    developer = action_details.get('developer')
                    if developer:
                        dev_counts[developer] = dev_counts.get(developer, 0) + 1
            
            top_devs = sorted(dev_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n👑 最も活動的な開発者 TOP 10:")
            for i, (dev, count) in enumerate(top_devs, 1):
                print(f"  {i:2d}. {dev:<20} : {count:4d}回の行動")
            
        else:
            print("❌ Expert trajectories が見つからないか、形式が不正です")
            irl_developer_count = 0
            
    except Exception as e:
        print(f"❌ Expert trajectories読み込みエラー: {e}")
        irl_developer_count = 0
    
    # 2. RL設定の開発者数確認
    print(f"\n2️⃣ RL設定ファイルの開発者数")
    print("-" * 50)
    
    config_files = [
        ('/Users/kazuki-h/rl/kazoo/configs/base_training.yaml', 'トレーニング用'),
        ('/Users/kazuki-h/rl/kazoo/configs/base_test_2022.yaml', 'テスト用(2022)'),
        ('/Users/kazuki-h/rl/kazoo/configs/rl_experiment.yaml', '実験用'),
    ]
    
    rl_configs = {}
    for config_path, desc in config_files:
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                num_developers = config.get('num_developers', 'N/A')
                print(f"  📋 {desc}: {num_developers}人 ({Path(config_path).name})")
                rl_configs[desc] = num_developers
                
            except Exception as e:
                print(f"  ❌ {desc}: 読み込みエラー ({e})")
        else:
            print(f"  ❌ {desc}: ファイルが見つかりません")
    
    # 3. 開発者プロファイルの実際の数
    print(f"\n3️⃣ 開発者プロファイルファイルの実際の数")
    print("-" * 50)
    
    profile_files = [
        ('/Users/kazuki-h/rl/kazoo/configs/dev_profiles_training.yaml', 'トレーニング用'),
        ('/Users/kazuki-h/rl/kazoo/configs/dev_profiles_test_2022.yaml', 'テスト用(2022)'),
        ('/Users/kazuki-h/rl/kazoo/configs/dev_profiles.yaml', '通常用'),
    ]
    
    profile_counts = {}
    for profile_path, desc in profile_files:
        if Path(profile_path).exists():
            try:
                with open(profile_path, 'r') as f:
                    profiles = yaml.safe_load(f)
                
                if isinstance(profiles, dict):
                    count = len(profiles)
                    print(f"  📋 {desc}: {count:,}人 ({Path(profile_path).name})")
                    profile_counts[desc] = count
                else:
                    print(f"  ❌ {desc}: プロファイル形式が不正")
                    
            except Exception as e:
                print(f"  ❌ {desc}: 読み込みエラー ({e})")
        else:
            print(f"  ❌ {desc}: ファイルが見つかりません")
    
    # 4. 一致性分析
    print(f"\n4️⃣ 一致性分析")
    print("-" * 50)
    
    print(f"🔍 IRL専門家軌跡の開発者数: {irl_developer_count}人")
    
    if rl_configs:
        print(f"\n📊 RL設定との比較:")
        for desc, count in rl_configs.items():
            if isinstance(count, int):
                match_status = "✅ 一致" if count == irl_developer_count else "❌ 不一致"
                ratio = f"({count/irl_developer_count:.1f}倍)" if irl_developer_count > 0 else ""
                print(f"  {desc}: {count}人 {match_status} {ratio}")
    
    if profile_counts:
        print(f"\n📊 開発者プロファイルとの比較:")
        for desc, count in profile_counts.items():
            if isinstance(count, int) and irl_developer_count > 0:
                match_status = "✅ 一致" if count == irl_developer_count else "❌ 不一致"
                ratio = f"({count/irl_developer_count:.1f}倍)" if irl_developer_count > 0 else ""
                print(f"  {desc}: {count:,}人 {match_status} {ratio}")
    
    # 5. 推奨設定
    print(f"\n5️⃣ 推奨設定")
    print("-" * 50)
    
    print(f"""
🎯 開発者数設定の推奨事項:

✅ 【基本原則】
   IRL専門家軌跡の開発者数 = RL設定の num_developers

✅ 【現在の状況】
   • IRL専門家軌跡: {irl_developer_count}人
   • この数に合わせてRL設定を調整することを推奨

✅ 【設定変更が必要なファイル】""")
    
    for desc, count in rl_configs.items():
        if isinstance(count, int) and count != irl_developer_count:
            print(f"   • {desc}: {count}人 → {irl_developer_count}人 に変更")
    
    print(f"""
🔧 【変更手順】
   1. 各設定ファイルの num_developers を {irl_developer_count} に変更
   2. 特に小規模実験の場合は、部分的な軌跡を使用することも可能
   3. 大規模実験の場合は、追加の専門家軌跡生成を検討

💡 【実験段階別の推奨】
   • デバッグ・概念実証: 20-50人 (IRL軌跡をサンプリング)
   • 本格実験: {irl_developer_count}人 (完全な軌跡を使用)
   • 大規模評価: {irl_developer_count}人以上 (追加軌跡生成)
""")
    
    # 6. IRL重みと特徴量の次元も確認
    print(f"\n6️⃣ IRL重みと特徴量の次元確認")
    print("-" * 50)
    
    # IRL重みファイル確認
    weights_files = [
        '/Users/kazuki-h/rl/kazoo/data/learned_weights_training.npy',
        '/Users/kazuki-h/rl/kazoo/reward_weights.npy',
        '/Users/kazuki-h/rl/kazoo/data/learned_reward_weights.npy'
    ]
    
    for weights_path in weights_files:
        if Path(weights_path).exists():
            try:
                weights = np.load(weights_path)
                print(f"  📊 {Path(weights_path).name}: {weights.shape[0]}次元")
            except Exception as e:
                print(f"  ❌ {Path(weights_path).name}: 読み込みエラー ({e})")
        else:
            print(f"  ❌ {Path(weights_path).name}: ファイルが見つかりません")

if __name__ == "__main__":
    analyze_rl_irl_consistency()
