#!/usr/bin/env python3
"""Analyze the temporal structure of the simulation and GitHub data."""

import glob
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


def analyze_temporal_structure():
    """Analyze the temporal structure of the simulation."""
    print("=" * 80)
    print("🕰️ 強化学習シミュレーションの時系列分析")
    print("=" * 80)
    
    # 1. Expert trajectories の詳細分析
    print("\n1️⃣ Expert Trajectories の時系列構造")
    print("-" * 50)
    
    try:
        with open('/Users/kazuki-h/rl/kazoo/data/expert_trajectories.pkl', 'rb') as f:
            trajectories = pickle.load(f)
        
        if isinstance(trajectories, list) and len(trajectories) > 0:
            trajectory = trajectories[0]  # 最初の軌跡
            print(f"軌跡数: {len(trajectories)}")
            print(f"第1軌跡のステップ数: {len(trajectory)}")
            
            # 最初の数ステップを詳しく見る
            print(f"\n📋 最初の5ステップの詳細:")
            for i, step in enumerate(trajectory[:5]):
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    timestamp = action_details.get('timestamp', 'Unknown')
                    developer = action_details.get('developer', 'Unknown')
                    action = step.get('action', 'Unknown')
                    task_id = action_details.get('task_id', 'Unknown')
                    
                    print(f"  ステップ {i+1}:")
                    print(f"    時刻: {timestamp}")
                    print(f"    開発者: {developer}")
                    print(f"    行動: {action}")
                    print(f"    タスクID: {task_id}")
            
            # 時系列の範囲を確認
            print(f"\n⏰ 時系列の範囲:")
            timestamps = []
            for step in trajectory[:100]:  # 最初の100ステップをチェック
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    timestamp = action_details.get('timestamp')
                    if timestamp:
                        timestamps.append(timestamp)
            
            if timestamps:
                print(f"  最古のタイムスタンプ: {min(timestamps)}")
                print(f"  最新のタイムスタンプ: {max(timestamps)}")
                print(f"  時間範囲: {len(set([t[:7] for t in timestamps if len(t) >= 7]))} 年月")
            
    except Exception as e:
        print(f"Expert trajectories読み込みエラー: {e}")
    
    # 2. GitHubアーカイブデータの時系列確認
    print(f"\n2️⃣ GitHubアーカイブデータの時系列")
    print("-" * 50)
    
    github_files = glob.glob('/Users/kazuki-h/rl/kazoo/data/gharchive_*.jsonl')
    if github_files:
        print(f"GitHubアーカイブファイル数: {len(github_files)}")
        for file_path in sorted(github_files):
            file_name = Path(file_path).name
            print(f"  {file_name}")
            
            # ファイルから年月を抽出
            if '2020' in file_name or '2021' in file_name or '2022' in file_name:
                year_month = file_name.split('_')[-1].replace('.jsonl', '')
                print(f"    時期: {year_month}")
    
    # 3. Status ディレクトリ内の年別データ確認
    print(f"\n3️⃣ Status ディレクトリの年別データ")
    print("-" * 50)
    
    status_dir = Path('/Users/kazuki-h/rl/kazoo/data/status')
    if status_dir.exists():
        year_dirs = sorted([d for d in status_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        print(f"利用可能な年: {[d.name for d in year_dirs]}")
        
        for year_dir in year_dirs:
            jsonl_files = list(year_dir.glob('*.jsonl'))
            if jsonl_files:
                print(f"  {year_dir.name}年: {len(jsonl_files)} ファイル")
                # 最初のファイルからサンプルデータを読む
                if len(jsonl_files) > 0:
                    sample_file = jsonl_files[0]
                    try:
                        with open(sample_file, 'r') as f:
                            for i, line in enumerate(f):
                                if i >= 1:  # 最初の1行だけ
                                    break
                                try:
                                    event = json.loads(line.strip())
                                    created_at = event.get('created_at', 'Unknown')
                                    event_type = event.get('type', 'Unknown')
                                    print(f"    サンプル: {created_at} - {event_type}")
                                except:
                                    continue
                    except Exception as e:
                        print(f"    ファイル読み込みエラー: {e}")
    
    # 4. バックログの時系列確認
    print(f"\n4️⃣ バックログデータの時系列")
    print("-" * 50)
    
    backlog_files = [
        '/Users/kazuki-h/rl/kazoo/data/backlog.json',
        '/Users/kazuki-h/rl/kazoo/data/backlog_training.json',
        '/Users/kazuki-h/rl/kazoo/data/backlog_test_2022.json'
    ]
    
    for backlog_path in backlog_files:
        if Path(backlog_path).exists():
            try:
                with open(backlog_path, 'r') as f:
                    tasks = json.load(f)
                
                print(f"\n  📋 {Path(backlog_path).name}:")
                print(f"    タスク数: {len(tasks)}")
                
                # 時系列の範囲を確認
                created_dates = []
                for task in tasks[:100]:  # 最初の100タスクをチェック
                    created_at = task.get('created_at')
                    if created_at:
                        created_dates.append(created_at)
                
                if created_dates:
                    print(f"    最古の作成日: {min(created_dates)}")
                    print(f"    最新の作成日: {max(created_dates)}")
                    
                    # 年別の分布
                    years = {}
                    for date in created_dates:
                        year = date[:4] if len(date) >= 4 else 'Unknown'
                        years[year] = years.get(year, 0) + 1
                    
                    print(f"    年別分布 (最初100件): {dict(sorted(years.items()))}")
                    
            except Exception as e:
                print(f"  エラー: {e}")
        else:
            print(f"\n  ❌ {Path(backlog_path).name}: ファイルが見つかりません")
    
    # 5. シミュレーション設定の確認
    print(f"\n5️⃣ シミュレーション時系列設定")
    print("-" * 50)
    
    # 環境設定ファイルを確認
    config_files = [
        '/Users/kazuki-h/rl/kazoo/configs/base.yaml',
        '/Users/kazuki-h/rl/kazoo/configs/base_training.yaml',
        '/Users/kazuki-h/rl/kazoo/configs/base_test_2022.yaml'
    ]
    
    for config_path in config_files:
        if Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"\n  ⚙️ {Path(config_path).name}:")
                
                # 環境設定
                env_config = config.get('env', {})
                sim_config = env_config.get('simulation', {})
                
                print(f"    バックログ: {env_config.get('backlog_path', 'N/A')}")
                print(f"    時間ステップ: {sim_config.get('time_step_hours', 'N/A')}時間")
                print(f"    最大日数: {sim_config.get('max_days', 'N/A')}日")
                print(f"    開発者数: {config.get('num_developers', 'N/A')}人")
                
            except Exception as e:
                print(f"  設定ファイル読み込みエラー: {e}")
        else:
            print(f"  ❌ {Path(config_path).name}: ファイルが見つかりません")
    
    print(f"\n" + "=" * 80)
    print("🔍 時系列構造のサマリー")
    print("=" * 80)
    
    print("""
📊 データの時系列構造:

1️⃣ 【GitHubアーカイブデータ】
   • 期間: 2019-2024年 (data/status/以下に年別保存)
   • 形式: 月別JSONLファイル (gharchive_*.jsonl)
   • 内容: 実際のGitHub OSS開発イベント

2️⃣ 【Expert Trajectories】  
   • 元データ: GitHubアーカイブから生成
   • 軌跡: 時系列順の (状態, 行動) ペア
   • 用途: IRL学習のお手本データ

3️⃣ 【バックログデータ】
   • training: 2019-2021年 (2022年除外)
   • test: 2022年のみ
   • 構造: タスクID、作成日時、ラベル等

4️⃣ 【シミュレーション】
   • 時間進行: 8時間ステップ 
   • 期間: 最大365日 (設定により30-365日)
   • 開始点: バックログ最古日時から

⚡ 重要なポイント:
   • 学習は過去データ(2019-2021)で行い、評価は未来データ(2022)で行う
   • シミュレーション内時間 ≠ 実際の学習時間
   • 強化学習エージェントは歴史的な開発プロセスを再現学習する
""")

if __name__ == "__main__":
    analyze_temporal_structure()
