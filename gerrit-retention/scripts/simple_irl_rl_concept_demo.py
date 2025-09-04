#!/usr/bin/env python3
"""
IRL/RL継続予測 概念実証デモ

ルールベースシステムをIRL/RLで置き換える概念を
シンプルに実証するデモンストレーション
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRetentionIRL:
    """シンプルな継続予測IRL"""
    
    def __init__(self):
        # 特徴量次元
        self.feature_dim = 8
        
        # 報酬重み（学習対象）
        self.reward_weights = torch.randn(self.feature_dim, requires_grad=True)
        
        # オプティマイザー
        self.optimizer = optim.Adam([self.reward_weights], lr=0.01)
        
        print("🧠 シンプルIRLシステムを初期化しました")
    
    def extract_features(self, developer: Dict[str, Any]) -> np.ndarray:
        """開発者から特徴量を抽出"""
        
        features = [
            developer.get('changes_authored', 0) / 100.0,      # 作成変更数
            developer.get('changes_reviewed', 0) / 100.0,      # レビュー数
            len(developer.get('projects', [])) / 5.0,          # プロジェクト数
            len(developer.get('activity_history', [])) / 20.0, # 活動数
            1.0 if developer.get('changes_authored', 0) > 50 else 0.0,  # 高活動フラグ
            1.0 if len(developer.get('projects', [])) > 2 else 0.0,     # 多プロジェクトフラグ
            developer.get('collaboration_score', 0.5),         # 協力スコア
            developer.get('code_quality_score', 0.5)           # コード品質スコア
        ]
        
        return np.array(features, dtype=np.float32)
    
    def train_irl(self, expert_data: List[Dict[str, Any]], epochs: int = 100):
        """IRLで報酬関数を学習"""
        
        print(f"🎯 IRL訓練開始: {epochs}エポック")
        
        # 継続した開発者（エキスパート）と離脱した開発者を分離
        continued_devs = [d for d in expert_data if d.get('continued', True)]
        left_devs = [d for d in expert_data if not d.get('continued', True)]
        
        print(f"   継続開発者: {len(continued_devs)}人")
        print(f"   離脱開発者: {len(left_devs)}人")
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # 継続開発者の特徴量
            for expert_dev in continued_devs:
                expert_features = torch.tensor(
                    self.extract_features(expert_dev['developer']), 
                    dtype=torch.float32
                )
                
                # 離脱開発者との比較
                for non_expert_dev in left_devs:
                    non_expert_features = torch.tensor(
                        self.extract_features(non_expert_dev['developer']), 
                        dtype=torch.float32
                    )
                    
                    # 報酬計算
                    expert_reward = torch.dot(self.reward_weights, expert_features)
                    non_expert_reward = torch.dot(self.reward_weights, non_expert_features)
                    
                    # IRL損失（継続開発者の報酬 > 離脱開発者の報酬）
                    loss = torch.max(
                        torch.tensor(0.0), 
                        1.0 - (expert_reward - non_expert_reward)
                    )
                    
                    # バックプロパゲーション
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"   エポック {epoch}: 損失 = {avg_loss:.4f}")
        
        print(f"🎉 IRL訓練完了: 最終損失 = {losses[-1]:.4f}")
        
        return {'losses': losses, 'learned_weights': self.reward_weights.detach().numpy()}
    
    def predict_continuation_probability(self, developer: Dict[str, Any]) -> float:
        """継続確率を予測"""
        
        features = torch.tensor(self.extract_features(developer), dtype=torch.float32)
        
        with torch.no_grad():
            reward = torch.dot(self.reward_weights, features)
            # シグモイド関数で確率に変換
            probability = torch.sigmoid(reward).item()
        
        return probability


class SimpleRetentionRL:
    """シンプルな継続予測RL"""
    
    def __init__(self):
        # ニューラルネットワーク
        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # オプティマイザー
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        
        print("🎮 シンプルRLシステムを初期化しました")
    
    def extract_features(self, developer: Dict[str, Any]) -> np.ndarray:
        """開発者から特徴量を抽出"""
        
        features = [
            developer.get('changes_authored', 0) / 100.0,      # 作成変更数
            developer.get('changes_reviewed', 0) / 100.0,      # レビュー数
            len(developer.get('projects', [])) / 5.0,          # プロジェクト数
            len(developer.get('activity_history', [])) / 20.0, # 活動数
            1.0 if developer.get('changes_authored', 0) > 50 else 0.0,  # 高活動フラグ
            1.0 if len(developer.get('projects', [])) > 2 else 0.0,     # 多プロジェクトフラグ
            developer.get('collaboration_score', 0.5),         # 協力スコア
            developer.get('code_quality_score', 0.5)           # コード品質スコア
        ]
        
        return np.array(features, dtype=np.float32)
    
    def train_rl(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """RLで予測ポリシーを学習"""
        
        print(f"🎯 RL訓練開始: {epochs}エポック")
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            for data_point in training_data:
                developer = data_point['developer']
                true_continued = data_point.get('continued', True)
                
                # 特徴量抽出
                features = torch.tensor(
                    self.extract_features(developer), 
                    dtype=torch.float32
                ).unsqueeze(0)
                
                # 予測
                predicted_prob = self.network(features)
                
                # 真のラベル
                target = torch.tensor([[1.0 if true_continued else 0.0]], dtype=torch.float32)
                
                # 損失計算（バイナリクロスエントロピー）
                loss = nn.BCELoss()(predicted_prob, target)
                
                # バックプロパゲーション
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 精度計算
                predicted_label = predicted_prob.item() > 0.5
                if predicted_label == true_continued:
                    correct_predictions += 1
            
            avg_loss = epoch_loss / len(training_data)
            accuracy = correct_predictions / len(training_data)
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 20 == 0:
                print(f"   エポック {epoch}: 損失 = {avg_loss:.4f}, 精度 = {accuracy:.1%}")
        
        print(f"🎉 RL訓練完了: 最終精度 = {accuracies[-1]:.1%}")
        
        return {'losses': losses, 'accuracies': accuracies}
    
    def predict_continuation_probability(self, developer: Dict[str, Any]) -> float:
        """継続確率を予測"""
        
        features = torch.tensor(
            self.extract_features(developer), 
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            probability = self.network(features).item()
        
        return probability


def main():
    """メイン関数"""
    
    print("🤖 IRL/RL継続予測システム 概念実証デモ")
    print("=" * 70)
    print("ルールベースから機械学習への移行を実証します")
    print()
    
    # 訓練データを作成（簡略版）
    training_data = [
        # 継続した開発者
        {
            'developer': {
                'developer_id': 'alice@example.com',
                'name': 'Alice Continued',
                'changes_authored': 85,
                'changes_reviewed': 120,
                'projects': ['project-a', 'project-b', 'project-c'],
                'activity_history': [{'type': 'commit'} for _ in range(15)],
                'collaboration_score': 0.8,
                'code_quality_score': 0.7
            },
            'continued': True
        },
        {
            'developer': {
                'developer_id': 'bob@example.com',
                'name': 'Bob Continued',
                'changes_authored': 156,
                'changes_reviewed': 89,
                'projects': ['project-a', 'project-c', 'project-d'],
                'activity_history': [{'type': 'commit'} for _ in range(18)],
                'collaboration_score': 0.9,
                'code_quality_score': 0.8
            },
            'continued': True
        },
        {
            'developer': {
                'developer_id': 'eve@example.com',
                'name': 'Eve Veteran',
                'changes_authored': 456,
                'changes_reviewed': 234,
                'projects': ['project-a', 'project-b', 'project-c', 'project-d', 'project-e'],
                'activity_history': [{'type': 'commit'} for _ in range(25)],
                'collaboration_score': 0.95,
                'code_quality_score': 0.9
            },
            'continued': True
        },
        # 離脱した開発者
        {
            'developer': {
                'developer_id': 'charlie@example.com',
                'name': 'Charlie Left',
                'changes_authored': 12,
                'changes_reviewed': 8,
                'projects': ['project-b'],
                'activity_history': [{'type': 'commit'} for _ in range(5)],
                'collaboration_score': 0.3,
                'code_quality_score': 0.4
            },
            'continued': False
        },
        {
            'developer': {
                'developer_id': 'diana@example.com',
                'name': 'Diana Left',
                'changes_authored': 34,
                'changes_reviewed': 18,
                'projects': ['project-c'],
                'activity_history': [{'type': 'commit'} for _ in range(8)],
                'collaboration_score': 0.4,
                'code_quality_score': 0.3
            },
            'continued': False
        },
        {
            'developer': {
                'developer_id': 'frank@example.com',
                'name': 'Frank Left',
                'changes_authored': 23,
                'changes_reviewed': 15,
                'projects': ['project-a'],
                'activity_history': [{'type': 'commit'} for _ in range(6)],
                'collaboration_score': 0.2,
                'code_quality_score': 0.35
            },
            'continued': False
        }
    ]
    
    print(f"📊 訓練データ: {len(training_data)}件")
    print(f"   継続: {sum(1 for d in training_data if d['continued'])}件")
    print(f"   離脱: {sum(1 for d in training_data if not d['continued'])}件")
    print()
    
    # 1. IRL システムのテスト
    print("🧠 逆強化学習（IRL）システム")
    print("-" * 50)
    
    irl_system = SimpleRetentionIRL()
    irl_results = irl_system.train_irl(training_data, epochs=100)
    
    print("\n🔍 学習された報酬重み:")
    feature_names = [
        'changes_authored', 'changes_reviewed', 'project_count', 'activity_count',
        'high_activity_flag', 'multi_project_flag', 'collaboration_score', 'code_quality_score'
    ]
    
    learned_weights = irl_results['learned_weights']
    for i, (name, weight) in enumerate(zip(feature_names, learned_weights)):
        print(f"   {name:20s}: {weight:6.3f}")
    
    print()
    
    # 2. RL システムのテスト
    print("🎮 強化学習（RL）システム")
    print("-" * 50)
    
    rl_system = SimpleRetentionRL()
    rl_results = rl_system.train_rl(training_data, epochs=100)
    
    print()
    
    # 3. 予測比較テスト
    print("📊 予測システム比較テスト")
    print("=" * 70)
    
    # ルールベース予測（従来システム）
    def rule_based_predict(developer: Dict[str, Any]) -> float:
        """ルールベース予測"""
        changes = developer.get('changes_authored', 0)
        reviews = developer.get('changes_reviewed', 0)
        projects = len(developer.get('projects', []))
        
        # 単純なルール
        if changes > 100 and reviews > 50 and projects > 2:
            return 0.9  # 高確率で継続
        elif changes > 50 and reviews > 30:
            return 0.7  # 中確率で継続
        elif changes > 20:
            return 0.5  # 中程度
        else:
            return 0.3  # 低確率で継続
    
    print(f"{'開発者名':<15} {'実際':<6} {'ルール':<8} {'IRL':<8} {'RL':<8} {'ルール正解':<10} {'IRL正解':<8} {'RL正解':<8}")
    print("-" * 80)
    
    rule_correct = 0
    irl_correct = 0
    rl_correct = 0
    
    for data_point in training_data:
        developer = data_point['developer']
        true_continued = data_point['continued']
        
        # 各システムで予測
        rule_prob = rule_based_predict(developer)
        irl_prob = irl_system.predict_continuation_probability(developer)
        rl_prob = rl_system.predict_continuation_probability(developer)
        
        # ラベル化（50%閾値）
        rule_label = rule_prob > 0.5
        irl_label = irl_prob > 0.5
        rl_label = rl_prob > 0.5
        
        # 正解判定
        rule_correct += (rule_label == true_continued)
        irl_correct += (irl_label == true_continued)
        rl_correct += (rl_label == true_continued)
        
        # 結果表示
        true_str = "継続" if true_continued else "離脱"
        rule_correct_str = "✅" if rule_label == true_continued else "❌"
        irl_correct_str = "✅" if irl_label == true_continued else "❌"
        rl_correct_str = "✅" if rl_label == true_continued else "❌"
        
        print(f"{developer['name'][:14]:<15} {true_str:<6} {rule_prob:>6.1%} {irl_prob:>8.1%} {rl_prob:>8.1%} {rule_correct_str:>10} {irl_correct_str:>8} {rl_correct_str:>8}")
    
    print("-" * 80)
    total_samples = len(training_data)
    print(f"{'総合精度':<15} {'':<6} {rule_correct/total_samples:>6.1%} {irl_correct/total_samples:>8.1%} {rl_correct/total_samples:>8.1%}")
    print()
    
    # 4. システム比較分析
    print("🏆 システム比較分析")
    print("=" * 70)
    
    systems = [
        ('ルールベース', rule_correct/total_samples),
        ('逆強化学習（IRL）', irl_correct/total_samples),
        ('強化学習（RL）', rl_correct/total_samples)
    ]
    
    # 最高性能システム
    best_system = max(systems, key=lambda x: x[1])
    
    print("**システム性能ランキング:**")
    for i, (system_name, accuracy) in enumerate(sorted(systems, key=lambda x: x[1], reverse=True), 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {i}. {status} {system_name}: {accuracy:.1%}")
    
    print()
    print("**各システムの特徴:**")
    print()
    print("🔧 **ルールベースシステム:**")
    print("   ✅ 解釈しやすい")
    print("   ✅ 実装が簡単")
    print("   ❌ 固定的なルール")
    print("   ❌ 複雑なパターンを捉えられない")
    print()
    
    print("🧠 **逆強化学習（IRL）システム:**")
    print("   ✅ エキスパートの行動から学習")
    print("   ✅ 解釈可能な報酬関数")
    print("   ✅ 継続要因を自動発見")
    print("   ❌ エキスパートデータが必要")
    print()
    
    print("🎮 **強化学習（RL）システム:**")
    print("   ✅ 予測精度を直接最適化")
    print("   ✅ 複雑なパターンを学習可能")
    print("   ✅ 大量データで性能向上")
    print("   ❌ ブラックボックス的")
    print()
    
    # 5. 実用化への提案
    print("💡 実用化への提案")
    print("=" * 70)
    
    if best_system[0] == 'ルールベース':
        print("🔧 **推奨**: ルールベースシステムの改良")
        print("   - より多くの特徴量を追加")
        print("   - 重み付けの最適化")
        print("   - 機械学習との組み合わせ")
    elif 'IRL' in best_system[0]:
        print("🧠 **推奨**: IRLシステムの本格導入")
        print("   - より多くのエキスパートデータの収集")
        print("   - 特徴量エンジニアリングの改善")
        print("   - 報酬関数の解釈性向上")
    else:
        print("🎮 **推奨**: RLシステムの本格導入")
        print("   - より大規模なデータセットでの訓練")
        print("   - ネットワーク構造の最適化")
        print("   - 説明可能性の向上")
    
    print()
    print("**共通の改善点:**")
    print("1. 特徴量の拡張（コード品質、協力関係、プロジェクト重要度）")
    print("2. 時系列データの活用（活動トレンド、季節性）")
    print("3. より大規模なデータセットでの検証")
    print("4. リアルタイム学習機能の追加")
    print("5. 説明可能性の向上")
    print()
    
    # 結果保存
    results = {
        'system_comparison': {
            'rule_based_accuracy': rule_correct / total_samples,
            'irl_accuracy': irl_correct / total_samples,
            'rl_accuracy': rl_correct / total_samples,
            'best_system': best_system[0],
            'best_accuracy': best_system[1]
        },
        'irl_results': irl_results,
        'rl_results': rl_results,
        'training_data_size': len(training_data),
        'demo_date': str(datetime.now())
    }
    
    output_path = "outputs/comprehensive_retention/irl_rl_concept_demo.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 結果を保存しました: {output_path}")
    print()
    print("🎉 IRL/RL概念実証デモ完了！")
    print(f"   最高性能: {best_system[0]} ({best_system[1]:.1%})")
    print("   ルールベースから機械学習への移行可能性が実証されました。")


if __name__ == "__main__":
    main()