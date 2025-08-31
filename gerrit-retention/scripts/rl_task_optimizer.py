#!/usr/bin/env python3
"""
強化学習タスク最適化システム
訓練済みモデルを使用した実際のタスク割り当て最適化
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import sys

from gerrit_retention.utils.logger import get_logger

sys.path.append('scripts')
from advanced_rl_system import AdvancedActorCritic

logger = get_logger(__name__)

@dataclass
class TaskAssignmentRecommendation:
    """タスク割り当て推奨結果"""
    developer_id: str
    developer_name: str
    task_id: str
    action: str  # 'assign', 'reject', 'defer'
    confidence: float
    reasoning: List[str]
    action_probabilities: Dict[str, float]
    expected_value: float

class RLTaskOptimizer:
    """強化学習ベースのタスク最適化器"""
    
    def __init__(self, model_path: str = 'models/advanced_ppo_agent.pth'):
        self.device = torch.device('cpu')
        self.model_path = model_path
        
        # モデルの読み込み
        self.actor_critic = AdvancedActorCritic(obs_dim=20, action_dim=3)
        self._load_model()
        
        # 行動マッピング
        self.action_names = ['assign', 'reject', 'defer']
        self.action_descriptions = {
            'assign': '割り当て推奨',
            'reject': '割り当て非推奨',
            'defer': '後で検討'
        }
        
        logger.info("強化学習タスク最適化器を初期化")
    
    def _load_model(self):
        """訓練済みモデルの読み込み"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.actor_critic.eval()
            logger.info(f"訓練済みモデルを読み込み: {self.model_path}")
        except Exception as e:
            logger.warning(f"モデル読み込み失敗: {e}. ランダム初期化を使用")
    
    def _build_observation(self, developer: Dict, task: Dict, context: Dict = None) -> np.ndarray:
        """観測ベクトルの構築"""
        obs = np.zeros(20, dtype=np.float32)
        
        if context is None:
            context = {}
        
        # 開発者特徴量 (0-9)
        obs[0] = np.tanh((developer['changes_authored'] + developer['changes_reviewed']) / 100.0)
        obs[1] = np.tanh(developer['changes_authored'] / 50.0)
        obs[2] = np.tanh(developer['changes_reviewed'] / 100.0)
        obs[3] = np.tanh(len(developer['projects']) / 10.0)
        obs[4] = context.get('stress_level', np.random.normal(0, 0.3))
        obs[5] = context.get('workload', np.random.uniform(-0.5, 0.5))
        obs[6] = context.get('expertise_match', np.random.uniform(-0.3, 0.7))
        obs[7] = context.get('availability', np.random.uniform(0.2, 1.0))
        obs[8] = context.get('collaboration_score', np.random.uniform(-0.2, 0.8))
        obs[9] = context.get('recent_performance', np.random.uniform(-0.3, 0.7))
        
        # タスク特徴量 (10-14)
        obs[10] = np.tanh(task.get('lines_added', 0) / 500.0)
        obs[11] = np.tanh(task.get('files_changed', 0) / 10.0)
        obs[12] = 1.0 if task.get('status') == 'NEW' else -1.0
        obs[13] = np.tanh(task.get('score', 0) / 2.0)
        obs[14] = context.get('task_priority', np.random.uniform(-0.5, 0.5))
        
        # 環境状態 (15-19)
        obs[15] = context.get('team_workload', np.random.uniform(-0.3, 0.3))
        obs[16] = context.get('deadline_pressure', np.random.uniform(-0.5, 0.5))
        obs[17] = context.get('team_stress', np.random.uniform(-0.3, 0.3))
        obs[18] = context.get('sprint_progress', np.random.uniform(-0.2, 0.8))
        obs[19] = context.get('resource_availability', np.random.uniform(0.0, 1.0))
        
        return obs
    
    def get_recommendation(
        self, 
        developer: Dict, 
        task: Dict, 
        context: Dict = None
    ) -> TaskAssignmentRecommendation:
        """単一のタスク割り当て推奨を取得"""
        
        # 観測ベクトルの構築
        obs = self._build_observation(developer, task, context)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # モデル推論
        with torch.no_grad():
            action_logits, value = self.actor_critic(obs_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            
            # 最適行動の選択
            best_action_idx = torch.argmax(probs, dim=-1).item()
            best_action = self.action_names[best_action_idx]
            confidence = probs[0, best_action_idx].item()
            
            # 行動確率の辞書
            action_probs = {
                action: probs[0, i].item() 
                for i, action in enumerate(self.action_names)
            }
        
        # 推奨理由の生成
        reasoning = self._generate_reasoning(
            developer, task, best_action, confidence, obs, context
        )
        
        return TaskAssignmentRecommendation(
            developer_id=developer['developer_id'],
            developer_name=developer['name'],
            task_id=task.get('change_id', 'unknown'),
            action=best_action,
            confidence=confidence,
            reasoning=reasoning,
            action_probabilities=action_probs,
            expected_value=value.item()
        )
    
    def _generate_reasoning(
        self, 
        developer: Dict, 
        task: Dict, 
        action: str, 
        confidence: float, 
        obs: np.ndarray,
        context: Dict = None
    ) -> List[str]:
        """推奨理由の生成"""
        reasons = []
        
        # 開発者の活動レベル
        total_activity = developer['changes_authored'] + developer['changes_reviewed']
        
        # タスクの複雑さ
        task_complexity = task.get('lines_added', 0) + task.get('files_changed', 0) * 10
        
        if action == 'assign':
            reasons.append(f"高い信頼度 ({confidence:.1%}) で割り当てを推奨")
            
            if total_activity > 50:
                reasons.append(f"豊富な経験 (総活動: {total_activity}件)")
            
            if task_complexity < 200:
                reasons.append("適度なサイズのタスク")
            
            if obs[4] < 0:  # ストレスレベル
                reasons.append("開発者のストレスレベルが適正")
            
            if obs[6] > 0.3:  # 専門性マッチ
                reasons.append("専門性が適合")
            
            if len(developer['projects']) > 0:
                reasons.append("関連プロジェクトでの経験あり")
        
        elif action == 'reject':
            reasons.append(f"高い信頼度 ({confidence:.1%}) で割り当てを非推奨")
            
            if total_activity < 10:
                reasons.append("開発者の経験が不足")
            
            if task_complexity > 500:
                reasons.append("タスクが複雑すぎる")
            
            if obs[4] > 0.5:  # 高ストレス
                reasons.append("開発者のストレスレベルが高い")
            
            if obs[5] > 0.5:  # 高負荷
                reasons.append("現在の作業負荷が高い")
        
        else:  # defer
            reasons.append("より詳細な検討が必要")
            
            if 0.3 < confidence < 0.7:
                reasons.append("判断が困難なケース")
            
            if obs[16] > 0.5:  # 締切プレッシャー
                reasons.append("締切を考慮した慎重な判断")
        
        # 最低1つの理由を保証
        if not reasons:
            reasons.append(f"AI判断による {self.action_descriptions[action]}")
        
        return reasons
    
    def optimize_team_assignments(
        self, 
        developers: List[Dict], 
        tasks: List[Dict],
        max_assignments_per_developer: int = 3
    ) -> Dict:
        """チーム全体のタスク割り当て最適化"""
        
        logger.info(f"チーム最適化開始: {len(developers)}名の開発者, {len(tasks)}件のタスク")
        
        # 全ての組み合わせで推奨を取得
        all_recommendations = []
        
        for task in tasks:
            task_recommendations = []
            
            for developer in developers:
                # コンテキスト情報の設定
                context = {
                    'stress_level': np.random.normal(0, 0.3),
                    'workload': np.random.uniform(-0.5, 0.5),
                    'expertise_match': np.random.uniform(-0.3, 0.7),
                    'availability': np.random.uniform(0.2, 1.0),
                    'task_priority': np.random.uniform(-0.5, 0.5)
                }
                
                recommendation = self.get_recommendation(developer, task, context)
                task_recommendations.append(recommendation)
            
            # タスクごとに最適な割り当てを選択
            assign_recommendations = [
                rec for rec in task_recommendations 
                if rec.action == 'assign'
            ]
            
            if assign_recommendations:
                # 信頼度が最も高い推奨を選択
                best_recommendation = max(
                    assign_recommendations, 
                    key=lambda x: x.confidence
                )
                all_recommendations.append(best_recommendation)
        
        # 開発者ごとの割り当て数を制限
        developer_assignments = {}
        final_assignments = []
        
        # 信頼度順にソート
        sorted_recommendations = sorted(
            all_recommendations, 
            key=lambda x: x.confidence, 
            reverse=True
        )
        
        for rec in sorted_recommendations:
            dev_id = rec.developer_id
            current_count = developer_assignments.get(dev_id, 0)
            
            if current_count < max_assignments_per_developer:
                final_assignments.append(rec)
                developer_assignments[dev_id] = current_count + 1
        
        # 統計情報の計算
        total_tasks = len(tasks)
        assigned_tasks = len(final_assignments)
        assignment_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0
        
        avg_confidence = np.mean([rec.confidence for rec in final_assignments]) if final_assignments else 0
        
        # 開発者別統計
        developer_stats = {}
        for dev_id, count in developer_assignments.items():
            developer = next(d for d in developers if d['developer_id'] == dev_id)
            developer_stats[dev_id] = {
                'name': developer['name'],
                'assignments': count,
                'avg_confidence': np.mean([
                    rec.confidence for rec in final_assignments 
                    if rec.developer_id == dev_id
                ])
            }
        
        optimization_result = {
            'assignments': final_assignments,
            'statistics': {
                'total_tasks': total_tasks,
                'assigned_tasks': assigned_tasks,
                'assignment_rate': assignment_rate,
                'average_confidence': avg_confidence,
                'developer_stats': developer_stats
            }
        }
        
        logger.info(f"最適化完了: {assigned_tasks}/{total_tasks}件のタスクを割り当て (成功率: {assignment_rate:.1%})")
        
        return optimization_result


def demonstrate_rl_optimization():
    """強化学習最適化のデモンストレーション"""
    logger.info("=== 強化学習タスク最適化デモ ===")
    
    # データの読み込み
    with open('data/processed/unified/all_developers.json', 'r') as f:
        developers_data = json.load(f)
    
    with open('data/processed/unified/all_reviews.json', 'r') as f:
        reviews_data = json.load(f)
    
    # 人間の開発者のみを抽出
    human_developers = []
    for dev in developers_data:
        name_lower = dev['name'].lower()
        email_lower = dev['developer_id'].lower()
        
        is_bot = any(keyword in name_lower for keyword in ['bot', 'robot', 'lint', 'presubmit', 'treehugger'])
        is_bot = is_bot or any(keyword in email_lower for keyword in ['bot', 'robot', 'system.gserviceaccount', 'presubmit'])
        
        if not is_bot and (dev['changes_authored'] > 0 or dev['changes_reviewed'] > 5):
            human_developers.append(dev)
    
    # 上位開発者とタスクを選択
    top_developers = sorted(
        human_developers, 
        key=lambda x: x['changes_authored'] + x['changes_reviewed'], 
        reverse=True
    )[:8]
    
    interesting_tasks = [
        task for task in reviews_data 
        if task.get('lines_added', 0) > 0  # より緩い条件
    ][:6]
    
    # タスクが見つからない場合は全てのタスクを使用
    if not interesting_tasks:
        interesting_tasks = reviews_data[:6]
    
    # 最適化器の初期化
    optimizer = RLTaskOptimizer()
    
    # チーム最適化の実行
    optimization_result = optimizer.optimize_team_assignments(
        top_developers, interesting_tasks, max_assignments_per_developer=2
    )
    
    # 結果の表示
    print("\\n🤖 強化学習による最適タスク割り当て結果")
    print("=" * 60)
    
    assignments = optimization_result['assignments']
    stats = optimization_result['statistics']
    
    print(f"\\n📊 全体統計:")
    print(f"   総タスク数: {stats['total_tasks']}件")
    print(f"   割り当て済み: {stats['assigned_tasks']}件")
    print(f"   割り当て率: {stats['assignment_rate']:.1%}")
    print(f"   平均信頼度: {stats['average_confidence']:.1%}")
    
    print(f"\\n👥 開発者別割り当て:")
    for dev_id, dev_stats in stats['developer_stats'].items():
        print(f"   {dev_stats['name']}: {dev_stats['assignments']}件 (信頼度: {dev_stats['avg_confidence']:.1%})")
    
    print(f"\\n📋 詳細な割り当て推奨:")
    for i, assignment in enumerate(assignments, 1):
        print(f"\\n   {i}. タスク {assignment.task_id}")
        print(f"      → {assignment.developer_name} ({assignment.developer_id})")
        print(f"      推奨: {assignment.action} (信頼度: {assignment.confidence:.1%})")
        print(f"      期待価値: {assignment.expected_value:.2f}")
        print(f"      理由:")
        for reason in assignment.reasoning:
            print(f"        • {reason}")
        
        # 行動確率の詳細
        probs = assignment.action_probabilities
        print(f"      確率分布: 割当{probs['assign']:.1%}, 拒否{probs['reject']:.1%}, 延期{probs['defer']:.1%}")
    
    print("\\n" + "=" * 60)
    logger.info("強化学習最適化デモ完了")


if __name__ == "__main__":
    demonstrate_rl_optimization()