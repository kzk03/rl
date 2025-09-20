"""
レビュー受諾強化学習環境

このモジュールは、開発者のレビュー受諾行動を最適化するための強化学習環境を提供する。
Gymnasium インターフェースに準拠し、20次元の状態空間と3つの行動（受諾/拒否/待機）を実装。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..behavior_analysis.review_behavior import ReviewBehaviorAnalyzer
from ..behavior_analysis.similarity_calculator import SimilarityCalculator
from ..prediction.stress_analyzer import StressAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewRequest:
    """レビュー依頼データクラス"""
    change_id: str
    author_email: str
    project: str
    branch: str
    subject: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    complexity_score: float
    technical_domain: str
    urgency_level: float
    estimated_review_effort: float
    required_expertise: List[str]
    created_at: datetime
    deadline: Optional[datetime] = None
    expertise_match: float = 0.5
    requester_relationship: float = 0.5


@dataclass
class DeveloperState:
    """開発者状態データクラス"""
    developer_email: str
    name: str
    expertise_level: float
    stress_level: float
    activity_pattern: Dict[str, float]
    recent_review_acceptance_rate: float
    workload_ratio: float
    collaboration_quality: float
    learning_velocity: float
    satisfaction_level: float
    boiling_point_estimate: float
    avg_review_score_given: float
    avg_review_score_received: float
    review_response_time_avg: float
    code_review_thoroughness: float
    last_updated: datetime


class ReviewAcceptanceEnvironment(gym.Env):
    """
    レビュー受諾強化学習環境
    
    開発者のレビュー受諾行動を最適化するための強化学習環境。
    長期的な開発者定着を重視した報酬設計を採用。
    """
    
    # 行動定数
    ACTION_REJECT = 0  # 拒否
    ACTION_ACCEPT = 1  # 受諾
    ACTION_WAIT = 2    # 待機
    
    def __init__(self, config: Dict[str, Any]):
        """
        環境を初期化
        
        Args:
            config: 環境設定辞書
        """
        super().__init__()
        self.config = config

        # 状態空間の定義（20次元）
        self.observation_space = self._define_observation_space()

        # 行動空間の定義（受諾/拒否/待機の3つ）
        self.action_space = spaces.Discrete(3)

        # 環境パラメータ
        self.max_episode_length = config.get('max_episode_length', 100)
        self.max_queue_size = config.get('max_queue_size', 10)
        self.stress_threshold = config.get('stress_threshold', 0.8)
        # ランダム生成制御フラグ
        self.use_random_initial_queue = config.get('use_random_initial_queue', True)
        self.enable_random_new_reviews = config.get('enable_random_new_reviews', True)

        # 分析器の初期化
        self.stress_analyzer = StressAnalyzer(config.get('stress_config', {}))
        self.behavior_analyzer = ReviewBehaviorAnalyzer(config.get('behavior_config', {}))
        self.similarity_calculator = SimilarityCalculator(config.get('similarity_config', {}))

        # 環境状態の初期化
        self.reset()

        logger.info("レビュー受諾環境を初期化しました")
    
    def _define_observation_space(self) -> spaces.Box:
        """
        20次元の状態空間を定義
        
        状態空間の構成:
        [0-4]: 開発者状態 (専門性、ストレス、ワークロード、満足度、沸点余裕)
        [5-9]: レビュー特徴 (複雑度、規模、緊急度、専門性適合度、関係性)
        [10-14]: 時間的特徴 (時刻、曜日、プロジェクトフェーズ、締切余裕、応答履歴)
        [15-19]: 協力関係特徴 (協力品質、ネットワーク密度、過去成功率、学習機会、負荷分散)
        
        Returns:
            gymnasium.spaces.Box: 20次元の連続状態空間
        """
        # すべての特徴量を0-1の範囲に正規化
        low = np.zeros(20, dtype=np.float32)
        high = np.ones(20, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        環境をリセット
        
        Args:
            seed: ランダムシード
            options: リセットオプション
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 初期観測と情報辞書
        """
        super().reset(seed=seed)
        
        # 環境状態の初期化
        self.current_step = 0
        self.developer_state = self._initialize_developer_state()
        # 初期レビューキュー
        if self.use_random_initial_queue:
            self.review_queue = self._initialize_review_queue()
        else:
            self.review_queue = []
        self.stress_accumulator = 0.0
        self.acceptance_history = []
        self.episode_rewards = []
        self.total_reviews_processed = 0
        self.total_reviews_accepted = 0
        
        # 初期観測を取得
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"環境をリセットしました (ステップ: {self.current_step})")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        1ステップ実行
        
        Args:
            action: 実行する行動 (0: 拒否, 1: 受諾, 2: 待機)
            
        Returns:
            Tuple: (次の観測, 報酬, 終了フラグ, 切り詰めフラグ, 情報辞書)
        """
        # 行動の検証
        if action not in [self.ACTION_REJECT, self.ACTION_ACCEPT, self.ACTION_WAIT]:
            raise ValueError(f"無効な行動: {action}")
        
        # 行動を実行して報酬を計算
        reward = self._execute_action(action)
        self.episode_rewards.append(reward)
        
        # 状態を更新
        self._update_state(action)
        
        # 終了条件をチェック
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # 次の観測を取得
        next_observation = self._get_observation()
        
        # 情報を収集
        info = self._get_info()
        
        self.current_step += 1
        
        logger.debug(f"ステップ {self.current_step}: 行動={action}, 報酬={reward:.3f}")
        
        return next_observation, reward, terminated, truncated, info
    
    def _initialize_developer_state(self) -> DeveloperState:
        """
        開発者状態を初期化
        
        Returns:
            DeveloperState: 初期化された開発者状態
        """
        return DeveloperState(
            developer_email="developer@example.com",
            name="Test Developer",
            expertise_level=np.random.uniform(0.3, 0.9),
            stress_level=np.random.uniform(0.1, 0.5),
            activity_pattern={"morning": 0.7, "afternoon": 0.8, "evening": 0.3},
            recent_review_acceptance_rate=np.random.uniform(0.5, 0.9),
            workload_ratio=np.random.uniform(0.2, 0.8),
            collaboration_quality=np.random.uniform(0.4, 0.9),
            learning_velocity=np.random.uniform(0.3, 0.8),
            satisfaction_level=np.random.uniform(0.4, 0.9),
            boiling_point_estimate=np.random.uniform(0.7, 0.95),
            avg_review_score_given=np.random.uniform(0.6, 0.9),
            avg_review_score_received=np.random.uniform(0.6, 0.9),
            review_response_time_avg=np.random.uniform(2.0, 24.0),
            code_review_thoroughness=np.random.uniform(0.5, 0.9),
            last_updated=datetime.now()
        )
    
    def _initialize_review_queue(self) -> List[ReviewRequest]:
        """
        レビューキューを初期化
        
        Returns:
            List[ReviewRequest]: 初期化されたレビューキュー
        """
        queue = []
        initial_queue_size = np.random.randint(1, min(5, self.max_queue_size + 1))
        
        for i in range(initial_queue_size):
            review = ReviewRequest(
                change_id=f"change_{i}_{np.random.randint(1000, 9999)}",
                author_email=f"author_{i}@example.com",
                project="test-project",
                branch="main",
                subject=f"Test change {i}",
                files_changed=np.random.randint(1, 20),
                lines_added=np.random.randint(10, 500),
                lines_deleted=np.random.randint(0, 200),
                complexity_score=np.random.uniform(0.1, 1.0),
                technical_domain=np.random.choice(["backend", "frontend", "database", "api"]),
                urgency_level=np.random.uniform(0.1, 1.0),
                estimated_review_effort=np.random.uniform(0.5, 4.0),
                required_expertise=["python", "testing"],
                created_at=datetime.now() - timedelta(hours=np.random.randint(1, 48)),
                deadline=datetime.now() + timedelta(hours=np.random.randint(24, 168)),
                expertise_match=np.random.uniform(0.2, 1.0),
                requester_relationship=np.random.uniform(0.0, 1.0)
            )
            queue.append(review)
        
        return queue
    
    def _get_observation(self) -> np.ndarray:
        """
        現在の状態から観測ベクトルを生成
        
        Returns:
            np.ndarray: 20次元の観測ベクトル
        """
        obs = np.zeros(20, dtype=np.float32)
        
        # [0-4]: 開発者状態
        obs[0] = self.developer_state.expertise_level
        obs[1] = self.developer_state.stress_level
        obs[2] = self.developer_state.workload_ratio
        obs[3] = self.developer_state.satisfaction_level
        obs[4] = max(0.0, self.developer_state.boiling_point_estimate - self.developer_state.stress_level)
        
        # [5-9]: レビュー特徴（キューの先頭レビュー、なければ0）
        if self.review_queue:
            current_review = self.review_queue[0]
            obs[5] = current_review.complexity_score
            obs[6] = min(1.0, (current_review.lines_added + current_review.lines_deleted) / 1000.0)
            obs[7] = current_review.urgency_level
            obs[8] = current_review.expertise_match
            obs[9] = current_review.requester_relationship
        
        # [10-14]: 時間的特徴
        current_time = datetime.now()
        obs[10] = current_time.hour / 24.0  # 時刻正規化
        obs[11] = current_time.weekday() / 6.0  # 曜日正規化
        obs[12] = min(1.0, self.current_step / self.max_episode_length)  # エピソード進行度
        
        if self.review_queue and self.review_queue[0].deadline:
            time_to_deadline = (self.review_queue[0].deadline - current_time).total_seconds() / (24 * 3600)
            obs[13] = max(0.0, min(1.0, time_to_deadline / 7.0))  # 締切余裕（週単位）
        
        # 最近の応答履歴
        recent_acceptance_rate = self._calculate_recent_acceptance_rate()
        obs[14] = recent_acceptance_rate
        
        # [15-19]: 協力関係特徴
        obs[15] = self.developer_state.collaboration_quality
        obs[16] = min(1.0, len(self.review_queue) / self.max_queue_size)  # キュー充填率
        obs[17] = self.developer_state.recent_review_acceptance_rate
        obs[18] = self.developer_state.learning_velocity
        obs[19] = self._calculate_load_balance()
        
        return obs
    
    def _execute_action(self, action: int) -> float:
        """
        行動を実行して報酬を計算
        
        Args:
            action: 実行する行動
            
        Returns:
            float: 計算された報酬
        """
        if not self.review_queue:
            # レビューがない場合は小さな負の報酬
            return -0.1

        current_review = self.review_queue[0]
        self.total_reviews_processed += 1

        if action == self.ACTION_REJECT:  # 拒否
            reward = self._calculate_rejection_reward(current_review)
            self.review_queue.pop(0)
            self.acceptance_history.append({
                'action': 'reject',
                'review': current_review,
                'timestamp': datetime.now(),
                'step': self.current_step,
                'reviewer': self.developer_state.developer_email,
                'reward': reward
            })

        elif action == self.ACTION_ACCEPT:  # 受諾
            reward = self._calculate_acceptance_reward(current_review)
            self.review_queue.pop(0)
            self.acceptance_history.append({
                'action': 'accept',
                'review': current_review,
                'timestamp': datetime.now(),
                'step': self.current_step,
                'reviewer': self.developer_state.developer_email,
                'reward': reward
            })
            self.total_reviews_accepted += 1
            self._update_developer_state_after_acceptance(current_review)

        else:  # 待機
            reward = self._calculate_waiting_reward(current_review)
            # レビューはキューに残る
            self.acceptance_history.append({
                'action': 'wait',
                'review': current_review,
                'timestamp': datetime.now(),
                'step': self.current_step,
                'reviewer': self.developer_state.developer_email,
                'reward': reward
            })

        # 新しいレビューをキューに追加（確率的）
        self._maybe_add_new_review()

        return reward
    
    def _calculate_acceptance_reward(self, review: ReviewRequest) -> float:
        """
        受諾時の報酬を計算
        
        Args:
            review: レビュー依頼
            
        Returns:
            float: 計算された報酬
        """
        base_reward = getattr(self, 'accept_base_reward', 0.8)  # 基本受諾報酬（下げる）

        # 継続報酬（reject を挟んでも直近受諾があれば継続とみなす）
        continuity_reward = 0.0
        weight = float(getattr(self, 'continuity_weight', 0.3))
        mode = getattr(self, 'continuity_mode', 'decay')
        # 継続は「同一レビュアー（本環境の開発者）によるレビュー継続」が定義
        target_reviewer = getattr(self.developer_state, 'developer_email', None)
        if weight > 0.0:
            if mode == 'decay':
                last_accept_step = None
                for h in reversed(self.acceptance_history):
                    if h.get('action') == 'accept':
                        # 同じレビュアーに限定
                        if target_reviewer is not None and h.get('reviewer') != target_reviewer:
                            continue
                        last_accept_step = h.get('step')
                        break
                if last_accept_step is not None and isinstance(last_accept_step, int):
                    tau = float(getattr(self, 'continuity_tau', 2.0))
                    delta = max(1, self.current_step - last_accept_step)
                    continuity_reward = weight * float(np.exp(-delta / max(1e-6, tau)))
            else:
                window = int(getattr(self, 'continuity_window', 5))
                recent = self.acceptance_history[-window:]
                k = 0
                for h in recent:
                    if h.get('action') != 'accept':
                        continue
                    if target_reviewer is not None and h.get('reviewer') != target_reviewer:
                        continue
                    k += 1
                continuity_reward = weight * float(1.0 - np.exp(-float(k)))
        
        # ストレス報酬
        expertise_match = review.expertise_match
        if expertise_match > 0.7:
            stress_reward = 0.2  # 専門性に合う場合はストレス軽減
        else:
            stress_reward = -0.4  # 専門性に合わない場合はストレス増加
            # 開発者のストレスレベルを更新
            self.developer_state.stress_level = min(1.0, self.developer_state.stress_level + 0.1)
        
        # 品質報酬（予測）
        expected_quality = self._predict_review_quality(review)
        quality_reward = 0.1 * expected_quality
        
        # 協力報酬
        if review.requester_relationship < 0.3:
            collaboration_reward = 0.15  # 新しい協力関係
        else:
            collaboration_reward = 0.1  # 既存関係の強化
        
        # 過負荷ペナルティ
        if self.developer_state.stress_level > self.stress_threshold:
            overload_penalty = -0.5 * (self.developer_state.stress_level - self.stress_threshold)
        else:
            overload_penalty = 0.0
        
        total_reward = (base_reward + continuity_reward + stress_reward + 
                       quality_reward + collaboration_reward + overload_penalty)
        
        return total_reward
    
    def _calculate_rejection_reward(self, review: ReviewRequest) -> float:
        """
        拒否時の報酬を計算
        
        Args:
            review: レビュー依頼
            
        Returns:
            float: 計算された報酬
        """
        base_penalty = -0.5  # 基本拒否ペナルティ
        
        # ストレス軽減報酬（高ストレス時の適切な拒否）
        if self.developer_state.stress_level > self.stress_threshold:
            stress_relief = 0.3  # 過負荷回避報酬
            # ストレスレベルを軽減
            self.developer_state.stress_level = max(0.0, self.developer_state.stress_level - 0.05)
        else:
            stress_relief = 0.0
        
        # 専門性不適合による正当な拒否
        if review.expertise_match < 0.3:
            expertise_justification = 0.2
        else:
            expertise_justification = 0.0
        
        # 緊急度による調整
        urgency_penalty = -0.2 * review.urgency_level
        
        total_reward = base_penalty + stress_relief + expertise_justification + urgency_penalty
        
        return total_reward
    
    def _calculate_waiting_reward(self, review: ReviewRequest) -> float:
        """
        待機時の報酬を計算
        
        Args:
            review: レビュー依頼
            
        Returns:
            float: 計算された報酬
        """
        base_penalty = -0.1  # 基本待機ペナルティ
        
        # 情報収集価値（複雑なレビューの場合）
        if review.complexity_score > 0.7:
            information_value = 0.05
        else:
            information_value = 0.0
        
        # 緊急度による調整
        urgency_penalty = -0.1 * review.urgency_level
        
        total_reward = base_penalty + information_value + urgency_penalty
        
        return total_reward
    
    def _predict_review_quality(self, review: ReviewRequest) -> float:
        """
        レビュー品質を予測
        
        Args:
            review: レビュー依頼
            
        Returns:
            float: 予測品質スコア (0-1)
        """
        # 専門性適合度とレビュー品質の相関を模擬
        base_quality = review.expertise_match
        
        # 開発者の専門性レベルによる調整
        expertise_bonus = self.developer_state.expertise_level * 0.2
        
        # 複雑度による調整（適度な複雑度が最適）
        complexity_factor = 1.0 - abs(review.complexity_score - 0.6)
        
        predicted_quality = min(1.0, base_quality + expertise_bonus * complexity_factor)
        
        return predicted_quality
    
    def _update_developer_state_after_acceptance(self, review: ReviewRequest) -> None:
        """
        レビュー受諾後に開発者状態を更新
        
        Args:
            review: 受諾したレビュー依頼
        """
        # ワークロード増加
        effort_impact = review.estimated_review_effort / 10.0  # 正規化
        self.developer_state.workload_ratio = min(1.0, 
            self.developer_state.workload_ratio + effort_impact)
        
        # 専門性向上（適合度が高い場合）
        if review.expertise_match > 0.7:
            learning_gain = 0.01 * self.developer_state.learning_velocity
            self.developer_state.expertise_level = min(1.0,
                self.developer_state.expertise_level + learning_gain)
        
        # 協力関係向上
        if review.requester_relationship < 0.8:
            relationship_gain = 0.05
            # 実際の実装では、特定の開発者との関係性を更新する
        
        # 満足度調整
        if review.expertise_match > 0.6:
            self.developer_state.satisfaction_level = min(1.0,
                self.developer_state.satisfaction_level + 0.02)
        else:
            self.developer_state.satisfaction_level = max(0.0,
                self.developer_state.satisfaction_level - 0.05)
        
        # 受諾率更新
        recent_acceptances = sum(
            1 for h in self.acceptance_history[-10:] 
            if h['action'] == 'accept'
        )
        recent_total = min(len(self.acceptance_history), 10)
        if recent_total > 0:
            self.developer_state.recent_review_acceptance_rate = recent_acceptances / recent_total
    
    def _update_state(self, action: int) -> None:
        """
        行動実行後の状態更新
        
        Args:
            action: 実行された行動
        """
        # 時間経過による自然なストレス軽減
        natural_stress_decay = 0.01
        self.developer_state.stress_level = max(0.0,
            self.developer_state.stress_level - natural_stress_decay)
        
        # ワークロード自然減少
        natural_workload_decay = 0.02
        self.developer_state.workload_ratio = max(0.0,
            self.developer_state.workload_ratio - natural_workload_decay)
        
        # 状態更新時刻を記録
        self.developer_state.last_updated = datetime.now()
    
    def _maybe_add_new_review(self) -> None:
        """
        確率的に新しいレビューをキューに追加
        """
        if not self.enable_random_new_reviews:
            return
        if len(self.review_queue) < self.max_queue_size and np.random.random() < 0.3:
            new_review = ReviewRequest(
                change_id=f"change_{self.current_step}_{np.random.randint(1000, 9999)}",
                author_email=f"author_{np.random.randint(100, 999)}@example.com",
                project="test-project",
                branch="main",
                subject=f"New change at step {self.current_step}",
                files_changed=np.random.randint(1, 15),
                lines_added=np.random.randint(5, 300),
                lines_deleted=np.random.randint(0, 100),
                complexity_score=np.random.uniform(0.1, 1.0),
                technical_domain=np.random.choice(["backend", "frontend", "database", "api"]),
                urgency_level=np.random.uniform(0.1, 1.0),
                estimated_review_effort=np.random.uniform(0.5, 3.0),
                required_expertise=["python"],
                created_at=datetime.now(),
                deadline=datetime.now() + timedelta(hours=np.random.randint(24, 120)),
                expertise_match=np.random.uniform(0.2, 1.0),
                requester_relationship=np.random.uniform(0.0, 1.0)
            )
            self.review_queue.append(new_review)
    
    def _calculate_recent_acceptance_rate(self) -> float:
        """
        最近の受諾率を計算
        
        Returns:
            float: 最近の受諾率
        """
        if not self.acceptance_history:
            return 0.5  # デフォルト値
        
        recent_history = self.acceptance_history[-10:]  # 最近10件
        acceptances = sum(1 for h in recent_history if h['action'] == 'accept')
        
        return acceptances / len(recent_history) if recent_history else 0.5
    
    def _calculate_load_balance(self) -> float:
        """
        負荷バランスを計算
        
        Returns:
            float: 負荷バランススコア
        """
        # キューサイズとワークロードの組み合わせ
        queue_pressure = len(self.review_queue) / self.max_queue_size
        workload_pressure = self.developer_state.workload_ratio
        
        # バランススコア（低いほど良い）
        balance_score = 1.0 - (queue_pressure + workload_pressure) / 2.0
        
        return max(0.0, balance_score)
    
    def _check_terminated(self) -> bool:
        """
        エピソード終了条件をチェック
        
        Returns:
            bool: 終了フラグ
        """
        # 開発者が沸点に達した場合
        if self.developer_state.stress_level >= self.developer_state.boiling_point_estimate:
            logger.info("開発者が沸点に達しました - エピソード終了")
            return True
        
        # 満足度が極端に低下した場合
        if self.developer_state.satisfaction_level <= 0.1:
            logger.info("開発者満足度が極端に低下しました - エピソード終了")
            return True
        
        return False
    
    def _check_truncated(self) -> bool:
        """
        エピソード切り詰め条件をチェック
        
        Returns:
            bool: 切り詰めフラグ
        """
        # 最大ステップ数に達した場合
        if self.current_step >= self.max_episode_length:
            logger.info(f"最大ステップ数 {self.max_episode_length} に達しました - エピソード切り詰め")
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """
        環境情報を取得
        
        Returns:
            Dict[str, Any]: 環境情報辞書
        """
        acceptance_rate = (self.total_reviews_accepted / self.total_reviews_processed 
                          if self.total_reviews_processed > 0 else 0.0)
        
        return {
            'step': self.current_step,
            'developer_stress': self.developer_state.stress_level,
            'developer_satisfaction': self.developer_state.satisfaction_level,
            'queue_size': len(self.review_queue),
            'total_reviews_processed': self.total_reviews_processed,
            'total_reviews_accepted': self.total_reviews_accepted,
            'acceptance_rate': acceptance_rate,
            'episode_reward_sum': sum(self.episode_rewards),
            'boiling_point_margin': (self.developer_state.boiling_point_estimate - 
                                   self.developer_state.stress_level),
            'workload_ratio': self.developer_state.workload_ratio
        }
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        環境の可視化
        
        Args:
            mode: レンダリングモード
            
        Returns:
            Optional[str]: レンダリング結果
        """
        if mode == 'human':
            info = self._get_info()
            output = f"""
=== レビュー受諾環境状態 ===
ステップ: {info['step']}/{self.max_episode_length}
開発者ストレス: {info['developer_stress']:.3f}
開発者満足度: {info['developer_satisfaction']:.3f}
沸点余裕: {info['boiling_point_margin']:.3f}
キューサイズ: {info['queue_size']}/{self.max_queue_size}
処理済みレビュー: {info['total_reviews_processed']}
受諾済みレビュー: {info['total_reviews_accepted']}
受諾率: {info['acceptance_rate']:.3f}
エピソード累積報酬: {info['episode_reward_sum']:.3f}
ワークロード: {info['workload_ratio']:.3f}
"""
            print(output)
            return output
        
        return None
    
    def close(self) -> None:
        """環境をクローズ"""
        logger.info("レビュー受諾環境をクローズしました")
        pass