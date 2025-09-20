"""
IRL報酬ラッパー

ReviewAcceptanceEnvironment の報酬を、学習済みの IRL 報酬モデルで
置き換える or ブレンドする薄い Gym ラッパー。

使い方:
    env = ReviewAcceptanceEnvironment(env_config)
    irl = RetentionIRLSystem({'state_dim': 20, 'action_dim': 3})
    irl.load_model('path/to/irl_model.pth')
    wrapped = IRLRewardWrapper(env, irl, mode='blend', alpha=0.7)
    obs, info = wrapped.reset()
    ...
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch


class IRLRewardWrapper(gym.Wrapper):
    """IRL 予測報酬で元の環境報酬を置換/混合するラッパー。

    Args:
        env: ベース環境 (ReviewAcceptanceEnvironment を想定)
        irl_system: RetentionIRLSystem のインスタンス（学習済み or これから学習）
        mode: 'replace' なら IRL の報酬に置換, 'blend' なら線形結合
        alpha: blend のときの係数。new = alpha * irl + (1-alpha) * original
    """

    def __init__(
        self,
        env: gym.Env,
        irl_system: Any,
        mode: str = "replace",
        alpha: float = 1.0,
        engagement_bonus_weight: float = 0.0,
        accept_action_id: int | None = None,
    ) -> None:
        super().__init__(env)
        assert mode in ("replace", "blend"), "mode must be 'replace' or 'blend'"
        self.irl_system = irl_system
        self.mode = mode
        self.alpha = float(alpha)
        self.engagement_bonus_weight = float(engagement_bonus_weight)
        # 受諾行動ID（既定: ReviewAcceptanceEnvironment の ACTION_ACCEPT=1 を想定）
        self.accept_action_id = 1 if accept_action_id is None else int(accept_action_id)

        # 推論用に state_dim / action_dim を決定
        # 既存 IRL モデルの設定と合うことが望ましい。
        self._state_dim = getattr(irl_system, "state_dim", None) or env.observation_space.shape[0]
        self._action_dim = getattr(irl_system, "action_dim", None) or env.action_space.n

        # 実際の観測/行動次元と IRL モデルの次元が違う場合は、
        # 事前に IRL モデル側の再学習 or 変換を行ってください。

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: int):
        next_obs, orig_reward, terminated, truncated, info = self.env.step(action)

        # IRL モデルで報酬を推定
        try:
            irl_reward = self._predict_irl_reward(next_obs, action)
        except Exception as e:  # フォールバック: IRL 失敗時は元の報酬
            irl_reward = orig_reward
            info = dict(info or {})
            info.setdefault("warnings", []).append(f"IRL reward failed: {e}")

        if self.mode == "replace":
            new_reward = float(irl_reward)
        else:  # blend
            new_reward = float(self.alpha * irl_reward + (1.0 - self.alpha) * orig_reward)

        # エンゲージメント（レビューしてくれる＝受諾）ボーナス
        engagement_bonus = 0.0
        if self.engagement_bonus_weight != 0.0 and int(action) == int(self.accept_action_id):
            # ボーナスはキュー充填率やストレスに応じて控えめに重み付け可能
            # ここでは単純に定数重みを加算（必要なら info から調整可能）
            engagement_bonus = float(self.engagement_bonus_weight)
            new_reward += engagement_bonus

        # デバッグ用に情報を付与
        info = dict(info or {})
        info.update({
            "orig_reward": float(orig_reward),
            "irl_reward": float(irl_reward),
            "reward_mode": self.mode,
            "reward_alpha": self.alpha,
            "engagement_bonus": float(engagement_bonus),
            "accept_action_id": int(self.accept_action_id),
        })

        return next_obs, new_reward, terminated, truncated, info

    @torch.no_grad()
    def _predict_irl_reward(self, obs: np.ndarray, action: int) -> float:
        """IRL ネットワークで (state, action) から報酬を推定する。

        ここでは最小限の前提として:
          - state は観測ベクトル (env.observation_space) をそのまま使用
          - action は one-hot (env.action_space.n)
        を用いる。
        IRL を別の特徴で学習している場合は、対応する前処理/変換を実装してください。
        """
        device = getattr(self.irl_system, "device", torch.device("cpu"))
        # state
        s = np.asarray(obs, dtype=np.float32)
        if s.shape[-1] != self._state_dim:
            # 次元不一致時はゼロパディング/トリム（暫定フォールバック）
            if s.shape[-1] < self._state_dim:
                pad = np.zeros((self._state_dim - s.shape[-1],), dtype=np.float32)
                s = np.concatenate([s, pad], axis=0)
            else:
                s = s[: self._state_dim]
        s_t = torch.from_numpy(s).unsqueeze(0).to(device)

        # action one-hot
        a = np.zeros((self._action_dim,), dtype=np.float32)
        if 0 <= int(action) < self._action_dim:
            a[int(action)] = 1.0
        a_t = torch.from_numpy(a).unsqueeze(0).to(device)

        # IRL forward: (reward, continuation_prob)
        reward_pred, _ = self.irl_system.network(s_t, a_t)
        return float(reward_pred.item())
