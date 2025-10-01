from __future__ import annotations

"""
2アクション方策（0=非受諾, 1=受諾）を、基底の3アクション環境
ReviewAcceptanceEnvironment (0=reject,1=accept,2=wait) に写像するラッパ。

- action_space: Discrete(2)
  - 0 -> base_env.ACTION_REJECT (0)
  - 1 -> base_env.ACTION_ACCEPT (1)
  - wait(2) は使用しない
"""

import gymnasium as gym
from gymnasium import spaces


class BinaryActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        # 0/1 以外は安全側で 0 に丸める
        a = int(action)
        if a not in (0, 1):
            a = 0
        # 0 -> reject(0), 1 -> accept(1)
        return self.env.step(a)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
