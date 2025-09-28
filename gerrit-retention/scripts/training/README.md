# training ディレクトリ概要

- 目的: 学習スクリプト（RL/オフライン/評価）。
- 主な内容:
  - `rl_training/stable_ppo_training.py`: 単一受理環境の PPO 学習。
  - `rl_training/train_assignment_ppo.py`: 割当環境の PPO 学習（IRL/受理確率報酬、変更単位分割）。
  - `offline_rl/`: 行動クローニング等。
  - `evaluation/` 連携: AB テスト/統合テストの補助コードと併用。
