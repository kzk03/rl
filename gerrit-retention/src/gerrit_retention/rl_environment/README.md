# rl_environment ディレクトリ概要

- 目的: 強化学習の環境/報酬/エージェント関連。
- 主なファイル:
  - `review_env.py`: 単一受理環境（実データ遷移、継続ボーナス、IRL ラッパー対応）。
  - `multi_reviewer_assignment_env.py`: 複数候補からの割当環境（報酬: GT 一致/IRL 確率/受理確率）。
  - `irl_reward_wrapper.py`: 受理環境の報酬を IRL に置換/加重。
  - `ppo_agent.py`: PPO 関連ヘルパ。
  - `reward_calculator.py`: 報酬計算補助。
  - `time_split_data_wrapper.py`: 変更単位の学習/評価分割補助。
