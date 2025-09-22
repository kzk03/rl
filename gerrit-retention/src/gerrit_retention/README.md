# src/gerrit_retention 概要

- 目的: ライブラリ実装本体（IRL/推薦/RL/可視化/ユーティリティ）。
- 下位ディレクトリ:
  - `irl/`: 受理/継続の IRL 実装（例: MaxEnt 二値）
  - `recommendation/`: レビュアー招待ランキング/受理確率など
  - `rl_environment/`: RL 環境と報酬計算、PPO 支援
  - `offline/`: オフラインデータセット構築
  - `prediction/`: 予測器
  - `utils/`: 共通ユーティリティ
  - `visualization/`: 可視化
  - `analysis/`, `behavior_analysis/`: 解析
