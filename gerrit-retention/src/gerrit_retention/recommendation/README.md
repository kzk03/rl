# recommendation ディレクトリ概要

- 目的: 招待（誰を選ぶか）と受理（参加するか）のモデリング。
- 主なファイル:
  - `reviewer_invitation_ranking.py`: 条件付きロジット/Plackett–Luce の IRL。`evaluate_invitation_irl(_plackett)` など。
  - `reviewer_acceptance_after_invite.py`: 受理確率モデル（ロジスティックなど）。
  - `reviewer_acceptance.py`: 受理行動の補助/分析。
  - `task_recommendation_pipeline.py`: 推薦パイプライン。
