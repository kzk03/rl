# scripts ディレクトリ詳細

- 目的: データ収集/前処理/IRL/評価/可視化/運用までのワンショット実行スクリプト集。

## 主なスクリプト一覧（要約）

- `collect_real_data.py`: Gerrit からデータ収集。
- `connect_real_gerrit.py`: Gerrit 接続の確認/設定。
- `run_full_training_pipeline.py`: 収集 → 前処理 → 特徴 → 学習まで一気通貫。
- `run_reviewer_invitation_ranking.py`: 招待ランキング IRL の学習/出力。
- `run_reviewer_acceptance_after_invite.py`: 招待後受理モデルの学習/出力。
- `run_reviewer_acceptance_eval.py`: 受理モデルの評価。
- `run_api_server.py`: FastAPI サーバ起動。
- `run_task_recommender.py`: タスク推薦一括実行。
- `run_retention_analysis.py`: 継続率の分析。
- `evaluate_future_window_retention.py`: 将来窓での継続率評価。
- `generate_visualizations.py`: 可視化生成。
- `analyze_prediction_vs_reality.py`: 予測と実績の乖離分析。
- `evaluate_reviewer_irl_holdout.py`: 招待 IRL のホールドアウト評価。
- `train_irl_author_engagement.py`: 著者側 IRL 学習。
- `train_irl_reviewer_engagement.py`: レビュアー側 IRL 学習。
- `advanced_rl_system.py`: 先進 RL シナリオの一括実行。
- `rl_task_optimizer.py`: RL タスク最適化のユーティリティ。
- `verify_project_structure.py`: 構成検証。
- `...` その他: データ整備/生成、例示用スクリプト。

## 実行のヒント

- ほとんどのスクリプトは `--help` で引数説明が表示されます。
- 入出力は `configs/*.yaml` に合わせて指定するのがおすすめです。
- uv 経由の実行例:

```bash
uv run python scripts/run_reviewer_invitation_ranking.py --input data/processed/unified/all_reviews.json --output outputs/reviewer_invitation_ranking/
```
