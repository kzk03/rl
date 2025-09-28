# examples ディレクトリ概要

- 目的: 使い方のサンプルコード。
- 主なファイル:
  - `training_pipeline_example.py`: 学習パイプライン例。
  - `ab_testing_example.py`: AB テスト例。
  - `visualization_example.py`: 可視化例。
  - `retention_auc_demo.py`: 合成データでの AUC/AP/Brier/ECE デモ。
  - `real_data_evaluate.py`: 実データ(CSV/Parquet)で RetentionPredictor を評価。
  - `build_eval_from_changes.py`: 抽出した changes\_\*.json から評価用 CSV を生成。

使い方の例:

1. Gerrit の変更を抽出（`data_processing/gerrit_extraction/extract_changes.py` を参照）

   - 出力: `data/raw/gerrit_changes/changes_YYYYmmdd_HHMMSS.json`

2. 評価用 CSV に変換

   - uv run python examples/build*eval_from_changes.py --input data/raw/gerrit_changes/changes*\*.json --output data/retention_samples.csv

3. 予測器で評価（IRL 特徴の有無は任意）

   - uv run python examples/real_data_evaluate.py --input data/retention_samples.csv --format csv --enable-irl --irl-model-path path/to/irl.joblib

   ## レビュー依頼に対する応答予測（新フロー）

   1. Gerrit から詳細つき変更を抽出（messages / reviewers / reviewer_updates が必要）

      - uv run python data_processing/gerrit_extraction/extract_changes.py --projects <proj...> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --output-prefix <prefix>

   2. レビュー依頼データセットを生成

      - uv run python examples/build_eval_from_review_requests.py --input data/raw/gerrit_changes/<prefix>\*.json --output data/review_requests.csv --response-window-days 14

   3. 時系列分割 + 開発者グループ分離で評価（ギャップ特徴の無効化でフェア性確認）

      - uv run python examples/real_data_evaluate.py --input data/review_requests.csv --format csv --split-mode time --group-by-email --exclude-gap-features

   出力 CSV の主な列: developer_email, context_date(=request_time), label (0/1), days_since_last_activity など。
