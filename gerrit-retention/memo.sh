#!/bin/bash
# 同期コマンド
# S → L
rsync -avz --progress /Users/kazuki-h/rl/gerrit-retention/ socsel:/mnt/data1/kazuki-h/gerrit-retention/


# L → S
rsync socsel:/mnt/data1/kazuki-h/rl/gerrit-retention/ /Users/kazuki-h/rl/






# ================================================================================
# IRL 8×8行列評価メモ
# ================================================================================

# 現在実行中（バックグラウンド ID: 5f90d2）:
# - 8×8行列評価（2023-01-01スナップショット）
#   進捗: 12/64実験完了
#   完了予定: 約15分後

# ================================================================================
# 完了後に実行するコマンド
# ================================================================================

# 1. ヒートマップ作成（全メトリクス）
uv run python scripts/visualization/create_heatmap_8x8.py \
  --csv importants/irl_matrix_8x8_2023q1/sliding_window_results_fixed_pop_seq.csv \
  --output importants/irl_matrix_8x8_2023q1/heatmaps \
  --title-suffix " (2023 Q1)"

# 2. 詳細分析レポート
uv run python scripts/analysis/analyze_8x8_matrix.py \
  --csv importants/irl_matrix_8x8_2023q1/sliding_window_results_fixed_pop_seq.csv \
  --output importants/irl_matrix_8x8_2023q1/analysis_report.md

# ================================================================================
# 統合スクリプト（評価+可視化+分析を一括実行）
# ================================================================================

# 2023 Q1（既に実行中）
./scripts/evaluation/run_8x8_and_visualize.sh --snapshot-date 2023-01-01

# 他のスナップショット日でも実行する場合:
./scripts/evaluation/run_8x8_and_visualize.sh --snapshot-date 2022-10-01
./scripts/evaluation/run_8x8_and_visualize.sh --snapshot-date 2022-07-01
./scripts/evaluation/run_8x8_and_visualize.sh --snapshot-date 2022-04-01