#!/bin/bash

# eval-snapshot を固定 (2023-07-01)
EVAL_DATE="2023-07-01"

# 学習データの遡り期間 (eval から何ヶ月前まで学習するか)
for train_offset in 1 3 6 12; do
  # train-snapshot を eval から train_offset ヶ月前に設定
  TRAIN_DATE=$(date -j -f "%Y-%m-%d" -v-${train_offset}m "$EVAL_DATE" +"%Y-%m-%d")
  
  # 予測期間 (eval から何ヶ月後まで予測するか)
  for predict_months in 3 6 12; do
    echo "Running: train_offset=${train_offset} months, predict_months=${predict_months} months"
    uv run python scripts/training/retention/run_retention_baseline_pipeline.py \
      --train-snapshot-date "$TRAIN_DATE" \
      --eval-snapshot-date "$EVAL_DATE" \
      --review-requests data/review_requests_openstack_multi_5y_detail.csv \
      --feature-history-months 18 \
      --target-window-months "$predict_months" \
      --activity-after-months "$predict_months" \
      --label-column is_active_after_threshold \
      --output-dir outputs/retention/models \
      --overwrite
  done
done
