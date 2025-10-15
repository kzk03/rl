#!/bin/bash

# パターン2: eval を固定し、train-snapshot の距離 (eval から何ヶ月前) を変化させて精度変化を検証
EVAL_DATE="2023-07-01"
HISTORY_MONTHS=18  # feature-history-months を固定
PREDICT_MONTHS=12

for offset in 1 3 6 12 18; do
  # train-snapshot を eval から offset ヶ月前に設定
  TRAIN_DATE=$(date -j -f "%Y-%m-%d" -v-${offset}m "$EVAL_DATE" +"%Y-%m-%d")
  echo "Pattern 2: train_offset=${offset} months (train_date=$TRAIN_DATE)"
  uv run python scripts/training/retention/run_retention_baseline_pipeline.py \
    --train-snapshot-date "$TRAIN_DATE" \
    --eval-snapshot-date "$EVAL_DATE" \
    --review-requests data/review_requests_openstack_multi_5y_detail.csv \
    --feature-history-months $HISTORY_MONTHS \
    --target-window-months $PREDICT_MONTHS \
    --activity-after-months $PREDICT_MONTHS \
    --label-column is_active_after_threshold \
    --output-dir outputs/retention/models \
    --overwrite
done
