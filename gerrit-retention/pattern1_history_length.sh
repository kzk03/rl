#!/bin/bash

# パターン1: eval を固定し、学習データの履歴長 (feature-history-months) を変化させて精度変化を検証
EVAL_DATE="2023-07-01"
TRAIN_DATE="2022-07-01"  # train-snapshot を固定 (eval の1年前)
PREDICT_MONTHS=12

for history in 1 3 6 12 18; do
  echo "Pattern 1: history=${history} months"
  uv run python scripts/training/retention/run_retention_baseline_pipeline.py \
    --train-snapshot-date "$TRAIN_DATE" \
    --eval-snapshot-date "$EVAL_DATE" \
    --review-requests data/review_requests_openstack_multi_5y_detail.csv \
    --feature-history-months $history \
    --target-window-months $PREDICT_MONTHS \
    --activity-after-months $PREDICT_MONTHS \
    --label-column is_active_after_threshold \
    --output-dir outputs/retention/models \
    --overwrite
done
