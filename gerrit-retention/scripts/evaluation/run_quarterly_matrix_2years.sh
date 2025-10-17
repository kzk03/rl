#!/bin/bash
# 2年分の四半期ごと4×4行列評価スクリプト

set -e

BASE_CMD="uv run python scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py"
DATA_FILE="data/review_requests_openstack_multi_5y_detail.csv"
HISTORY_MONTHS="3 6 9 12"
TARGET_MONTHS="3 6 9 12"
SEQ_LEN=15
EPOCHS=30
REFERENCE_PERIOD=6

# 8つの四半期スナップショット（2年分）
SNAPSHOTS=(
    "2021-01-01:2021q1"
    "2021-04-01:2021q2"
    "2021-07-01:2021q3"
    "2021-10-01:2021q4"
    "2022-01-01:2022q1"
    "2022-04-01:2022q2"
    "2022-07-01:2022q3"
    "2022-10-01:2022q4"
)

echo "=================================="
echo "2年分四半期評価（8スナップショット×16実験）"
echo "=================================="

for snapshot in "${SNAPSHOTS[@]}"; do
    IFS=':' read -r date label <<< "$snapshot"

    echo ""
    echo "===================="
    echo "スナップショット: $date ($label)"
    echo "===================="

    OUTPUT_DIR="importants/irl_matrix_$label"

    $BASE_CMD \
        --reviews $DATA_FILE \
        --snapshot-date $date \
        --reference-period $REFERENCE_PERIOD \
        --history-months $HISTORY_MONTHS \
        --target-months $TARGET_MONTHS \
        --sequence \
        --seq-len $SEQ_LEN \
        --epochs $EPOCHS \
        --output $OUTPUT_DIR

    echo "✅ $label 完了"
done

echo ""
echo "=================================="
echo "全ての四半期評価が完了しました！"
echo "総スナップショット数: 8"
echo "総実験数: 128 (8×16)"
echo "=================================="
