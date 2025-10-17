#!/bin/bash
# 8×8行列評価とヒートマップ可視化の統合スクリプト

set -e

# デフォルト値
SNAPSHOT_DATE="2023-01-01"
DATA_FILE="data/review_requests_openstack_multi_5y_detail.csv"
OUTPUT_BASE="importants/irl_matrix_8x8"

# 引数処理
while [[ $# -gt 0 ]]; do
    case $1 in
        --snapshot-date)
            SNAPSHOT_DATE="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# スナップショット日からラベルを生成（例: 2023-01-01 -> 2023q1）
LABEL=$(echo $SNAPSHOT_DATE | awk -F'-' '{
    year=$1
    month=$2
    if (month <= 3) quarter=1
    else if (month <= 6) quarter=2
    else if (month <= 9) quarter=3
    else quarter=4
    printf "%sq%d", year, quarter
}')

OUTPUT_DIR="${OUTPUT_BASE}_${LABEL}"

echo "========================================"
echo "8×8 IRL評価と可視化"
echo "========================================"
echo "スナップショット日: $SNAPSHOT_DATE"
echo "データファイル: $DATA_FILE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "========================================"
echo ""

# ステップ1: 8×8行列評価（64実験）
echo "ステップ1: 8×8行列評価を実行中..."
echo ""

uv run python scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py \
    --reviews $DATA_FILE \
    --snapshot-date $SNAPSHOT_DATE \
    --reference-period 6 \
    --history-months 3 6 9 12 15 18 21 24 \
    --target-months 3 6 9 12 15 18 21 24 \
    --sequence \
    --seq-len 15 \
    --epochs 30 \
    --output $OUTPUT_DIR

echo ""
echo "✅ ステップ1完了"
echo ""

# ステップ2: ヒートマップ作成
echo "ステップ2: ヒートマップを作成中..."
echo ""

CSV_FILE="$OUTPUT_DIR/sliding_window_results_fixed_pop_seq.csv"
HEATMAP_DIR="$OUTPUT_DIR/heatmaps"

uv run python scripts/visualization/create_heatmap_8x8.py \
    --csv $CSV_FILE \
    --output $HEATMAP_DIR \
    --title-suffix " ($LABEL)"

echo ""
echo "✅ ステップ2完了"
echo ""

# ステップ3: 詳細分析レポート作成
echo "ステップ3: 詳細分析レポートを作成中..."
echo ""

ANALYSIS_FILE="$OUTPUT_DIR/analysis_report.md"

uv run python scripts/analysis/analyze_8x8_matrix.py \
    --csv $CSV_FILE \
    --output $ANALYSIS_FILE

echo ""
echo "✅ ステップ3完了"
echo ""

# 完了メッセージ
echo "========================================"
echo "✅ すべての処理が完了しました！"
echo "========================================"
echo ""
echo "生成されたファイル:"
echo "  - 評価結果CSV: $CSV_FILE"
echo "  - ヒートマップ: $HEATMAP_DIR/"
echo "  - 分析レポート: $ANALYSIS_FILE"
echo ""
echo "========================================"
