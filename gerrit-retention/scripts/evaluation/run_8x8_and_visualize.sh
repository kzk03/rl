#!/bin/bash
# 8×8マトリクス評価の実行と可視化（3ヶ月単位スライディングウィンドウ）

set -e

# デフォルト設定
REVIEWS_FILE="data/review_requests_openstack_no_bots.csv"
START_DATE="2023-01-01"
OUTPUT_BASE="importants/irl_matrix_8x8"

# ヘルプ表示
show_help() {
    cat << EOF
使用方法: $0 [OPTIONS]

3ヶ月単位のスライディングウィンドウで2年間（8×8マトリクス）のIRL評価を実行

オプション:
    -r, --reviews FILE      レビューCSVファイル (デフォルト: $REVIEWS_FILE)
    -s, --start-date DATE   開始日 YYYY-MM-DD (デフォルト: $START_DATE)
    -e, --enhanced          拡張特徴量を使用
    -o, --output DIR        出力ディレクトリ (デフォルト: $OUTPUT_BASE)
    -h, --help              このヘルプを表示

使用例:
    # 基本版で実行
    $0 --start-date 2023-01-01

    # 拡張特徴量で実行
    $0 --start-date 2023-01-01 --enhanced --output importants/irl_matrix_8x8_enhanced

    # カスタムデータで実行
    $0 --reviews data/custom_reviews.csv --start-date 2022-01-01
EOF
}

# パラメータ解析
ENHANCED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--reviews)
            REVIEWS_FILE="$2"
            shift 2
            ;;
        -s|--start-date)
            START_DATE="$2"
            shift 2
            ;;
        -e|--enhanced)
            ENHANCED="--use-enhanced-features"
            shift
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# 出力ディレクトリを決定
if [ -n "$ENHANCED" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}_enhanced_${START_DATE//-/}"
else
    OUTPUT_DIR="${OUTPUT_BASE}_${START_DATE//-/}"
fi

echo "=========================================="
echo "8×8マトリクス評価開始"
echo "=========================================="
echo "レビューファイル: $REVIEWS_FILE"
echo "開始日: $START_DATE"
echo "拡張特徴量: $([ -n "$ENHANCED" ] && echo 'はい' || echo 'いいえ')"
echo "出力先: $OUTPUT_DIR"
echo "=========================================="

# データファイルの存在確認
if [ ! -f "$REVIEWS_FILE" ]; then
    echo "エラー: レビューファイルが見つかりません: $REVIEWS_FILE"
    exit 1
fi

# ステップ1: 8×8マトリクス評価実行
echo ""
echo "ステップ1: 8×8マトリクス評価を実行中..."
echo ""

uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
    --reviews "$REVIEWS_FILE" \
    --start-date "$START_DATE" \
    $ENHANCED \
    --sequence \
    --seq-len 15 \
    --epochs 30 \
    --output "$OUTPUT_DIR"

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 8×8マトリクス評価が完了しました"
    echo ""
else
    echo ""
    echo "❌ エラーが発生しました"
    exit 1
fi

# ステップ2: 結果サマリー表示
echo "=========================================="
echo "結果サマリー"
echo "=========================================="

if [ -f "$OUTPUT_DIR/MATRIX_8x8_REPORT.md" ]; then
    echo ""
    echo "📊 マトリクスレポート:"
    head -n 30 "$OUTPUT_DIR/MATRIX_8x8_REPORT.md"
    echo ""
    echo "（続きは $OUTPUT_DIR/MATRIX_8x8_REPORT.md を参照）"
fi

# ステップ3: ファイル一覧表示
echo ""
echo "=========================================="
echo "生成されたファイル"
echo "=========================================="
ls -lh "$OUTPUT_DIR"

# ステップ4: 次のステップの提案
echo ""
echo "=========================================="
echo "次のステップ"
echo "=========================================="
echo ""
echo "📄 詳細レポート:"
echo "  cat $OUTPUT_DIR/MATRIX_8x8_REPORT.md"
echo ""
echo "📊 NumPy行列データ:"
echo "  - $OUTPUT_DIR/matrix_auc_roc.npy"
echo "  - $OUTPUT_DIR/matrix_auc_pr.npy"
echo "  - $OUTPUT_DIR/matrix_f1.npy"
echo ""
echo "📈 Pythonで可視化:"
echo "  import numpy as np"
echo "  import matplotlib.pyplot as plt"
echo "  matrix = np.load('$OUTPUT_DIR/matrix_auc_roc.npy')"
echo "  plt.imshow(matrix, cmap='viridis')"
echo "  plt.colorbar()"
echo "  plt.show()"
echo ""
