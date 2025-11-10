# Nova Review Acceptance - importants 準拠実験

## 概要

importants のレビュー受諾予測実験を完全に再現し、Enhanced IRL (Attention)との公平な比較を実施。

## 実験設定（importants と完全一致）

### データ

- プロジェクト: OpenStack Nova
- 期間: 2021-01-01 ~ 2024-01-01 (36 ヶ月)
- 訓練: 2021-01-01 ~ 2023-01-01 (24 ヶ月、全パターン共通)
- 評価: 2023-01-01 ~ 2024-01-01 (12 ヶ月、4 期間)

### Future Window 設定

各月末から動的に Future Window を適用（importants 方式）:

- 0-3m: 各月末から 0~3 ヶ月後を見る
- 3-6m: 各月末から 3~6 ヶ月後を見る
- 6-9m: 各月末から 6~9 ヶ月後を見る
- 9-12m: 各月末から 9~12 ヶ月後を見る

### 月次軌跡生成

- 訓練期間の各月末を基準点として軌跡を生成
- train_end でクリップしてデータリーク防止
- Future Window が train_end を超える月は自動的にスキップ

### 予想される軌跡数

- 0-3m: 約 121 軌跡 (23 ヶ月使用可能)
- 3-6m: 約 100 軌跡 (17 ヶ月使用可能)
- 6-9m: 約 80 軌跡 (11 ヶ月使用可能)
- 9-12m: 約 60 軌跡 (5 ヶ月使用可能)

## 実装

### 1. Attention-less IRL (importants baseline)

スクリプト: `run_cross_eval_importants.py`

- importants の`train_irl_review_acceptance.py`をそのまま使用
- エポック数: 50 (Enhanced IRL と揃える)
- 結果ディレクトリ: `results_importants/`

### 2. Enhanced IRL (Attention)

スクリプト: `run_cross_eval_enhanced_irl.py`

- データ準備: importants と完全同一の方式
- モデル: RetentionIRLSystem (Attention 有効)
- エポック数: 50
- 結果ディレクトリ: `results_enhanced_irl/`

## 実行コマンド

```bash
# Enhanced IRL (Attention) - 4×4クロス評価
cd /Users/kazuki-h/rl/gerrit-retention
nohup uv run python experiments/nova_review_acceptance/scripts/run_cross_eval_enhanced_irl.py \
  > experiments/nova_review_acceptance/results_enhanced_irl/run.log 2>&1 &

# Attention-less IRL (importants baseline) - 4×4クロス評価
nohup uv run python experiments/nova_review_acceptance/scripts/run_cross_eval_importants.py \
  > experiments/nova_review_acceptance/results_importants/run.log 2>&1 &
```

## 進捗確認

```bash
# Enhanced IRL進捗
tail -f experiments/nova_review_acceptance/results_enhanced_irl/run.log

# importants進捗
tail -f experiments/nova_review_acceptance/results_importants/run.log

# 結果確認
ls experiments/nova_review_acceptance/results_enhanced_irl/train_*/eval_*/metrics.json
ls experiments/nova_review_acceptance/results_importants/train_*/eval_*/metrics.json
```

## 期待される結果

### importants baseline (AUC-ROC 予想)

- 対角線平均: ~0.75
- 最高性能: ~0.82 (train: 3-6m)
- 全体平均: ~0.75

### Enhanced IRL (Attention) 予想

- 対角線平均: ~0.80 (Attention による改善)
- 最高性能: ~0.85+
- Attention による改善幅: +0.05 ~ +0.10

## 検証ポイント

1. ✅ データ準備方式が importants と一致
2. ✅ 軌跡数が一致 (0-3m: 121 軌跡)
3. ✅ Future Window 方式が一致（月末からの動的オフセット）
4. ✅ クリップロジックが一致（train_end で切る）
5. ⏳ 実験実行中
6. ⏳ 結果比較待ち

## タイムライン

- 2025-11-10 01:27: Enhanced IRL 訓練開始
- 推定完了時刻: 2025-11-10 03:00 (約 1.5 時間、4 パターン × 50 エポック)

## ファイル構成

```
experiments/nova_review_acceptance/
├── scripts/
│   ├── train_enhanced_irl_importants.py  # Enhanced IRL訓練（importants準拠）
│   ├── run_cross_eval_enhanced_irl.py    # Enhanced IRLクロス評価
│   └── run_cross_eval_importants.py      # importantsクロス評価
├── results_enhanced_irl/                 # Enhanced IRL結果
│   ├── run.log
│   └── train_{period}/eval_{period}/
│       ├── metrics.json
│       └── enhanced_irl_model.pt
├── results_importants/                   # importants結果
│   ├── run.log
│   └── train_{period}/eval_{period}/
│       ├── metrics.json
│       └── irl_model.pt
└── README_EXPERIMENT.md                  # このファイル
```
