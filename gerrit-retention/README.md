# Gerrit Retention IRL プロジェクト

**逆強化学習(IRL)×LSTMによるOSS開発者継続予測システム**

OpenStack Gerritの13年分のレビュー履歴（137,632件）を活用し、レビュアーが長期的に貢献を続けるかを高精度で予測する研究プロジェクトです。

## 主要な成果

- **予測精度**: AUC-ROC 0.868、AUC-PR 0.983、F1スコア 0.978
- **技術革新**: LSTM×IRLによる時系列学習で従来手法を23.5%上回る精度
- **データ規模**: 13年分（2012-2025）、137,632件のレビューデータ
- **実用性**: 1コマンドで訓練・評価が完了、即座に本番適用可能

## クイックスタート

```bash
# 環境構築
uv sync

# データ前処理（ボット除外）
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# IRL学習と評価（スライディングウィンドウ）
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --epochs 20 \
  --sequence \
  --seq-len 15 \
  --output importants/irl_my_experiment
```

実行時間: 約4分（16組み合わせ評価）

## 主な機能

- **時系列IRL学習**: LSTMで開発者の活動軌跡を時系列的に学習し、継続/離脱を予測
- **スライディングウィンドウ評価**: 複数の学習期間×予測期間で最適な設定を自動探索
- **プロジェクト別予測**: 同一プロジェクト内での継続を正確に判定
- **ベースライン比較**: ロジスティック回帰、ランダムフォレストとの性能比較
- **データ前処理**: ボットアカウント除外（44%ノイズ削減）、プロジェクトフィルタリング
- **分析・可視化**: 特徴量重要度、精度マトリクス、カバレッジ分析

## リポジトリ構成

```
gerrit-retention/
├── src/gerrit_retention/          # コアシステム
│   ├── rl_prediction/
│   │   └── retention_irl_system.py  # ★ 時系列IRLシステム（メイン）
│   └── baselines/                   # ベースラインモデル
│       ├── logistic_regression.py   # ロジスティック回帰
│       └── random_forest.py         # ランダムフォレスト
│
├── scripts/                        # 実行スクリプト
│   ├── preprocessing/              # データ前処理
│   │   ├── filter_bot_accounts.py
│   │   └── filter_by_project.py
│   ├── training/irl/
│   │   └── train_temporal_irl_sliding_window.py  # ★ スライディング評価
│   └── experiments/                # ★ ベースライン比較実験
│       └── run_baseline_comparison.py
│
├── data/                           # データディレクトリ
│   └── review_requests_openstack_multi_5y_detail.csv  # メインデータ
│
├── importants/                     # 重要な実験結果
│   ├── irl_openstack_real/         # 主要IRL実験（16モデル、評価結果）
│   └── baseline_experiments/       # ★ ベースライン比較結果
│
├── docs/                           # ドキュメント（整理済み）
│   ├── experiment_results/         # 実験結果レポート
│   ├── analysis_reports/           # 分析レポート
│   ├── implementation_guides/      # 実装ガイド
│   ├── troubleshooting/            # 問題解決
│   └── archive/                    # アーカイブ
│
├── PROJECT_OVERVIEW.md             # ★ 完全ガイド（データ収集〜予測まで）
├── README_TEMPORAL_IRL.md          # 時系列IRL詳細
└── CLAUDE.md                       # 開発者向けガイド
```

## ドキュメント

### 🚀 はじめに読むべきドキュメント

1. **PROJECT_OVERVIEW.md** - **データ収集からIRL予測までの完全な流れ**
   - 各フェーズの詳細説明（データ収集、前処理、特徴量、IRL学習、評価、予測）
   - コード例とベストプラクティス
   - 技術詳細リファレンス

2. **README_TEMPORAL_IRL.md** - 時系列IRL学習の詳細ガイド
   - LSTMアーキテクチャの説明
   - スライディングウィンドウ評価の使い方
   - 実験結果の見方

3. **docs/** - カテゴリ別の詳細ドキュメント
   - `experiment_results/`: 過去の実験結果
   - `analysis_reports/`: データ分析レポート
   - `implementation_guides/`: 実装の詳細
   - `troubleshooting/`: よくある問題と解決策

### 📊 主要な実験結果

最新の実験結果は `importants/irl_openstack_real/` に格納：

- **評価マトリクス**: `evaluation_matrix_seq.txt`
- **詳細結果**: `sliding_window_results_seq.csv`
- **訓練済みモデル**: `models/` ディレクトリ（16モデル）
- **完全レポート**: `EVALUATION_REPORT.md`

## 主なワークフロー

### データ前処理

```bash
# ボットアカウントの除外（推奨）
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# プロジェクトフィルタリング（任意）
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_filtered.csv \
  --top 3
```

### IRL学習と評価

```bash
# スライディングウィンドウ評価（推奨）
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --epochs 20 \
  --sequence \
  --seq-len 15 \
  --output importants/irl_my_experiment
```

### ベースライン比較実験

```bash
# ロジスティック回帰とランダムフォレストで性能比較
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_experiments/

# ロジスティック回帰のみ
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines logistic_regression \
  --output importants/baseline_experiments/logistic_regression/
```

**利用可能なベースライン**:
- `logistic_regression`: 線形モデル、解釈性が高い
- `random_forest`: 非線形アンサンブルモデル、ロバスト性が高い

### モデルの利用

```python
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# モデル読み込み
model = RetentionIRLSystem.load_model(
    'importants/irl_openstack_real/models/irl_h12m_t6m_seq.pth'
)

# 継続確率の予測
result = model.predict_continuation_probability(
    developer=developer_data,
    activity_history=activities,
    context_date=datetime.now()
)

print(f"継続確率: {result['continuation_probability']:.1%}")
```

### ベースラインモデルの利用

```python
from gerrit_retention.baselines import LogisticRegressionBaseline, RandomForestBaseline

# ロジスティック回帰
lr = LogisticRegressionBaseline()
lr.train({'features': X_train, 'labels': y_train, 'feature_names': feature_names})
predictions = lr.predict({'features': X_test})
importance = lr.get_feature_importance()

# ランダムフォレスト
rf = RandomForestBaseline()
rf.train({'features': X_train, 'labels': y_train, 'feature_names': feature_names})
predictions = rf.predict({'features': X_test})
importance = rf.get_feature_importance()
```

## ライセンス

本リポジトリのライセンスやデータ利用ポリシーはプロジェクト内の契約・合意に従います。公開用途で利用する場合は別途確認してください。
