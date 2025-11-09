# Nova 単一プロジェクト Enhanced IRL 実験

## 目的

Attention なしモデルとの公平な比較のため、**同一データセット（nova 単一）**で Attention あり IRL モデルを評価する。

## 実験設定

### データ

- **プロジェクト**: `openstack/nova` のみ
- **元データ**: `review_requests_openstack_multi_5y_detail.csv`
- **フィルタ後**: 約 27,000 レコード、241 reviewers

### 期間

- **Train**: 2021-01-01 → 2023-01-01
- **Eval**: 2023-01-01 → 2024-01-01
- **履歴窓**: 12 ヶ月

### 予測窓組み合わせ

時系列的に有効な 10 組み合わせ:

1. 0-3m → 0-3m
2. 0-3m → 3-6m
3. 0-3m → 6-9m
4. 0-3m → 9-12m
5. 3-6m → 3-6m
6. 3-6m → 6-9m
7. 3-6m → 9-12m
8. 6-9m → 6-9m
9. 6-9m → 9-12m
10. 9-12m → 9-12m

### シード

5 シード: 42, 123, 777, 2024, 9999

**総実験数**: 5 シード × 10 組み合わせ = **50 実験**

## モデル構成

### Enhanced IRL (Attention あり)

- **State 特徴量**: 10 次元

  - 経験日数、総変更数、総レビュー数、プロジェクト数
  - 最近の活動頻度、平均活動間隔、活動トレンド
  - 協力度スコア、コード品質スコア、最終活動からの経過日数

- **Temporal 特徴量**: 97 次元

  - 30 日間の受け入れシーケンス (30 次元)
  - 30 日間のロードシーケンス (30 次元)
  - 30 日間の応答時間シーケンス (30 次元)
  - 7 日間の週次パターン (7 次元)

- **モデルアーキテクチャ**:
  - State Encoder: 10 → 64 次元
  - Temporal Encoder: 97 → 64 次元
  - LSTM: 128 次元 × 2 層
  - Attention 機構
  - Dropout: 0.3

### 訓練設定

- **Epochs**: 50
- **Batch size**: 32
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Binary Cross Entropy

## 比較対象

### Attention なしモデル（過去実験）

- **ソース**: `/Users/kazuki-h/rl/gerrit-retention/importants/review_acceptance_cross_eval_nova/`
- **データ**: nova 単一プロジェクト（同一）
- **平均 AUC-ROC**: **0.7536** (対角線)
- **評価方法**: クロス評価（4 訓練 × 4 評価）

### 期待される比較

同一データセットでの比較により、Attention 機構の純粋な効果を測定できる。

## 実行方法

```bash
# 実験実行（バックグラウンド）
cd /Users/kazuki-h/rl/gerrit-retention
nohup uv run python experiments/enhanced_irl_nova/scripts/run_nova_multi_seed.py > experiments/enhanced_irl_nova/nova_multi_seed.log 2>&1 &

# ステータス確認（推奨）
./experiments/enhanced_irl_nova/check_status.sh

# プロセス保護（停止時に自動再起動）
./experiments/enhanced_irl_nova/keep_alive.sh

# 継続的な監視（1分ごとに自動チェック）
watch -n 60 ./experiments/enhanced_irl_nova/check_status.sh

# ログをリアルタイム監視
tail -f experiments/enhanced_irl_nova/nova_multi_seed.log

# 手動でプロセス確認
ps aux | grep run_nova_multi_seed.py
```

### 保護機能

実験が長時間実行されるため、以下の保護措置を実装：

1. **PID ファイル**: `nova_experiment.pid` にプロセス ID を保存
2. **keep_alive.sh**: プロセスが停止した場合に自動再起動
3. **check_status.sh**: 実験の進捗状況を確認

```bash
# 定期的なヘルスチェック（crontabに登録可能）
*/10 * * * * /Users/kazuki-h/rl/gerrit-retention/experiments/enhanced_irl_nova/keep_alive.sh >> /tmp/nova_keepalive.log 2>&1
```

## 結果

### 実験状況

- **開始日時**: 2025-11-09 10:05 AM
- **PID**: 38507
- **ステータス**: 実行中 ⏳

### 最終結果（実験完了後）

結果ファイル: `experiments/enhanced_irl_nova/results/nova_multi_seed_results.csv`

比較分析は実験完了後に実施予定。

## ディレクトリ構造

```
experiments/enhanced_irl_nova/
├── README.md                          # このファイル
├── scripts/
│   └── run_nova_multi_seed.py        # 実験スクリプト
├── results/
│   └── nova_multi_seed_results.csv   # 結果（実験完了後）
├── nova_multi_seed.log                # 実行ログ
├── nova_experiment.pid                # プロセスID（保護用）
├── check_status.sh                    # ステータス確認スクリプト ✨
├── keep_alive.sh                      # プロセス保護スクリプト ✨
└── monitor_progress.sh                # 進捗モニター（廃止予定）
```

## 注意事項

- 複数プロジェクト版 (`experiments/enhanced_irl/`) とは異なるデータセット
- importants/ の Attention なしモデルと同一条件での比較が目的
- 実行時間: 約 3-4 時間（50 実験）
