# Python 版クロス評価実行ガイド

## 🎯 概要

シェルスクリプト版の`run_enhanced_cross_evaluation.sh`を Python で実行できるようにしました。

### メリット

- ✅ **クロスプラットフォーム**: Windows/Mac/Linux で動作
- ✅ **進捗管理**: リアルタイムで進捗を確認可能
- ✅ **エラーハンドリング**: より詳細なエラー情報
- ✅ **スキップ機能**: 既存モデルは自動スキップ
- ✅ **統計情報**: 実行時間や成功/失敗数を自動集計

---

## 📋 使用方法

### 1. 基本実行

```bash
# ローカルで実行
python scripts/training/irl/run_enhanced_cross_evaluation.py

# または uv run
uv run python scripts/training/irl/run_enhanced_cross_evaluation.py
```

### 2. サーバーでバックグラウンド実行

```bash
# サーバーにログイン
docker compose exec app bash

# nohup で実行
nohup python scripts/training/irl/run_enhanced_cross_evaluation.py \
  > /tmp/enhanced_cross_eval.log 2>&1 &

# または
nohup uv run python scripts/training/irl/run_enhanced_cross_evaluation.py \
  > /tmp/enhanced_cross_eval.log 2>&1 &
```

### 3. カスタム設定で実行

```bash
# エポック数を変更
python scripts/training/irl/run_enhanced_cross_evaluation.py \
  --epochs 30

# 出力先を変更
python scripts/training/irl/run_enhanced_cross_evaluation.py \
  --output outputs/my_enhanced_eval

# 訓練期間を変更
python scripts/training/irl/run_enhanced_cross_evaluation.py \
  --train-start 2020-01-01 \
  --train-end 2022-01-01
```

### 4. ドライラン（テスト実行）

```bash
# 実際には実行せず、コマンドだけ確認
python scripts/training/irl/run_enhanced_cross_evaluation.py --dry-run
```

---

## ⚙️ オプション一覧

| オプション         | デフォルト値                                         | 説明                     |
| ------------------ | ---------------------------------------------------- | ------------------------ |
| `--reviews`        | `data/review_requests_openstack_multi_5y_detail.csv` | レビューデータファイル   |
| `--train-start`    | `2021-01-01`                                         | 訓練開始日               |
| `--train-end`      | `2023-01-01`                                         | 訓練終了日               |
| `--eval-start`     | `2023-01-01`                                         | 評価開始日               |
| `--eval-end`       | `2024-01-01`                                         | 評価終了日               |
| `--history-window` | `12`                                                 | 履歴ウィンドウ（月）     |
| `--epochs`         | `20`                                                 | 訓練エポック数           |
| `--output`         | `outputs/enhanced_cross_eval`                        | 出力ディレクトリ         |
| `--dry-run`        | -                                                    | ドライラン（実行しない） |

---

## 📊 進捗確認

### リアルタイムログ監視（サーバー）

```bash
# ログをリアルタイム表示
tail -f /tmp/enhanced_cross_eval.log

# または個別ログ
tail -f outputs/enhanced_cross_eval/logs/main.log
```

### 完了状況確認

```bash
# Pythonスクリプトが自動で表示するサマリー
# 実行中は以下で確認可能

# 完了モデル数
ls outputs/enhanced_cross_eval/train_*/irl_model.pt | wc -l

# 完了評価数
ls outputs/enhanced_cross_eval/train_*/eval_*/metrics.json | wc -l
```

---

## 🔄 シェル版との比較

### シェルスクリプト版

```bash
nohup bash scripts/training/irl/run_enhanced_cross_evaluation.sh \
  > /tmp/enhanced_cross_eval.log 2>&1 &
```

### Python 版

```bash
nohup python scripts/training/irl/run_enhanced_cross_evaluation.py \
  > /tmp/enhanced_cross_eval.log 2>&1 &
```

**主な違い**:

| 機能                   | シェル版 | Python 版 |
| ---------------------- | -------- | --------- |
| クロスプラットフォーム | ❌       | ✅        |
| 進捗表示               | 基本     | 詳細      |
| エラーハンドリング     | 基本     | 詳細      |
| 統計情報               | なし     | あり      |
| 既存モデルスキップ     | なし     | あり      |
| ドライラン             | なし     | あり      |

---

## 📁 出力構造

```
outputs/enhanced_cross_eval/
├── logs/
│   ├── main.log                           # メインログ
│   ├── train_0-1m.log                     # 訓練ログ
│   ├── train_0-1m_eval_0-3m.log          # 評価ログ
│   └── ...
├── train_0-1m/
│   ├── irl_model.pt                       # 訓練済みモデル
│   ├── predictions.csv
│   ├── metrics.json
│   ├── eval_0-3m/                         # 評価結果
│   │   ├── predictions.csv
│   │   └── metrics.json
│   └── ...
├── train_0-3m/（同上）
├── train_0-6m/（同上）
├── train_0-9m/（同上）
└── train_0-12m/（同上）
```

---

## 💡 便利な使い方

### 1. 途中から再開

既存のモデルがある場合は自動的にスキップされるため、途中で停止しても再実行すれば続きから実行されます。

```bash
# 途中で停止した場合
# そのまま再実行すれば、完了済みの訓練/評価はスキップされる
python scripts/training/irl/run_enhanced_cross_evaluation.py
```

### 2. 特定のエポック数でテスト

```bash
# エポック数を減らして動作確認
python scripts/training/irl/run_enhanced_cross_evaluation.py --epochs 5
```

### 3. 異なる期間で実験

```bash
# 別の訓練期間で実験
python scripts/training/irl/run_enhanced_cross_evaluation.py \
  --train-start 2020-01-01 \
  --train-end 2022-01-01 \
  --output outputs/enhanced_cross_eval_2020_2022
```

---

## 🛑 実行を停止

### プロセス ID で停止

```bash
# プロセスを確認
ps aux | grep run_enhanced_cross_evaluation.py

# プロセスIDを指定して停止
kill <PID>
```

### 強制停止

```bash
# Python版
pkill -f run_enhanced_cross_evaluation.py

# または Ctrl+C（フォアグラウンド実行の場合）
```

---

## 🔍 トラブルシューティング

### エラーが発生した場合

```bash
# 個別ログを確認
tail -50 outputs/enhanced_cross_eval/logs/train_0-*m.log

# エラーメッセージを検索
grep -i "error\|traceback" outputs/enhanced_cross_eval/logs/main.log
```

### 既存モデルを削除して再実行

```bash
# 特定モデルを削除
rm -rf outputs/enhanced_cross_eval/train_0-12m/

# 全て削除して最初から
rm -rf outputs/enhanced_cross_eval/
```

---

## 📊 実行例

### 基本実行（ローカル）

```bash
$ python scripts/training/irl/run_enhanced_cross_evaluation.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
拡張IRL完全クロス評価
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2025-10-24 16:30:00] 開始時刻: 2025-10-24 16:30:00

[2025-10-24 16:30:00] 訓練ラベル: 5個 (0-1m, 0-3m, 0-6m, 0-9m, 0-12m)
[2025-10-24 16:30:00] 評価期間: 4個 (0-3m, 3-6m, 6-9m, 9-12m)
[2025-10-24 16:30:00] 総評価数: 20個
...
```

### ドライラン

```bash
$ python scripts/training/irl/run_enhanced_cross_evaluation.py --dry-run

[2025-10-24 16:30:00] [DRY RUN] コマンド: uv run python scripts/training/irl/train_enhanced_irl_per_timestep_labels.py ...
[2025-10-24 16:30:00] ✓ 拡張IRL訓練完了: 0-1m
...
```

---

## 📌 注意事項

1. **Python 3.8 以上が必要**
2. **`uv`コマンドが必要** （または通常の Python 環境）
3. **既存モデルは自動スキップ** されるため、削除してから再実行する場合は手動で削除が必要
4. **バックグラウンド実行時** は `nohup` を使用してログを保存

---

## 🎯 まとめ

| コマンド                                                                                   | 用途                 |
| ------------------------------------------------------------------------------------------ | -------------------- |
| `python scripts/training/irl/run_enhanced_cross_evaluation.py`                             | 基本実行             |
| `python scripts/training/irl/run_enhanced_cross_evaluation.py --dry-run`                   | テスト実行           |
| `python scripts/training/irl/run_enhanced_cross_evaluation.py --epochs 30`                 | カスタム設定         |
| `nohup python scripts/training/irl/run_enhanced_cross_evaluation.py > /tmp/log.log 2>&1 &` | バックグラウンド実行 |

---

**作成日時**: 2025-10-24 16:30
