# 拡張 IRL クロス評価 進捗確認方法

## 🚀 実行コマンド（確認用）

```bash
kazuki-h@50d4d3445737:~/workspace$ nohup bash scripts/training/irl/run_enhanced_cross_evaluation.sh \
  > /tmp/enhanced_cross_eval.log 2>&1 &
[1] 194
```

---

## 📊 進捗確認方法

### 1. メインログをリアルタイム監視

```bash
# サーバーにログイン
docker compose exec app bash

# ログをリアルタイム表示
tail -f /tmp/enhanced_cross_eval.log
```

**Ctrl+C で終了**

---

### 2. 最新の進捗を確認

```bash
# 最新20行を表示
tail -20 /tmp/enhanced_cross_eval.log

# または最新50行
tail -50 /tmp/enhanced_cross_eval.log
```

---

### 3. プロセスが実行中か確認

```bash
# プロセス確認
ps aux | grep run_enhanced_cross_evaluation | grep -v grep

# または
ps aux | grep enhanced | grep -v grep
```

**出力があれば実行中、なければ終了済み**

---

### 4. 出力ディレクトリの状況確認

```bash
# 完成したモデル数を確認
ls outputs/enhanced_cross_eval/train_*/irl_model.pt 2>/dev/null | wc -l

# 完成した評価数を確認
ls outputs/enhanced_cross_eval/train_*/eval_*/metrics.json 2>/dev/null | wc -l

# ディレクトリ構造を確認
ls -la outputs/enhanced_cross_eval/
```

**期待値**: 5 モデル、20 評価（5 訓練 × 4 評価期間）

---

### 5. 個別ログを確認

```bash
# 個別ログディレクトリに移動
cd outputs/enhanced_cross_eval/logs/

# メインログ
tail -10 main.log

# 特定の訓練ログ（例: 0-12m）
tail -20 train_0-12m_eval_0-3m.log

# 全ログファイル一覧
ls -lh *.log
```

---

## 📋 進捗チェックリスト

### 訓練フェーズ（5 モデル）

```bash
# 各訓練の進捗を一括確認
for label in 0-1m 0-3m 0-6m 0-9m 0-12m; do
  echo "【${label}】"
  if [ -f "outputs/enhanced_cross_eval/train_${label}/irl_model.pt" ]; then
    echo "  ✓ 訓練完了"
  else
    echo "  ⏳ 訓練中または未開始"
  fi
done
```

### 評価フェーズ（20 評価）

```bash
# 完了した評価数を確認
EVAL_COUNT=$(ls outputs/enhanced_cross_eval/train_*/eval_*/metrics.json 2>/dev/null | wc -l)
echo "完了: ${EVAL_COUNT}/20 評価"
```

---

## 🔍 詳細分析コマンド

### エポック進捗を確認

```bash
# 最新のエポック情報を抽出
grep "Epoch" /tmp/enhanced_cross_eval.log | tail -10

# または
grep "平均損失" /tmp/enhanced_cross_eval.log | tail -10
```

### エラーチェック

```bash
# エラーがないか確認
grep -i "error\|traceback\|exception" /tmp/enhanced_cross_eval.log

# または
grep -i "エラー\|失敗" /tmp/enhanced_cross_eval.log
```

### 実行時間を推定

```bash
# 開始時刻を確認
head -20 /tmp/enhanced_cross_eval.log | grep "開始\|Start"

# 現在の進捗を確認
tail -20 /tmp/enhanced_cross_eval.log
```

---

## ⚡ クイック確認コマンド（コピペ用）

```bash
# サーバーログイン → 進捗確認（ワンライナー）
docker compose exec app bash -c "
echo '【プロセス状態】'
ps aux | grep enhanced | grep -v grep || echo '停止中'
echo ''
echo '【完了モデル】'
ls outputs/enhanced_cross_eval/train_*/irl_model.pt 2>/dev/null | wc -l | xargs echo 'モデル数:'
echo ''
echo '【最新ログ（10行）】'
tail -10 /tmp/enhanced_cross_eval.log
"
```

---

## 🛑 実行を停止したい場合

```bash
# プロセスIDを確認
ps aux | grep run_enhanced_cross_evaluation | grep -v grep

# プロセスを停止（PIDを確認してから）
kill <PID>

# または強制停止
kill -9 <PID>

# 関連プロセスを全て停止
pkill -f run_enhanced_cross_evaluation
```

---

## 📊 完了確認

### 全て完了したか確認

```bash
# モデル数確認（期待: 5）
ls outputs/enhanced_cross_eval/train_*/irl_model.pt 2>/dev/null | wc -l

# 評価数確認（期待: 20）
ls outputs/enhanced_cross_eval/train_*/eval_*/metrics.json 2>/dev/null | wc -l

# プロセス確認（期待: 何も表示されない）
ps aux | grep enhanced | grep -v grep
```

**全て完了していれば**:

- モデル数: 5
- 評価数: 20
- プロセス: なし

---

## 📁 ログファイルの場所

| ログタイプ     | パス                                           |
| -------------- | ---------------------------------------------- |
| メインログ     | `/tmp/enhanced_cross_eval.log`                 |
| スクリプトログ | `outputs/enhanced_cross_eval/logs/main.log`    |
| 個別訓練ログ   | `outputs/enhanced_cross_eval/logs/train_*.log` |

---

## ⏱️ 推定実行時間

**通常 IRL の実績**: 17 時間（5 モデル × 4 評価）

**拡張 IRL**: 特徴量が多い（32 次元 vs 10 次元）ため、やや長くなる可能性

- **推定**: 18-20 時間
- **各モデル**: 約 3.5-4 時間

---

## 💡 トラブルシューティング

### ログが見つからない

```bash
# ログファイルが存在するか確認
ls -la /tmp/enhanced_cross_eval.log

# 別の場所を確認
find . -name "*enhanced*log*" -type f
```

### プロセスが見つからないが完了していない

```bash
# nohup.outを確認
cat nohup.out

# システムログを確認
dmesg | tail -50
```

### 途中で止まっている場合

```bash
# 最後の更新時刻を確認
ls -lt /tmp/enhanced_cross_eval.log

# 最新10行で状態確認
tail -10 /tmp/enhanced_cross_eval.log
```

---

**作成日時**: 2025-10-24 16:30
