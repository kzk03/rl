# サーバ実行ガイド

**クロス評価をサーバで長時間実行する方法**

---

## 🚀 クイックスタート

### 1. サーバにログイン

```bash
ssh your-server
cd /path/to/gerrit-retention  # ディレクトリ構成は同じ
```

### 2. スクリプトを実行可能にする

```bash
chmod +x scripts/training/irl/run_cross_eval_server.sh
```

### 3. 実行方法（3 つのオプション）

---

## 📋 実行オプション

### オプション A: nohup で実行（推奨）

**最もシンプル。SSH 切断後も実行継続。**

```bash
# 実行
nohup bash scripts/training/irl/run_cross_eval_server.sh > /tmp/cross_eval_nohup.log 2>&1 &

# プロセスIDを確認
echo $!

# ログ監視
tail -f /tmp/cross_eval_nohup.log

# 進捗確認（別のターミナル）
tail -50 outputs/cross_eval_simple/logs/main.log

# プロセス確認
ps aux | grep run_cross_eval_server
```

**Ctrl+C で監視を終了してもプロセスは継続します。**

---

### オプション B: screen で実行

**セッション管理が柔軟。**

```bash
# screenセッション開始
screen -S cross_eval

# スクリプト実行
bash scripts/training/irl/run_cross_eval_server.sh

# デタッチ（セッションから抜ける）: Ctrl+A, D

# セッション一覧確認
screen -ls

# セッションに再接続
screen -r cross_eval

# セッション終了（実行中のプロセスも終了）
# セッション内で Ctrl+C または exit
```

---

### オプション C: tmux で実行

**screen の高機能版。**

```bash
# tmuxセッション開始
tmux new -s cross_eval

# スクリプト実行
bash scripts/training/irl/run_cross_eval_server.sh

# デタッチ（セッションから抜ける）: Ctrl+B, D

# セッション一覧確認
tmux ls

# セッションに再接続
tmux attach -t cross_eval

# セッション終了
# セッション内で Ctrl+C または exit
```

---

## 📊 進捗確認方法

### 1. メインログを確認

```bash
# リアルタイム監視
tail -f outputs/cross_eval_simple/logs/main.log

# 最新50行
tail -50 outputs/cross_eval_simple/logs/main.log

# エラーのみ表示
grep "エラー\|Error\|失敗" outputs/cross_eval_simple/logs/main.log
```

### 2. 個別モデルのログ確認

```bash
# 特定モデルのログ
tail -f outputs/cross_eval_simple/logs/train_0-3m.log

# 全ログ一覧
ls -lh outputs/cross_eval_simple/logs/

# エポック進捗確認
grep "エポック" outputs/cross_eval_simple/logs/train_0-3m.log
```

### 3. 完了したモデルの確認

```bash
# メトリクスファイルの存在確認
ls -lh outputs/cross_eval_simple/*/metrics.json

# 完了数カウント
ls outputs/cross_eval_simple/*/metrics.json 2>/dev/null | wc -l

# メトリクス内容確認
cat outputs/cross_eval_simple/train_eval_0-3m/metrics.json | python3 -m json.tool
```

### 4. プロセス確認

```bash
# Pythonプロセス確認
ps aux | grep train_irl_per_timestep

# CPU/メモリ使用率確認
top -p $(pgrep -f train_irl_per_timestep)

# GPUを使用している場合
nvidia-smi  # GPU使用状況
watch -n 1 nvidia-smi  # 1秒ごとに更新
```

---

## 🛠️ トラブルシューティング

### プロセスが停止した場合

```bash
# プロセス確認
ps aux | grep run_cross_eval_server

# 停止していた場合、再開
# まず、どこまで完了したか確認
ls outputs/cross_eval_simple/*/metrics.json

# 未完了のモデルだけ再実行（スクリプト編集が必要）
# または全体を再実行
nohup bash scripts/training/irl/run_cross_eval_server.sh > /tmp/cross_eval_nohup.log 2>&1 &
```

### メモリ不足エラー

```bash
# メモリ使用状況確認
free -h

# スワップ使用状況
swapon --show

# エポック数を減らす（スクリプト編集）
# run_cross_eval_server.sh の EPOCHS=20 を EPOCHS=10 に変更
```

### ディスク容量不足

```bash
# ディスク使用量確認
df -h

# outputsディレクトリのサイズ
du -sh outputs/

# 古い実験結果を削除
rm -rf outputs/old_experiment_name
```

### SSH 切断されてしまった

```bash
# nohupで実行していれば問題なし
# プロセス確認
ps aux | grep run_cross_eval_server

# ログで進捗確認
tail -f /tmp/cross_eval_nohup.log
```

---

## ⏱️ 実行時間の見積もり

### 基準値（OpenStack データ、355 レビュアー）

| エポック数 | 1 モデル | 5 モデル合計 |
| ---------- | -------- | ------------ |
| 5          | 15 分    | 75 分        |
| 10         | 30 分    | 2.5 時間     |
| 20         | 60 分    | 5 時間       |
| 30         | 90 分    | 7.5 時間     |

**注意**: データ量やサーバスペックにより大きく変動します。

---

## 📧 通知設定（オプション）

### メール通知

```bash
# 完了時にメールを送信
nohup bash -c "
bash scripts/training/irl/run_cross_eval_server.sh && \
echo '実験完了' | mail -s 'クロス評価完了' your-email@example.com
" > /tmp/cross_eval_nohup.log 2>&1 &
```

### Slack に通知

```bash
# 完了時にSlackに通知
nohup bash -c "
bash scripts/training/irl/run_cross_eval_server.sh && \
curl -X POST -H 'Content-type: application/json' \
--data '{\"text\":\"クロス評価完了！\"}' \
YOUR_SLACK_WEBHOOK_URL
" > /tmp/cross_eval_nohup.log 2>&1 &
```

---

## 📁 ファイル構成

実行後のディレクトリ構成:

```
outputs/cross_eval_simple/
├── logs/
│   ├── main.log               # メインログ（全体の進捗）
│   ├── train_0-1m.log         # 0-1mモデルの詳細ログ
│   ├── train_0-3m.log         # 0-3mモデルの詳細ログ
│   ├── train_0-6m.log         # 0-6mモデルの詳細ログ
│   ├── train_0-9m.log         # 0-9mモデルの詳細ログ
│   └── train_0-12m.log        # 0-12mモデルの詳細ログ
├── train_eval_0-1m/
│   ├── irl_model.pt
│   ├── metrics.json
│   └── training.log
├── train_eval_0-3m/
│   └── ... (同様)
├── train_eval_0-6m/
│   └── ... (同様)
├── train_eval_0-9m/
│   └── ... (同様)
├── train_eval_0-12m/
│   └── ... (同様)
└── summary.csv                # 集計結果
```

---

## ✅ 実行チェックリスト

### 実行前

- [ ] サーバにログインした
- [ ] ディレクトリ構成が正しい
- [ ] データファイルが存在する (`data/review_requests_openstack_multi_5y_detail.csv`)
- [ ] 十分なディスク容量がある（約 10GB）
- [ ] スクリプトが実行可能 (`chmod +x`)

### 実行中

- [ ] プロセスが起動している
- [ ] ログが出力されている
- [ ] エラーが出ていないか定期確認
- [ ] ディスク容量を定期確認

### 実行後

- [ ] 全 5 モデルが完了
- [ ] metrics.json が全て生成されている
- [ ] summary.csv で結果を確認
- [ ] 最高性能のモデルを特定

---

## 🎯 サーバ実行のベストプラクティス

### 1. 長時間実行の準備

```bash
# ディスク容量確認
df -h

# メモリ確認
free -h

# 実行権限確認
chmod +x scripts/training/irl/run_cross_eval_server.sh

# テストラン（エポック1で動作確認）
# run_cross_eval_server.sh の EPOCHS を 1 に変更して実行
```

### 2. 実行開始

```bash
# nohupで実行（SSH切断に強い）
nohup bash scripts/training/irl/run_cross_eval_server.sh > /tmp/cross_eval_nohup.log 2>&1 &

# プロセスIDを記録
echo $! > /tmp/cross_eval.pid
```

### 3. 定期確認

```bash
# 進捗確認（1日1-2回）
tail -50 outputs/cross_eval_simple/logs/main.log

# プロセス確認
cat /tmp/cross_eval.pid | xargs ps -p

# 完了モデル数確認
ls outputs/cross_eval_simple/*/metrics.json 2>/dev/null | wc -l
```

### 4. 結果取得

```bash
# ローカルにダウンロード（実行後）
scp -r your-server:/path/to/gerrit-retention/outputs/cross_eval_simple ./outputs/

# または summary.csv のみ
scp your-server:/path/to/gerrit-retention/outputs/cross_eval_simple/summary.csv ./
```

---

## 📞 ヘルプ

### よくある質問

**Q: SSH 切断しても大丈夫？**  
A: `nohup`、`screen`、`tmux` いずれかを使えば大丈夫です。

**Q: 途中で停止させるには？**  
A: プロセス ID を確認して `kill <PID>` を実行。

**Q: 途中から再開できる？**  
A: 現状は全体を再実行する必要があります。完了したモデルはスキップされません。

**Q: エポック数を途中で変更できる？**  
A: プロセスを停止してから `run_cross_eval_server.sh` の `EPOCHS` を変更して再実行。

---

## 🚀 今すぐ実行

```bash
# サーバにログイン
ssh your-server
cd /path/to/gerrit-retention

# 実行可能にする
chmod +x scripts/training/irl/run_cross_eval_server.sh

# 実行（nohup推奨）
nohup bash scripts/training/irl/run_cross_eval_server.sh > /tmp/cross_eval_nohup.log 2>&1 &

# プロセスID確認
echo $!

# ログ監視（Ctrl+Cで終了しても実行は継続）
tail -f /tmp/cross_eval_nohup.log
```

**実行が完了したら:**

```bash
# 結果確認
cat outputs/cross_eval_simple/summary.csv

# ローカルにダウンロード
scp -r your-server:/path/to/gerrit-retention/outputs/cross_eval_simple ./outputs/
```
