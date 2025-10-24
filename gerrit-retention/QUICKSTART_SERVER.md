# サーバ実行クイックスタート

**30 秒で始めるクロス評価**

---

## 🚀 コピペで実行

### Step 1: サーバにログイン

```bash
ssh your-server
cd /path/to/gerrit-retention
```

### Step 2: ワンライナー実行

```bash
chmod +x scripts/training/irl/run_cross_eval_server.sh && \
nohup bash scripts/training/irl/run_cross_eval_server.sh > /tmp/cross_eval.log 2>&1 & \
echo "PID: $!" | tee /tmp/cross_eval.pid && \
sleep 3 && tail -f /tmp/cross_eval.log
```

**これで実行開始！Ctrl+C でログ監視終了（実行は継続）**

---

## 📊 進捗確認

```bash
# メインログ
tail -50 outputs/cross_eval_simple/logs/main.log

# 完了数（5個になれば完了）
ls outputs/cross_eval_simple/*/metrics.json 2>/dev/null | wc -l
```

---

## 📥 結果取得

```bash
# サーバで結果確認
cat outputs/cross_eval_simple/summary.csv

# ローカルにダウンロード（別ターミナル）
scp -r your-server:/path/to/gerrit-retention/outputs/cross_eval_simple ./outputs/
```

---

## ⏱️ 実行時間

- **エポック 5**: 約 75 分
- **エポック 10**: 約 2.5 時間
- **エポック 20**: 約 5 時間 ← デフォルト

---

## 🔧 トラブル時

```bash
# プロセス確認
ps aux | grep run_cross_eval_server

# 停止
cat /tmp/cross_eval.pid | xargs kill

# 強制停止
pkill -f run_cross_eval_server
```

---

## 📚 詳細は

- [SERVER_EXECUTION_GUIDE.md](./SERVER_EXECUTION_GUIDE.md) - 詳細な実行ガイド
- [docs/クロス評価実行ガイド.md](./docs/クロス評価実行ガイド.md) - 設定と結果の見方
- [docs/全シーケンス月次集約ラベル実験結果.md](./docs/全シーケンス月次集約ラベル実験結果.md) - 実験結果

---

## ✅ 期待される結果

```
訓練ラベル  AUC-ROC  F1
0-1m       0.70+    0.65+
0-3m       0.74     0.69   ← 基準値
0-6m       0.75+    0.70+
0-9m       0.75+    0.70+
0-12m      0.75+    0.70+
```
