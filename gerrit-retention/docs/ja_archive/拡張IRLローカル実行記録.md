# 拡張 IRL ローカル実行記録

## 📊 実行情報

### 実行日時

- **開始**: 2025-10-25 16:02:38
- **完了予定**: 2025-10-25 18:30（約 2.5 時間後 / 評価のみ 16 個）

### プロセス情報

- **メインプロセス ID**: 18253
- **caffeinate PID**: 18261（`-i -s -w` オプションで起動）
- **コマンド**: `uv run python scripts/training/irl/run_enhanced_cross_evaluation.py`
- **ログファイル**: `/tmp/enhanced_cross_eval_success.log`

### 修正内容

1. ✓ pandas 再インストール（2.3.3）
2. ✓ モデルパス修正: `enhanced_irl_model.pt`
3. ✓ 評価スクリプト修正: `train_enhanced_irl_per_timestep_labels.py`
4. ✓ `--model` オプション追加
5. ✓ インポートパス修正: `gerrit_retention`
6. ✓ `torch.load`: `weights_only=False`

### 重要な更新（2025-10-24 21:50）

- ⚠️ **実行時間の大幅な見直し**:

  - 1 エポック ≈ 3.5-4 時間（予想の約 10 倍）
  - 1 モデル（20 エポック） ≈ 70-80 時間（約 3 日）
  - 全体（5 モデル） ≈ 350-400 時間（約 15-17 日）

- ✅ **caffeinate 再設定**:
  - 旧: `caffeinate -i -w 68194`（-s なし → 蓋を閉じるとスリープ）
  - 新: `caffeinate -dims -w 68194`（全オプション → 蓋を閉じても OK）

---

## 🔒 保護設定

### 実施済み対策

1. ✅ **nohup 実行**

   - ターミナルを閉じても継続実行
   - ログを `/tmp/enhanced_cross_eval_py.log` に保存

2. ✅ **caffeinate 実行中**
   - Mac がスリープしない
   - プロセス 68194 が終了するまで継続
   - PID: 68687

### 推奨事項

- ⚠️ **電源を接続してください**（バッテリーのみは危険）
- ⚠️ 約 20 時間実行予定
- ⚠️ ディスク容量を確認

---

## 📋 実行内容

### 訓練設定

- **訓練ラベル**: 5 個（0-1m, 0-3m, 0-6m, 0-9m, 0-12m）
- **評価期間**: 4 個（0-3m, 3-6m, 6-9m, 9-12m）
- **総評価数**: 20 個（5 訓練 × 4 評価）
- **エポック数**: 20

### データ

- **データセット**: `data/review_requests_openstack_multi_5y_detail.csv`
- **訓練期間**: 2021-01-01 ～ 2023-01-01
- **評価期間**: 2023-01-01 ～ 2024-01-01
- **出力先**: `outputs/enhanced_cross_eval/`

---

## 🔍 進捗確認方法

### リアルタイム監視

```bash
tail -f /tmp/enhanced_cross_eval_py.log
```

### 最新状況確認

```bash
# 最新30行
tail -30 /tmp/enhanced_cross_eval_py.log

# プロセス確認
ps -p 68194

# caffeinate確認
ps -p 68687
```

### 完了数確認

```bash
# 完了モデル数（期待: 5）
ls outputs/enhanced_cross_eval/train_*/irl_model.pt 2>/dev/null | wc -l

# 完了評価数（期待: 20）
ls outputs/enhanced_cross_eval/train_*/eval_*/metrics.json 2>/dev/null | wc -l
```

---

## 🛑 停止方法

### 通常停止

```bash
# メインプロセス停止
kill 68194

# caffeinate停止
kill $(cat /tmp/caffeinate_enhanced_irl.pid)
```

### 強制停止

```bash
# 強制停止
kill -9 68194

# caffeinate強制停止
kill -9 $(cat /tmp/caffeinate_enhanced_irl.pid)
```

---

## 📊 推定タイムライン

| 時刻  | イベント                     |
| ----- | ---------------------------- |
| 18:08 | 実行開始                     |
| 22:00 | 0-1m 完了予定（約 4 時間）   |
| 02:00 | 0-3m 完了予定（約 8 時間）   |
| 06:00 | 0-6m 完了予定（約 12 時間）  |
| 10:00 | 0-9m 完了予定（約 16 時間）  |
| 14:00 | 0-12m 完了予定（約 20 時間） |

---

## 📁 期待される出力

```
outputs/enhanced_cross_eval/
├── logs/
│   ├── main.log
│   ├── train_0-1m.log
│   ├── train_0-1m_eval_0-3m.log
│   └── ...（計25ログファイル）
├── train_0-1m/
│   ├── irl_model.pt
│   ├── predictions.csv
│   ├── metrics.json
│   ├── eval_0-3m/
│   ├── eval_3-6m/
│   ├── eval_6-9m/
│   └── eval_9-12m/
├── train_0-3m/（同上）
├── train_0-6m/（同上）
├── train_0-9m/（同上）
└── train_0-12m/（同上）
```

---

## ⚠️ トラブルシューティング

### プロセスが停止している場合

```bash
# 状態確認
ps -p 68194

# ログで原因確認
tail -50 /tmp/enhanced_cross_eval_py.log | grep -i "error\|exception"
```

### Mac がスリープした場合

```bash
# caffeinate再起動
caffeinate -i -w 68194 &
echo $! > /tmp/caffeinate_enhanced_irl.pid
```

### ディスク容量不足

```bash
# 容量確認
df -h .

# 不要ファイル削除
rm -rf outputs/enhanced_cross_eval/
```

---

## 📝 メモ

### 特徴量の違い

| 項目        | 通常 IRL | 拡張 IRL           |
| ----------- | -------- | ------------------ |
| State 次元  | 10       | 32                 |
| Action 次元 | 5        | 9                  |
| Hidden 次元 | 128      | 256                |
| 実行時間    | 17 時間  | 18-20 時間（推定） |

### 実行結果の比較予定

拡張 IRL 完了後、通常 IRL との性能比較を実施予定:

- AUC-ROC 比較
- F1 スコア比較
- 特徴量重要度の違い

---

**作成日時**: 2025-10-24 18:11
**更新日時**: 2025-10-24 18:11
