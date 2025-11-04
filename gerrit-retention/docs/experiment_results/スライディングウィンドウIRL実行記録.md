# スライディングウィンドウ IRL 実行記録

## 概要

訓練ラベルを独立したスライディングウィンドウで定義し、従来版（累積ラベル）との違いを検証。

- **開始時刻**: 2025-10-26 21:26
- **実行方式**: 複数プロジェクト + nova 単一を並列実行
- **推定完了**: 2025-10-27 00:30 頃（約 3 時間）

---

## 訓練ラベルの違い

### 従来版（累積ラベル）

```
0-1m ⊂ 0-3m ⊂ 0-6m ⊂ 0-9m ⊂ 0-12m
```

- **問題点**: ラベルが重複（包含関係）
- **例**: 2 ヶ月後に活動 → 0-3m, 0-6m, 0-9m, 0-12m 全て True

### スライディング版（今回）

```
0-3m, 3-6m, 6-9m, 9-12m （独立）
```

- **特徴**: 各ラベルが独立
- **例**: 2 ヶ月後に活動 → 0-3m のみ True、他は False

---

## 実行設定

### 共通設定

- **データ**: `data/review_requests_openstack_multi_5y_detail.csv`
- **学習期間**: 2021-01-01 ～ 2023-01-01
- **評価期間**: 2023-01-01 ～ 2024-01-01
- **エポック数**: 20
- **min-history-events**: 1

### 訓練ラベル（4 個、スライディング）

- **0-3m**: 0 ～ 3 ヶ月後に活動
- **3-6m**: 3 ～ 6 ヶ月後に活動（0-3m は除く）
- **6-9m**: 6 ～ 9 ヶ月後に活動（0-6m は除く）
- **9-12m**: 9 ～ 12 ヶ月後に活動（0-9m は除く）

### 評価期間（4 個、スライディング）

- **0-3m**: cutoff から 0 ～ 3 ヶ月後
- **3-6m**: cutoff から 3 ～ 6 ヶ月後
- **6-9m**: cutoff から 6 ～ 9 ヶ月後
- **9-12m**: cutoff から 9 ～ 12 ヶ月後

### 合計評価数

- **4 訓練 × 4 評価 = 16 評価** × 2（複数 + nova） = **32 評価**

---

## プロセス情報

### 複数プロジェクト

- **PID**: 64088
- **ログ**: `/tmp/sliding_multi.log`
- **出力**: `outputs/sliding_cross_eval/`
- **特徴**: 全プロジェクト対象

### nova 単一

- **PID**: 64127
- **ログ**: `/tmp/sliding_nova.log`
- **出力**: `outputs/sliding_cross_eval_nova/`
- **特徴**: openstack/nova のみ

### caffeinate 監視

- **PID**: 64164
- **オプション**: `-s -w 64088 64127`

---

## 進捗確認

### リアルタイムログ

```bash
# 複数プロジェクト
tail -f /tmp/sliding_multi.log

# nova単一
tail -f /tmp/sliding_nova.log
```

### プロセス確認

```bash
ps aux | grep -E "64088|64127"
```

---

## 期待される結果

### 訓練時の継続率

各訓練ラベルで異なる継続率が期待される：

| 訓練ラベル | 予想継続率 | 理由             |
| ---------- | ---------- | ---------------- |
| 0-3m       | 高い       | 短期継続は多い   |
| 3-6m       | 中程度     | 0-3m より低い    |
| 6-9m       | 低い       | 長期継続は少ない |
| 9-12m      | 最も低い   | 最も長期         |

### クロス評価結果

**対角線（train=eval）で高性能が期待される：**

| Train↓ / Eval→ | 0-3m   | 3-6m   | 6-9m   | 9-12m  |
| -------------- | ------ | ------ | ------ | ------ |
| 0-3m           | **高** | 中     | 低     | 低     |
| 3-6m           | 中     | **高** | 中     | 低     |
| 6-9m           | 低     | 中     | **高** | 中     |
| 9-12m          | 低     | 低     | 中     | **高** |

**理由:** 訓練時の時間スケールと評価時の時間スケールが一致する場合に最も性能が高いはず。

---

## 実行後の分析

### サマリー生成

```bash
# 複数プロジェクト
uv run python scripts/training/irl/summarize_full_cross_evaluation.py outputs/sliding_cross_eval

# nova単一
uv run python scripts/training/irl/summarize_full_cross_evaluation.py outputs/sliding_cross_eval_nova
```

### ヒートマップ生成

```bash
# 複数プロジェクト
uv run python scripts/analysis/visualize_cross_evaluation.py outputs/sliding_cross_eval

# nova単一
uv run python scripts/analysis/visualize_cross_evaluation.py outputs/sliding_cross_eval_nova
```

### 従来版との比較

```bash
# 複数プロジェクト: 従来 vs スライディング
uv run python scripts/analysis/compare_irl_heatmaps.py \
    --normal outputs/full_cross_eval \
    --enhanced outputs/sliding_cross_eval \
    --output outputs/cumulative_vs_sliding

# nova単一: 従来 vs スライディング
uv run python scripts/analysis/compare_irl_heatmaps.py \
    --normal outputs/full_cross_eval_nova \
    --enhanced outputs/sliding_cross_eval_nova \
    --output outputs/cumulative_vs_sliding_nova
```

---

## 新規作成ファイル

1. **scripts/training/irl/train_irl_sliding_window.py**

   - スライディングウィンドウ版データ抽出関数
   - 訓練・評価スクリプト

2. **scripts/training/irl/run_sliding_cross_evaluation.sh**

   - 複数プロジェクト用実行スクリプト

3. **scripts/training/irl/run_sliding_cross_evaluation_nova.sh**
   - nova 単一用実行スクリプト

---

## 完了チェックリスト

- [ ] 複数プロジェクト実行完了（16 評価）
- [ ] nova 単一実行完了（16 評価）
- [ ] 複数プロジェクトサマリー生成
- [ ] nova 単一サマリー生成
- [ ] 複数プロジェクトヒートマップ生成
- [ ] nova 単一ヒートマップ生成
- [ ] 従来版との比較分析

---

## メモ

### 仮説

- **訓練ラベルが独立** → より明確な時間スケールの学習
- **対角線で高性能** → 時間スケールの一致が重要
- **従来版より安定** → ラベルの重複がなく、学習が安定

### 注意点

- 継続率の大幅な変化が予想される
- 特に 3-6m, 6-9m, 9-12m は継続率が低い可能性
- データ不足による不安定性に注意

---

## 実行履歴

### 初回実行

- **開始**: 2025-10-26 21:26
- **推定完了**: 2025-10-27 00:30 頃
- **状態**: 実行中 ✓
