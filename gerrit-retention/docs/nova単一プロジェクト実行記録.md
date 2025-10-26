# nova 単一プロジェクト実行記録

## 概要

openstack/nova プロジェクト単一での完全クロス評価を実行。

- **初回実行開始**: 2025-10-26 17:31:10（エラーにより中断）
- **修正版実行開始**: 2025-10-26 17:53:19
- **プロジェクト**: openstack/nova
- **実行内容**: 通常 IRL + 拡張 IRL を並列実行
- **推定完了時刻**: 2025-10-26 22:30 頃（4-5 時間）

### 修正履歴

**初回実行（17:31-17:50）**

- 問題: `train_irl_per_timestep_labels.py`に`--model`引数がなく、評価時にエラー
- 対応: `--model`引数とモデルロード処理を追加

**修正版実行（17:53-18:40）**

- 評価も含めて全て正常に動作するように修正
- エラー: モデルロード時の形式不一致（`KeyError: 'model_state_dict'`）
- 対応: `RetentionIRLSystem.load_model`を修正（state_dict 直接保存形式に対応）

**再修正版実行（18:54-）**

- モデルロード処理を両方のフォーマットに対応
- 通常 IRL: 再実行開始（18:54:32）
- 拡張 IRL: 継続中（正常）

---

## 実行設定

### 共通設定

- **データ**: `data/review_requests_openstack_multi_5y_detail.csv`
- **プロジェクトフィルタ**: `openstack/nova`
- **学習期間**: 2021-01-01 ～ 2023-01-01
- **評価期間**: 2023-01-01 ～ 2024-01-01
- **履歴ウィンドウ**: 12 ヶ月
- **エポック数**: 20
- **min-history-events**: 1

### 訓練ラベル（5 個）

- 0-1m
- 0-3m
- 0-6m
- 0-9m
- 0-12m

### 評価期間（4 個）

- 0-3m
- 3-6m
- 6-9m
- 9-12m

### 合計評価数

- **5 訓練ラベル × 4 評価期間 = 20 評価** × 2（通常 + 拡張） = **40 評価**

---

## プロセス情報

### 通常 IRL

- **PID**: 55753
- **ログ**: `/tmp/nova_irl.log`
- **出力**: `outputs/full_cross_eval_nova/`
- **特徴**: 10D state, 5D action

### 拡張 IRL

- **PID**: 55754
- **ログ**: `/tmp/nova_enhanced_irl.log`
- **出力**: `outputs/enhanced_cross_eval_nova/`
- **特徴**: 32D state, 9D action

---

## 進捗確認

### リアルタイムログ

```bash
# 通常IRL
tail -f /tmp/nova_irl.log

# 拡張IRL
tail -f /tmp/nova_enhanced_irl.log
```

### プロセス確認

```bash
ps aux | grep 'run_.*_nova.sh'
```

### caffeinate 確認

```bash
ps aux | grep caffeinate
```

---

## データ規模（推定）

### openstack/nova

- **レビュー数**: 27,328 件
- **期間**: 5 年分
- **全プロジェクトの割合**: 19.9%

### 予想される評価母集団

- **複数プロジェクト（全体）**: 260 人
- **nova 単一**: 約 50-100 人（推定）

---

## 結果の分析

### サマリー生成

```bash
# 通常IRL
uv run python scripts/training/irl/summarize_full_cross_evaluation.py \
    --input outputs/full_cross_eval_nova

# 拡張IRL
uv run python scripts/training/irl/summarize_full_cross_evaluation.py \
    --input outputs/enhanced_cross_eval_nova
```

### 比較ヒートマップ

```bash
# nova単一: 通常 vs 拡張
uv run python scripts/analysis/compare_irl_heatmaps.py \
    --normal outputs/full_cross_eval_nova \
    --enhanced outputs/enhanced_cross_eval_nova \
    --output outputs/nova_comparison

# 複数プロジェクト vs nova単一（通常IRL）
uv run python scripts/analysis/compare_irl_heatmaps.py \
    --normal outputs/full_cross_eval \
    --enhanced outputs/full_cross_eval_nova \
    --output outputs/multi_vs_nova_normal

# 複数プロジェクト vs nova単一（拡張IRL）
uv run python scripts/analysis/compare_irl_heatmaps.py \
    --normal outputs/enhanced_cross_eval \
    --enhanced outputs/enhanced_cross_eval_nova \
    --output outputs/multi_vs_nova_enhanced
```

---

## 期待される結果

### 複数プロジェクト vs 単一プロジェクト

1. **データ規模**

   - 単一プロジェクト: より小規模（50-100 人）
   - 複数プロジェクト: より大規模（260 人）

2. **プロジェクト固有性**

   - 単一プロジェクト: プロジェクト固有のパターンを学習
   - 複数プロジェクト: 汎用的なパターンを学習

3. **パフォーマンス予想**
   - 単一プロジェクト: 特定プロジェクトでは高精度の可能性
   - 複数プロジェクト: 汎化性能が高い可能性

---

## 実装の詳細

### 追加機能

- **プロジェクトフィルタ**: データ抽出時に単一プロジェクトのみを使用
- **`--project`引数**: 訓練/評価スクリプトに追加

### 修正ファイル

1. `scripts/training/irl/train_irl_within_training_period.py`

   - `extract_full_sequence_monthly_label_trajectories()`: `project`引数を追加
   - `extract_cutoff_evaluation_trajectories()`: `project`引数を追加
   - `extract_monthly_aggregated_label_trajectories()`: `project`引数を追加
   - `extract_multi_step_label_trajectories()`: `project`引数を追加

2. `scripts/training/irl/train_irl_per_timestep_labels.py`

   - `--project`引数を追加

3. `scripts/training/irl/train_enhanced_irl_per_timestep_labels.py`
   - `--project`引数を追加

### 新規ファイル

1. `scripts/training/irl/run_full_cross_evaluation_nova.sh`

   - nova 単一プロジェクトでの通常 IRL 実行

2. `scripts/training/irl/run_enhanced_cross_evaluation_nova.sh`
   - nova 単一プロジェクトでの拡張 IRL 実行

---

## 完了チェックリスト

- [x] 通常 IRL 完了（20 評価）✅ 2025-10-26 20:00:20
- [x] 拡張 IRL 完了（20 評価）✅ 2025-10-26 19:51:12
- [x] サマリー生成（通常）✅
- [x] サマリー生成（拡張）✅
- [x] ヒートマップ生成（nova: 通常 vs 拡張）✅
- [ ] ヒートマップ生成（複数 vs nova: 通常）⚠️ 複数プロジェクト側のデータ不足
- [x] ヒートマップ生成（複数 vs nova: 拡張）✅
- [x] 結果分析レポート作成 ✅

---

## 実行結果サマリー

### 完了時刻

- **通常 IRL**: 2025-10-26 20:00:20（実行時間: 約 1 時間 6 分）
- **拡張 IRL**: 2025-10-26 19:51:12（実行時間: 約 1 時間 58 分）

### 評価母集団

- **レビュアー数**: 77 人
- **継続率（0-3m）**: 46.8%
- **継続率（3-6m）**: 51.9%
- **継続率（6-9m）**: 37.7%
- **継続率（9-12m）**: 40.3%

### 主要な発見

#### nova 単一プロジェクト

**通常 IRL:**

- 最高 AUC-ROC: **0.852** (0-6m → 3-6m)
- 平均 AUC-ROC: **0.435**
- F1 スコア: 全て **0.610** 付近（分散が小さい）

**拡張 IRL:**

- 最高 AUC-ROC: **0.820** (0-12m → 6-9m)
- 平均 AUC-ROC: **0.491**
- F1 スコア: 全て **0.610** 付近（分散が小さい）

#### nova 単一 vs 複数プロジェクト（拡張 IRL）

| 項目            | 複数プロジェクト | nova 単一 | 差分    |
| --------------- | ---------------- | --------- | ------- |
| AUC-ROC（平均） | 0.573            | 0.491     | +0.082  |
| F1（平均）      | 0.707            | 0.611     | +0.096  |
| レビュアー数    | 260 人           | 77 人     | -       |
| 継続率（0-3m）  | 61.5%            | 46.8%     | -14.7pt |

**結論:**

- **複数プロジェクトの方が性能が高い**
- データ規模が大きいため、より汎化性能が高い
- nova 単一はデータ規模が小さく、継続率も低いため学習が困難

### 生成されたファイル

1. **nova 単一プロジェクト結果**

   - `outputs/full_cross_eval_nova/` (通常 IRL)
   - `outputs/enhanced_cross_eval_nova/` (拡張 IRL)
   - サマリー CSV、マトリクス CSV、ヒートマップ画像

2. **比較分析**
   - `outputs/nova_comparison/` (通常 vs 拡張)
   - `outputs/multi_vs_nova_enhanced/` (複数 vs nova: 拡張)
   - ヒートマップ、統計サマリー、モデル別比較

---

## メモ

- データ規模が小さいため、過学習のリスクあり → **実際に影響**
- 拡張 IRL の不安定性が改善されるか要確認 → **nova 単一でも不安定（AUC-ROC 0.491）**
- 複数プロジェクトとの性能比較が重要 → **複数プロジェクトの方が優れている（AUC-ROC +0.082）**
