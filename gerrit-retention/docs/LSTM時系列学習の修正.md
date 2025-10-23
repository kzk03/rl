# LSTM 時系列学習の修正

**作成日**: 2025-10-22  
**問題**: 予測精度が極端に悪い（AUC-ROC=0.5, pred_mean=1.0）  
**原因**: 時系列状態の不適切な実装

---

## 🔴 発見された問題

### 問題 1: 全タイムステップで同じ状態を使用

**Before（誤り）**:

```python
# ❌ 全タイムステップで同じ状態
state_tensors = [self.state_to_tensor(state) for _ in range(self.seq_len)]
state_seq = torch.stack(state_tensors)  # [20, state_dim]

# 例: 20ステップすべてで同じ状態
state_seq = [
    [0.5, 0.3, 0.2, ...],  # ステップ1（19ヶ月前）← 同じ
    [0.5, 0.3, 0.2, ...],  # ステップ2（18ヶ月前）← 同じ
    ...
    [0.5, 0.3, 0.2, ...],  # ステップ20（現在）  ← 同じ
]

結果:
- LSTM が state から学習できない
- action のパターンのみを見る
- しかし action だけでは予測困難
- → モデルが平均値（継続率70%）に収束
```

**After（正しい）**:

```python
# ✅ 各タイムステップでその時点の状態を再構築
step_dates = []
for i in range(self.seq_len):
    days_before = (self.seq_len - 1 - i) * 30
    step_date = context_date - timedelta(days=days_before)
    step_dates.append(step_date)

state_tensors = []
for i, step_date in enumerate(step_dates):
    # その時点までの履歴
    history_up_to_step = [
        act for act in activity_history
        if act.get('timestamp', context_date) <= step_date
    ]

    # その時点での状態を計算
    state_at_step = self.extract_developer_state(
        developer,
        history_up_to_step,
        step_date
    )

    state_tensors.append(self.state_to_tensor(state_at_step))

state_seq = torch.stack(state_tensors)  # [20, state_dim]

# 例: 各ステップで異なる状態
state_seq = [
    [0.1, 0.2, 0.1, ...],  # ステップ1（19ヶ月前）← 初期状態
    [0.2, 0.3, 0.2, ...],  # ステップ2（18ヶ月前）← 少し成長
    [0.3, 0.4, 0.3, ...],  # ステップ3
    ...
    [0.5, 0.6, 0.5, ...],  # ステップ20（現在）  ← 現在の状態
]

結果:
- LSTM が state の時系列変化を学習できる
- 経験の蓄積パターンを捉えられる
- action と state の相互作用を学習
```

### 問題 2: クラス不均衡（継続率 70%）

**問題**:

```python
継続率 = 70%
→ 「全員継続」と予測すれば 70% 正解
→ モデルが局所最適解に陥る
→ pred_mean = 1.0, AUC-ROC = 0.5
```

**解決策**: Focal Loss + クラスウェイト

```python
# Focal Loss: 難しいサンプルに焦点
def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - p_t) ** gamma  # 難しいサンプル（p_t ≈ 0.5）の重みを上げる
    loss = alpha * focal_weight * bce
    return loss.mean()

# クラスウェイト: 少数派（非継続者）の重みを上げる
if positive_rate > 0.5:
    pos_weight = 1.0
    neg_weight = positive_rate / (1 - positive_rate)  # 例: 70%/30% = 2.33
else:
    pos_weight = (1 - positive_rate) / positive_rate
    neg_weight = 1.0

# 損失計算
continuation_loss = self.focal_loss(pred, target) * sample_weight
```

### 問題 3: 学習率が高すぎる

**Before**: `lr = 0.001`

- 初期損失: 0.0155
- エポック 10: 損失 = 0.0000 ← 異常に早い収束

**After**: `lr = 0.0001`

- より慎重な学習
- 局所最適解を回避

---

## 📝 修正内容

### 1. 時系列状態の再構築（訓練時）

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**関数**: `train_irl`  
**行**: 352-416

```python
# Before（誤り）
state_tensors = [self.state_to_tensor(state) for _ in range(self.seq_len)]

# After（正しい）
# 各タイムステップでの日付を計算
step_dates = []
for i in range(self.seq_len):
    days_before = (self.seq_len - 1 - i) * 30
    step_date = context_date - timedelta(days=days_before)
    step_dates.append(step_date)

# 各タイムステップで状態を再構築
state_tensors = []
for i, step_date in enumerate(step_dates):
    history_up_to_step = [
        act for act in activity_history
        if act.get('timestamp', context_date) <= step_date
    ]

    if history_up_to_step:
        state_at_step = self.extract_developer_state(
            developer,
            history_up_to_step,
            step_date
        )
    else:
        state_at_step = state  # 初期状態

    state_tensors.append(self.state_to_tensor(state_at_step))

state_seq = torch.stack(state_tensors).unsqueeze(0)
```

### 2. 時系列状態の再構築（予測時）

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**関数**: `predict_continuation_probability`  
**行**: 548-592

同様のロジックを予測関数にも適用。

### 3. Focal Loss の追加

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**行**: 306-320

```python
def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss: 難しいサンプルに焦点を当てる
    """
    bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    p_t = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - p_t) ** gamma
    loss = alpha * focal_weight * bce
    return loss.mean()
```

### 4. クラスウェイトの計算

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**行**: 335-352

```python
# クラス不均衡を計算
positive_count = sum(1 for t in expert_trajectories if t.get('continued', True))
negative_count = len(expert_trajectories) - positive_count
positive_rate = positive_count / len(expert_trajectories)

# クラスウェイトを計算
if positive_rate > 0.5:
    pos_weight = 1.0
    neg_weight = positive_rate / (1 - positive_rate)
else:
    pos_weight = (1 - positive_rate) / positive_rate
    neg_weight = 1.0

logger.info(f"継続率: {positive_rate:.1%}")
logger.info(f"クラスウェイト: 継続={pos_weight:.2f}, 非継続={neg_weight:.2f}")
```

### 5. 損失計算の修正

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**行**: 438-453

```python
# Before
continuation_loss = self.bce_loss(predicted_continuation, target_continuation)

# After
continuation_loss = self.focal_loss(
    predicted_continuation,
    target_continuation,
    alpha=0.25,
    gamma=2.0
)

# クラスウェイトを追加で適用
sample_weight = pos_weight if continuation_label else neg_weight
continuation_loss = continuation_loss * sample_weight
```

### 6. 学習率の削減

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`  
**行**: 151-155

```python
# Before
lr=config.get('learning_rate', 0.001)

# After
lr=config.get('learning_rate', 0.0001)  # 10分の1に削減
```

---

## 📊 期待される改善

### Before（修正前）

```
訓練:
  エポック0:  損失 = 0.0155
  エポック10: 損失 = 0.0000  ← 異常に早い
  エポック30: 損失 = 0.0000

評価:
  pred_mean: 1.0              ← 全員「継続」予測
  pred_std:  0.0              ← 分散ゼロ
  AUC-ROC:   0.5              ← ランダム予測
  AUC-PR:    0.849            ← 継続率が高いだけ
  F1:        0.823            ← Recall=1.0のみ
  Precision: 0.699 (=継続率) ← 全員予測の結果
  Recall:    1.0              ← 全員予測
```

### After（修正後・期待値）

```
訓練:
  エポック0:  損失 = 0.15-0.30   ← より高い初期損失
  エポック10: 損失 = 0.10-0.15   ← 緩やかな収束
  エポック30: 損失 = 0.05-0.10   ← 継続的な改善

評価:
  pred_mean: 0.5-0.7           ← 多様な予測
  pred_std:  0.1-0.2           ← 適切な分散
  AUC-ROC:   0.65-0.75         ← 識別能力あり ✅
  AUC-PR:    0.70-0.80         ← 改善
  F1:        0.60-0.75         ← バランスの良い予測
  Precision: 0.60-0.70         ← 適切な精度
  Recall:    0.70-0.85         ← 過剰予測の回避
```

---

## 🎯 理論的根拠

### なぜ各ステップで状態を再構築するのか？

**時系列の本質**:

```
開発者の成長プロセス:
t=0（19ヶ月前）: 初心者、経験少ない、プロジェクト1個
t=10（9ヶ月前）: 中級者、経験増加、プロジェクト3個
t=20（現在）:    ベテラン、経験豊富、プロジェクト5個

→ 各時点で「異なる状態」にいる
→ 状態の変化パターンが継続性を予測する鍵
```

**LSTM が学習すべきパターン**:

1. **成長パターン**: 経験が順調に増加 → 継続しやすい
2. **停滞パターン**: 経験が増えない → 離脱リスク
3. **加速パターン**: 活動頻度が増加 → 高い継続性
4. **減速パターン**: 活動頻度が減少 → 離脱の前兆

**誤った実装の問題**:

```python
# 全ステップで同じ状態
state = [経験5年, プロジェクト5個, ...]

→ LSTMは「現在の状態」しか見えない
→ 「成長パターン」を学習できない
→ actionの時系列のみに依存
→ しかしactionだけでは不十分
→ モデルが平均値に収束
```

---

## 🚀 次のステップ

### 1. 既存モデルを削除

```bash
cd /Users/kazuki-h/rl/gerrit-retention
rm -rf outputs/cross_evaluation/model_*
```

### 2. 再訓練

```bash
bash scripts/training/irl/run_cross_evaluation.sh
```

### 3. 結果の確認

```bash
# 訓練ログを確認（損失の変化）
tail -50 outputs/cross_evaluation/model_0_6m/training.log

# 評価結果を確認
cat outputs/cross_evaluation/model_0_6m/evaluation_results.json

# 期待される改善:
# - pred_mean: 0.5-0.7 （現状: 1.0）
# - pred_std: > 0.05 （現状: 0.0）
# - AUC-ROC: > 0.60 （現状: 0.5）
```

### 4. 詳細な評価

```bash
# クロス評価結果
cat outputs/cross_evaluation/cross_evaluation_results.csv

# ヒートマップ確認
open outputs/cross_evaluation/cross_evaluation_heatmaps.png
```

---

## 🔧 トラブルシューティング

### エラー: timestamp 属性がない

```python
# エラーメッセージ
KeyError: 'timestamp'

# 原因
activity_history の各要素に 'timestamp' キーがない

# 解決策
データ抽出スクリプトで timestamp を追加
または、別のキー名（'date', 'request_time'）を使用
```

### 依然として pred_mean = 1.0

```python
# 考えられる原因
1. データの継続率が非常に高い（> 90%）
   → さらに強いクラスウェイトが必要

2. 学習率がまだ高い
   → 0.00001 に下げる

3. Focal Loss の gamma が不適切
   → gamma=3.0 または 4.0 に上げる
```

### 訓練が遅い

```python
# 各ステップで状態を再計算するため遅くなる

# 高速化オプション:
1. 状態を事前計算してキャッシュ
2. バッチ処理の最適化
3. GPU使用（CUDA）
```

---

## 📚 関連ドキュメント

- [精度問題の分析と解決策.md](./精度問題の分析と解決策.md) - 閾値問題の分析
- [閾値統一実装の詳細.md](./閾値統一実装の詳細.md) - 閾値の統一
- [IRL 設計と実験結果サマリー.md](./IRL設計と実験結果サマリー.md) - 全体サマリー

---

**最終更新**: 2025-10-22  
**実装ステータス**: ✅ 完了  
**次のアクション**: モデルの再訓練と評価
