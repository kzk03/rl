# 重み付きラベル IRL - 実装確認と精度調査

## 概要

重み付きラベル IRL の実装状況と、精度低下の原因調査を行います。

---

## 現在の実装状況

### 1. 軌跡データ構造

#### 追加されたフィールド

```python
trajectory = {
    'future_acceptance': 0 or 1,      # ラベル（2値）
    'had_requests': True or False,    # 依頼があったか
    'sample_weight': 1.0 or 0.7       # サンプル重み
}
```

**重みの設定**:

- `had_requests=True` (依頼あり): `sample_weight = 1.0` （通常の重み）
- `had_requests=False` (依頼なし): `sample_weight = 0.7` （中程度の重み、0.5→0.7 に増加）

### 2. ラベル付けロジック（修正版）

#### train_irl_review_acceptance.py: 行 482-518

```python
# 評価期間にレビュー依頼を受けていない場合、拡張期間をチェック
if len(reviewer_eval) == 0:
    # 拡張期間のレビュー依頼をチェック
    reviewer_extended_eval = extended_eval_df[extended_eval_df[reviewer_col] == reviewer]

    if len(reviewer_extended_eval) == 0:
        # 拡張期間にも依頼がない → 本当に離脱したと判断、除外
        continue
    else:
        # 拡張期間には依頼がある → この期間では活動しなかったので負例（0）
        future_acceptance = False
        had_requests = False
        sample_weight = 0.7  # 中程度の重み

        rescued_by_extended += 1
        negative_count += 1
        negative_extended_count += 1
        negative_without_requests += 1
else:
    # 通常の評価期間に依頼がある場合
    accepted_requests = reviewer_eval[reviewer_eval[label_col] == 1]
    rejected_requests = reviewer_eval[reviewer_eval[label_col] == 0]
    future_acceptance = len(accepted_requests) > 0
    had_requests = True
    sample_weight = 1.0  # 通常の重み

    if future_acceptance:
        positive_count += 1
    else:
        negative_count += 1
        negative_with_requests += 1
```

### 3. 損失計算での重み適用

#### retention_irl_system.py: 行 1322-1329

```python
# サンプル重みを取得（依頼なし=0.5、依頼あり=1.0）
sample_weight = trajectory.get('sample_weight', 1.0)
sample_weights = torch.full([min_len], sample_weight, device=self.device)

# 継続予測損失（Focal Loss を使用、重み付き）
continuation_loss = self.focal_loss(
    predicted_continuation_flat, targets, sample_weights
)
```

#### focal_loss 関数: 行 291-319

```python
def focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
               sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Focal Loss の計算

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) * sample_weight
    """
    predictions = predictions.squeeze()
    targets = targets.squeeze()

    # BCE loss
    bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

    # p_t の計算
    p_t = predictions * targets + (1 - predictions) * (1 - targets)

    # alpha_t の計算
    alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)

    # Focal Loss
    focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
    focal_loss = focal_weight * bce_loss

    # サンプル重みを適用
    if sample_weights is not None:
        sample_weights = sample_weights.squeeze()
        focal_loss = focal_loss * sample_weights

    return focal_loss.mean()
```

---

## 実装の妥当性

### ✅ 正しく実装されている部分

1. **軌跡データに`sample_weight`が追加されている**: 行 566
2. **損失計算で重みを適用している**: 行 1323-1329
3. **拡張期間チェックで負例を追加している**: 行 482-504

### ❌ 問題点

1. **重みの効果が不明確**

   - `weight=0.7` と `weight=1.0` の差が小さい
   - 「依頼なし」と「依頼あり → 拒否」を区別できるか疑問

2. **訓練サンプル数の増加が性能に影響**

   - 拡張期間チェックでサンプル数が増えた
   - しかし、増加したサンプルは全て負例（ラベル=0）
   - 正例率が下がり、性能が低下した可能性

3. **重みの値の妥当性**
   - 初期設計: `weight=0.5` for 依頼なし
   - 現在: `weight=0.7` for 依頼なし
   - 最適値が不明

---

## 現在のモデル精度

### バランス調整後（Seed 888, LR 0.000075, Epoch 25, Dropout 0.3）

```
train_0-3m: AUC-PR=0.518, R=0.952
train_3-6m: AUC-PR=0.612, R=0.944
train_6-9m: AUC-PR=0.460, R=1.000 ❌
train_9-12m: AUC-PR=0.432, R=1.000 ❌
平均AUC-PR: 0.506
```

### 最良設定（Seed 777, LR 0.00005, Epoch 20, Dropout 0.3）

```
train_0-3m: AUC-PR=0.689, R=0.857 ✅
train_3-6m: AUC-PR=0.770, R=0.667 ✅
train_6-9m: AUC-PR=0.529, R=1.000 ❌
train_9-12m: AUC-PR=0.670, R=0.562 ✅
平均AUC-PR: 0.664
```

### 比較

- **平均 AUC-PR**: 0.664 → 0.506 （**-23.8%**）
- **Recall=1.0 問題**: train_6-9m, train_9-12m で悪化

---

## 原因分析

### 仮説 1: 訓練サンプル数の変化

**問題**:

- 拡張期間チェックでサンプル数が増加
- 増加したサンプルは全て負例（ラベル=0）
- 正例率が下がる

**確認方法**:

- 訓練前後のサンプル数を比較
- 正例率の変化を確認

### 仮説 2: 重みの効果が不十分

**問題**:

- `weight=0.7` が `weight=1.0` に近すぎる
- 「依頼なし」と「依頼あり → 拒否」を区別できない

**確認方法**:

- 重み=0.7 のサンプル数と分布を確認
- 損失への影響を分析

### 仮説 3: 拡張期間チェックのロジックが不適切

**問題**:

- 拡張期間（0-12m）をチェックしている
- ラベル期間（0-3m）に依頼がなくても拡張期間にあれば負例
- しかし、これは「その期間に活動しなかった」だけで「離脱」ではない

**確認方法**:

- 拡張期間で「救済」されたサンプル数と分布
- これらのサンプルが性能に与える影響

---

## 調査項目

### 1. サンプル数の変化

**現在（重み付きラベル実装後）**:

- train_0-3m: ?人
- train_3-6m: ?人
- train_6-9m: ?人
- train_9-12m: ?人

**以前（最良設定）**:

- train_0-3m: 44 人
- train_3-6m: 45 人
- train_6-9m: 32 人
- train_9-12m: 39 人

### 2. 正例率の変化

**現在**: ?

**以前**: 各期間で異なる

### 3. 重みの分布

- `weight=0.7` (依頼なし): ?人
- `weight=1.0` (依頼あり): ?人

### 4. 拡張期間で「救済」されたサンプル数

- `rescued_by_extended`: ?人

---

## 次のステップ

1. **データ確認**: 実際の訓練データを読み込んで統計を確認
2. **重み調整**: 重みの値を変更してテスト（例: 0.5, 0.3, 0.1）
3. **拡張期間の見直し**: 拡張期間のチェックロジックを調整
4. **最良設定に戻す**: Seed 777, LR 0.00005, Epoch 20, Dropout 0.3 に戻して比較

---

## 関連ファイル

- `scripts/training/irl/train_irl_review_acceptance.py`: 行 482-566（軌跡抽出）
- `src/gerrit_retention/rl_prediction/retention_irl_system.py`: 行 1322-1329（損失計算）
- `docs/ラベル付けロジック_詳細解説.md`: 詳細なロジック解説

---

## 備考

- 本ドキュメントは実装コードを基に作成
- 実際のデータを確認して統計を更新する必要がある
