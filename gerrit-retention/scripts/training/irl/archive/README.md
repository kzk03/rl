# アーカイブ - 旧版 IRL モデル

このディレクトリには、レビュー継続予測の旧版実装が含まれています。

## アーカイブされたファイル

### `train_irl_sliding_window_old.py`

**問題点**:

- レビュー依頼の有無を考慮していない
- 評価期間内にレビュー依頼を受けていない開発者も負例として扱っていた
- 「レビュー活動があったかどうか」という不正確な継続判定

**アーカイブ理由**:

- 予測の目的と実装が不一致
- レビュー依頼を拒否した人と、そもそも依頼を受けていない人を区別できない
- より正確な予測ロジックに置き換えられた

## 新版の実装

### `train_irl_review_acceptance.py`

**改善点**:

- レビュー依頼の有無を適切に考慮
- 評価期間内にレビュー依頼を受けた開発者のみを対象
- 正しい継続判定：「レビュー依頼を承諾したかどうか」

**継続判定ロジック**:

```
評価期間内にレビュー依頼なし → 除外（判定対象外）
評価期間内にレビュー依頼あり + 少なくとも1つ承諾 → 正例（継続）
評価期間内にレビュー依頼あり + 全て拒否 → 負例（離脱）
```

## データ構造

CSV ファイル (`review_requests_openstack_multi_5y_detail.csv`) の `label` カラム:

- `label = 1`: レビュー依頼に応答（承諾）
- `label = 0`: レビュー依頼に応答せず（拒否/無視）

## 使用方法

新版の実装を使用してください：

```bash
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2023-04-01 \
  --future-window-start 0 \
  --future-window-end 3 \
  --epochs 20 \
  --min-history-events 3 \
  --output outputs/review_acceptance_nova \
  --project openstack/nova
```

## アーカイブ日

2025-10-29
