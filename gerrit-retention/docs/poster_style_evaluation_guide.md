# ポスター風評価ガイド（選定 IRL + 受諾モデル）

このガイドは、ポスター記載の方針に沿って「招待の選定」と「招待後の受諾/実参加」までをオフラインで評価するための手順と実行コマンドをまとめたものです。

## 最新出力参照先（Last Artifacts）

- 最終更新: 2025-09-18
- 正式（現行）出力:
  - 招待の選定（IRL, Plackett–Luce）: `outputs/reviewer_invitation_irl_pl_full/`
  - 招待後の受諾/実参加: `outputs/reviewer_acceptance_after_invite_full/`
- 旧版/整理済み出力: `outputs/_legacy/`（不要特徴を含む旧成果物を移設）

## 目的（Objective）

- 選定（誰を招待するか）の相対確率を IRL で学習・評価
- 招待後の受諾/実参加確率を別モデルで学習・評価
- ランキング指標（Recall/MAP/NDCG@K）と確率的指標（AUC/Brier）で性能を把握
- （オプション）ロギング方策 μ と提案方策 π を用いた OPE（IPS/SNIPS/DR）

## データと分割（Data & Split）

- 入力データ（例）: `data/processed/unified/all_reviews.json`
- 時系列分割で評価（既存スクリプトはテスト 20%の時系列後方を用いる）

---

## 1. 招待の選定（IRL: Plackett–Luce）

- 概要: 同一変更に対する候補レビュアー集合内での選択（招待）確率を学習。
- スクリプト: `scripts/run_reviewer_invitation_ranking.py`
- 主な機能:
  - 候補集合の構築（活動者＋実招待者）
  - 特徴量生成（活動/適合/負荷/内容類似）
  - IRL 学習（`--irl-mode plackett`）と評価（ランキング指標、AUC/Brier、NLL）
  - 重み/分析のエクスポート

実行例（Plackett–Luce, full window）:

```bash
uv run python scripts/run_reviewer_invitation_ranking.py \
  --changes data/processed/unified/all_reviews.json \
  --mode irl --irl-mode plackett \
  --min-total-reviews 1 --recent-days 3650 \
  --output outputs/reviewer_invitation_irl_pl_full
```

主な出力:

- `metrics_irl.json`（または `metrics_irl_plackett.json`）
- `test_predictions_irl_plackett.json`（候補内確率、先頭 1000 件）
- `weights_irl_plackett.json`, `weights_summary_irl_plackett.csv`
- `reward_analysis_irl_plackett.json`

---

## 2. 招待後の受諾/実参加（Acceptance after Invite）

- 概要: 招待済みレビュアーのみを母集団にして、実参加（コメント等）の確率を推定。
- スクリプト: `scripts/run_reviewer_acceptance_after_invite.py`
- 主な機能:
  - 招待者のみサンプリング、リーケージ除去済み `pending` 計算
  - 特徴は選定 IRL と整合（TF‑IDF 類似を含む）
  - 時系列分割で学習・評価、ランキング指標は「招待者内」で計算
  - 重み/分析のエクスポート

実行例（full window）:

```bash
uv run python scripts/run_reviewer_acceptance_after_invite.py \
  --changes data/processed/unified/all_reviews.json \
  --min-total-reviews 1 --recent-days 3650 \
  --output outputs/reviewer_acceptance_after_invite_full
```

主な出力:

- `metrics.json`
- `test_predictions.json`
- `weights.json`, `weights_summary.csv`
- `reward_analysis.json`

---

## 3. 指標の読み方（Metrics）

- AUC/Brier: 確率予測の識別力/校正度
- Recall@K/MAP@K/NDCG@K: 変更ごとの候補集合（選定）/招待者集合（受諾）でのランキング品質
- IRL の重み: 標準化後係数（+1SD オッズ比、Δ 確率）で直感比較

---

## 4. （オプション）OPE: オフライン方策評価の手順

- 目的: ログ（ロギング方策 μ）から、提案方策 π の期待報酬を推定。
- 必要情報:
  - ログ上の招待と実参加（reward r）
  - μ(a|x): 手順 1 の IRL（Plackett）で近似した確率
  - π(a|x): 別設定の IRL（または学習済みモデル）での確率
  - q(a,x): 手順 2 の受諾確率
- 推定量:
  - IPS: 1/N Σ r·π/μ
  - SNIPS: Σ r·w / Σ w（w=π/μ）
  - DR: 1/N Σ [ w·(r−q) + E\_{a~π}[q] ]
- 実装メモ:
  - 変更単位でのスレート（Top-K）を扱う場合、Plackett の逐次確率で μ と π を組み立て。
  - 分散抑制: w のクリッピング、SNIPS 推奨、ブートストラップ CI（95%）。

（必要に応じ `scripts/` に OPE 集計スクリプトを追加可能です。入力は test_predictions と受諾確率を結合した CSV/JSON を想定。）

---

## 5. トラブルシュート

- ローカル `src` を優先するため、実行スクリプトは `sys.path` に `src` を追加済み。
- 旧成果物に不要特徴が残る場合、`scripts/filter_reward_analysis_features.py` でクリーニング可能。

---

## 6. 参考

- 出力例フォルダ:
  - `outputs/reviewer_invitation_irl_pl_full/`
  - `outputs/reviewer_acceptance_after_invite_full/`
- 主要スクリプト:
  - `scripts/run_reviewer_invitation_ranking.py`
  - `scripts/run_reviewer_acceptance_after_invite.py`
