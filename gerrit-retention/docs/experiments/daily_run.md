# 日次実行ログ — 2025-10-01（2 クラス方策, IRL→PPO）

作成者: @kzk03

## 目的 / 今日やったこと(10/1)

- 目的: IRL 報酬で学習した 2 クラス（非受諾/受諾）のレビュア方策を評価し、時間窓シフトも含めて過去再現度（リプレイ一致率）を確認する。
- 本日: sequences→extended 変換を本番規模に耐えるよう強化し、現状の sequences で変換 → オフライン生成 →IRL→PPO→ 評価までエンドツーエンドで実行。

## データ / 入力

- Sequences（本日）: `outputs/irl/reviewer_sequences.json`（小規模デモ相当）
- 変換後 extended: `data/extended_test_data.json`
- オフライン分割 cutoff: `2024-07-01T00:00:00Z`

## 主要スクリプト（パス）

- 変換（sequences→extended）: `scripts/offline/sequences_to_extended_data.py`
- オフライン生成（extended→JSONL）: `scripts/training/offline_rl/build_offline_data.py`
- IRL 学習（offline）: `scripts/training/offline_rl/train_retention_irl_from_offline.py`
- PPO 学習（二値 + IRL）: `scripts/training/rl_training/stable_ppo_training.py`
- リプレイ評価: `scripts/evaluation/policy_replay_match_eval.py`
- 期間ウィンドウ評価: `scripts/evaluation/policy_replay_time_windows.py`

## 本日の変更点（重要）

- `sequences_to_extended_data.py` を本番対応へ強化:
  - JSON / JSONL / .gz 入力対応
  - 巨大 JSON 配列のストリーミングパース（低メモリ）
  - 逐次書き出し（全体をメモリ保持しない）
  - CLI オプション追加: `--projects`, `--keep-empty`

## 実行サマリ

1. 変換

- 実施: sequences → extended
- 出力: `data/extended_test_data.json`

2. オフライン JSONL

- 実施: extended → `outputs/offline/dataset_train.jsonl`, `outputs/offline/dataset_eval.jsonl`
- 本日のメタ: train=2, eval=2（入力が小さいため）

3. IRL（offline）

- 実施: 30 エポック（本日のベースライン）
- モデル: `outputs/irl/retention_irl_offline_long.pth`
- 損失: 約 0.97 → 約 0.55 まで改善

4. PPO（二値, IRL=replace, 受諾ボーナス=0.3）

- 実施: 600 エピソード, 評価 100 エピソード
- 保存ポリシー:
  - `outputs/policies/stable_ppo_policy_20251001_165030.pt`
  - `outputs/policies/stable_ppo_policy_20251001_165625.pt`

5. リプレイ評価

- 使用ポリシー: `stable_ppo_policy_20251001_165030.pt`
- 出力: `outputs/irl/policy_replay_report_binary_prod.json`
- メモ: `<SAVED_POLICY_FROM_PPO>` などのプレースホルダは zsh でエラーになるため、実ファイル名を指定。

6. 期間ウィンドウ評価（累積）

- cutoff: `2024-07-01T00:00:00`
- 出力例: `outputs/irl/policy_replay_time_windows_binary.json`
- 注記: `*_prod.json` の実行はプレースホルダが原因で失敗。実ファイル名で再実行が必要（下のコマンド参照）。

## 結果（本日の入力は小規模 → 数値は参考程度）

- リプレイ（eval=2 ステップ）:
  - Multiclass overall: 0.5
  - Binary overall: 0.5
- 期間窓（binary, cutoff=2024-07-01）:
  - Train: 1.0, +1m: 0.0, +3m: 0.5, +6m: 0.5, +12m: 0.5

## 明日用コマンド（コピペ可・プレースホルダ無し）

1. sequences → extended

```bash
uv run python scripts/offline/sequences_to_extended_data.py \
  --input outputs/irl/reviewer_sequences.json \
  --out data/extended_test_data.json \
  --projects historical-project
```

2. extended → offline JSONL

```bash
uv run python scripts/training/offline_rl/build_offline_data.py \
  --data data/extended_test_data.json \
  --cutoff 2024-07-01T00:00:00Z \
  --out outputs/offline
```

3. IRL 学習

```bash
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/dataset_train.jsonl \
  --epochs 100 \
  --batch-size 4096 \
  --negatives-per-pos 6 \
  --out outputs/irl/retention_irl_offline_prod.pth
```

4. PPO 学習（二値 + IRL 報酬）

```bash
STABLE_RL_DATA=data/extended_test_data.json \
STABLE_RL_CUTOFF=2024-07-01T00:00:00Z \
uv run python scripts/training/rl_training/stable_ppo_training.py \
  --use-irl-reward \
  --irl-model-path outputs/irl/retention_irl_offline_prod.pth \
  --irl-reward-mode replace \
  --engagement-bonus-weight 0.3 \
  --binary-actions \
  --train-episodes 1500 \
  --eval-episodes 300
```

5. リプレイ一致率（eval JSONL, 二値併記）

```bash
uv run python scripts/evaluation/policy_replay_match_eval.py \
  --offline outputs/offline/dataset_eval.jsonl \
  --policy outputs/policies/stable_ppo_policy_20251001_165030.pt \
  --collapse-wait-reject \
  --out outputs/irl/policy_replay_report_binary_prod.json
```

6. 期間ウィンドウ比較（同じポリシー）

```bash
uv run python scripts/evaluation/policy_replay_time_windows.py \
  --outdir outputs/offline \
  --policy outputs/policies/stable_ppo_policy_20251001_165030.pt \
  --windows 1m,3m,6m,12m \
  --collapse-wait-reject \
  --out outputs/irl/policy_replay_time_windows_binary_prod.json
```

## チューニングのヒント（方策が偏る/受諾過多のとき）

- IRL blend: `--irl-reward-mode blend --irl-reward-alpha 0.5`
- 受諾ボーナス調整: `--engagement-bonus-weight 0.2〜0.4`
- PPO エピソード増: `--train-episodes 2000+`
- 3 クラス分析を見たい場合は、`--binary-actions` を外し、評価で collapse フラグを外して wait の挙動を可視化。

## 成果物チェックリスト

- Extended: `data/extended_test_data.json`
- Offline: `outputs/offline/{dataset_train.jsonl, dataset_eval.jsonl, offline_dataset_meta.json}`
- IRL model: `outputs/irl/retention_irl_offline_*.pth`
- Policies: `outputs/policies/stable_ppo_policy_*.pt`
- Reports: `outputs/irl/policy_replay_report_*.json`, `outputs/irl/policy_replay_time_windows_*.json`

## 既知の注意点 / Tips

- `<SAVED_POLICY_FROM_PPO>` のようなプレースホルダは zsh でエラーになります。実ファイル名を指定（`ls outputs/policies/` で確認）。
- eval が極小だと数値が不安定。より長期の sequences（大規模）を推奨。
- 変換スクリプトは `.jsonl` と `.gz` に対応。超大規模ファイルは `.gz` 圧縮で IO を軽減可能。

---

## 本日のレポートまとめ（要点）(10/2)

- ハイライト
  - sequences→extended 変換を本番対応（JSON/JSONL/.gz、ストリーミング、逐次書き出し）に刷新。
  - 2クラス（二値）方策で IRL→PPO→評価（リプレイ・期間窓）まで一通り通し、成果物を保存。
  - オフライン生成の希薄さを緩和する第一歩として、レビュアー連結エピソード生成オプション（`--episode-group reviewer`）を追加（別スクリプト）。

- 主要結果（小規模データのため参考値）
  - IRL: 30エポックで損失 ~0.97 → ~0.55。
  - PPO（二値, IRL replace, bonus=0.3, 600ep）: 学習中行動分布は accept≈55%。
  - リプレイ（eval=2step）: multiclass/binary ともに overall=0.5。
  - 期間窓（cutoff=2024-07-01）: Train=1.0, +1m=0.0, +3m=0.5, +6m=0.5, +12m=0.5。

- データ規模/注意点と原因分析
  - 本日の sequences は小規模で、offline の eval が2ステップのみ。期間+1m に該当イベントが1件→ `num_steps=1, num_episodes=1`。
  - 現行 offline 生成は「1イベント=1ステップ」設計が基本（即 done）。非レビューイベントは既定では除外。

- 実装の差分（今日のΔ）
  - `scripts/offline/sequences_to_extended_data.py`: 本番対応（入力形式拡張、ストリーミング、逐次書き出し、CLI）。
  - `scripts/offline/build_offline_from_reviewer_sequences.py`: `--episode-group reviewer` を追加（レビュアー単位で連結し、内部 done を切らずに連続 step）。

- 失敗/対処
  - `<SAVED_POLICY_FROM_PPO>` プレースホルダ使用でコマンド失敗 → 実ファイル名に差し替えて再実行。

- 次のアクション（優先順）
  1) より長期・大規模 sequences で再実行（eval の安定化）。
  2) offline 生成で `--include-non-review` を併用し、サンプル数を増やす（必要に応じ wait 注入）。
  3) タスク中心（Task-centric）MDP の試作（Task×Reviewer ペアで IRL、割当 RL 環境）。
  4) 方策チューニング: IRL blend（alpha調整）、受諾ボーナス、PPOエピソード増。

- 成果物
  - Extended: `data/extended_test_data.json`
  - Offline: `outputs/offline/`
  - IRL: `outputs/irl/retention_irl_offline_long.pth`
  - Policies: `outputs/policies/stable_ppo_policy_*.pt`
  - Reports: `outputs/irl/policy_replay_report_*.json`, `outputs/irl/policy_replay_time_windows_*.json`

品質ゲート（スモーク）
- スクリプト実行: PASS（uv 経由で全行程を通過）
- 例外/停止: プレースホルダ以外の致命的エラーなし
- データ漏洩対策: cutoff 準拠で過去情報のみ使用（既存方針踏襲）
