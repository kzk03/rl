# タスク中心 MDP 実験レポート（2025-10-03）

作成者: @kzk03

## 目的と背景

- 目的: Gerrit の変更ごとに「どのレビュアを割り当てるか」を学習するタスク中心 MDP（MultiReviewerAssignmentEnv）で、学習済み方策の過去再現度（行動一致率）を時間窓別に評価する。
- 背景: これまでのレビュア単位の受諾 MDP から、より意思決定に直結する「タスク → レビュア割当」へピボット。候補レビューア集合と候補別特徴をタスク時点に整合させ、現実的な割当選好を再現できるかを検証。

## データとソース

- 元データ（Gerrit 変更）
  - 統合元: `data/raw/gerrit_changes/openstack_multi_5y_detail_*.json`
  - ユニファイ: `scripts/offline/unify_gerrit_changes.py` で `data/processed/unified/all_reviews.json` を生成
  - ファイル構造: `{metadata, data: {project: [changes...]}}`（トップ辞書）
- reviewer_sequences 抽出（レビュア軸）
  - スクリプト: `scripts/extract_reviewer_sequences.py`
  - 主要パラメータ: `--idle-gap 40`, `--tail-inactive-days 90`
  - 出力: `outputs/irl/reviewer_sequences.json`
  - 備考: ユニファイド形式（上記の data 辞書）からフラット変更配列に自動展開するよう対応済み
- タスク生成（Task× 候補レビュア）
  - スクリプト: `scripts/offline/build_task_assignment_from_sequences.py`
  - ハイライト:
    - 候補選定: `--candidate-sampling time-local`（タスク時刻 ±60 日で活動のあるレビュアを優先）
    - 候補別特徴: 候補 r のタスク時刻直前の最新 state を使用（なければ元 state にフォールバック）
  - 実行設定: `--candidate-window-days 60`, `--max-candidates 8`, `--cutoff 2024-07-01T00:00:00Z`
  - 出力: `outputs/task_assign/tasks_{train,eval}.jsonl`, メタ: `tasks_meta.json`

## 環境と方策

- 環境: `MultiReviewerAssignmentEnv`
  - 観測: 最大 K 候補の特徴ベクトル連結（+ action mask）
  - 行動: Discrete(K)（1 名選択）
  - 報酬: `reward_mode = 'match_gt'`（選択が正解集合に含まれれば+1.0）
- 方策: 小型 MLP（学習スクリプト `train_task_assignment.py`）
  - ネット: 128→64→K の全結合（ReLU）
  - 学習: 600 episodes（方策勾配の最小実装）

## 実験設計（評価）

- スプリット: タスクの `timestamp` で `cutoff = 2024-07-01T00:00:00Z`
  - train: `timestamp <= cutoff`
  - eval: `timestamp > cutoff`
- 指標: 行動一致率（Action Match Rate = 選択レビュア ∈ 正解集合）
- 時間窓: eval 側で累積窓を評価（`1m, 3m, 6m, 12m`）
- 実行スクリプト: `scripts/evaluation/task_assignment_replay_eval.py`
  - 進捗: tqdm 表示（ウィンドウごと + タスクステップ）

### 最小契約（contract）

- 入力: tasks\_{train,eval}.jsonl（各行: change_id, candidates[{reviewer_id, features}], positive_reviewer_ids, timestamp）
- 出力: 窓ごとの {steps, action_match_rate}
- 失敗モード: 候補 0/正解 0 のタスクは reward=0 のみ、steps にはカウント。タスク 0 件の窓は steps=0, rate=None。

## 再現手順（コマンド）

```bash
# 1) Raw -> Unified
uv run python scripts/offline/unify_gerrit_changes.py \
  --inputs 'data/raw/gerrit_changes/openstack_multi_5y_detail_*.json' \
  --out data/processed/unified/all_reviews.json

# 2) Unified -> reviewer_sequences
uv run python scripts/extract_reviewer_sequences.py \
  --idle-gap 40 \
  --tail-inactive-days 90 \
  --output outputs/irl/reviewer_sequences.json

# 3) sequences -> tasks(train/eval)
uv run python scripts/offline/build_task_assignment_from_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2024-07-01T00:00:00Z \
  --outdir outputs/task_assign \
  --max-candidates 8 \
  --candidate-sampling time-local \
  --candidate-window-days 60

# 4) 学習
uv run python scripts/training/rl_training/train_task_assignment.py \
  --train-tasks outputs/task_assign/tasks_train.jsonl \
  --eval-tasks outputs/task_assign/tasks_eval.jsonl \
  --outdir outputs/task_assign \
  --episodes 600

# 5) 評価（時間窓）
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign/tasks_eval.jsonl \
  --cutoff 2024-07-01T00:00:00Z \
  --policy outputs/task_assign/task_assign_policy.pt \
  --windows 1m,3m,6m,12m \
  --out outputs/task_assign/replay_eval_post.json
```

## 結果

- 評価出力（抜粋）: `outputs/task_assign/replay_eval_post.json`
  - 1m: steps=2,284, action_match_rate=0.9873
  - 3m: steps=6,154, action_match_rate=0.9855
  - 6m: steps=10,448, action_match_rate=0.9860
  - 12m: steps=18,890, action_match_rate=0.9864

### 所見

- 非常に高い一致率（≈98.5–98.7%）。要因として:
  - 正解集合（実参加者）に「タスク発生 source reviewer」を含めており、time-local 候補化により負例が非現実的になりにくい
  - match_gt 報酬は模倣設定に近く（挙動模倣/分類タスクに近い）、再現率は高く出やすい
- 一方で generalization の観点では、候補生成や特徴に依存したバイアスが残る可能性
  - 真の意思決定要因（プロジェクト/パス親和性、過去の関係性など）を候補別特徴に注入し、難易度を上げて検証が必要

## リスクと妥当性

- リーク防止:
  - タスク時刻より「前」の候補別最新 state を使用（後方の情報は未使用）
  - 時系列 split（cutoff）で train/eval を分離
- バイアス:
  - 候補生成（time-local）により対象期間周辺の活動者が候補に入りやすい（現実的だが易化要因にもなり得る）
  - 正解集合が「実参加者」のため、複数正解があるケースでは単純一致率が高くなる傾向

## 次のアクション

1. IRL 連携（Task×Reviewer 特徴での IRL 報酬）
   - 環境の `reward_mode` を `irl_softmax`/`accept_prob` へ切替、`irl_model`（θ/スケーラ）を注入
   - これに合わせて IRL 学習用のオフラインデータ（タスク × 候補）ビルダーを追加
2. 候補別特徴の拡充
   - 活動量/負荷、オーナーとのペア関係、パス親和性（overlap/jaccard/…）を注入（既存の per-review-request 特徴群を再利用）
3. 公平性・難易度調整
   - 候補生成にプロジェクト/パス/履歴近傍条件を組み合わせ、多様な負例を構成
   - 期間ごとの generalization（さらに先の窓 18m, 24m）
4. ベースライン比較
   - 「常に source reviewer を選ぶ」等の単純ベースラインとの比較
   - 上位 k 選択や確率較正（ECE）評価

## 成果物一覧

- Unified: `data/processed/unified/all_reviews.json`
- Sequences: `outputs/irl/reviewer_sequences.json`
- Tasks: `outputs/task_assign/tasks_{train,eval}.jsonl`, `tasks_meta.json`
- Policy: `outputs/task_assign/task_assign_policy.pt`, `train_log.json`
- Eval: `outputs/task_assign/replay_eval_post.json`

## 品質ゲート（スモーク）

- Build/Lint: スクリプト実行に致命的例外なし（uv で一連のフローが完走）
- データ件数: eval steps は窓に応じ 2,284→18,890 と十分（小規模時の不安定性を解消）
- 再現性: コマンドは上記の通り。候補生成は `seed=42`、time-local 窓は 60 日

# 1 raw -> unified
uv run python scripts/offline/unify_gerrit_changes.py \
  --inputs "data/raw/gerrit_changes/*detail*.json" \
  --out data/processed/unified/all_reviews.json

# 2 unified -> reviewer_sequences
uv run python scripts/extract_reviewer_sequences.py \
  --output outputs/irl/reviewer_sequences.json

# 3 sequences -> tasks（候補シャッフルは既定ON）
uv run python scripts/offline/build_task_assignment_from_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2024-07-01T00:00:00Z \
  --outdir outputs/task_assign \
  --max-candidates 8 \
  --candidate-sampling time-local \
  --candidate-window-days 60

# 4 学習（簡易方策）
uv run python scripts/training/rl_training/train_task_assignment.py \
  --train-tasks outputs/task_assign/tasks_train.jsonl \
  --eval-tasks outputs/task_assign/tasks_eval.jsonl \
  --outdir outputs/task_assign \
  --episodes 200

# 5 評価（時間窓 + 監査指標）
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign/tasks_eval.jsonl \
  --cutoff 2024-07-01T00:00:00Z \
  --policy outputs/task_assign/task_assign_policy.pt \
  --windows 1m,3m,6m,12m \
  --out outputs/task_assign/replay_eval.json

---

本レポートは、タスク中心 MDP の初期到達点（高再現度）を整理したものです。次は IRL 報酬の導入と候補別特徴の拡充で、より難度の高い一般化検証へ進みます。
