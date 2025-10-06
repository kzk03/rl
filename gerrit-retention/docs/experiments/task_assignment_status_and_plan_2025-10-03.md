# タスク中心 MDP の現状サマリと今後の方針（2025-10-03）

作成者: @kzk03

## 現状サマリ

- データパイプライン（再現可能・uv 実行）
  - raw Gerrit JSON → 統合: `scripts/offline/unify_gerrit_changes.py`
  - 統合 → レビュア系列: `scripts/extract_reviewer_sequences.py`
  - 系列 → タスク（Task× 候補レビュアー）: `scripts/offline/build_task_assignment_from_sequences.py`
    - 位置バイアス削減のため「候補シャッフル（決定的）」を既定 ON で実装済み
    - 候補特徴は「タスク時刻より前の各候補の最新 state」を利用（リーク防止）
- 強化学習の実行
  - 環境: `MultiReviewerAssignmentEnv`（K 択一、action_mask 対応）
  - 学習: `scripts/training/rl_training/train_task_assignment.py`（簡易 MLP、模倣に近い学習）
  - 評価: `scripts/evaluation/task_assignment_replay_eval.py`（時間窓、tqdm、監査指標）
- 最新の規模感
  - tasks_train.jsonl ≈ 139,649 行、tasks_eval.jsonl ≈ 23,261 行
  - max_candidates = 8、candidate_sampling = time-local（±60 日）、shuffle_candidates = true

## 評価結果（候補シャッフル導入後）

- 代表結果（post-cutoff 窓）
  - 1m: steps=2,284, action_match≈0.119, index0_positive_rate≈0.121, random_top1≈0.125
  - 3m: steps=6,154, action_match≈0.119, index0_positive_rate≈0.123, random_top1≈0.125
  - 6m: steps=10,450, action_match≈0.123, index0_positive_rate≈0.120, random_top1≈0.125
  - 12m: steps=18,892, action_match≈0.122, index0_positive_rate≈0.120, random_top1≈0.125
- 所見
  - シャッフル前の「一致率 ≈0.986」は位置バイアスの影響が大きかったと判断。
  - シャッフル後は Top-1 一致率がランダム基準（1/K=0.125）近傍に収束し、位置バイアスは抑制。
  - ただし現行の特徴・報酬設計では、まだランダム以上の識別が困難。

## 強化学習の報酬（現状）

- 環境設定: `reward_mode = 'match_gt'`
  - 選択した候補が ground-truth（実際に参加/受諾したレビュアー集合）に含まれれば +1.0、そうでなければ 0.0。
  - 候補が 1 人も正解でないタスクでは、どの行動も 0.0（学習信号が弱い）。
  - 無効アクション（パディング）に対してのみ小負報酬（`invalid_action_penalty=-0.1`）。
- 影響
  - 分類に近い模倣型の設定で、複数正解や正解欠如タスクの扱いが課題。
  - 情報量の少ない特徴 + 易しい候補集合では、性能がランダム付近に留まりがち。

## 逆強化学習（IRL）の活用状況

- 現状：タスク中心 MDP の学習に IRL は未統合。
  - 既存の IRL 実装は「レビュア受諾（3 クラス →2 クラス）」側で動作実績あり。
  - タスク × 候補レビュアーの文脈では、IRL 用のデータセット（state, action, candidate set, 行動確率）構築が未完。
- 予定している統合方法
  1. オフライン行動ログから Task×Candidate の特徴で IRL を学習（MaxEnt/ロジスティック等）
  2. 学習した IRL モデルの効用/確率を環境報酬に置換
     - `reward_mode='irl_softmax'`：候補集合上の softmax 確率を報酬化
     - `reward_mode='accept_prob'`：シグモイド確率（連続報酬）
  3. 同一特徴順序（feature_order）・スケーリング一貫性を担保
  4. 校正（温度/Platt 等）と ECE 監視で過信を抑制

## これからの方針（短期 → 中期）

- 短期（すぐに着手）

  - 候補難易度の引き上げ（hard negatives）
    - 同プロジェクト・パス親和・過去関係（owner↔reviewer）を優先して負例候補を生成
    - time-local のみ依存を緩和し、多様で紛らわしい候補集合を形成
  - 指標拡充と監査
    - Top-k（k=3/5）、mAP、ECE 追加。index ヒスト、candidate[0]正解率の監査継続
    - 「正解が候補集合に含まれる率」や「正解空タスク比率」の定点計測
  - データ健全性
    - タスク時刻より未来情報の不使用を再検証（すでに前向き切り）、seed 固定で再現

- 中期（次のスプリント）
  - IRL の統合
    - Task×Candidate の IRL データビルダー実装 → IRL 学習 → `irl_softmax/accept_prob` で RL を訓練
  - 設定の厳格化
    - review request ベースのタスク化（owner→reviewer）へ切替、より実務に沿う問題設定に
  - フェアネス・一般化検証
    - 時系列分割 + メール（開発者）単位のグループ分割
    - プロジェクト/期間別の比較、アブレーションで特徴の寄与を検証

## 既存実装の使い分け（要点）

- 候補シャッフル（位置バイアス対策）
  - 既定 ON。無効化は `--no-shuffle-candidates`（デバッグ用途のみ推奨）
- 報酬モード
  - いまは `match_gt` のみで学習・評価（模倣的）
  - IRL 統合後に `irl_softmax` / `accept_prob` を評価
- 欠損正解タスクの扱い
  - 正解空（positive なし）のタスクは学習信号が弱い → 訓練時の重み調整/フィルタ/報酬シェイピングを検討

## 参考: 直近の実行（任意）

```bash
# Raw -> Unified -> Sequences -> Tasks（shuffle ON）
uv run python scripts/offline/unify_gerrit_changes.py \
  --inputs "data/raw/gerrit_changes/*detail*.json" \
  --out data/processed/unified/all_reviews.json

uv run python scripts/extract_reviewer_sequences.py \
  --output outputs/irl/reviewer_sequences.json

uv run python scripts/offline/build_task_assignment_from_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2024-07-01T00:00:00Z \
  --outdir outputs/task_assign \
  --max-candidates 8 \
  --candidate-sampling time-local \
  --candidate-window-days 60

# Train & Evaluate
uv run python scripts/training/rl_training/train_task_assignment.py \
  --train-tasks outputs/task_assign/tasks_train.jsonl \
  --eval-tasks outputs/task_assign/tasks_eval.jsonl \
  --outdir outputs/task_assign \
  --episodes 200

uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign/tasks_eval.jsonl \
  --cutoff 2024-07-01T00:00:00Z \
  --policy outputs/task_assign/task_assign_policy.pt \
  --windows 1m,3m,6m,12m \
  --out outputs/task_assign/replay_eval.json
```

---

本メモの目的は、位置バイアスを除去したうえで現状の性能を正しく把握し、IRL の統合・候補難易度の引き上げ・評価指標の拡充という三段構えで「一般化可能な方策学習」に進める道筋を明確化することです。
