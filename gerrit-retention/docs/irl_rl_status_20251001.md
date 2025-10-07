# IRL/RL 現状サマリ（2025-10-01）

本ドキュメントは、現行の IRL→RL 実験の状態を「どのデータを使っているか」「環境（ストレスなど）の設定」「学習済みモデル/評価結果」「次アクション」の観点で整理したものです。

---

## 使用データと分割条件

- ソースデータ
  - TimeSplit 用: `outputs/irl/reviewer_sequences.json`
  - TimeSplit 切替（例）: `cutoff = 2025-01-01T00:00:00Z`
    - train フェーズ: cutoff より過去のイベント
    - eval フェーズ: cutoff より未来のイベント
- オフライン評価データ（3 クラス化: wait を注入）
  - 生成スクリプト: `scripts/offline/build_offline_from_reviewer_sequences.py`
  - 生成パラメータ（使用例）: `--inject-wait-prob 0.15 --seed 7 --cutoff 2025-01-01T00:00:00Z`
  - 出力:
    - Train JSONL: `outputs/offline/expanded_wait/dataset_train.jsonl`（5,505 行）
    - Eval JSONL: `outputs/offline/expanded_wait/dataset_eval.jsonl`（12,901 行）
  - Eval のクラス分布（混同行列の行合計より）
    - reject: 518（約 4.0%）
    - accept: 10,719（約 83.1%）
    - wait: 1,664（約 12.9%）

> 3 クラス化は実履歴の「accept 偏重」を維持しつつ、wait 行動を人工的に追加するため `inject_wait_prob` を用いています。

---

## 環境設定（ストレス・終了条件など）

- 強化学習環境: ReviewAcceptanceEnvironment（行動: 0=reject, 1=accept, 2=wait）
- データ供給: TimeSplitDataWrapper（train/eval でイベント供給を切替）
- 終了・切り詰めの代表例（ログ）
  - 「開発者が沸点に達しました - エピソード終了」
  - 「開発者満足度が極端に低下しました - エピソード終了」
  - 「最大ステップ数 100 に達しました - エピソード切り詰め」

> 評価フェーズでは `phase='eval'` に切り替わり、cutoff 以後のイベントを利用します。

---

### 環境デフォルト報酬とは？

ここで言う「環境デフォルト報酬」は、IRL を使わずに環境（`ReviewAcceptanceEnvironment`/`reward_calculator`）が内部ルールで計算するベース報酬です。主な特徴は次の通りです（コードより要約）。

- 共通要素
  - ストレスしきい値（`stress_threshold=0.8`）を超えると過負荷ペナルティが乗る
  - 毎ステップ、ストレスは自然減衰（`natural_stress_decay ≈ 0.01`）
  - エピソード進行やキュー状況に応じた小さな継続性/緊急度成分（urgency/continuity）
- accept（受諾）
  - ベース報酬に加え、専門性マッチ時はストレス軽減寄り（例: +0.2）、ミスマッチ時はペナルティ寄り（例: -0.4）
  - ミスマッチ受諾ではストレスが蓄積（例: +0.1）
  - 高ストレス時は過負荷ペナルティが上乗せ
- reject（拒否）
  - ベースはペナルティだが、高ストレス時はストレス緩和ボーナス（例: +0.3）
  - 緊急度/専門性に応じた小さな加減項
- wait（保留）
  - 緊急度に応じた小さな報酬/ペナルティ
  - 高ストレス時は微小な緩和方向、低ストレス時は機会損失ペナルティ寄り

この「環境デフォルト報酬」に対し、IRL 報酬 PPO では IRL 報酬を blend/replace で組み合わせて学習します。

---

## IRL 学習

- スクリプト: `scripts/training/offline_rl/train_retention_irl_from_offline.py`
- 入力: `outputs/offline/expanded_wait/dataset_train.jsonl`
- 主な設定（抜粋）
  - `epochs = 3`, `batch_size = 1024`, `negatives_per_pos = 2`
  - モデル: `state_dim=20, action_dim=3, hidden_dim=128`（Retention IRL System）
- 学習ログ（平均損失）
  - Epoch 1: 0.9054 → Epoch 2: 0.8490 → Epoch 3: 0.8095
- 保存物
  - `outputs/irl/retention_irl_offline_expanded_wait.pth`

---

## リプレイ評価（オフライン JSONL 上）

- 評価スクリプト（IRL 貪欲）: `scripts/evaluation/irl_rl_replay_eval.py`
- 評価スクリプト（保存 PPO）: `scripts/evaluation/policy_replay_match_eval.py`
- 評価セット: `outputs/offline/expanded_wait/dataset_eval.jsonl`（12,901 行 / 1,466 エピソード）

### 1) IRL-greedy（報酬貪欲で argmax）

- モデル: `outputs/irl/retention_irl_offline_expanded_wait.pth`
- 結果
  - overall_action_match_rate: 0.8309
  - episode_action_match_rate_mean/std: 0.8798 / 0.1278
  - exact_episode_match_rate: 0.4059
  - 混同行列（gt 行 × pred 列）
    - reject: [0, 518, 0]
    - accept: [0, 10719, 0]
    - wait: [0, 1664, 0]
- 備考
  - すべてを accept 予測。accept が 8 割超のため、この単純方策が高い一致率を示す。

### 2) 既存 PPO ポリシー（過去保存）

- 同 eval に対して IRL-greedy と同等（ほぼ全 accept）。
  - 訓練時の報酬源: 環境デフォルト報酬（IRL なし）
  - 傾向: 実データ分布（accept 優勢）に引っ張られ、ほぼ全 accept に収束

### 3) 新規 Baseline PPO（IRL 報酬なし）

- 学習: `stable_ppo_training.py`（IRL 報酬 off）
- ポリシー: `outputs/policies/stable_ppo_policy_20251001_144522.pt`
- リプレイ結果
  - overall_action_match_rate: 0.5876
  - exact_episode_match_rate: 0.1112
  - 混同行列
    - reject: [0, 93, 425]
    - accept: [0, 7055, 3664]
    - wait: [0, 1138, 526]
- 備考
  - accept と wait を混在予測（reject は出さない）。分布からの乖離で一致率が低下。

### 4) IRL 報酬 PPO（例: blend α=0.8, engagement=0.05）

- ポリシー: `outputs/policies/stable_ppo_policy_20251001_144718.pt`
- リプレイ結果
  - overall_action_match_rate: 0.4273
  - exact_episode_match_rate: 0.1392
  - 混同行列
    - reject: [388, 130, 0]
    - accept: [5595, 5124, 0]
    - wait: [887, 777, 0]
- 備考
  - reject/accept のみ（wait 無し）。reject 過多で一致率が低下。
  - 訓練時の報酬源: IRL 報酬と環境報酬のブレンド（α=0.8）+ 受諾行動への engagement ボーナス（0.05）。既存 PPO との最大の違いは「IRL 由来の報酬を用いるかどうか」。

### 5) IRL 報酬 PPO スイープ（軽量）

- α=0.4, engagement=0.1 → overall=0.1281, exact=0.00136（wait 過多）
- α=0.6, engagement=0.2 → overall=0.2856, exact=0.00136（wait 過多）

### 6) IRL-only PPO（replace モード・engagement=0.0）

- 学習: `stable_ppo_training.py --use-irl-reward --irl-reward-mode replace --engagement-bonus-weight 0.0`
- IRL モデル: `outputs/irl/retention_irl_offline_expanded_wait.pth`
- ポリシー: `outputs/policies/stable_ppo_policy_20251001_151401.pt`
- 訓練サマリ（短時間ラン）
  - train-episodes=150, eval-episodes=60
  - 訓練中行動分布（概況）: reject≈33%, accept≈35%, wait≈33%
- リプレイ結果（eval JSONL 上）
  - overall_action_match_rate: 0.8309
  - exact_episode_match_rate: 0.4059
  - 混同行列（gt 行 × pred 列）
    - reject: [0, 518, 0]
    - accept: [0, 10719, 0]
    - wait: [0, 1664, 0]
- 備考
  - 評価では全 accept に収束（IRL-greedy と同一挙動）。現行 IRL 報酬が accept を強く好むためと考えられる。

#### (補足) IRL-only 長時間学習（本格設定）

- IRL モデル: `outputs/irl/retention_irl_offline_expanded_wait_e30_np4.pth`（epochs=30, negatives_per_pos=4）
- 学習: `--irl-reward-mode replace --engagement-bonus-weight 0.0 --train-episodes 2000 --eval-episodes 400`
- ポリシー: `outputs/policies/stable_ppo_policy_20251001_151901.pt`
- リプレイ結果（offline eval JSONL 上）
  - overall_action_match_rate: 0.4398
  - episode_action_match_rate_mean/std: 0.4579 / 0.2337
  - exact_episode_match_rate: 0.0048
  - 混同行列（gt 行 × pred 列）
    - reject: [16, 60, 442]
    - accept: [37, 4614, 6068]
    - wait: [3, 617, 1044]
- 所見
  - 長時間学習により、訓練中の行動は reject/accept/wait が概ね 1/3 ずつに分散。評価では accept 優勢の実分布とミスマッチが拡大し、一致率は低下。
  - IRL-only で分布調整を図るには、IRL 側のクラスバランス/正則化強化、PPO 側の模倣正則化（分布 KL）や小さな accept ボーナスの導入が有効。

##### 2 クラス評価（wait と reject を統合）

- 目的: wait と reject の意味的な近さ（「非受諾」）を前提に、2 クラス（非受諾/受諾）での一致を確認
- 結果（IRL-only 長時間ポリシー）
  - overall_action_match_rate: 0.4743
  - episode_action_match_rate_mean/std: 0.4656 / 0.2314
  - exact_episode_match_rate: 0.0075
  - 2x2 混同行列（[gt][pred] = 非受諾/受諾）
    - 非受諾: [1505, 677]
    - 受諾: [6105, 4614]
- 所見: 2 クラスでも一致率は限定的。受諾多数の実分布に比べ、非受諾（特に wait）選好が相対的に強い。

---

## 直近の所見

- 本 eval セットは accept が 83% と支配的で、「全 accept」が非常に強いベースライン（overall ≈ 0.83）。
- 現状の IRL-greedy は全 accept に収束しており、IRL 報酬 PPO は逆に reject / wait へ振れやすい（パラメータ次第）。
- IRL の負例サンプリングや重み付けを見直し、wait/reject の識別力を少し高めつつ、PPO 側は accept に寄せるシェーピングが有効と推定。

---

## 次のアクション（提案）

1. IRL 報酬 PPO のパラメータ調整（短時間スイープ）
   - 例: blend α ∈ {0.90, 0.95}, engagement ∈ {0.3, 0.5}, mode=replace の比較も 1 本。
   - 目的: accept の比率を実分布に寄せ、overall を 0.6 以上まで回復。
2. IRL 学習の調整（軽量）
   - `negatives_per_pos`: 2 → 3〜4、クラス重みを追加（負例の重みをやや下げる等）。
   - エポック+2〜3 で安定化。
3. PPO への模倣正則化を少量追加（要小改修）
   - policy 出力とターゲット分布（{r≈0.04, a≈0.83, w≈0.13}）との KL を λ=0.02〜0.05 で付加。
   - 極端な reject / wait 偏重を抑制。

---

## 2025-10-07 Nova Task Assignment IRL 再現レシピ

> 目的: `outputs/task_assign_nova_cut2023` に保存した **新特徴量版 (2025-10-07)** と、`irl_model_prev.json` に対応する **旧ベースライン版** の両方を再現できるよう、作業ブックマークとコマンドをまとめる。

### 共通準備

- 依存関係: ルートで `uv sync` 済みであること。
- 入力データ: `outputs/irl/reviewer_sequences.json` と `data/processed/unified/all_reviews.json` が存在していること。
- すべてのコマンドは `uv run` で実行し、カレントディレクトリはリポジトリルートとする。

### A. 新特徴量版（HEAD=84c1fa1, registry 37 features）

1. ブランチを最新に揃える。

```bash
git checkout refactor
git pull
```

2. タスクデータセット生成。

```bash
uv run python scripts/offline/build_task_assignment_from_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2023-07-01T00:00:00 \
  --outdir outputs/task_assign_nova_cut2023 \
  --candidate-sampling time-local \
  --candidate-window-days 60 \
  --train-window-months 24 \
  --eval-window-months 24 \
  --project openstack/nova \
  --seed 42
```

- 出力: `tasks_train.jsonl` 12,493 件中 12,259 件が学習に利用。`tasks_meta.json` の `registry_features_loaded` が 37 件であることを確認。

3. IRL モデル再学習。

```bash
uv run python scripts/training/irl/train_task_candidate_irl.py \
  --train-tasks outputs/task_assign_nova_cut2023/tasks_train.jsonl \
  --out outputs/task_assign_nova_cut2023/irl_model.json \
  --iters 400 --lr 0.1 --reg 1e-4 --l1 0.001 --temperature 0.8
```

4. リプレイ評価（IRL 直接順位）。

```bash
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign_nova_cut2023/tasks_eval.jsonl \
  --cutoff 2023-07-01T00:00:00 \
  --irl-model outputs/task_assign_nova_cut2023/irl_model.json \
  --eval-mode irl \
  --out outputs/task_assign_nova_cut2023/replay_eval_irl.json
```

- 指標例（12m ウィンドウ）: Top-1=0.151, Top-3=0.396, Top-5=0.563, mAP=0.347。

### B. 旧ベースライン版（commit 515d1a8, registry 未導入）

> 新旧を同一ワークツリーで混在させないため、以下では **作業用ワークツリーを追加** する手順を推奨する。別ディレクトリの用意が難しい場合は、対象ファイルのみ `git checkout 515d1a8 -- <path>` で切り戻し、完了後に `git restore` で戻すこと。

1. ワークツリー作成または切り戻し。

```bash
git worktree add ../gerrit-retention-baseline 515d1a8
cd ../gerrit-retention-baseline
```

（単一ツリーで作業する場合）

```bash
git checkout 515d1a8
```

2. ベースラインタスク生成（出力先は衝突防止のため別ディレクトリに変更）。

```bash
uv run python scripts/offline/build_task_assignment_from_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2023-07-01T00:00:00 \
  --outdir outputs/task_assign_nova_cut2023_baseline \
  --candidate-sampling time-local \
  --candidate-window-days 60 \
  --train-window-months 24 \
  --eval-window-months 24 \
  --project openstack/nova \
  --seed 42
```

- このコミットではフィーチャーレジストリが存在しないため、各候補の特徴量は 17 次元（activity/gap/workload 系 + reviewer_tenure 等）のみ。

3. IRL モデル学習（係数 17 次元 + バイアス）。

```bash
uv run python scripts/training/irl/train_task_candidate_irl.py \
  --train-tasks outputs/task_assign_nova_cut2023_baseline/tasks_train.jsonl \
  --out outputs/task_assign_nova_cut2023_baseline/irl_model.json \
  --iters 400 --lr 0.1 --reg 1e-4 --l1 0.001 --temperature 0.8
```

4. リプレイ評価。

```bash
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign_nova_cut2023_baseline/tasks_eval.jsonl \
  --cutoff 2023-07-01T00:00:00 \
  --irl-model outputs/task_assign_nova_cut2023_baseline/irl_model.json \
  --eval-mode irl \
  --out outputs/task_assign_nova_cut2023_baseline/replay_eval_irl.json
```

- 指標例（12m ウィンドウ）: Top-1≈0.219, Top-3≈0.331, Top-5≈0.400, mAP≈0.337。

5. 現行ブランチへ戻る。

```bash
cd ../gerrit-retention   # worktreeを使った場合
git checkout refactor    # 単一ツリーの場合
git worktree remove ../gerrit-retention-baseline --force  # 任意
```

### C. 追加メモ

- 比較用に HEAD 側では `outputs/task_assign_nova_cut2023/irl_model_prev.json` を保持しており、旧モデルの θ を直接参照できる。
- 係数差分を再確認するには以下のスクリプトを利用。

  ```bash
  uv run python - <<'PY'
  import json
  from pathlib import Path
  new = json.loads(Path('outputs/task_assign_nova_cut2023/irl_model.json').read_text())
  old = json.loads(Path('outputs/task_assign_nova_cut2023/irl_model_prev.json').read_text())
  shared = [f for f in old['feature_order'] if f in new['feature_order']]
  print('feature,old,new')
  for name in shared:
    oi = old['feature_order'].index(name)
    ni = new['feature_order'].index(name)
    print(f"{name},{old['theta'][oi]:.3f},{new['theta'][ni]:.3f}")
  PY
  ```

- 指標比較サマリ

  | モデル                      | 特徴次元 | 12m Top-1 | 12m Top-3 | 12m mAP | eval JSON                                                        |
  | --------------------------- | -------- | --------- | --------- | ------- | ---------------------------------------------------------------- |
  | baseline (515d1a8)          | 17       | ≈0.219    | ≈0.331    | ≈0.337  | `outputs/task_assign_nova_cut2023_baseline/replay_eval_irl.json` |
  | feature-expansion (84c1fa1) | 41       | 0.151     | 0.396     | 0.347   | `outputs/task_assign_nova_cut2023/replay_eval_irl.json`          |

- 新特徴量版は Top-1 が低下する一方で Top-3/Top-5 や mAP が改善している。位置決定重みを維持したい場合は L1 を弱める、もしくは change\_\* 系特徴を RL 方策専用に切り出すのが次の改善案。

---

## 参考: 「2) 既存 PPO」と「4) IRL 報酬 PPO」の違い（要点）

- 報酬設計
  - 既存 PPO: 環境のデフォルト報酬のみ（IRL なし）
  - IRL 報酬 PPO: IRL 報酬を blend（α=0.8 例）で加味し、さらに accept に小さなボーナス（engagement）を付与
- 振る舞い
  - 既存 PPO: ほぼ全 accept（IRL-greedy と同様）
  - IRL 報酬 PPO: reject を相対的に多く選択（設定によっては wait 過多）しやすい
- 目的
  - 既存 PPO: 環境設計の報酬に忠実なベースライン
  - IRL 報酬 PPO: 実履歴から推定した IRL 報酬で行動傾向を再現・誘導

---

## 参考: IRL のエポック数について

- 現状の `epochs = 3` はスモークテストとしては妥当ですが、実用には少なめです。
- 推奨: `epochs = 15〜30` 程度 + 簡易 EarlyStopping（可能なら）/ あるいは固定エポックで複数保存し、リプレイ一致率でベスト選定。
- あわせて `negatives_per_pos` を 3〜4 に増やし、クラス重みを調整すると reject / wait の識別が安定しやすくなります。

再学習コマンド例（オプション）

```bash
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/expanded_wait/dataset_train.jsonl \
  --epochs 20 \
  --batch-size 1024 \
  --negatives-per-pos 3 \
  --out outputs/irl/retention_irl_offline_expanded_wait_e20_np3.pth
```

---

## 再現用メモ（任意）

> 実行環境: macOS, Python 3.11（uv 経由）。プロジェクトルートで実行。

- 3 クラスオフライン生成（例）

```bash
uv run python scripts/offline/build_offline_from_reviewer_sequences.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2025-01-01T00:00:00Z \
  --outdir outputs/offline/expanded_wait \
  --inject-wait-prob 0.15 \
  --seed 7
```

- IRL 学習（例）

```bash
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/expanded_wait/dataset_train.jsonl \
  --epochs 3 --batch-size 1024 \
  --out outputs/irl/retention_irl_offline_expanded_wait.pth
```

- IRL-greedy リプレイ（例）

```bash
uv run python scripts/evaluation/irl_rl_replay_eval.py \
  --offline outputs/offline/expanded_wait/dataset_eval.jsonl \
  --irl-model outputs/irl/retention_irl_offline_expanded_wait.pth \
  --out outputs/irl/expanded_wait_irl_greedy_replay.json
```

- 安定版 PPO 学習（IRL 報酬あり・例）

```bash
STABLE_RL_DATA=outputs/irl/reviewer_sequences.json \\
STABLE_RL_CUTOFF=2025-01-01T00:00:00Z \\
uv run python scripts/training/rl_training/stable_ppo_training.py \\
  --use-irl-reward \\
  --irl-model-path outputs/irl/retention_irl_offline_expanded_wait.pth \\
  --irl-reward-mode blend \\
  --irl-reward-alpha 0.8 \\
  --engagement-bonus-weight 0.05 \\
  --train-episodes 400 --eval-episodes 100
```

- PPO ポリシーのリプレイ一致率評価（例）

```bash
uv run python scripts/evaluation/policy_replay_match_eval.py \
  --offline outputs/offline/expanded_wait/dataset_eval.jsonl \
  --policy outputs/policies/stable_ppo_policy_YYYYMMDD_HHMMSS.pt \
  --out outputs/irl/expanded_wait_policy_replay_report.json
```

- IRL-only（replace・長時間）学習（例）

```bash
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/expanded_wait/dataset_train.jsonl \
  --epochs 30 --batch-size 1024 --negatives-per-pos 4 \
  --out outputs/irl/retention_irl_offline_expanded_wait_e30_np4.pth

STABLE_RL_DATA=outputs/irl/reviewer_sequences.json \\
STABLE_RL_CUTOFF=2025-01-01T00:00:00Z \\
uv run python scripts/training/rl_training/stable_ppo_training.py \\
  --use-irl-reward \\
  --irl-model-path outputs/irl/retention_irl_offline_expanded_wait_e30_np4.pth \\
  --irl-reward-mode replace \\
  --engagement-bonus-weight 0.0 \\
  --train-episodes 2000 --eval-episodes 400

uv run python scripts/evaluation/policy_replay_match_eval.py \
  --offline outputs/offline/expanded_wait/dataset_eval.jsonl \
  --policy outputs/policies/stable_ppo_policy_20251001_151901.pt \
  --out outputs/irl/expanded_wait_policy_replay_report_irl_only_e30_np4_long.json
```

---

## 付記（フェアネス/リーク対策）

- TimeSplit による時系列分割（cutoff 前後）での評価。
- オフライン評価は `dataset_eval.jsonl` を固定し、学習/推論とも未来情報の不使用を徹底。
- メールなど PII の扱いは外部送信せずローカルで完結。

---

以上。次の反復（パラメータスイープ / IRL 再学習 / 模倣正則化の追加）をご希望なら、このレポートに追記更新します。
