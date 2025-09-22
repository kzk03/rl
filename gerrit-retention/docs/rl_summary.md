# 現状の強化学習まとめ（Reviewer Assignment/Acceptance）

最終更新: 2025-09-22

## 概要

- 目的: 実ログに基づき、レビュー担当者の割当や受理方針を RL で学習。
- 学習対象:
  - 単一レビュアー受理環境（既存）
  - 複数候補からのレビュアー割当環境（新規）
- 手法: PPO（stable-baselines3）。
- 報酬: 2 系統
  - match_gt: 実際に参加したレビュアーと一致で+1
  - irl_softmax: 逆強化学習（条件付きロジット）の確率（候補集合ソフトマックス）を報酬とし、継続ボーナスを加算

## データと分割

- 入力: `data/processed/unified/all_reviews.json`（変更とレビュー履歴）
- 特徴抽出/サンプル化: `reviewer_invitation_ranking.build_invitation_ranking_samples`
- タスク化（複数候補）: `offline/build_assignment_tasks.py`
- 分割: 変更 ID（change_idx）単位の時系列分割（80/20）
  - `train_assignment_ppo.py` の `change_level_split` で実装
  - データが極小の場合はガード（train/eval を最低 1 変更ずつ確保、eval が空なら train を代用）

## 環境

### 単一レビュアー受理環境

- ファイル: `src/gerrit_retention/rl_environment/review_env.py`
- 特徴:
  - 実データのみで遷移（ランダム挿入 OFF）
  - IRL 報酬ラッパー（任意）
  - 継続ボーナス: 同一レビュアーの継続に指数減衰ボーナス（reject→ 次 accept でも継続可）

### 複数レビュアー割当環境（新）

- ファイル: `src/gerrit_retention/rl_environment/multi_reviewer_assignment_env.py`
- 仕様:
  - 観測: 候補 K 人 × 特徴をフラット化（必要ならアクションマスク付与）
  - 行動: Discrete(K)（選択したレビュアーのインデックス）
  - 終了: 1 変更=1 ステップ（タスク列を順に消化）
- 報酬モード:
  - `match_gt`: 選択が実参加者集合に含まれれば 1、それ以外 0
  - `irl_softmax`: IRL の θ/スケーラで候補集合のソフトマックス確率を算出し、その確率を報酬とする
  - 継続ボーナス: 同一レビュアーが再度選ばれたときに `weight * exp(-gap/tau)` を加算

## 逆強化学習（IRL）

- 実装: `src/gerrit_retention/recommendation/reviewer_invitation_ranking.py`
  - レベル 1: 条件付きロジット（候補集合ソフトマックス）
  - レベル 2: Plackett–Luce（Top-k の逐次ソフトマックス）
  - `evaluate_invitation_irl`（条件付きロジット）/ `evaluate_invitation_irl_plackett`（Plackett–Luce）が学習・評価し、共通して `features`, `scaler`, `theta` を返す
- RL 統合:
  - `train_assignment_ppo.py` で `--reward-mode irl_softmax` の場合に `evaluate_invitation_irl` を実行し、Env に `theta/scaler` を渡す

### IRL 2 パターンまとめ

- 条件付きロジット（Conditional Logit）

  - 前提: 各変更に対して「同時に」候補集合から 1 名（または招待者の分だけ独立に）選ばれると仮定。
  - 学習: グループごとに候補の線形効用 `u = w^T x + b` を推定し、softmax 確率を対数尤度最大化で最適化。
  - 出力: `theta`（重み+切片）、`scaler`、`features`。
  - 使い所: 「最初の 1 名」や「同時招待の確率」を近似したい場合のベースラインに適する。

- Plackett–Luce（逐次）
  - 前提: 候補集合から「順番に」選択（without replacement）される過程を仮定（Top-k 選出の近似）。
  - 学習: 残り集合に対する softmax を逐次最適化する損失で `theta` を推定。
  - 出力: `theta`（重み+切片）、`scaler`、`features`。
  - 使い所: 実際に複数名を順に招待するプロセスの近似や、Top-k のランキング忠実度を高めたい場合。

注意: 現在の RL トレーナ（`train_assignment_ppo.py`）では IRL 報酬の取得は条件付きロジットを使用しています。Plackett–Luce 版の θ/スケーラも互換ですが、トレーナ側の呼び出し切替（`evaluate_invitation_irl_plackett` を利用）を追加することで利用可能です。

### 行動の定義と IRL の対象（2 系統）

1. レビュアー推薦を「行動」にする（割当アクション）

- 何を学ぶか: 候補集合から誰を招待/割り当てるか（Discrete(K)）。
- IRL の役割: 候補ごとの効用を学習（Conditional Logit/Plackett–Luce）。RL への報酬として softmax 確率を利用（+継続ボーナス）。
- 実装ポイント:
  - データ: `build_invitation_groups_irl` が変更ごとの候補集合を構成。
  - IRL 学習/推論: `evaluate_invitation_irl` または `evaluate_invitation_irl_plackett`。
  - RL 環境/学習: `multi_reviewer_assignment_env.py` ＆ `training/rl_training/train_assignment_ppo.py`。

2. 「推薦後にレビューするか否か」を「行動」にする（受理/参加アクション）

- 何を学ぶか: そのレビュアーがレビューを継続（参加）するか否か（Binary）。
- IRL の役割: 受理/継続の報酬近似（MaxEntBinaryIRL や RetentionIRLSystem）を学習し、環境報酬を置換/混合。
- 実装ポイント:
  - IRL 本体: `src/gerrit_retention/irl/maxent_binary_irl.py`、`src/gerrit_retention/rl_prediction/retention_irl_system.py`。
  - 環境ラップ: `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`（`ReviewAcceptanceEnvironment` の報酬を IRL で置換/加重）。
  - RL 学習: `training/rl_training/stable_ppo_training.py`（単一レビュアー受理環境）。

選び方の目安

- 最適化したい対象が「誰に依頼するか」なら 1) を使用。
- 「依頼後に参加してもらえるか（継続性/負荷/相性）」を強く建模したいなら 2) を使用。
- 将来的には、上位（割当）× 下位（受理）を組み合わせた 2 段構成も可能。

## 学習/評価

- 単一レビュアー（従来）: `training/rl_training/stable_ppo_training.py`
- 複数割当（新）: `training/rl_training/train_assignment_ppo.py`
  - 主要引数:
    - `--changes`: 入力 JSON（デフォルト `data/processed/unified/all_reviews.json`）
    - `--max-candidates`: 候補上限 K
    - `--timesteps`: 学習総ステップ数
  - `--reward-mode`: `match_gt` | `irl_softmax` | `accept_prob`
    - `--continuity-weight`, `--continuity-tau`: 継続ボーナスの重みと減衰
    - `--output`: 出力先ディレクトリ
  - 出力: `eval_summary.txt`（簡易評価）
  - 注意: 環境は「1 変更=1 ステップ」。評価のステップ数は評価用変更数に依存。

## 実行例

- GT 一致報酬（短時間スモーク）
  - `uv run python training/rl_training/train_assignment_ppo.py --timesteps 2000 --output outputs/assignment_rl_match_gt`
- IRL+継続ボーナス（短時間スモーク）
  - `uv run python training/rl_training/train_assignment_ppo.py --reward-mode irl_softmax --continuity-weight 0.2 --continuity-tau 3.0 --timesteps 2000 --output outputs/assignment_rl_irl`
- 受理確率ベース（IRL-受理/参加アクション）
  - `uv run python training/rl_training/train_assignment_ppo.py --reward-mode accept_prob --continuity-weight 0.2 --continuity-tau 3.0 --timesteps 2000 --output outputs/assignment_rl_accept`

## 現状の挙動と注意点

- 小規模データでは、評価が 1 ステップ（=1 変更）になることがあります。
- IRL 確率報酬は 0〜1 の値で、短評価では 0 に寄ることがあり異常ではありません。
- 継続ボーナスは同一レビュアー連続選択に効くため、評価対象が少ないと寄与しにくいです。

## 次の一歩（任意）

- 評価の拡充: 変更単位での Top-1 精度、MAP@K、NDCG などの追加
- 学習安定化: IRL 確率の温度パラメータ導入、報酬スケーリング、学習ステップ増加
- 個人最適化: レビュアー ID 埋め込みやパーソナライズ特徴の導入
- ログ出力強化: タスク/候補数の分布、報酬分布、継続ボーナス寄与の可視化
