# IRL/RL 特徴量と前処理マップ（src 配下）

最終更新: 2025-09-22

本ドキュメントは、src 内の IRL/RL で実際に使用している特徴量名、算出方法、前処理ルール、ならびに関連ファイルの対応関係を一覧化します。

## 対象モジュール

- 招待ランキング（誰を招待/割当するか）
  - `src/gerrit_retention/recommendation/reviewer_invitation_ranking.py`
- 招待後の受理（招待されたレビュアーが参加するか）
  - `src/gerrit_retention/recommendation/reviewer_acceptance_after_invite.py`
- 単一レビュアー受理環境（RL）
  - `src/gerrit_retention/rl_environment/review_env.py`, `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`
- 複数候補割当環境（RL）
  - `src/gerrit_retention/rl_environment/multi_reviewer_assignment_env.py`

---

## 特徴量一覧（共通の順序）

以下は `InvitationRankingModel().features` で定義され、受理モデルでも同じ順序を再利用します。

- `reviewer_recent_reviews_7d`: 作成時刻直近 7 日での当該レビュアー参加回数（歴史から算出）
- `reviewer_recent_reviews_30d`: 直近 30 日での参加回数
- `reviewer_gap_days`: 直近の参加からの経過日数（参加履歴がないときは 999）
- `reviewer_total_reviews`: 作成時点までの累積参加回数
- `reviewer_proj_prev_reviews_30d`: 同一プロジェクトで直近 30 日に参加した回数
- `reviewer_proj_share_30d`: `proj_prev_30 / recent_30`（recent_30=0 のとき 0）
- `change_current_invited_cnt`: その変更に既に招待されている人数
- `reviewer_active_flag_30d`: 直近 30 日で 1 回以上参加なら 1、それ以外 0
- `reviewer_pending_reviews`: 作成時点で「開始済みだが未クローズ」の参加件数（リーケージ防止の時間条件で算出）
- `reviewer_night_activity_share_30d`: 直近 30 日の夜間（22-5 時）の参加比率（夜間回数/recent_30）
- `reviewer_overload_flag`: `recent_30 >= (global_mean + global_std)` かつ recent_30>0 なら 1
- `reviewer_workload_deviation_z`: `(recent_30 - global_mean) / (global_std + 1e-6)`（std=0 のとき 0）
- `match_off_specialty_flag`: 直近 30 日に同一プロジェクト参加が 0 なら 1（専門外）
- `off_specialty_recent_ratio`: `1 - (proj_prev_30 / recent_30)`（recent_30=0 のとき 0）
- `reviewer_file_tfidf_cosine_30d`: 変更のファイルトークンとレビュアー直近 30 日のファイルトークンの TF-IDF コサイン類似度

### ファイルトークンと IDF

- トークン化（`_path_tokens`）:
  - ディレクトリ名セグメント
  - ファイル名 stem を非英数境界で分割したトークン
  - 拡張子トークン（`ext:py` のように付与）
- IDF（グローバル）:
  - 平滑化: `idf = log((N+1)/(df+1)) + 1`
  - 招待ランキングでは `idf_mode='recent'` も選択可能（スライディングウィンドウ）。既定はグローバル。

---

## 前処理ルール（学習時）

- 標準化（`StandardScaler`）:
  - 招待ランキングのロジスティック回帰、受理モデルのロジスティック回帰の両方で、訓練データの特徴行列に対して適用（`scale=True`）。
  - RL 環境で IRL パラメータ（θ）を用いて効用を計算する際も、同じスケーラを適用して整合を取ります。
- クラス不均衡の重み付け:
  - `w_pos = 0.5 / pos_ratio`、`w_neg = 0.5 / (1 - pos_ratio)` をサンプル重みに適用。
- 時系列分割（評価）:
  - `ts` 昇順で 80/20 の時間分割。
  - 割当 RL 側は「変更単位（change_idx）」で分割し、1 変更=1 ステップを保証。
- ネガティブサンプリング（招待ランキング）:
  - `max_neg_per_pos`、`hard_fraction` に基づき、直近活動の多いレビュアーをハードネガに、残りはランダム補完。
  - 候補プールは `recent_days` と `min_total_reviews` でフィルタ。
- ファイル抽出/正規化:
  - 変更のファイル一覧は複数のフィールドから集約し、`COMMIT_MSG`/`MergeList*` を除外。
  - 抽出上限は 500 ファイル。

---

## 主要ファイルと依存関係

- 招待ランキング（誰を招待するか）

  - サンプル生成: `recommendation/reviewer_invitation_ranking.py` の `build_invitation_ranking_samples`
  - 学習/評価: `evaluate_invitation_ranking`（確率校正/ランキング指標）
  - IRL（候補集合ソフトマックス）: `build_invitation_groups_irl` と Conditional Logit 系（同ファイル内）
  - RL 連携:
    - 環境: `rl_environment/multi_reviewer_assignment_env.py`
    - 学習: `training/rl_training/train_assignment_ppo.py`（`--reward-mode irl_softmax`）
    - タスク化: `offline/build_assignment_tasks.py`
    - スクリプト: `scripts/run_reviewer_invitation_ranking.py`

- 受理（招待後に参加するか）

  - サンプル生成: `recommendation/reviewer_acceptance_after_invite.py` の `build_invitee_acceptance_samples`
  - 学習/評価: `evaluate_acceptance_after_invite`（ロジスティック、スケーラ/係数）
  - RL 連携:
    - 割当 RL の報酬: `--reward-mode accept_prob`（効用 u の `sigmoid(u)`）
    - 単一受理環境の報酬置換: `rl_environment/irl_reward_wrapper.py` + `rl_environment/review_env.py`
    - IRL（MaxEnt 二値）: `irl/maxent_binary_irl.py`、統合: `rl_prediction/retention_irl_system.py`
    - スクリプト: `scripts/run_reviewer_acceptance_after_invite.py`

- 共有/補助
  - 特徴順序の共有: 受理モデルは `InvitationRankingModel().features` を再利用（順序厳密一致）
  - スケーラ/係数の受け渡し: 環境に `{theta, scaler}` を渡し、同順序でベクトル化して効用算出
  - 分割補助: `rl_environment/time_split_data_wrapper.py`

---

## RL 環境での利用（効用 → 報酬）

- `irl_softmax`（割当環境）:
  - 候補ごとの効用 `u_i` を計算し、`p_i = softmax(u)` を報酬（継続ボーナスを加算可）。
  - 数値安定化のため max-shift を実施（`u - max(u)`）。
- `accept_prob`（割当環境）:
  - 単一候補の効用 `u` に対して `sigmoid(u)` を報酬。
- 単一受理環境:
  - IRL 報酬ラッパーで MaxEnt IRL スコア等を組み込み、同一レビュアーの継続ボーナス（`weight * exp(-gap/tau)`）を付与。

---

## プログラムから特徴名を取得

```python
from gerrit_retention.recommendation.reviewer_invitation_ranking import InvitationRankingModel
print(InvitationRankingModel().features)
```

---

## 備考

- 特徴の欠損は `0.0` で埋めています（ベクトル化時）。
- プロジェクト/ファイル由来のバイアスを軽減するため、IDF と最近傾向を併用しています。
- 受理/割当の 2 段構成も可能で、その場合は割当で選んだ候補の受理確率を次段の報酬やフィルタに活用します。
