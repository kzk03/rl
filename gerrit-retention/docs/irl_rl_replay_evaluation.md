# IRL→RL リプレイ評価ガイド

本ドキュメントは、逆強化学習（IRL）で得た報酬関数を用いて強化学習（RL）方策を評価し、過去履歴（オフライン軌跡）をどの程度再現できているか（エピソード一致率など）を整理する実務ガイドです。

## 目的と全体像

- 目的: OSS 開発履歴を元に学習した IRL 報酬で方策を構築し、実際の過去行動に対してどの程度一致するかを測る。
- 全体像:
  - 1. 実履歴 → オフライン RL データセット化（(s,a,r,s',done) JSONL）
  - 2. IRL 学習（状態 × 行動 → 報酬/継続確率）
  - 3. IRL 報酬による方策（貪欲 or PPO）を適用
  - 4. 履歴の行動と一致率（ステップ/エピソード）を評価

---

## 使用する IRL（逆強化学習）

本リポジトリには 2 系統の IRL が存在します。リプレイ評価で主に利用するのは後者のニューラル IRL です。

- MaxEntBinaryIRL（レビュア応答の二値 IRL）

  - 実装: `src/gerrit_retention/irl/maxent_binary_irl.py`
  - 入力: 依頼 → 応答/不応答の時系列遷移（Engage=1/Idle=0）
  - 出力: Engage 確率、ロジット、重み（`explain_weights()`）
  - 用途: 応答性の分析・特徴化（本評価の補助）

- RetentionIRLSystem（状態 × 行動 → 報酬/継続確率）
  - 実装: `src/gerrit_retention/rl_prediction/retention_irl_system.py`
  - 入力: state ベクトル（連続）、action One-hot（離散）。config で `state_dim` と `action_dim`（本系では 3）を指定
  - 出力: `(predicted_reward, continuation_prob)`
  - 学習: 本ドキュメントでは、オフライン (state, action) から簡易教師で学習する補助スクリプトを利用
    - スクリプト: `scripts/training/offline_rl/train_retention_irl_from_offline.py`
    - 教師づけ: 真の行動=報酬 1/継続 1、その他行動=報酬 0/継続 0.5 を少数サンプル
    - 損失: MSE(報酬)+BCE(継続)

注意:

- リプレイ評価スクリプトは `action_dim=3`（reject/accept/wait）を前提とします。
- `state_dim` は既定 20（環境の観測次元）。不一致時はゼロパディング/トリムで暫定対応。

---

## オフライン RL データセット

- 生成: `src/gerrit_retention/offline/offline_dataset.py`
  - 入力: `extended_test_data.json` の開発者アクティビティ（commit / review）
  - マッピング（既定）:
    - review.score >= +1 → accept(1)
    - review.score <= -1 → reject(0)
    - 上記以外 → wait(2)
    - commit 等の非レビューは通常除外（オプションで wait として含め可）
  - 出力: `outputs/offline/dataset_{train,eval}.jsonl`（各行: `{timestamp, state[20], action(0/1/2), reward, next_state, done}`）

---

## 強化学習（RL）環境の要点

- 環境: `src/gerrit_retention/rl_environment/review_env.py`（Gymnasium 互換）
- 行動（3 種類）
  - 0: 拒否（reject） / 1: 受諾（accept） / 2: 待機（wait）
- 観測（20 次元, 0-1 範囲に正規化）
  - [0-4] 開発者状態: 専門性、ストレス、負荷、満足度、沸点余裕
  - [5-9] レビュー特徴: 複雑度、規模、緊急度、専門適合、関係性（先頭レビュー）
  - [10-14] 時間特徴: 時刻、曜日、エピ進行度、締切余裕、直近受諾率
  - [15-19] 協力/負荷: 協力品質、キュー充填率、直近受諾率、学習速度、負荷バランス
- 報酬（概略）
  - accept: 基本報酬 + 継続ボーナス（decay/window） + ストレス項 ± + 品質項 + 協力項 − 過負荷ペナルティ
  - reject: 基本ペナルティ + 高ストレス時の回避報酬 + 不適合の正当化 + 緊急度ペナルティ
  - wait: 基本ペナルティ + 情報価値（高複雑度） − 緊急度ペナルティ
- 実データ供給
  - `TimeSplitDataWrapper` で `extended_test_data.json` を時系列分割し、キューへ供給
  - ランダムレビュー投入/初期キューを無効化して「データ駆動」に切替可

### IRL 報酬の適用

- ラッパ: `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`
  - mode: `replace`（置換）/ `blend`（線形結合: `alpha*IRL + (1-alpha)*original`）
  - 受諾ボーナス（任意）: `engagement_bonus_weight` で調整

---

## リプレイ評価（エピソード一致率）

- スクリプト: `scripts/evaluation/irl_rl_replay_eval.py`
- 入力:
  - オフライン評価データ: `outputs/offline/dataset_eval.jsonl`
  - IRL モデル: `RetentionIRLSystem` で保存した `.pth`
- 方策: IRL 予測報酬を各行動に対して計算し、argmax を貪欲選択（同値時 tie-break: first/random）
- 指標:
  - overall_action_match_rate: 全ステップ一致率
  - episode_action_match_rate_mean/std: 各エピソードの一致率の平均/標準偏差
  - exact_episode_match_rate: エピソード内の全ステップが一致した割合
  - confusion_matrix: 3×3（[gt][pred]）
- エピソード境界: サンプルの `done == True` で区切る（末尾に `done` 無しの残片は 1 エピソードとして扱う）

---

## クイック実行例

IRL モデルの簡易学習（オフライン学習）→ リプレイ評価まで。

```bash
# 1) IRL（RetentionIRLSystem）をオフライン学習（短時間例）
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/dataset_train.jsonl \
  --out outputs/irl/retention_irl_offline.pth \
  --epochs 10 \
  --batch-size 2048 \
  --negatives-per-pos 2 \
  --hidden 128 \
  --lr 0.001

# 2) IRL→RL リプレイ評価（エピソード一致率の算出）
uv run python scripts/evaluation/irl_rl_replay_eval.py \
  --offline outputs/offline/dataset_eval.jsonl \
  --irl-model outputs/irl/retention_irl_offline.pth \
  --out outputs/irl/irl_rl_replay_report.json \
  --tie-break first
```

出力: `outputs/irl/irl_rl_replay_report.json`（指標と混同行列を含む）

---

## よくある注意点 / トラブルシュート

- IRL の `action_dim` は 3 必須（reject/accept/wait）。異なる場合は学習時 config を合わせる。
- `state_dim` 不一致は評価側でパディング/トリムするが、原則は 20 に揃えることを推奨。
- 一致率が低い場合:
  - 学習エポック/バッチサイズ/負例サンプル数を増やす
  - 正則化やドロップアウトの導入、学習率スケジュール調整
  - 特徴拡張（時間・締切・相互作用項など）や標準化を検討
- データが極端に偏っている場合は、混同行列やクラス別指標（Precision/Recall）を確認

---

## 次の拡張案

- PPO 学習（IRL 報酬使用）: `scripts/training/rl_training/stable_ppo_training.py` を IRL ラッパ付きで実行し、
  - IRL 貪欲方策 vs PPO 方策の一致率・報酬・行動分布を比較
- 分解評価: プロジェクト/レビュア/期間別の一致率、行動別 Precision/Recall、エピソード長別比較
- ブレンド比 `alpha` や受諾ボーナスのスイープで再現性を最適化

---

## 参考コードへのリンク

- IRL（MaxEnt 二値）: `src/gerrit_retention/irl/maxent_binary_irl.py`
- IRL（RetentionIRLSystem）: `src/gerrit_retention/rl_prediction/retention_irl_system.py`
- RL 環境: `src/gerrit_retention/rl_environment/review_env.py`
- IRL 報酬ラッパ: `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`
- オフラインデータ生成: `src/gerrit_retention/offline/offline_dataset.py`
- オフライン IRL 学習: `scripts/training/offline_rl/train_retention_irl_from_offline.py`
- リプレイ評価: `scripts/evaluation/irl_rl_replay_eval.py`

---

以上。IRL→RL のリプレイ評価を起点として、PPO 学習や分解評価に拡張することで、過去履歴の再現性と一般化挙動をより深く検証できます。

---

## 付録: IRL→RL は one-shot か？ RL の時系列性は？

結論: 本流（IRL 学習 → RL 方策学習）は one-shot ではありません。いずれも反復最適化です。

- IRL 学習（RetentionIRLSystem/MaxEnt 系）

  - 複数エポックの学習で損失（例: MSE(報酬)+BCE(継続)）を最小化する反復過程です。
  - 学習後の推論自体は 1 ステップの forward ですが、モデル獲得までは反復的です。

- RL 方策学習（PPO など）

  - IRL で得た報酬関数を環境に適用し、エピソードを跨いだ多ステップ遷移の中で方策を更新します。
  - 価値推定・GAE・クリップ損失など、反復更新で収束を目指します。

- リプレイ評価の IRL 貪欲方策（本書の `irl_rl_replay_eval.py`）
  - これは「各ステップの state に対し IRL 予測報酬の argmax を選ぶ」という one-shot 推論の繰り返しです（方策学習は行わない）。
  - 目的は「IRL の出力だけで履歴行動にどれだけ一致するか」を素早く測るベースライン提供です。

### RL のステップは時系列に沿って遷移する？

はい。`ReviewAcceptanceEnvironment` は `done`/`truncated` まで time-step を進める逐次遷移（MDP/POMDP 風）です。

- `step(action)` を呼ぶと、
  - 状態（観測ベクトル 20 次元）が次状態へ更新され、
  - 報酬が返り、
  - エピソード終了（`terminated`/`truncated`）判定が行われます。
- `TimeSplitDataWrapper` などを用いると、実データの時系列でキュー/状態が供給され、より履歴に沿った遷移を再現できます。
- 一方、リプレイ評価の IRL 貪欲方策は、各ステップで one-shot に行動を選ぶだけで、RL 的な方策更新は行いません（「単発の予測を時系列に沿って繰り返している」イメージ）。
