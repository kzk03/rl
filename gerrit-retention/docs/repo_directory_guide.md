# gerrit-retention ディレクトリ徹底ガイド（初学者向け）

最終更新: 2025-09-22

このドキュメントは、リポジトリの全体像と各ディレクトリの役割を初学者にも分かるように丁寧に解説します。実データ処理 → 特徴量 →IRL→RL→API/可視化までの流れを把握できます。

## 全体像（ざっくり）

- data: 入力データ（raw/external/processed）とサンプル JSON
- data_processing: データ抽出・前処理・特徴量生成
- src/gerrit_retention: 本体ライブラリ（IRL、推薦、RL 環境、可視化、ユーティリティ）
- training: 学習スクリプト（RL/評価含む）
- outputs/models/logs: 学習結果やモデル成果物
- api: FastAPI による API 実装
- docs: ドキュメント集（本ファイル含む）
- docker: 実行環境の Docker 定義
- configs/config: 設定ファイル群（環境別、モジュール別）

---

## ルート直下

- README.md: プロジェクト概要と基本の使い方
- pyproject.toml / setup.py: パッケージ設定
- uv.lock: 依存ロック
- coverage.xml / htmlcov/: テストカバレッジ関連
- データソース拡張計画.md: 将来のデータ拡張に関するメモ

## data/

- raw/: 生データ（Gerrit からのダンプ等）
- external/: 外部ソース由来の補助データ
- processed/: 前処理済み/統合データ（例: unified/all_reviews.json）
- sample_test_data.json / extended_test_data.json: 小規模検証用データ
- pipeline_history.json: パイプライン実行履歴

## data_processing/

- gerrit_extraction/: Gerrit からのデータ抽出
- preprocessing/: クリーニング、正規化、時系列整形
- feature_engineering/: 特徴量生成（例: 開発者の活動量、過去受理率など）

## src/gerrit_retention/（ライブラリ本体）

- cli.py: CLI エントリ（必要処理の呼び出し口）
- utils/: 共通ユーティリティ（ロギング、設定、時間処理など）
- data_integration/: 複数データソースの結合・整形
- analysis/, behavior_analysis/: 解析モジュール（行動/傾向分析）
- visualization/: 可視化（チャート/ダッシュボード生成）

### IRL（逆強化学習）関連: src/gerrit_retention/irl/

- maxent_binary_irl.py: MaxEnt 型の二値 IRL（受理/継続の確率を推定）

### 推薦/ランキング: src/gerrit_retention/recommendation/

- reviewer_invitation_ranking.py: 招待（誰に依頼するか）のランキング IRL（条件付きロジット/Plackett–Luce）
- reviewer_acceptance_after_invite.py: 招待後の受理確率モデル（ロジスティック等）
- reviewer_acceptance.py: 受理行動の分析/補助関数
- task_recommendation_pipeline.py: タスク推薦の一連処理

### RL 環境・学習: src/gerrit_retention/rl_environment/

- review_env.py: 単一レビュアー受理環境（実データ遷移、継続ボーナス、IRL ラッパー対応）
- multi_reviewer_assignment_env.py: 複数候補からの割当環境（K 人から 1 人選択、IRL 確率/受理確率報酬対応）
- irl_reward_wrapper.py: 受理環境の報酬を IRL で置換/加重するラッパー
- ppo_agent.py / reward_calculator.py: エージェント/報酬関連
- time_split_data_wrapper.py: 安定した学習/評価分割の補助

### オフライン/予測: src/gerrit_retention/offline/, prediction/

- offline/: ログからのオフラインデータセット生成（割当タスク化など）
- prediction/: 受理率や次アクションの予測器群

## training/

- rl_training/
  - stable_ppo_training.py: 単一受理環境での学習（PPO）
  - train_assignment_ppo.py: 割当環境での学習（PPO, IRL 確率/受理確率報酬, 変更単位分割）
- offline_rl/: オフライン RL や模倣学習（BC）
- 評価系: evaluation/ 配下も参照（AB テスト、統合テスト）

## evaluation/

- ab_testing/: AB テストシナリオ
- integration_tests/: 統合テスト
- model_evaluation/: 学習済みモデルの評価ユーティリティ

## examples/

- training_pipeline_example.py ほか: 使い方の最小例

## api/

- main.py: FastAPI エントリ
- routers/, services/, models/, core/, utils/: API のルーティング/ビジネスロジック/スキーマ/補助

## configs/ と config/

- configs/: 実行環境（development/production/testing など）やパイプライン構成
- config/: API やモデルのロギング/モデル設定（小規模まとめ）

## docker/

- Dockerfile / docker-compose\*.yml: コンテナ環境
- init-db.sql / redis.conf / nginx.conf / prometheus.yml: 周辺サービス設定

## outputs/, models/, logs/

- outputs/: 実行結果（評価レポート、可視化、RL の評価ログなど）
- models/: 学習済みモデル（.pth/.zip/.pkl）やチェックポイント
- logs/: 学習時のログ出力

---

## 代表的なワークフロー

1. データ準備 → 特徴量 →IRL→RL（割当）

- 原データ整形（入力）

  - 目的: Gerrit 実ログを学習用に整形
  - 場所: `data_processing/gerrit_extraction/`, `data_processing/preprocessing/`
  - 出力: `data/processed/unified/all_reviews.json`
  - 参考: `scripts/collect_real_data.py`, `scripts/run_full_training_pipeline.py`

- サンプル化（招待ランキング用特徴）

  - 目的: 変更ごとの候補レビュアー集合と特徴量を作る
  - 場所: `src/gerrit_retention/recommendation/reviewer_invitation_ranking.py`
  - コマンド例:
    ```bash
    uv run python scripts/run_reviewer_invitation_ranking.py \
      --input data/processed/unified/all_reviews.json \
      --output outputs/reviewer_invitation_ranking/
    ```
  - 生成物: 候補ごとの特徴・学習済み θ/スケーラ（IRL の場合）

- タスク化（割当タスクに再構成）

  - 目的: 1 変更=1 ステップの割当タスクへグルーピング
  - 場所: `src/gerrit_retention/offline/build_assignment_tasks.py`
  - 入力: 招待ランキングのサンプル
  - 出力: 変更単位のタスク（候補 K/正解集合/特徴）

- 学習（割当 RL: PPO）
  - 目的: K 候補から 1 人を選ぶ方策を学習
  - スクリプト: `training/rl_training/train_assignment_ppo.py`
  - 代表オプション:
    - `--reward-mode`: `match_gt` | `irl_softmax` | `accept_prob`
    - `--max-candidates`: 候補上限 K（不足分はパディング/切り詰め）
    - `--timesteps`: 学習総ステップ数（変更数に依存）
    - `--continuity-weight`, `--continuity-tau`: 継続ボーナスの強さ/減衰
  - 出力: `outputs/assignment_rl_*/eval_summary.txt` ほかチェックポイント

2. 受理行動の学習 → 受理環境 RL

- IRL(受理/継続) の事前学習

  - 目的: 招待後に「レビュー参加するか」を推定
  - 実装: `src/gerrit_retention/irl/maxent_binary_irl.py`, `src/gerrit_retention/recommendation/reviewer_acceptance_after_invite.py`
  - コマンド例:
    ```bash
    uv run python scripts/run_reviewer_acceptance_after_invite.py \
      --input data/processed/unified/all_reviews.json \
      --output outputs/reviewer_acceptance_after_invite_full/
    ```
  - 生成物: 係数 θ/スケーラ、評価レポート

- 環境と統合（受理環境）

  - 環境: `src/gerrit_retention/rl_environment/review_env.py`
  - IRL ラッパー: `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`
  - 目的: 受理/継続に基づく報酬で RL 学習（継続ボーナス込み）

- 学習（PPO）
  - スクリプト: `training/rl_training/stable_ppo_training.py`
  - 出力: `models/` にチェックポイント、`outputs/` に評価ログ

## よく使うコマンド

```bash
# 割当RL（IRLソフトマックス報酬）
# 目的: 招待ランキングIRL（条件付きロジット）の候補集合softmax確率を報酬にして、
#       K候補から誰を割り当てるべきかの方策を学習します。
# 主な引数:
#   --reward-mode irl_softmax  IRLのsoftmax確率を報酬に使用
#   --continuity-weight        同一レビュアー継続ボーナスの重み（0なら無効）
#   --continuity-tau           継続ボーナスの時間減衰の速さ（大→ゆっくり減衰）
#   --timesteps                学習総ステップ（=学習用変更数×エポック相当）
#   --output                   出力先ディレクトリ（評価サマリ/チェックポイント保存）
uv run python training/rl_training/train_assignment_ppo.py \
  --reward-mode irl_softmax \
  --continuity-weight 0.2 --continuity-tau 3.0 \
  --timesteps 2000 \
  --output outputs/assignment_rl_irl

# 割当RL（受理確率報酬）
# 目的: 招待後に「参加する確率」（ロジスティック由来のsigmoid(u)）を報酬として、
#       参加されやすい担当者を選ぶ方策を学習します。
# 備考: IRL(受理)モデルのθ/スケーラを内部で推定し、環境が報酬化します。
uv run python training/rl_training/train_assignment_ppo.py \
  --reward-mode accept_prob \
  --continuity-weight 0.2 --continuity-tau 3.0 \
  --timesteps 2000 \
  --output outputs/assignment_rl_accept

# 単一受理環境の学習
# 目的: 1人のレビュアー軸で、受理/継続を最大化する方策を学習します。
# 備考: 実データのみの遷移、IRLラッパーで受理IRL報酬に置換/加重が可能、
#       同一レビュアー継続ボーナスあり。
uv run python training/rl_training/stable_ppo_training.py

# API 起動（必要に応じて）
# 目的: 学習済みのモデルや予測/推薦をAPI経由で提供します（FastAPI）。
# 備考: 設定は `configs/*.yaml` を参照。docker-compose でも起動可。
uv run python scripts/run_api_server.py

# 主要スクリプト一覧（実行例あり）
# 目的: 用途別スクリプトの確認。例: データ収集、IRL学習、可視化など。
ls scripts | sed -n '1,20p'
```

## つまずきやすいポイント

- 分割粒度: 割当環境は「変更=1 ステップ」。分割は変更単位（`change_idx`）で。
- 特徴量順序: IRL/受理モデルと環境の特徴順序がズレるとスコアが不安定。警告が出たら要確認。
- データ極小時のガード: 学習/評価が 0 ステップにならないよう `train_assignment_ppo.py` に保護あり。
- 継続ボーナス: 同一レビュアー連続選択時に寄与。短い評価では影響が見えづらい。

## 参考ドキュメント

- `docs/rl_summary.md`: RL/IRL の全体整理
- `docs/visualization_system_summary.md`: 可視化やダッシュボード概要
- `outputs/` 内の各種レポート: 実行結果のサマリ
