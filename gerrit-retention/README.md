# Gerrit Retention IRL プロジェクト

OpenStack Gerrit のレビュー履歴を用いたタスク割り当て・維持率改善のための強化学習 (RL / IRL) 研究リポジトリです。過去レビュー記録から候補レビュアーを推薦し、その品質をオフラインで評価するワークフローと、推定モデルを活用した可視化・分析ツール群を提供します。

## 主な機能

- **データ抽出・前処理**: Gerrit から取得した JSON/CSV をクレンジングし、特徴量付きのタスクデータセットへ整形。
- **IRL モデル学習**: `scripts/training/offline_rl` 配下のスクリプトで専門家軌跡から報酬関数を推定。
- **タスク割り当てリプレイ評価**: `scripts/evaluation/task_assignment_replay_eval.py` によりウィンドウ別 métrics を算出し、評価結果を JSON/CSV に出力。
- **分析・可視化**: `analysis/` 以下で多角的なレポート生成や可視化を実施。
- **API プロトタイプ**: `api/` ディレクトリで FastAPI ベースのエンドポイントを提供。

本プロジェクトのレビュアー推薦は、単なる「担当者割り当て」ではなく、**レビュアーが実際にタスクへ取り組んでくれる確率**を推定し、その確率の変動を逆強化学習・強化学習の両面で活用することが狙いです。逆強化学習で学習した報酬関数は「レビュアーが何を重視して意思決定しているか」を示し、タスク × レビュアー組み合わせごとに取り組み意欲の推定値を与えます。これらをスライディングウィンドウで評価し、個人ごとの確率遷移を追跡しながら、将来的には強化学習により「長期的に貢献してくれるレビュアーを増やすタスク依頼シナリオ」のシミュレーションまで発展させることを目標としています。

### 今後の分析・発展方向

- 逆強化学習で得た報酬関数をスライディングウィンドウで評価し、時間変化やドリフトを定量化。
- レビュアー個人の「タスクに取り組む確率」の遷移を追跡し、離脱兆候の検知指標を構築。
- 長期的な貢献者を増やすタスク依頼ポリシーを強化学習で探索・シミュレート。

## リポジトリ構成

| ディレクトリ       | 内容                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| `analysis/`        | 評価結果の統計処理・可視化スクリプト。                                |
| `api/`             | 推薦 API の FastAPI 実装と関連サービス。                              |
| `configs/`         | 各種実験・サービス用設定ファイル。                                    |
| `data/`            | 元データおよび加工済みデータ (Git 管理推奨外の大容量ファイルを格納)。 |
| `data_processing/` | データ抽出・前処理・特徴量生成ユーティリティ。                        |
| `docker/`          | 本番 / 開発用 Docker 設定。                                           |
| `docs/`            | 実験記録、設計ノート、ステータスレポート。                            |
| `examples/`        | ワークフロー解説用サンプルスクリプト。                                |
| `outputs/`         | 学習・評価の成果物 (JSON/CSV/図表)。                                  |
| `scripts/`         | 学習・評価・ユーティリティ実行スクリプト。                            |

各最大成果物の例として `outputs/task_assign_multilabel/2023_cutoff_full/README.md` に 2023 年カットオフ実験の詳細サマリーを掲載しています。

## セットアップ

1. **依存整備**: 本プロジェクトは [uv](https://github.com/astral-sh/uv) を前提にしています。
   ```bash
   uv sync
   ```
2. **環境変数・設定**: `configs/` 以下の `development.yaml` などを参照して必要に応じて調整してください。
3. **データ配置**: `data/` 配下に Gerrit から抽出した JSON/CSV を配置します (大容量ファイルは Git LFS などで管理推奨)。

## 代表的なワークフロー

### 1. レビュー依頼データのビルド

```bash
uv run python examples/build_eval_from_review_requests.py \
  --input data/raw/review_requests_openstack.json \
  --output data/review_requests_openstack_w14.csv
```

### 2. タスクラベルおよび特徴量生成

```bash
uv run python data_processing/extract_action_match.py \
  --input data/review_requests_openstack_w14.csv \
  --output data/processed/action_match.jsonl
```

### 3. IRL モデル学習

```bash
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --config configs/rl_config.yaml \
  --output-dir outputs/irl_model_latest
```

### 4. リプレイ評価 (タスク割り当て)

```bash
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign_multilabel/2023_cutoff_full/tasks_eval.jsonl \
  --cutoff 2023-07-01T00:00:00 \
  --eval-mode irl \
  --irl-model outputs/task_assign_multilabel/2023_cutoff_full/irl_model.json \
  --windows "0m-3m,3m-6m,6m-9m,9m-12m" \
  --csv-dir outputs/task_assign_multilabel/2023_cutoff_full/eval_detail \
  --out outputs/task_assign_multilabel/2023_cutoff_full/replay_eval_eval.json
```

### 5. 可視化

`analysis/visualization` や `outputs/task_assign_multilabel/2023_cutoff_full/figs/window_match_rates.png` を参照してください。カスタム可視化を追加する場合は `analysis/` 内のテンプレートを活用できます。

## 便利ツール

- レビュアーごとの行抽出: `data_processing/extract_reviwer.py`。
- 評価結果サマリー: `outputs/task_assign_multilabel/2023_cutoff_full/replay_eval_summary.md`。
- API サーバー起動: `uv run python api/main.py`。

## ドキュメント

- 長期的な IRL/RL の歩み: `docs/irl_rl_status_20251001.md`
- 実験ログ: `docs/experiments/`
- 将来方針メモ: `docs/irl_future_directions.md`

## ライセンス

本リポジトリのライセンスやデータ利用ポリシーはプロジェクト内の契約・合意に従います。公開用途で利用する場合は別途確認してください。
