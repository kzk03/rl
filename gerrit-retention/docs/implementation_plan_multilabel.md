# マルチラベル版タスク生成の実装方針（再現用）

## 前提

- リポジトリ直下で `uv` コマンドが利用可能であること。
- 入力となるレビュアシーケンス: `outputs/irl/reviewer_sequences.json`
- 既存の単一ラベル版ディレクトリ `outputs/task_assign/window_*` は上書き対象。
- マルチラベル用スクリプト: `scripts/offline/build_task_assignment_multilabel.py`
- 生成先ディレクトリは学習期間（月）ごとに `outputs/task_assign/window_{Nm}` を利用する。

## 方針概要

1. 学習期間ごとの出力先ディレクトリを事前に確認し、既存ファイルがある場合はバックアップまたは上書きを許容する。
2. 各期間 (1, 3, 6, 9, 12 ヶ月) について以下の共通パラメータでスクリプトを実行する。
   - `--input outputs/irl/reviewer_sequences.json`
   - `--cutoff 2023-07-01T00:00:00Z`
   - `--candidate-window-days 30`
   - `--candidate-activity-window-months 6`
   - `--seed 42`
   - `--train-window-months <対象期間>`
   - `--outdir outputs/task_assign/window_<対象期間>`
3. コマンド実行後、各ディレクトリに `tasks_train.jsonl` / `tasks_eval.jsonl` / `tasks_meta.json` が生成されることを確認する。
4. `tasks_meta.json` の `label_type` や `train_count` などのメタ情報を確認し、マルチラベル設定になっていることを検証する。
5. 必要に応じて差分確認 (`git status`) を実施し、再生成結果をレビューの上コミットする。

## 実行コマンド例

以下は 12 ヶ月ウィンドウの例。

```bash
uv run python scripts/offline/build_task_assignment_multilabel.py \
  --input outputs/irl/reviewer_sequences.json \
  --cutoff 2023-07-01T00:00:00Z \
  --outdir outputs/task_assign/window_12m \
  --train-window-months 12 \
  --candidate-window-days 30 \
  --candidate-activity-window-months 6 \
  --seed 42
```

他の月数 (1, 3, 6, 9) についても `--train-window-months` と `--outdir` を変更して同様に実行する。

## 生成後の検証

- `jq '.[0].positive_reviewer_ids' outputs/task_assign/window_12m/tasks_eval.jsonl` などで複数正例が保持されているかをサンプリング確認。
- 評価パイプラインで CSV 出力を生成し、`positive_reviewers` 列が複数リスト化されているか確認。

## 再生成状況サマリ（2025-10-12 時点）

- 全ウィンドウ（1m/3m/6m/9m/12m）の `tasks_meta.json` で `label_type: "multi"` を確認済み。
- 共通パラメータ:
  - `candidate_window_days = 30`
  - `candidate_activity_window_months = 6`
  - `seed = 42`
- 件数とサンプル:

| window     | train_count | eval_count | sample_change_id          | positive_reviewers（一部）                                                     |
| ---------- | ----------- | ---------- | ------------------------- | ------------------------------------------------------------------------------ |
| window_1m  | 205         | 6,591      | openstack%2Fnova~444517   | gibizer@gmail.com, john.garbutt@stackhpc.com, kvmpower@linux.vnet.ibm.com, ... |
| window_3m  | 668         | 6,591      | openstack%2Fnova~444517   | gibizer@gmail.com, john.garbutt@stackhpc.com, kvmpower@linux.vnet.ibm.com, ... |
| window_6m  | 1,494       | 6,591      | openstack%2Fcinder~887443 | caiql@toyou.com.cn, cinder+dellpvme@tristero.net, eharney@redhat.com, ...      |
| window_9m  | 2,162       | 6,591      | openstack%2Fnova~444517   | gibizer@gmail.com, john.garbutt@stackhpc.com, kvmpower@linux.vnet.ibm.com, ... |
| window_12m | 2,919       | 6,591      | openstack%2Fnova~444517   | gibizer@gmail.com, john.garbutt@stackhpc.com, kvmpower@linux.vnet.ibm.com, ... |

- `tasks_eval.jsonl` からのサンプル確認:
  - `openstack%2Fcinder~887443` で正例 11 名（例: caiql@toyou.com.cn, cinder+dellpvme@tristero.net, eharney@redhat.com など）。
  - `openstack%2Fglance~887476` で正例 2 名。
  - `openstack%2Fneutron~887477` で正例 2 名。
- いずれの窓でも複数レビュアが保持されており、単一ラベル問題は再現していないことを確認。

## メモ

- 既存の IRL モデル学習・評価スクリプトは、生成済みタスクファイルを参照して再実行する必要がある。
- 大規模ファイルの検証には `jq` や `head` を組み合わせ、OS の制約に注意する。

## 今後の実行計画（モデル学習〜評価〜分析）

### 1. モデル学習フェーズ

- **ロジスティック IRL (ベースライン)**

  - 各ウィンドウの `tasks_train.jsonl` を入力に `scripts/training/irl/train_task_candidate_irl.py` を実行。
  - 推奨コマンド雛形:

    ```bash
    uv run python scripts/training/irl/train_task_candidate_irl.py \
      --train-tasks outputs/task_assign/window_6m/tasks_train.jsonl \
      --out outputs/irl/window_6m/irl_model.json \
      --iters 400 --lr 0.05 --reg 1e-4 --l1 1e-3
    ```

    ```
    uv run python scripts/training/irl/train_task_candidate_irl.py --train-tasks outputs/task_assign/window_6m/tasks_train.jsonl --out outputs/irl/window_6m/irl_model.json --iters 400 --lr 0.05 --reg 1e-4 --l1 1e-3`
    ```

  - 1m/3m/6m/9m/12m で `--train-tasks` と `--out` のパスを変更しつつ実行。`tasks_used` が極端に少ない窓は L1 を弱めたり `--iters` を増やしてフィットを安定化。

- **(任意) オフライン RL/深層 IRL**
  - `scripts/offline/build_offline_from_reviewer_sequences.py` で `dataset_{train,eval}.jsonl` を生成。
  - `scripts/training/offline_rl/train_retention_irl_from_offline.py` で `.pth` を作成し、`outputs/irl/window_{Nm}/retention_irl_offline.pth` へ保存。
  - ハイパラは `--negatives-per-pos 3`, `--epochs 10` など少しサーチし、学習ログを記録。
- **成果物の整理**
  - 生成ファイルを `outputs/irl/window_{Nm}/` にまとめ、`irl_model.json` と (必要なら) `.pth` を揃える。
  - `docs/implementation_plan_multilabel.md` に実行日・ハイパラ・出力先を追記し再現性を担保。

### 2. リプレイ評価フェーズ

- `scripts/evaluation/task_assignment_replay_eval.py` を用いてメトリクスと推薦 CSV を取得。
- IRL 直ランキング評価例:

  ```bash
  uv run python scripts/evaluation/task_assignment_replay_eval.py \
    --tasks outputs/task_assign/window_6m/tasks_eval.jsonl \
    --cutoff 2023-07-01T00:00:00 \
    --eval-mode irl \
    --irl-model outputs/irl/window_6m/irl_model.json \
    --windows 1m,3m,6m,9m,12m \
    --csv-dir outputs/analysis/window_6m/replay_csv \
    --out outputs/analysis/window_6m/replay_eval.json
  ```

  uv run python scripts/evaluation/task_assignment_replay_eval.py --tasks outputs/task_assign/window_6m/tasks_eval.jsonl --cutoff 2023-07-01T00:00:00 --eval-mode irl --irl-model outputs/irl/window_6m/irl_model.json --windows 1m,3m,6m,9m,12m --csv-dir outputs/analysis/window_6m/replay_csv --out outputs/analysis/window_6m/replay_eval.json

- ベースライン比較として `--irl-model` を外した `reward-mode match_gt`、あるいは RL ポリシーによる `--eval-mode policy --policy <checkpoint>` も実行。
- 温度チューニングが必要なら `scripts/analysis/temperature_tune_irl.py` を活用し ECE を最小化。

### 3. 詳細分析フェーズ

- `outputs/analysis/window_*/replay_csv/*.csv` を対象に以下を実施:
  - `data_processing/extract_action_match.py` でヒット率集計。
  - `analysis/reviewer_diversity_analysis.py` やカスタムノートブックで `precision@k`, `recall@k`, `positive_coverage` を可視化。
  - 失敗事例抽出: `positive_reviewers` を含みつつ top-k に入らなかった候補を調査し、特徴量パターンを確認。
- 分析結果は `docs/irl_rl_status_YYYYMMDD.md` などに記録し、成功・失敗パターンと改善仮説を整理。

### 4. 進捗管理

- 各ステップ完了後に本ドキュメントへ日時・実行コマンド・要点を追記。
- Git 差分をこまめに確認し、モデルファイルは必要に応じて LFS や外部ストレージに配置する。

## 実行ログ

### 2025-10-12: window_6m ロジスティック IRL 学習 + リプレイ評価

- 実行コマンド

  ```bash
  uv run python scripts/training/irl/train_task_candidate_irl.py \
    --train-tasks outputs/task_assign/window_6m/tasks_train.jsonl \
    --out outputs/irl/window_6m/irl_model.json \
    --iters 400 --lr 0.05 --reg 1e-4 --l1 1e-3

  uv run python scripts/evaluation/task_assignment_replay_eval.py \
    --tasks outputs/task_assign/window_6m/tasks_eval.jsonl \
    --cutoff 2023-07-01T00:00:00 \
    --eval-mode irl \
    --irl-model outputs/irl/window_6m/irl_model.json \
    --windows 1m,3m,6m,9m,12m \
    --csv-dir outputs/analysis/window_6m/replay_csv \
    --out outputs/analysis/window_6m/replay_eval.json
  ```

- 生成物

  - `outputs/irl/window_6m/irl_model.json` （theta 次元=13, tasks_used=1494）
  - `outputs/analysis/window_6m/replay_eval.json`
  - `outputs/analysis/window_6m/replay_csv/{1m,3m,6m,9m,12m}.csv`

- 主なリプレイ指標（IRL ダイレクト）

| window | steps | action_match_rate | top3_hit_rate | mAP    | ECE    | positive_coverage |
| ------ | ----- | ----------------- | ------------- | ------ | ------ | ----------------- |
| 1m     | 207   | 0.6570            | 0.8551        | 0.6579 | 0.3490 | 0.9227            |
| 3m     | 672   | 0.6622            | 0.8810        | 0.6540 | 0.4096 | 0.9286            |
| 6m     | 1,291 | 0.6816            | 0.8908        | 0.6633 | 0.4402 | 0.9411            |
| 9m     | 2,032 | 0.6905            | 0.8981        | 0.6784 | 0.4468 | 0.9542            |
| 12m    | 2,510 | 0.6936            | 0.9004        | 0.6804 | 0.4514 | 0.9530            |

- メモ
  - 候補数が平均 100 件超のため `ECE` が 0.35〜0.45 と高め。温度調整で改善余地あり。
  - `positive_coverage` は全ウィンドウで 92%以上を確保。
