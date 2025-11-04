# 長期貢献者予測の実装方針

## 目的とゴール

- **目的**: Gerrit レビュー履歴や IRL で得られる特徴量を活用し、開発者が一定期間以上アクティブに貢献し続けるかを確率で予測する。
- **利用シナリオ**:
  - レビュアアサイン時に「長期的に活動を継続しそうな候補」を優先表示する。
  - コミュニティマネージャが離脱リスクの高い開発者を早期に把握しフォローする。
- **成功指標**:
  - バックテストでの AUC/PR-AUC がベースライン (例: 最近の活動回数のみ) を上回る。
  - 重要開発者群に対するリコールが向上し、ノイズレベルが許容範囲内に収まる。
  - プロダクション導入前にサバイバル曲線などで説明可能な形で可視化できる。

## 現在の優先アクション（2025-10-13 時点）

- 長期貢献者予測を最優先の研究テーマとして進捗管理する。レビューア推薦 IRL から得られる知見は特徴量として活用しつつ、目的関数を「長期継続確率」に再定義する。
- **短期ロードマップ**:
  1.  既存ログを用いた `feature_history_months` = 18 / `target_window_months` = 12 のベースラインデータセット生成。
  2.  LightGBM・ロジスティック回帰による分類モデルを構築し、AUC / PR-AUC を算出。
  3.  サバイバルモデル（CoxPH）を試作し、C-index とサバイバル曲線を比較。
- **意思決定ポイント**:
  - ベースラインとサバイバルモデルの性能差が小さい場合は解釈性の高い手法を優先。
  - 期間パラメータ (`target_window_months`, `feature_history_months`) は本研究を通じて再チューニングし、最適値を決めたら `configs/retention_config.yaml` に固定する。

## 想定するデータソースとスキーマ

| データ種別         | ファイル/テーブル例                                                 | 主な属性                                              | 備考                            |
| ------------------ | ------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------- |
| レビュー履歴       | `outputs/irl/reviewer_sequences.json`、`data/review_requests_*.csv` | change_id, reviewer_id, patch_set, status, timestamps | 既存 IRL 学習と同様の整形を利用 |
| アクティビティログ | `data/retention_*`、`outputs/task_assign/**`                        | review_count_windowed, approvals, comments            | 月次集計を追加する              |
| IRL 派生特徴       | `outputs/irl/window_{Nm}/irl_model.json`、`replay_eval.json`        | reward_score, action_match_rate, policy_entropy       | 開発者単位に再集計する          |
| メタ情報           | プロジェクト/モジュール情報 (`configs/gerrit_config.yaml`)          | module, company (あれば)                              | 多様性・ネットワーク指標に利用  |

### ラベル化定義

- **観測基準日 (`snapshot_date`)**: 月末や四半期末など、特徴量を切り出す時点。各開発者の活動履歴をこの日まででスナップショット化する。
- **予測対象期間 (`target_window_months`)**: `snapshot_date` の直後から何ヶ月先までの継続を判定するか。標準は 12 ヶ月。
- **学習に用いる履歴長 (`feature_history_months`)**: 特徴量生成で遡る期間。標準は 18 ヶ月だが、`target_window_months` に応じて 1〜2 倍程度でチューニング可能。
- **活動閾値 (`min_reviews`)**: 予測期間の各月について「最低限アクティブであった」とみなすレビュー件数。標準は 1 件以上。
- **アクティブ率しきい値 (`active_ratio`)**: 予測期間内で活動閾値を満たした月数 ÷ `target_window_months`。標準では 0.75 (つまり 12 ヶ月中 9 ヶ月以上活動)。
- **継続確認ラベル (`is_active_after_threshold`)**: `snapshot_date` から `activity_after_months` 経過後にも 1 件以上のレビュー活動があれば 1、それ以降まったく登場しなければ 0。長期ラベル (`is_long_term`) が厳しすぎて完全分離する場合の代替指標として利用する。
- **短期観測者の扱い**: `target_window_months` に満たない履歴しかない場合は (a) ラベル生成対象外にする、(b) 分母を実月数に調整した補助ラベルを使う、の 2 パターンを検討。
- **離脱イベント定義 (`grace_period_days`)**: 生存分析では「最後のアクティブ日から 60 日 (仮) 経過」などの猶予を超えた時点を離脱とみなし、イベント時間 `event_time` と打ち切りフラグ `event_observed` を生成する。

これらのパラメータは YAML (`configs/retention_config.yaml` など) で管理し、分析比較時に変更履歴を追えるようにする。

### パラメータ設定方法

- **YAML 設定**: 研究用の `configs/retention_config.yaml` に以下のようなブロックを設け、`target_window_months` と `feature_history_months` を任意に指定できるようにする。

  ```yaml
  retention_prediction:
    target_window_months: 12 # ここを 6 や 18 など任意の値に変更
    feature_history_months: 18 # 過去データの参照期間も自由に設定
    min_reviews: 1
    active_ratio: 0.75
    grace_period_days: 60
      activity_after_months: 12 # is_active_after_threshold 用に確認する「以降」の開始タイミング
  ```

- **コマンドライン引数**: スクリプト実行時に上書きできるよう、`scripts/offline/build_contributor_retention_labels.py` や `scripts/evaluation/retention_backtest.py` に `--target-window-months` / `--feature-history-months` オプションを用意し、手元で即座に変更を試せるようにする。

  ```bash
  uv run python scripts/evaluation/retention_backtest.py \
     --target-window-months 12 \
     --feature-history-months 18 \
     --config configs/retention_config.yaml
  ```

- **バッチ探索**: 複数値を試すときは YAML にリストを記述するか、CLI で複数指定を解釈するロジック（例: `--target-window-months 6 12 18`）を実装してループ処理を行う。

  新しい継続確認ラベルを使う場合は `--activity-after-months` で閾値を上書きし、学習・評価時に `--label-column is_active_after_threshold` を指定する。

#### ウィンドウ設定のバリエーション

- `target_window_months` を {3, 6, 9, 12, 18, 24} など複数パターンで評価し、短期の離脱検知と長期の残存率予測を切り分ける。
- 各 `target_window_months` ごとに特徴量に含める履歴長 (`feature_history_months`) を {target, target×1.5, target×2} の候補で試行する。例: 12 ヶ月先を予測するモデルなら、過去 {12, 18, 24} ヶ月のデータを入力して精度を比較し、長すぎる履歴でノイズが混ざる/短すぎて情報が足りないケースを避ける。
- サバイバル分析では観測期間 (`observation_window_years`) を複数設定し、長期フォロー時の打ち切り率をモニタリングする。
- 期間ごとのモデルを別々に学習する方式と、マルチタスク学習（shared encoder + head）で同時に出力する方式の比較を検討する。

#### ハイパーパラメータ探索の流れ

1. `configs/retention_config.yaml` に期間パラメータの候補リストを記述。
2. `scripts/evaluation/retention_backtest.py` に `--target-window-months` と `--feature-history-months` の複数指定を許可し、バッチ実行で全組み合わせを評価。
3. 指標 (AUC, PR-AUC, C-index, Brier) を期間別に集計し、モデルの最適な期間設定を選定。
4. 選定後は同パラメータで再学習し、解釈可能性の分析 (SHAP 等) を実施。

#### 生存期間分析への影響

- 短い `target_window_months` ではイベント発生率が高まり、サバイバル曲線が急峻になるため、早期離脱予兆の可視化に向く。
- 長い `target_window_months` を採用すると打ち切りデータが増えるため、CoxPH では比例ハザード仮定の検証 (Schoenfeld 残差など) を併用する。
- `grace_period_days` や `active_ratio` を期間に応じて再スケーリングし、例えば 6 ヶ月予測では `active_ratio=0.67` (4/6 ヶ月) を基準にするなど柔軟に調整する。
- Kaplan-Meier 曲線を期間別に重ねて描画し、長期化するとベースライン生存率がどの程度低下するかを比較する。

## 予測タイミングと評価設計

- **特徴量生成のタイミング**: 毎月末に `snapshot_date` を設定し、過去 `feature_history_months` (例: 18 ヶ月) の活動を集約して特徴量テーブルを作成する。IRL 派生値も同じカットオフで再集計する。
- **予測の出力タイミング**: `snapshot_date` ごとにモデルへ特徴量を入力し、各開発者の「`target_window_months` 先まで継続する確率」を推定。推定値は `outputs/retention/scores/snapshot=YYYYMM.parquet` 等として保存。
- **ラベル付与**: 予測後に実績データが揃ったタイミングで、同じ `snapshot_date` に対応する `target_window_months` の活動実績からラベルを生成し、学習・評価に利用する。
- **時系列分割**: 学習/検証/テストを時間順で分離する。例: 2019-2021 年のスナップショットを学習、2022 年をバリデーション、2023 年をテストとする。データ漏洩を避けるため、未来のスナップショット情報やラベルは学習から除外する。
- **評価設計**:
  - バックテストでは各 `snapshot_date` での予測確率と実績ラベルを突き合わせ、AUC・PR-AUC・Precision@K を算出。
  - サバイバルモデルは C-index や Integrated Brier Score をスナップショット単位で集計し、月次推移を可視化。
  - 予測と実績の時差を意識し、評価期間が重複しないよう `target_window_months` ごとに区切る (例: 12 ヶ月予測なら 2023/01 スナップショットの評価は 2023/02-2024/01 の実績のみを使用)。

## システム構成の概要

```mermaid
digraph G {
   rankdir=LR;
   subgraph cluster_data {
      label="Data Processing";
      raw["Raw Review Logs"];
      features["Contributor Feature Builder"];
      raw -> features;
   }
   subgraph cluster_model {
      label="Modeling";
      irl_feats["IRL-derived Features"];
      survival["Survival / Classification Models"];
      features -> survival;
      irl_feats -> survival;
   }
   subgraph cluster_eval {
      label="Evaluation";
      eval["Backtesting & Metrics"];
      survival -> eval;
   }
}
```

## 実装ステップ

### フェーズ 1: データ基盤整備

1. `data_processing/feature_engineering/build_contributor_retention_features.py` を利用し、月次アクティビティと IRL 派生値を統合した特徴量テーブルを生成。

   - 入力: review リクエスト CSV、`outputs/irl/**/replay_eval.json` (任意)。
   - 出力: `outputs/retention/features/contributor_features_{YYYYMMDD}.parquet`。
   - 実行例:

     ```bash
     uv run python data_processing/feature_engineering/build_contributor_retention_features.py \
        --review-requests data/review_requests_openstack_multi_5y_detail.csv \
        --snapshot-date 2023-07-01 \
        --feature-history-months 18 \
        --target-window-months 12 \
        --irl-replay outputs/analysis/window_6m/replay_eval.json \
        --output-dir outputs/retention/features \
        --config configs/retention_config.yaml \
        --overwrite
     ```

   - 実行後、標準出力に JSON 形式のサマリ（件数、期間、出力先）が表示される。

2. ラベル生成スクリプト `scripts/offline/build_contributor_retention_labels.py` を使い、スナップショット以降の活動から長期貢献ラベル / サバイバル用メタ情報を生成。

   - 出力: `outputs/retention/labels/contributor_labels_{YYYYMMDD}.parquet`。
   - 実行例:

     ```bash
     uv run python scripts/offline/build_contributor_retention_labels.py \
        --review-requests data/review_requests_openstack_multi_5y_detail.csv \
        --snapshot-date 2023-07-01 \
        --target-window-months 12 \
        --developer-list outputs/retention/features/contributor_features_20230701.parquet \
        --output-dir outputs/retention/labels \
        --config configs/retention_config.yaml \
        --overwrite
     ```

   - `event_time_days` と `event_observed` をサバイバル分析に、`is_long_term` を分類モデルに利用する。
     - `is_active_after_threshold` と `activity_after_months` を同梱し、単純な「n ヶ月後以降に再登場するか」ラベルとして利用できる。

3. データ検証: 欠損値、レビュー期間の偏り、ラベル比率をレポート (`analysis/retention/eda.ipynb`) にまとめる。

### フェーズ 2: モデル開発

1. ベースラインモデル

   - `analysis/retention/modeling_baseline.ipynb` に加えて、CLI ベースの `scripts/training/retention/train_retention_baseline.py` でロジスティック回帰モデルを学習し、モデル/指標を保存。
   - 代表的な実行例:

     ```bash
     uv run python scripts/training/retention/train_retention_baseline.py \
        --features outputs/retention/features/contributor_features_20230701.parquet \
        --labels outputs/retention/labels/contributor_labels_20230701.parquet \
        --output-dir outputs/retention/models/baseline_20230701 \
        --test-size 0.2 \
        --random-state 42 \
        --overwrite
     ```

   - 学習済みモデルのオフライン評価は `scripts/evaluation/evaluate_retention_model.py` を利用して実施。

     ```bash
     uv run python scripts/evaluation/evaluate_retention_model.py \
        --model outputs/retention/models/baseline_20230701/baseline_model.joblib \
        --features outputs/retention/features/contributor_features_20240101.parquet \
        --labels outputs/retention/labels/contributor_labels_20240101.parquet \
        --output outputs/retention/eval/baseline_eval_20240101.json
     ```

   - 学習と検証をワンコマンドで分離実行する場合は `scripts/training/retention/run_retention_baseline_pipeline.py` を利用。`--train-snapshot-date` と `--eval-snapshot-date` を分けて指定すると、各スナップショットで特徴量/ラベル生成 → 学習 → 評価を自動で実施できる。

     ```bash
     uv run python scripts/training/retention/run_retention_baseline_pipeline.py \
        --train-snapshot-date 2022-07-01 \
        --eval-snapshot-date 2023-07-01 \
        --review-requests data/review_requests_openstack_multi_5y_detail.csv \
        --feature-history-months 18 \
        --target-window-months 12 \
        --activity-after-months 12 \
        --label-column is_active_after_threshold \
        --output-dir outputs/retention/models \
        --overwrite
     ```

     既存の `--snapshot-date` は学習側のショートカットとして引き続き利用でき、検証側を明示しない場合は学習と同じスナップショットが評価に用いられる。

   - 評価指標: AUC, PR-AUC, F1、サバイバル視点の C-index。

2. サバイバルモデル
   - `src/retention/models/survival.py` を新設し、`scikit-survival` または `lifelines` を利用した Cox PH / Weibull AFT モデルを実装。
   - IRL 報酬やポリシーエントロピーを時間依存共変量として扱えるよう投入。
3. 時系列モデル (任意)
   - LSTM/Transformer を `torch` ベースで試す (`src/retention/models/sequence.py`)。
   - temporal split (train/val/test) を徹底。
4. モデル選定
   - モデル毎に SHAP/Permutation importance を算出し、主要特徴を可視化。
   - 最終的に精度・解釈性・計算コストを比較し推奨モデルを決定。

### フェーズ 3: バックテストと評価

1. `scripts/evaluation/retention_backtest.py` を追加し、時系列の検証用フォールド (例: 2019-2021 で学習、2022-2023 で検証) を自動生成。
2. モデル出力を元に以下を算出:
   - 月次 AUC 推移、PR-AUC、top-K precision。
   - サバイバル曲線 (Kaplan-Meier) と予測 C-index。
   - 開発者セグメント別 (プロジェクト/会社) の離脱拾い上げ率。
3. 結果を `docs/retention_evaluation_report.md` に記録。差分が大きい場合は `docs/closure/retention_case_studies.md` を作成し事例を整理。

## 主要コンポーネントの詳細

### 特徴量候補

- 活動量: 月次レビュー数、コメント数、PatchSet 更新数、バグ修正数。
- 品質: 承認率、レビューラグ、リオープン率。
- ネットワーク: 被レビュー関係の多様性、Graph centrality。
- IRL 派生: 推定報酬、ポリシーの Shannon エントロピー、方策と実観測の KL divergence。
- 履歴: 過去 `n` ヶ月の活動リズム (季節性対策としてローリング統計を付与)。

### ラベル生成ロジック例 (擬似コード)

```python
def build_labels(activity_df, target_window_months=12, min_reviews=1, active_ratio=0.75):
    labels = []
    for developer, group in activity_df.groupby("developer_id"):
        window_activities = group.set_index("month").sort_index()
        total_months = window_activities.shape[0]
        active_months = (window_activities["reviews"] >= min_reviews).sum()
        is_long_term = (total_months >= target_window_months) and (
            active_months / target_window_months >= active_ratio
        )
        labels.append({
            "developer_id": developer,
            "is_long_term": int(is_long_term),
            "active_rate": active_months / target_window_months,
            "last_active": window_activities.index.max(),
        })
    return pd.DataFrame(labels)
```

サバイバル分析用には最後の活動月から離脱イベントまでの期間 (`event_time`) と、観測打ち切りフラグ (`event_observed`) を生成する。

## 評価指標とモニタリング

- **分類指標**: AUC, PR-AUC, Precision@K, Recall@K, Brier Score。
- **サバイバル指標**: Concordance Index, Integrated Brier Score。
- **ビジネス指標**: 上位 K% に含まれる実残存率、離脱予兆の早期検知率。
- **キャリブレーション**: Reliability diagram、温度スケーリングの適用を検討。

## ロードマップ (目安)

| フェーズ                  | 期間     | 主な成果物                                         |
| ------------------------- | -------- | -------------------------------------------------- |
| P0: 要件定義 & PoC        | 1-2 週間 | データ定義、EDA、ベースラインモデル (分類)         |
| P1: サバイバルモデル拡張  | 2-3 週間 | Cox/DeepSurv 実装、バックテストレポート            |
| P2: 高度分析 & モデル選定 | 2 週間   | 特徴量重要度分析、キャリブレーション、レポート整備 |

## 想定リスクと緩和策

- **データスパース**: 新規参加者は履歴が短くラベルが付けにくい → 低活動者専用モデル or ルールベース補完。
- **IRL モデル更新との整合**: 報酬スケールが変わると特徴量がドリフト → 正規化ルールとメタ情報 (モデルバージョン) を記録。
- **サバイバルモデルの学習コスト**: サンプル数が大きい場合に計算時間が増大 → 事前特徴選択とバッチ学習で対応。
- **解釈性の不足**: 重要特徴や影響を可視化するダッシュボードを整備。

## 次アクション

1. `outputs/` 内の既存データを用いた EDA 用ノートブック雛形を作成。
2. 特徴量生成スクリプトのドラフトを `data_processing/` 配下に追加。
3. 12 ヶ月ウィンドウを対象にベースライン分類モデルを実装し精度を確認。
4. 結果を踏まえサバイバルモデルへの拡張を検討。


evalから何ヶ月分学習すれば精度が良くなるかの変化と，trainの基準がevalからどのくらい離れても精度が担保できるかの２パターン検証したい