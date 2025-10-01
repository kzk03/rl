## 目的と範囲

本ドキュメントは、本リポジトリで動作する自動化エージェント（抽出・特徴量・評価・IRL/RL・可視化）の行動規範、役割、入出力、品質ゲート、運用ワークフローをまとめた実装/運用ガイドです。対象は以下です。

- Gerrit の履歴データ抽出と、レビュア応答性の予測（介入なし前提）
- 逆強化学習（IRL）による報酬関数の推定と、強化学習（RL）によるポリシー学習/シミュレーション
- 評価におけるリーク防止・公平性の担保、および長期データでの一般化検証

## 基本原則（エージェントが従うルール）

- 日本語で応答すること
- 必要に応じてユーザへ確認質問を行い、要求や前提を明確化すること
- 作業後に「やったこと」「結果」「次に取れる行動」を簡潔に提示すること
- 可能な限り自動で実行し、ブロックされた場合のみユーザに判断を仰ぐこと
- セキュリティ/プライバシー配慮（メール等 PII の扱い、外部送信の抑制）
- コメントを詳細に記載すること（できれば行単位）

## システム全体像（高レベル）

1. 変更履歴抽出 → 2) per-review-request データセット構築 → 3) 予測器の学習/評価 → 4) IRL で報酬推定 → 5) RL でポリシー学習/シミュレーション → 6) 可視化/レポーティング

- リーク防止: 割当時刻より未来の情報は使用しない
- 公平性: 時系列分割 + メール単位のグループ分割を優先し、gap 系特徴は評価時に既定で除外可

## エージェント一覧と役割（Inputs/Outputs/失敗時）

### 1. 変更抽出エージェント（Gerrit Extraction）

- 入力: プロジェクト群、期間、詳細フラグ（messages / reviewers / reviewer_updates 必須）
- 出力: 詳細付き変更 JSON（複数プロジェクト統合可）
- 実装: `data_processing/gerrit_extraction/extract_changes.py`
- 重要フィールド: `messages`, `reviewers`, `reviewer_updates`, `attention_set`, `revisions.*.files`
- 失敗時: 接続エラーはリトライ/スキップ、完了時はメタ情報（抽出時刻/件数）を JSON へ付与

### 2. データセットビルダー（Per-Review-Request）

- 入力: 上記 JSON
- 出力: 行=「(change_id, reviewer_email, request_time)」、ラベル=「W 日以内に応答(メッセージ/投票)」の CSV
- 実装: `examples/build_eval_from_review_requests.py`
- 主な特徴量:
  - 活動量: reviewer/owner 過去 30/90/180 日メッセージ数
  - 負荷: reviewer 過去 7/30/180 日の割当件数
  - ペア関係: owner→reviewer 過去割当数、±D 日相互近接メッセージ数（全体/プロジェクト内）
  - パス親和性: 直近 180 日の担当パス集合との重なり/類似（overlap, Jaccard, Dice, Overlap Coef, Cosine）を files/dir1/dir2 × global/project で計算
  - 代理応答率: 過去 180 日のメッセージ/割当
  - テンュア: 初回発言からの経過日数（owner/reviewer）
  - 変更複雑度: insertions/deletions/files_count、WIP、subject 長
- リーク対策: request_time より未来の情報を使用しない（時刻フィルタ、bisect による前方切り）

### 3. 予測・評価エージェント

- 入力: 上記 CSV（`developer_email`, `context_date`, `label` を含む）
- 出力: AUC/AP/Brier/ECE 等の評価指標
- 実装: `scripts/evaluation/real_data_evaluate.py`, 予測器: `src/gerrit_retention/prediction/retention_predictor.py`
- 分割:
  - 既定: 時系列分割（`context_date`）、テスト比率指定または分位自動選定
  - グループ分割: `--group-by-email` で同一メールの交差を防止
  - 例外ガード: 単一クラスや極端な不均衡での分割失敗時は安全なフォールバック
- 特徴量制御: `--exclude-gap-features` でギャップ類を無効化（リーク/過適合の抑制）

### 4. IRL エージェント（逆強化学習）

- 目的: 過去の開発/レビュー行動から報酬関数を推定（MaxEnt 系を想定）
- 入力: 行動系列（例: 招待 → 応答/不応答、割当 → 応答時刻）と特徴（確率/ロジット/エントロピーなど）
- 出力: 報酬重み/関数、期間別の汎化/再現性指標
- 期間分割: 作成期間を時系列/幅で複数に分け、どの期間/直近データが再現に効くかを比較

### 5. RL 訓練/シミュレーションエージェント

- 目的: IRL で得た報酬に基づきポリシーを学習し、シミュレーションで活動履歴の再現性を検証
- 入力: 報酬関数、MDP/POMDP 環境（例: マルチレビュア割当環境）
- 出力: 学習済みポリシー、シミュレーションログ、再現率/軌跡類似度
- 検証: 実履歴との距離（参加率、待ち時間分布、アクション配分など）

### 6. 可視化/レポートエージェント

- 評価曲線（ROC/PR）、較正（信頼度-実現率）、重要特徴、期間/プロジェクト別比較を生成
- 成果物: `outputs/` や `docs/` 下へ図表/サマリを保存

## 公平性・リーク防止の原則

- 時系列分割を既定とし、未来情報の使用を禁止
- 同一メール（開発者/レビュア）が学習/評価を跨がないようグループ分割
- ギャップ特徴（例: days_since_last_activity）は評価では既定 OFF（比較用に ON も可）
- 特徴量生成時は request_time より前の情報のみを参照（messages/assignments/path 履歴の時間フィルタ）

## 主な特徴量辞書（抜粋）

- reviewer_past_reviews_30d/90d/180d: レビュアの過去メッセージ数
- owner_past_messages_30d/90d/180d: オーナーの過去メッセージ数
- owner_reviewer_past_interactions_180d: ±D 日近接の相互メッセージ（180 日）
- owner_reviewer_project_interactions_180d: プロジェクト内での近接相互
- reviewer_assignment_load_7d/30d/180d: 過去割当負荷
- owner_reviewer_past_assignments_180d: 過去のペア割当回数
- reviewer_past_response_rate_180d: 過去 180 日のメッセージ/割当（0〜1）
- reviewer_tenure_days / owner_tenure_days: 初回発言からの経過日
- change_insertions/deletions/files_count, work_in_progress, subject_len
- path\_\*: overlap/jaccard/dice/overlap_coeff/cosine（files/dir1/dir2 × global/project）

## 品質ゲート（Quality Gates）

- Build/Lint/型: Python は基本スクリプト駆動。スクリプト実行時に例外が出ないこと
- 単体/スモーク: 小規模データでビルド → 評価まで通ること
- 評価指標: ROC-AUC, AP, Brier, ECE を出力し、カバレッジ/単一クラス発生時はガードが作用すること
- Green-before-done: 評価スクリプトが通る状態でコミットを締める

## 推奨運用フロー（チェックリスト）

1. 変更抽出（詳細付き）
2. per-review-request データセット構築（W/response-type を明示）
3. 公平設定で評価（time split + group-by-email + gap OFF）
4. 比較用に gap ON 評価（解釈注意）
5. 期間/プロジェクト別、特徴アブレーション、窓 W やラベル定義のスイープ
6. IRL/RL: 期間別の報酬学習 → シミュレーション再現性評価

## コンフィグ/コード参照

- 抽出: `data_processing/gerrit_extraction/extract_changes.py`
- ビルド: `examples/build_eval_from_review_requests.py`
- 評価: `scripts/evaluation/real_data_evaluate.py`
- 予測器/特徴抽出: `src/gerrit_retention/prediction/retention_predictor.py`
- 設定ファイル: `configs/*.yaml`（実行環境、ロギング、分析設定 等）

## トラブルシュート（よくある失敗）

- 抽出時の接続エラー: 自動リトライ後も失敗する ID はスキップし、全体は継続
- reviewer_updates 欠落: `reviewers`/`attention_set`/メッセージ文面からフォールバック復元（精度低）
- 単一クラス分割: 時系列閾値の再選定、またはランダム分割フォールバック
- 極端な不均衡: しきい値調整、PR-AUC 基準での比較、負例サンプリングの工夫

## クイック実行例（任意）

```bash
# 1) per-request データセット生成（14日、vote-or-message）
uv run python examples/build_eval_from_review_requests.py \
	--input data/raw/gerrit_changes/openstack_multi_5y_detail_*.json \
	--output data/review_requests.csv \
	--response-window-days 14 \
	--response-type vote-or-message

# 2) 公平評価（gap OFF）
uv run python scripts/evaluation/real_data_evaluate.py \
	--input data/review_requests.csv \
	--format csv \
	--split-mode time \
	--group-by-email \
	--exclude-gap-features

# 3) 比較評価（gap ON）
uv run python scripts/evaluation/real_data_evaluate.py \
	--input data/review_requests.csv \
	--format csv \
	--split-mode time \
	--group-by-email
```

## 強化学習/逆強化学習の到達目標（刷新）

- OSS の過去の開発履歴から逆強化学習を用いて報酬関数を作成する
- 逆強化学習で算出した報酬に基づき強化学習を実行する
- 強化学習のシミュレーションが実際の活動履歴をどの程度再現できるか検証する
- 報酬関数の作成期間を時系列や期間で分割し、直近/長期のどの範囲が再現に効くかを検証する

## 用語集（抜粋）

- 応答: 依頼後 W 日以内のメッセージまたは投票
- ギャップ特徴: 最終活動からの経過日数など、ラベルと強相関な時系列特徴
- パス親和性: 過去担当パスと対象変更のパス集合の重なり/類似度
- ECE: Expected Calibration Error（確率較正のズレ）

---

このドキュメントは、エージェントの開発・運用を加速する「実務ガイド」です。改善提案（新しい特徴量、評価軸、ワークフロー）は随時歓迎します。
