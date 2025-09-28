# configs ディレクトリ概要

- 目的: 実行環境/パイプラインごとの包括的設定。
- 主なファイル:
  - `development.yaml` / `production.yaml` / `testing.yaml`: 環境別設定。
  - `gerrit_config.yaml`: Gerrit 接続や取得範囲の設定。
  - `retention_config.yaml` / `retention_analysis_config.yaml`: 継続率関連の設定。
  - `rl_config.yaml`: 強化学習の設定。
  - `logging_config.yaml`: ログ設定（環境向け、アプリ全体のロギングポリシー）。
  - `visualization_config.yaml`: 可視化設定。
  - `deployment_config.yaml`: デプロイ関連。
  - `comprehensive_retention_config.yaml`: 包括的な実行設定サンプル。

## API 用設定の配置ポリシー

- API 専用の設定は `configs/api/` に統一しました（唯一の探索先）。
  - `configs/api/api_config.yaml`: API の基本設定（ポート、CORS、機能フラグ）。
  - `configs/api/model_config.yaml`: API が読み込むモデル定義。
  - `configs/api/logging_config.yaml`: API の dictConfig 形式のロギング設定。

`config/` は廃止済みポリシーです。今後は `configs/api/` のみをご利用ください。
