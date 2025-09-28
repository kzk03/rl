# api ディレクトリ概要

- 目的: FastAPI によるサーバー実装。
- 主な内容:
  - `main.py`: アプリエントリ。ルータのマウントや起動設定。
  - `core/`: 設定/例外/ロギングのコア機能。
  - `routers/`: エンドポイント定義。
  - `services/`: ビジネスロジック層。
  - `models/`: Pydantic モデル/スキーマ。
  - `utils/`: 補助ユーティリティ。

## 設定ファイルの配置ポリシー（重要）

- API 専用の設定ファイルは `configs/api/` に配置します（唯一の探索先）。
  - `configs/api/api_config.yaml`: サーバー/ CORS / 既定値など。
  - `configs/api/model_config.yaml`: API が参照するモデル一覧/パス等。
  - `configs/api/logging_config.yaml`: API 用の dictConfig ログ設定。

起動時の設定解決順:

1. `configs/api/` のみを探索
2. ログ設定のみ、互換目的で `configs/logging_config.yaml` を参照することがあります

環境変数でも上書きが可能です（例: `API_SERVER_PORT=8080`）。
