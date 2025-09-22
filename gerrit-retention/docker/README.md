# docker ディレクトリ概要

- 目的: コンテナ化と周辺サービス設定。
- 主な内容:
  - `Dockerfile`: アプリのベースイメージ定義。
  - `docker-compose*.yml`: 開発/本番用の compose 設定。
  - `init-db.sql`: DB 初期化スクリプト。
  - `nginx.conf` / `redis.conf` / `prometheus.yml`: 周辺サービス設定。
  - `healthcheck.sh`: ヘルスチェック。
