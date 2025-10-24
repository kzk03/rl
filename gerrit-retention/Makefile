.PHONY: setup build up down restart clean

# 環境変数の取得
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
export USERNAME := $(shell whoami)

setup:
	@echo "🚀 環境変数を設定します..."
	@echo "HOST_UID=$(HOST_UID)" > .env
	@echo "HOST_GID=$(HOST_GID)" >> .env
	@echo "USERNAME=$(USERNAME)" >> .env
	@echo "✅ .env ファイルを生成しました"
	@cat .env

build: setup
	@echo "🔨 Dockerイメージをビルドします..."
	docker compose build

up: build
	@echo "🚀 コンテナを起動します..."
	docker compose up -d

down:
	@echo "🛑 コンテナを停止します..."
	docker compose down

restart: down up

clean: down
	@echo "🧹 ボリュームとイメージを削除します..."
	docker compose down -v --rmi all

shell:
	@echo "💻 コンテナに入ります..."
	docker compose exec app bash

logs:
	docker compose logs -f

.DEFAULT_GOAL := up
