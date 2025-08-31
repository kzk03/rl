# Gerrit開発者定着予測システム - 本番対応Dockerfile

# マルチステージビルドを使用してイメージサイズを最適化
FROM python:3.11-slim as builder

# ビルド時の依存関係をインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# uvをインストール
RUN pip install --no-cache-dir uv

# 作業ディレクトリを設定
WORKDIR /app

# プロジェクトファイルをコピー（依存関係のみ先にコピーしてキャッシュを活用）
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY README.md ./

# 依存関係をインストール（仮想環境を作成）
RUN uv sync --frozen

# 本番用イメージ
FROM python:3.11-slim as production

# 非rootユーザーを作成
RUN groupadd -r gerrit && useradd -r -g gerrit gerrit

# 必要なランタイム依存関係のみインストール
RUN apt-get update && apt-get install -y \
    curl \
    libhdf5-310 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 作業ディレクトリを設定
WORKDIR /app

# ビルダーステージから仮想環境をコピー
COPY --from=builder /app/.venv /app/.venv

# アプリケーションファイルをコピー
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY training/ training/
COPY data_processing/ data_processing/
COPY evaluation/ evaluation/
COPY analysis/ analysis/
COPY pipelines/ pipelines/

# ディレクトリを作成
RUN mkdir -p data logs models outputs && \
    chown -R gerrit:gerrit /app

# 環境変数を設定
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1

# パフォーマンス最適化の環境変数
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_MAX_THREADS=4

# ヘルスチェック用のスクリプトを作成
COPY docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# ヘルスチェックを設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /healthcheck.sh

# 非rootユーザーに切り替え
USER gerrit

# ポートを公開
EXPOSE 8080 8081

# デフォルトコマンド（APIサーバーを起動）
CMD ["python", "scripts/run_api_server.py"]

# 開発用イメージ
FROM production as development

# rootユーザーに戻る（開発用ツールのインストールのため）
USER root

# 開発用ツールをインストール
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    git \
    && rm -rf /var/lib/apt/lists/*

# Jupyter Notebookをインストール
RUN . /app/.venv/bin/activate && pip install --no-cache-dir jupyter jupyterlab

# 開発用の環境変数
ENV LOG_LEVEL=DEBUG
ENV ENVIRONMENT=development

# 開発用ユーザーに切り替え
USER gerrit

# Jupyter用ポートを追加
EXPOSE 8888

# 開発用デフォルトコマンド
CMD ["python", "scripts/run_full_training_pipeline.py", "--development"]