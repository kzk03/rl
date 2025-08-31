"""
メイン FastAPI アプリケーション

改善された API サーバーのエントリーポイントです。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.exceptions import APIException
from .core.logging import api_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時の処理
    settings = get_settings()
    
    # ログ設定を初期化
    setup_logging()
    api_logger.info("API サーバーを起動しています...")
    
    # 設定情報をログ出力
    api_logger.info(
        "設定情報",
        extra_data={
            "host": settings.server.host,
            "port": settings.server.port,
            "environment": "development" if settings.is_development else "production",
            "models_enabled": list(settings.model_configs.keys())
        }
    )
    
    yield
    
    # 終了時の処理
    api_logger.info("API サーバーを終了しています...")


def create_app() -> FastAPI:
    """FastAPI アプリケーションを作成"""
    settings = get_settings()
    
    # FastAPI アプリケーション作成
    app = FastAPI(
        title="Gerrit 開発者定着予測 API",
        description="企業レベルの品質を持つ堅牢な予測 API サーバー",
        version="1.0.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan
    )
    
    # CORS 設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    
    return app


# アプリケーションインスタンス
app = create_app()


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Gerrit 開発者定着予測 API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers if not settings.server.reload else 1,
        log_config=str(settings.log_config_path) if settings.log_config_path.exists() else None
    )