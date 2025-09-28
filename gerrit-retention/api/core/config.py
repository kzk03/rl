"""
設定管理システム

Pydantic Settings を使用した設定管理を実装します。
環境変数でのオーバーライドをサポートします。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, validator


class ServerConfig(BaseSettings):
    """サーバー設定"""
    host: str = Field(default="0.0.0.0", description="サーバーホスト")
    port: int = Field(default=8080, description="サーバーポート")
    reload: bool = Field(default=False, description="開発モード（自動リロード）")
    workers: int = Field(default=1, description="ワーカー数")

    class Config:
        env_prefix = "API_SERVER_"


class CORSConfig(BaseSettings):
    """CORS設定"""
    allow_origins: List[str] = Field(default=["*"], description="許可するオリジン")
    allow_credentials: bool = Field(default=True, description="クレデンシャル許可")
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="許可するHTTPメソッド"
    )
    allow_headers: List[str] = Field(default=["*"], description="許可するヘッダー")

    class Config:
        env_prefix = "API_CORS_"


class ModelConfig(BaseSettings):
    """個別モデル設定"""
    path: str = Field(..., description="モデルファイルパス")
    type: str = Field(..., description="モデルタイプ")
    enabled: bool = Field(default=True, description="モデル有効化フラグ")
    version: Optional[str] = Field(default=None, description="モデルバージョン")


class SecurityConfig(BaseSettings):
    """セキュリティ設定"""
    rate_limit_enabled: bool = Field(default=True, description="レート制限有効化")
    requests_per_minute: int = Field(default=60, description="分あたりリクエスト数")
    api_key_enabled: bool = Field(default=False, description="APIキー認証有効化")
    api_key_header: str = Field(default="X-API-Key", description="APIキーヘッダー名")

    class Config:
        env_prefix = "API_SECURITY_"


class MonitoringConfig(BaseSettings):
    """監視設定"""
    metrics_enabled: bool = Field(default=True, description="メトリクス収集有効化")
    health_check_enabled: bool = Field(default=True, description="ヘルスチェック有効化")
    request_logging: bool = Field(default=True, description="リクエストログ有効化")

    class Config:
        env_prefix = "API_MONITORING_"


class DefaultsConfig(BaseSettings):
    """デフォルト値設定"""
    satisfaction_level: float = Field(default=0.7, description="デフォルト満足度")
    confidence_threshold: float = Field(default=0.5, description="信頼度閾値")
    max_request_size: int = Field(default=10485760, description="最大リクエストサイズ")
    request_timeout: int = Field(default=30, description="リクエストタイムアウト")

    @validator('satisfaction_level', 'confidence_threshold')
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('値は0.0から1.0の間である必要があります')
        return v

    class Config:
        env_prefix = "API_DEFAULTS_"


class Settings(BaseSettings):
    """メイン設定クラス"""
    
    # 設定ファイルパス
    config_dir: Path = Field(
        default=Path("configs/api"),
        description="設定ディレクトリ（既定は configs/api/ のみを使用）",
    )
    
    # サブ設定
    server: ServerConfig = Field(default_factory=ServerConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    
    # モデル設定
    model_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # ログ設定
    log_level: str = Field(default="INFO", description="ログレベル")
    log_format: str = Field(default="json", description="ログフォーマット")

    class Config:
        env_prefix = "API_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_config_files()

    def _load_config_files(self):
        """設定ファイルを読み込み"""
        try:
            def first_existing(*paths: Path) -> Optional[Path]:
                for p in paths:
                    if p is not None and p.exists():
                        return p
                return None

            # 探索対象ディレクトリ: 指定 config_dir（既定は configs/api/）
            primary_dir = self.config_dir

            # API設定ファイル読み込み（configs/api/ のみ）
            api_config_path = first_existing(
                primary_dir / "api_config.yaml",
            )
            if api_config_path:
                with open(api_config_path, 'r', encoding='utf-8') as f:
                    api_config = yaml.safe_load(f)
                    if api_config:
                        self._update_from_dict(api_config)

            # モデル設定ファイル読み込み（configs/api/ のみ）
            model_config_path = first_existing(
                primary_dir / "model_config.yaml",
            )
            if model_config_path:
                with open(model_config_path, 'r', encoding='utf-8') as f:
                    model_config = yaml.safe_load(f)
                    if model_config and 'models' in model_config:
                        self.model_configs = model_config['models']

        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")

    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定を更新"""
        if 'server' in config_dict:
            self.server = ServerConfig(**config_dict['server'])
        if 'cors' in config_dict:
            self.cors = CORSConfig(**config_dict['cors'])
        if 'security' in config_dict:
            security_config = config_dict['security']
            # ネストした設定を平坦化
            flat_security = {}
            if 'rate_limit' in security_config:
                flat_security.update({
                    'rate_limit_enabled': security_config['rate_limit'].get('enabled', True),
                    'requests_per_minute': security_config['rate_limit'].get('requests_per_minute', 60)
                })
            if 'api_key' in security_config:
                flat_security.update({
                    'api_key_enabled': security_config['api_key'].get('enabled', False),
                    'api_key_header': security_config['api_key'].get('header_name', 'X-API-Key')
                })
            self.security = SecurityConfig(**flat_security)
        if 'monitoring' in config_dict:
            self.monitoring = MonitoringConfig(**config_dict['monitoring'])
        if 'defaults' in config_dict:
            self.defaults = DefaultsConfig(**config_dict['defaults'])
        if 'models' in config_dict:
            self.model_configs = config_dict['models']

    @property
    def is_development(self) -> bool:
        """開発モードかどうか"""
        return self.server.reload

    @property
    def log_config_path(self) -> Path:
        """ログ設定ファイルパス"""
        # ログ設定は API 用の dictConfig を期待
        # 優先: configs/api/logging_config.yaml → 最後: configs/logging_config.yaml（互換性がある場合のみ）
        candidates = [
            self.config_dir / "logging_config.yaml",
            Path("configs") / "logging_config.yaml",
        ]
        for p in candidates:
            if p.exists():
                return p
        # 既定値（存在しない場合でもパスを返す）
        return self.config_dir / "logging_config.yaml"

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """指定されたモデルの設定を取得"""
        return self.model_configs.get(model_name)

    def is_model_enabled(self, model_name: str) -> bool:
        """モデルが有効かどうか"""
        model_config = self.get_model_config(model_name)
        return model_config is not None and model_config.get('enabled', True)


# グローバル設定インスタンス
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """設定インスタンスを取得（シングルトンパターン）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """設定を再読み込み"""
    global _settings
    _settings = Settings()
    return _settings