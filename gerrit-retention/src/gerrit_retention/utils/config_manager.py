"""
設定管理ユーティリティ

YAML設定ファイルの読み込み、環境変数の管理、設定の検証を行います。
環境別設定管理と設定変更影響分析機能を提供します。

ポリシー:
- 本ユーティリティはリポジトリ横断の環境/パイプライン設定を対象とし、
    既定の探索先は `configs/` 配下（例: `configs/gerrit_config.yaml`, `configs/development.yaml`）。
- `config/` はAPI専用のレガシー/互換ディレクトリであり、本ユーティリティの探索対象ではありません。
    誤って `config/` を指定した場合は警告を出します（読み込みは行いません）。
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigChange:
    """設定変更情報"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    change_type: str  # 'added', 'modified', 'deleted'
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    description: str


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 environment: Optional[str] = None):
        """
        設定管理を初期化
        
        Args:
            config_path: 設定ファイルのパス（オプション）
            environment: 環境名（development, testing, production）
        """
        self.config_path = config_path
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config: Optional[DictConfig] = None
        self.config_history: List[ConfigChange] = []
        self.config_hash: Optional[str] = None
        
        # 重要設定キーの定義（影響度分析用）
        self.critical_keys = {
            "gerrit.url", "gerrit.auth.username", "gerrit.auth.password",
            "ml.random_seed", "temporal_consistency.enable_strict_validation"
        }
        self.high_impact_keys = {
            "retention_prediction.model_type", "stress_analysis.weights",
            "rl_environment.reward_weights", "ppo_agent.learning_rates"
        }
        
        # レガシーAPI用ディレクトリ（config/）が誤指定されていないかの注意喚起
        self._warn_if_legacy_api_dir()

        self._load_config()

    def _warn_if_legacy_api_dir(self) -> None:
        """`config/`（API 専用）を誤って指定した場合に警告する"""
        if not self.config_path:
            return
        try:
            p = Path(self.config_path)
        except Exception:
            return
        # `config` ディレクトリ配下、あるいはそのファイルを指している場合に警告
        parts = [part.lower() for part in p.parts]
        if "config" in parts and "configs" not in parts:
            logger.warning(
                "ConfigManager は `configs/` 配下の環境設定を対象とします。"
                " 指定されたパスは `config/`（API 用）配下の可能性があります: %s",
                str(self.config_path)
            )
    
    def _load_config(self) -> None:
        """設定ファイルを読み込み（環境別設定対応）"""
        # 環境別設定ファイルの優先順位
        config_files = self._get_config_file_paths()
        
        # 基本設定を読み込み
        base_config = {}
        for config_file in config_files:
            if Path(config_file).exists():
                logger.info(f"設定ファイルを読み込み中: {config_file}")
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # 設定をマージ
                base_config = self._deep_merge_dict(base_config, file_config)
        
        if base_config:
            self.config = OmegaConf.create(base_config)
            
            # 環境変数で設定を上書き
            self._override_with_env_vars()
            
            # 設定ハッシュを計算
            self._update_config_hash()
            
            logger.info(f"設定ファイルの読み込みが完了しました（環境: {self.environment}）")
        else:
            logger.warning("設定ファイルが見つかりません。デフォルト設定を使用します")
            self.config = OmegaConf.create({})
    
    def _get_config_file_paths(self) -> List[str]:
        """環境別設定ファイルパスを取得"""
        if self.config_path:
            # 単一のファイルパスが指定された場合のみそのまま使用。
            # ディレクトリが指定された場合は `configs/` 相当の想定で標準ファイル名を探索。
            p = Path(self.config_path)
            if p.is_file():
                return [str(p)]
            if p.is_dir():
                return [
                    str(p / "gerrit_config.yaml"),
                    str(p / f"{self.environment}.yaml"),
                    str(p / "local.yaml"),
                ]
            # 存在しない場合はそのまま返す（呼び出し側の意図を尊重）
            return [str(p)]
        
        # 環境別設定ファイルの優先順位
        config_files = [
            "configs/gerrit_config.yaml",  # ベース設定
            f"configs/{self.environment}.yaml",  # 環境別設定
            "configs/local.yaml",  # ローカル設定（オプション）
        ]
        
        return config_files
    
    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の深いマージ"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _update_config_hash(self) -> None:
        """設定のハッシュ値を更新"""
        if self.config:
            config_str = OmegaConf.to_yaml(self.config)
            self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    def get_config_hash(self) -> Optional[str]:
        """設定のハッシュ値を取得"""
        return self.config_hash
    
    def _override_with_env_vars(self) -> None:
        """環境変数で設定を上書き"""
        env_mappings = {
            "GERRIT_URL": "gerrit.url",
            "GERRIT_USERNAME": "gerrit.auth.username", 
            "GERRIT_PASSWORD": "gerrit.auth.password",
            "LOG_LEVEL": "logging.level",
            "ML_RANDOM_SEED": "ml.random_seed",
            "ML_DEVICE": "ml.device",
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # ネストしたキーを設定
                keys = config_key.split('.')
                current = self.config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = env_value
                logger.debug(f"環境変数 {env_var} で設定を上書き: {config_key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得
        
        Args:
            key: 設定キー（ドット記法対応）
            default: デフォルト値
            
        Returns:
            設定値
        """
        if self.config is None:
            return default
        
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception as e:
            logger.warning(f"設定キー '{key}' の取得に失敗: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        設定値を設定
        
        Args:
            key: 設定キー（ドット記法対応）
            value: 設定値
        """
        if self.config is None:
            self.config = OmegaConf.create({})
        
        # ネストしたキーを設定
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        logger.debug(f"設定を更新: {key} = {value}")
    
    def get_config(self) -> DictConfig:
        """
        全設定を取得
        
        Returns:
            DictConfig: 全設定
        """
        return self.config or OmegaConf.create({})
    
    def validate_config(self) -> bool:
        """
        設定の妥当性を検証
        
        Returns:
            bool: 設定が有効かどうか
        """
        required_keys = [
            "gerrit.url",
            "gerrit.auth.username",
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"必須設定が不足しています: {key}")
                return False
        
        logger.info("設定の検証が完了しました")
        return True
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        設定をファイルに保存
        
        Args:
            output_path: 出力ファイルパス
        """
        if self.config is None:
            logger.warning("保存する設定がありません")
            return
        
        config_dict = OmegaConf.to_yaml(self.config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(config_dict)
        
        logger.info(f"設定をファイルに保存しました: {output_path}")
    
    def merge_config(self, other_config: Union[Dict[str, Any], DictConfig]) -> None:
        """
        他の設定をマージ
        
        Args:
            other_config: マージする設定
        """
        if self.config is None:
            self.config = OmegaConf.create({})
        
        # 変更前の設定を保存
        old_config = OmegaConf.to_container(self.config, resolve=True)
        
        if isinstance(other_config, dict):
            other_config = OmegaConf.create(other_config)
        
        self.config = OmegaConf.merge(self.config, other_config)
        
        # 変更を記録
        new_config = OmegaConf.to_container(self.config, resolve=True)
        self._record_config_changes(old_config, new_config)
        
        # ハッシュを更新
        self._update_config_hash()
        
        logger.info("設定をマージしました")
    
    def analyze_config_impact(self, new_config: Union[Dict[str, Any], DictConfig]) -> List[ConfigChange]:
        """
        設定変更の影響を分析
        
        Args:
            new_config: 新しい設定
            
        Returns:
            List[ConfigChange]: 設定変更のリスト
        """
        if self.config is None:
            return []
        
        old_config = OmegaConf.to_container(self.config, resolve=True)
        
        if isinstance(new_config, DictConfig):
            new_config = OmegaConf.to_container(new_config, resolve=True)
        
        changes = []
        self._compare_configs(old_config, new_config, "", changes)
        
        return changes
    
    def _record_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """設定変更を記録"""
        changes = []
        self._compare_configs(old_config, new_config, "", changes)
        self.config_history.extend(changes)
        
        # 履歴の上限を設定（最新100件まで）
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
    
    def _compare_configs(self, old: Any, new: Any, path: str, changes: List[ConfigChange]) -> None:
        """設定を再帰的に比較"""
        if isinstance(old, dict) and isinstance(new, dict):
            # 削除されたキー
            for key in old.keys() - new.keys():
                full_path = f"{path}.{key}" if path else key
                changes.append(ConfigChange(
                    timestamp=datetime.now(),
                    key=full_path,
                    old_value=old[key],
                    new_value=None,
                    change_type="deleted",
                    impact_level=self._assess_impact_level(full_path),
                    description=f"設定キー '{full_path}' が削除されました"
                ))
            
            # 追加されたキー
            for key in new.keys() - old.keys():
                full_path = f"{path}.{key}" if path else key
                changes.append(ConfigChange(
                    timestamp=datetime.now(),
                    key=full_path,
                    old_value=None,
                    new_value=new[key],
                    change_type="added",
                    impact_level=self._assess_impact_level(full_path),
                    description=f"設定キー '{full_path}' が追加されました"
                ))
            
            # 変更されたキー
            for key in old.keys() & new.keys():
                full_path = f"{path}.{key}" if path else key
                self._compare_configs(old[key], new[key], full_path, changes)
        
        elif old != new:
            changes.append(ConfigChange(
                timestamp=datetime.now(),
                key=path,
                old_value=old,
                new_value=new,
                change_type="modified",
                impact_level=self._assess_impact_level(path),
                description=f"設定 '{path}' が {old} から {new} に変更されました"
            ))
    
    def _assess_impact_level(self, key: str) -> str:
        """設定変更の影響レベルを評価"""
        if key in self.critical_keys:
            return "critical"
        elif key in self.high_impact_keys:
            return "high"
        elif any(pattern in key for pattern in ["auth", "password", "secret", "key"]):
            return "critical"
        elif any(pattern in key for pattern in ["model", "learning", "training"]):
            return "high"
        elif any(pattern in key for pattern in ["visualization", "logging", "cache"]):
            return "low"
        else:
            return "medium"
    
    def get_config_history(self, limit: Optional[int] = None) -> List[ConfigChange]:
        """
        設定変更履歴を取得
        
        Args:
            limit: 取得する履歴の上限数
            
        Returns:
            List[ConfigChange]: 設定変更履歴
        """
        history = sorted(self.config_history, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def export_config_changes(self, output_path: Union[str, Path]) -> None:
        """
        設定変更履歴をファイルにエクスポート
        
        Args:
            output_path: 出力ファイルパス
        """
        changes_data = []
        
        for change in self.config_history:
            changes_data.append({
                "timestamp": change.timestamp.isoformat(),
                "key": change.key,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "change_type": change.change_type,
                "impact_level": change.impact_level,
                "description": change.description
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(changes_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"設定変更履歴をエクスポートしました: {output_path}")
    
    def validate_environment_config(self) -> Tuple[bool, List[str]]:
        """
        環境別設定の妥当性を検証
        
        Returns:
            Tuple[bool, List[str]]: (検証結果, エラーメッセージリスト)
        """
        errors = []
        
        # 環境別必須設定の検証
        env_required_keys = {
            "production": [
                "gerrit.url", "gerrit.auth.username", "gerrit.auth.password",
                "security.enable_ssl_verification", "monitoring.enable_metrics"
            ],
            "development": [
                "gerrit.url", "debug.enabled"
            ],
            "testing": [
                "testing.mock_gerrit_api", "testing.use_sample_data"
            ]
        }
        
        required_keys = env_required_keys.get(self.environment, [])
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"環境 '{self.environment}' で必須設定が不足: {key}")
        
        # 環境固有の検証
        if self.environment == "production":
            # 本番環境での追加検証
            if self.get("logging.level") == "DEBUG":
                errors.append("本番環境でDEBUGログレベルは推奨されません")
            
            if not self.get("security.enable_ssl_verification", True):
                errors.append("本番環境でSSL検証を無効にすることは推奨されません")
        
        elif self.environment == "development":
            # 開発環境での追加検証
            if self.get("cache.backend") == "redis":
                logger.warning("開発環境でRedisキャッシュを使用しています")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"環境 '{self.environment}' の設定検証が完了しました")
        else:
            logger.error(f"環境 '{self.environment}' の設定に問題があります: {errors}")
        
        return is_valid, errors


# グローバル設定管理インスタンス
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Union[str, Path]] = None,
                      environment: Optional[str] = None) -> ConfigManager:
    """
    グローバル設定管理インスタンスを取得
    
    Args:
        config_path: 設定ファイルのパス（初回のみ使用）
        environment: 環境名（初回のみ使用）
        
    Returns:
        ConfigManager: 設定管理インスタンス
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_path, environment)
    
    return _global_config_manager


def reload_config_manager(config_path: Optional[Union[str, Path]] = None,
                         environment: Optional[str] = None) -> ConfigManager:
    """
    設定管理インスタンスを再読み込み
    
    Args:
        config_path: 設定ファイルのパス
        environment: 環境名
        
    Returns:
        ConfigManager: 新しい設定管理インスタンス
    """
    global _global_config_manager
    _global_config_manager = ConfigManager(config_path, environment)
    return _global_config_manager


def get_environment_config(environment: str) -> ConfigManager:
    """
    指定環境の設定管理インスタンスを取得
    
    Args:
        environment: 環境名
        
    Returns:
        ConfigManager: 環境別設定管理インスタンス
    """
    return ConfigManager(environment=environment)