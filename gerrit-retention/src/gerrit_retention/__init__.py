"""
Gerrit開発者定着予測システム

開発者の長期的な貢献を促進する「開発者定着予測システム」のメインパッケージです。
このシステムは、開発者の「沸点」（ストレス限界点）を予測し、レビュー受諾行動を
最適化することで、持続可能な開発者コミュニティを構築することを目的とします。

主要モジュール:
- data_integration: Gerritデータ統合
- prediction: 定着予測・ストレス分析
- behavior_analysis: レビュー行動分析
- rl_environment: 強化学習環境
- visualization: 可視化システム
- adaptive_strategy: 適応的戦略
- utils: ユーティリティ
"""

__version__ = "0.1.0"
__author__ = "Gerrit Retention Team"
__email__ = "team@example.com"

# パッケージレベルでの主要クラス・関数のインポート
from gerrit_retention.utils.config_manager import ConfigManager
from gerrit_retention.utils.logger import get_logger

# バージョン情報
VERSION = __version__

# ログ設定の初期化
logger = get_logger(__name__)

def get_version() -> str:
    """パッケージバージョンを取得"""
    return __version__

def initialize_system(config_path: str = None) -> ConfigManager:
    """
    システムを初期化
    
    Args:
        config_path: 設定ファイルのパス（オプション）
        
    Returns:
        ConfigManager: 設定管理インスタンス
    """
    logger.info(f"Gerrit開発者定着予測システム v{__version__} を初期化中...")
    
    config_manager = ConfigManager(config_path)
    
    logger.info("システムの初期化が完了しました")
    return config_manager

# パッケージレベルでの設定
__all__ = [
    "VERSION",
    "get_version", 
    "initialize_system",
    "ConfigManager",
    "get_logger",
]