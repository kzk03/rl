#!/usr/bin/env python3
"""
Gerrit接続設定スクリプト

Gerrit APIへの接続をテストし、基本的な設定を検証します。
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention import initialize_system
from gerrit_retention.utils.logger import get_logger, setup_logging

# ログ設定を初期化
setup_logging(level="INFO")
logger = get_logger(__name__)


def test_gerrit_connection():
    """Gerrit接続をテスト"""
    logger.info("Gerrit接続設定をテスト中...")
    
    try:
        # システムを初期化
        config_manager = initialize_system()
        
        # 設定を検証
        if not config_manager.validate_config():
            logger.error("設定の検証に失敗しました")
            return False
        
        # Gerrit設定を取得
        gerrit_url = config_manager.get("gerrit.url")
        gerrit_username = config_manager.get("gerrit.auth.username")
        
        if not gerrit_url:
            logger.error("GERRIT_URLが設定されていません")
            return False
        
        if not gerrit_username:
            logger.error("GERRIT_USERNAMEが設定されていません")
            return False
        
        logger.info(f"Gerrit URL: {gerrit_url}")
        logger.info(f"Gerrit Username: {gerrit_username}")
        
        # TODO: 実際のGerrit APIクライアントが実装されたら接続テストを追加
        logger.info("Gerrit接続設定が正常に読み込まれました")
        logger.warning("注意: 実際のAPI接続テストはGerritクライアント実装後に追加されます")
        
        return True
        
    except Exception as e:
        logger.error(f"Gerrit接続テスト中にエラーが発生しました: {e}")
        return False


def main():
    """メイン関数"""
    logger.info("Gerrit開発者定着予測システム - 接続設定テスト")
    
    if test_gerrit_connection():
        logger.info("接続設定テストが成功しました")
        return 0
    else:
        logger.error("接続設定テストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())