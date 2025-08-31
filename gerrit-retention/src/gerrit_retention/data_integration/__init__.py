"""
Gerritデータ統合モジュール

Gerrit APIからのデータ抽出、変換、前処理を担当するモジュールです。

主要コンポーネント:
- gerrit_client: Gerrit APIクライアント
- data_extractor: データ抽出器
- data_transformer: データ変換器
"""

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

# モジュールレベルでの設定
__all__ = []

logger.debug("Gerritデータ統合モジュールが読み込まれました")