#!/bin/bash

# Gerrit開発者定着予測システム - ヘルスチェックスクリプト

set -e

# 基本的なPythonインポートテスト
python -c "
import sys
import os
sys.path.insert(0, '/app/src')

try:
    # コアモジュールのインポートテスト
    import gerrit_retention
    from gerrit_retention.utils.config_manager import get_config_manager
    from gerrit_retention.utils.logger import get_logger
    
    # 設定管理のテスト
    config_manager = get_config_manager()
    
    # ログシステムのテスト
    logger = get_logger('healthcheck')
    logger.info('ヘルスチェック実行中')
    
    print('OK: 基本モジュールの読み込み成功')
    
except ImportError as e:
    print(f'ERROR: モジュールインポートエラー: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: ヘルスチェックエラー: {e}')
    sys.exit(1)
"

# ディスク容量チェック
DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "WARNING: ディスク使用量が90%を超えています: ${DISK_USAGE}%"
fi

# メモリ使用量チェック（利用可能な場合）
if command -v free >/dev/null 2>&1; then
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$MEMORY_USAGE" -gt 90 ]; then
        echo "WARNING: メモリ使用量が90%を超えています: ${MEMORY_USAGE}%"
    fi
fi

# 必要なディレクトリの存在確認
for dir in data logs models outputs; do
    if [ ! -d "/app/$dir" ]; then
        echo "ERROR: 必要なディレクトリが存在しません: $dir"
        exit 1
    fi
done

# 設定ファイルの存在確認
if [ ! -f "/app/configs/gerrit_config.yaml" ]; then
    echo "ERROR: 基本設定ファイルが存在しません"
    exit 1
fi

echo "OK: ヘルスチェック完了"
exit 0