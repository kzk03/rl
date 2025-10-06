#!/usr/bin/env python3
"""
実際のGerritサーバーへの接続スクリプト
オープンソースプロジェクトのGerritからデータを取得
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import requests

# 公開Gerritインスタンスの例
PUBLIC_GERRIT_INSTANCES = {
    "android": "https://android-review.googlesource.com",
    "chromium": "https://chromium-review.googlesource.com", 
    "eclipse": "https://git.eclipse.org/r",
    "libreoffice": "https://gerrit.libreoffice.org"
}

def connect_to_gerrit(base_url: str, project: str = None):
    """実際のGerritサーバーに接続してデータを取得"""
    print(f"🔗 Gerritサーバーに接続中: {base_url}")
    
    try:
        # 基本的なAPI呼び出し例
        changes_url = f"{base_url}/changes/"
        
        params = {
            "q": f"project:{project}" if project else "status:merged",
            "n": 25,  # 最大25件
            "o": ["DETAILED_ACCOUNTS", "CURRENT_REVISION"]
        }
        
        response = requests.get(changes_url, params=params, timeout=10)
        
        if response.status_code == 200:
            # Gerrit APIは ")]}'" プレフィックスを返すことがある
            content = response.text
            if content.startswith(")]}'\n"):
                content = content[5:]
            
            changes = json.loads(content)
            print(f"✅ {len(changes)}件のChangeを取得しました")
            return changes
        else:
            print(f"❌ API呼び出し失敗: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return None

def demo_real_connection():
    """実際の接続デモ"""
    print("=== 実際のGerritサーバー接続デモ ===")
    
    # Android Gerritに接続してみる（公開API）
    changes = connect_to_gerrit(
        PUBLIC_GERRIT_INSTANCES["android"], 
        "platform/frameworks/base"
    )
    
    if changes:
        print(f"📊 取得したデータサンプル:")
        for i, change in enumerate(changes[:3]):
            print(f"  {i+1}. {change.get('subject', 'N/A')[:50]}...")
            print(f"     作成者: {change.get('owner', {}).get('name', 'N/A')}")
            print(f"     状態: {change.get('status', 'N/A')}")
    
    return changes is not None

if __name__ == "__main__":
    demo_real_connection()