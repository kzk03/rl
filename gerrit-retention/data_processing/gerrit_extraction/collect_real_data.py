#!/usr/bin/env python3
"""
実際のGerritデータ収集スクリプト
複数のオープンソースプロジェクトから実データを収集
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

# 公開Gerritインスタンス（実際のオープンソースプロジェクト）
GERRIT_SOURCES = {
    "android": {
        "url": "https://android-review.googlesource.com",
        "projects": [
            "platform/frameworks/base",
            "platform/packages/apps/Settings", 
            "platform/system/core"
        ]
    },
    "chromium": {
        "url": "https://chromium-review.googlesource.com",
        "projects": [
            "chromium/src",
            "chromium/src/chrome"
        ]
    },
    "eclipse": {
        "url": "https://git.eclipse.org/r",
        "projects": [
            "platform/eclipse.platform.ui",
            "jdt/eclipse.jdt.core"
        ]
    }
}

class RealDataCollector:
    """実際のGerritデータ収集器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Gerrit-Retention-Research/1.0'
        })
        
    def collect_changes(self, base_url: str, project: str, limit: int = 100) -> List[Dict]:
        """Changeデータを収集"""
        logger.info(f"📥 Changeデータ収集開始: {project}")
        
        try:
            url = f"{base_url}/changes/"
            params = {
                "q": f"project:{project} status:merged",
                "n": limit,
                "o": [
                    "DETAILED_ACCOUNTS",
                    "CURRENT_REVISION", 
                    "MESSAGES",
                    "DETAILED_LABELS"
                ]
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                if content.startswith(")]}'\n"):
                    content = content[5:]
                
                changes = json.loads(content)
                logger.info(f"✅ {len(changes)}件のChangeを取得: {project}")
                return changes
            else:
                logger.error(f"❌ API呼び出し失敗: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ データ収集エラー ({project}): {e}")
            return []
    
    def collect_reviews(self, base_url: str, change_id: str) -> List[Dict]:
        """レビューデータを収集"""
        try:
            url = f"{base_url}/changes/{change_id}/revisions/current/review"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                if content.startswith(")]}'\n"):
                    content = content[5:]
                return json.loads(content)
            return []
            
        except Exception as e:
            logger.warning(f"レビューデータ取得失敗 ({change_id}): {e}")
            return []
    
    def extract_developer_info(self, changes: List[Dict]) -> List[Dict]:
        """開発者情報を抽出"""
        developers = {}
        
        for change in changes:
            # 作成者情報
            owner = change.get('owner', {})
            if owner and 'email' in owner:
                dev_id = owner['email']
                if dev_id not in developers:
                    developers[dev_id] = {
                        'developer_id': dev_id,
                        'name': owner.get('name', 'Unknown'),
                        'first_seen': change.get('created', ''),
                        'changes_authored': 0,
                        'changes_reviewed': 0,
                        'total_insertions': 0,
                        'total_deletions': 0,
                        'projects': set()
                    }
                
                developers[dev_id]['changes_authored'] += 1
                developers[dev_id]['total_insertions'] += change.get('insertions', 0)
                developers[dev_id]['total_deletions'] += change.get('deletions', 0)
                developers[dev_id]['projects'].add(change.get('project', ''))
            
            # レビュアー情報
            for label_name, label_info in change.get('labels', {}).items():
                for vote in label_info.get('all', []):
                    reviewer = vote.get('email')
                    if reviewer and reviewer != owner.get('email'):
                        if reviewer not in developers:
                            developers[reviewer] = {
                                'developer_id': reviewer,
                                'name': vote.get('name', 'Unknown'),
                                'first_seen': change.get('created', ''),
                                'changes_authored': 0,
                                'changes_reviewed': 0,
                                'total_insertions': 0,
                                'total_deletions': 0,
                                'projects': set()
                            }
                        developers[reviewer]['changes_reviewed'] += 1
        
        # セットをリストに変換
        for dev in developers.values():
            dev['projects'] = list(dev['projects'])
            
        return list(developers.values())
    
    def calculate_features(self, developers: List[Dict], changes: List[Dict]) -> List[Dict]:
        """特徴量を計算"""
        features = []
        
        for dev in developers:
            dev_id = dev['developer_id']
            
            # 基本統計
            authored = dev['changes_authored']
            reviewed = dev['changes_reviewed']
            total_activity = authored + reviewed
            
            if total_activity == 0:
                continue
                
            # 特徴量計算
            feature = {
                'developer_id': dev_id,
                'expertise_level': min(1.0, total_activity / 50.0),  # 正規化
                'collaboration_quality': min(1.0, reviewed / max(1, authored)),
                'satisfaction_level': 0.7 + (total_activity / 200.0) * 0.3,  # 推定
                'stress_accumulation': max(0.0, min(1.0, (authored - 10) / 40.0)),
                'review_load': reviewed,
                'response_time_avg': 24.0 - min(20.0, total_activity / 5.0),  # 推定
                'specialization_score': len(dev['projects']) / 10.0,
                'workload_balance': 1.0 - abs(authored - reviewed) / max(1, total_activity),
                'social_integration': min(1.0, reviewed / 20.0),
                'growth_trajectory': min(1.0, total_activity / 100.0),
                'burnout_risk': max(0.0, (authored - 20) / 30.0),
                'retention_prediction': 0.8 - max(0.0, (authored - 15) / 50.0)
            }
            
            # 値を0-1の範囲に正規化
            for key in ['satisfaction_level', 'specialization_score', 'burnout_risk', 'retention_prediction']:
                feature[key] = max(0.0, min(1.0, feature[key]))
            
            features.append(feature)
        
        return features
    
    def collect_all_data(self, max_changes_per_project: int = 50) -> Dict:
        """全データを収集"""
        logger.info("🚀 実際のGerritデータ収集開始")
        
        all_changes = []
        all_developers = []
        all_features = []
        
        for source_name, source_info in GERRIT_SOURCES.items():
            logger.info(f"📡 {source_name} からデータ収集中...")
            
            for project in source_info['projects'][:2]:  # 最初の2プロジェクトのみ
                changes = self.collect_changes(
                    source_info['url'], 
                    project, 
                    max_changes_per_project
                )
                
                if changes:
                    all_changes.extend(changes)
                    
                    # 開発者情報を抽出
                    developers = self.extract_developer_info(changes)
                    all_developers.extend(developers)
                    
                    # 特徴量を計算
                    features = self.calculate_features(developers, changes)
                    all_features.extend(features)
                
                # レート制限対策
                time.sleep(1)
        
        logger.info(f"📊 収集完了: {len(all_changes)}件のChange, {len(all_developers)}名の開発者")
        
        return {
            'changes': all_changes,
            'developers': all_developers,
            'features': all_features,
            'collection_timestamp': datetime.now().isoformat(),
            'sources': list(GERRIT_SOURCES.keys())
        }

def save_real_data():
    """実際のデータを収集・保存"""
    collector = RealDataCollector()
    
    # データ収集
    data = collector.collect_all_data(max_changes_per_project=30)
    
    # データディレクトリを作成
    data_dir = Path("data/processed/unified")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path("data/raw/gerrit_real")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 生データを保存
    with open(raw_dir / "collected_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 処理済みデータを保存
    with open(data_dir / "all_developers.json", "w", encoding="utf-8") as f:
        json.dump(data['developers'], f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "all_reviews.json", "w", encoding="utf-8") as f:
        json.dump(data['changes'], f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "all_features.json", "w", encoding="utf-8") as f:
        json.dump(data['features'], f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 実際のデータを保存しました:")
    logger.info(f"   📊 開発者: {len(data['developers'])}名")
    logger.info(f"   📝 Change: {len(data['changes'])}件")
    logger.info(f"   🎯 特徴量: {len(data['features'])}セット")
    logger.info(f"   📁 保存先: {data_dir}")
    
    return True

if __name__ == "__main__":
    save_real_data()