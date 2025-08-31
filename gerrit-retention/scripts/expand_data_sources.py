#!/usr/bin/env python3
"""
データソース拡張スクリプト
Mozilla、OpenStack Nova、LibreOfficeなど他の主要OSSプロジェクトを追加
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

# 拡張Gerritソース（主要OSSプロジェクト）
EXPANDED_GERRIT_SOURCES = {
    "android": {
        "url": "https://android-review.googlesource.com",
        "projects": [
            "platform/frameworks/base",
            "platform/packages/apps/Settings", 
            "platform/system/core",
            "platform/packages/apps/Camera2",
            "platform/frameworks/native"
        ]
    },
    "chromium": {
        "url": "https://chromium-review.googlesource.com",
        "projects": [
            "chromium/src",
            "chromium/src/chrome",
            "chromium/src/v8",
            "chromium/src/third_party/blink"
        ]
    },
    "eclipse": {
        "url": "https://git.eclipse.org/r",
        "projects": [
            "platform/eclipse.platform.ui",
            "jdt/eclipse.jdt.core",
            "platform/eclipse.platform.runtime",
            "jdt/eclipse.jdt.ui"
        ]
    },
    "openstack": {
        "url": "https://review.opendev.org",
        "projects": [
            "openstack/nova",           # コンピュートサービス
            "openstack/neutron",        # ネットワークサービス
            "openstack/cinder",         # ブロックストレージ
            "openstack/keystone",       # アイデンティティサービス
            "openstack/glance",         # イメージサービス
            "openstack/swift",          # オブジェクトストレージ
            "openstack/heat",           # オーケストレーション
            "openstack/horizon"         # ダッシュボード
        ]
    },
    "libreoffice": {
        "url": "https://gerrit.libreoffice.org",
        "projects": [
            "core",                     # メインコア
            "help",                     # ヘルプシステム
            "translations",             # 翻訳
            "dictionaries"              # 辞書
        ]
    },
    "wikimedia": {
        "url": "https://gerrit.wikimedia.org/r",
        "projects": [
            "mediawiki/core",           # MediaWikiコア
            "mediawiki/extensions/VisualEditor",
            "mediawiki/extensions/Wikibase",
            "operations/puppet"         # インフラ管理
        ]
    },
    "go": {
        "url": "https://go-review.googlesource.com",
        "projects": [
            "go",                       # Go言語本体
            "tools",                    # Go開発ツール
            "crypto",                   # 暗号化ライブラリ
            "net"                       # ネットワークライブラリ
        ]
    },
    "qt": {
        "url": "https://codereview.qt-project.org",
        "projects": [
            "qt/qtbase",                # Qtベース
            "qt/qtdeclarative",         # QML/Qt Quick
            "qt/qtwidgets",             # ウィジェット
            "qt/qtnetwork"              # ネットワーク
        ]
    }
}

class ExpandedDataCollector:
    """拡張データ収集器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Gerrit-Retention-Research/2.0'
        })
        
    def collect_changes_with_retry(self, base_url: str, project: str, limit: int = 100, max_retries: int = 3) -> List[Dict]:
        """リトライ機能付きChangeデータ収集"""
        logger.info(f"📥 Changeデータ収集開始: {project}")
        
        for attempt in range(max_retries):
            try:
                url = f"{base_url}/changes/"
                params = {
                    "q": f"project:{project} status:merged",
                    "n": limit,
                    "o": [
                        "DETAILED_ACCOUNTS",
                        "CURRENT_REVISION", 
                        "MESSAGES",
                        "DETAILED_LABELS",
                        "CURRENT_FILES"
                    ]
                }
                
                response = self.session.get(url, params=params, timeout=45)
                
                if response.status_code == 200:
                    content = response.text
                    if content.startswith(")]}'\n"):
                        content = content[5:]
                    
                    changes = json.loads(content)
                    logger.info(f"✅ {len(changes)}件のChangeを取得: {project}")
                    return changes
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"⏳ レート制限 - {wait_time}秒待機中...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"❌ API呼び出し失敗: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []
                    
            except Exception as e:
                logger.error(f"❌ データ収集エラー ({project}, 試行 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return []
        
        return []
    
    def analyze_project_activity(self, changes: List[Dict]) -> Dict:
        """プロジェクト活動分析"""
        if not changes:
            return {}
        
        # 期間分析
        dates = []
        for change in changes:
            if change.get('created'):
                dates.append(change['created'][:10])  # YYYY-MM-DD
        
        # 開発者分析
        authors = set()
        reviewers = set()
        
        for change in changes:
            if change.get('owner', {}).get('email'):
                authors.add(change['owner']['email'])
            
            for label_name, label_info in change.get('labels', {}).items():
                for vote in label_info.get('all', []):
                    if vote.get('email'):
                        reviewers.add(vote['email'])
        
        # 複雑度分析
        total_insertions = sum(change.get('insertions', 0) for change in changes)
        total_deletions = sum(change.get('deletions', 0) for change in changes)
        
        return {
            'total_changes': len(changes),
            'unique_authors': len(authors),
            'unique_reviewers': len(reviewers),
            'date_range': {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None
            },
            'code_changes': {
                'total_insertions': total_insertions,
                'total_deletions': total_deletions,
                'avg_insertions_per_change': total_insertions / len(changes) if changes else 0,
                'avg_deletions_per_change': total_deletions / len(changes) if changes else 0
            }
        }
    
    def collect_comprehensive_data(self, 
                                 sources_to_collect: List[str] = None,
                                 max_changes_per_project: int = 100) -> Dict:
        """包括的データ収集"""
        logger.info("🚀 拡張Gerritデータ収集開始")
        
        if sources_to_collect is None:
            sources_to_collect = list(EXPANDED_GERRIT_SOURCES.keys())
        
        all_changes = []
        all_developers = []
        project_stats = {}
        
        for source_name in sources_to_collect:
            if source_name not in EXPANDED_GERRIT_SOURCES:
                logger.warning(f"⚠️ 未知のソース: {source_name}")
                continue
                
            source_info = EXPANDED_GERRIT_SOURCES[source_name]
            logger.info(f"📡 {source_name} からデータ収集中...")
            
            source_changes = []
            source_developers = []
            
            for project in source_info['projects']:
                logger.info(f"  🔍 プロジェクト: {project}")
                
                changes = self.collect_changes_with_retry(
                    source_info['url'], 
                    project, 
                    max_changes_per_project
                )
                
                if changes:
                    source_changes.extend(changes)
                    all_changes.extend(changes)
                    
                    # プロジェクト統計
                    project_stats[f"{source_name}/{project}"] = self.analyze_project_activity(changes)
                    
                    # 開発者情報を抽出
                    developers = self.extract_developer_info(changes, source_name, project)
                    source_developers.extend(developers)
                    all_developers.extend(developers)
                
                # レート制限対策
                time.sleep(2)
            
            logger.info(f"✅ {source_name}: {len(source_changes)}件のChange, {len(source_developers)}名の開発者")
        
        # 重複除去
        unique_developers = self.deduplicate_developers(all_developers)
        
        logger.info(f"📊 総収集結果:")
        logger.info(f"   📝 Change: {len(all_changes)}件")
        logger.info(f"   👥 開発者: {len(unique_developers)}名（重複除去後）")
        logger.info(f"   🏢 プロジェクト: {len(project_stats)}個")
        
        return {
            'changes': all_changes,
            'developers': unique_developers,
            'project_statistics': project_stats,
            'collection_timestamp': datetime.now().isoformat(),
            'sources': sources_to_collect,
            'collection_summary': {
                'total_changes': len(all_changes),
                'total_developers': len(unique_developers),
                'total_projects': len(project_stats)
            }
        }
    
    def extract_developer_info(self, changes: List[Dict], source: str, project: str) -> List[Dict]:
        """開発者情報抽出（拡張版）"""
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
                        'projects': set(),
                        'sources': set(),
                        'last_activity': change.get('created', ''),
                        'review_scores': []
                    }
                
                developers[dev_id]['changes_authored'] += 1
                developers[dev_id]['total_insertions'] += change.get('insertions', 0)
                developers[dev_id]['total_deletions'] += change.get('deletions', 0)
                developers[dev_id]['projects'].add(project)
                developers[dev_id]['sources'].add(source)
                
                # 最新活動日を更新
                if change.get('created', '') > developers[dev_id]['last_activity']:
                    developers[dev_id]['last_activity'] = change.get('created', '')
            
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
                                'projects': set(),
                                'sources': set(),
                                'last_activity': change.get('created', ''),
                                'review_scores': []
                            }
                        
                        developers[reviewer]['changes_reviewed'] += 1
                        developers[reviewer]['projects'].add(project)
                        developers[reviewer]['sources'].add(source)
                        
                        # レビュースコアを記録
                        if 'value' in vote:
                            developers[reviewer]['review_scores'].append(vote['value'])
        
        # セットをリストに変換
        for dev in developers.values():
            dev['projects'] = list(dev['projects'])
            dev['sources'] = list(dev['sources'])
            
        return list(developers.values())
    
    def deduplicate_developers(self, developers: List[Dict]) -> List[Dict]:
        """開発者重複除去"""
        unique_devs = {}
        
        for dev in developers:
            dev_id = dev['developer_id']
            if dev_id not in unique_devs:
                unique_devs[dev_id] = dev
            else:
                # 既存の開発者情報をマージ
                existing = unique_devs[dev_id]
                existing['changes_authored'] += dev['changes_authored']
                existing['changes_reviewed'] += dev['changes_reviewed']
                existing['total_insertions'] += dev['total_insertions']
                existing['total_deletions'] += dev['total_deletions']
                
                # プロジェクトとソースをマージ
                existing['projects'] = list(set(existing['projects'] + dev['projects']))
                existing['sources'] = list(set(existing['sources'] + dev['sources']))
                
                # より早い初回活動日を採用
                if dev['first_seen'] < existing['first_seen']:
                    existing['first_seen'] = dev['first_seen']
                
                # より遅い最終活動日を採用
                if dev['last_activity'] > existing['last_activity']:
                    existing['last_activity'] = dev['last_activity']
                
                # レビュースコアをマージ
                existing['review_scores'].extend(dev['review_scores'])
        
        return list(unique_devs.values())

def collect_expanded_data():
    """拡張データ収集の実行"""
    collector = ExpandedDataCollector()
    
    # 収集するソースを選択（全部または一部）
    sources_to_collect = [
        "openstack",      # Nova, Neutron, Cinder等
        "libreoffice",    # LibreOffice
        "wikimedia",      # MediaWiki
        "android",        # Android（既存）
        "chromium"        # Chromium（既存）
    ]
    
    logger.info(f"🎯 収集対象: {', '.join(sources_to_collect)}")
    
    # データ収集実行
    data = collector.collect_comprehensive_data(
        sources_to_collect=sources_to_collect,
        max_changes_per_project=150  # プロジェクトあたり最大150件
    )
    
    # データディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data_dir = Path("data/processed/unified")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path("data/raw/expanded_gerrit")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 拡張生データを保存
    with open(raw_dir / f"expanded_data_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 統合処理済みデータを保存（既存データと統合）
    with open(data_dir / "all_developers.json", "w", encoding="utf-8") as f:
        json.dump(data['developers'], f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "all_reviews.json", "w", encoding="utf-8") as f:
        json.dump(data['changes'], f, indent=2, ensure_ascii=False)
    
    # プロジェクト統計を保存
    with open(data_dir / f"project_statistics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(data['project_statistics'], f, indent=2, ensure_ascii=False)
    
    # サマリーレポートを生成
    summary_report = generate_summary_report(data)
    with open(data_dir / f"collection_summary_{timestamp}.md", "w", encoding="utf-8") as f:
        f.write(summary_report)
    
    logger.info(f"✅ 拡張データ収集完了:")
    logger.info(f"   📊 開発者: {len(data['developers'])}名")
    logger.info(f"   📝 Change: {len(data['changes'])}件")
    logger.info(f"   🏢 プロジェクト: {len(data['project_statistics'])}個")
    logger.info(f"   📁 保存先: {data_dir}")
    
    return data

def generate_summary_report(data: Dict) -> str:
    """サマリーレポート生成"""
    report = f"""# 拡張Gerritデータ収集サマリー

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 収集概要

- **総Change数**: {data['collection_summary']['total_changes']:,}件
- **総開発者数**: {data['collection_summary']['total_developers']:,}名
- **総プロジェクト数**: {data['collection_summary']['total_projects']}個
- **収集ソース**: {', '.join(data['sources'])}

## プロジェクト別統計

"""
    
    for project_name, stats in data['project_statistics'].items():
        report += f"""### {project_name}

- Change数: {stats['total_changes']}件
- 作成者数: {stats['unique_authors']}名
- レビュアー数: {stats['unique_reviewers']}名
- 期間: {stats['date_range']['earliest']} ～ {stats['date_range']['latest']}
- 平均挿入行数: {stats['code_changes']['avg_insertions_per_change']:.1f}行
- 平均削除行数: {stats['code_changes']['avg_deletions_per_change']:.1f}行

"""
    
    # 開発者活動統計
    developers = data['developers']
    if developers:
        authored_counts = [dev.get('changes_authored', 0) for dev in developers]
        reviewed_counts = [dev.get('changes_reviewed', 0) for dev in developers]
        
        report += f"""## 開発者活動統計

- 平均作成Change数: {sum(authored_counts)/len(authored_counts):.2f}件
- 平均レビューChange数: {sum(reviewed_counts)/len(reviewed_counts):.2f}件
- 最大作成Change数: {max(authored_counts)}件
- 最大レビューChange数: {max(reviewed_counts)}件
- 複数プロジェクト参加者: {len([d for d in developers if len(d.get('projects', [])) > 1])}名

## 次のステップ

1. **継続要因分析の実行**
   ```bash
   uv run --directory gerrit-retention python -m gerrit_retention.cli analyze --type all
   ```

2. **可視化の生成**
   ```bash
   uv run --directory gerrit-retention python scripts/generate_visualizations.py
   ```

3. **予測モデルの訓練**
   ```bash
   uv run --directory gerrit-retention python -m gerrit_retention.cli train --component all
   ```
"""
    
    return report

if __name__ == "__main__":
    collect_expanded_data()