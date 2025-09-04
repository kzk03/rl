#!/usr/bin/env python3
"""
強化された開発者分析システム

作業負荷、専門性一致率、バーンアウトリスクを考慮した
包括的な開発者継続率予測システム
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.gerrit_retention.prediction.advanced_accuracy_improver import (
    AdvancedAccuracyImprover,
)
from src.gerrit_retention.prediction.workload_expertise_analyzer import (
    WorkloadExpertiseAnalyzer,
)


class EnhancedDeveloperAnalyzer:
    """強化された開発者分析システム"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.developer_data = None
        self.workload_analyzer = None
        self.accuracy_improver = None
        self.enhanced_predictions = {}
        
    def load_data_and_initialize(self):
        """データの読み込みとシステム初期化"""
        print("📊 データを読み込み中...")
        
        # 開発者データの読み込み
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.developer_data = json.load(f)
        
        # 分析システムの初期化
        config = {'output_path': 'outputs/enhanced_analysis'}
        self.workload_analyzer = WorkloadExpertiseAnalyzer(config)
        self.accuracy_improver = AdvancedAccuracyImprover(config)
        
        print(f"✅ {len(self.developer_data)}人の開発者データを読み込み完了")
    
    def analyze_all_developers_enhanced(self):
        """全開発者の強化分析"""
        print("🔍 強化された全開発者分析を開始...")
        
        for i, dev in enumerate(self.developer_data):
            if i % 100 == 0:
                print(f"   進捗: {i}/{len(self.developer_data)} ({i/len(self.developer_data)*100:.1f}%)")
            
            dev_id = dev.get('developer_id', f'developer_{i}')
            enhanced_analysis = self.analyze_single_developer_enhanced(dev, dev_id)
            self.enhanced_predictions[dev_id] = enhanced_analysis
        
        print("✅ 強化された全開発者分析完了")
    
    def analyze_single_developer_enhanced(self, dev_data: Dict[str, Any], dev_id: str) -> Dict[str, Any]:
        """単一開発者の強化分析"""
        # 基本的な継続率予測
        base_features = self.accuracy_improver.extract_advanced_features(dev_data)
        
        # 作業負荷・専門性・バーンアウト分析
        comprehensive_analysis = self.workload_analyzer.generate_comprehensive_analysis(dev_data)
        
        # 統合スコアの計算
        wellness_scores = comprehensive_analysis['wellness_scores']
        
        # 継続率予測の調整（作業負荷と専門性を考慮）
        base_retention_score = self._calculate_base_retention_score(dev_data)
        
        # 作業負荷による調整
        workload_adjustment = (wellness_scores['workload_score'] - 0.5) * 0.2
        
        # 専門性による調整
        expertise_adjustment = (wellness_scores['expertise_score'] - 0.5) * 0.15
        
        # バーンアウトリスクによる調整
        burnout_adjustment = -(1.0 - wellness_scores['burnout_score']) * 0.25
        
        # 調整後の継続率スコア
        adjusted_retention_score = base_retention_score + workload_adjustment + expertise_adjustment + burnout_adjustment
        adjusted_retention_score = max(0.0, min(1.0, adjusted_retention_score))
        
        # カテゴリとリスクレベルの再計算
        if adjusted_retention_score >= 0.8:
            category = "高継続率"
            risk_level = "低リスク"
            color = "🟢"
        elif adjusted_retention_score >= 0.5:
            category = "中継続率"
            risk_level = "中リスク"
            color = "🟡"
        else:
            category = "低継続率"
            risk_level = "高リスク"
            color = "🔴"
        
        # バーンアウトリスクによる追加分類
        burnout_level = comprehensive_analysis['burnout_risk']['burnout_level']
        if burnout_level in ['critical', 'high']:
            risk_level = "バーンアウトリスク"
            color = "🔥"
        
        # 包括的な推奨アクションの生成
        comprehensive_recommendations = self._generate_comprehensive_recommendations(
            dev_data, comprehensive_analysis, adjusted_retention_score
        )
        
        return {
            'basic_info': {
                'developer_id': dev_id,
                'name': dev_data.get('name', 'Unknown'),
                'first_seen': dev_data.get('first_seen', ''),
                'last_activity': dev_data.get('last_activity', ''),
                'projects': dev_data.get('projects', []),
                'sources': dev_data.get('sources', [])
            },
            'enhanced_prediction': {
                'base_retention_score': base_retention_score,
                'adjusted_retention_score': adjusted_retention_score,
                'workload_adjustment': workload_adjustment,
                'expertise_adjustment': expertise_adjustment,
                'burnout_adjustment': burnout_adjustment,
                'category': category,
                'risk_level': risk_level,
                'color': color
            },
            'workload_analysis': comprehensive_analysis['workload_analysis'],
            'expertise_analysis': comprehensive_analysis['expertise_analysis'],
            'burnout_risk': comprehensive_analysis['burnout_risk'],
            'wellness_scores': wellness_scores,
            'comprehensive_recommendations': comprehensive_recommendations,
            'activity_stats': {
                'changes_authored': dev_data.get('changes_authored', 0),
                'changes_reviewed': dev_data.get('changes_reviewed', 0),
                'total_insertions': dev_data.get('total_insertions', 0),
                'total_deletions': dev_data.get('total_deletions', 0),
                'project_count': len(dev_data.get('projects', [])),
                'source_count': len(dev_data.get('sources', []))
            }
        }
    
    def _calculate_base_retention_score(self, dev_data: Dict[str, Any]) -> float:
        """基本継続率スコアの計算"""
        try:
            current_time = datetime.now()
            last_activity = datetime.fromisoformat(
                dev_data.get('last_activity', '').replace(' ', 'T')
            )
            days_since_last = (current_time - last_activity).days
            
            # 時間ベーススコア
            if days_since_last <= 7:
                time_score = 1.0
            elif days_since_last <= 30:
                time_score = 0.8
            elif days_since_last <= 90:
                time_score = 0.5
            elif days_since_last <= 180:
                time_score = 0.3
            else:
                time_score = 0.1
            
            # 活動量スコア
            total_activity = dev_data.get('changes_authored', 0) + dev_data.get('changes_reviewed', 0)
            if total_activity >= 100:
                activity_score = 1.0
            elif total_activity >= 50:
                activity_score = 0.8
            elif total_activity >= 20:
                activity_score = 0.6
            elif total_activity >= 5:
                activity_score = 0.4
            else:
                activity_score = 0.2
            
            # 統合スコア
            base_score = time_score * 0.6 + activity_score * 0.4
            return base_score
            
        except:
            return 0.3
    
    def _generate_comprehensive_recommendations(self, dev_data: Dict[str, Any],
                                             comprehensive_analysis: Dict[str, Any],
                                             retention_score: float) -> List[str]:
        """包括的な推奨アクションの生成"""
        recommendations = []
        
        # 基本的な推奨アクション
        workload_recs = comprehensive_analysis['recommendations']
        recommendations.extend(workload_recs)
        
        # 継続率スコアベースの追加推奨
        if retention_score >= 0.8:
            recommendations.append("🌟 総合的に優秀な開発者です。リーダーシップ機会を提供してください")
        elif retention_score >= 0.5:
            recommendations.append("📈 継続率向上の余地があります。個別サポートを強化してください")
        else:
            recommendations.append("🚨 継続率が低いです。包括的な支援策が必要です")
        
        # 作業負荷調整の推奨
        workload_stress = comprehensive_analysis['workload_analysis'].get('workload_stress', 0.0)
        if workload_stress > 0.5:
            recommendations.append("⚖️ 作業負荷の再配分を検討してください")
        
        # 専門性マッチングの推奨
        expertise_match = comprehensive_analysis['expertise_analysis'].get('expertise_match_score', 0.0)
        if expertise_match < 0.5:
            recommendations.append("🎯 スキルマッチングの改善が必要です")
        
        # バーンアウト予防の推奨
        burnout_level = comprehensive_analysis['burnout_risk']['burnout_level']
        if burnout_level in ['high', 'critical']:
            recommendations.append("🛡️ バーンアウト予防策の即座の実施が必要です")
        
        return recommendations
    
    def display_enhanced_developer_details(self, dev_id: str):
        """強化された開発者詳細の表示"""
        analysis = self.enhanced_predictions.get(dev_id)
        if not analysis:
            print(f"❌ 開発者 {dev_id} が見つかりません")
            return
        
        basic = analysis['basic_info']
        pred = analysis['enhanced_prediction']
        workload = analysis['workload_analysis']
        expertise = analysis['expertise_analysis']
        burnout = analysis['burnout_risk']
        wellness = analysis['wellness_scores']
        stats = analysis['activity_stats']
        
        print(f"\n{pred['color']} 強化された開発者分析: {basic['name']}")
        print("=" * 80)
        
        print(f"📧 ID: {basic['developer_id']}")
        print(f"👤 名前: {basic['name']}")
        print(f"📅 初回参加: {basic['first_seen']}")
        print(f"🕒 最終活動: {basic['last_activity']}")
        
        print(f"\n🎯 強化された継続率予測:")
        print(f"   基本スコア: {pred['base_retention_score']:.4f}")
        print(f"   調整後スコア: {pred['adjusted_retention_score']:.4f} ({pred['adjusted_retention_score']*100:.1f}%)")
        print(f"   作業負荷調整: {pred['workload_adjustment']:+.4f}")
        print(f"   専門性調整: {pred['expertise_adjustment']:+.4f}")
        print(f"   バーンアウト調整: {pred['burnout_adjustment']:+.4f}")
        print(f"   カテゴリ: {pred['category']}")
        print(f"   リスクレベル: {pred['risk_level']}")
        
        print(f"\n📊 ウェルネススコア:")
        print(f"   作業負荷スコア: {wellness['workload_score']:.4f} ({wellness['workload_score']*100:.1f}%)")
        print(f"   専門性スコア: {wellness['expertise_score']:.4f} ({wellness['expertise_score']*100:.1f}%)")
        print(f"   バーンアウト耐性: {wellness['burnout_score']:.4f} ({wellness['burnout_score']*100:.1f}%)")
        print(f"   総合ウェルネス: {wellness['overall_wellness']:.4f} ({wellness['overall_wellness']*100:.1f}%)")
        
        print(f"\n🔧 作業負荷分析:")
        print(f"   日次チェンジ負荷: {workload.get('daily_changes_load', 0):.4f}")
        print(f"   日次コード負荷: {workload.get('daily_code_load', 0):.4f}")
        print(f"   作業強度: {workload.get('code_intensity', 0):.4f}")
        print(f"   負荷レベル: {workload.get('workload_level', 0):.1f}")
        print(f"   ストレス指標: {workload.get('workload_stress', 0):.4f}")
        
        print(f"\n🎯 専門性分析:")
        print(f"   専門性マッチ: {expertise.get('expertise_match_score', 0):.4f}")
        print(f"   ドメイン一貫性: {expertise.get('domain_consistency', 0):.4f}")
        print(f"   スキルアライメント: {expertise.get('skill_alignment', 0):.4f}")
        print(f"   専門性信頼度: {expertise.get('expertise_confidence', 0):.4f}")
        
        print(f"\n🔥 バーンアウトリスク:")
        print(f"   総合リスク: {burnout.get('total_burnout_risk', 0):.4f}")
        print(f"   リスクレベル: {burnout.get('burnout_level', 'unknown')}")
        print(f"   作業負荷リスク: {burnout.get('workload_burnout_risk', 0):.4f}")
        print(f"   専門性ミスマッチリスク: {burnout.get('expertise_mismatch_risk', 0):.4f}")
        print(f"   継続性リスク: {burnout.get('continuity_risk', 0):.4f}")
        
        print(f"\n📊 活動統計:")
        print(f"   作成したチェンジ: {stats['changes_authored']:,}")
        print(f"   レビューしたチェンジ: {stats['changes_reviewed']:,}")
        print(f"   総挿入行数: {stats['total_insertions']:,}")
        print(f"   総削除行数: {stats['total_deletions']:,}")
        print(f"   参加プロジェクト数: {stats['project_count']}")
        
        print(f"\n💡 包括的推奨アクション:")
        for i, rec in enumerate(analysis['comprehensive_recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🏢 参加プロジェクト:")
        for project in basic['projects'][:5]:
            print(f"   • {project}")
        if len(basic['projects']) > 5:
            print(f"   ... 他 {len(basic['projects']) - 5} プロジェクト")
    
    def get_enhanced_summary_report(self) -> Dict[str, Any]:
        """強化されたサマリーレポートの生成"""
        if not self.enhanced_predictions:
            return {}
        
        # カテゴリ別集計
        categories = {}
        risk_levels = {}
        burnout_levels = {}
        total_devs = len(self.enhanced_predictions)
        
        retention_scores = []
        wellness_scores = []
        workload_scores = []
        expertise_scores = []
        burnout_scores = []
        
        for analysis in self.enhanced_predictions.values():
            pred = analysis['enhanced_prediction']
            wellness = analysis['wellness_scores']
            burnout = analysis['burnout_risk']
            
            category = pred['category']
            risk_level = pred['risk_level']
            burnout_level = burnout['burnout_level']
            
            categories[category] = categories.get(category, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            burnout_levels[burnout_level] = burnout_levels.get(burnout_level, 0) + 1
            
            retention_scores.append(pred['adjusted_retention_score'])
            wellness_scores.append(wellness['overall_wellness'])
            workload_scores.append(wellness['workload_score'])
            expertise_scores.append(wellness['expertise_score'])
            burnout_scores.append(wellness['burnout_score'])
        
        return {
            'total_developers': total_devs,
            'categories': categories,
            'risk_levels': risk_levels,
            'burnout_levels': burnout_levels,
            'statistics': {
                'retention_scores': {
                    'mean': np.mean(retention_scores),
                    'std': np.std(retention_scores),
                    'min': np.min(retention_scores),
                    'max': np.max(retention_scores)
                },
                'wellness_scores': {
                    'mean': np.mean(wellness_scores),
                    'std': np.std(wellness_scores),
                    'min': np.min(wellness_scores),
                    'max': np.max(wellness_scores)
                },
                'workload_scores': {
                    'mean': np.mean(workload_scores),
                    'std': np.std(workload_scores)
                },
                'expertise_scores': {
                    'mean': np.mean(expertise_scores),
                    'std': np.std(expertise_scores)
                },
                'burnout_scores': {
                    'mean': np.mean(burnout_scores),
                    'std': np.std(burnout_scores)
                }
            }
        }

def main():
    """メイン実行関数"""
    print("🚀 強化された開発者分析システムを開始します")
    print("=" * 80)
    print("📋 新機能:")
    print("   • 作業負荷分析")
    print("   • 専門性一致率分析")
    print("   • バーンアウトリスク評価")
    print("   • ウェルネススコア")
    print("   • 包括的推奨アクション")
    print("=" * 80)
    
    # データパスの設定
    data_path = "data/processed/unified/all_developers.json"
    
    # システムの初期化
    analyzer = EnhancedDeveloperAnalyzer(data_path)
    
    try:
        # データの読み込みと初期化
        analyzer.load_data_and_initialize()
        
        # 全開発者の強化分析
        analyzer.analyze_all_developers_enhanced()
        
        # サマリーレポートの生成
        summary = analyzer.get_enhanced_summary_report()
        
        print("\n📊 強化された全体サマリー:")
        print("-" * 60)
        print(f"総開発者数: {summary['total_developers']:,}人")
        
        print(f"\n📈 継続率カテゴリ分布:")
        for category, count in summary['categories'].items():
            percentage = (count / summary['total_developers']) * 100
            print(f"   {category}: {count:,}人 ({percentage:.1f}%)")
        
        print(f"\n⚠️ リスクレベル分布:")
        for risk, count in summary['risk_levels'].items():
            percentage = (count / summary['total_developers']) * 100
            print(f"   {risk}: {count:,}人 ({percentage:.1f}%)")
        
        print(f"\n🔥 バーンアウトリスク分布:")
        for burnout, count in summary['burnout_levels'].items():
            percentage = (count / summary['total_developers']) * 100
            print(f"   {burnout}: {count:,}人 ({percentage:.1f}%)")
        
        print(f"\n📊 ウェルネス統計:")
        stats = summary['statistics']
        print(f"   調整後継続率: 平均={stats['retention_scores']['mean']:.4f}, "
              f"標準偏差={stats['retention_scores']['std']:.4f}")
        print(f"   総合ウェルネス: 平均={stats['wellness_scores']['mean']:.4f}, "
              f"標準偏差={stats['wellness_scores']['std']:.4f}")
        print(f"   作業負荷スコア: 平均={stats['workload_scores']['mean']:.4f}")
        print(f"   専門性スコア: 平均={stats['expertise_scores']['mean']:.4f}")
        print(f"   バーンアウト耐性: 平均={stats['burnout_scores']['mean']:.4f}")
        
        # 高リスク開発者の特定
        high_risk_devs = [analysis for analysis in analyzer.enhanced_predictions.values()
                         if analysis['enhanced_prediction']['risk_level'] in ['高リスク', 'バーンアウトリスク']]
        
        print(f"\n🚨 高リスク・バーンアウトリスク開発者: {len(high_risk_devs)}人")
        if high_risk_devs:
            print("上位5人:")
            sorted_high_risk = sorted(high_risk_devs, 
                                    key=lambda x: x['enhanced_prediction']['adjusted_retention_score'])
            
            for i, analysis in enumerate(sorted_high_risk[:5], 1):
                basic = analysis['basic_info']
                pred = analysis['enhanced_prediction']
                burnout = analysis['burnout_risk']
                print(f"{i}. {basic['name']} ({basic['developer_id']})")
                print(f"   継続率: {pred['adjusted_retention_score']:.4f}, "
                      f"バーンアウトリスク: {burnout['burnout_level']}")
        
        # インタラクティブモード
        print(f"\n🔍 強化された個別開発者詳細表示")
        print("=" * 80)
        print("開発者IDまたは名前を入力してください（'quit'で終了）:")
        
        while True:
            query = input("\n検索 > ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # 検索実行
            results = []
            query_lower = query.lower()
            
            for analysis in analyzer.enhanced_predictions.values():
                basic = analysis['basic_info']
                if (query_lower in basic['developer_id'].lower() or
                    query_lower in basic['name'].lower()):
                    results.append(analysis)
            
            if not results:
                print(f"❌ '{query}' に一致する開発者が見つかりません")
                continue
            
            if len(results) == 1:
                # 1人の場合は詳細表示
                analyzer.display_enhanced_developer_details(results[0]['basic_info']['developer_id'])
            else:
                # 複数の場合はリスト表示
                print(f"\n🔍 検索結果: {len(results)}人")
                print("-" * 60)
                for i, analysis in enumerate(results[:10], 1):
                    basic = analysis['basic_info']
                    pred = analysis['enhanced_prediction']
                    wellness = analysis['wellness_scores']
                    print(f"{i}. {basic['name']} ({basic['developer_id']})")
                    print(f"   継続率: {pred['adjusted_retention_score']:.4f}, "
                          f"ウェルネス: {wellness['overall_wellness']:.4f}")
                
                # 詳細表示の選択
                try:
                    choice = input("\n詳細を見たい番号を入力 (Enter でスキップ): ").strip()
                    if choice and choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < min(len(results), 10):
                            analyzer.display_enhanced_developer_details(results[idx]['basic_info']['developer_id'])
                except:
                    pass
        
        print("\n✅ 強化された開発者分析システムを終了します")
        return analyzer
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = main()