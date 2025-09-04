#!/usr/bin/env python3
"""
個別開発者分析システム

各開発者ごとの継続率予測、特徴量分析、推奨アクションを
詳細に表示するシステム
"""

import json
import pickle
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


class IndividualDeveloperAnalyzer:
    """個別開発者分析システム"""
    
    def __init__(self, models_path: str, data_path: str):
        self.models_path = models_path
        self.data_path = data_path
        self.developer_data = None
        self.improver = None
        self.feature_names = None
        self.predictions = {}
        self.feature_analysis = {}
        
    def load_data_and_models(self):
        """データとモデルの読み込み"""
        print("📊 データとモデルを読み込み中...")
        
        # 開発者データの読み込み
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.developer_data = json.load(f)
        
        # モデルの読み込み
        with open(self.models_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # AdvancedAccuracyImproverの初期化
        config = {'output_path': 'outputs/individual_analysis'}
        self.improver = AdvancedAccuracyImprover(config)
        self.improver.models = model_data['models']
        self.improver.scalers = model_data['scalers']
        self.improver.ensemble_weights = model_data['ensemble_weights']
        self.improver.feature_importance = model_data['feature_importance']
        
        print(f"✅ {len(self.developer_data)}人の開発者データを読み込み完了")
        print(f"✅ {len(self.improver.models)}個のモデルを読み込み完了")
    
    def analyze_all_developers(self):
        """全開発者の分析"""
        print("🔍 全開発者の分析を開始...")
        
        for i, dev in enumerate(self.developer_data):
            if i % 100 == 0:
                print(f"   進捗: {i}/{len(self.developer_data)} ({i/len(self.developer_data)*100:.1f}%)")
            
            dev_id = dev.get('developer_id', f'developer_{i}')
            analysis = self.analyze_single_developer(dev, dev_id)
            self.predictions[dev_id] = analysis
        
        print("✅ 全開発者の分析完了")
    
    def analyze_single_developer(self, dev_data: Dict[str, Any], dev_id: str) -> Dict[str, Any]:
        """単一開発者の詳細分析"""
        # 特徴量の抽出
        features = self.improver.extract_advanced_features(dev_data)
        
        if self.feature_names is None:
            self.feature_names = list(features.keys())
        
        feature_vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        
        # 予測の実行
        try:
            prediction, uncertainty = self.improver.predict_with_ensemble(feature_vector)
            retention_score = float(prediction[0])
            confidence = 1.0 - float(uncertainty[0])
        except Exception as e:
            retention_score = 0.5
            confidence = 0.0
        
        # 継続率カテゴリの判定
        if retention_score >= 0.8:
            category = "高継続率"
            risk_level = "低リスク"
            color = "🟢"
        elif retention_score >= 0.5:
            category = "中継続率"
            risk_level = "中リスク"
            color = "🟡"
        else:
            category = "低継続率"
            risk_level = "高リスク"
            color = "🔴"
        
        # 特徴量の重要度分析
        feature_analysis = self._analyze_developer_features(features)
        
        # 推奨アクションの生成
        recommendations = self._generate_recommendations(dev_data, features, retention_score)
        
        # 詳細情報の作成
        analysis = {
            'basic_info': {
                'developer_id': dev_id,
                'name': dev_data.get('name', 'Unknown'),
                'first_seen': dev_data.get('first_seen', ''),
                'last_activity': dev_data.get('last_activity', ''),
                'projects': dev_data.get('projects', []),
                'sources': dev_data.get('sources', [])
            },
            'prediction': {
                'retention_score': retention_score,
                'confidence': confidence,
                'category': category,
                'risk_level': risk_level,
                'color': color
            },
            'activity_stats': {
                'changes_authored': dev_data.get('changes_authored', 0),
                'changes_reviewed': dev_data.get('changes_reviewed', 0),
                'total_insertions': dev_data.get('total_insertions', 0),
                'total_deletions': dev_data.get('total_deletions', 0),
                'project_count': len(dev_data.get('projects', [])),
                'source_count': len(dev_data.get('sources', []))
            },
            'feature_analysis': feature_analysis,
            'recommendations': recommendations,
            'raw_features': features
        }
        
        return analysis
    
    def _analyze_developer_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """開発者の特徴量分析"""
        # 特徴量を重要度順にソート
        if self.improver.feature_importance:
            # アンサンブルの重み付き重要度を計算
            weighted_importance = {}
            total_weight = 0
            
            for model_name, importance in self.improver.feature_importance.items():
                if model_name in self.improver.ensemble_weights:
                    weight = self.improver.ensemble_weights[model_name]
                    for i, feature_name in enumerate(self.feature_names):
                        if feature_name not in weighted_importance:
                            weighted_importance[feature_name] = 0
                        weighted_importance[feature_name] += importance[i] * weight
                    total_weight += weight
            
            # 正規化
            if total_weight > 0:
                for feature_name in weighted_importance:
                    weighted_importance[feature_name] /= total_weight
            
            # 重要度順にソート
            sorted_features = sorted(
                [(name, features.get(name, 0), weighted_importance.get(name, 0)) 
                 for name in self.feature_names],
                key=lambda x: x[2], reverse=True
            )
        else:
            sorted_features = [(name, features.get(name, 0), 0) for name in self.feature_names]
        
        return {
            'top_features': sorted_features[:10],
            'all_features': sorted_features
        }
    
    def _generate_recommendations(self, dev_data: Dict[str, Any], 
                                features: Dict[str, float], 
                                retention_score: float) -> List[str]:
        """個別推奨アクションの生成"""
        recommendations = []
        
        # 継続率レベル別の基本推奨
        if retention_score >= 0.8:
            recommendations.append("🌟 優秀な開発者です。現在の活動を継続してください")
            recommendations.append("🎯 メンターやリーダー役への昇格を検討してください")
        elif retention_score >= 0.5:
            recommendations.append("⚠️ 継続率に注意が必要です。サポートを強化してください")
        else:
            recommendations.append("🚨 離脱リスクが高いです。緊急の介入が必要です")
        
        # 特徴量ベースの具体的推奨
        recent_activity_score = features.get('recent_activity_score', 0)
        if recent_activity_score < 0.3:
            recommendations.append("📅 最近の活動が少ないです。プロジェクトへの再参加を促してください")
        
        activity_frequency = features.get('activity_frequency', 0)
        if activity_frequency < 0.1:
            recommendations.append("📈 活動頻度が低いです。定期的なタスク割り当てを検討してください")
        
        project_diversity = features.get('project_diversity', 0)
        if project_diversity < 2:
            recommendations.append("🔄 プロジェクトの多様性を増やすことで関心を維持してください")
        
        avg_review_score = features.get('avg_review_score', 0)
        if avg_review_score < 0:
            recommendations.append("📝 レビュースコアが低いです。コードレビューのサポートを提供してください")
        
        contribution_balance = features.get('contribution_balance', 0)
        if contribution_balance < 0.3:
            recommendations.append("⚖️ 作成とレビューのバランスを改善してください")
        
        # 活動量ベースの推奨
        total_activity = dev_data.get('changes_authored', 0) + dev_data.get('changes_reviewed', 0)
        if total_activity < 10:
            recommendations.append("🚀 活動量を増やすため、小さなタスクから始めることを推奨します")
        elif total_activity > 100:
            recommendations.append("🏆 高い活動量を維持しています。他の開発者のメンターを検討してください")
        
        return recommendations
    
    def get_developer_by_id(self, dev_id: str) -> Optional[Dict[str, Any]]:
        """IDによる開発者情報の取得"""
        return self.predictions.get(dev_id)
    
    def get_developers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """カテゴリ別開発者リストの取得"""
        return [analysis for analysis in self.predictions.values() 
                if analysis['prediction']['category'] == category]
    
    def get_top_risk_developers(self, n: int = 10) -> List[Dict[str, Any]]:
        """高リスク開発者トップN"""
        sorted_devs = sorted(
            self.predictions.values(),
            key=lambda x: x['prediction']['retention_score']
        )
        return sorted_devs[:n]
    
    def get_top_performers(self, n: int = 10) -> List[Dict[str, Any]]:
        """トップパフォーマー開発者"""
        sorted_devs = sorted(
            self.predictions.values(),
            key=lambda x: x['prediction']['retention_score'],
            reverse=True
        )
        return sorted_devs[:n]
    
    def search_developers(self, query: str) -> List[Dict[str, Any]]:
        """開発者検索"""
        query = query.lower()
        results = []
        
        for analysis in self.predictions.values():
            basic_info = analysis['basic_info']
            if (query in basic_info['developer_id'].lower() or
                query in basic_info['name'].lower() or
                any(query in project.lower() for project in basic_info['projects'])):
                results.append(analysis)
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """サマリーレポートの生成"""
        if not self.predictions:
            return {}
        
        # カテゴリ別集計
        categories = {}
        risk_levels = {}
        total_devs = len(self.predictions)
        
        retention_scores = []
        confidence_scores = []
        
        for analysis in self.predictions.values():
            pred = analysis['prediction']
            category = pred['category']
            risk_level = pred['risk_level']
            
            categories[category] = categories.get(category, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            
            retention_scores.append(pred['retention_score'])
            confidence_scores.append(pred['confidence'])
        
        # 統計情報
        retention_stats = {
            'mean': np.mean(retention_scores),
            'std': np.std(retention_scores),
            'min': np.min(retention_scores),
            'max': np.max(retention_scores),
            'median': np.median(retention_scores)
        }
        
        confidence_stats = {
            'mean': np.mean(confidence_scores),
            'std': np.std(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores)
        }
        
        return {
            'total_developers': total_devs,
            'categories': categories,
            'risk_levels': risk_levels,
            'retention_stats': retention_stats,
            'confidence_stats': confidence_stats,
            'category_percentages': {k: (v/total_devs)*100 for k, v in categories.items()},
            'risk_percentages': {k: (v/total_devs)*100 for k, v in risk_levels.items()}
        }
    
    def display_developer_details(self, dev_id: str):
        """開発者詳細の表示"""
        analysis = self.get_developer_by_id(dev_id)
        if not analysis:
            print(f"❌ 開発者 {dev_id} が見つかりません")
            return
        
        basic = analysis['basic_info']
        pred = analysis['prediction']
        stats = analysis['activity_stats']
        
        print(f"\n{pred['color']} 開発者詳細分析: {basic['name']}")
        print("=" * 60)
        
        print(f"📧 ID: {basic['developer_id']}")
        print(f"👤 名前: {basic['name']}")
        print(f"📅 初回参加: {basic['first_seen']}")
        print(f"🕒 最終活動: {basic['last_activity']}")
        
        print(f"\n🎯 継続率予測:")
        print(f"   スコア: {pred['retention_score']:.4f} ({pred['retention_score']*100:.1f}%)")
        print(f"   信頼度: {pred['confidence']:.4f} ({pred['confidence']*100:.1f}%)")
        print(f"   カテゴリ: {pred['category']}")
        print(f"   リスクレベル: {pred['risk_level']}")
        
        print(f"\n📊 活動統計:")
        print(f"   作成したチェンジ: {stats['changes_authored']:,}")
        print(f"   レビューしたチェンジ: {stats['changes_reviewed']:,}")
        print(f"   総挿入行数: {stats['total_insertions']:,}")
        print(f"   総削除行数: {stats['total_deletions']:,}")
        print(f"   参加プロジェクト数: {stats['project_count']}")
        print(f"   参加ソース数: {stats['source_count']}")
        
        print(f"\n🔍 重要特徴量トップ5:")
        for i, (name, value, importance) in enumerate(analysis['feature_analysis']['top_features'][:5], 1):
            print(f"   {i}. {name}: {value:.4f} (重要度: {importance:.4f})")
        
        print(f"\n💡 推奨アクション:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🏢 参加プロジェクト:")
        for project in basic['projects'][:5]:  # 最初の5つを表示
            print(f"   • {project}")
        if len(basic['projects']) > 5:
            print(f"   ... 他 {len(basic['projects']) - 5} プロジェクト")

def main():
    """メイン実行関数"""
    print("🔍 個別開発者分析システムを開始します")
    print("=" * 60)
    
    # パスの設定
    models_path = "outputs/comprehensive_accuracy_improvement/improved_models_20250904_225449.pkl"
    data_path = "data/processed/unified/all_developers.json"
    
    # システムの初期化
    analyzer = IndividualDeveloperAnalyzer(models_path, data_path)
    
    try:
        # データとモデルの読み込み
        analyzer.load_data_and_models()
        
        # 全開発者の分析
        analyzer.analyze_all_developers()
        
        # サマリーレポートの生成
        summary = analyzer.generate_summary_report()
        
        print("\n📊 全体サマリー:")
        print("-" * 40)
        print(f"総開発者数: {summary['total_developers']:,}人")
        
        print(f"\n📈 継続率カテゴリ分布:")
        for category, count in summary['categories'].items():
            percentage = summary['category_percentages'][category]
            print(f"   {category}: {count:,}人 ({percentage:.1f}%)")
        
        print(f"\n⚠️ リスクレベル分布:")
        for risk, count in summary['risk_levels'].items():
            percentage = summary['risk_percentages'][risk]
            print(f"   {risk}: {count:,}人 ({percentage:.1f}%)")
        
        print(f"\n📊 継続率統計:")
        stats = summary['retention_stats']
        print(f"   平均: {stats['mean']:.4f}")
        print(f"   標準偏差: {stats['std']:.4f}")
        print(f"   最小値: {stats['min']:.4f}")
        print(f"   最大値: {stats['max']:.4f}")
        print(f"   中央値: {stats['median']:.4f}")
        
        # 高リスク開発者トップ5
        print(f"\n🚨 高リスク開発者トップ5:")
        print("-" * 40)
        top_risk = analyzer.get_top_risk_developers(5)
        for i, analysis in enumerate(top_risk, 1):
            basic = analysis['basic_info']
            pred = analysis['prediction']
            print(f"{i}. {basic['name']} ({basic['developer_id']})")
            print(f"   継続率: {pred['retention_score']:.4f} ({pred['category']})")
        
        # トップパフォーマー5
        print(f"\n🌟 トップパフォーマー5:")
        print("-" * 40)
        top_performers = analyzer.get_top_performers(5)
        for i, analysis in enumerate(top_performers, 1):
            basic = analysis['basic_info']
            pred = analysis['prediction']
            print(f"{i}. {basic['name']} ({basic['developer_id']})")
            print(f"   継続率: {pred['retention_score']:.4f} ({pred['category']})")
        
        # インタラクティブモード
        print(f"\n🔍 個別開発者詳細表示")
        print("=" * 60)
        print("開発者IDまたは名前を入力してください（'quit'で終了）:")
        
        while True:
            query = input("\n検索 > ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # 検索実行
            results = analyzer.search_developers(query)
            
            if not results:
                print(f"❌ '{query}' に一致する開発者が見つかりません")
                continue
            
            if len(results) == 1:
                # 1人の場合は詳細表示
                analyzer.display_developer_details(results[0]['basic_info']['developer_id'])
            else:
                # 複数の場合はリスト表示
                print(f"\n🔍 検索結果: {len(results)}人")
                print("-" * 40)
                for i, analysis in enumerate(results[:10], 1):  # 最初の10人
                    basic = analysis['basic_info']
                    pred = analysis['prediction']
                    print(f"{i}. {basic['name']} ({basic['developer_id']})")
                    print(f"   継続率: {pred['retention_score']:.4f} ({pred['category']})")
                
                if len(results) > 10:
                    print(f"... 他 {len(results) - 10} 人")
                
                # 詳細表示の選択
                try:
                    choice = input("\n詳細を見たい番号を入力 (Enter でスキップ): ").strip()
                    if choice and choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < min(len(results), 10):
                            analyzer.display_developer_details(results[idx]['basic_info']['developer_id'])
                except:
                    pass
        
        print("\n✅ 個別開発者分析システムを終了します")
        return analyzer
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = main()