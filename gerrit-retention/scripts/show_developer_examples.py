#!/usr/bin/env python3
"""
開発者分析例の表示

トップパフォーマーと高リスク開発者の詳細例を表示
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.individual_developer_analysis import IndividualDeveloperAnalyzer


def show_examples():
    """開発者分析例の表示"""
    print("🔍 開発者分析例を表示します")
    print("=" * 80)
    
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
        
        # トップパフォーマーの詳細表示
        print("\n🌟 トップパフォーマー詳細分析")
        print("=" * 80)
        
        top_performers = analyzer.get_top_performers(3)
        for i, analysis in enumerate(top_performers, 1):
            print(f"\n【トップパフォーマー #{i}】")
            analyzer.display_developer_details(analysis['basic_info']['developer_id'])
            print("-" * 60)
        
        # 高リスク開発者の詳細表示
        print("\n🚨 高リスク開発者詳細分析")
        print("=" * 80)
        
        high_risk = analyzer.get_top_risk_developers(3)
        for i, analysis in enumerate(high_risk, 1):
            print(f"\n【高リスク開発者 #{i}】")
            analyzer.display_developer_details(analysis['basic_info']['developer_id'])
            print("-" * 60)
        
        # 中継続率開発者の例
        print("\n⚖️ 中継続率開発者詳細分析")
        print("=" * 80)
        
        medium_devs = analyzer.get_developers_by_category("中継続率")
        if medium_devs:
            # 中継続率の中でも上位3人を表示
            sorted_medium = sorted(medium_devs, 
                                 key=lambda x: x['prediction']['retention_score'], 
                                 reverse=True)
            
            for i, analysis in enumerate(sorted_medium[:3], 1):
                print(f"\n【中継続率開発者 #{i}】")
                analyzer.display_developer_details(analysis['basic_info']['developer_id'])
                print("-" * 60)
        
        # 特徴量分析のサマリー
        print("\n📊 特徴量重要度分析")
        print("=" * 80)
        
        if analyzer.improver.feature_importance:
            # アンサンブルの重み付き重要度を計算
            weighted_importance = {}
            total_weight = 0
            
            for model_name, importance in analyzer.improver.feature_importance.items():
                if model_name in analyzer.improver.ensemble_weights:
                    weight = analyzer.improver.ensemble_weights[model_name]
                    for i, feature_name in enumerate(analyzer.feature_names):
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
                weighted_importance.items(),
                key=lambda x: x[1], reverse=True
            )
            
            print("🔍 最重要特徴量トップ15:")
            for i, (feature_name, importance) in enumerate(sorted_features[:15], 1):
                print(f"{i:2d}. {feature_name:30s}: {importance:.6f}")
        
        # カテゴリ別統計
        print(f"\n📈 カテゴリ別詳細統計")
        print("=" * 80)
        
        categories = ["高継続率", "中継続率", "低継続率"]
        for category in categories:
            devs = analyzer.get_developers_by_category(category)
            if devs:
                scores = [d['prediction']['retention_score'] for d in devs]
                confidences = [d['prediction']['confidence'] for d in devs]
                
                print(f"\n{category} ({len(devs)}人):")
                print(f"   継続率スコア: 平均={sum(scores)/len(scores):.4f}, "
                      f"最小={min(scores):.4f}, 最大={max(scores):.4f}")
                print(f"   予測信頼度: 平均={sum(confidences)/len(confidences):.4f}, "
                      f"最小={min(confidences):.4f}, 最大={max(confidences):.4f}")
                
                # 活動統計
                authored = [d['activity_stats']['changes_authored'] for d in devs]
                reviewed = [d['activity_stats']['changes_reviewed'] for d in devs]
                projects = [d['activity_stats']['project_count'] for d in devs]
                
                print(f"   作成チェンジ: 平均={sum(authored)/len(authored):.1f}, "
                      f"最大={max(authored)}")
                print(f"   レビューチェンジ: 平均={sum(reviewed)/len(reviewed):.1f}, "
                      f"最大={max(reviewed)}")
                print(f"   参加プロジェクト: 平均={sum(projects)/len(projects):.1f}, "
                      f"最大={max(projects)}")
        
        print(f"\n✅ 開発者分析例の表示完了")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_examples()