#!/usr/bin/env python3
"""
強化された開発者分析例の表示

作業負荷、専門性、バーンアウトリスクを考慮した
詳細な開発者分析例を表示
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.enhanced_developer_analysis import EnhancedDeveloperAnalyzer


def show_enhanced_examples():
    """強化された開発者分析例の表示"""
    print("🚀 強化された開発者分析例を表示します")
    print("=" * 100)
    print("📋 分析内容:")
    print("   • 基本継続率 + 作業負荷・専門性・バーンアウト調整")
    print("   • ウェルネススコア（作業負荷・専門性・バーンアウト耐性）")
    print("   • 包括的推奨アクション")
    print("=" * 100)
    
    # データパスの設定
    data_path = "data/processed/unified/all_developers.json"
    
    # システムの初期化
    analyzer = EnhancedDeveloperAnalyzer(data_path)
    
    try:
        # データの読み込みと初期化
        analyzer.load_data_and_initialize()
        
        # 全開発者の強化分析
        analyzer.analyze_all_developers_enhanced()
        
        # 高継続率開発者の詳細表示
        print("\n🌟 高継続率開発者の強化分析")
        print("=" * 100)
        
        high_performers = [analysis for analysis in analyzer.enhanced_predictions.values()
                          if analysis['enhanced_prediction']['category'] == '高継続率']
        
        # ウェルネススコア順でソート
        sorted_high = sorted(high_performers, 
                           key=lambda x: x['wellness_scores']['overall_wellness'], 
                           reverse=True)
        
        for i, analysis in enumerate(sorted_high[:3], 1):
            print(f"\n【高継続率開発者 #{i}】")
            analyzer.display_enhanced_developer_details(analysis['basic_info']['developer_id'])
            print("-" * 80)
        
        # バーンアウトリスク開発者の詳細表示
        print("\n🔥 バーンアウトリスク開発者の強化分析")
        print("=" * 100)
        
        burnout_risk_devs = [analysis for analysis in analyzer.enhanced_predictions.values()
                           if analysis['burnout_risk']['burnout_level'] in ['high', 'critical']]
        
        if burnout_risk_devs:
            # バーンアウトリスク順でソート
            sorted_burnout = sorted(burnout_risk_devs, 
                                  key=lambda x: x['burnout_risk']['total_burnout_risk'], 
                                  reverse=True)
            
            for i, analysis in enumerate(sorted_burnout[:3], 1):
                print(f"\n【バーンアウトリスク開発者 #{i}】")
                analyzer.display_enhanced_developer_details(analysis['basic_info']['developer_id'])
                print("-" * 80)
        else:
            print("🎉 クリティカル・高バーンアウトリスク開発者は検出されませんでした")
        
        # 作業負荷過多開発者の分析
        print("\n⚖️ 作業負荷過多開発者の強化分析")
        print("=" * 100)
        
        high_workload_devs = [analysis for analysis in analyzer.enhanced_predictions.values()
                            if analysis['workload_analysis'].get('workload_stress', 0) > 0.5]
        
        if high_workload_devs:
            # 作業負荷ストレス順でソート
            sorted_workload = sorted(high_workload_devs, 
                                   key=lambda x: x['workload_analysis']['workload_stress'], 
                                   reverse=True)
            
            print(f"作業負荷過多開発者: {len(high_workload_devs)}人")
            for i, analysis in enumerate(sorted_workload[:3], 1):
                print(f"\n【作業負荷過多開発者 #{i}】")
                analyzer.display_enhanced_developer_details(analysis['basic_info']['developer_id'])
                print("-" * 80)
        else:
            print("✅ 作業負荷過多の開発者は検出されませんでした")
        
        # 専門性ミスマッチ開発者の分析
        print("\n🎯 専門性ミスマッチ開発者の強化分析")
        print("=" * 100)
        
        mismatch_devs = [analysis for analysis in analyzer.enhanced_predictions.values()
                        if analysis['expertise_analysis'].get('expertise_match_score', 0) < 0.4]
        
        if mismatch_devs:
            # 専門性ミスマッチ順でソート
            sorted_mismatch = sorted(mismatch_devs, 
                                   key=lambda x: x['expertise_analysis']['expertise_match_score'])
            
            print(f"専門性ミスマッチ開発者: {len(mismatch_devs)}人")
            for i, analysis in enumerate(sorted_mismatch[:3], 1):
                print(f"\n【専門性ミスマッチ開発者 #{i}】")
                analyzer.display_enhanced_developer_details(analysis['basic_info']['developer_id'])
                print("-" * 80)
        else:
            print("✅ 深刻な専門性ミスマッチ開発者は検出されませんでした")
        
        # 最適バランス開発者の分析
        print("\n⚖️ 最適バランス開発者の強化分析")
        print("=" * 100)
        
        balanced_devs = [analysis for analysis in analyzer.enhanced_predictions.values()
                        if (analysis['wellness_scores']['overall_wellness'] > 0.8 and
                            analysis['workload_analysis'].get('workload_stress', 0) < 0.3 and
                            analysis['expertise_analysis'].get('expertise_match_score', 0) > 0.6)]
        
        if balanced_devs:
            # 総合ウェルネス順でソート
            sorted_balanced = sorted(balanced_devs, 
                                   key=lambda x: x['wellness_scores']['overall_wellness'], 
                                   reverse=True)
            
            print(f"最適バランス開発者: {len(balanced_devs)}人")
            for i, analysis in enumerate(sorted_balanced[:3], 1):
                print(f"\n【最適バランス開発者 #{i}】")
                analyzer.display_enhanced_developer_details(analysis['basic_info']['developer_id'])
                print("-" * 80)
        else:
            print("⚠️ 最適バランス開発者は少数です")
        
        # 統計サマリー
        print(f"\n📊 強化分析統計サマリー")
        print("=" * 100)
        
        summary = analyzer.get_enhanced_summary_report()
        
        print(f"📈 継続率改善効果:")
        base_scores = []
        adjusted_scores = []
        
        for analysis in analyzer.enhanced_predictions.values():
            pred = analysis['enhanced_prediction']
            base_scores.append(pred['base_retention_score'])
            adjusted_scores.append(pred['adjusted_retention_score'])
        
        import numpy as np
        base_mean = np.mean(base_scores)
        adjusted_mean = np.mean(adjusted_scores)
        improvement = ((adjusted_mean - base_mean) / base_mean) * 100
        
        print(f"   基本継続率平均: {base_mean:.4f}")
        print(f"   調整後継続率平均: {adjusted_mean:.4f}")
        print(f"   改善効果: {improvement:+.2f}%")
        
        print(f"\n🔧 作業負荷分析結果:")
        workload_scores = [analysis['wellness_scores']['workload_score'] 
                          for analysis in analyzer.enhanced_predictions.values()]
        print(f"   平均作業負荷スコア: {np.mean(workload_scores):.4f}")
        print(f"   作業負荷過多開発者: {len(high_workload_devs)}人 ({len(high_workload_devs)/len(analyzer.enhanced_predictions)*100:.1f}%)")
        
        print(f"\n🎯 専門性分析結果:")
        expertise_scores = [analysis['wellness_scores']['expertise_score'] 
                           for analysis in analyzer.enhanced_predictions.values()]
        print(f"   平均専門性スコア: {np.mean(expertise_scores):.4f}")
        print(f"   専門性ミスマッチ開発者: {len(mismatch_devs)}人 ({len(mismatch_devs)/len(analyzer.enhanced_predictions)*100:.1f}%)")
        
        print(f"\n🔥 バーンアウトリスク分析結果:")
        burnout_scores = [analysis['wellness_scores']['burnout_score'] 
                         for analysis in analyzer.enhanced_predictions.values()]
        print(f"   平均バーンアウト耐性: {np.mean(burnout_scores):.4f}")
        print(f"   高リスク開発者: {len(burnout_risk_devs)}人 ({len(burnout_risk_devs)/len(analyzer.enhanced_predictions)*100:.1f}%)")
        
        print(f"\n⚖️ 総合ウェルネス結果:")
        wellness_scores = [analysis['wellness_scores']['overall_wellness'] 
                          for analysis in analyzer.enhanced_predictions.values()]
        print(f"   平均総合ウェルネス: {np.mean(wellness_scores):.4f}")
        print(f"   最適バランス開発者: {len(balanced_devs)}人 ({len(balanced_devs)/len(analyzer.enhanced_predictions)*100:.1f}%)")
        
        print(f"\n✅ 強化された開発者分析例の表示完了")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_enhanced_examples()