#!/usr/bin/env python3
"""
実際のGoogle開発者データで予測テスト
"""

import json
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from retention_training.train_retention_model import SimpleRetentionModel
from stress_training.train_stress_model import SimpleStressModel


def test_real_predictions():
    """実際のデータで予測テスト"""
    
    # 実際のデータを読み込み
    with open('data/processed/unified/all_developers.json', 'r') as f:
        developers = json.load(f)
    
    with open('data/processed/unified/all_features.json', 'r') as f:
        features = json.load(f)
    
    print("=== 実際のGoogle開発者データで予測テスト ===")
    
    # 興味深い開発者を選択
    target_developers = [
        "rbraunstein@google.com",  # Ronald Braunstein - 高活動レビュアー
        "jji@google.com",          # Jing Ji - 作成者+レビュアー
        "vinhdaitran@google.com"   # Vinh Tran - 新規開発者
    ]
    
    for dev_id in target_developers:
        # 開発者データを検索
        dev_data = next((d for d in developers if d['developer_id'] == dev_id), None)
        dev_features = next((f for f in features if f['developer_id'] == dev_id), None)
        
        if not dev_data or not dev_features:
            print(f"❌ 開発者 {dev_id} のデータが見つかりません")
            continue
        
        print(f"\n👤 開発者: {dev_data['name']} ({dev_id})")
        print(f"   活動: 作成{dev_data['changes_authored']}件, レビュー{dev_data['changes_reviewed']}件")
        print(f"   プロジェクト: {dev_data['projects']}")
        
        # 特徴量表示
        print(f"\n📊 特徴量:")
        print(f"   専門性レベル: {dev_features['expertise_level']:.3f}")
        print(f"   協力品質: {dev_features['collaboration_quality']:.3f}")
        print(f"   満足度: {dev_features['satisfaction_level']:.3f}")
        print(f"   ストレス蓄積: {dev_features['stress_accumulation']:.3f}")
        
        # 予測実行
        retention_model = SimpleRetentionModel()
        stress_model = SimpleStressModel()
        
        # 定着予測
        retention_prob = retention_model.predict(dev_features)
        
        # ストレス分析（簡易版）
        stress_level = dev_features['stress_accumulation']
        burnout_risk = dev_features.get('burnout_risk', 0.0)
        
        stress_analysis = {
            'total_stress': max(stress_level, burnout_risk),
            'risk_level': 'HIGH' if stress_level > 0.7 else 'MEDIUM' if stress_level > 0.4 else 'LOW'
        }
        
        print(f"\n🔮 予測結果:")
        print(f"   定着確率: {retention_prob:.1%}")
        print(f"   ストレスレベル: {stress_analysis['total_stress']:.2f}/1.0")
        print(f"   リスク評価: {stress_analysis['risk_level']}")
        
        # リスク評価とアクション提案
        if retention_prob < 0.6:
            risk_level = "🔴 高リスク"
            actions = [
                "緊急1on1ミーティング設定",
                "ワークロード即座調整", 
                "ストレス要因の特定と除去",
                "短期休暇の提案"
            ]
        elif retention_prob < 0.8:
            risk_level = "🟡 中リスク"
            actions = [
                "定期チェックイン強化",
                "スキル開発機会の提供",
                "プロジェクト配置の見直し",
                "メンタリング支援"
            ]
        else:
            risk_level = "🟢 低リスク"
            actions = [
                "現状維持",
                "新しい挑戦機会の提供",
                "他開発者のメンタリング役割",
                "技術リーダーシップ機会"
            ]
        
        print(f"\n{risk_level}")
        print("   推奨アクション:")
        for action in actions:
            print(f"   • {action}")
        
        # 実際のデータとの比較
        actual_retention = dev_features['retention_prediction']
        prediction_diff = abs(retention_prob - actual_retention)
        print(f"\n📈 予測精度:")
        print(f"   実際の定着予測: {actual_retention:.1%}")
        print(f"   モデル予測: {retention_prob:.1%}")
        print(f"   差異: {prediction_diff:.1%}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_real_predictions()