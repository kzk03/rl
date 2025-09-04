#!/usr/bin/env python3
"""
シンプルな継続予測デモ

実際に「どのくらい先で継続しているか」を予測するシステムのデモンストレーション
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.prediction.dynamic_threshold_calculator import (
    DynamicThresholdCalculator,
)


class SimpleContinuationPredictor:
    """シンプルな継続予測器"""
    
    def __init__(self):
        self.threshold_calculator = DynamicThresholdCalculator({
            'min_threshold_days': 7,
            'max_threshold_days': 365,
            'default_threshold_days': 90
        })
    
    def predict_continuation_probability(self, 
                                       developer: Dict[str, Any], 
                                       activity_history: List[Dict[str, Any]],
                                       future_days: int) -> Dict[str, Any]:
        """
        指定日数後の継続確率を予測
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            future_days: 予測したい日数後
            
        Returns:
            Dict[str, Any]: 予測結果
        """
        # 動的閾値を計算
        threshold_info = self.threshold_calculator.calculate_dynamic_threshold(
            developer, activity_history
        )
        
        # 活動パターンを分析
        activity_patterns = threshold_info['activity_patterns']
        
        # 基本的な継続確率計算
        base_probability = self._calculate_base_probability(
            activity_patterns, threshold_info, future_days
        )
        
        # 開発者タイプによる調整
        type_adjusted_probability = self._adjust_by_developer_type(
            base_probability, threshold_info['developer_type']
        )
        
        # 最終確率
        final_probability = max(0.0, min(1.0, type_adjusted_probability))
        
        # リスクレベルの判定
        risk_level = self._classify_risk_level(final_probability)
        
        return {
            'developer_id': developer.get('developer_id', 'unknown'),
            'prediction_days': future_days,
            'continuation_probability': final_probability,
            'risk_level': risk_level,
            'dynamic_threshold_days': threshold_info['threshold_days'],
            'developer_type': threshold_info['developer_type'],
            'confidence': threshold_info['confidence'],
            'reasoning': self._generate_reasoning(
                future_days, final_probability, threshold_info
            )
        }
    
    def _calculate_base_probability(self, 
                                  activity_patterns: Dict[str, Any], 
                                  threshold_info: Dict[str, Any],
                                  future_days: int) -> float:
        """基本継続確率を計算"""
        
        threshold_days = threshold_info['threshold_days']
        avg_gap = activity_patterns.get('avg_gap_days', 30)
        activity_frequency = activity_patterns.get('activity_frequency', 0.1)
        
        # 予測期間が閾値より短い場合は高確率
        if future_days <= threshold_days:
            # 活動頻度に基づく確率
            frequency_factor = min(activity_frequency * 10, 1.0)  # 正規化
            gap_factor = max(0.1, 1.0 - (avg_gap / threshold_days))
            base_prob = 0.7 + (frequency_factor * 0.2) + (gap_factor * 0.1)
        else:
            # 予測期間が閾値より長い場合は低確率
            excess_ratio = future_days / threshold_days
            decay_factor = 1.0 / (1.0 + excess_ratio * 0.5)  # 指数的減衰
            base_prob = 0.5 * decay_factor
        
        return base_prob
    
    def _adjust_by_developer_type(self, base_probability: float, dev_type: str) -> float:
        """開発者タイプによる調整"""
        
        type_multipliers = {
            'newcomer': 0.8,      # 新人は離脱リスクが高い
            'regular': 1.0,       # 通常開発者
            'veteran': 1.2,       # ベテランは継続しやすい
            'maintainer': 1.4,    # メンテナーは最も継続しやすい
            'occasional': 0.6,    # 時々参加は離脱しやすい
            'unknown': 1.0
        }
        
        multiplier = type_multipliers.get(dev_type, 1.0)
        return base_probability * multiplier
    
    def _classify_risk_level(self, probability: float) -> str:
        """リスクレベルを分類"""
        
        if probability >= 0.8:
            return 'low'        # 低リスク
        elif probability >= 0.6:
            return 'moderate'   # 中リスク
        elif probability >= 0.4:
            return 'high'       # 高リスク
        else:
            return 'critical'   # 危険
    
    def _generate_reasoning(self, 
                          future_days: int, 
                          probability: float, 
                          threshold_info: Dict[str, Any]) -> str:
        """予測の理由を生成"""
        
        threshold_days = threshold_info['threshold_days']
        dev_type = threshold_info['developer_type']
        
        reasoning_parts = []
        
        # 期間と閾値の関係
        if future_days <= threshold_days:
            reasoning_parts.append(f"{future_days}日後は動的閾値（{threshold_days}日）以内のため基本的に継続予想")
        else:
            reasoning_parts.append(f"{future_days}日後は動的閾値（{threshold_days}日）を超えるため継続確率が低下")
        
        # 開発者タイプの影響
        type_effects = {
            'veteran': '経験豊富なため継続確率が向上',
            'maintainer': 'メンテナー役割により継続確率が大幅向上',
            'newcomer': '新人のため継続確率がやや低下',
            'occasional': '時々参加のため継続確率が低下'
        }
        
        if dev_type in type_effects:
            reasoning_parts.append(type_effects[dev_type])
        
        # 最終判定
        reasoning_parts.append(f"最終継続確率: {probability:.1%}")
        
        return "。".join(reasoning_parts)


def main():
    """メイン関数"""
    
    print("🔮 継続予測システム デモンストレーション")
    print("=" * 60)
    
    # データを読み込み
    data_path = "data/extended_test_data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        developers_data = json.load(f)
    
    # 予測器を初期化
    predictor = SimpleContinuationPredictor()
    
    # 予測期間
    prediction_periods = [7, 30, 90, 180, 365]  # 1週間、1ヶ月、3ヶ月、6ヶ月、1年
    
    print(f"📊 分析対象: {len(developers_data)}人の開発者")
    print(f"🔍 予測期間: {prediction_periods} 日後")
    print()
    
    # 各開発者の予測を実行
    all_predictions = {}
    
    for dev_data in developers_data:
        developer = dev_data['developer']
        activity_history = dev_data['activity_history']
        developer_id = developer['developer_id']
        developer_name = developer['name']
        
        print(f"👤 {developer_name} ({developer_id})")
        print("-" * 50)
        
        dev_predictions = {}
        
        for days in prediction_periods:
            prediction = predictor.predict_continuation_probability(
                developer, activity_history, days
            )
            
            dev_predictions[f"{days}d"] = prediction
            
            prob = prediction['continuation_probability']
            risk = prediction['risk_level']
            
            # リスクレベルの色分け（テキスト）
            risk_symbols = {
                'low': '🟢',
                'moderate': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }
            
            symbol = risk_symbols.get(risk, '⚪')
            
            print(f"  {days:3d}日後: {prob:5.1%} {symbol} ({risk})")
        
        # 推論の表示（30日後の例）
        reasoning = dev_predictions['30d']['reasoning']
        print(f"  💭 30日後予測の根拠: {reasoning}")
        print()
        
        all_predictions[developer_id] = dev_predictions
    
    # サマリー統計
    print("📈 予測サマリー")
    print("=" * 60)
    
    for days in prediction_periods:
        probabilities = [
            pred[f"{days}d"]['continuation_probability'] 
            for pred in all_predictions.values()
        ]
        
        avg_prob = np.mean(probabilities)
        min_prob = min(probabilities)
        max_prob = max(probabilities)
        
        # リスクレベル分布
        risk_counts = {}
        for pred in all_predictions.values():
            risk = pred[f"{days}d"]['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print(f"{days:3d}日後予測:")
        print(f"  平均継続確率: {avg_prob:.1%}")
        print(f"  確率範囲: {min_prob:.1%} - {max_prob:.1%}")
        print(f"  リスク分布: {dict(risk_counts)}")
        print()
    
    # 結果を保存
    output_path = "outputs/comprehensive_retention/continuation_predictions.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'predictions': all_predictions,
            'summary': {
                'total_developers': len(developers_data),
                'prediction_periods': prediction_periods,
                'analysis_date': datetime.now().isoformat()
            }
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 結果を保存しました: {output_path}")
    print()
    print("🎉 継続予測デモ完了！")
    print("   各開発者の7日後〜1年後の継続確率を予測しました。")


if __name__ == "__main__":
    main()