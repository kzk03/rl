#!/usr/bin/env python3
"""
予測精度分析システム

継続予測の確率と実際の離脱結果を比較分析し、
予測システムの精度を評価する。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.prediction.dynamic_threshold_calculator import (
    DynamicThresholdCalculator,
)


class PredictionAccuracyAnalyzer:
    """予測精度分析器"""
    
    def __init__(self):
        self.threshold_calculator = DynamicThresholdCalculator({
            'min_threshold_days': 7,
            'max_threshold_days': 365,
            'default_threshold_days': 90
        })
    
    def create_test_scenario_with_outcomes(self) -> List[Dict[str, Any]]:
        """
        予測と実際の結果を含むテストシナリオを作成
        
        Returns:
            List[Dict[str, Any]]: テストデータ（予測時点と結果時点を含む）
        """
        
        # 実際のプロジェクトでは過去のデータを使用するが、
        # ここではシミュレーションデータを作成
        
        scenarios = [
            {
                'developer': {
                    'developer_id': 'alice@example.com',
                    'name': 'Alice Active',
                    'first_seen': '2023-01-15T09:00:00Z',
                    'changes_authored': 45,
                    'changes_reviewed': 67,
                    'projects': ['project-a', 'project-b']
                },
                'prediction_date': '2023-03-20T00:00:00Z',
                'activity_history_at_prediction': [
                    {'timestamp': '2023-01-20T10:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-02-05T14:15:00Z', 'type': 'review'},
                    {'timestamp': '2023-02-18T11:45:00Z', 'type': 'commit'},
                    {'timestamp': '2023-03-02T16:20:00Z', 'type': 'merge'},
                    {'timestamp': '2023-03-15T09:10:00Z', 'type': 'review'}
                ],
                'actual_future_activities': [
                    {'timestamp': '2023-04-01T12:00:00Z', 'type': 'commit'},
                    {'timestamp': '2023-04-20T15:30:00Z', 'type': 'review'},
                    {'timestamp': '2023-05-10T11:15:00Z', 'type': 'commit'},
                    {'timestamp': '2023-06-05T14:30:00Z', 'type': 'review'}
                ]
            },
            {
                'developer': {
                    'developer_id': 'bob@example.com',
                    'name': 'Bob Declining',
                    'first_seen': '2022-11-10T14:30:00Z',
                    'changes_authored': 123,
                    'changes_reviewed': 89,
                    'projects': ['project-a', 'project-c']
                },
                'prediction_date': '2023-03-20T00:00:00Z',
                'activity_history_at_prediction': [
                    {'timestamp': '2023-01-05T08:45:00Z', 'type': 'commit'},
                    {'timestamp': '2023-01-12T13:20:00Z', 'type': 'review'},
                    {'timestamp': '2023-02-08T15:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-02-20T11:00:00Z', 'type': 'documentation'},
                    {'timestamp': '2023-03-05T14:45:00Z', 'type': 'review'}
                ],
                'actual_future_activities': [
                    {'timestamp': '2023-04-15T09:30:00Z', 'type': 'commit'}
                    # その後活動なし（離脱）
                ]
            },
            {
                'developer': {
                    'developer_id': 'charlie@example.com',
                    'name': 'Charlie Newbie',
                    'first_seen': '2023-02-28T16:00:00Z',
                    'changes_authored': 8,
                    'changes_reviewed': 12,
                    'projects': ['project-b']
                },
                'prediction_date': '2023-03-20T00:00:00Z',
                'activity_history_at_prediction': [
                    {'timestamp': '2023-03-01T09:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-03-03T11:15:00Z', 'type': 'review'},
                    {'timestamp': '2023-03-08T14:20:00Z', 'type': 'commit'},
                    {'timestamp': '2023-03-12T10:45:00Z', 'type': 'issue_comment'}
                ],
                'actual_future_activities': []  # 完全に離脱
            },
            {
                'developer': {
                    'developer_id': 'diana@example.com',
                    'name': 'Diana Veteran',
                    'first_seen': '2021-06-15T10:00:00Z',
                    'changes_authored': 456,
                    'changes_reviewed': 234,
                    'projects': ['project-a', 'project-b', 'project-c', 'project-d']
                },
                'prediction_date': '2023-03-20T00:00:00Z',
                'activity_history_at_prediction': [
                    {'timestamp': '2023-01-03T08:00:00Z', 'type': 'commit'},
                    {'timestamp': '2023-01-10T09:30:00Z', 'type': 'review'},
                    {'timestamp': '2023-01-25T11:20:00Z', 'type': 'documentation'},
                    {'timestamp': '2023-02-15T10:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-03-01T14:00:00Z', 'type': 'review'},
                    {'timestamp': '2023-03-10T16:45:00Z', 'type': 'collaboration'}
                ],
                'actual_future_activities': [
                    {'timestamp': '2023-04-05T12:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-04-25T14:15:00Z', 'type': 'review'},
                    {'timestamp': '2023-05-15T10:00:00Z', 'type': 'commit'},
                    {'timestamp': '2023-06-10T13:45:00Z', 'type': 'review'},
                    {'timestamp': '2023-07-02T11:20:00Z', 'type': 'commit'},
                    {'timestamp': '2023-08-18T15:30:00Z', 'type': 'review'}
                ]
            },
            {
                'developer': {
                    'developer_id': 'eve@example.com',
                    'name': 'Eve Occasional',
                    'first_seen': '2022-08-20T12:00:00Z',
                    'changes_authored': 23,
                    'changes_reviewed': 15,
                    'projects': ['project-c']
                },
                'prediction_date': '2023-03-20T00:00:00Z',
                'activity_history_at_prediction': [
                    {'timestamp': '2022-12-15T14:30:00Z', 'type': 'commit'},
                    {'timestamp': '2023-01-20T16:45:00Z', 'type': 'review'},
                    {'timestamp': '2023-02-28T11:30:00Z', 'type': 'commit'}
                ],
                'actual_future_activities': [
                    {'timestamp': '2023-05-10T14:20:00Z', 'type': 'commit'},
                    {'timestamp': '2023-08-15T10:30:00Z', 'type': 'review'}
                ]
            }
        ]
        
        return scenarios
    
    def predict_continuation_probability(self, 
                                       developer: Dict[str, Any], 
                                       activity_history: List[Dict[str, Any]],
                                       future_days: int,
                                       prediction_date: datetime) -> float:
        """継続確率を予測（簡略版）"""
        
        # 動的閾値を計算
        threshold_info = self.threshold_calculator.calculate_dynamic_threshold(
            developer, activity_history, prediction_date
        )
        
        activity_patterns = threshold_info['activity_patterns']
        threshold_days = threshold_info['threshold_days']
        dev_type = threshold_info['developer_type']
        
        # 基本確率計算
        if future_days <= threshold_days:
            activity_frequency = activity_patterns.get('activity_frequency', 0.1)
            base_prob = 0.7 + min(activity_frequency * 10, 0.2)
        else:
            excess_ratio = future_days / threshold_days
            base_prob = 0.5 / (1.0 + excess_ratio * 0.5)
        
        # 開発者タイプ調整
        type_multipliers = {
            'newcomer': 0.8, 'regular': 1.0, 'veteran': 1.2, 
            'maintainer': 1.4, 'occasional': 0.6, 'unknown': 1.0
        }
        
        adjusted_prob = base_prob * type_multipliers.get(dev_type, 1.0)
        return max(0.0, min(1.0, adjusted_prob))
    
    def determine_actual_continuation(self, 
                                    future_activities: List[Dict[str, Any]], 
                                    prediction_date: datetime,
                                    target_days: int) -> bool:
        """実際の継続状況を判定"""
        
        target_date = prediction_date + timedelta(days=target_days)
        
        # target_date以降に活動があるかチェック
        for activity in future_activities:
            try:
                activity_date_str = activity['timestamp']
                if 'T' in activity_date_str:
                    activity_date = datetime.fromisoformat(activity_date_str.replace('Z', '+00:00'))
                else:
                    activity_date = datetime.strptime(activity_date_str, '%Y-%m-%d')
                
                if activity_date >= target_date:
                    return True
            except:
                continue
        
        return False
    
    def analyze_prediction_accuracy(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """予測精度を分析"""
        
        print("🔍 予測精度分析を開始...")
        print("=" * 60)
        
        prediction_periods = [7, 30, 90, 180]
        results = {}
        
        for period_days in prediction_periods:
            print(f"\n📊 {period_days}日後予測の分析")
            print("-" * 40)
            
            predictions = []
            actuals = []
            probabilities = []
            details = []
            
            for scenario in scenarios:
                developer = scenario['developer']
                prediction_date = datetime.fromisoformat(
                    scenario['prediction_date'].replace('Z', '+00:00')
                )
                activity_history = scenario['activity_history_at_prediction']
                future_activities = scenario['actual_future_activities']
                
                # 予測実行
                predicted_prob = self.predict_continuation_probability(
                    developer, activity_history, period_days, prediction_date
                )
                
                # 実際の結果判定
                actual_continuation = self.determine_actual_continuation(
                    future_activities, prediction_date, period_days
                )
                
                # 予測ラベル（50%を閾値とする）
                predicted_continuation = predicted_prob > 0.5
                
                predictions.append(predicted_continuation)
                actuals.append(actual_continuation)
                probabilities.append(predicted_prob)
                
                details.append({
                    'developer_id': developer['developer_id'],
                    'developer_name': developer['name'],
                    'predicted_probability': predicted_prob,
                    'predicted_continuation': predicted_continuation,
                    'actual_continuation': actual_continuation,
                    'correct_prediction': predicted_continuation == actual_continuation
                })
                
                # 個別結果表示
                status = "✅" if predicted_continuation == actual_continuation else "❌"
                actual_str = "継続" if actual_continuation else "離脱"
                pred_str = "継続" if predicted_continuation else "離脱"
                
                print(f"  {status} {developer['name'][:15]:15s}: "
                      f"予測{predicted_prob:5.1%}({pred_str}) vs 実際({actual_str})")
            
            # メトリクス計算
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, zero_division=0)
            recall = recall_score(actuals, predictions, zero_division=0)
            f1 = f1_score(actuals, predictions, zero_division=0)
            
            # AUC計算（両クラスが存在する場合のみ）
            auc = None
            if len(set(actuals)) > 1:
                auc = roc_auc_score(actuals, probabilities)
            
            # 混同行列
            cm = confusion_matrix(actuals, predictions)
            
            period_results = {
                'period_days': period_days,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'confusion_matrix': cm.tolist(),
                'details': details,
                'total_samples': len(scenarios),
                'correct_predictions': sum(d['correct_prediction'] for d in details)
            }
            
            results[f'{period_days}d'] = period_results
            
            # 結果サマリー表示
            print(f"\n📈 {period_days}日後予測の精度:")
            print(f"  正解率: {accuracy:.1%} ({period_results['correct_predictions']}/{len(scenarios)})")
            print(f"  適合率: {precision:.1%}")
            print(f"  再現率: {recall:.1%}")
            print(f"  F1スコア: {f1:.1%}")
            if auc:
                print(f"  AUC-ROC: {auc:.3f}")
            
            # 混同行列表示
            print(f"  混同行列:")
            print(f"    実際継続 | 実際離脱")
            if cm.shape == (2, 2):
                print(f"  予測継続 |    {cm[1,1]:2d}    |    {cm[0,1]:2d}")
                print(f"  予測離脱 |    {cm[1,0]:2d}    |    {cm[0,0]:2d}")
            else:
                print(f"  単一クラスのため混同行列は {cm.shape} 形状")
        
        return results
    
    def generate_calibration_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """予測確率の校正分析"""
        
        print("\n🎯 予測確率の校正分析")
        print("=" * 60)
        
        calibration_results = {}
        
        for period_key, period_data in results.items():
            period_days = period_data['period_days']
            details = period_data['details']
            
            # 確率区間別の分析
            prob_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            bin_analysis = []
            
            print(f"\n📊 {period_days}日後予測の校正:")
            print("確率区間 | 予測数 | 実際継続率 | 期待継続率 | 校正誤差")
            print("-" * 55)
            
            for bin_start, bin_end in prob_bins:
                bin_details = [
                    d for d in details 
                    if bin_start <= d['predicted_probability'] < bin_end
                ]
                
                if not bin_details:
                    continue
                
                predicted_probs = [d['predicted_probability'] for d in bin_details]
                actual_outcomes = [d['actual_continuation'] for d in bin_details]
                
                count = len(bin_details)
                actual_rate = np.mean(actual_outcomes)
                expected_rate = np.mean(predicted_probs)
                calibration_error = abs(actual_rate - expected_rate)
                
                bin_analysis.append({
                    'bin_range': f"{bin_start:.1f}-{bin_end:.1f}",
                    'count': count,
                    'actual_rate': actual_rate,
                    'expected_rate': expected_rate,
                    'calibration_error': calibration_error
                })
                
                print(f"{bin_start:.1f}-{bin_end:.1f}  |   {count:2d}   |   {actual_rate:5.1%}   |   {expected_rate:5.1%}   |   {calibration_error:5.1%}")
            
            # 全体の校正誤差
            all_probs = [d['predicted_probability'] for d in details]
            all_actuals = [d['actual_continuation'] for d in details]
            overall_calibration_error = abs(np.mean(all_actuals) - np.mean(all_probs))
            
            calibration_results[period_key] = {
                'bin_analysis': bin_analysis,
                'overall_calibration_error': overall_calibration_error
            }
            
            print(f"全体校正誤差: {overall_calibration_error:.1%}")
        
        return calibration_results
    
    def generate_insights_and_recommendations(self, 
                                            results: Dict[str, Any], 
                                            calibration: Dict[str, Any]) -> Dict[str, Any]:
        """洞察と推奨事項を生成"""
        
        print("\n💡 分析結果の洞察と推奨事項")
        print("=" * 60)
        
        insights = {
            'accuracy_trends': {},
            'calibration_quality': {},
            'recommendations': []
        }
        
        # 精度トレンド分析
        periods = sorted([int(k.replace('d', '')) for k in results.keys()])
        accuracies = [results[f'{p}d']['accuracy'] for p in periods]
        
        print("\n📈 精度トレンド:")
        for i, period in enumerate(periods):
            accuracy = accuracies[i]
            trend = ""
            if i > 0:
                if accuracy > accuracies[i-1]:
                    trend = " ⬆️"
                elif accuracy < accuracies[i-1]:
                    trend = " ⬇️"
                else:
                    trend = " ➡️"
            
            print(f"  {period:3d}日後: {accuracy:5.1%}{trend}")
        
        insights['accuracy_trends'] = {
            'periods': periods,
            'accuracies': accuracies,
            'best_period': periods[np.argmax(accuracies)],
            'worst_period': periods[np.argmin(accuracies)]
        }
        
        # 校正品質評価
        print("\n🎯 校正品質:")
        for period_key, calib_data in calibration.items():
            period_days = results[period_key]['period_days']
            error = calib_data['overall_calibration_error']
            
            quality = "優秀" if error < 0.1 else "良好" if error < 0.2 else "要改善"
            print(f"  {period_days:3d}日後: {error:5.1%} ({quality})")
        
        # 推奨事項生成
        recommendations = []
        
        # 精度に基づく推奨
        best_accuracy = max(accuracies)
        if best_accuracy < 0.7:
            recommendations.append("全体的な予測精度が70%未満のため、特徴量エンジニアリングの改善が必要")
        
        # 校正に基づく推奨
        high_calibration_errors = [
            (period_key, calib_data['overall_calibration_error'])
            for period_key, calib_data in calibration.items()
            if calib_data['overall_calibration_error'] > 0.15
        ]
        
        if high_calibration_errors:
            recommendations.append("一部期間で校正誤差が15%を超えているため、確率校正の改善が必要")
        
        # 期間別推奨
        if insights['accuracy_trends']['best_period'] <= 30:
            recommendations.append("短期予測の精度が高いため、短期アラートシステムの導入を推奨")
        
        if insights['accuracy_trends']['worst_period'] >= 180:
            recommendations.append("長期予測の精度が低いため、長期予測は参考程度に留めることを推奨")
        
        insights['recommendations'] = recommendations
        
        print("\n🔧 推奨事項:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return insights
    
    def save_analysis_results(self, 
                            results: Dict[str, Any], 
                            calibration: Dict[str, Any], 
                            insights: Dict[str, Any]) -> None:
        """分析結果を保存"""
        
        output_data = {
            'analysis_date': datetime.now().isoformat(),
            'prediction_accuracy': results,
            'calibration_analysis': calibration,
            'insights_and_recommendations': insights,
            'summary': {
                'total_scenarios': len(results['7d']['details']),
                'prediction_periods': [results[k]['period_days'] for k in results.keys()],
                'overall_best_accuracy': max(results[k]['accuracy'] for k in results.keys()),
                'overall_worst_accuracy': min(results[k]['accuracy'] for k in results.keys())
            }
        }
        
        output_path = "outputs/comprehensive_retention/prediction_accuracy_analysis.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 分析結果を保存しました: {output_path}")


def main():
    """メイン関数"""
    
    print("🔍 継続予測精度分析システム")
    print("=" * 60)
    print("予測確率と実際の離脱結果を比較分析します")
    print()
    
    # 分析器を初期化
    analyzer = PredictionAccuracyAnalyzer()
    
    # テストシナリオを作成
    scenarios = analyzer.create_test_scenario_with_outcomes()
    print(f"📊 分析対象: {len(scenarios)}人の開発者")
    print("🎯 予測期間: 7日、30日、90日、180日後")
    
    # 予測精度分析
    accuracy_results = analyzer.analyze_prediction_accuracy(scenarios)
    
    # 校正分析
    calibration_results = analyzer.generate_calibration_analysis(accuracy_results)
    
    # 洞察と推奨事項
    insights = analyzer.generate_insights_and_recommendations(
        accuracy_results, calibration_results
    )
    
    # 結果保存
    analyzer.save_analysis_results(accuracy_results, calibration_results, insights)
    
    print("\n🎉 予測精度分析完了！")
    print("   予測システムの性能が定量的に評価されました。")


if __name__ == "__main__":
    main()