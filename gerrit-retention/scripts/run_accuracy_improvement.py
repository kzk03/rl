#!/usr/bin/env python3
"""
予測精度継続改善システムの実行スクリプト

現在の217.7%改善を達成したシステムをベースに、
アンサンブル学習とハイパーパラメータ最適化による
さらなる精度向上を実現します。
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.gerrit_retention.prediction.advanced_accuracy_improver import (
    AdvancedAccuracyImprover,
)


def load_developer_data(data_path: str) -> List[Dict[str, Any]]:
    """開発者データの読み込み"""
    print(f"📊 開発者データを読み込み中: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ {len(data)}人の開発者データを読み込みました")
    return data

def create_retention_labels(developer_data: List[Dict[str, Any]]) -> np.ndarray:
    """継続率ラベルの作成"""
    print("🎯 継続率ラベルを作成中...")
    
    labels = []
    current_time = datetime.now()
    
    for dev in developer_data:
        try:
            # 最後の活動からの経過日数
            last_activity = datetime.fromisoformat(
                dev.get('last_activity', '').replace(' ', 'T')
            )
            days_since_last = (current_time - last_activity).days
            
            # 継続率スコアの計算（0-1の範囲）
            if days_since_last <= 7:
                retention_score = 1.0  # 高い継続率
            elif days_since_last <= 30:
                retention_score = 0.8  # 中程度の継続率
            elif days_since_last <= 90:
                retention_score = 0.4  # 低い継続率
            elif days_since_last <= 180:
                retention_score = 0.2  # 非常に低い継続率
            else:
                retention_score = 0.0  # 離脱状態
            
            # 活動量による調整
            total_activity = dev.get('changes_authored', 0) + dev.get('changes_reviewed', 0)
            if total_activity > 100:
                retention_score *= 1.2  # 高活動者はボーナス
            elif total_activity < 10:
                retention_score *= 0.8  # 低活動者はペナルティ
            
            # 0-1の範囲に正規化
            retention_score = min(1.0, max(0.0, retention_score))
            labels.append(retention_score)
            
        except (ValueError, TypeError):
            labels.append(0.0)  # エラーの場合は離脱とみなす
    
    labels = np.array(labels)
    print(f"✅ 継続率ラベルを作成完了")
    print(f"   平均継続率: {labels.mean():.3f}")
    print(f"   継続率分布: 高(>0.8): {(labels > 0.8).sum()}人, "
          f"中(0.4-0.8): {((labels >= 0.4) & (labels <= 0.8)).sum()}人, "
          f"低(<0.4): {(labels < 0.4).sum()}人")
    
    return labels

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """データの分割"""
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_accuracy_improvement():
    """予測精度改善の実行"""
    print("🚀 予測精度継続改善システムを開始します")
    print("=" * 80)
    
    # 設定
    config = {
        'data_path': 'data/processed/unified/all_developers.json',
        'output_path': 'outputs/accuracy_improvement',
        'test_size': 0.2,
        'random_state': 42
    }
    
    # 出力ディレクトリの作成
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. データの読み込み
        developer_data = load_developer_data(config['data_path'])
        
        # 2. システムの初期化
        improver = AdvancedAccuracyImprover(config)
        
        # 3. 特徴量の抽出
        print("\n🔧 高度な特徴量を抽出中...")
        features_list = []
        feature_names = None
        
        for i, dev in enumerate(developer_data):
            if i % 100 == 0:
                print(f"   進捗: {i}/{len(developer_data)} ({i/len(developer_data)*100:.1f}%)")
            
            features = improver.extract_advanced_features(dev)
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            features_list.append([features.get(name, 0.0) for name in feature_names])
        
        X = np.array(features_list)
        print(f"✅ 特徴量抽出完了: {X.shape[1]}次元の特徴量")
        print(f"   特徴量名: {', '.join(feature_names[:5])}...")
        
        # 4. ラベルの作成
        y = create_retention_labels(developer_data)
        
        # 5. データの分割
        print(f"\n📊 データを分割中 (テスト比率: {config['test_size']})")
        X_train, X_test, y_train, y_test = split_data(
            X, y, config['test_size'], config['random_state']
        )
        print(f"✅ 訓練データ: {X_train.shape[0]}サンプル, テストデータ: {X_test.shape[0]}サンプル")
        
        # 6. アンサンブルモデルの訓練
        print(f"\n🤖 アンサンブルモデルの訓練を開始...")
        print("-" * 60)
        
        trained_models = improver.train_ensemble_models(X_train, y_train)
        print(f"✅ {len(trained_models)}個のモデルを訓練完了")
        
        # 7. モデル性能の評価
        print(f"\n📈 モデル性能を評価中...")
        performance_results = improver.evaluate_model_performance(X_test, y_test)
        
        print("\n🏆 性能評価結果:")
        print("-" * 40)
        for model_name, metrics in performance_results.items():
            print(f"{model_name:15s}: R² = {metrics['r2']:.4f}, "
                  f"RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
        
        # 8. 特徴量重要度の分析
        print(f"\n🔍 特徴量重要度を分析中...")
        feature_analysis = improver.analyze_feature_importance(feature_names)
        
        if 'top_10' in feature_analysis:
            print("\n📊 重要特徴量トップ10:")
            print("-" * 50)
            for i, (name, importance) in enumerate(feature_analysis['top_10'], 1):
                print(f"{i:2d}. {name:30s}: {importance:.4f}")
        
        # 9. 改善提案の生成
        print(f"\n💡 改善提案を生成中...")
        recommendations = improver.generate_improvement_recommendations(
            performance_results, feature_analysis
        )
        
        print("\n🎯 改善提案:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # 10. 結果の保存
        print(f"\n💾 結果を保存中...")
        
        # 包括的な結果の作成
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(developer_data),
                'features_count': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'performance': performance_results,
            'feature_analysis': feature_analysis,
            'recommendations': recommendations,
            'model_weights': improver.ensemble_weights,
            'feature_names': feature_names
        }
        
        results_file, models_file = improver.save_improvement_results(
            comprehensive_results, config['output_path']
        )
        
        # 11. 最終サマリーの表示
        print("\n" + "=" * 80)
        print("🎉 予測精度改善システム実行完了！")
        print("=" * 80)
        
        ensemble_r2 = performance_results['ensemble']['r2']
        ensemble_rmse = performance_results['ensemble']['rmse']
        
        print(f"📊 最終性能:")
        print(f"   アンサンブルR²スコア: {ensemble_r2:.4f}")
        print(f"   RMSE: {ensemble_rmse:.4f}")
        
        if ensemble_r2 > 0.8:
            print("🏆 優秀な予測精度を達成しました！")
        elif ensemble_r2 > 0.6:
            print("✅ 良好な予測精度を達成しました")
        else:
            print("⚠️  予測精度の改善が必要です")
        
        print(f"\n📁 保存ファイル:")
        print(f"   結果: {results_file}")
        print(f"   モデル: {models_file}")
        
        # 前回の217.7%改善との比較
        print(f"\n🔄 前回の217.7%改善システムとの統合:")
        print(f"   現在のアンサンブル精度: {ensemble_r2:.1%}")
        print(f"   統合により更なる精度向上が期待されます")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_accuracy_improvement()
    
    if results:
        print("\n✅ 予測精度改善システムが正常に完了しました")
    else:
        print("\n❌ 予測精度改善システムでエラーが発生しました")
        sys.exit(1)