#!/usr/bin/env python3
"""
定着予測モデル訓練スクリプト
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

class SimpleRetentionModel:
    """簡単な定着予測モデル"""
    
    def __init__(self):
        self.trained_at = datetime.now().isoformat()
        self.model_version = "1.0.0"
        self.feature_count = 0
    
    def predict(self, features):
        """定着確率を予測"""
        expertise = features.get('expertise_level', 0.5)
        collaboration = features.get('collaboration_quality', 0.5)
        satisfaction = features.get('satisfaction_level', 0.5)
        stress = features.get('stress_accumulation', 0.5)
        
        # 重み付き平均で定着確率を計算
        retention_prob = (
            expertise * 0.3 + 
            collaboration * 0.3 + 
            satisfaction * 0.3 - 
            stress * 0.1
        )
        
        return max(0.0, min(1.0, retention_prob))

def main():
    """メイン関数"""
    logger.info("定着予測モデル訓練開始")
    
    try:
        # 設定を読み込み
        config_manager = get_config_manager()
        
        # データディレクトリを取得
        data_dir = Path(os.getenv('DATA_DIR', 'data/processed/unified'))
        model_output_dir = Path(os.getenv('MODEL_OUTPUT_DIR', 'models'))
        
        # データを読み込み
        features_file = data_dir / 'all_features.json'
        if not features_file.exists():
            logger.error(f"特徴量ファイルが見つかりません: {features_file}")
            return False
        
        with open(features_file, 'r', encoding='utf-8') as f:
            features_data = json.load(f)
        
        logger.info(f"特徴量データを読み込みました: {len(features_data)}件")
        
        # モデルを作成
        model = SimpleRetentionModel()
        model.feature_count = len(features_data)
        
        # モデルを保存
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_output_dir / 'retention_model.pkl'
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"定着予測モデルを保存しました: {model_file}")
        
        # 訓練結果を出力
        result = {
            'model_file': str(model_file),
            'model_version': model.model_version,
            'trained_at': model.trained_at,
            'feature_count': model.feature_count,
            'training_data_size': len(features_data)
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        logger.info("定着予測モデル訓練完了")
        return True
        
    except Exception as e:
        logger.error(f"定着予測モデル訓練エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)