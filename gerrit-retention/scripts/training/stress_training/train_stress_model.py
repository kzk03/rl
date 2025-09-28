#!/usr/bin/env python3
"""
ストレス分析モデル訓練スクリプト
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

class SimpleStressModel:
    """簡単なストレス分析モデル"""
    
    def __init__(self):
        self.trained_at = datetime.now().isoformat()
        self.model_version = "1.0.0"
        self.developer_count = 0
        self.review_count = 0
    
    def predict_stress(self, developer_data, context_data=None):
        """ストレスレベルを予測"""
        base_stress = developer_data.get('stress_level', 0.5)
        expertise = developer_data.get('expertise_level', 0.5)
        collaboration = developer_data.get('collaboration_quality', 0.5)
        
        # ワークロード分析
        activity = developer_data.get('activity_pattern', {})
        commits_per_week = activity.get('commits_per_week', 10)
        reviews_per_week = activity.get('reviews_per_week', 10)
        
        workload_stress = min(1.0, (commits_per_week + reviews_per_week) / 30.0)
        
        # 総合ストレス計算
        total_stress = (
            base_stress * 0.4 +
            workload_stress * 0.3 +
            (1.0 - collaboration) * 0.2 +
            (1.0 - expertise) * 0.1
        )
        
        return max(0.0, min(1.0, total_stress))
    
    def predict_boiling_point(self, developer_data, context_data=None):
        """沸点予測（ストレスが限界に達するまでの時間）"""
        current_stress = self.predict_stress(developer_data, context_data)
        stress_tolerance = developer_data.get('expertise_level', 0.5)
        
        if current_stress < 0.3:
            return 30  # 30日以上
        elif current_stress < 0.6:
            return 14  # 2週間
        elif current_stress < 0.8:
            return 7   # 1週間
        else:
            return 3   # 3日以内

def main():
    """メイン関数"""
    logger.info("ストレス分析モデル訓練開始")
    
    try:
        # 設定を読み込み
        config_manager = get_config_manager()
        
        # データディレクトリを取得
        data_dir = Path(os.getenv('DATA_DIR', 'data/processed/unified'))
        model_output_dir = Path(os.getenv('MODEL_OUTPUT_DIR', 'models'))
        
        # データを読み込み
        developers_file = data_dir / 'all_developers.json'
        reviews_file = data_dir / 'all_reviews.json'
        
        if not developers_file.exists():
            logger.error(f"開発者ファイルが見つかりません: {developers_file}")
            return False
        
        if not reviews_file.exists():
            logger.error(f"レビューファイルが見つかりません: {reviews_file}")
            return False
        
        with open(developers_file, 'r', encoding='utf-8') as f:
            developers_data = json.load(f)
        
        with open(reviews_file, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
        
        logger.info(f"開発者データを読み込みました: {len(developers_data)}件")
        logger.info(f"レビューデータを読み込みました: {len(reviews_data)}件")
        
        # モデルを作成
        model = SimpleStressModel()
        model.developer_count = len(developers_data)
        model.review_count = len(reviews_data)
        
        # モデルを保存
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_output_dir / 'stress_model.pkl'
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"ストレス分析モデルを保存しました: {model_file}")
        
        # 訓練結果を出力
        result = {
            'model_file': str(model_file),
            'model_version': model.model_version,
            'trained_at': model.trained_at,
            'developer_count': model.developer_count,
            'review_count': model.review_count
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        logger.info("ストレス分析モデル訓練完了")
        return True
        
    except Exception as e:
        logger.error(f"ストレス分析モデル訓練エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)