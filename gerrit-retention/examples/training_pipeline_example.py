"""
訓練パイプライン使用例

開発者定着予測システムの訓練パイプラインの使用方法を示すサンプルスクリプト。
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from gerrit_retention.scripts.run_full_training_pipeline import FullTrainingPipeline
from gerrit_retention.utils.logger import setup_logger


def create_sample_data():
    """サンプル訓練データを作成"""
    logger = setup_logger(__name__)
    logger.info("サンプル訓練データを作成中...")
    
    # データディレクトリの作成
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプルサイズ
    n_samples = 1000
    
    # 定着予測用サンプルデータ
    retention_data = pd.DataFrame({
        'developer_id': range(n_samples),
        'expertise_level': np.random.uniform(0, 1, n_samples),
        'activity_pattern': np.random.uniform(0, 1, n_samples),
        'collaboration_score': np.random.uniform(0, 1, n_samples),
        'workload_ratio': np.random.uniform(0, 2, n_samples),
        'satisfaction_level': np.random.uniform(0, 1, n_samples),
        'avg_review_score': np.random.uniform(-2, 2, n_samples),
        'response_time_avg': np.random.uniform(1, 48, n_samples),
        'retention_label': np.random.binomial(1, 0.7, n_samples)  # 70%の定着率
    })
    
    retention_path = data_dir / "retention_training_data.csv"
    retention_data.to_csv(retention_path, index=False)
    logger.info(f"定着予測データを保存: {retention_path}")
    
    # ストレス分析用サンプルデータ
    stress_data = pd.DataFrame({
        'developer_id': range(n_samples),
        'expertise_level': np.random.uniform(0, 1, n_samples),
        'workload_ratio': np.random.uniform(0, 2, n_samples),
        'collaboration_score': np.random.uniform(0, 1, n_samples),
        'recent_changes': np.random.poisson(5, n_samples),
        'avg_complexity': np.random.uniform(0, 1, n_samples),
        'task_compatibility_stress': np.random.uniform(0, 1, n_samples),
        'workload_stress': np.random.uniform(0, 1, n_samples),
        'social_stress': np.random.uniform(0, 1, n_samples),
        'temporal_stress': np.random.uniform(0, 1, n_samples),
        'total_stress_score': np.random.uniform(0, 1, n_samples),
        'boiling_point_threshold': np.random.uniform(0.5, 1.0, n_samples),
        'risk_level': np.random.choice(['low', 'medium', 'high'], n_samples)
    })
    
    stress_path = data_dir / "stress_training_data.csv"
    stress_data.to_csv(stress_path, index=False)
    logger.info(f"ストレス分析データを保存: {stress_path}")
    
    # 強化学習用サンプルデータ
    rl_data = pd.DataFrame({
        'developer_id': range(n_samples),
        'expertise_level': np.random.uniform(0, 1, n_samples),
        'current_workload': np.random.uniform(0, 1, n_samples),
        'stress_level': np.random.uniform(0, 1, n_samples),
        'review_complexity': np.random.uniform(0, 1, n_samples),
        'requester_relationship': np.random.uniform(0, 1, n_samples),
        'time_pressure': np.random.uniform(0, 1, n_samples),
        'acceptance_history': np.random.uniform(0, 1, n_samples),
        'collaboration_quality': np.random.uniform(0, 1, n_samples)
    })
    
    rl_path = data_dir / "rl_training_data.csv"
    rl_data.to_csv(rl_path, index=False)
    logger.info(f"強化学習データを保存: {rl_path}")
    
    return {
        'retention': str(retention_path),
        'stress': str(stress_path),
        'rl': str(rl_path)
    }


def run_training_example():
    """訓練パイプラインの実行例"""
    logger = setup_logger(__name__)
    logger.info("=== 訓練パイプライン実行例開始 ===")
    
    try:
        # サンプルデータの作成
        data_paths = create_sample_data()
        
        # 設定ファイルのパス
        config_path = "configs/gerrit_config.yaml"
        
        # 訓練パイプラインの初期化
        pipeline = FullTrainingPipeline(config_path)
        
        # 全コンポーネントの訓練実行
        logger.info("全コンポーネントの訓練を開始...")
        results = pipeline.run_full_pipeline(data_paths)
        
        # 結果の表示
        logger.info("=== 訓練結果 ===")
        for component, model_path in results.items():
            logger.info(f"{component}: {model_path}")
        
        logger.info("=== 訓練パイプライン実行例完了 ===")
        
        return results
        
    except Exception as e:
        logger.error(f"訓練パイプライン実行例に失敗: {e}")
        raise


def run_individual_training_example():
    """個別コンポーネント訓練の実行例"""
    logger = setup_logger(__name__)
    logger.info("=== 個別コンポーネント訓練例開始 ===")
    
    try:
        # サンプルデータの作成
        data_paths = create_sample_data()
        
        # 設定ファイルのパス
        config_path = "configs/gerrit_config.yaml"
        
        # 訓練パイプラインの初期化
        pipeline = FullTrainingPipeline(config_path)
        
        # 定着予測モデルのみ訓練
        logger.info("定着予測モデルのみ訓練...")
        retention_path = pipeline.train_retention_model(data_paths['retention'])
        if retention_path:
            logger.info(f"定着予測モデル訓練完了: {retention_path}")
        
        # ストレス分析モデルのみ訓練
        logger.info("ストレス分析モデルのみ訓練...")
        stress_path = pipeline.train_stress_model(data_paths['stress'])
        if stress_path:
            logger.info(f"ストレス分析モデル訓練完了: {stress_path}")
        
        # 強化学習エージェントのみ訓練
        logger.info("強化学習エージェントのみ訓練...")
        rl_path = pipeline.train_rl_agent(data_paths['rl'])
        if rl_path:
            logger.info(f"強化学習エージェント訓練完了: {rl_path}")
        
        logger.info("=== 個別コンポーネント訓練例完了 ===")
        
    except Exception as e:
        logger.error(f"個別コンポーネント訓練例に失敗: {e}")
        raise


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練パイプライン使用例")
    parser.add_argument("--mode", choices=['full', 'individual'], default='full',
                       help="実行モード: full=統合訓練, individual=個別訓練")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'full':
            run_training_example()
        else:
            run_individual_training_example()
            
    except Exception as e:
        print(f"実行に失敗: {e}")
        exit(1)


if __name__ == "__main__":
    main()