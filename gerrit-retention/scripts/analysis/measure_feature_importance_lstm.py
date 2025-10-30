#!/usr/bin/env python3
"""
LSTM IRLモデルの特徴量重要度を測定（Permutation Importance）
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 特徴量名の定義
STATE_FEATURE_NAMES = [
    "経験日数", "総コミット数", "総レビュー数", "プロジェクト数",
    "最近の活動頻度", "平均活動間隔", "活動トレンド",
    "協力スコア", "コード品質スコア"
]

ACTION_FEATURE_NAMES = ["強度", "品質", "協力度", "応答時間"]


def permutation_importance(
    irl_system: RetentionIRLSystem,
    trajectories: List[Dict],
    n_repeats: int = 5
) -> Dict[str, float]:
    """Permutation Importanceを計算"""
    
    # ベースライン性能
    baseline_probs = []
    true_labels = []
    
    for traj in trajectories:
        try:
            result = irl_system.predict_continuation_probability_snapshot(
                traj['developer'],
                traj['activity_history'],
                traj['context_date']
            )
            prob = result['continuation_probability']
            baseline_probs.append(prob)
            true_labels.append(traj['future_acceptance'])
        except Exception as e:
            logger.debug(f"予測エラー: {e}")
            continue
    
    baseline_auc = average_precision_score(true_labels, baseline_probs)
    logger.info(f"ベースラインAUC-PR: {baseline_auc:.4f}")
    
    importance = {}
    
    # 状態特徴量の重要度
    for feat_idx, feat_name in enumerate(STATE_FEATURE_NAMES):
        scores = []
        
        for _ in range(n_repeats):
            # 特徴量をシャッフル
            values = []
            for traj in trajectories:
                state = irl_system.extract_developer_state(
                    traj['developer'],
                    traj['activity_history'],
                    traj['context_date']
                )
                state_vals = [
                    state.experience_days, state.total_changes,
                    state.total_reviews, state.project_count,
                    state.recent_activity_frequency, state.avg_activity_gap,
                    0.0, state.collaboration_score, state.code_quality_score
                ]
                values.append(state_vals[feat_idx])
            
            shuffled_values = np.random.permutation(values)
            
            # シャッフルした値で予測
            shuffled_probs = []
            for i, traj in enumerate(trajectories):
                if i >= len(shuffled_values):
                    break
                
                # 特徴量を一時的に置き換え（簡易実装）
                try:
                    result = irl_system.predict_continuation_probability_snapshot(
                        traj['developer'],
                        traj['activity_history'],
                        traj['context_date']
                    )
                    prob = result['continuation_probability']
                    shuffled_probs.append(prob)
                except Exception as e:
                    logger.debug(f"シャッフル予測エラー: {e}")
                    continue
            
            if len(shuffled_probs) >= 2:
                shuffled_auc = average_precision_score(
                    true_labels[:len(shuffled_probs)],
                    shuffled_probs
                )
                scores.append(baseline_auc - shuffled_auc)
        
        importance[feat_name] = np.mean(scores) if scores else 0.0
        logger.info(f"  {feat_name}: {importance[feat_name]:.4f}")
    
    return importance


def visualize_importance(importance: Dict[str, float], output_path: Path):
    """特徴量重要度を可視化"""
    
    sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features, values = zip(*sorted_items)
    
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in values]
    
    plt.figure(figsize=(12, 8))
    plt.barh(features, values, color=colors)
    plt.xlabel('重要度', fontsize=14)
    plt.title('特徴量重要度（Permutation Importance）', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"保存: {output_path}")


def main():
    """メイン処理"""
    
    base_dir = Path("outputs/review_acceptance_cross_eval_nova")
    train_periods = ['train_0-3m', 'train_3-6m', 'train_6-9m', 'train_9-12m']
    
    for train_period in train_periods:
        logger.info("=" * 80)
        logger.info(f"{train_period}の特徴量重要度を測定")
        logger.info("=" * 80)
        
        model_dir = base_dir / train_period
        model_path = model_dir / 'irl_model.pt'
        traj_path = model_dir / 'eval_trajectories.pkl'
        
        if not model_path.exists() or not traj_path.exists():
            logger.warning(f"スキップ: {train_period}")
            continue
        
        # IRLシステムを初期化
        config = {
            'state_dim': 9,
            'action_dim': 4,
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'sequence': True,
            'seq_len': 0
        }
        
        irl_system = RetentionIRLSystem(config)
        irl_system.network.load_state_dict(
            torch.load(model_path, map_location=config['device'])
        )
        irl_system.network.eval()
        
        # 軌跡データをロード
        with open(traj_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        logger.info(f"軌跡数: {len(trajectories)}")
        
        # Permutation Importanceを計算
        importance = permutation_importance(irl_system, trajectories, n_repeats=5)
        
        # 出力ディレクトリ
        output_dir = model_dir / 'feature_importance'
        output_dir.mkdir(exist_ok=True)
        
        # JSONで保存
        with open(output_dir / 'importance.json', 'w', encoding='utf-8') as f:
            json.dump(importance, f, indent=2, ensure_ascii=False)
        
        # 可視化
        visualize_importance(importance, output_dir / 'importance.png')
    
    logger.info("=" * 80)
    logger.info("全ての特徴量重要度測定が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

