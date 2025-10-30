#!/usr/bin/env python3
"""
LSTM IRLモデルの特徴量重要度を分析

Permutation ImportanceとGradient-based Importanceの両方を計算
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 特徴量名の定義
STATE_FEATURE_NAMES = [
    "経験日数",
    "総コミット数",
    "総レビュー数",
    "プロジェクト数",
    "最近の活動頻度",
    "平均活動間隔",
    "活動トレンド",
    "協力スコア",
    "コード品質スコア"
]

ACTION_FEATURE_NAMES = [
    "強度",
    "品質",
    "協力度",
    "応答時間"
]


def load_trajectories_from_predictions(predictions_csv: Path) -> List[Dict]:
    """予測結果CSVから軌跡データを再構築"""
    df = pd.read_csv(predictions_csv)
    
    trajectories = []
    for _, row in df.iterrows():
        traj = {
            'reviewer': row['reviewer_email'],
            'predicted_prob': row['predicted_prob'],
            'true_label': row['true_label']
        }
        trajectories.append(traj)
    
    return trajectories


def calculate_permutation_importance(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict],
    n_repeats: int = 10
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Permutation Importanceを計算
    
    各特徴量をシャッフルして、性能低下を測定
    """
    logger.info("Permutation Importanceを計算中...")
    
    # ベースライン性能を計算
    baseline_preds = []
    true_labels = []
    
    for traj in eval_trajectories:
        if 'developer_info' not in traj or 'activity_history' not in traj:
            continue
        
        developer = traj['developer_info']
        activity_history = traj['activity_history']
        context_date = traj['context_date']
        
        try:
            prob = irl_system.predict_continuation_probability_snapshot(
                developer, activity_history, context_date
            )
            baseline_preds.append(prob)
            true_labels.append(traj.get('future_acceptance', 0))
        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            continue
    
    if len(baseline_preds) < 2:
        logger.warning("予測データが不足しています")
        return {}, {}
    
    baseline_auc_pr = average_precision_score(true_labels, baseline_preds)
    logger.info(f"ベースラインAUC-PR: {baseline_auc_pr:.4f}")
    
    # 状態特徴量の重要度
    state_importance = {}
    state_dim = irl_system.state_dim
    
    for feat_idx in range(state_dim):
        if feat_idx >= len(STATE_FEATURE_NAMES):
            break
        
        feat_name = STATE_FEATURE_NAMES[feat_idx]
        importance_scores = []
        
        for _ in range(n_repeats):
            # この特徴量をシャッフル
            shuffled_preds = []
            
            for traj in eval_trajectories:
                if 'developer_info' not in traj or 'activity_history' not in traj:
                    continue
                
                developer = traj['developer_info'].copy()
                activity_history = traj['activity_history']
                context_date = traj['context_date']
                
                # 状態を抽出
                state = irl_system.extract_developer_state(developer, activity_history, context_date)
                
                # 特徴量をシャッフル（ランダム値で置き換え）
                state_dict = state.__dict__.copy()
                state_values = [
                    state.experience_days,
                    state.total_changes,
                    state.total_reviews,
                    state.project_count,
                    state.recent_activity_frequency,
                    state.avg_activity_gap,
                    0.0 if state.activity_trend == 'increasing' else 1.0,  # カテゴリ変数
                    state.collaboration_score,
                    state.code_quality_score
                ]
                
                # ランダムシャッフル
                shuffled_value = np.random.permutation([state_values[feat_idx]])[0]
                
                # 状態を更新
                if feat_idx == 0:
                    state_dict['experience_days'] = shuffled_value
                elif feat_idx == 1:
                    state_dict['total_changes'] = shuffled_value
                elif feat_idx == 2:
                    state_dict['total_reviews'] = shuffled_value
                elif feat_idx == 3:
                    state_dict['project_count'] = shuffled_value
                elif feat_idx == 4:
                    state_dict['recent_activity_frequency'] = shuffled_value
                elif feat_idx == 5:
                    state_dict['avg_activity_gap'] = shuffled_value
                elif feat_idx == 6:
                    state_dict['activity_trend'] = 'increasing' if shuffled_value < 0.5 else 'decreasing'
                elif feat_idx == 7:
                    state_dict['collaboration_score'] = shuffled_value
                elif feat_idx == 8:
                    state_dict['code_quality_score'] = shuffled_value
                
                # 予測（状態のみシャッフル）
                try:
                    prob = irl_system.predict_continuation_probability_snapshot(
                        developer, activity_history, context_date
                    )
                    shuffled_preds.append(prob)
                except Exception as e:
                    continue
            
            if len(shuffled_preds) >= 2:
                shuffled_auc_pr = average_precision_score(true_labels[:len(shuffled_preds)], shuffled_preds)
                importance = baseline_auc_pr - shuffled_auc_pr
                importance_scores.append(importance)
        
        if importance_scores:
            state_importance[feat_name] = np.mean(importance_scores)
            logger.info(f"  {feat_name}: {np.mean(importance_scores):.4f}")
    
    # 行動特徴量の重要度も同様に計算（簡略版）
    action_importance = {}
    for feat_name in ACTION_FEATURE_NAMES:
        action_importance[feat_name] = 0.0  # 実装簡略化のため0とする
    
    return state_importance, action_importance


def calculate_gradient_importance(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Gradient-based Importanceを計算
    
    各特徴量に対する勾配の大きさを測定
    """
    logger.info("Gradient-based Importanceを計算中...")
    
    irl_system.network.eval()
    
    state_gradients = []
    action_gradients = []
    
    for traj in eval_trajectories[:50]:  # 計算コスト削減のため50サンプル
        if 'developer_info' not in traj or 'activity_history' not in traj:
            continue
        
        developer = traj['developer_info']
        activity_history = traj['activity_history']
        context_date = traj['context_date']
        
        try:
            # 状態と行動を抽出
            state = irl_system.extract_developer_state(developer, activity_history, context_date)
            actions = irl_system.extract_developer_actions(activity_history, context_date)
            
            if not actions:
                continue
            
            action = actions[-1]
            
            # テンソルに変換（勾配計算を有効化）
            state_tensor = irl_system.state_to_tensor(state).unsqueeze(0)
            action_tensor = irl_system.action_to_tensor(action).unsqueeze(0)
            
            state_tensor.requires_grad = True
            action_tensor.requires_grad = True
            
            # 予測
            _, continuation_prob = irl_system.network(state_tensor, action_tensor)
            
            # 勾配計算
            continuation_prob.backward()
            
            # 勾配の絶対値を記録
            if state_tensor.grad is not None:
                state_gradients.append(state_tensor.grad.abs().squeeze().cpu().detach().numpy())
            if action_tensor.grad is not None:
                action_gradients.append(action_tensor.grad.abs().squeeze().cpu().detach().numpy())
        
        except Exception as e:
            logger.warning(f"勾配計算エラー: {e}")
            continue
    
    # 平均勾配を計算
    state_importance = {}
    if state_gradients:
        mean_gradients = np.mean(state_gradients, axis=0)
        for i, feat_name in enumerate(STATE_FEATURE_NAMES[:len(mean_gradients)]):
            state_importance[feat_name] = float(mean_gradients[i])
            logger.info(f"  {feat_name}: {mean_gradients[i]:.6f}")
    
    action_importance = {}
    if action_gradients:
        mean_gradients = np.mean(action_gradients, axis=0)
        for i, feat_name in enumerate(ACTION_FEATURE_NAMES[:len(mean_gradients)]):
            action_importance[feat_name] = float(mean_gradients[i])
            logger.info(f"  {feat_name}: {mean_gradients[i]:.6f}")
    
    return state_importance, action_importance


def visualize_feature_importance(
    state_importance: Dict[str, float],
    action_importance: Dict[str, float],
    output_dir: Path,
    method: str = "permutation"
):
    """特徴量重要度を可視化"""
    
    # 状態特徴量のプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 状態特徴量
    if state_importance:
        sorted_items = sorted(state_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        features, importances = zip(*sorted_items)
        
        colors = ['#d62728' if imp < 0 else '#2ca02c' for imp in importances]
        
        ax1.barh(features, importances, color=colors)
        ax1.set_xlabel('重要度', fontsize=12)
        ax1.set_title(f'状態特徴量の重要度 ({method})', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
    
    # 行動特徴量
    if action_importance:
        sorted_items = sorted(action_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        features, importances = zip(*sorted_items)
        
        colors = ['#d62728' if imp < 0 else '#2ca02c' for imp in importances]
        
        ax2.barh(features, importances, color=colors)
        ax2.set_xlabel('重要度', fontsize=12)
        ax2.set_title(f'行動特徴量の重要度 ({method})', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'feature_importance_{method}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可視化を保存: {output_path}")


def main():
    """メイン処理"""
    
    base_dir = Path("outputs/review_acceptance_cross_eval_nova")
    
    # 各訓練期間のモデルを分析
    train_periods = ['train_0-3m', 'train_3-6m', 'train_6-9m', 'train_9-12m']
    
    for train_period in train_periods:
        logger.info("=" * 80)
        logger.info(f"{train_period}モデルの特徴量重要度を分析")
        logger.info("=" * 80)
        
        model_dir = base_dir / train_period
        model_path = model_dir / 'irl_model.pt'
        
        if not model_path.exists():
            logger.warning(f"モデルが見つかりません: {model_path}")
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
        
        # モデルをロード
        try:
            irl_system.network.load_state_dict(torch.load(model_path, map_location=config['device']))
            irl_system.network.eval()
            logger.info(f"モデルをロード: {model_path}")
        except Exception as e:
            logger.error(f"モデルロードエラー: {e}")
            continue
        
        # 評価データをロード
        eval_csv = model_dir / 'eval_0-3m' / 'predictions.csv'
        if not eval_csv.exists():
            logger.warning(f"評価データが見つかりません: {eval_csv}")
            continue
        
        # 軌跡データを読み込み（簡易版）
        eval_df = pd.read_csv(eval_csv)
        logger.info(f"評価データを読み込み: {len(eval_df)}件")
        
        # 注意: 完全な特徴量重要度計算には、元の軌跡データが必要
        # ここでは簡易的にGradient-based Importanceのみ計算
        
        logger.info("注意: 完全なPermutation Importanceには元データが必要です")
        logger.info("Gradient-based Importanceを計算します...")
        
        # 出力ディレクトリを作成
        output_dir = model_dir / 'feature_importance'
        output_dir.mkdir(exist_ok=True)
        
        # ダミーの軌跡データを作成（実際には元データから再構築が必要）
        logger.warning("完全な実装には、訓練・評価時の軌跡データの保存が必要です")
        logger.info(f"スキップ: {train_period}")
        
        # JSONで保存（将来の実装用）
        importance_data = {
            'train_period': train_period,
            'state_importance': {},
            'action_importance': {},
            'method': 'gradient',
            'note': '完全な実装には軌跡データが必要'
        }
        
        with open(output_dir / 'feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("注意: 特徴量重要度の完全な計算には、")
    logger.info("訓練・評価スクリプトで軌跡データを保存する必要があります。")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

