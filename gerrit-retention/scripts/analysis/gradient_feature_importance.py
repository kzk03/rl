#!/usr/bin/env python3
"""
LSTM IRLモデルの特徴量重要度を勾配ベースで測定
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

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 特徴量名の定義（10次元版）
STATE_FEATURE_NAMES = [
    "経験日数", "総コミット数", "総レビュー数",
    "最近の活動頻度", "平均活動間隔", "活動トレンド",
    "協力スコア", "コード品質スコア",
    "最近の受諾率", "レビュー負荷"
]

ACTION_FEATURE_NAMES = ["強度", "品質", "協力度", "応答速度"]


def calculate_gradient_importance(
    irl_system: RetentionIRLSystem,
    trajectories: List[Dict]
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    勾配ベースの特徴量重要度を計算
    
    各特徴量に対する予測への勾配の平均絶対値を計算
    """
    logger.info("勾配ベースの特徴量重要度を計算中...")
    
    irl_system.network.eval()
    
    state_gradients = []
    action_gradients = []
    
    for i, traj in enumerate(trajectories):
        if i >= 50:  # 計算コスト削減のため50サンプル
            break
        
        try:
            # 状態と行動を抽出
            state = irl_system.extract_developer_state(
                traj['developer'],
                traj['activity_history'],
                traj['context_date']
            )
            actions = irl_system.extract_developer_actions(
                traj['activity_history'],
                traj['context_date']
            )
            
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
            
            # 勾配を記録（正負を保持）
            if state_tensor.grad is not None:
                state_gradients.append(state_tensor.grad.squeeze().cpu().detach().numpy())
            if action_tensor.grad is not None:
                action_gradients.append(action_tensor.grad.squeeze().cpu().detach().numpy())
        
        except Exception as e:
            logger.debug(f"勾配計算エラー: {e}")
            continue
    
    # 平均勾配を計算
    state_importance = {}
    if state_gradients:
        mean_gradients = np.mean(state_gradients, axis=0)
        for i, feat_name in enumerate(STATE_FEATURE_NAMES[:len(mean_gradients)]):
            state_importance[feat_name] = float(mean_gradients[i])
            logger.info(f"  状態 - {feat_name}: {mean_gradients[i]:.6f}")
    
    action_importance = {}
    if action_gradients:
        mean_gradients = np.mean(action_gradients, axis=0)
        for i, feat_name in enumerate(ACTION_FEATURE_NAMES[:len(mean_gradients)]):
            action_importance[feat_name] = float(mean_gradients[i])
            logger.info(f"  行動 - {feat_name}: {mean_gradients[i]:.6f}")
    
    return state_importance, action_importance


def visualize_importance(
    state_importance: Dict[str, float],
    action_importance: Dict[str, float],
    output_dir: Path
):
    """特徴量重要度を可視化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 状態特徴量
    if state_importance:
        sorted_items = sorted(state_importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_items)
        
        colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
        
        ax1.barh(features, values, color=colors)
        ax1.set_xlabel('勾配の大きさ', fontsize=14)
        ax1.set_title('状態特徴量の重要度（勾配ベース）', fontsize=16, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    
    # 行動特徴量
    if action_importance:
        sorted_items = sorted(action_importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_items)
        
        colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
        
        ax2.barh(features, values, color=colors)
        ax2.set_xlabel('勾配の大きさ', fontsize=14)
        ax2.set_title('行動特徴量の重要度（勾配ベース）', fontsize=16, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"保存: {output_path}")


def visualize_combined_importance(
    state_importance: Dict[str, float],
    action_importance: Dict[str, float],
    output_dir: Path
):
    """状態と行動の特徴量重要度を統合してプロット"""
    
    # 全特徴量を統合
    all_importance = {}
    for feat, val in state_importance.items():
        all_importance[f"状態: {feat}"] = val
    for feat, val in action_importance.items():
        all_importance[f"行動: {feat}"] = val
    
    # ソート
    sorted_items = sorted(all_importance.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_items)
    
    # プロット
    plt.figure(figsize=(12, 10))
    colors = ['#1f77b4' if '状態' in f else '#ff7f0e' for f in features]
    
    plt.barh(features, values, color=colors)
    plt.xlabel('勾配の大きさ', fontsize=14)
    plt.title('全特徴量の重要度（勾配ベース）', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='状態特徴量'),
        Patch(facecolor='#ff7f0e', label='行動特徴量')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_importance_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"保存: {output_path}")


def main():
    """メイン処理"""
    
    base_dir = Path("outputs/review_acceptance_cross_eval_nova")
    train_periods = ['train_0-3m', 'train_3-6m', 'train_6-9m', 'train_9-12m']
    
    all_results = {}
    
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
            'state_dim': 10,  # 最近の受諾率+レビュー負荷を追加
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
        
        # 勾配ベースの重要度を計算
        state_importance, action_importance = calculate_gradient_importance(
            irl_system, trajectories
        )
        
        # 結果を保存
        all_results[train_period] = {
            'state_importance': state_importance,
            'action_importance': action_importance
        }
        
        # 出力ディレクトリ
        output_dir = model_dir / 'feature_importance'
        output_dir.mkdir(exist_ok=True)
        
        # JSONで保存
        with open(output_dir / 'gradient_importance.json', 'w', encoding='utf-8') as f:
            json.dump({
                'state_importance': state_importance,
                'action_importance': action_importance,
                'method': 'gradient_based'
            }, f, indent=2, ensure_ascii=False)
        
        # 可視化
        visualize_importance(state_importance, action_importance, output_dir)
        visualize_combined_importance(state_importance, action_importance, output_dir)
    
    # 全訓練期間の平均を計算
    if all_results:
        logger.info("=" * 80)
        logger.info("全訓練期間の平均重要度")
        logger.info("=" * 80)
        
        avg_state = {}
        avg_action = {}
        
        for feat in STATE_FEATURE_NAMES:
            values = [r['state_importance'].get(feat, 0.0) for r in all_results.values()]
            avg_state[feat] = np.mean(values)
        
        for feat in ACTION_FEATURE_NAMES:
            values = [r['action_importance'].get(feat, 0.0) for r in all_results.values()]
            avg_action[feat] = np.mean(values)
        
        logger.info("状態特徴量:")
        for feat, val in sorted(avg_state.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feat}: {val:.6f}")
        
        logger.info("行動特徴量:")
        for feat, val in sorted(avg_action.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feat}: {val:.6f}")
        
        # 平均結果を保存
        avg_output_dir = base_dir / 'average_feature_importance'
        avg_output_dir.mkdir(exist_ok=True)
        
        with open(avg_output_dir / 'gradient_importance_average.json', 'w', encoding='utf-8') as f:
            json.dump({
                'state_importance': avg_state,
                'action_importance': avg_action,
                'method': 'gradient_based_average'
            }, f, indent=2, ensure_ascii=False)
        
        visualize_importance(avg_state, avg_action, avg_output_dir)
        visualize_combined_importance(avg_state, avg_action, avg_output_dir)
    
    logger.info("=" * 80)
    logger.info("全ての特徴量重要度測定が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

