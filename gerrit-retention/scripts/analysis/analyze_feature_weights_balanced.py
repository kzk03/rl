#!/usr/bin/env python3
"""
バランス型 Focal Loss モデルの特徴量重要度分析

勾配ベースの特徴量重要度を計算し、各特徴量が予測にどう影響するかを分析
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['font.size'] = 10

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 特徴量名（9次元 state + 4次元 action）
STATE_FEATURE_NAMES = [
    '総変更数',
    '総レビュー数',
    'プロジェクト数',
    '最近の活動頻度',
    '平均活動間隔',
    '活動トレンド',
    '協力スコア',
    'コード品質スコア',
    '在籍日数'
]

ACTION_FEATURE_NAMES = [
    '変更の挿入行数',
    '変更の削除行数',
    '変更ファイル数',
    'メッセージ品質'
]


def compute_gradient_importance(
    model: RetentionIRLSystem,
    trajectories: list,
    device: torch.device
) -> dict:
    """
    勾配ベースの特徴量重要度を計算
    
    Args:
        model: 訓練済みモデル
        trajectories: 評価用軌跡
        device: デバイス
    
    Returns:
        特徴量重要度の辞書
    """
    model.network.eval()
    
    state_gradients = []
    action_gradients = []
    
    for traj in trajectories:
        developer = traj['developer']
        activity_history = traj['activity_history']
        context_date = traj['context_date']
        
        # 特徴量を計算
        result = model.predict_continuation_probability_snapshot(
            developer=developer,
            activity_history=activity_history,
            context_date=context_date
        )
        
        state_features = result['state_features']
        action_features = result['action_features']
        
        # Tensor に変換して勾配を有効化
        state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device, requires_grad=True)
        action_tensor = torch.tensor(action_features, dtype=torch.float32, device=device, requires_grad=True)
        
        # 予測
        with torch.enable_grad():
            # ネットワークに通す
            state_encoded = model.network.state_encoder(state_tensor.unsqueeze(0))
            action_encoded = model.network.action_encoder(action_tensor.unsqueeze(0))
            combined = torch.cat([state_encoded, action_encoded], dim=1)
            reward = model.network.reward_head(combined)
            continuation_prob = torch.sigmoid(reward)
            
            # 逆伝播
            continuation_prob.backward()
        
        # 勾配を取得（符号付き）
        if state_tensor.grad is not None:
            state_gradients.append(state_tensor.grad.cpu().numpy())
        if action_tensor.grad is not None:
            action_gradients.append(action_tensor.grad.cpu().numpy())
    
    # 平均勾配を計算（符号を保持）
    state_importance = np.mean(state_gradients, axis=0)
    action_importance = np.mean(action_gradients, axis=0)
    
    return {
        'state': state_importance,
        'action': action_importance
    }


def visualize_feature_importance(
    state_importance: np.ndarray,
    action_importance: np.ndarray,
    output_path: Path,
    title: str
):
    """
    特徴量重要度を可視化
    
    Args:
        state_importance: 状態特徴量の重要度
        action_importance: 行動特徴量の重要度
        output_path: 出力先
        title: グラフタイトル
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 状態特徴量
    colors_state = ['red' if x < 0 else 'blue' for x in state_importance]
    ax1.barh(STATE_FEATURE_NAMES, state_importance, color=colors_state)
    ax1.set_xlabel('重要度（勾配の平均）', fontsize=12)
    ax1.set_title('状態特徴量の重要度', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # 行動特徴量
    colors_action = ['red' if x < 0 else 'blue' for x in action_importance]
    ax2.barh(ACTION_FEATURE_NAMES, action_importance, color=colors_action)
    ax2.set_xlabel('重要度（勾配の平均）', fontsize=12)
    ax2.set_title('行動特徴量の重要度', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可視化を保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='特徴量重要度分析（バランス型 Focal Loss）')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='outputs/review_acceptance_cross_eval_nova_balanced',
        help='モデルディレクトリ'
    )
    parser.add_argument(
        '--train-period',
        type=str,
        default='0-3m',
        help='訓練期間'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir) / f'train_{args.train_period}'
    
    if not model_dir.exists():
        logger.error(f"モデルディレクトリが存在しません: {model_dir}")
        return
    
    logger.info("=" * 80)
    logger.info(f"特徴量重要度分析: train_{args.train_period}")
    logger.info("=" * 80)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"デバイス: {device}")
    
    # モデル設定
    config = {
        'network': {
            'state_dim': 9,
            'action_dim': 4,
            'hidden_dim': 128,
            'learning_rate': 0.0001,
            'sequence': True,
            'seq_len': 0
        }
    }
    
    # モデルを読み込み
    model = RetentionIRLSystem(config=config)
    model_path = model_dir / 'irl_model.pt'
    
    if not model_path.exists():
        logger.error(f"モデルファイルが存在しません: {model_path}")
        return
    
    model.network.load_state_dict(torch.load(model_path, map_location=device))
    model.network.to(device)
    model.network.eval()
    
    logger.info(f"モデルを読み込みました: {model_path}")
    
    # 評価用軌跡を読み込み
    eval_traj_path = model_dir / 'eval_trajectories.pkl'
    
    if not eval_traj_path.exists():
        logger.error(f"評価用軌跡が存在しません: {eval_traj_path}")
        return
    
    with open(eval_traj_path, 'rb') as f:
        eval_trajectories = pickle.load(f)
    
    logger.info(f"評価用軌跡を読み込みました: {len(eval_trajectories)}サンプル")
    
    # 特徴量重要度を計算
    logger.info("特徴量重要度を計算中...")
    importance = compute_gradient_importance(model, eval_trajectories, device)
    
    # 結果を表示
    logger.info("=" * 80)
    logger.info("状態特徴量の重要度:")
    logger.info("=" * 80)
    for name, value in zip(STATE_FEATURE_NAMES, importance['state']):
        sign = "➕" if value > 0 else "➖"
        logger.info(f"  {sign} {name}: {value:.6f}")
    
    logger.info("=" * 80)
    logger.info("行動特徴量の重要度:")
    logger.info("=" * 80)
    for name, value in zip(ACTION_FEATURE_NAMES, importance['action']):
        sign = "➕" if value > 0 else "➖"
        logger.info(f"  {sign} {name}: {value:.6f}")
    
    # 可視化
    output_path = model_dir / f'feature_importance_train_{args.train_period}.png'
    visualize_feature_importance(
        importance['state'],
        importance['action'],
        output_path,
        title=f'特徴量重要度分析 (train_{args.train_period}, バランス型 Focal Loss)'
    )
    
    # JSON で保存
    importance_dict = {
        'state': {name: float(value) for name, value in zip(STATE_FEATURE_NAMES, importance['state'])},
        'action': {name: float(value) for name, value in zip(ACTION_FEATURE_NAMES, importance['action'])}
    }
    
    json_path = model_dir / f'feature_importance_train_{args.train_period}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(importance_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果を保存しました: {json_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

