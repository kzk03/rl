#!/usr/bin/env python3
"""
特徴量の重要度分析スクリプト

手法:
1. Permutation Importance: 各特徴量をランダム化してモデル性能の変化を測定
2. Gradient-based Importance: 予測に対する各特徴量の勾配を計算
3. Zero-out Importance: 特徴量をゼロにした時の性能変化を測定
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from training.irl.train_irl_within_training_period import (
    extract_cutoff_evaluation_trajectories,
    load_review_logs,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 特徴量名の定義
STATE_FEATURE_NAMES = [
    "経験日数(年)",
    "総変更数(/100)",
    "総レビュー数(/100)",
    "プロジェクト数(/10)",
    "最近の活動頻度",
    "平均活動間隔(月)",
    "活動トレンド",
    "協力スコア",
    "コード品質スコア",
    "時間経過(年)"
]

ACTION_FEATURE_NAMES = [
    "行動タイプ",
    "強度",
    "品質",
    "協力度",
    "時間経過(年)"
]


def compute_baseline_performance(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict[str, Any]]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """ベースライン性能を計算"""
    predictions = []
    true_labels = []
    
    for trajectory in eval_trajectories:
        developer = trajectory.get('developer', trajectory.get('developer_info', {}))
        activity_history = trajectory['activity_history']
        context_date = trajectory.get('context_date')
        true_label = trajectory.get('future_contribution', False)
        
        result = irl_system.predict_continuation_probability(
            developer, activity_history, context_date
        )
        
        predictions.append(result['continuation_probability'])
        true_labels.append(1 if true_label else 0)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # AUC-ROCを性能指標として使用
    from sklearn.metrics import roc_auc_score
    try:
        auc_score = roc_auc_score(true_labels, predictions)
    except:
        auc_score = 0.5
    
    return auc_score, predictions, true_labels


def permutation_importance(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict[str, Any]],
    baseline_score: float,
    feature_type: str = 'state',
    n_repeats: int = 5
) -> Dict[int, float]:
    """
    Permutation Importance を計算
    
    Args:
        irl_system: 訓練済みモデル
        eval_trajectories: 評価データ
        baseline_score: ベースライン性能
        feature_type: 'state' or 'action'
        n_repeats: 繰り返し回数
        
    Returns:
        Dict[int, float]: 特徴量インデックス -> 重要度スコア
    """
    logger.info(f"Permutation Importance計算開始 ({feature_type})")
    
    if feature_type == 'state':
        n_features = 10
    else:  # action
        n_features = 5
    
    importance_scores = {}
    
    for feature_idx in range(n_features):
        scores_drop = []
        
        for repeat in range(n_repeats):
            # この特徴量をシャッフルして予測
            permuted_predictions = []
            true_labels = []
            
            for trajectory in eval_trajectories:
                developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                activity_history = trajectory['activity_history']
                context_date = trajectory.get('context_date')
                true_label = trajectory.get('future_contribution', False)
                
                # 状態と行動を抽出
                state = irl_system.extract_developer_state(developer, activity_history, context_date)
                actions = irl_system.extract_developer_actions(activity_history, context_date)
                
                if not actions:
                    continue
                
                # テンソルに変換
                state_tensor = irl_system.state_to_tensor(state)
                
                if feature_type == 'state':
                    # 特定の特徴量をランダムな値に置換
                    original_value = state_tensor[feature_idx].item()
                    # 他のサンプルからランダムにサンプリング
                    random_value = np.random.uniform(0, 1)
                    state_tensor[feature_idx] = random_value
                
                # シーケンスの場合
                if irl_system.sequence:
                    recent_actions = actions[-min(len(actions), irl_system.seq_len if irl_system.seq_len else len(actions)):]
                    
                    if feature_type == 'action':
                        # 行動の特徴量をシャッフル
                        action_tensors = []
                        for action in recent_actions:
                            action_tensor = irl_system.action_to_tensor(action)
                            random_value = np.random.uniform(0, 1)
                            action_tensor[feature_idx] = random_value
                            action_tensors.append(action_tensor)
                    else:
                        action_tensors = [irl_system.action_to_tensor(action) for action in recent_actions]
                    
                    state_seq = state_tensor.unsqueeze(0).repeat(len(action_tensors), 1).unsqueeze(0)
                    action_seq = torch.stack(action_tensors).unsqueeze(0)
                    
                    _, pred_prob = irl_system.network(state_seq, action_seq)
                else:
                    action_tensor = irl_system.action_to_tensor(actions[-1])
                    
                    if feature_type == 'action':
                        random_value = np.random.uniform(0, 1)
                        action_tensor[feature_idx] = random_value
                    
                    _, pred_prob = irl_system.network(
                        state_tensor.unsqueeze(0),
                        action_tensor.unsqueeze(0)
                    )
                
                permuted_predictions.append(pred_prob.item())
                true_labels.append(1 if true_label else 0)
            
            # 性能を計算
            permuted_predictions = np.array(permuted_predictions)
            true_labels = np.array(true_labels)
            
            from sklearn.metrics import roc_auc_score
            try:
                permuted_score = roc_auc_score(true_labels, permuted_predictions)
            except:
                permuted_score = 0.5
            
            # 性能の低下 = 重要度
            score_drop = baseline_score - permuted_score
            scores_drop.append(score_drop)
        
        # 平均を取る
        importance_scores[feature_idx] = np.mean(scores_drop)
        logger.info(f"特徴量 {feature_idx}: 重要度 = {importance_scores[feature_idx]:.4f}")
    
    return importance_scores


def gradient_based_importance(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict[str, Any]],
    feature_type: str = 'state'
) -> Dict[int, float]:
    """
    勾配ベースの特徴量重要度を計算
    
    予測に対する各特徴量の勾配の絶対値の平均を重要度とする
    """
    logger.info(f"Gradient-based Importance計算開始 ({feature_type})")
    
    if feature_type == 'state':
        n_features = 10
    else:
        n_features = 5
    
    gradients_sum = np.zeros(n_features)
    count = 0
    
    irl_system.network.eval()
    
    for trajectory in eval_trajectories:
        developer = trajectory.get('developer', trajectory.get('developer_info', {}))
        activity_history = trajectory['activity_history']
        context_date = trajectory.get('context_date')
        
        # 状態と行動を抽出
        state = irl_system.extract_developer_state(developer, activity_history, context_date)
        actions = irl_system.extract_developer_actions(activity_history, context_date)
        
        if not actions:
            continue
        
        # テンソルに変換（requires_grad=True）
        state_tensor = irl_system.state_to_tensor(state)
        state_tensor.requires_grad = True
        
        if irl_system.sequence:
            recent_actions = actions[-min(len(actions), irl_system.seq_len if irl_system.seq_len else len(actions)):]
            action_tensors = [irl_system.action_to_tensor(action) for action in recent_actions]
            
            if feature_type == 'action':
                for act_tensor in action_tensors:
                    act_tensor.requires_grad = True
            
            state_seq = state_tensor.unsqueeze(0).repeat(len(action_tensors), 1).unsqueeze(0)
            action_seq = torch.stack(action_tensors).unsqueeze(0)
            
            _, pred_prob = irl_system.network(state_seq, action_seq)
        else:
            action_tensor = irl_system.action_to_tensor(actions[-1])
            if feature_type == 'action':
                action_tensor.requires_grad = True
            
            _, pred_prob = irl_system.network(
                state_tensor.unsqueeze(0),
                action_tensor.unsqueeze(0)
            )
        
        # 勾配を計算
        pred_prob.backward()
        
        if feature_type == 'state':
            if state_tensor.grad is not None:
                gradients_sum += np.abs(state_tensor.grad.cpu().numpy())
                count += 1
        else:  # action
            if irl_system.sequence:
                for act_tensor in action_tensors:
                    if act_tensor.grad is not None:
                        gradients_sum += np.abs(act_tensor.grad.cpu().numpy())
                        count += 1
            else:
                if action_tensor.grad is not None:
                    gradients_sum += np.abs(action_tensor.grad.cpu().numpy())
                    count += 1
    
    # 平均を取る
    importance_scores = {}
    if count > 0:
        avg_gradients = gradients_sum / count
        for i in range(n_features):
            importance_scores[i] = float(avg_gradients[i])
            logger.info(f"特徴量 {i}: 重要度 = {importance_scores[i]:.6f}")
    
    return importance_scores


def visualize_importance(
    state_importance: Dict[int, float],
    action_importance: Dict[int, float],
    output_dir: Path,
    method_name: str = "Permutation"
):
    """特徴量重要度を可視化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 状態特徴量
    indices = sorted(state_importance.keys())
    values = [state_importance[i] for i in indices]
    names = [STATE_FEATURE_NAMES[i] for i in indices]
    
    ax1.barh(names, values, color='steelblue')
    ax1.set_xlabel(f'{method_name} Importance', fontsize=12)
    ax1.set_title('状態特徴量の重要度', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 行動特徴量
    indices = sorted(action_importance.keys())
    values = [action_importance[i] for i in indices]
    names = [ACTION_FEATURE_NAMES[i] for i in indices]
    
    ax2.barh(names, values, color='coral')
    ax2.set_xlabel(f'{method_name} Importance', fontsize=12)
    ax2.set_title('行動特徴量の重要度', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'feature_importance_{method_name.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"可視化保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='特徴量重要度分析')
    parser.add_argument('--model', type=str, required=True,
                        help='訓練済みモデルのパス (.pt)')
    parser.add_argument('--reviews', type=str, required=True,
                        help='レビューデータCSVファイル')
    parser.add_argument('--eval-start', type=str, required=True,
                        help='評価開始日 (YYYY-MM-DD)')
    parser.add_argument('--eval-end', type=str, required=True,
                        help='評価終了日 (YYYY-MM-DD)')
    parser.add_argument('--history-window', type=int, default=12,
                        help='履歴ウィンドウ（月）')
    parser.add_argument('--future-window-start', type=int, default=0,
                        help='将来窓開始（月）')
    parser.add_argument('--future-window-end', type=int, default=3,
                        help='将来窓終了（月）')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Permutation Importanceの繰り返し回数')
    parser.add_argument('--method', type=str, default='both',
                        choices=['permutation', 'gradient', 'both'],
                        help='重要度計算手法')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("特徴量重要度分析開始")
    logger.info(f"モデル: {args.model}")
    logger.info(f"データ: {args.reviews}")
    logger.info(f"評価期間: {args.eval_start} ～ {args.eval_end}")
    logger.info(f"手法: {args.method}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # 評価データ抽出
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    eval_trajectories = extract_cutoff_evaluation_trajectories(
        df=df,
        cutoff_date=eval_start,
        history_window_months=args.history_window,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_events=3
    )
    
    logger.info(f"評価サンプル数: {len(eval_trajectories)}")
    
    # モデル読み込み
    config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 128,  # 訓練時のhidden_dimに合わせる
        'sequence': True,
        'seq_len': 0  # 可変長
    }
    
    irl_system = RetentionIRLSystem(config)
    
    # モデル読み込み（直接state_dictとして保存されている場合）
    checkpoint = torch.load(args.model, map_location=irl_system.device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 通常の保存形式
        irl_system.network.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 直接state_dictとして保存されている場合
        irl_system.network.load_state_dict(checkpoint)
    
    irl_system.network.eval()
    logger.info(f"モデル読み込み完了: {args.model}")
    
    # ベースライン性能計算
    baseline_score, baseline_preds, true_labels = compute_baseline_performance(
        irl_system, eval_trajectories
    )
    logger.info(f"ベースライン AUC-ROC: {baseline_score:.4f}")
    
    results = {
        'baseline_auc': float(baseline_score),
        'n_samples': len(eval_trajectories),
        'methods': {}
    }
    
    # Permutation Importance
    if args.method in ['permutation', 'both']:
        state_perm_importance = permutation_importance(
            irl_system, eval_trajectories, baseline_score, 'state', args.n_repeats
        )
        action_perm_importance = permutation_importance(
            irl_system, eval_trajectories, baseline_score, 'action', args.n_repeats
        )
        
        results['methods']['permutation'] = {
            'state': state_perm_importance,
            'action': action_perm_importance
        }
        
        visualize_importance(
            state_perm_importance, action_perm_importance,
            output_dir, 'Permutation'
        )
    
    # Gradient-based Importance
    if args.method in ['gradient', 'both']:
        state_grad_importance = gradient_based_importance(
            irl_system, eval_trajectories, 'state'
        )
        action_grad_importance = gradient_based_importance(
            irl_system, eval_trajectories, 'action'
        )
        
        results['methods']['gradient'] = {
            'state': state_grad_importance,
            'action': action_grad_importance
        }
        
        visualize_importance(
            state_grad_importance, action_grad_importance,
            output_dir, 'Gradient'
        )
    
    # 結果を保存
    results_path = output_dir / 'feature_importance.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"結果保存: {results_path}")
    
    logger.info("=" * 80)
    logger.info("特徴量重要度分析完了")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

