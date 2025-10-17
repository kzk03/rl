"""
パス類似度計算モジュール - 階層的コサイン類似度

Jaccard係数の代替として、ファイルパスの階層構造を考慮した類似度計算
"""

import numpy as np
from collections import Counter
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def hierarchical_cosine_similarity(paths_A: List[str],
                                   paths_B: List[str],
                                   depth_weights: Optional[List[float]] = None) -> float:
    """
    階層的重み付きコサイン類似度

    ファイルパスを階層ごとに分解し、深い階層（ファイル名）に高い重みを付与してコサイン類似度を計算

    例:
        paths_A = ['/src/core/auth.py', '/src/utils/helpers.py']
        paths_B = ['/src/core/user.py', '/src/utils/helpers.py']

        トークン化:
        A: ['src'(0.33), 'core'(0.67), 'auth.py'(1.0), 'src'(0.33), 'utils'(0.67), 'helpers.py'(1.0)]
        B: ['src'(0.33), 'core'(0.67), 'user.py'(1.0), 'src'(0.33), 'utils'(0.67), 'helpers.py'(1.0)]

        → コサイン類似度 ≈ 0.75 (Jaccardなら0.33)

    Args:
        paths_A: レビュアーAが変更したパスのリスト
        paths_B: レビュアーBが変更したパスのリスト
        depth_weights: 階層ごとの重み（Noneの場合は線形増加: [0.33, 0.67, 1.0]）

    Returns:
        階層的コサイン類似度（0.0-1.0）
    """
    if not paths_A or not paths_B:
        return 0.0

    # デフォルト重み: 深い階層ほど重要（線形増加）
    if depth_weights is None:
        max_depth = max(
            max(len(p.split('/')) for p in paths_A),
            max(len(p.split('/')) for p in paths_B)
        )
        # [1/max_depth, 2/max_depth, ..., max_depth/max_depth]
        depth_weights = [(i + 1) / max_depth for i in range(max_depth)]

    # トークン化と重み付き頻度計算
    def weighted_tokens(paths):
        weighted_counter = Counter()
        for path in paths:
            # パスの区切り文字を統一（\ → /）
            normalized_path = path.replace('\\', '/')
            tokens = normalized_path.split('/')

            for depth, token in enumerate(tokens):
                if not token:  # 空文字列をスキップ
                    continue
                weight = depth_weights[min(depth, len(depth_weights) - 1)]
                weighted_counter[token] += weight

        return weighted_counter

    counter_A = weighted_tokens(paths_A)
    counter_B = weighted_tokens(paths_B)

    # 共通語彙
    vocab = sorted(set(counter_A.keys()) | set(counter_B.keys()))

    if not vocab:
        return 0.0

    # ベクトル化
    vec_A = np.array([counter_A.get(token, 0) for token in vocab])
    vec_B = np.array([counter_B.get(token, 0) for token in vocab])

    # コサイン類似度
    norm_A = np.linalg.norm(vec_A)
    norm_B = np.linalg.norm(vec_B)

    if norm_A == 0 or norm_B == 0:
        return 0.0

    return float(np.dot(vec_A, vec_B) / (norm_A * norm_B))


# 使用例
if __name__ == "__main__":
    # テストデータ
    paths_A = [
        '/src/core/auth.py',
        '/src/core/user.py',
        '/src/utils/helpers.py'
    ]

    paths_B = [
        '/src/core/authentication.py',
        '/src/core/user.py',
        '/tests/test_auth.py'
    ]

    # Jaccard係数（比較用）
    set_A = set(paths_A)
    set_B = set(paths_B)
    jaccard = len(set_A & set_B) / len(set_A | set_B)

    # 階層的コサイン類似度
    hierarchical = hierarchical_cosine_similarity(paths_A, paths_B)

    print("=== パス類似度比較 ===")
    print(f"Paths A: {paths_A}")
    print(f"Paths B: {paths_B}")
    print()
    print(f"Jaccard係数:            {jaccard:.4f}")
    print(f"階層的コサイン類似度:    {hierarchical:.4f}")
    print()
    print(f"改善率: {(hierarchical - jaccard) / jaccard * 100:.1f}%")
