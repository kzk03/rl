"""
多目的最適化システム

短期効率と長期定着のトレードオフを最適化し、
パレート最適解を探索するシステムです。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Objective:
    """最適化目的関数"""
    name: str
    function: Callable[[Dict[str, Any]], float]
    weight: float
    maximize: bool = True
    description: str = ""


@dataclass
class ParetoSolution:
    """パレート最適解"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    dominance_rank: int
    crowding_distance: float
    timestamp: datetime


class MultiObjectiveOptimizer:
    """
    多目的最適化システム
    
    短期効率と長期定着のトレードオフを最適化し、
    パレート最適解を探索します。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 最適化設定
        """
        self.config = config
        self.objectives = self._initialize_objectives()
        self.scaler = StandardScaler()
        
        # パレート解集合
        self.pareto_solutions: List[ParetoSolution] = []
        
        # 最適化履歴
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("多目的最適化システムを初期化しました")
    
    def _initialize_objectives(self) -> Dict[str, Objective]:
        """目的関数を初期化"""
        objectives = {
            # 短期効率目的
            "task_completion_rate": Objective(
                name="タスク完了率",
                function=self._calculate_task_completion_rate,
                weight=self.config.get('task_completion_weight', 0.3),
                maximize=True,
                description="短期的なタスク完了効率"
            ),   
         
            "review_quality": Objective(
                name="レビュー品質",
                function=self._calculate_review_quality,
                weight=self.config.get('review_quality_weight', 0.2),
                maximize=True,
                description="レビューの品質指標"
            ),
            
            # 長期定着目的
            "developer_retention": Objective(
                name="開発者定着率",
                function=self._calculate_developer_retention,
                weight=self.config.get('retention_weight', 0.4),
                maximize=True,
                description="長期的な開発者定着率"
            ),
            
            "stress_minimization": Objective(
                name="ストレス最小化",
                function=self._calculate_stress_minimization,
                weight=self.config.get('stress_weight', 0.3),
                maximize=True,
                description="開発者ストレスの最小化"
            ),
            
            "satisfaction_maximization": Objective(
                name="満足度最大化",
                function=self._calculate_satisfaction_maximization,
                weight=self.config.get('satisfaction_weight', 0.2),
                maximize=True,
                description="開発者満足度の最大化"
            ),
            
            # バランス目的
            "workload_balance": Objective(
                name="ワークロードバランス",
                function=self._calculate_workload_balance,
                weight=self.config.get('balance_weight', 0.15),
                maximize=True,
                description="チーム全体のワークロードバランス"
            )
        }
        
        return objectives
    
    def optimize_recommendation_strategy(self, 
                                       current_state: Dict[str, Any],
                                       constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """
        推薦戦略を多目的最適化
        
        Args:
            current_state: 現在の状態
            constraints: 制約条件
            
        Returns:
            Dict[str, float]: 最適化されたパラメータ
        """
        logger.info("推薦戦略の多目的最適化を開始")
        
        # 初期パラメータ設定
        initial_params = self._get_initial_parameters()
        
        # 制約条件の設定
        if constraints is None:
            constraints = self._get_default_constraints()
        
        # パレート最適解の探索
        pareto_solutions = self._find_pareto_optimal_solutions(
            current_state, initial_params, constraints
        )
        
        # 最適解の選択
        best_solution = self._select_best_solution(pareto_solutions, current_state)
        
        # 履歴に記録
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'state': current_state,
            'solution': best_solution,
            'pareto_size': len(pareto_solutions)
        })
        
        logger.info(f"最適化完了: パレート解数={len(pareto_solutions)}")
        
        return best_solution.parameters    
   
 def _find_pareto_optimal_solutions(self, 
                                     state: Dict[str, Any],
                                     initial_params: Dict[str, float],
                                     constraints: Dict[str, Any]) -> List[ParetoSolution]:
        """パレート最適解を探索"""
        solutions = []
        
        # 複数の初期点から最適化を実行
        num_runs = self.config.get('optimization_runs', 10)
        
        for run in range(num_runs):
            # ランダムな初期点を生成
            params = self._generate_random_parameters(initial_params)
            
            # 重み付きスカラー化による最適化
            for weight_combination in self._generate_weight_combinations():
                try:
                    result = self._optimize_weighted_sum(
                        params, state, weight_combination, constraints
                    )
                    
                    if result.success:
                        solution = self._create_pareto_solution(result, state)
                        solutions.append(solution)
                        
                except Exception as e:
                    logger.warning(f"最適化実行中にエラー: {e}")
                    continue
        
        # パレート最適解をフィルタリング
        pareto_solutions = self._filter_pareto_optimal(solutions)
        
        # 支配ランクとクラウディング距離を計算
        pareto_solutions = self._calculate_dominance_and_crowding(pareto_solutions)
        
        return pareto_solutions
    
    def _optimize_weighted_sum(self, 
                             initial_params: Dict[str, float],
                             state: Dict[str, Any],
                             weights: Dict[str, float],
                             constraints: Dict[str, Any]) -> Any:
        """重み付きスカラー化による最適化"""
        
        def objective_function(x):
            params = self._array_to_params(x)
            total_objective = 0.0
            
            for obj_name, objective in self.objectives.items():
                value = objective.function({**state, 'params': params})
                weight = weights.get(obj_name, objective.weight)
                
                if objective.maximize:
                    total_objective += weight * value
                else:
                    total_objective -= weight * value
            
            return -total_objective  # 最小化問題として解く
        
        # パラメータを配列に変換
        x0 = self._params_to_array(initial_params)
        
        # 制約条件を設定
        bounds = self._get_parameter_bounds(constraints)
        
        # 最適化実行
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config.get('max_iterations', 100)}
        )
        
        return result  
  
    def _filter_pareto_optimal(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """パレート最適解をフィルタリング"""
        if not solutions:
            return []
        
        pareto_solutions = []
        
        for i, solution_i in enumerate(solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(solutions):
                if i != j and self._dominates(solution_j, solution_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution_i)
        
        return pareto_solutions
    
    def _dominates(self, solution_a: ParetoSolution, solution_b: ParetoSolution) -> bool:
        """解Aが解Bを支配するかを判定"""
        better_in_at_least_one = False
        
        for obj_name in solution_a.objectives:
            obj_a = solution_a.objectives[obj_name]
            obj_b = solution_b.objectives[obj_name]
            
            objective = self.objectives[obj_name]
            
            if objective.maximize:
                if obj_a < obj_b:
                    return False
                elif obj_a > obj_b:
                    better_in_at_least_one = True
            else:
                if obj_a > obj_b:
                    return False
                elif obj_a < obj_b:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _calculate_dominance_and_crowding(self, 
                                        solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """支配ランクとクラウディング距離を計算"""
        # 支配ランクの計算
        for i, solution in enumerate(solutions):
            rank = 0
            for other_solution in solutions:
                if self._dominates(other_solution, solution):
                    rank += 1
            solution.dominance_rank = rank
        
        # クラウディング距離の計算
        if len(solutions) <= 2:
            for solution in solutions:
                solution.crowding_distance = float('inf')
        else:
            # 各目的関数について距離を計算
            for obj_name in self.objectives:
                solutions.sort(key=lambda s: s.objectives[obj_name])
                
                solutions[0].crowding_distance = float('inf')
                solutions[-1].crowding_distance = float('inf')
                
                obj_range = (solutions[-1].objectives[obj_name] - 
                           solutions[0].objectives[obj_name])
                
                if obj_range > 0:
                    for i in range(1, len(solutions) - 1):
                        distance = (solutions[i+1].objectives[obj_name] - 
                                  solutions[i-1].objectives[obj_name]) / obj_range
                        solutions[i].crowding_distance += distance
        
        return solutions    

    def _select_best_solution(self, 
                            pareto_solutions: List[ParetoSolution],
                            current_state: Dict[str, Any]) -> ParetoSolution:
        """最適解を選択"""
        if not pareto_solutions:
            # デフォルト解を返す
            return self._create_default_solution()
        
        # 支配ランクが最小の解を選択
        min_rank = min(sol.dominance_rank for sol in pareto_solutions)
        best_rank_solutions = [sol for sol in pareto_solutions 
                              if sol.dominance_rank == min_rank]
        
        # 同じランクの中でクラウディング距離が最大の解を選択
        best_solution = max(best_rank_solutions, 
                           key=lambda s: s.crowding_distance)
        
        return best_solution
    
    # 目的関数の実装
    def _calculate_task_completion_rate(self, context: Dict[str, Any]) -> float:
        """タスク完了率を計算"""
        params = context.get('params', {})
        efficiency_factor = params.get('efficiency_factor', 0.5)
        workload_factor = params.get('workload_factor', 0.5)
        
        # 簡略化された計算（実際の実装では履歴データを使用）
        base_rate = context.get('historical_completion_rate', 0.7)
        adjusted_rate = base_rate * (1 + efficiency_factor * 0.2) * (1 - workload_factor * 0.1)
        
        return np.clip(adjusted_rate, 0.0, 1.0)
    
    def _calculate_review_quality(self, context: Dict[str, Any]) -> float:
        """レビュー品質を計算"""
        params = context.get('params', {})
        expertise_match = params.get('expertise_match', 0.5)
        time_allocation = params.get('time_allocation', 0.5)
        
        base_quality = context.get('historical_review_quality', 0.6)
        adjusted_quality = base_quality * (1 + expertise_match * 0.3) * (1 + time_allocation * 0.2)
        
        return np.clip(adjusted_quality, 0.0, 1.0)
    
    def _calculate_developer_retention(self, context: Dict[str, Any]) -> float:
        """開発者定着率を計算"""
        params = context.get('params', {})
        satisfaction_factor = params.get('satisfaction_factor', 0.5)
        stress_factor = params.get('stress_factor', 0.5)
        
        base_retention = context.get('historical_retention_rate', 0.8)
        adjusted_retention = base_retention * (1 + satisfaction_factor * 0.2) * (1 - stress_factor * 0.3)
        
        return np.clip(adjusted_retention, 0.0, 1.0)
    
    def _calculate_stress_minimization(self, context: Dict[str, Any]) -> float:
        """ストレス最小化を計算"""
        params = context.get('params', {})
        workload_balance = params.get('workload_balance', 0.5)
        task_match = params.get('task_match', 0.5)
        
        stress_level = context.get('current_stress_level', 0.5)
        stress_reduction = workload_balance * 0.3 + task_match * 0.4
        
        return 1.0 - np.clip(stress_level - stress_reduction, 0.0, 1.0)   
 
    def _calculate_satisfaction_maximization(self, context: Dict[str, Any]) -> float:
        """満足度最大化を計算"""
        params = context.get('params', {})
        autonomy_factor = params.get('autonomy_factor', 0.5)
        growth_opportunity = params.get('growth_opportunity', 0.5)
        
        base_satisfaction = context.get('current_satisfaction', 0.6)
        satisfaction_boost = autonomy_factor * 0.2 + growth_opportunity * 0.3
        
        return np.clip(base_satisfaction + satisfaction_boost, 0.0, 1.0)
    
    def _calculate_workload_balance(self, context: Dict[str, Any]) -> float:
        """ワークロードバランスを計算"""
        params = context.get('params', {})
        distribution_factor = params.get('distribution_factor', 0.5)
        
        workload_variance = context.get('team_workload_variance', 0.3)
        balanced_variance = workload_variance * (1 - distribution_factor * 0.5)
        
        return 1.0 - np.clip(balanced_variance, 0.0, 1.0)
    
    # ユーティリティメソッド
    def _get_initial_parameters(self) -> Dict[str, float]:
        """初期パラメータを取得"""
        return {
            'efficiency_factor': 0.5,
            'workload_factor': 0.5,
            'expertise_match': 0.5,
            'time_allocation': 0.5,
            'satisfaction_factor': 0.5,
            'stress_factor': 0.5,
            'workload_balance': 0.5,
            'task_match': 0.5,
            'autonomy_factor': 0.5,
            'growth_opportunity': 0.5,
            'distribution_factor': 0.5
        }
    
    def _generate_random_parameters(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """ランダムなパラメータを生成"""
        params = {}
        for key, base_value in base_params.items():
            # ±20%の範囲でランダム化
            noise = np.random.uniform(-0.2, 0.2)
            params[key] = np.clip(base_value + noise, 0.0, 1.0)
        return params
    
    def _generate_weight_combinations(self) -> List[Dict[str, float]]:
        """重み組み合わせを生成"""
        combinations = []
        
        # 均等重み
        equal_weight = 1.0 / len(self.objectives)
        combinations.append({name: equal_weight for name in self.objectives})
        
        # 各目的関数を重視した組み合わせ
        for obj_name in self.objectives:
            weights = {name: 0.1 for name in self.objectives}
            weights[obj_name] = 0.7
            combinations.append(weights)
        
        # ランダム重み
        for _ in range(5):
            weights = np.random.dirichlet(np.ones(len(self.objectives)))
            weight_dict = {name: weight for name, weight in zip(self.objectives.keys(), weights)}
            combinations.append(weight_dict)
        
        return combinations    
  
  def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """パラメータを配列に変換"""
        return np.array(list(params.values()))
    
    def _array_to_params(self, array: np.ndarray) -> Dict[str, float]:
        """配列をパラメータに変換"""
        param_names = list(self._get_initial_parameters().keys())
        return {name: value for name, value in zip(param_names, array)}
    
    def _get_parameter_bounds(self, constraints: Dict[str, Any]) -> List[Tuple[float, float]]:
        """パラメータの境界を取得"""
        bounds = []
        for param_name in self._get_initial_parameters().keys():
            lower = constraints.get(f'{param_name}_min', 0.0)
            upper = constraints.get(f'{param_name}_max', 1.0)
            bounds.append((lower, upper))
        return bounds
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """デフォルト制約条件を取得"""
        return {param + '_min': 0.0 for param in self._get_initial_parameters().keys()} | \
               {param + '_max': 1.0 for param in self._get_initial_parameters().keys()}
    
    def _create_pareto_solution(self, result: Any, state: Dict[str, Any]) -> ParetoSolution:
        """パレート解を作成"""
        params = self._array_to_params(result.x)
        
        # 各目的関数の値を計算
        objectives = {}
        for obj_name, objective in self.objectives.items():
            objectives[obj_name] = objective.function({**state, 'params': params})
        
        return ParetoSolution(
            parameters=params,
            objectives=objectives,
            dominance_rank=0,  # 後で計算
            crowding_distance=0.0,  # 後で計算
            timestamp=datetime.now()
        )
    
    def _create_default_solution(self) -> ParetoSolution:
        """デフォルト解を作成"""
        params = self._get_initial_parameters()
        objectives = {name: 0.5 for name in self.objectives.keys()}
        
        return ParetoSolution(
            parameters=params,
            objectives=objectives,
            dominance_rank=0,
            crowding_distance=0.0,
            timestamp=datetime.now()
        )
    
    def get_pareto_front(self) -> List[ParetoSolution]:
        """現在のパレートフロントを取得"""
        return [sol for sol in self.pareto_solutions if sol.dominance_rank == 0]
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """トレードオフ分析を実行"""
        if not self.pareto_solutions:
            return {}
        
        pareto_front = self.get_pareto_front()
        
        if not pareto_front:
            return {}
        
        analysis = {}
        
        # 各目的関数の範囲
        for obj_name in self.objectives:
            values = [sol.objectives[obj_name] for sol in pareto_front]
            analysis[f'{obj_name}_range'] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # 相関分析
        obj_names = list(self.objectives.keys())
        correlations = {}
        for i, obj1 in enumerate(obj_names):
            for j, obj2 in enumerate(obj_names[i+1:], i+1):
                values1 = [sol.objectives[obj1] for sol in pareto_front]
                values2 = [sol.objectives[obj2] for sol in pareto_front]
                correlation = np.corrcoef(values1, values2)[0, 1]
                correlations[f'{obj1}_vs_{obj2}'] = correlation
        
        analysis['correlations'] = correlations
        analysis['pareto_front_size'] = len(pareto_front)
        
        return analysis