#!/usr/bin/env python3
"""
Gerrit開発者定着予測システム - ワーカースクリプト

並列処理とバックグラウンドタスクを処理するワーカープロセス
"""

import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, Dict, Optional

# プロジェクトパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gerrit_retention.data_integration.gerrit_client import GerritClient
from gerrit_retention.prediction.retention_predictor import RetentionPredictor
from gerrit_retention.prediction.stress_analyzer import StressAnalyzer
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger, performance_monitor

logger = get_logger(__name__)


class TaskQueue:
    """タスクキュー管理"""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = Queue(maxsize=maxsize)
        self.running = True
    
    def put_task(self, task_type: str, task_data: Dict[str, Any], priority: int = 0):
        """タスクをキューに追加"""
        task = {
            'type': task_type,
            'data': task_data,
            'priority': priority,
            'timestamp': time.time()
        }
        
        try:
            self.queue.put(task, timeout=5)
            logger.info(f"タスクをキューに追加: {task_type}")
        except Exception as e:
            logger.error(f"タスクキューへの追加に失敗: {e}")
    
    def get_task(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """タスクをキューから取得"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def task_done(self):
        """タスク完了を通知"""
        self.queue.task_done()
    
    def stop(self):
        """キューを停止"""
        self.running = False


class Worker:
    """ワーカークラス"""
    
    def __init__(self, worker_id: str, config_manager):
        self.worker_id = worker_id
        self.config_manager = config_manager
        self.task_queue = TaskQueue()
        self.running = True
        
        # コンポーネントの初期化
        self.gerrit_client = None
        self.retention_predictor = None
        self.stress_analyzer = None
        
        self._initialize_components()
        
        logger.info(f"ワーカー {worker_id} を初期化しました")
    
    def _initialize_components(self):
        """コンポーネントを初期化"""
        try:
            # Gerritクライアント
            gerrit_config = self.config_manager.get('gerrit', {})
            if gerrit_config:
                self.gerrit_client = GerritClient(gerrit_config)
            
            # 定着予測器
            retention_config = self.config_manager.get('retention_prediction', {})
            if retention_config:
                self.retention_predictor = RetentionPredictor(retention_config)
            
            # ストレス分析器
            stress_config = self.config_manager.get('stress_analysis', {})
            if stress_config:
                self.stress_analyzer = StressAnalyzer(stress_config)
            
            logger.info("コンポーネントの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"コンポーネント初期化エラー: {e}")
    
    @performance_monitor("task_execution")
    def process_task(self, task: Dict[str, Any]) -> bool:
        """タスクを処理"""
        task_type = task['type']
        task_data = task['data']
        
        try:
            if task_type == 'data_extraction':
                return self._process_data_extraction(task_data)
            elif task_type == 'retention_prediction':
                return self._process_retention_prediction(task_data)
            elif task_type == 'stress_analysis':
                return self._process_stress_analysis(task_data)
            elif task_type == 'model_training':
                return self._process_model_training(task_data)
            elif task_type == 'batch_prediction':
                return self._process_batch_prediction(task_data)
            else:
                logger.warning(f"未知のタスクタイプ: {task_type}")
                return False
                
        except Exception as e:
            logger.error(f"タスク処理エラー ({task_type}): {e}")
            return False
    
    def _process_data_extraction(self, task_data: Dict[str, Any]) -> bool:
        """データ抽出タスクを処理"""
        if not self.gerrit_client:
            logger.error("Gerritクライアントが初期化されていません")
            return False
        
        project = task_data.get('project')
        time_range = task_data.get('time_range')
        
        logger.info(f"データ抽出開始: プロジェクト={project}")
        
        # データ抽出の実行
        review_data = self.gerrit_client.extract_review_data(project, time_range)
        
        # 結果の保存
        output_path = task_data.get('output_path', f'data/raw/{project}_reviews.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"データ抽出完了: {len(review_data)}件のレビューを抽出")
        return True
    
    def _process_retention_prediction(self, task_data: Dict[str, Any]) -> bool:
        """定着予測タスクを処理"""
        if not self.retention_predictor:
            logger.error("定着予測器が初期化されていません")
            return False
        
        developer_data = task_data.get('developer_data')
        context_data = task_data.get('context_data', {})
        
        logger.info(f"定着予測開始: 開発者={developer_data.get('email', 'unknown')}")
        
        # 定着確率を予測
        retention_prob = self.retention_predictor.predict_retention_probability(
            developer_data, context_data
        )
        
        # 結果を保存
        result = {
            'developer_email': developer_data.get('email'),
            'retention_probability': retention_prob,
            'timestamp': time.time()
        }
        
        # データベースまたはファイルに保存
        self._save_prediction_result('retention', result)
        
        logger.info(f"定着予測完了: 確率={retention_prob:.3f}")
        return True
    
    def _process_stress_analysis(self, task_data: Dict[str, Any]) -> bool:
        """ストレス分析タスクを処理"""
        if not self.stress_analyzer:
            logger.error("ストレス分析器が初期化されていません")
            return False
        
        developer_data = task_data.get('developer_data')
        context_data = task_data.get('context_data', {})
        
        logger.info(f"ストレス分析開始: 開発者={developer_data.get('email', 'unknown')}")
        
        # ストレス指標を計算
        stress_indicators = self.stress_analyzer.calculate_stress_indicators(
            developer_data, context_data
        )
        
        # 沸点を予測
        boiling_point = self.stress_analyzer.predict_boiling_point(
            developer_data, context_data
        )
        
        # 結果を保存
        result = {
            'developer_email': developer_data.get('email'),
            'stress_indicators': stress_indicators,
            'boiling_point': boiling_point,
            'timestamp': time.time()
        }
        
        self._save_prediction_result('stress', result)
        
        logger.info(f"ストレス分析完了: 総合ストレス={stress_indicators.get('total_stress', 0):.3f}")
        return True
    
    def _process_model_training(self, task_data: Dict[str, Any]) -> bool:
        """モデル訓練タスクを処理"""
        model_type = task_data.get('model_type')
        training_data_path = task_data.get('training_data_path')
        
        logger.info(f"モデル訓練開始: タイプ={model_type}")
        
        # 訓練スクリプトを実行
        if model_type == 'retention':
            from training.retention_training.train_retention_model import (
                main as train_retention,
            )
            success = train_retention(training_data_path)
        elif model_type == 'stress':
            from training.stress_training.train_stress_model import main as train_stress
            success = train_stress(training_data_path)
        else:
            logger.error(f"未知のモデルタイプ: {model_type}")
            return False
        
        logger.info(f"モデル訓練完了: 成功={success}")
        return success
    
    def _process_batch_prediction(self, task_data: Dict[str, Any]) -> bool:
        """バッチ予測タスクを処理"""
        developers_data = task_data.get('developers_data', [])
        prediction_type = task_data.get('prediction_type', 'retention')
        
        logger.info(f"バッチ予測開始: {len(developers_data)}人の開発者")
        
        results = []
        
        for developer_data in developers_data:
            try:
                if prediction_type == 'retention':
                    result = self._process_retention_prediction({
                        'developer_data': developer_data,
                        'context_data': task_data.get('context_data', {})
                    })
                elif prediction_type == 'stress':
                    result = self._process_stress_analysis({
                        'developer_data': developer_data,
                        'context_data': task_data.get('context_data', {})
                    })
                else:
                    logger.warning(f"未知の予測タイプ: {prediction_type}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"開発者 {developer_data.get('email')} の予測エラー: {e}")
        
        success_rate = sum(results) / len(results) if results else 0
        logger.info(f"バッチ予測完了: 成功率={success_rate:.2%}")
        
        return success_rate > 0.8  # 80%以上成功で成功とみなす
    
    def _save_prediction_result(self, prediction_type: str, result: Dict[str, Any]):
        """予測結果を保存"""
        # 実装: データベースまたはファイルに保存
        output_dir = f"outputs/predictions/{prediction_type}"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        output_file = f"{output_dir}/{result.get('developer_email', 'unknown')}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def run(self):
        """ワーカーを実行"""
        logger.info(f"ワーカー {self.worker_id} を開始します")
        
        while self.running:
            task = self.task_queue.get_task()
            
            if task is None:
                continue
            
            try:
                success = self.process_task(task)
                
                if success:
                    logger.debug(f"タスク処理成功: {task['type']}")
                else:
                    logger.warning(f"タスク処理失敗: {task['type']}")
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"タスク処理中の予期しないエラー: {e}")
                self.task_queue.task_done()
        
        logger.info(f"ワーカー {self.worker_id} を停止しました")
    
    def stop(self):
        """ワーカーを停止"""
        self.running = False
        self.task_queue.stop()


class WorkerManager:
    """ワーカー管理クラス"""
    
    def __init__(self, num_workers: int = None):
        self.config_manager = get_config_manager()
        self.num_workers = num_workers or self.config_manager.get('performance.max_workers', mp.cpu_count())
        self.workers = []
        self.running = True
        
        # シグナルハンドラーを設定
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"ワーカーマネージャーを初期化: {self.num_workers}個のワーカー")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"シグナル {signum} を受信しました。ワーカーを停止します...")
        self.stop()
    
    def start(self):
        """ワーカーを開始"""
        logger.info("ワーカーを開始します")
        
        # ワーカーを作成・開始
        for i in range(self.num_workers):
            worker_id = f"worker-{i+1}"
            worker = Worker(worker_id, self.config_manager)
            
            # ワーカーをスレッドで実行
            worker_thread = threading.Thread(target=worker.run, name=worker_id)
            worker_thread.daemon = True
            worker_thread.start()
            
            self.workers.append((worker, worker_thread))
        
        logger.info(f"{len(self.workers)}個のワーカーを開始しました")
        
        # メインループ
        try:
            while self.running:
                time.sleep(1)
                
                # ワーカーの健全性をチェック
                self._check_worker_health()
                
        except KeyboardInterrupt:
            logger.info("キーボード割り込みを受信しました")
        finally:
            self.stop()
    
    def _check_worker_health(self):
        """ワーカーの健全性をチェック"""
        for i, (worker, thread) in enumerate(self.workers):
            if not thread.is_alive():
                logger.warning(f"ワーカー {worker.worker_id} が停止しています。再起動します...")
                
                # 新しいワーカーを作成
                new_worker = Worker(worker.worker_id, self.config_manager)
                new_thread = threading.Thread(target=new_worker.run, name=worker.worker_id)
                new_thread.daemon = True
                new_thread.start()
                
                # リストを更新
                self.workers[i] = (new_worker, new_thread)
    
    def stop(self):
        """ワーカーを停止"""
        logger.info("ワーカーを停止します")
        self.running = False
        
        # 全ワーカーを停止
        for worker, thread in self.workers:
            worker.stop()
        
        # スレッドの終了を待機
        for worker, thread in self.workers:
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning(f"ワーカー {worker.worker_id} の停止がタイムアウトしました")
        
        logger.info("全ワーカーを停止しました")


def main():
    """メイン関数"""
    logger.info("Gerrit開発者定着予測システム - ワーカー開始")
    
    # 設定を読み込み
    config_manager = get_config_manager()
    
    # ワーカー数を設定から取得
    num_workers = config_manager.get('performance.max_workers', mp.cpu_count())
    
    # ワーカーマネージャーを作成・開始
    manager = WorkerManager(num_workers)
    
    try:
        manager.start()
    except Exception as e:
        logger.error(f"ワーカー実行エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())