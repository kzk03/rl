"""
ログ管理ユーティリティ

システム全体で統一されたログ出力を提供します。
構造化ログ、パフォーマンス監視、アラート・通知機能を含みます。
"""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger as loguru_logger


@dataclass
class PerformanceMetric:
    """パフォーマンスメトリクス"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AlertRule:
    """アラートルール"""
    name: str
    condition: Callable[[Any], bool]
    message: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.alert_rules: List[AlertRule] = []
        self.alert_handlers: List[Callable[[str, str, str], None]] = []
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """メトリクスを記録"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            
            # メトリクス履歴の上限管理（最新1000件まで）
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
        
        # アラートチェック
        self._check_alerts(metric)
    
    def add_alert_rule(self, rule: AlertRule):
        """アラートルールを追加"""
        self.alert_rules.append(rule)
    
    def add_alert_handler(self, handler: Callable[[str, str, str], None]):
        """アラートハンドラーを追加"""
        self.alert_handlers.append(handler)
    
    def _check_alerts(self, metric: PerformanceMetric):
        """アラートをチェック"""
        for rule in self.alert_rules:
            if rule.condition(metric):
                # クールダウン期間をチェック
                if (rule.last_triggered is None or 
                    datetime.now() - rule.last_triggered > timedelta(minutes=rule.cooldown_minutes)):
                    
                    rule.last_triggered = datetime.now()
                    
                    # アラートを発火
                    for handler in self.alert_handlers:
                        try:
                            handler(rule.name, rule.message, rule.severity)
                        except Exception as e:
                            loguru_logger.error(f"アラートハンドラーでエラー: {e}")
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """メトリクスを取得"""
        with self._lock:
            metrics = self.metrics.copy()
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def export_metrics(self, output_path: str):
        """メトリクスをファイルにエクスポート"""
        with self._lock:
            metrics_data = [metric.to_dict() for metric in self.metrics]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)


# グローバルパフォーマンス監視インスタンス
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """パフォーマンス監視インスタンスを取得"""
    return _performance_monitor


def get_logger(name: str, level: Optional[str] = None) -> loguru_logger:
    """
    ログインスタンスを取得
    
    Args:
        name: ログ名（通常は __name__ を使用）
        level: ログレベル（オプション）
        
    Returns:
        loguru.Logger: ログインスタンス
    """
    # 環境変数からログレベルを取得
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    
    # ログフォーマットを環境変数から取得
    log_format = os.getenv("LOG_FORMAT", "text")
    
    if log_format == "json":
        format_string = (
            "{"
            '"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"name": "' + name + '", '
            '"message": "{message}", '
            '"file": "{file}", '
            '"line": {line}'
            "}"
        )
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # 既存のハンドラーを削除
    loguru_logger.remove()
    
    # 新しいハンドラーを追加
    loguru_logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=log_format != "json",
        backtrace=True,
        diagnose=True,
    )
    
    # ファイル出力も追加（オプション）
    log_file = os.getenv("LOG_FILE")
    if log_file:
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )
    
    return loguru_logger


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None
) -> None:
    """
    ログ設定を初期化
    
    Args:
        level: ログレベル
        format_type: フォーマットタイプ（text または json）
        log_file: ログファイルパス（オプション）
    """
    os.environ["LOG_LEVEL"] = level
    os.environ["LOG_FORMAT"] = format_type
    
    if log_file:
        os.environ["LOG_FILE"] = log_file
    
    # ルートロガーを初期化
    get_logger("gerrit_retention")


def performance_monitor(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    パフォーマンス監視デコレータ
    
    Args:
        metric_name: メトリクス名
        tags: タグ辞書
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                
                # メトリクスを記録
                metric_tags = (tags or {}).copy()
                metric_tags.update({
                    'function': func.__name__,
                    'success': str(success)
                })
                
                if error:
                    metric_tags['error'] = error
                
                _performance_monitor.record_metric(
                    name=metric_name,
                    value=execution_time,
                    unit="seconds",
                    tags=metric_tags
                )
                
                # ログ出力
                logger = get_logger(func.__module__)
                if success:
                    logger.info(f"{func.__name__} 実行完了: {execution_time:.3f}秒")
                else:
                    logger.error(f"{func.__name__} 実行エラー: {execution_time:.3f}秒, エラー: {error}")
            
            return result
        return wrapper
    return decorator


@contextmanager
def performance_context(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    パフォーマンス監視コンテキストマネージャー
    
    Args:
        metric_name: メトリクス名
        tags: タグ辞書
    """
    start_time = time.time()
    
    try:
        yield
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        
        # メトリクスを記録
        metric_tags = (tags or {}).copy()
        metric_tags.update({
            'success': str(success)
        })
        
        if error:
            metric_tags['error'] = error
        
        _performance_monitor.record_metric(
            name=metric_name,
            value=execution_time,
            unit="seconds",
            tags=metric_tags
        )


def log_structured(logger_instance, level: str, message: str, **kwargs):
    """
    構造化ログ出力
    
    Args:
        logger_instance: ロガーインスタンス
        level: ログレベル
        message: メッセージ
        **kwargs: 追加のフィールド
    """
    # 構造化データを準備
    log_data = {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    # JSON形式でログ出力
    if os.getenv("LOG_FORMAT") == "json":
        log_message = json.dumps(log_data, ensure_ascii=False)
    else:
        # テキスト形式の場合は追加情報を含める
        extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        log_message = f"{message} [{extra_info}]" if extra_info else message
    
    # レベル別にログ出力
    if level.upper() == "DEBUG":
        logger_instance.debug(log_message)
    elif level.upper() == "INFO":
        logger_instance.info(log_message)
    elif level.upper() == "WARNING":
        logger_instance.warning(log_message)
    elif level.upper() == "ERROR":
        logger_instance.error(log_message)
    elif level.upper() == "CRITICAL":
        logger_instance.critical(log_message)


def setup_default_alerts():
    """デフォルトのアラートルールを設定"""
    monitor = get_performance_monitor()
    
    # 実行時間アラート
    monitor.add_alert_rule(AlertRule(
        name="slow_execution",
        condition=lambda metric: metric.name == "execution_time" and metric.value > 30.0,
        message="実行時間が30秒を超えました",
        severity="warning",
        cooldown_minutes=5
    ))
    
    # エラー率アラート
    monitor.add_alert_rule(AlertRule(
        name="high_error_rate",
        condition=lambda metric: (
            metric.name == "error_rate" and metric.value > 0.1
        ),
        message="エラー率が10%を超えました",
        severity="error",
        cooldown_minutes=10
    ))
    
    # メモリ使用量アラート
    monitor.add_alert_rule(AlertRule(
        name="high_memory_usage",
        condition=lambda metric: (
            metric.name == "memory_usage" and metric.value > 0.8
        ),
        message="メモリ使用量が80%を超えました",
        severity="warning",
        cooldown_minutes=5
    ))


def setup_file_alert_handler(log_file: str):
    """ファイルアラートハンドラーを設定"""
    def file_alert_handler(rule_name: str, message: str, severity: str):
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'rule_name': rule_name,
            'message': message,
            'severity': severity
        }
        
        # アラートログファイルに出力
        alert_log_file = Path(log_file).parent / "alerts.log"
        with open(alert_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')
    
    get_performance_monitor().add_alert_handler(file_alert_handler)


def setup_console_alert_handler():
    """コンソールアラートハンドラーを設定"""
    def console_alert_handler(rule_name: str, message: str, severity: str):
        alert_logger = get_logger("alerts")
        
        if severity == "critical":
            alert_logger.critical(f"🚨 ALERT [{rule_name}]: {message}")
        elif severity == "error":
            alert_logger.error(f"❌ ALERT [{rule_name}]: {message}")
        elif severity == "warning":
            alert_logger.warning(f"⚠️  ALERT [{rule_name}]: {message}")
        else:
            alert_logger.info(f"ℹ️  ALERT [{rule_name}]: {message}")
    
    get_performance_monitor().add_alert_handler(console_alert_handler)


def record_system_metrics():
    """システムメトリクスを記録"""
    import psutil
    
    monitor = get_performance_monitor()
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    monitor.record_metric("cpu_usage", cpu_percent / 100.0, "ratio")
    
    # メモリ使用率
    memory = psutil.virtual_memory()
    monitor.record_metric("memory_usage", memory.percent / 100.0, "ratio")
    
    # ディスク使用率
    disk = psutil.disk_usage('/')
    monitor.record_metric("disk_usage", disk.percent / 100.0, "ratio")


def start_system_monitoring(interval_seconds: int = 60):
    """システム監視を開始"""
    def monitoring_loop():
        while True:
            try:
                record_system_metrics()
            except Exception as e:
                logger = get_logger("system_monitor")
                logger.error(f"システムメトリクス記録エラー: {e}")
            
            time.sleep(interval_seconds)
    
    # バックグラウンドスレッドで監視開始
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    
    logger = get_logger("system_monitor")
    logger.info(f"システム監視を開始しました（間隔: {interval_seconds}秒）")


# デフォルトロガー
logger = get_logger(__name__)

# デフォルト設定を初期化
setup_default_alerts()
setup_console_alert_handler()