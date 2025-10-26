#!/usr/bin/env python3
"""
拡張IRL完全クロス評価スクリプト（Python版）

訓練ラベル: 0-1m, 0-3m, 0-6m, 0-9m, 0-12m (5個)
評価期間: 0-3m, 3-6m, 6-9m, 9-12m (4個)
合計: 5訓練 × 4評価 = 20評価
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnhancedCrossEvaluationRunner:
    """拡張IRLクロス評価実行クラス"""
    
    def __init__(
        self,
        reviews_file: str,
        train_start: str,
        train_end: str,
        eval_start: str,
        eval_end: str,
        history_window: int,
        epochs: int,
        output_base: str,
        dry_run: bool = False
    ):
        self.reviews_file = reviews_file
        self.train_start = train_start
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.history_window = history_window
        self.epochs = epochs
        self.output_base = Path(output_base)
        self.dry_run = dry_run
        
        # ログディレクトリ
        self.log_dir = self.output_base / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # メインログファイル
        self.main_log = self.log_dir / 'main.log'
        
        # ファイルハンドラを追加
        file_handler = logging.FileHandler(self.main_log, mode='a')
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(file_handler)
        
        # 訓練ラベルと評価期間の定義
        self.train_labels = [1, 3, 6, 9, 12]  # 0-1m, 0-3m, 0-6m, 0-9m, 0-12m
        self.eval_windows = [
            (0, 3),   # 0-3m
            (3, 6),   # 3-6m
            (6, 9),   # 6-9m
            (9, 12),  # 9-12m
        ]
        
        # 統計情報
        self.stats = {
            'total_trainings': len(self.train_labels),
            'total_evaluations': len(self.train_labels) * len(self.eval_windows),
            'completed_trainings': 0,
            'completed_evaluations': 0,
            'failed_trainings': 0,
            'failed_evaluations': 0,
            'start_time': None,
            'end_time': None
        }
    
    def log_message(self, message: str, level: str = 'info'):
        """ログメッセージを出力"""
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
    
    def run_training(self, label: int) -> Tuple[bool, Path]:
        """訓練を実行
        
        Args:
            label: 訓練ラベル（月数）
            
        Returns:
            (成功フラグ, モデルパス)
        """
        output_dir = self.output_base / f"train_0-{label}m"
        log_file = self.log_dir / f"train_0-{label}m.log"
        model_path = output_dir / "enhanced_irl_model.pt"
        
        # 既に訓練済みならスキップ
        if model_path.exists():
            self.log_message(f"⏭️  訓練スキップ（既存）: 0-{label}m")
            return True, model_path
        
        self.log_message(f"拡張IRL訓練開始: 0-{label}m")
        
        cmd = [
            'uv', 'run', 'python',
            'scripts/training/irl/train_enhanced_irl_per_timestep_labels.py',
            '--reviews', self.reviews_file,
            '--train-start', self.train_start,
            '--train-end', self.train_end,
            '--eval-start', self.eval_start,
            '--eval-end', self.eval_end,
            '--history-window', str(self.history_window),
            '--future-window-start', '0',
            '--future-window-end', str(label),
            '--seq-len', '0',
            '--use-full-sequence',
            '--epochs', str(self.epochs),
            '--min-history-events', '1',
            '--output', str(output_dir)
        ]
        
        if self.dry_run:
            self.log_message(f"[DRY RUN] コマンド: {' '.join(cmd)}")
            return True, model_path
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.log_message(f"✓ 拡張IRL訓練完了: 0-{label}m")
            self.stats['completed_trainings'] += 1
            return True, model_path
            
        except subprocess.CalledProcessError as e:
            self.log_message(f"✗ エラー発生: 0-{label}m (exit code: {e.returncode})", 'error')
            self.log_message(f"詳細: {log_file}", 'error')
            self.stats['failed_trainings'] += 1
            return False, model_path
    
    def run_evaluation(
        self,
        train_label: int,
        eval_start: int,
        eval_end: int,
        model_path: Path
    ) -> bool:
        """評価を実行
        
        Args:
            train_label: 訓練ラベル（月数）
            eval_start: 評価開始月
            eval_end: 評価終了月
            model_path: モデルパス
            
        Returns:
            成功フラグ
        """
        eval_output_dir = self.output_base / f"train_0-{train_label}m" / f"eval_{eval_start}-{eval_end}m"
        log_file = self.log_dir / f"train_0-{train_label}m_eval_{eval_start}-{eval_end}m.log"
        metrics_file = eval_output_dir / "metrics.json"
        
        # 既に評価済みならスキップ
        if metrics_file.exists():
            self.log_message(f"  ⏭️  評価スキップ（既存）: {eval_start}-{eval_end}m (訓練: 0-{train_label}m)")
            return True
        
        self.log_message(f"  評価: {eval_start}-{eval_end}m (拡張IRL訓練: 0-{train_label}m)")
        
        cmd = [
            'uv', 'run', 'python',
            'scripts/training/irl/train_enhanced_irl_per_timestep_labels.py',
            '--model', str(model_path),
            '--reviews', self.reviews_file,
            '--train-start', self.train_start,
            '--train-end', self.train_end,
            '--eval-start', self.eval_start,
            '--eval-end', self.eval_end,
            '--history-window', str(self.history_window),
            '--future-window-start', str(eval_start),
            '--future-window-end', str(eval_end),
            '--seq-len', '0',
            '--use-full-sequence',
            '--epochs', '0',  # 評価のみ
            '--min-history-events', '1',
            '--output', str(eval_output_dir)
        ]
        
        if self.dry_run:
            self.log_message(f"[DRY RUN] コマンド: {' '.join(cmd)}")
            return True
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.log_message(f"  ✓ 評価完了: {eval_start}-{eval_end}m")
            self.stats['completed_evaluations'] += 1
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_message(f"  ✗ 評価エラー: {eval_start}-{eval_end}m (exit code: {e.returncode})", 'error')
            self.log_message(f"詳細: {log_file}", 'error')
            self.stats['failed_evaluations'] += 1
            return False
    
    def print_header(self):
        """ヘッダーを表示"""
        header = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
拡張IRL完全クロス評価
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.log_message(header)
        self.log_message(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("")
        self.log_message(f"訓練ラベル: {len(self.train_labels)}個 (0-1m, 0-3m, 0-6m, 0-9m, 0-12m)")
        self.log_message(f"評価期間: {len(self.eval_windows)}個 (0-3m, 3-6m, 6-9m, 9-12m)")
        self.log_message(f"総評価数: {self.stats['total_evaluations']}個")
        self.log_message("")
        self.log_message(f"データ: {self.reviews_file}")
        self.log_message(f"訓練期間: {self.train_start} ～ {self.train_end}")
        self.log_message(f"評価期間: {self.eval_start} ～ {self.eval_end}")
        self.log_message(f"エポック数: {self.epochs}")
        self.log_message(f"出力先: {self.output_base}")
        self.log_message("")
        self.log_message("=" * 80)
        self.log_message("")
    
    def print_summary(self):
        """サマリーを表示"""
        self.log_message("")
        self.log_message("=" * 80)
        self.log_message("実行完了")
        self.log_message("=" * 80)
        self.log_message("")
        self.log_message(f"訓練: {self.stats['completed_trainings']}/{self.stats['total_trainings']} 成功, "
                        f"{self.stats['failed_trainings']} 失敗")
        self.log_message(f"評価: {self.stats['completed_evaluations']}/{self.stats['total_evaluations']} 成功, "
                        f"{self.stats['failed_evaluations']} 失敗")
        self.log_message("")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            hours = duration.total_seconds() / 3600
            self.log_message(f"実行時間: {hours:.2f}時間")
        
        self.log_message("")
        self.log_message(f"出力先: {self.output_base}")
        self.log_message(f"メインログ: {self.main_log}")
        self.log_message(f"個別ログ: {self.log_dir}/*.log")
        self.log_message("")
        self.log_message("=" * 80)
    
    def run(self):
        """メイン実行"""
        self.stats['start_time'] = datetime.now()
        self.print_header()
        
        try:
            # 各訓練ラベルで実行
            for train_label in self.train_labels:
                self.log_message("")
                self.log_message("=" * 80)
                self.log_message(f"訓練ラベル: 0-{train_label}m")
                self.log_message("=" * 80)
                self.log_message("")
                
                # 訓練実行
                success, model_path = self.run_training(train_label)
                
                if not success:
                    self.log_message(f"訓練失敗のため、0-{train_label}m の評価をスキップ", 'warning')
                    continue
                
                # モデルが存在しない場合はスキップ
                if not model_path.exists():
                    self.log_message(f"モデルが存在しないため、0-{train_label}m の評価をスキップ", 'warning')
                    continue
                
                # 各評価期間で評価
                self.log_message("")
                self.log_message(f"評価開始: 0-{train_label}m モデル")
                self.log_message("")
                
                for eval_start, eval_end in self.eval_windows:
                    self.run_evaluation(train_label, eval_start, eval_end, model_path)
                
                self.log_message("")
            
            self.stats['end_time'] = datetime.now()
            self.print_summary()
            
            # 失敗があった場合は終了コード1
            if self.stats['failed_trainings'] > 0 or self.stats['failed_evaluations'] > 0:
                return 1
            return 0
            
        except KeyboardInterrupt:
            self.log_message("\n中断されました", 'warning')
            self.stats['end_time'] = datetime.now()
            self.print_summary()
            return 130
        except Exception as e:
            self.log_message(f"予期しないエラー: {e}", 'error')
            import traceback
            self.log_message(traceback.format_exc(), 'error')
            self.stats['end_time'] = datetime.now()
            self.print_summary()
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='拡張IRL完全クロス評価（Python版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本実行
  python scripts/training/irl/run_enhanced_cross_evaluation.py

  # カスタム設定
  python scripts/training/irl/run_enhanced_cross_evaluation.py \\
    --epochs 30 \\
    --output outputs/my_enhanced_eval

  # ドライラン（実際には実行しない）
  python scripts/training/irl/run_enhanced_cross_evaluation.py --dry-run
        """
    )
    
    parser.add_argument(
        '--reviews',
        default='data/review_requests_openstack_multi_5y_detail.csv',
        help='レビューデータファイル（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--train-start',
        default='2021-01-01',
        help='訓練開始日（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--train-end',
        default='2023-01-01',
        help='訓練終了日（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--eval-start',
        default='2023-01-01',
        help='評価開始日（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--eval-end',
        default='2024-01-01',
        help='評価終了日（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--history-window',
        type=int,
        default=12,
        help='履歴ウィンドウ（月）（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='訓練エポック数（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--output',
        default='outputs/enhanced_cross_eval',
        help='出力ディレクトリ（デフォルト: %(default)s）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='ドライラン（実際には実行しない）'
    )
    
    args = parser.parse_args()
    
    # 実行
    runner = EnhancedCrossEvaluationRunner(
        reviews_file=args.reviews,
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        history_window=args.history_window,
        epochs=args.epochs,
        output_base=args.output,
        dry_run=args.dry_run
    )
    
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

