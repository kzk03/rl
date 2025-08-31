"""
Gerrit開発者定着予測システム - コマンドラインインターフェース

システムの主要機能をコマンドラインから実行するためのCLIです。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from gerrit_retention import get_version, initialize_system
from gerrit_retention.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="gerrit-retention",
        description="Gerrit開発者定着予測システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  gerrit-retention --version
  gerrit-retention --config configs/gerrit_config.yaml train
  gerrit-retention predict --developer john.doe@example.com
  gerrit-retention analyze --output outputs/analysis.html
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"gerrit-retention {get_version()}"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイルのパス"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベル"
    )
    
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default="text",
        help="ログフォーマット"
    )
    
    # サブコマンドを追加
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # 訓練コマンド
    train_parser = subparsers.add_parser("train", help="モデルを訓練")
    train_parser.add_argument(
        "--component",
        choices=["retention", "stress", "rl", "all"],
        default="all",
        help="訓練するコンポーネント"
    )
    
    # 予測コマンド
    predict_parser = subparsers.add_parser("predict", help="定着確率を予測")
    predict_parser.add_argument(
        "--developer",
        required=True,
        help="開発者のメールアドレス"
    )
    predict_parser.add_argument(
        "--output",
        help="出力ファイルのパス"
    )
    
    # 分析コマンド
    analyze_parser = subparsers.add_parser("analyze", help="分析レポートを生成")
    analyze_parser.add_argument(
        "--type",
        choices=["retention", "stress", "behavior", "all"],
        default="all",
        help="分析タイプ"
    )
    analyze_parser.add_argument(
        "--output",
        help="出力ディレクトリのパス"
    )
    
    # データ抽出コマンド
    extract_parser = subparsers.add_parser("extract", help="Gerritからデータを抽出")
    extract_parser.add_argument(
        "--project",
        required=True,
        help="Gerritプロジェクト名"
    )
    extract_parser.add_argument(
        "--start-date",
        help="開始日（YYYY-MM-DD形式）"
    )
    extract_parser.add_argument(
        "--end-date",
        help="終了日（YYYY-MM-DD形式）"
    )
    
    # 設定検証コマンド
    subparsers.add_parser("validate", help="設定を検証")
    
    return parser


def handle_train_command(args: argparse.Namespace) -> int:
    """訓練コマンドを処理"""
    logger.info(f"訓練を開始: {args.component}")
    
    try:
        # 訓練パイプラインを使用してモデルを訓練
        from gerrit_retention.pipelines.training_pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline(args.config)
        
        # 訓練するモデルを決定
        models_to_train = []
        if args.component in ["retention", "all"]:
            logger.info("定着予測モデルを訓練中...")
            models_to_train.append("retention_model")
            
        if args.component in ["stress", "all"]:
            logger.info("ストレス分析モデルを訓練中...")
            models_to_train.append("stress_model")
            
        if args.component in ["rl", "all"]:
            logger.info("強化学習エージェントを訓練中...")
            models_to_train.append("rl_agent")
        
        # 訓練パイプラインを実行
        result = pipeline.run_training_pipeline(
            models=models_to_train,
            backup_existing=True,
            evaluate_after_training=True
        )
        
        if result['success']:
            logger.info(f"訓練完了: {result['summary']['successful_models']}/{result['summary']['total_models']}モデル成功")
        else:
            logger.error(f"訓練失敗: {result['summary']['successful_models']}/{result['summary']['total_models']}モデル成功")
            return 1
            
        logger.info("訓練が完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"訓練中にエラーが発生しました: {e}")
        return 1


def handle_predict_command(args: argparse.Namespace) -> int:
    """予測コマンドを処理"""
    logger.info(f"開発者の定着確率を予測中: {args.developer}")
    
    try:
        # 定着予測システムを使用して予測を実行
        from gerrit_retention.prediction.retention_predictor import RetentionPredictor
        from gerrit_retention.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager(args.config)
        predictor = RetentionPredictor(config_manager.config)
        
        # 開発者の定着確率を予測
        prediction_result = predictor.predict_retention(args.developer)
        
        # 結果を出力
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prediction_result, f, ensure_ascii=False, indent=2)
            logger.info(f"予測結果を保存しました: {output_path}")
        else:
            print(json.dumps(prediction_result, ensure_ascii=False, indent=2))
        
        logger.info("予測が完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"予測中にエラーが発生しました: {e}")
        return 1


def handle_analyze_command(args: argparse.Namespace) -> int:
    """分析コマンドを処理"""
    logger.info(f"分析レポートを生成中: {args.type}")
    
    try:
        # 分析レポートシステムを使用して分析を実行
        from gerrit_retention.analysis.reports.advanced_retention_insights import (
            AdvancedRetentionInsights,
        )
        from gerrit_retention.analysis.reports.retention_factor_analysis import (
            RetentionFactorAnalyzer,
        )
        from gerrit_retention.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager(args.config)
        
        # 出力ディレクトリを準備
        output_dir = Path(args.output) if args.output else Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {}
        
        if args.type in ["retention", "all"]:
            logger.info("継続要因分析を実行中...")
            analyzer = RetentionFactorAnalyzer(config_manager.config)
            retention_result = analyzer.run_comprehensive_analysis()
            analysis_results['retention'] = retention_result
            
            # レポートを保存
            report_path = output_dir / f"retention_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(retention_result, f, ensure_ascii=False, indent=2)
            logger.info(f"継続要因分析レポートを保存: {report_path}")
        
        if args.type in ["behavior", "all"]:
            logger.info("高度分析を実行中...")
            insights = AdvancedRetentionInsights(config_manager.config)
            behavior_result = insights.generate_comprehensive_insights()
            analysis_results['behavior'] = behavior_result
            
            # レポートを保存
            report_path = output_dir / f"behavior_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(behavior_result, f, ensure_ascii=False, indent=2)
            logger.info(f"行動分析レポートを保存: {report_path}")
        
        if args.type in ["stress", "all"]:
            logger.info("ストレス分析を実行中...")
            # ストレス分析は既存の実装を使用
            from gerrit_retention.prediction.stress_analyzer import StressAnalyzer
            stress_analyzer = StressAnalyzer(config_manager.config)
            stress_result = stress_analyzer.analyze_stress_patterns()
            analysis_results['stress'] = stress_result
            
            # レポートを保存
            report_path = output_dir / f"stress_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stress_result, f, ensure_ascii=False, indent=2)
            logger.info(f"ストレス分析レポートを保存: {report_path}")
        
        # 統合レポートを生成
        summary_report_path = output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析が完了しました。結果は {output_dir} に保存されました")
        return 0
        
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {e}")
        return 1


def handle_extract_command(args: argparse.Namespace) -> int:
    """データ抽出コマンドを処理"""
    logger.info(f"Gerritからデータを抽出中: {args.project}")
    
    try:
        # Gerritデータ抽出システムを使用してデータを抽出
        from gerrit_retention.data_processing.gerrit_extraction.gerrit_client import (
            GerritClient,
        )
        from gerrit_retention.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager(args.config)
        gerrit_config = config_manager.get('gerrit', {})
        
        # Gerritクライアントを初期化
        client = GerritClient(gerrit_config)
        
        # データ抽出パラメータを設定
        extraction_params = {
            'project': args.project,
            'start_date': args.start_date,
            'end_date': args.end_date
        }
        
        logger.info(f"Gerritからデータを抽出中: {args.project}")
        
        # データを抽出
        extraction_result = client.extract_project_data(**extraction_params)
        
        # 結果を保存
        output_dir = Path("data/raw") / args.project
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 変更データを保存
        if extraction_result.get('changes'):
            changes_file = output_dir / f"changes_{timestamp}.json"
            with open(changes_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['changes'], f, ensure_ascii=False, indent=2)
            logger.info(f"変更データを保存: {changes_file}")
        
        # 開発者データを保存
        if extraction_result.get('developers'):
            developers_file = output_dir / f"developers_{timestamp}.json"
            with open(developers_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['developers'], f, ensure_ascii=False, indent=2)
            logger.info(f"開発者データを保存: {developers_file}")
        
        # 抽出サマリーを保存
        summary_file = output_dir / f"extraction_summary_{timestamp}.json"
        summary = {
            'project': args.project,
            'extraction_date': datetime.now().isoformat(),
            'parameters': extraction_params,
            'results': {
                'changes_count': len(extraction_result.get('changes', [])),
                'developers_count': len(extraction_result.get('developers', [])),
                'date_range': {
                    'start': args.start_date,
                    'end': args.end_date
                }
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"データ抽出が完了しました。結果は {output_dir} に保存されました")
        logger.info(f"抽出件数: 変更 {summary['results']['changes_count']}件, 開発者 {summary['results']['developers_count']}名")
        return 0
        
    except Exception as e:
        logger.error(f"データ抽出中にエラーが発生しました: {e}")
        return 1


def handle_validate_command(args: argparse.Namespace) -> int:
    """設定検証コマンドを処理"""
    logger.info("設定を検証中...")
    
    try:
        config_manager = initialize_system(args.config)
        
        if config_manager.validate_config():
            logger.info("設定の検証が成功しました")
            return 0
        else:
            logger.error("設定の検証が失敗しました")
            return 1
            
    except Exception as e:
        logger.error(f"設定検証中にエラーが発生しました: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # ログ設定を初期化
    setup_logging(level=args.log_level, format_type=args.log_format)
    
    logger.info(f"Gerrit開発者定着予測システム v{get_version()} を開始")
    
    # システムを初期化
    try:
        initialize_system(args.config)
    except Exception as e:
        logger.error(f"システムの初期化に失敗しました: {e}")
        return 1
    
    # コマンドを処理
    if args.command == "train":
        return handle_train_command(args)
    elif args.command == "predict":
        return handle_predict_command(args)
    elif args.command == "analyze":
        return handle_analyze_command(args)
    elif args.command == "extract":
        return handle_extract_command(args)
    elif args.command == "validate":
        return handle_validate_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())