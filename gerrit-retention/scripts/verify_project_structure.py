#!/usr/bin/env python3
"""
プロジェクト構造検証スクリプト

プロジェクトの基本構造が正しく作成されているかを確認します。
"""

import sys
from pathlib import Path
from typing import List, Tuple

# プロジェクトルートを取得
project_root = Path(__file__).parent.parent

def check_directory_structure() -> Tuple[bool, List[str]]:
    """ディレクトリ構造をチェック"""
    required_dirs = [
        "src/gerrit_retention",
        "src/gerrit_retention/data_integration",
        "src/gerrit_retention/prediction", 
        "src/gerrit_retention/behavior_analysis",
        "src/gerrit_retention/rl_environment",
        "src/gerrit_retention/visualization",
        "src/gerrit_retention/adaptive_strategy",
        "src/gerrit_retention/utils",
        "training/retention_training",
        "training/stress_training",
        "training/rl_training",
        "data_processing/gerrit_extraction",
        "data_processing/feature_engineering",
        "data_processing/preprocessing",
        "analysis/reports",
        "analysis/visualization",
        "evaluation/model_evaluation",
        "evaluation/ab_testing",
        "evaluation/integration_tests",
        "pipelines",
        "scripts",
        "configs",
        "data/raw",
        "data/processed",
        "data/external",
        "models",
        "logs",
        "outputs",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "docker",
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0, missing_dirs


def check_required_files() -> Tuple[bool, List[str]]:
    """必須ファイルをチェック"""
    required_files = [
        "README.md",
        "pyproject.toml",
        "setup.py",
        ".env.example",
        ".gitignore",
        "src/gerrit_retention/__init__.py",
        "src/gerrit_retention/utils/logger.py",
        "src/gerrit_retention/utils/config_manager.py",
        "src/gerrit_retention/cli.py",
        "configs/gerrit_config.yaml",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "scripts/setup_gerrit_connection.py",
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def check_python_imports() -> Tuple[bool, List[str]]:
    """Pythonインポートをチェック"""
    import_errors = []
    
    try:
        sys.path.insert(0, str(project_root / "src"))
        
        # 基本インポートをテスト
        import gerrit_retention
        from gerrit_retention.utils.config_manager import ConfigManager
        from gerrit_retention.utils.logger import get_logger
        
        print(f"✓ gerrit_retention v{gerrit_retention.get_version()} が正常にインポートされました")
        
    except ImportError as e:
        import_errors.append(f"インポートエラー: {e}")
    
    return len(import_errors) == 0, import_errors


def main():
    """メイン関数"""
    print("Gerrit開発者定着予測システム - プロジェクト構造検証")
    print("=" * 60)
    
    all_checks_passed = True
    
    # ディレクトリ構造をチェック
    print("\n1. ディレクトリ構造をチェック中...")
    dirs_ok, missing_dirs = check_directory_structure()
    
    if dirs_ok:
        print("✓ すべての必須ディレクトリが存在します")
    else:
        print("✗ 以下のディレクトリが不足しています:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        all_checks_passed = False
    
    # 必須ファイルをチェック
    print("\n2. 必須ファイルをチェック中...")
    files_ok, missing_files = check_required_files()
    
    if files_ok:
        print("✓ すべての必須ファイルが存在します")
    else:
        print("✗ 以下のファイルが不足しています:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        all_checks_passed = False
    
    # Pythonインポートをチェック
    print("\n3. Pythonインポートをチェック中...")
    imports_ok, import_errors = check_python_imports()
    
    if imports_ok:
        print("✓ すべてのPythonモジュールが正常にインポートされます")
    else:
        print("✗ 以下のインポートエラーがあります:")
        for error in import_errors:
            print(f"  - {error}")
        all_checks_passed = False
    
    # 結果を表示
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ プロジェクト構造の検証が成功しました！")
        print("\n次のステップ:")
        print("1. .env ファイルを作成してGerrit接続情報を設定")
        print("2. uv sync で依存関係をインストール")
        print("3. python scripts/setup_gerrit_connection.py で接続をテスト")
        return 0
    else:
        print("✗ プロジェクト構造に問題があります。上記のエラーを修正してください。")
        return 1


if __name__ == "__main__":
    sys.exit(main())