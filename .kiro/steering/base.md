---
inclusion: always
---

# Kazoo 開発基本ルール

## 必須要件

### パッケージ管理

- **必須**: `uv` を使用してパッケージ管理・実行を行う
- **コマンド例**: ``uv run script.py`

### 言語・コメント

- **日本語**: コメント、docstring、エラーメッセージは日本語で記述
- **英語**: 変数名、関数名、ファイル名は英語（snake_case）

## コードスタイル規則

### 命名規則

```python
# ファイル名: snake_case
train_gat.py, evaluate_models.py

# 関数・変数: snake_case + 型ヒント
def train_model(config: Dict[str, Any]) -> torch.nn.Module:
    """モデルを訓練する関数"""
    pass

# 設定ファイル: 用途_環境.yaml
base_training.yaml, dev_profiles_test_2022.yaml
```

### インポート順序

1. 標準ライブラリ
2. サードパーティ
3. プロジェクト内モジュール

## 重要な開発原則

### 時系列データ（絶対厳守）

- **禁止**: 未来データを過去の予測に使用
- **必須**: IRL (2019-2021) → RL 訓練 (2022) → テスト (2023)
- **検証**: 訓練前に `scripts/split_temporal_data.py` で時系列整合性を確認

### 再現性

- **設定**: すべての実験は YAML 設定ファイルで制御
- **シード**: ランダムシード固定で結果の再現性を保証
- **バージョン**: モデル・データセットのバージョン管理

### エラーハンドリング

```python
try:
    model = load_model(model_path)
except FileNotFoundError:
    logger.error(f"モデルファイルが見つかりません: {model_path}")
    raise
```

## ファイル管理

### 保存場所

- **モデル**: `models/{component}_{timestamp}.{ext}`
- **データ**: `data/` (処理済みデータセット)
- **設定**: `configs/` (YAML 設定ファイル)
- **出力**: `outputs/` (実験結果)

### 命名パターン

- **タイムスタンプ**: `YYYYMMDD_HHMMSS`
- **コンポーネント**: `gat_`, `irl_`, `rl_`
- **用途**: `_training`, `_test`, `_best`

## 実行時の注意点

### メモリ・パフォーマンス

- 大きなデータセットはバッチ処理
- GPU 使用時は適切なバッチサイズ設定
- 不要オブジェクトの明示的削除

### セキュリティ

- 機密情報は `.env` ファイルで管理
- GitHub トークン等をコードに直接記述禁止
