---
inclusion: always
---

# Kazoo プロダクトガイドライン

Kazoo は、GitHub コラボレーションデータで訓練された **GAT → IRL → RL** パイプラインを使用する、OSS タスク推薦のためのマルチエージェント強化学習システムです。

## 重要なアーキテクチャルール

### 1. 時系列データの整合性（必須）

- **絶対に** 過去の予測に未来のデータを使用しない
- **必ず** 時系列順序を維持する: 訓練 ≤ 検証 ≤ テスト
- **必須**: 訓練前に `scripts/split_temporal_data.py` で時系列分割を検証する
- **確認**: すべてのデータファイルの時系列整合性を検証する

### 2. 3 段階パイプラインシーケンス

```
GAT (埋め込み) → IRL (報酬) → RL (ポリシー)
```

**GAT 要件:**

- 入力: `data_processing/generate_graph.py` → コラボレーショングラフ
- 出力: `data/` ディレクトリ内の `.pt` テンソル
- 実装: `training/gat/train_gat.py`

**IRL 要件:**

- 入力: `data/expert_trajectories*.pkl` (ボットフィルタ済み)
- 出力: RL 用の `.npy` 重みファイル
- 実装: `training/irl/train_irl.py`

**RL 要件:**

- 入力: GAT 埋め込み + IRL 報酬
- 出力: `.zip` ポリシーファイル (Stable-Baselines3 形式)
- 実装: `training/rl/train_rl.py`

## データ品質・処理ルール

### ボットフィルタリング（必須）

- **必ず** エキスパート軌跡からボットアカウントを除外する
- **確認**: `expert_trajectories*.pkl` ファイルがボットフィルタ済みであることを検証する
- **検証**: 自動化されたコミットが訓練データから除去されていることを確認する

### グラフ検証要件

- **GAT 前**: グラフの接続性を検証する
- **IRL 前**: 軌跡の完全性をチェックする
- **最低限**: コラボレーション閾値が満たされていることを確認する（YAML で設定可能）

### ファイル処理順序

1. `data_processing/generate_backlog.py` → タスクデータ
2. `data_processing/generate_graph.py` → コラボレーションネットワーク
3. `data_processing/create_expert_trajectories.py` → エキスパートデータ
4. 訓練パイプライン: GAT → IRL → RL

## 設定要件

### 必須 YAML 構造

- **ベース**: すべての訓練に `configs/base_training.yaml`
- **テスト**: 評価に `configs/base_test_2022.yaml`
- **本番**: デプロイに `configs/production.yaml`

### 必須設定キー（必ず含める）

```yaml
temporal_split:
  train_end_date: "YYYY-MM-DD" # 必須
  val_start_date: "YYYY-MM-DD" # 必須
  test_start_date: "YYYY-MM-DD" # 必須

components:
  gat_enabled: true # コンポーネントフラグ
  irl_enabled: true
  rl_enabled: true
```

### 設定検証

- **訓練前**: 時系列分割日付が時系列順であることを検証する
- **必須**: すべての実験は YAML 設定を使用する（ハードコードされたパラメータは禁止）
- **HYDRA**: 実験バリエーションには合成を使用する

## モデル保存・命名

### ファイル命名規則（厳格）

```
{component}_{timestamp}.{ext}
例:
- gat_model_20250714_102645.pt
- irl_weights_20250714_103000.npy
- rl_policy_20250714_104500.zip
```

### 保存場所

- **訓練モデル**: `models/{component}_{timestamp}.{ext}`
- **最良モデル**: `models/{component}_best/best_model.{ext}`
- **メタデータ**: `models/training_metadata.json` (設定 + メトリクス)

### ファイル形式要件

- **GAT**: PyTorch state dicts (`.pt`)
- **IRL**: NumPy arrays (`.npy`)
- **RL**: Stable-Baselines3 ZIP (`.zip`)
- **グラフ**: PyTorch Geometric (`.pt`)

## 訓練・評価基準

### 実行コマンド

```bash
# フルパイプライン（推奨）
python scripts/full_training_pipeline.py

# 個別コンポーネント
python training/gat/train_gat.py
python training/irl/train_irl.py
python training/rl/train_rl.py

# 評価
python evaluation/evaluate_models.py
```

### 検証要件

- **主要メトリクス**: 時系列テストセットでのタスク割り当て精度
- **評価**: 一貫したテストのために `evaluation/evaluate_models.py` を使用する
- **時系列テスト**: 保留された未来データでのみテストする
- **再現性**: すべての訓練スクリプトで固定ランダムシードを設定する

### 品質チェック（必須）

1. **訓練前**: `scripts/split_temporal_data.py` を実行して分割を検証する
2. **訓練後**: `analysis/reports/` で分析レポートを生成する
3. **デプロイ前**: 2022 年テストセットでモデル性能を検証する

## 開発ワークフロー

### コード修正ルール

- **新機能**: 時系列分割検証を含める必要がある
- **モデル変更**: 対応する分析レポートを更新する
- **実験**: YAML 設定のみを使用する（ハードコードされたパラメータは禁止）
- **ログ**: 構造化された出力でコンポーネント固有のロガーを使用する

### 分析・レポート

- **必須**: 訓練後に `analysis/reports/` を使用してレポートを生成する
- **比較**: 実験間で一貫した評価プロトコルを使用する
- **ドキュメント**: モデルアーキテクチャを変更する際は分析を更新する
