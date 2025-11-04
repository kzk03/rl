# データフィルタリングガイド

OpenStack Gerritデータのフィルタリングスクリプトの使用方法

**最終更新**: 2025-10-17
**関連スクリプト**:
- [scripts/preprocessing/filter_bot_accounts.py](../scripts/preprocessing/filter_bot_accounts.py)
- [scripts/preprocessing/filter_by_project.py](../scripts/preprocessing/filter_by_project.py)

---

## 📋 目次

1. [ボットアカウント除外](#1-ボットアカウント除外)
2. [プロジェクト別フィルター](#2-プロジェクト別フィルター)
3. [実験結果](#3-実験結果)
4. [組み合わせ利用](#4-組み合わせ利用)

---

## 1. ボットアカウント除外

### 概要

OpenStackのGerritデータからCI/CDボットアカウントを除外するスクリプト。

**除外されるアカウント**: 44.41%のレビュー（61,120件）、133人のボットアカウント

### 推奨ボットパターン

```
bot, ci, automation, jenkins, build, deploy, zuul, gerrit, infra,
DL-ARC, openstack-ci, noreply, service
```

### 基本的な使用方法

#### 統計情報のみ表示（dry-run）

```bash
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --dry-run
```

**出力例**:
```
元データ: 137,632件, 1,379人
ボットアカウント数: 133人
ボット関連レビュー数: 61,120件
全レビューに占める割合: 44.41%

上位10ボットアカウント:
  emc.scaleio.ci@emc.com: 3,625件
  hp.cinder.blr.ci@groups.ext.hpe.com: 3,127件
  neutron_hyperv_ci@cloudbasesolutions.com: 2,953件
  ...

フィルタリング後: 76,512件, 1,246人
除外されたレビュー数: 61,120件 (44.41%)
除外されたレビュアー数: 133人
```

#### ボットを除外してファイル保存

```bash
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_openstack_no_bots.csv
```

#### 追加のボットパターンを指定

```bash
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_openstack_no_bots.csv \
  --additional-patterns "deploy" "release" "test"
```

### オプション一覧

| オプション | 説明 | デフォルト |
|----------|------|----------|
| `--input` | 入力CSVファイルパス（必須） | - |
| `--output` | 出力CSVファイルパス | なし（保存しない） |
| `--email-column` | メールアドレスのカラム名 | `reviewer_email` |
| `--additional-patterns` | 追加のボット検出パターン | なし |
| `--dry-run` | 統計情報のみ表示（保存しない） | False |

---

## 2. プロジェクト別フィルター

### 概要

OpenStackのGerritデータを特定のプロジェクトで絞り込むスクリプト。
手動指定、上位N個の自動抽出、プロジェクト別分割に対応。

### OpenStackプロジェクト統計

```
総プロジェクト数: 5
総レビュー数: 137,632件

順位  プロジェクト名              レビュー数    レビュアー数   期間
1     openstack/cinder           71,604        615          2013-03-04 ~ 2025-09-27
2     openstack/neutron          32,888        503          2012-06-20 ~ 2025-09-27
3     openstack/nova             27,328        565          2012-08-02 ~ 2025-09-26
4     openstack/glance            3,273        203          2015-11-12 ~ 2025-09-26
5     openstack/keystone          2,539        202          2014-04-07 ~ 2025-09-23
```

### 基本的な使用方法

#### 統計情報のみ表示

```bash
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --stats-only
```

#### 特定のプロジェクトでフィルタリング

```bash
# 1つのプロジェクト
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_nova.csv \
  --projects "openstack/nova"

# 複数のプロジェクト
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_nova_neutron.csv \
  --projects "openstack/nova" "openstack/neutron"
```

**結果**: 60,216件, 2プロジェクト（元データの43.75%）

#### 上位N個のプロジェクトを自動抽出

```bash
# レビュー数上位3個のプロジェクトを抽出
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_top3.csv \
  --top 3
```

**抽出されるプロジェクト**: cinder, neutron, nova

#### 各プロジェクトごとに個別ファイルを作成（自動化）

```bash
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --split-by-project \
  --output-dir data/projects/ \
  --min-reviews 500
```

**出力例**:
```
data/projects/
├── openstack_cinder.csv    (71,604件)
├── openstack_neutron.csv   (32,888件)
├── openstack_nova.csv      (27,328件)
├── openstack_glance.csv    (3,273件)
└── openstack_keystone.csv  (2,539件)
```

### オプション一覧

| オプション | 説明 | デフォルト |
|----------|------|----------|
| `--input` | 入力CSVファイルパス（必須） | - |
| `--output` | 出力CSVファイルパス | なし |
| `--project-column` | プロジェクト名のカラム名 | `project` |
| `--projects` | フィルタリングするプロジェクト名 | なし |
| `--top` | 上位N個のプロジェクトを抽出 | なし |
| `--split-by-project` | プロジェクト別に個別ファイル作成 | False |
| `--output-dir` | プロジェクト別ファイルの出力先 | `data/projects` |
| `--min-reviews` | 分割時の最小レビュー数 | 100 |
| `--stats-only` | 統計情報のみ表示 | False |

---

## 3. 実験結果

### ボット除外の効果

| 項目 | 元データ | ボット除外後 | 変化 |
|-----|---------|------------|------|
| レビュー数 | 137,632件 | 76,512件 | -44.41% |
| レビュアー数 | 1,379人 | 1,246人 | -9.65% |
| ボットアカウント数 | 133人 | 0人 | -100% |

**主要なボットアカウント**:
- `emc.scaleio.ci@emc.com`: 3,625件
- `hp.cinder.blr.ci@groups.ext.hpe.com`: 3,127件
- `neutron_hyperv_ci@cloudbasesolutions.com`: 2,953件
- `cisco-cinder-ci@cisco.com`: 2,801件

### プロジェクト別フィルターの効果

#### nova + neutron（2プロジェクト）

| 項目 | 元データ | フィルタリング後 | 変化 |
|-----|---------|----------------|------|
| レビュー数 | 137,632件 | 60,216件 | -56.25% |
| プロジェクト数 | 5 | 2 | -60% |

#### 上位3プロジェクト（cinder + neutron + nova）

| 項目 | 元データ | フィルタリング後 | 変化 |
|-----|---------|----------------|------|
| レビュー数 | 137,632件 | 131,820件 | -4.22% |
| プロジェクト数 | 5 | 3 | -40% |

---

## 4. 組み合わせ利用

### パターン1: ボット除外 → プロジェクト指定

```bash
# ステップ1: ボット除外
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_openstack_no_bots.csv

# ステップ2: プロジェクト指定
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_no_bots.csv \
  --output data/review_requests_nova_no_bots.csv \
  --projects "openstack/nova"
```

### パターン2: プロジェクト別にボット除外（自動化）

```bash
# 各プロジェクトごとにボット除外済みファイルを作成
for project in "openstack/nova" "openstack/neutron" "openstack/cinder"; do
  safe_name=$(echo $project | sed 's/\//_/g')

  # プロジェクト抽出
  uv run python scripts/preprocessing/filter_by_project.py \
    --input data/review_requests_openstack_multi_5y_detail.csv \
    --output data/temp_${safe_name}.csv \
    --projects "$project"

  # ボット除外
  uv run python scripts/preprocessing/filter_bot_accounts.py \
    --input data/temp_${safe_name}.csv \
    --output data/${safe_name}_no_bots.csv

  # 一時ファイル削除
  rm data/temp_${safe_name}.csv
done
```

### パターン3: IRL訓練用データ準備（推奨ワークフロー）

```bash
# 1. ボット除外
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# 2. 上位3プロジェクトを抽出
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_top3_no_bots.csv \
  --top 3

# 3. IRL訓練
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_top3_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 12 \
  --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_top3_no_bots
```

---

## 5. トラブルシューティング

### Q1: 「存在しないプロジェクト」エラー

```bash
# まず統計情報を確認
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --stats-only

# 正確なプロジェクト名を指定
--projects "openstack/nova"  # ✅ 正しい
--projects "nova"            # ❌ 間違い
```

### Q2: ボット除外後もボットアカウントが残っている

```bash
# 追加パターンを指定
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/filtered.csv \
  --additional-patterns "custom_bot_pattern"
```

### Q3: ファイルサイズが大きすぎる

```bash
# 上位プロジェクトのみに絞る
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/top1.csv \
  --top 1

# またはプロジェクト別に分割
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --split-by-project \
  --output-dir data/projects/
```

---

## 6. まとめ

### 推奨フィルタリング手順

1. **ボット除外**: データ品質向上のため、必ず実施（44%のノイズ除去）
2. **プロジェクト指定**: 実験目的に応じて選択
   - 全プロジェクト: 包括的な分析
   - 上位3プロジェクト: 主要プロジェクトに集中
   - 特定プロジェクト: プロジェクト特化型分析

### フィルタリング前後の比較

| データセット | レビュー数 | レビュアー数 | 用途 |
|------------|-----------|------------|------|
| 元データ | 137,632件 | 1,379人 | - |
| ボット除外 | 76,512件 | 1,246人 | 推奨（品質向上） |
| 上位3プロジェクト（ボット除外済み） | ~73,000件 | ~1,100人 | 主要プロジェクト分析 |
| nova単体（ボット除外済み） | ~15,000件 | ~400人 | プロジェクト特化分析 |

---

## 📚 関連ドキュメント

- [DATA_PROCESSING_DETAILS.md](DATA_PROCESSING_DETAILS.md): データ処理の詳細
- [IRL_COMPREHENSIVE_GUIDE.md](IRL_COMPREHENSIVE_GUIDE.md): IRL全体ガイド
- [README_TEMPORAL_IRL.md](../README_TEMPORAL_IRL.md): Temporal IRL実験結果

---

**最終更新**: 2025-10-17
**作成者**: Claude + Kazuki-h
**ステータス**: 完成
