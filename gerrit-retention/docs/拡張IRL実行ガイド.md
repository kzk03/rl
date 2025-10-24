# 拡張 IRL 実行ガイド

## 概要

拡張 IRL（EnhancedRetentionIRLSystem）は、通常 IRL の 10 次元の状態特徴量を 32 次元に拡張し、より豊富な特徴量を使用して開発者の継続予測を行います。

## 🆚 通常 IRL vs 拡張 IRL の比較

| 項目             | **通常 IRL**       | **拡張 IRL**             |
| ---------------- | ------------------ | ------------------------ |
| **状態特徴量**   | 10 次元            | **32 次元** ⭐           |
| **行動特徴量**   | 5 次元             | **9 次元** ⭐            |
| **ネットワーク** | hidden_dim=128     | **hidden_dim=256**       |
| **特徴量**       | 基本的な特徴量のみ | **高優先度特徴量を統合** |
| **実行速度**     | 速い               | やや遅い                 |
| **メモリ**       | 少ない             | やや多い                 |

## 📊 拡張特徴量の詳細

### 状態特徴量（32 次元）

#### **基本特徴量**（10 次元）

通常 IRL と同じ：

1. 経験日数（年単位）
2. 総変更数（/100）
3. 総レビュー数（/100）
4. プロジェクト数（/10）
5. 最近の活動頻度
6. 平均活動間隔（月単位）
7. 活動トレンド
8. 協力スコア
9. コード品質スコア
10. 時間経過（年単位）

#### **A1: 活動頻度の多期間比較**（5 次元）

11. 7 日間の活動頻度
12. 30 日間の活動頻度
13. 90 日間の活動頻度
14. 活動加速度（短期 vs 中期）
15. 一貫性スコア（標準偏差ベース）

#### **B1: レビュー負荷指標**（6 次元）

16. 7 日間のレビュー負荷（件/日）
17. 30 日間のレビュー負荷
18. 180 日間のレビュー負荷
19. レビュー負荷トレンド
20. 過負荷フラグ（5 件/日以上）
21. 高負荷フラグ（2 件/日以上）

#### **C1: 相互作用の深さ**（4 次元）

22. 180 日間の相互作用数
23. 相互作用強度（interactions/months_active）
24. プロジェクト固有の相互作用
25. 割り当て履歴（180 日）

#### **D1: 専門性の一致度**（2 次元）

26. パス類似度スコア（Jaccard 平均）
27. パスオーバーラップスコア

#### **その他**（5 次元）

28. 平均応答時間（日）
29. 応答率（180 日）
30. 在籍日数
31. 平均変更サイズ
32. 平均変更ファイル数

### 行動特徴量（9 次元）

#### **基本特徴量**（5 次元）

通常 IRL と同じ：

1. 行動タイプ
2. 強度
3. 品質
4. 協力度
5. 時間経過

#### **拡張特徴量**（4 次元）

6. 変更挿入行数
7. 変更削除行数
8. 変更ファイル数
9. 応答遅延時間（日）

## 🚀 実行方法

### サーバでの実行（推奨）

```bash
# 拡張IRL完全クロス評価を実行
cd /path/to/gerrit-retention
bash scripts/training/irl/run_enhanced_cross_evaluation.sh
```

### 実行される内容

**5 つの訓練ラベル × 4 つの評価期間 = 20 評価**

#### 訓練ラベル

- 0-1m
- 0-3m
- 0-6m
- 0-9m
- 0-12m

#### 評価期間

- 0-3m
- 3-6m
- 6-9m
- 9-12m

### パラメータ設定

```bash
# デフォルト設定（scripts/training/irl/run_enhanced_cross_evaluation.sh）
REVIEWS_FILE="data/review_requests_openstack_multi_5y_detail.csv"
TRAIN_START="2021-01-01"
TRAIN_END="2023-01-01"
EVAL_START="2023-01-01"
EVAL_END="2024-01-01"
HISTORY_WINDOW=12  # ヶ月
EPOCHS=20
```

## ⏱️ 予想実行時間

### ローカル（Macbook）

- 1 モデル訓練: 約 1-1.5 時間
- 完全クロス評価（5 モデル）: **約 5-7 時間**

### サーバ（強力な GPU）

- 1 モデル訓練: 約 30-40 分
- 完全クロス評価（5 モデル）: **約 2.5-3.5 時間**

⚠️ **注意**: 拡張 IRL は通常 IRL より約 1.2-1.5 倍時間がかかります

## 📊 出力結果

### ディレクトリ構造

```
outputs/enhanced_cross_eval/
├── logs/
│   ├── main.log
│   ├── train_0-1m.log
│   ├── train_0-3m.log
│   ├── ...
│   └── train_0-12m_eval_9-12m.log
├── train_0-1m/
│   ├── enhanced_irl_model.pt  ← 拡張IRLモデル
│   ├── metrics.json
│   ├── predictions.csv
│   ├── train_trajectories.pkl
│   └── eval_trajectories.pkl
├── train_0-3m/
│   └── ...
└── train_0-12m/
    └── ...
```

### 結果の確認

```bash
# メインログを確認
tail -f outputs/enhanced_cross_eval/logs/main.log

# 訓練ログを確認
tail -f outputs/enhanced_cross_eval/logs/train_0-3m.log

# 結果サマリーを生成
uv run python scripts/training/irl/summarize_full_cross_evaluation.py \
  --input outputs/enhanced_cross_eval \
  --output outputs/enhanced_cross_eval/summary.md
```

## 🔄 通常 IRL との比較

### 比較実行

通常 IRL との AUC-ROC 比較：

```bash
# 通常IRL（実行済み）
ls outputs/full_cross_eval/train_*/metrics.json

# 拡張IRL（実行後）
ls outputs/enhanced_cross_eval/train_*/metrics.json

# 比較スクリプト（作成予定）
python scripts/analysis/compare_irl_versions.py \
  --baseline outputs/full_cross_eval \
  --enhanced outputs/enhanced_cross_eval \
  --output outputs/irl_comparison.md
```

### 期待される改善

拡張特徴量により以下の改善が期待されます：

- **AUC-ROC**: +0.02-0.05 程度の改善
- **AUC-PR**: +0.03-0.07 程度の改善
- **F1 スコア**: +0.01-0.03 程度の改善

特に以下のケースで改善が大きいと予想：

- レビュー負荷が高い開発者
- 専門性の一致度が重要な場合
- 活動パターンの変化が激しい開発者

## 🐛 トラブルシューティング

### メモリ不足エラー

```bash
# バッチサイズを減らす（未実装の場合は実装が必要）
# または、より小さなデータセットで試す
```

### 実行が遅い

```bash
# EPOCHSを減らして試す
# 例: EPOCHS=10
sed -i '' 's/EPOCHS=20/EPOCHS=10/' scripts/training/irl/run_enhanced_cross_evaluation.sh
```

### ScalerFitting エラー

拡張 IRL は初回実行時に特徴量のスケーリングを行います。エラーが出た場合：

```bash
# データが十分にあるか確認
# 最低100サンプル程度必要
```

## 📝 サーバ実行チェックリスト

### 実行前

- [ ] データファイルが存在する（`data/review_requests_openstack_multi_5y_detail.csv`）
- [ ] `uv`がインストールされている
- [ ] 十分なディスク容量がある（約 5GB 以上）
- [ ] 他の実行プロセスと競合しないか確認

### 実行中

- [ ] `caffeinate`でスリープ防止（macOS）
- [ ] ログファイルで進捗確認
- [ ] メモリ使用量モニタリング

### 実行後

- [ ] 全 5 モデルが完成したか確認（`ls outputs/enhanced_cross_eval/train_*/enhanced_irl_model.pt`）
- [ ] サマリーレポート生成
- [ ] 通常 IRL との比較

## 🎯 次のステップ

### 性能比較

```bash
# 1. サマリー生成
python scripts/training/irl/summarize_full_cross_evaluation.py \
  --input outputs/enhanced_cross_eval \
  --output outputs/enhanced_cross_eval/summary.md

# 2. 比較分析
python scripts/analysis/compare_irl_versions.py \
  --baseline outputs/full_cross_eval \
  --enhanced outputs/enhanced_cross_eval

# 3. 特徴量重要度分析
bash scripts/analysis/run_feature_importance_all_models.sh
```

### 論文・レポート作成

拡張特徴量の効果を検証した結果を論文やレポートにまとめる際のポイント：

1. **特徴量の設計根拠**

   - なぜこれらの特徴量を選んだか
   - 先行研究との関連

2. **性能改善の分析**

   - どの特徴量が最も効果的か
   - どのような状況で改善が大きいか

3. **計算コストとの トレードオフ**
   - 実行時間の増加
   - メモリ使用量の増加
   - 性能改善が妥当か

## 🤝 関連ドキュメント

- `docs/特徴量の重要度分析.md` - 特徴量重要度の測定方法
- `docs/完全クロス評価ガイド.md` - クロス評価の詳細
- `SERVER_EXECUTION_GUIDE.md` - サーバ実行の詳細ガイド
- `src/gerrit_retention/rl_prediction/enhanced_feature_extractor.py` - 特徴量抽出の実装

## ⚡ クイックスタート

```bash
# サーバで拡張IRLを実行（最短手順）
cd /path/to/gerrit-retention

# 実行（バックグラウンド）
nohup bash scripts/training/irl/run_enhanced_cross_evaluation.sh \
  > /tmp/enhanced_cross_eval.log 2>&1 &

# PIDを保存
echo $! > /tmp/enhanced_cross_eval.pid

# スリープ防止（macOS）
caffeinate -i -s -w $(cat /tmp/enhanced_cross_eval.pid)

# 進捗確認
tail -f outputs/enhanced_cross_eval/logs/main.log
```

## 📧 サポート

問題が発生した場合は、以下の情報を含めて報告してください：

- エラーメッセージ
- 実行環境（OS、メモリ、GPU 等）
- ログファイル（`outputs/enhanced_cross_eval/logs/*.log`）
- データサイズ（レビュー数、レビュアー数）
