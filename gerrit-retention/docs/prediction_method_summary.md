## Future-Window & Gap-Based Retention Prediction Summary (2025-09-05)

### 1. データソース

- Developers: `data/processed/unified/all_developers.json`
- Changes (activity events): `data/processed/unified/all_reviews.json`
- Project scoped 分析対象: `openstack/nova` (実データの `project` フィールドは `openstack/nova` 形式)

### 2. 2 種類のラベリング手法

| 手法                      | ラベル定義                                           | 利点                 | 課題                                                    |
| ------------------------- | ---------------------------------------------------- | -------------------- | ------------------------------------------------------- |
| Gap Threshold (例: 60 日) | "最終活動から閾値日数以内なら 1"                     | 単純・高速           | 将来継続を直接見ない / 全体正例多発しやすい             |
| Future Window (Δ=H 日)    | スナップショット日時 T の後 H 日以内に活動があれば 1 | 真の将来行動に基づく | horizon が長過ぎると正例飽和 / 短すぎると特徴不足で難化 |

### 3. Gap Threshold 分布 (openstack/nova)

| Threshold | pos_rate |
| --------- | -------- |
| 30        | 0.308    |
| 45        | 0.462    |
| 60        | 0.538    |
| 90        | 0.615    |

→ 45〜60 日がバランス良。60 日採用で pos 約 0.54。

### 4. Future Window スナップショット生成ロジック (脚本: `evaluate_future_window_retention.py`)

1. 開発者ごとに活動タイムスタンプを昇順取得。
2. 末尾を除く系列から最大 N 個 (均等サンプリング) のスナップショットを選択。
3. 各 snapshot_date に対し horizon_days 内に将来活動があれば label=1。
4. 特徴は _累積集計_ を activity 割合(fraction) で線形スケールする近似 (完全な過去再計算は未実装)。
5. 時系列 (日時昇順) 80/20 or 指定比率で前方=Train / 後方=Test 分割。

プロジェクト単位で評価したい場合は、`evaluate_future_window_retention.py` に `--project <name>` を渡して changes をフィルタしてください。
例: uv run python scripts/evaluate_future_window_retention.py --developers data/processed/unified/all_developers.json --changes data/processed/unified/all_reviews.json --project openstack/nova --horizon-days 90 --max-snapshots-per-dev 4 --test-ratio 0.3

### 5. ネガティブ不足問題 & 拡張

初期 30 日モデルでは test pos_rate=0.95 (負例=1) → 汎化指標不安定。
対策 (オプション): プロジェクト指定の上で以下のネガティブ拡張を有効化。

- `--add-gap-negatives`: 連続活動間ギャップが horizon を超える区間に中点付近 negative を挿入 (1 開発者あたり最大 K)。
- `--add-tail-negative`: 終端 (dataset_end まで) に horizon 以上のギャップがある場合 tail 内部に negative。
  結果: テスト負例が増加し pos_rate が現実的水準 (≈0.68) に低下。

### 6. Future Window 評価 (openstack/nova, max_snapshots_per_dev=8)

#### (A) ネガ拡張前

| Horizon | Test Count | pos_rate |  Acc |  Prec | Recall |    F1 |   AUC | Brier |
| ------- | ---------: | -------: | ---: | ----: | -----: | ----: | ----: | ----: |
| 14d     |         20 |     0.70 | 0.40 | 0.667 |  0.286 | 0.400 | 0.637 | 0.404 |
| 21d     |         20 |     0.80 | 0.60 | 0.786 |  0.688 | 0.733 | 0.500 | 0.242 |
| 30d     |         20 |     0.95 | 0.80 | 1.000 |  0.789 | 0.882 | 0.895 | 0.136 |

課題: 30d は容易すぎ (正例飽和)。14/21d は負例増だが特徴不足で AUC 低下。

#### (B) ネガティブ拡張後 (30d)

| Metric               | Value |
| -------------------- | ----: |
| Snapshot Total       |    92 |
| Test Count           |    28 |
| pos_rate (test)      | 0.679 |
| Accuracy             | 0.714 |
| Precision (label=1)  | 0.867 |
| Recall (label=1)     | 0.684 |
| F1 (label=1)         | 0.765 |
| AUC                  | 0.830 |
| Brier                | 0.185 |
| Added Gap Negatives  |    20 |
| Added Tail Negatives |     7 |

混同行列 (test 28): TP=13 / FP=2 / TN=7 / FN=6

離脱(0) 視点 (簡易):

- Recall_0 (TN/(TN+FP)) ≈ 0.778
- Precision_0 (TN/(TN+FN)) ≈ 0.538

誤分類詳細:

- FP (過検知): 確率 0.62〜0.68 (閾値近辺 / 閾値上げで抑制可能)
- FN (見逃し): p=0.24〜0.47 の中域 (最近活動勢いを捉えきれていない)

確率帯精度 (30d after augmentation):
| Prob Range | Count | Acc | Pos Rate |
|------------|------:|----:|---------:|
| 0.3–0.5 | 5 | 0.20 | 0.80 |
| 0.5–0.7 | 6 | 1.00 | 1.00 |
| 0.7–0.85 | 4 | 1.00 | 1.00 |
| 0.85–1.0 | 5 | 1.00 | 1.00 |
→ 誤りは 0.5 未満帯に集中 (閾値調整 / 特徴追加余地)。

### 7. Future vs Gap Alignment (30d future vs 60d gap)

出力: `outputs/prediction_vs_reality_analysis/future_gap_summary.json`
| 指標 | 値 |
|------|----:|
| aligned_pairs | 16 |
| uncertain_future (|p-0.5|<=0.1) | 5 |
| gap_high_conf_fp (gap label=0 & p>=0.8) | 5 |
| label_mismatch | 1 |
ギャップ側で高確信正例だが future が低めになる開発者 (例: wesley.hershberger@...) で recent 活動勾配特徴欠落を示唆。

### 8. 現行特徴の制限

- 主に累積カウント (changes_authored / reviewed / insertions / deletions) の割合スケール。
- 欠落: current*gap_days, recent_activity_count*{7,14,30}, activity ratios, decay weighted activity, authored vs reviewed recent mix。
- スナップショット特徴は“近似”であり厳密な過去再集計ではない (リーク低減は部分的)。

### 9. 改善計画 (提案順)

1. 追加派生特徴: gap, recent window counts, ratios, decay score。
2. ネガ多様化: ギャップ内複数位置 / horizon 境界直前 negative。
3. 閾値最適化: 0.5 → F1 / Youden J ベース (暫定 0.53–0.55 試行)。
4. キャリブレーション: Isotonic / Platt で Brier & 中域信頼性改善。
5. Horizon 再探索: 特徴強化後に 21d 再評価 (離脱早期検知向上)。
6. Feature importance / permutation で寄与度検証 & pruning。
7. 真の履歴再集計 (累積ではなく: 過去スナップショット時点までの正確計数)。

### 10. 現状まとめ

- 以前の“全員継続” に近い極端分布を是正し、離脱 (0) も一定精度 (Recall_0≈0.78) で検出可能に到達。
- ただし 中域 (p<0.5 帯) の FN が多く、短期勢いを表す局所特徴不足が主要ボトルネック。
- 次ステップは「局所活動ダイナミクス特徴 + 閾値/キャリブ最適化」で FN/FP を同時低減し AUC / Brier 安定化を狙う段階。

---

更新日: 2025-09-05
担当: 自動生成レポート
