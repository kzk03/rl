# Enhanced IRL - ベースライン打倒プロジェクト

## 目的

Random Forest/Logistic Regression ベースラインを超える IRL モデルの開発

## 現状

- **RF 平均 AUC-ROC**: 0.8603
- **LR 平均 AUC-ROC**: 0.8142
- **IRL 平均 AUC-ROC**: 0.8008（現状負けている）

## 目標

**AUC-ROC 0.87 以上**で RF を超える

## 改善戦略

### Phase 1: 時系列特徴量強化 ⭐⭐⭐⭐⭐

- 過去 30 日間の日次受諾率シーケンス
- 過去 30 日間の日次負荷シーケンス
- 受諾パターンの周期性
- 応答時間の変化トレンド

### Phase 2: アテンションメカニズム ⭐⭐⭐⭐⭐

- 時系列の重要な時点に焦点を当てる
- LSTM の出力に対するアテンション層

### Phase 3: コンテキスト情報追加 ⭐⭐⭐

- 依頼者との協力関係
- 変更内容の特性
- チーム状況

## ディレクトリ構造

```
experiments/enhanced_irl/
├── README.md                          # このファイル
├── models/
│   ├── __init__.py
│   ├── attention_irl.py              # アテンション付きIRLモデル
│   └── temporal_feature_extractor.py # 時系列特徴量抽出
├── scripts/
│   ├── train_enhanced_irl.py         # 訓練スクリプト
│   └── run_cross_eval.py             # クロス評価スクリプト
└── configs/
    └── enhanced_irl_config.yaml      # 設定ファイル
```

## 実行方法

### 小規模テスト（推奨）

```bash
# 2組み合わせ、1シード、10エポックで動作確認
# 実行時間: 約5分
cd /Users/kazuki-h/rl/gerrit-retention
cat > /tmp/test_small.py << 'EOF'
import sys
from pathlib import Path
project_root = Path("/Users/kazuki-h/rl/gerrit-retention")
sys.path.insert(0, str(project_root / "experiments/enhanced_irl/scripts"))
import run_multi_seed_eval as script
import pandas as pd

df = pd.read_csv(project_root / "data/review_requests_openstack_multi_5y_detail.csv")
combinations = [(3, 6, 3, 6), (3, 6, 6, 9)]
all_data = {}

for train_s, train_e, eval_s, eval_e in combinations:
    key = (train_s, train_e, eval_s, eval_e)
    all_data[key] = script.prepare_data(
        df, "2021-01-01", "2023-01-01", "2023-01-01", "2024-01-01",
        train_s, train_e, eval_s, eval_e
    )

for train_s, train_e, eval_s, eval_e in combinations:
    X_train_s, X_train_t, y_train, X_test_s, X_test_t, y_test = all_data[(train_s, train_e, eval_s, eval_e)]
    if len(set(y_test)) > 1:
        auc = script.train_and_evaluate(X_train_s, X_train_t, y_train, X_test_s, X_test_t, y_test, seed=777, epochs=10)
        print(f"Train={train_s}-{train_e}m→Eval={eval_s}-{eval_e}m: AUC={auc:.4f}")
EOF
uv run python /tmp/test_small.py
```

### フル実験

```bash
# 10組み合わせ、5シード、50エポック
# 実行時間: 約3-4時間（CPU環境）
cd experiments/enhanced_irl
uv run python scripts/run_multi_seed_eval.py
```

### データ準備のみ確認

```bash
# データ準備の動作確認（トレーニングなし）
# 実行時間: 約2-3分
uv run python /tmp/test_v2.py
```

## 進捗

- [x] Phase 1: 時系列特徴量強化 - **実装完了、月次サンプリング対応完了**
  - TemporalFeatureExtractor 実装（97 次元時系列特徴量）
  - AttentionIRLNetwork 実装（2 層 LSTM + Attention）
  - 月次サンプリング実装（Train データの大幅増加）
  - データリーク防止（最後 6 ヶ月はラベル専用）
  - `pre_filtered`による高速化実装
  - 小規模テスト成功: AUC 0.84-0.74（2 組み合わせ）
- [ ] Phase 2: フル実験実行
  - 10 組み合わせ × 5 シード × 50 エポック
  - 推定実行時間: 3-4 時間
- [ ] Phase 3: 結果分析とベースライン比較
  - RF との勝敗判定
  - 期間ごとの傾向分析
- [ ] Phase 4: ベースライン超え達成または次の改善策

## 実験結果

### Phase 1-A: 時系列特徴量 + Attention（初期評価）

**日時:** 2025-11-08

**モデル仕様:**

- 状態特徴量: 10 次元（既存 IRL と同一）
- 時系列特徴量: 97 次元
  - 過去 30 日の日別受諾率シーケンス (30)
  - 過去 30 日の日別レビュー負荷シーケンス (30)
  - 過去 30 日の日別応答時間シーケンス (30)
  - 曜日別受諾パターン (7)
- アーキテクチャ: 2 層 LSTM (hidden=128) + Attention
- 最適化: Adam (lr=0.001), Dropout=0.3

**単発評価（Train=0-3m → Eval=0-3m）:**

```
AUC: 0.8512
RF baseline: 0.8603
差分: -0.91%
```

**クロス評価（10 組み合わせ）:**

- 実行中...
- 結果は `results/cross_eval_results.csv` に保存予定

**所感:**

- 時系列特徴量の追加だけでは RF に勝てない可能性が高い
- アテンション機構は正常動作（過学習なし、Val AUC 0.8552 達成）
- データサイズが小さい（650 train, 361 test）ため、LSTM の優位性が発揮されにくい
- 次のステップ:
  1. クロス評価結果を待つ
  2. Multi-task learning 追加（応答時間予測など）
  3. より深い LSTM or Transformer アーキテクチャ検討
