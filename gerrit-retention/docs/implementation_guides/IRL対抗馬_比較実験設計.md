# IRL対抗馬: 比較実験設計

## 📋 概要

現在のIRL+LSTMモデル（AUC-PR 0.718）の有効性を検証するため、複数の対抗馬（ベースライン）との比較実験を設計する。

---

## 🎯 対抗馬の分類

### Tier 1: 必須ベースライン（最優先）

簡単に実装でき、必ず比較すべき手法

### Tier 2: 重要ベースライン（高優先）

時系列モデルとの公平な比較に必要

### Tier 3: 発展的ベースライン（中優先）

より深い洞察を得るための手法

---

## 📊 Tier 1: 必須ベースライン

### 1.1 ランダム予測（Random Baseline）

**概要**: 全員に0.5の確率を割り当てる

**実装難易度**: ★☆☆☆☆（超簡単）

**実装時間**: 10分

**期待性能**: AUC-PR ≈ 0.35（正例率に依存）

**コード例**:
```python
def random_baseline(test_data):
    """ランダムベースライン"""
    import numpy as np

    y_true = test_data['continued'].values
    y_pred = np.random.uniform(0.4, 0.6, len(y_true))  # 0.5付近のランダム値

    return {
        'predictions': y_pred,
        'method': 'random'
    }
```

**意義**:
- 最低限の性能基準
- "何もしない"場合との比較

---

### 1.2 単純ルールベース（Rule-based Baseline）

**概要**: 簡単なif-thenルールで予測

**実装難易度**: ★★☆☆☆（簡単）

**実装時間**: 30分

**期待性能**: AUC-PR ≈ 0.45-0.55

**ルール例**:
```python
def rule_based_baseline(developer, activity_history):
    """
    ルールベース予測

    Rule 1: 経験 > 200件 → 継続確率0.8
    Rule 2: 受諾率 > 20% → 継続確率0.7
    Rule 3: 最近30日の活動 > 5件 → 継続確率0.6
    Rule 4: それ以外 → 継続確率0.3
    """
    score = 0.3  # ベーススコア

    # ルール適用
    if developer['experience'] > 200:
        score = max(score, 0.8)

    if developer['acceptance_rate'] > 0.2:
        score = max(score, 0.7)

    recent_activities = [a for a in activity_history
                         if (datetime.now() - a['timestamp']).days <= 30]
    if len(recent_activities) > 5:
        score = max(score, 0.6)

    return {
        'continuation_probability': score,
        'method': 'rule_based',
        'applied_rules': []  # どのルールが適用されたか記録
    }
```

**意義**:
- ドメイン知識のみで到達可能な性能
- 解釈性が高い
- 実務での最低ライン

---

### 1.3 ロジスティック回帰（Logistic Regression）

**概要**: 伝統的な統計モデル

**実装難易度**: ★★☆☆☆（簡単）

**実装時間**: 1-2時間

**期待性能**: AUC-PR ≈ 0.55-0.65

**実装**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def logistic_regression_baseline(train_data, test_data):
    """ロジスティック回帰ベースライン"""

    # 特徴量抽出（時系列を集約）
    def extract_features(data):
        features = []
        for _, row in data.iterrows():
            # スナップショット特徴量（時系列を集約）
            feat = [
                row['experience_days'] / 730.0,
                row['total_reviews'] / 500.0,
                row['acceptance_rate'],
                row['recent_activity_frequency'],
                row['avg_activity_gap'] / 60.0,
                row['collaboration_score'],
                row['code_quality_score'],
                # 時系列統計量
                np.mean([a['intensity'] for a in row['activity_history']]),
                np.std([a['intensity'] for a in row['activity_history']]),
                len(row['activity_history'])
            ]
            features.append(feat)
        return np.array(features)

    X_train = extract_features(train_data)
    y_train = train_data['continued'].values

    X_test = extract_features(test_data)

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 訓練
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # クラス不均衡対応
        random_state=42
    )
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict_proba(X_test)[:, 1]

    return {
        'predictions': y_pred,
        'model': model,
        'feature_importance': model.coef_[0],  # 係数＝特徴量重要度
        'method': 'logistic_regression'
    }
```

**意義**:
- 統計学の標準手法
- 特徴量重要度が解釈可能
- 多くの研究で使われるベースライン

**利点**:
- 訓練が高速
- 過学習しにくい
- 解釈性が高い

---

### 1.4 ランダムフォレスト（Random Forest）

**概要**: アンサンブル学習による非線形モデル

**実装難易度**: ★★☆☆☆（簡単）

**実装時間**: 1-2時間

**期待性能**: AUC-PR ≈ 0.60-0.70

**実装**: ✅ **実装済み** (`src/gerrit_retention/baselines/random_forest.py`)

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest_baseline(train_data, test_data):
    """ランダムフォレストベースライン"""

    # 特徴量抽出（ロジスティック回帰と同じ）
    X_train = extract_features(train_data)
    y_train = train_data['continued'].values
    X_test = extract_features(test_data)

    # 訓練
    model = RandomForestClassifier(
        n_estimators=100,        # 木の数
        max_depth=None,          # 深さ制限なし
        min_samples_split=2,
        max_features='sqrt',     # sqrt(n_features)を使用
        class_weight='balanced', # クラス不均衡対応
        oob_score=True,          # Out-of-Bag評価
        n_jobs=-1,               # 並列処理
        random_state=42
    )
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict_proba(X_test)[:, 1]

    return {
        'predictions': y_pred,
        'model': model,
        'feature_importance': model.feature_importances_,
        'oob_score': model.oob_score_,  # OOB精度
        'method': 'random_forest'
    }
```

**意義**:
- **XGBoostより実装が簡単** - ハイパーパラメータ調整が少ない
- 非線形関係を捕捉できる
- 特徴量重要度が自然に得られる
- 過学習に強い（アンサンブル効果）

**利点**:
- ロバスト性が高い
- アンサンブル学習の効果を検証
- OOB scoreで追加の検証データ不要
- 並列処理で高速（n_jobs=-1）

**XGBoostとの比較**:
| 特性 | ランダムフォレスト | XGBoost |
|------|-------------------|---------|
| 実装難易度 | ★★☆☆☆ | ★★★☆☆ |
| チューニング | 簡単 | やや複雑 |
| 期待性能 | 0.60-0.70 | 0.65-0.75 |
| 訓練時間 | 速い | やや遅い |

**推奨理由**:
1. ロジスティック回帰（線形）とXGBoost（勾配ブースティング）の中間
2. 非線形モデルの基準として重要
3. 実装が簡単で再現性が高い

---

## 📊 Tier 2: 重要ベースライン

### 2.1 勾配ブースティング（XGBoost/LightGBM）

**概要**: 高性能な機械学習手法

**実装難易度**: ★★★☆☆（中程度）

**実装時間**: 3-4時間

**期待性能**: AUC-PR ≈ 0.65-0.75（IRLと同等の可能性）

**実装**:
```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def xgboost_baseline(train_data, test_data):
    """XGBoostベースライン"""

    # 特徴量抽出（時系列を集約 + より多くの統計量）
    def extract_features_advanced(data):
        features = []
        for _, row in data.iterrows():
            history = row['activity_history']

            feat = [
                # 基本統計
                row['experience_days'] / 730.0,
                row['total_reviews'] / 500.0,
                row['acceptance_rate'],
                row['recent_activity_frequency'],

                # 時系列統計（平均・標準偏差・最大・最小）
                np.mean([a['intensity'] for a in history]) if history else 0,
                np.std([a['intensity'] for a in history]) if history else 0,
                np.max([a['intensity'] for a in history]) if history else 0,
                np.min([a['intensity'] for a in history]) if history else 0,

                # トレンド（直近 vs 過去の比較）
                np.mean([a['intensity'] for a in history[-10:]]) if len(history) >= 10 else 0,
                np.mean([a['intensity'] for a in history[-30:-10]]) if len(history) >= 30 else 0,

                # 活動パターン
                len(history),
                len([a for a in history if a['collaboration'] > 0.5]) / max(len(history), 1),

                # 受諾率の時系列
                np.mean([a.get('accepted', 0) for a in history]) if history else 0,
                np.std([a.get('accepted', 0) for a in history]) if history else 0
            ]
            features.append(feat)
        return np.array(features)

    X_train = extract_features_advanced(train_data)
    y_train = train_data['continued'].values
    X_test = extract_features_advanced(test_data)

    # ハイパーパラメータチューニング
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42
    )

    # グリッドサーチ
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,
        scoring='average_precision',  # AUC-PR
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # 最良モデルで予測
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict_proba(X_test)[:, 1]

    return {
        'predictions': y_pred,
        'model': best_model,
        'feature_importance': best_model.feature_importances_,
        'best_params': grid_search.best_params_,
        'method': 'xgboost'
    }
```

**意義**:
- **最強のベースライン**（IRLより高性能の可能性）
- Kaggleコンペで頻繁に優勝
- 時系列を集約した場合の上限性能を示す

**重要**: XGBoostがIRLより高性能なら、時系列モデリングの価値を再考する必要がある

---

### 2.2 LSTM（IRLなし）

**概要**: 時系列モデリングだがIRLを使わない

**実装難易度**: ★★★☆☆（中程度）

**実装時間**: 1日

**期待性能**: AUC-PR ≈ 0.65-0.72

**実装**:
```python
import torch
import torch.nn as nn

class VanillaLSTMClassifier(nn.Module):
    """純粋なLSTM分類器（IRLなし）"""

    def __init__(self, feature_dim=15, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence):
        """
        Args:
            sequence: [batch, seq_len, feature_dim]
        Returns:
            prob: [batch, 1]
        """
        lstm_out, (h_n, c_n) = self.lstm(sequence)

        # 最終ステップの隠れ状態を使用
        final_hidden = h_n[-1]  # [batch, hidden_dim]

        prob = self.classifier(final_hidden)
        return prob

def vanilla_lstm_baseline(train_data, test_data, epochs=30):
    """純粋LSTM（IRLなし）ベースライン"""

    model = VanillaLSTMClassifier(
        feature_dim=15,  # 状態10次元 + 行動5次元
        hidden_dim=128,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 訓練ループ
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for trajectory in train_data:
            # 特徴量シーケンス構築
            sequence = build_sequence(trajectory)  # [1, seq_len, 15]
            label = torch.tensor([[1.0 if trajectory['continued'] else 0.0]])

            # 前向き計算
            pred = model(sequence)
            loss = criterion(pred, label)

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_data):.4f}")

    # 評価
    model.eval()
    predictions = []
    with torch.no_grad():
        for trajectory in test_data:
            sequence = build_sequence(trajectory)
            pred = model(sequence)
            predictions.append(pred.item())

    return {
        'predictions': np.array(predictions),
        'model': model,
        'method': 'vanilla_lstm'
    }
```

**意義**:
- **IRLの価値を測定**（LSTM vs IRL+LSTM）
- 時系列モデリングの効果を分離
- もしVanilla LSTMがIRL+LSTMと同等なら、IRLは不要

**重要**: これが最も重要な比較！

---

### 2.3 Transformer（IRLなし）

**概要**: 現代的な時系列モデル

**実装難易度**: ★★★★☆（やや難）

**実装時間**: 2-3日

**期待性能**: AUC-PR ≈ 0.68-0.75

**実装**:
```python
class TransformerClassifier(nn.Module):
    """Transformerベース分類器"""

    def __init__(self, feature_dim=15, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        # 入力埋め込み
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence):
        """
        Args:
            sequence: [batch, seq_len, feature_dim]
        Returns:
            prob: [batch, 1]
        """
        # 埋め込み + Positional Encoding
        x = self.input_projection(sequence)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)

        # Transformer
        transformer_out = self.transformer(x)  # [batch, seq_len, d_model]

        # 最終ステップ（または平均）を使用
        final_repr = transformer_out[:, -1, :]  # [batch, d_model]

        # 分類
        prob = self.classifier(final_repr)
        return prob

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]
```

**意義**:
- 最新のアーキテクチャとの比較
- Attention機構の効果を検証
- IRLなしでどこまで行けるか

---

## 📊 Tier 3: 発展的ベースライン

### 3.1 生存分析（Survival Analysis）

**概要**: 時間を明示的にモデル化

**実装難易度**: ★★★★☆（やや難）

**実装時間**: 3-4日

**期待性能**: AUC-PR ≈ 0.60-0.70

**手法**: Cox比例ハザードモデル

**実装**:
```python
from lifelines import CoxPHFitter
import pandas as pd

def survival_analysis_baseline(train_data, test_data):
    """生存分析ベースライン（Cox比例ハザードモデル）"""

    # データ形式変換
    def prepare_survival_data(data):
        """
        継続予測を生存分析の形式に変換

        duration: 観測期間（継続した場合は打ち切り）
        event: 離脱したかどうか（0=継続中, 1=離脱）
        """
        survival_data = []
        for _, row in data.iterrows():
            survival_data.append({
                'duration': row['observation_months'],  # 観測期間
                'event': 0 if row['continued'] else 1,  # 離脱=1
                'experience': row['experience_days'] / 730.0,
                'total_reviews': row['total_reviews'] / 500.0,
                'acceptance_rate': row['acceptance_rate'],
                'recent_activity': row['recent_activity_frequency'],
                'collaboration': row['collaboration_score']
            })
        return pd.DataFrame(survival_data)

    train_df = prepare_survival_data(train_data)
    test_df = prepare_survival_data(test_data)

    # Cox比例ハザードモデル
    cph = CoxPHFitter()
    cph.fit(
        train_df,
        duration_col='duration',
        event_col='event'
    )

    # 予測（生存確率）
    survival_probs = cph.predict_survival_function(test_df).iloc[-1].values

    return {
        'predictions': survival_probs,
        'model': cph,
        'hazard_ratios': cph.hazard_ratios_,  # 各特徴量のハザード比
        'method': 'cox_ph'
    }
```

**意義**:
- 時間を明示的にモデル化
- 医学・信頼性工学で標準的
- ハザード比で解釈可能

---

### 3.2 グラフニューラルネットワーク（GNN）

**概要**: 開発者間の協力ネットワークをモデル化

**実装難易度**: ★★★★★（難）

**実装時間**: 1週間

**期待性能**: AUC-PR ≈ 0.65-0.75

**手法**: GraphSAGE または GAT

**実装概要**:
```python
import torch_geometric as pyg
from torch_geometric.nn import SAGEConv, GATConv

class DeveloperGNN(nn.Module):
    """開発者ネットワークのGNN"""

    def __init__(self, node_features=10, hidden_dim=128):
        super().__init__()

        # Graph Convolution Layers
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, node_features]
            edge_index: [2, num_edges]（協力関係のエッジ）
        """
        # グラフ畳み込み
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # 分類
        prob = self.classifier(x)
        return prob

def build_collaboration_graph(data):
    """協力ネットワークグラフを構築"""

    # ノード: 開発者
    # エッジ: 共同でレビューした関係

    developers = list(set(r['reviewer'] for r in data))
    dev_to_idx = {dev: i for i, dev in enumerate(developers)}

    # ノード特徴量
    node_features = []
    for dev in developers:
        dev_data = [r for r in data if r['reviewer'] == dev]
        features = extract_developer_features(dev_data)
        node_features.append(features)

    # エッジ（協力関係）
    edges = []
    for r in data:
        if 'collaborators' in r:
            for collab in r['collaborators']:
                if collab in dev_to_idx:
                    edges.append([dev_to_idx[r['reviewer']], dev_to_idx[collab]])

    return torch.tensor(node_features), torch.tensor(edges).t()
```

**意義**:
- ネットワーク効果を捕捉
- 孤立した開発者の離脱リスクを検出
- 最先端の研究動向

---

## 🔬 比較実験設計

### 実験プロトコル

```python
# scripts/experiments/baseline_comparison.py

def comprehensive_baseline_comparison(data, output_dir):
    """全ベースラインとの包括的比較"""

    methods = {
        # Tier 1: 必須
        'random': random_baseline,
        'rule_based': rule_based_baseline,
        'logistic_regression': logistic_regression_baseline,

        # Tier 2: 重要
        'xgboost': xgboost_baseline,
        'vanilla_lstm': vanilla_lstm_baseline,
        'transformer': transformer_baseline,

        # Tier 3: 発展的
        'survival_analysis': survival_analysis_baseline,

        # 現在のモデル
        'irl_lstm': current_irl_model
    }

    results = {}
    for name, method in methods.items():
        print(f"\n{'='*60}")
        print(f"評価中: {name}")
        print(f"{'='*60}")

        # 訓練・予測
        predictions = method(train_data, test_data)

        # 評価
        metrics = evaluate_predictions(
            y_true=test_data['continued'],
            y_pred=predictions['predictions']
        )

        results[name] = {
            'metrics': metrics,
            'predictions': predictions
        }

        print(f"AUC-PR: {metrics['auc_pr']:.3f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")

    # 結果を保存
    save_comparison_results(results, output_dir)

    # 可視化
    plot_comparison(results, output_dir)

    return results
```

### 評価指標

すべてのベースラインで以下を測定：

| 指標 | 説明 |
|------|------|
| **AUC-PR** | メイン指標（不均衡データ） |
| **AUC-ROC** | 補助指標 |
| **F1スコア** | Precision/Recallのバランス |
| **Precision** | 継続予測の正解率 |
| **Recall** | 実際の継続者の捕捉率 |
| **訓練時間** | モデル訓練の所要時間 |
| **推論時間** | 1サンプルあたりの予測時間 |
| **モデルサイズ** | メモリ使用量 |

### 統計的検定

```python
from scipy.stats import wilcoxon

def statistical_test(irl_predictions, baseline_predictions, y_true):
    """
    Wilcoxon符号順位検定でIRLとベースラインの性能差を検定
    """

    # 各サンプルでのAUC-PR差を計算（ブートストラップ）
    n_bootstrap = 1000
    irl_scores = []
    baseline_scores = []

    for _ in range(n_bootstrap):
        # ブートストラップサンプリング
        indices = np.random.choice(len(y_true), len(y_true), replace=True)

        irl_auc = average_precision_score(
            y_true[indices],
            irl_predictions[indices]
        )
        baseline_auc = average_precision_score(
            y_true[indices],
            baseline_predictions[indices]
        )

        irl_scores.append(irl_auc)
        baseline_scores.append(baseline_auc)

    # Wilcoxon検定
    statistic, p_value = wilcoxon(irl_scores, baseline_scores)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'irl_mean': np.mean(irl_scores),
        'baseline_mean': np.mean(baseline_scores),
        'improvement': np.mean(irl_scores) - np.mean(baseline_scores)
    }
```

---

## 📈 可視化

### 性能比較グラフ

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(results, output_dir):
    """ベースライン比較の可視化"""

    # 1. 棒グラフ（AUC-PR比較）
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    methods = list(results.keys())
    metrics = ['auc_pr', 'auc_roc', 'f1', 'precision']
    titles = ['AUC-PR', 'AUC-ROC', 'F1 Score', 'Precision']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        values = [results[m]['metrics'][metric] for m in methods]
        colors = ['red' if m == 'irl_lstm' else 'skyblue' for m in methods]

        ax.barh(methods, values, color=colors)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1.0)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # 値をラベル表示
        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison_bars.png', dpi=300)

    # 2. レーダーチャート
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['AUC-PR', 'AUC-ROC', 'F1', 'Precision', 'Recall']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method in ['irl_lstm', 'xgboost', 'vanilla_lstm']:
        values = [
            results[method]['metrics']['auc_pr'],
            results[method]['metrics']['auc_roc'],
            results[method]['metrics']['f1'],
            results[method]['metrics']['precision'],
            results[method]['metrics']['recall']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('性能比較（レーダーチャート）')

    plt.savefig(output_dir / 'baseline_comparison_radar.png', dpi=300)
```

---

## 🎯 推奨実験スケジュール

### フェーズ1: 必須ベースライン（1週間）

| 日 | タスク | 期待所要時間 | 状態 |
|----|--------|-------------|------|
| 1 | ランダム・ルールベース実装 | 2時間 | ⬜ |
| 2 | ロジスティック回帰実装 | 3時間 | ✅ **完了** |
| 3 | ランダムフォレスト実装 | 2時間 | ✅ **完了** |
| 4-5 | XGBoost実装・チューニング | 8時間 | ⬜ |
| 6-7 | Vanilla LSTM実装・訓練 | 16時間 | ⬜ |

### フェーズ2: 発展的ベースライン（2週間）

| 週 | タスク |
|----|--------|
| 1 | Transformer実装 |
| 2 | 生存分析実装 |

### フェーズ3: 論文執筆（1週間）

- 結果分析
- 統計的検定
- 可視化
- 考察執筆

---

## 📊 期待される結果

### 予想される性能順位（AUC-PR基準）

1. **XGBoost**: 0.65-0.75（最強候補）
2. **IRL+LSTM**: **0.718**（現在の最良）
3. **Vanilla LSTM**: 0.65-0.72
4. **Transformer**: 0.68-0.75
5. **ランダムフォレスト**: 0.60-0.70 ✅ **実装済み**
6. **ロジスティック回帰**: 0.55-0.65 ✅ **実装済み**
7. **生存分析**: 0.60-0.70
8. **ルールベース**: 0.45-0.55
9. **ランダム**: 0.35

### 重要な比較

**比較1: IRL+LSTM vs Vanilla LSTM**
- **目的**: IRLの価値を直接測定
- **もし差が小さければ**: IRLは不要（単純なLSTMで十分）

**比較2: IRL+LSTM vs XGBoost**
- **目的**: 時系列モデリングの価値を測定
- **もしXGBoostが勝てば**: 時系列集約で十分（LSTMは不要）

**比較3: IRL+LSTM vs ランダムフォレスト** ✅ **実施可能**
- **目的**: 非線形モデルとの比較
- **ランダムフォレストの利点**: 実装が簡単、解釈性が高い
- **期待**: IRL+LSTMが10-15%上回ることを期待

**比較4: ロジスティック回帰 vs ランダムフォレスト** ✅ **実施可能**
- **目的**: 線形 vs 非線形の効果を測定
- **期待**: 非線形（RF）が5-10%上回る

**比較5: LSTM vs Transformer**
- **目的**: アーキテクチャの選択を検証

---

## 💡 論文への記載例

### Result Section

```markdown
### Baseline Comparison

We compare our IRL+LSTM model against 7 baselines:

**Table: Performance Comparison**

| Method | AUC-PR | AUC-ROC | F1 | Training Time |
|--------|--------|---------|-----|---------------|
| Random | 0.350 | 0.500 | 0.400 | - |
| Rule-based | 0.520 | 0.620 | 0.550 | - |
| Logistic Regression | 0.610 | 0.680 | 0.630 | 2 min |
| Random Forest | 0.650 | 0.720 | 0.660 | 5 min |
| XGBoost | **0.740** | **0.810** | **0.720** | 15 min |
| Vanilla LSTM | 0.680 | 0.750 | 0.670 | 45 min |
| Transformer | 0.705 | 0.770 | 0.690 | 60 min |
| **IRL+LSTM (Ours)** | **0.718** | 0.754 | 0.636 | 50 min |

Our IRL+LSTM model achieves competitive performance with XGBoost
while providing interpretable reward functions. The improvement over
Vanilla LSTM (+3.8% AUC-PR) demonstrates the value of incorporating
inverse reinforcement learning.
```

---

## 🚀 実装コマンド

### ロジスティック回帰とランダムフォレストの実行 ✅ **実装済み**

```bash
# ロジスティック回帰とランダムフォレストを実行
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_experiments/

# ロジスティック回帰のみ
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines logistic_regression \
  --output importants/baseline_experiments/logistic_regression/

# ランダムフォレストのみ
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines random_forest \
  --output importants/baseline_experiments/random_forest/
```

### 今後実装予定のコマンド

```bash
# 全ベースラインを一括実行（今後）
uv run python scripts/experiments/run_baseline_comparison.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 12 \
  --target-months 6 \
  --baselines logistic_regression random_forest xgboost vanilla_lstm \
  --output importants/baseline_experiments/comparison_results/

# 結果の可視化（今後実装）
uv run python scripts/experiments/visualize_baseline_comparison.py \
  --input importants/baseline_experiments/comparison_results/ \
  --output importants/baseline_experiments/comparison_results/figures/

# 統計的検定（今後実装）
uv run python scripts/experiments/statistical_test.py \
  --irl importants/irl_openstack_real/models/irl_h12m_t6m_seq.pth \
  --baselines importants/baseline_experiments/comparison_results/ \
  --output importants/baseline_experiments/comparison_results/statistical_test.json
```

---

## 📝 まとめ

### 実装済みベースライン ✅

1. **ロジスティック回帰** ✅
   - 線形モデルの代表
   - 実装完了、すぐに実験可能
   - 論文で必須のベースライン

2. **ランダムフォレスト** ✅
   - 非線形モデルの代表
   - 実装完了、すぐに実験可能
   - XGBoostより簡単で再現性が高い

### 今後実装すべきベースライン

3. **XGBoost**: 最強候補、IRLの優位性を示すために必須
4. **Vanilla LSTM**: IRLの価値を直接測定（最重要比較）

### これらを実装すれば

- ✅ 線形 vs 非線形の比較が可能（ロジスティック回帰 vs ランダムフォレスト）
- ✅ 論文の信頼性が向上（機械学習の標準ベースラインとの比較）
- ⬜ IRLの貢献を明確化（Vanilla LSTMとの比較が必要）
- ⬜ 時系列モデリングの価値を証明（XGBoostとの比較が必要）

### 次のステップ

1. **すぐに実行可能**: ロジスティック回帰とランダムフォレストで初回実験
2. **優先度高**: XGBoost実装（IRLとの性能差を確認）
3. **最重要**: Vanilla LSTM実装（IRLの価値を証明）

---

**作成日**: 2025年11月4日
**最終更新**: 2025年11月4日（ランダムフォレスト追加）
