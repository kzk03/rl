# 長期貢献者予測 戦略ドキュメント

## 目的 (Objective)

一定の観測期間 W 日 (例: 過去 60 日) の開発者活動ログ系列から、将来 H ヶ月 (例: 3 / 6 ヶ月) 時点でその開発者が「長期的貢献者 (active contributor)」であり続けているかを予測するシーケンス → ラベル問題を定義し、段階的に高度化 (Survival / 潜在変数 / RL 介入) するロードマップを示す。

---

## なぜ最初は IRL 特徴量を流用するか

既存逆強化学習 (IRL) で既に安定取得できている特徴 (ギャップ / 時間間隔など) は以下の利点:

- ETL・標準化・スケーラが確立済み → 迅速な MVP
- 行動 (レビュー受理/参加) と関連する「活動リズム」を長期継続にも転用可能
- 初期はシンプルな低次元特徴でリーク検証や評価パイプラインを固めやすい

その後、段階的にリッチな時系列/ネットワーク/テキスト特徴へ拡張する方針。

---

## ラベル定義 (Long-term Contributor Flag)

観測ウィンドウ終了時刻を T0 とする。
Y=1 (長期貢献者) 条件の例 (初期案):

1. T0 から H ヶ月の間に「活動日数 >= θ_days」 (例: 週 1 以上相当)
2. かつ 合計イベント数 (commit + review + comment) >= θ_events
3. かつ 最終イベント時刻 >= T0 + H − δ (離脱未発生; δ はバッファ例: 14 日)

部分観測 (H を完全に覆わない最新開発者) は right-censored として Survival モデル段階で利用、序盤の単純分類では除外 or ラベル欠損。

---

## 観測ウィンドウ & スライディング

- 観測長 W: 30 / 60 / 90 日を実験 (短期 vs 情報量トレードオフ)
- スライド間隔: 7 日 (週次) で複数サンプル化 (同一開発者の時系列スナップショット)
- 特徴作成は T0 時点より未来情報を一切含めない (ログ遅延に備え 24h ラグオプション)

---

## 初期特徴セット (IRL 流用)

`maxent_binary_irl.py` の基礎特徴:
| 名称 | 意味 | 長期継続との仮説的関連 |
|------|------|--------------------------|
| bias | 定数項 | 基準バイアス |
| gap*norm | 直近イベント間隔 / 正規化 | 長い → 離脱リスク上昇 |
| long_gap | ギャップが閾値超過か (0/1) | 長期不活性兆候 |
| gap_log | イベント間隔の log | 極端値の圧縮 |
| extra_numeric* | 状態 dict 中の追加数値 (例: 過去受理率など) | 活動品質/量指標 |

初期実装方針:

1. IRL と同じ feature_fn を再利用し、観測ウィンドウ末尾時点の状態から 1 ベクトル作成
2. オプションでウィンドウ内の (最新 k ステップ) シーケンス化 (k=10 など) → LSTM/Transformer へ
3. まだ存在しないが将来的に追加する可能性が高い extra フィールド: `recent_accept_rate`, `review_latency_avg`, `night_activity_ratio`

### 拡張フェーズで追加予定 (第 2 段階以降)

- Rolling 集約: commits_7d, reviews_14d, comments_30d, pending_reviews, rework_cycles
- 時間パターン: weekend_ratio, night_ratio, circadian_entropy
- トレンド: Δ(commits_7d vs 前 7d), EMA(latency)
- ソーシャル: unique_partners_30d, reciprocity, graph_centrality
- 品質: build_fail_rate, revert_rate
- テキスト: sentiment_mean, frustration_score (Sentence-BERT + 情緒分類)
- 潜在: stress_latent, engagement_latent (VAE / State-space encoder)

---

## One-shot 排除: 時系列 MDP/POMDP 設計

目的は「単発ラベル推定」ではなく「時系列の状態推移を踏まえて将来の継続確率を高める/予測する」こと。以下の定義で one-shot を排除します。

- 時間スケール: 1 ステップ = 1 日 or N イベント（構成に応じて）
- 観測 o_t: t 時点の生ログ由来の観測特徴（活動量、レビュー待ち、応答遅延、夜間比率、テキスト感情 など）
- 状態 s_t: 直近履歴からエンコードした表現（RNN/Transformer で h_t）＋ 潜在 z（stress/engagement）
  - 例: s*t = Encoder(o*{t−k:t})，z*t ~ q(z|o*{≤t})
  - POMDP として belief b*t = f(o*{≤t}) を状態とみなす実装でも可
- 行動 a_t（介入）: レビュー負荷調整(−1/0/+1)、メンターペア付与、タスク難易度調整、休息推奨、レビュー順序変更 等（離散/パラメータ化）
- 報酬 r_t: Retention 重視の複合
  - r*t = w1·Survival_t − w2·Stress_t − w3·負荷不均衡 + w4·Engagement*Δ − c(a_t)
  - 代替: 予測ハザード λ_t を別ヘッドで学習し、(1−λ_t) を価値の割引補正に使う Risk-aware 形式
- 遷移 p(s\_{t+1}|s_t,a_t): 実ログからのオフライン近似（行動は実際の介入ログ or 疑似介入を用いて学習/評価）
- 終了条件: 無活動が連続 L 日（離脱）、観測ホライズン終了、強制トランケーション
- 目的（長期）: J = E[Σ γ^t r_t] かつ H 末時点の長期貢献フラグの上昇

遷移イメージ（ASCII）:

```
時刻:    t            t+1           t+2          ...
観測:   o_t  ---->   o_{t+1}  ----> o_{t+2}
           \            \             \
状態:     s_t --a_t-->  s_{t+1} --a_{t+1}--> s_{t+2}
              \ r_t         \ r_{t+1}
```

---

### 行動設計（離散/パラメータ化）

- レビュー負荷調整: {-2, -1, 0, +1, +2}（週次想定の目標レビュー数に対する増減; 例: ±20%）
- メンタリング割当: {none, mentor_assigned}（将来は mentor_id でパラメータ化）
- タスク難易度: {easy, medium, hard}（難易度カタログを別管理）
- 休息推奨: {none, suggest_1d, suggest_2d}
- レビュー優先度: {expedite, normal, deprioritize}
- 通知/支援: {encourage_msg, offer_help}（軽量介入）

制約と運用:

- 予算/キャパ制約: チーム内で週あたりの総レビュー量の上限、各メンターの担当上限
- クールダウン: 同一開発者への介入間隔 >= Δt を強制
- 公平性: 仕事量分散/Gini を監視し、しきい値超過時は行動空間を制限

### 報酬設計（具体案）

指標は 0〜1 正規化を基本とし、スケール差を抑えます。

- Survival*t: 次時刻の存続/活動フラグ、または Survival ヘッドの継続確率 $
  \hat{p}*{survive}(t)$
- Stress*t: 夜間比率、応答遅延、ネガ感情、タスク切替頻度の合成スコア $z*{stress}(t)$
- 不均衡: 期間内の作業量配分の Gini/variance $
\mathrm{Ineq}(t)$
- Engagement_Δ: 未処理キュー減少、応答速度改善の差分 $
\Delta \mathrm{Eng}(t)$
- 行動コスト: 人的コスト/摩擦を $c(a_t)$ として控除

提案式（初期重みは開発時にグリッドで同定）:

$$
r_t = w_s \cdot \hat{p}_{survive}(t)
      - w_\sigma \cdot z_{stress}(t)
      - w_f \cdot \mathrm{Ineq}(t)
      + w_e \cdot \Delta \mathrm{Eng}(t)
      - c(a_t)
$$

リスク対応（ハザード併用）:

- Survival ヘッドでハザード $\lambda_t$ を推定し、$-w_h \cdot \lambda_t$ を加算、または $\gamma_t = \gamma (1-\lambda_t)$ で時変割引

実務上の注意:

- 早期は $w_s=1.0, w_\sigma=0.5, w_f=0.2, w_e=0.3$ 程度から調整（要検証）
- 各成分は min-max か分位正規化、季節性補正（週末/リリース週）

### 予測は最終的に何に基づく？

- 予測ヘッドは以下の 2 通りを併用可能：
  1. 「現状方針（介入なし/現状ポリシー）」下の長期貢献確率：$P(Y=1\mid s_W, \text{no-intervention})$
  2. 「推奨ポリシー適用」時の長期貢献確率：$P(Y=1\mid s_W, \pi)$（短いロールアウトで近似 or Decision Transformer）
- 製品上は 1) を標準提示、2) をシナリオ比較として提供すると安全（介入が未導入でも価値がある）。

### 行動を「タスク依頼」にするのは変？

- 領域的に「レビュー割当/タスク依頼」は正当な介入の一種。ただし以下を満たすこと：
  - 目的整合: 依頼で過負荷になれば Stress 上昇 →Retention 低下もあり得る。報酬に Stress/公平性を入れてガードする。
  - 制約尊重: 個人/チームの容量制約を MDP に組み込み、過度集中を抑制。
  - 反事実評価: 観測データは選好バイアスを含むため、オフライン RL（CQL/IQL）や Doubly Robust で安全に評価。
- 予測タスクとしては、「依頼をせずとも」状態系列から $P(Y=1)$ を推定できるようにモデルを分離（Multi-head: Survival/Classification と Policy の分離）。依頼を介入として扱うのは RL 側の責務に切り分けるのが無難。

---

## モデリング段階 (Phase Plan)

| Phase | モデル                               | 特徴                          | 目的               |
| ----- | ------------------------------------ | ----------------------------- | ------------------ |
| 0     | Logistic / XGBoost                   | 単一 IRL 特徴スナップショット | MVP, データ検証    |
| 1     | LSTM / Transformer (seq2one)         | IRL 特徴の時系列 (gap 系列)   | 時系列性評価       |
| 2     | Survival (Cox / DeepSurv)            | Phase1 特徴                   | 検閲対応・時間予測 |
| 3     | Multi-task (Label + Hazard + Stress) | 拡張特徴                      | 性能 + 解釈性      |
| 4     | Decision Transformer / RL (介入施策) | Multi-task 表現               | 継続促進介入最適化 |

---

## 評価指標

- 分類: ROC-AUC, PR-AUC, Lift@k, Recall@Precision=0.8
- 時間: C-index, Integrated Brier Score (IBS)
- 早期判別: 観測ウィンドウを短縮 (W/2) した場合の性能比較
- 安定性: 四半期別 AUC 変動, Population drift (feature mean/std)
- 解釈: 単純モデル係数 / SHAP (Phase2 以降)

---

## データフロー (MVP)

1. Raw events 正規化 (commit/review/comment)
2. セッション化 / ソート
3. gap 計算 (timestamp 差分)
4. IRL feature_fn 適用 → ベクトル化
5. ラベル付与 (H horizon)
6. 分割 (開発者単位 / 時間ベース holdout)
7. モデル学習・評価

後続で: rolling feature builder, sequence tensorizer, survival labeler を拡張。

---

## 実装タスク (初期 2 スプリント例)

| 優先 | タスク                        | 出力                                         |
| ---- | ----------------------------- | -------------------------------------------- |
| P0   | IRL 特徴抽出 util 切り出し    | `feature_engineering/irl_feature_adapter.py` |
| P0   | ラベル生成スクリプト          | `labeling/long_term_label.py`                |
| P0   | データスプリット              | train/val/test CSV (IDs)                     |
| P1   | Baseline (Logistic / XGBoost) | metrics.json, feature_importance.csv         |
| P1   | Gap シーケンス LSTM           | seq_model.pt, eval_report.json               |
| P2   | Survival (Cox)                | survival_metrics.json                        |
| P2   | Drift モニタ雛形              | drift_report.json                            |

---

## IRL 特徴流用 実装指針

- 既存 IRL の `default_feature_fn` を独立関数へ (副作用: extra_feature_names の固定化)
- 観測ウィンドウ内「最後のイベント状態」または「代表統計 (中央値イベント)」を選択
- シーケンス版は gap を時間順に並べ (padding) → embedding: log(1+gap) を連続値入力
- 欠損 (イベント < k): 左パディング + mask

---

## リスクと軽減

| リスク                       | 対策                                                          |
| ---------------------------- | ------------------------------------------------------------- |
| ギャップ特徴だけでは情報不足 | 早期に Phase1 で性能指標を測り不足を定量化                    |
| 未来リーク                   | ラベル生成前に T0 + H 以降の行動を除外するテスト (単体テスト) |
| 不均衡 (長期=少数)           | クラス重み, Focal loss, Stratified split                      |
| ドリフト                     | 週次で feature mean/std, AUC 監視                             |
| Cold start                   | 観測日数 < W/2 のサンプル別扱い (flag + 簡易 prior)           |

---

## 拡張ロードマップ (抜粋)

1. Rolling/トレンド特徴追加 → Ablation で寄与評価
2. Stress latent (VAE + proxy) → Multi-task で性能 + 早期警告強化
3. Survival モデルと分類モデルの ensembling (stacking)
4. 介入候補 (負荷調整 / メンタリング) の因果効果推定 (Doubly Robust)
5. Policy 最適化 (Offline RL / Conservative Q-Learning)

---

## 最小コード断片 (概念)

```python
# 既存 IRL feature_fn の再利用例
from gerrit_retention.irl.maxent_binary_irl import MaxEntBinaryIRL

irl = MaxEntBinaryIRL()

def extract_irl_features(event_state_dicts):  # event_state_dicts: 観測ウィンドウ内の状態リスト
    last_state = event_state_dicts[-1]
    vec = irl.default_feature_fn(last_state)  # np.ndarray
    names = irl.feature_names  # 固定化後
    return {n: float(v) for n, v in zip(names, vec)}
```

---

## 成功基準 (Definition of Done for MVP)

- IRL ベース特徴のみで Long-term (H=90d) ラベルに対し ROC-AUC > 0.60 (仮閾値)
- データリーク検証 (単体テスト) パス
- metrics.json + reproducible config (YAML) をコミット
- 次段階の追加特徴リスト & 優先度確定

---

## まとめ

まずは IRL で実績あるギャップ中心特徴をそのまま流用して、ラベル定義/評価フレーム/再現性の土台を固定。次に不足を測定しながら段階的にリッチな特徴 (活動量・ネットワーク・テキスト・潜在ストレス) に拡張し、最終的には Survival + Multi-task + 介入最適化 (RL) へ拡張する。これにより初期投資を抑えつつ、説明性と将来拡張性を両立するロードマップを確保する。

---

(本ドキュメントは初期ドラフト。フィードバックに応じて詳細なスキーマ・具体的閾値・テスト仕様を追記予定。)

---

## FAQ: 予測 vs ポリシー vs オフライン RL

- 短答まとめ

  - 予測は、ポリシーが無くても可能（現状方針=介入なしを前提に P(Y=1|s_W) を出す）。
  - ポリシーは、「どの状態でどの介入を取るか」を決める写像 π(a|s) の学習（RL 側の責務）。
  - オフライン RL は、過去ログ（バッチ）だけで安全にポリシーを学習・評価する枠組み。予測とは別物だが、同じエンコーダを共有しても良い。

- 用語の整理

  - 予測（Prediction/Survival）: 状態系列から将来の長期貢献確率 Y を推定する監督学習。介入なし前提が標準。
  - ポリシー（Policy）: 状態 → 行動を出す意思決定ルール。目的は長期報酬（Retention 重視の複合）最大化。
  - オフライン RL（Offline RL）: 実運用で試さず、既存ログのみでポリシーを学習（CQL/IQL 等）し、OPE（WIS/FQE/DR）で価値を見積もる。

- 貢献者の長期予測は、最終的に何に基づく？

  - 基本は P(Y=1 | s_W, no-intervention)。すなわち「今このまま」の前提での長期貢献確率。
  - 介入を前提にした確率 P(Y=1 | s_W, π) は、短いロールアウトやモデルベース近似でシナリオ比較として提供（過信しない）。

- 行動を「タスク依頼」にするのは変？

  - 変ではない。正当な介入。ただし Stress/公平性/コスト/制約を報酬と MDP に反映して、過負荷化を避けること。
  - 予測（P(Y=1)）と依頼の意思決定（π）は責務分離すると安全（まず予測の品質を固め、その後に RL を導入）。

- RL 側の責務（何をやる？）

  - 行動空間と制約の定義（負荷調整、メンタリング、休息、優先度 など）。
  - 報酬設計と安全ガード（Stress/公平性/コスト）。
  - ログからのポリシー学習（CQL/IQL など）とオフポリシー評価（WIS/FQE/DR）。
  - 本番導入の段階的 rollout（shadow → canary → ramp）と逸脱検知（stress 悪化時のフォールバック）。

- 予測はポリシーに基づくの？

  - 原則 No。予測は現状方針を暗黙の基準として学習する。ポリシーに依存しないため、介入未導入でも有用。
  - ただし「ポリシー条件付き予測」は可能（環境/動学の近似が前提）。これは分析用オプションと位置づける。

- 実務のすすめ方（安全な分割統治）
  1. 予測モデル: P(Y=1|s_W) と Survival/ハザードを確立（データリーク検証）。
  2. 共有エンコーダ上に RL: CQL/IQL で π をオフライン学習、OPE で価値推定。
  3. シナリオ比較: P(Y=1|s_W, π) を短ロールアウトで参考提示（不確実性も表示）。
  4. 段階導入と監視: Stress/公平性の SLO を満たす範囲で限定展開。

---

### 介入なし（no-intervention）の予測確率は何に基づく？

要点：観測ウィンドウ末の状態系列 s_W を入力として、将来 H での長期貢献ラベル Y を教師信号にした監督学習（分類/サバイバル）で推定します。学習データは実世界の「現状方針（実際に起きた介入の混合）」の結果を含みますが、推論時は追加の介入を仮定せず、そのままのダイナミクスを前提に P(Y=1|s_W) を返します。

- ラベル（教師信号）

  - 範囲: T0（観測末）から H ヶ月
  - 条件例（調整可能）: 活動日数 ≥ θ_days, イベント数 ≥ θ_events, 最終活動が T0+H−δ 以内
  - 検閲: H を覆えないケースは right-censored（サバイバルで使用、分類では除外/NaN）

- 入力（特徴）

  - ベース: IRL 流用の gap 系＋追加数値（初期 MVP）
  - 発展: rolling 集約、トレンド、時間帯パターン、ソーシャル、テキスト、潜在 z（stress/engagement）
  - 形式: スナップショット（集約）またはシーケンス（RNN/Transformer）

- モデルファミリ（いずれか/併用）

  1. 分類: ロジスティック/XGBoost/MLP（seq2one）で P(Y=1|s_W) を直接学習
  2. サバイバル: Cox/DeepSurv/DeepHit で時間依存の離脱ハザード λ(t|s_W) を学習し、H 時点の生存確率 S(H)=exp(−∫λ) を算出
     - 実務上は S(H) を「長期貢献確率」として扱う

- 学習・評価

  - Split: 開発者単位の 時間分割（最新期間をテスト）
  - 不均衡対策: クラス重み/Focal loss、閾値最適化（Precision 固定など）
  - 指標: ROC-AUC / PR-AUC、C-index / IBS（サバイバル）、Calibration
  - リーク検証: 特徴が T0 以降を参照していないことの単体テスト

- 数式のイメージ

  - 分類: \( \hat{p} = P\_\theta(Y=1 \mid s_W) \)
  - サバイバル: \( S*\theta(H \mid s_W) = \exp(-\int_0^H \hat{\lambda}*\theta(t \mid s_W) dt) \)

- 重要な含意
  - 「介入なし」は “新たに提案する介入を仮定しない” 意味で、データに存在する現実の運用方針の影響は背景に含まれます。
  - 後で「ポリシー条件付き P(Y=1|s_W, π)」を出す場合は、環境/動学の近似（短ロールアウト、モデルベース、DT）が別途必要になります。

---

## 予測への RL/IRL 活用アイディア

- IRL 由来の特徴量を注入（安全・低リスク）

  - MaxEntBinaryIRL の出力を特徴として付与: p_engage = sigmoid(θ·φ(s)), logit_engage = θ·φ(s)
  - 追加で、過去 W 期間の平均/分散、最近のトレンド Δp、エントロピー H(p)などを派生特徴に
  - 実装先候補: `src/gerrit_retention/prediction/retention_predictor.py` の `RetentionFeatureExtractor`

- OPE（オフポリシー評価）知見を事前分布に活用（中リスク）

  - 既存運用ポリシー π_b の下での価値/ハザード推定結果を、予測器の校正や事前重み付けに反映
  - 例: FQE で推定した将来報酬の期待値を特徴化、または確率の温度スケーリングに利用

- ポリシー条件付き確率 P(Y=1|s_W, π) の推定（拡張）
  - モデルベース短ロールアウト: 学習済み遷移モデル g(s,a) と IRL 報酬でシミュレーション → サンプル平均
  - 代替: 決定トランスフォーマー/BCQ/IQL/CQL で反実仮想を近似し、OPE（FQE/DR/WIS）でバイアスを制御
  - 注意: 分布外行動・オフサポートに要注意。行動マスクやリジェクションサンプリングで安全側に。

## 付録: 現行 RL/IRL の状態・報酬定義（コード根拠）

- 環境と報酬（レビュア割当て）

  - `src/gerrit_retention/rl_environment/multi_reviewer_assignment_env.py`
    - 観測: 候補レビュアごとの特徴を `feature_order` 固定順でベクトル化
    - 報酬モード:
      - `match_gt`: 教師データ一致
      - `irl_softmax`: IRL 効用の softmax (選択確率)
      - `accept_prob`: IRL ロジットの sigmoid (受理確率)
    - 連続性ボーナス等の追加あり
    - IRL 効用: `_irl_utility` で θ·[x;1]（標準化スケーラ込み）

- IRL（MaxEnt 二値）

  - `src/gerrit_retention/irl/maxent_binary_irl.py`
    - 目的: P(a=1|s) = sigmoid(w·φ(s))（ロジスティック同等）
    - 既定特徴: gap 系（gap_norm/long_gap/gap_log）＋動的に発見される数値特徴
    - 学習: ロジ損 + L2, 早期停止, 標準化対応
    - 予測: `predict_proba(states) -> List[float]`

- IRL 報酬の注入ラッパ

  - `src/gerrit_retention/rl_environment/irl_reward_wrapper.py`
    - ベース報酬を IRL 出力で置換/ブレンドし、エンゲージメントボーナス等を付加

- 予測器との接続（本ドキュメントで追加）
  - IRL 特徴アダプタ: `src/gerrit_retention/prediction/irl_feature_adapter.py`
    - 返却: [p_engage, logit_engage, entropy]
    - `RetentionPredictor` に設定で取り込み可能（`feature_extraction.irl_features.enabled: true`）
    - 学習済 IRL の差し込み: `RetentionPredictor.set_irl_model(irl)`
