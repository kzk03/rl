# 逆強化学習まわりの今後のアイデア集

## 1. 特徴量に依存しない（or 依存を減らす）報酬学習

- **Deep MaxEnt IRL**: 既存の最大エントロピー IRL をニューラルネット化し、非線形な報酬関数を学習する。表現力は高いが収束コストと不安定さに留意。
- **AIRL / GAIL 系**: GAN 風に「専門家 vs ポリシー」識別器を学習し、その内部表現を報酬として扱う。ハンドクラフト特徴が不要になる一方で、ハイパーパラメータ調整や安定化が難しい。
- **Representation + Linear Hybrid**: 自動特徴抽出モジュール（例: オートエンコーダ、シーケンス埋め込み）で低次元表現を作り、その上で現行の線形 IRL を走らせる。既存の L1 制御や係数解釈を活かせる保守的なアプローチ。

## 2. フィーチャー前処理の高度化

- **自動埋め込み生成**: パスやレビュア履歴を RNN/Transformer に通して学習済み埋め込みを作り、IRL へ入力。手作り特徴とのハイブリッドにして精度と安定性のバランスを取る。
- **クラスタリング系特徴**: K-means 等でレビュアー/パスクラスタを作り、その所属確率を特徴に追加。線形 IRL でも扱いやすい非線形情報源になる。

## 3. RL 側の改善（候補数拡大を支える）

- **エピソード拡張と高度なオプティマイザ**: PPO や Advantage Actor-Critic への移行、もしくは現在の簡易 PG でエピソード数・探索温度・entropy bonus を調整して安定化。
- **Policy 入力正規化**: LayerNorm・Feature scaling・マスク付きアテンションなどで候補数増によるスケール差を緩和。

## 4. 評価・キャリブレーションの強化

- **信頼度の再調整**: Policy ログ確率に対して温度スケーリングや Platt scaling を適用し、ECE を改善。IRL 確率 vs Policy 確率の比較評価も継続。
- **Permutation Importance / Ablation**: 特徴ごとの寄与度を定量化し、不必要な特徴を間引く。自動抽出系を導入した場合も説明可能性を確保するために併用。
- **スライディング区間評価**: `task_assignment_replay_eval.py` の `--windows` に `0-3m,3m-6m,...` のような開始・終了指定を渡せるようにしたので、学習直後区間と先の期間を切り分けてドリフトや劣化を把握可能。

## 5. 実験ロードマップ案

1. 既存線形 IRL + 追加埋め込み（ハイブリッド）で様子見。
2. Deep MaxEnt IRL の小規模実験（サブセットで安定性検証）。
3. 安定化できたら AIRL/GAIL を導入し、転移性や Top-k 指標の変化を評価。
4. 常に温度チューニング・校正・リプレイ評価で比較し、ベースライン（手作り特徴＋線形）を上回るか監視。

---

補足: 深層 IRL は高精度が期待できる一方、計算資源・チューニングコスト・可観測性のトレードオフが大きい。段階的に導入し、既存の監視指標（Top-k, mAP, ECE, index0_positive_rate）で効果検証するのが望ましい。

## 6. 1 年スライディング窓での学習・評価フロー

- **学習期間限定**: `build_task_assignment_from_sequences.py` の `--train-window-days` を 365 に設定し、直近 1 年分のタスクのみで IRL を学習。長期ドリフトを避けつつ最新傾向を反映する。
- **評価セット分割**:
  - 期間 A: 学習と同じ 1 年 (`--train` ウィンドウ) で再現率をチェックし、過学習/不足を見極め。
  - 期間 B: `--eval-window-days 365` でその先 1 年間を抽出し、汎化性能を評価。
- **パイプライン手順**:

1.  データ生成: `uv run python scripts/offline/build_task_assignment_from_sequences.py --cutoff <学習終了日> --train-window-days 365 --eval-window-days 365 ...`
2.  IRL 学習 + 温度チューニング: 生成された `tasks_train.jsonl` だけで再学習し、`temperature_tune_irl.py` で最適温度を再評価。
3.  RL 再学習: 新 IRL モデルと調整後の温度を使って `train_task_assignment.py --episodes <必要に応じて増やす> --max-candidates <設定値>` を実行。
4.  リプレイ評価: `task_assignment_replay_eval.py --windows train,12m --max-candidates <設定値>` で同期間・将来期間の指標を比較。

- **評価ポイント**: 行動一致率/Top-k/mAP/ECE の期間別推移を記録し、1 年以上学習したモデルとの差分をダッシュボード化する。

## 7. 1 年学習モデルの温度チューニング結果

- **実験設計**: `outputs/task_assign_1y/tasks_eval.jsonl` を評価セット、`outputs/task_assign_1y/irl_model_l1e-3.json`（17 特徴＋ L1=1e-3）を対象に `scripts/analysis/temperature_tune_irl.py` を実行。温度候補は {0.6, 0.8, 1.0, 1.2, 1.4}。
- **結果概要**:
  - `top1_acc` は全温度で 0.83399 と変化なし（IRL が最上位候補の順位付け自体は一貫）。
  - Expected Calibration Error (ECE) は T=0.8 が最小で 0.03534。T=0.6 では 0.06199、1.0 で 0.04648、1.2 で 0.09157、1.4 で 0.13952。
- **示唆**: 直近 1 年データで校正する際は T=0.8 を採用するのが最も信頼度が高い。RL 側 `irl_softmax` 報酬でもこの温度をデフォルトとしつつ、将来的には policy logits 自体のキャリブレーション（温度スケーリングや Platt scaling）も検討する。
