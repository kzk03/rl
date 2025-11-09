# IRL 専用の解析・実験ハブ

このディレクトリは、IRL ベースのレビュー受諾予測モデルを集中的に調整・評価するための作業場です。ベースライン（Logistic Regression や Random Forest）には触らず、IRL 系の改善のみを反復できるように以下を提供します。

- `run_irl_sensitivity.py` : 代表的なハイパーパラメータ（隠れ次元、学習率、dropout、対象期間など）を一括で試せるコマンドラインツール
- `experiments/` : 各実験の成果物（メトリクス、閾値、ログ）を保存するフォルダ
- `configs/` : 追加の設定テンプレートを配置する場所（必要に応じて拡張）

## 推奨ワークフロー

1. 事前に `uv pip install -r requirements_api.txt` を実行して依存関係をそろえる
2. IRL 感度分析を走らせる
   ```bash
   uv run python analysis/irl_only_experiments/run_irl_sensitivity.py --preset quick
   ```
3. 出力は `analysis/irl_only_experiments/experiments/<experiment_name>/` に集約されます
4. 追加で試したい設定があれば、`--hidden-dim` や `--dropout` 等を指定して再実行

## 既定の実験プリセット

| プリセット | 説明                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------- |
| `quick`    | 9-12 ヶ月訓練ウィンドウを除外しつつ、hidden_dim/learning_rate/dropout を少数パターン確認 |
| `extended` | quick に加え、seq_len や output_temperature の調整、正則化強化などを網羅                 |

出力には以下が含まれます。

- `metrics.json` : AUC/PR/F1 等の評価指標
- `threshold.json` : 訓練セットで最適化した閾値と確率統計
- `summary.md` : 実験条件と主要な結果のサマリ
- `model/` : 保存した IRL モデル（必要に応じて）

## 注意事項

- 実験は IRL のみを対象とします。ベースラインと比較したくなった場合も、このディレクトリからは参照だけに留めてください。
- 実験中に生成されるファイルサイズが大きくなる可能性があるため、不要になった成果物は適宜削除してください。
- `uv run` で生成されるキャッシュが気になる場合は、`UV_NO_CACHE=1` を付与して実行できます。
