# Reviewer Adoption IRL Pipeline

このディレクトリには、レビュアーが次に推薦されたタスクへ取り組む確率を推定するためのデータ整形・学習・評価スクリプトを配置しています。

## 構成

- `build_reviewer_adoption_dataset.py` — `outputs/irl/reviewer_sequences.json` からレビュアー視点の遷移データセットを生成します。
- `train_reviewer_adoption_irl.py` — MaxEnt 形式の二値 IRL モデルを学習し、パラメータを保存します。
- `evaluate_reviewer_adoption.py` — 学習済みモデルを用いて検証セット上の精度指標やキャリブレーションを計算します。
- `common.py` — 複数スクリプトで共有する入出力ユーティリティ。

## 典型フロー

```bash
uv run python scripts/reviewer_adoption/build_reviewer_adoption_dataset.py --cutoff 2023-07-01T00:00:00Z --outdir outputs/reviewer_adoption
uv run python scripts/reviewer_adoption/train_reviewer_adoption_irl.py --train outputs/reviewer_adoption/train.jsonl --outdir outputs/reviewer_adoption/model
uv run python scripts/reviewer_adoption/evaluate_reviewer_adoption.py --data outputs/reviewer_adoption/eval.jsonl --model outputs/reviewer_adoption/model/model.json
```

各ステップは CLI オプションで入力パスや閾値を切り替えられるように設計しています。詳しい使い方は各スクリプトの `--help` を参照してください。
