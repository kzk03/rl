# data_processing ディレクトリ概要

- 目的: データ抽出/前処理/特徴量生成の一連処理。
- 主な内容:
  - `gerrit_extraction/`: Gerrit からの抽出コード。
  - `preprocessing/`: クリーニングや時系列整形。
  - `feature_engineering/`: モデリング向け特徴量生成。

## ポリシー: data と data_processing の役割分担

- 本ディレクトリは「コードのみ」を置きます。永続データを置きません。
- 生成物の出力先は `data/` 配下（例: `data/raw/`, `data/processed/unified/`）。
- 入力/出力の既定パスは `configs/*.yaml`（例: `configs/gerrit_config.yaml` や `configs/retention_config.yaml`）から解決されます。
- スクリプトの CLI 引数で入出力パスを上書きできる場合は、そちらを優先します。
