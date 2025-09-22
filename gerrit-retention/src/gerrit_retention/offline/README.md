# offline ディレクトリ概要

- 目的: ログから学習用のオフラインデータセットを構築。
- 主なファイル:
  - `offline_dataset.py`: 共通のオフラインデータ読み出し。
  - `build_assignment_tasks.py`: 変更単位の割当タスク構築（候補 K、正解集合、特徴）。
