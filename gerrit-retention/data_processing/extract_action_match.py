import pandas as pd
from pathlib import Path

base_dir = Path("/Users/kazuki-h/rl/gerrit-retention/outputs/task_assign_cut2023/replay_csv_train")
csv_files = base_dir.glob("*.csv")

for input_path in csv_files:
    output_file_name = f"extracted_{input_path.name}"
    output_path = base_dir / output_file_name
    # 元のファイル名と出力ファイル名を表示（実行状況の確認に便利です）
    print(f"Processing {input_path.name} -> {output_path.name}")
    
    # CSVファイルを読み込み
    df = pd.read_csv(input_path)

        # 条件に合う行を抽出
    extracted_df = df[(df["is_positive"] == 1) & (df["selected"] == 1)]

    # 結果を保存
    extracted_df.to_csv(output_path, index=False)