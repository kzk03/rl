from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract_matches(base_dir: Path) -> None:
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return

    for input_path in csv_files:
        if input_path.name.startswith("extracted_"):
            continue
        output_file_name = f"extracted_{input_path.name}"
        output_path = base_dir / output_file_name
        print(f"Processing {input_path.name} -> {output_path.name}")

        df = pd.read_csv(input_path)
        extracted_df = df[(df["is_positive"] == 1) & (df["selected"] == 1)]
        extracted_df.to_csv(output_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract matched positive reviewer selections from replay detail CSVs.")
    ap.add_argument("--dir", dest="dirs", action="append", default=None, help="Directory containing replay_detail CSVs (can be specified multiple times).")
    args = ap.parse_args()

    base_dirs: list[Path]
    if not args.dirs:
        base_dirs = [Path("/Users/kazuki-h/rl/gerrit-retention/outputs/task_assign_multilabel/replay_detail")]
    else:
        base_dirs = [Path(d).expanduser().resolve() for d in args.dirs]

    for base_dir in base_dirs:
        extract_matches(base_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())