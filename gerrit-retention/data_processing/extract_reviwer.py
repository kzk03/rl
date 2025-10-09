#!/usr/bin/env python3
"""指定したレビュアー（開発者）の行を CSV 群から抽出するユーティリティ。

`outputs/task_assign_multilabel/2023_cutoff_full/eval_detail` のように
ウィンドウ単位で書き出された評価結果 CSV から、特定レビュアーに関する行だけを
抽出して別ファイルにまとめて保存する。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def _sanitize_filename(name: str) -> str:
	cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
	cleaned = cleaned.strip("._-")
	return cleaned[:200] or "unknown"


def _match(target: str, candidate: str, mode: str) -> bool:
	if mode == "exact":
		return candidate == target
	if mode == "contains":
		return target in candidate
	if mode == "glob":
		return fnmatch.fnmatch(candidate, target)
	if mode == "regex":
		return bool(re.search(target, candidate))
	raise ValueError(f"未知のマッチモードです: {mode}")


def _iter_csv_rows(csv_path: Path) -> Iterator[Dict[str, str]]:
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			yield row


def extract_rows(
	input_dir: Path,
	reviewers: Iterable[str],
	output_dir: Path,
	column: str,
	match_mode: str,
) -> List[Path]:
	csv_paths = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
	if not csv_paths:
		raise FileNotFoundError(f"CSV が見つかりませんでした: {input_dir}")

	reviewers = list(reviewers)
	if not reviewers:
		raise ValueError("抽出対象のレビュアーを 1 件以上指定してください。")

	output_dir.mkdir(parents=True, exist_ok=True)

	writers: Dict[str, csv.DictWriter] = {}
	file_handles: Dict[str, any] = {}
	written_paths: Dict[str, Path] = {}

	try:
		header: Optional[List[str]] = None
		for csv_path in csv_paths:
			for row in _iter_csv_rows(csv_path):
				if header is None:
					header = list(row.keys())
				candidate = row.get(column, "")
				for reviewer in reviewers:
					if _match(reviewer, candidate, match_mode):
						if reviewer not in writers:
							out_path = output_dir / f"{_sanitize_filename(reviewer)}.csv"
							handle = out_path.open("w", encoding="utf-8", newline="")
							file_handles[reviewer] = handle
							writer = csv.DictWriter(handle, fieldnames=header)
							writer.writeheader()
							writers[reviewer] = writer
							written_paths[reviewer] = out_path
						writers[reviewer].writerow(row)
		if not written_paths:
			raise ValueError("指定した条件に一致する行が見つかりませんでした。")
		return list(written_paths.values())
	finally:
		for handle in file_handles.values():
			handle.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="評価出力 CSV からレビュアー単位で行を抽出")
	parser.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="評価 CSV が格納されているディレクトリ",
	)
	parser.add_argument(
		"--reviewer",
		dest="reviewers",
		action="append",
		required=True,
		help="抽出対象の reviewer_id（複数指定可: --reviewer foo --reviewer bar）",
	)
	parser.add_argument(
		"--column",
		default="reviewer_id",
		help="マッチ対象とするカラム名（デフォルト: reviewer_id）",
	)
	parser.add_argument(
		"--match-mode",
		choices=["exact", "contains", "glob", "regex"],
		default="exact",
		help="レビュアー ID のマッチ方法",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs") / "extracted_reviewers",
		help="抽出結果を保存するディレクトリ",
	)
	args = parser.parse_args()

	written = extract_rows(
		input_dir=args.input_dir,
		reviewers=args.reviewers,
		output_dir=args.output_dir,
		column=args.column,
		match_mode=args.match_mode,
	)
	print("書き出し完了:")
	for path in written:
		print(f"  {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
