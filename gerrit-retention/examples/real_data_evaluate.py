#!/usr/bin/env python3
"""
Deprecated shim: the evaluator script moved to scripts/evaluation/real_data_evaluate.py
This shim forwards execution to the new path to avoid breaking existing commands.
"""
import runpy
import sys
from pathlib import Path

new_path = Path(__file__).parent.parent / "scripts" / "evaluation" / "real_data_evaluate.py"
print("[DEPRECATED] examples/real_data_evaluate.py is moved to scripts/evaluation/real_data_evaluate.py", file=sys.stderr)
if not new_path.exists():
    print(f"[ERROR] New evaluator not found at {new_path}", file=sys.stderr)
    sys.exit(1)
# Re-exec the target module in the current process
runpy.run_path(str(new_path), run_name="__main__")
