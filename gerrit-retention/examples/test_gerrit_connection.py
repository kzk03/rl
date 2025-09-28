#!/usr/bin/env python3
"""
Quickly test Gerrit connectivity using the configured URL/auth.
Prints server version and a small sample of projects.

Usage:
  uv run python examples/test_gerrit_connection.py

Environment overrides (optional):
  GERRIT_URL, GERRIT_USERNAME, GERRIT_PASSWORD
"""
from __future__ import annotations

import sys
from pathlib import Path

# add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.data_integration.gerrit_client import create_gerrit_client
from gerrit_retention.utils.logger import setup_logging


def main() -> int:
    setup_logging(level="INFO")
    client = create_gerrit_client()
    try:
        ok = client.test_connection()
        if not ok:
            print("Connection test failed. Check URL/auth in configs or env.")
            return 1
        # Try listing a few projects to verify auth/permissions
        try:
            projects = client.get_projects()
            names = [p.get("name") for p in projects[:10]]
            print(f"Projects sample (up to 10): {names}")
        except Exception as e:
            print(f"Connected, but failed to list projects: {e}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
