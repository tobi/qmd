#!/usr/bin/env python3
"""Validate JSONL files against the QMD training schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import VALID_OUTPUT_TYPES


def validate_file(path: Path) -> tuple[int, int]:
    """Return (total_lines, error_count)."""
    total = 0
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{line_num}: invalid JSON ({e})")
                errors += 1
                continue

            query = obj.get("query")
            output = obj.get("output")

            if not isinstance(query, str) or not query.strip():
                print(f"{path}:{line_num}: missing/invalid query")
                errors += 1
                continue

            if not isinstance(output, list):
                print(f"{path}:{line_num}: output must be a list")
                errors += 1
                continue

            for idx, item in enumerate(output):
                if not isinstance(item, list) or len(item) != 2:
                    print(f"{path}:{line_num}: output[{idx}] must be [type, text]")
                    errors += 1
                    continue
                kind, text = item
                if kind not in VALID_OUTPUT_TYPES:
                    print(f"{path}:{line_num}: invalid output type '{kind}'")
                    errors += 1
                if not isinstance(text, str) or not text.strip():
                    print(f"{path}:{line_num}: empty output text")
                    errors += 1

    return total, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QMD JSONL schema")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["finetune/data/*.jsonl"],
        help="JSONL files or glob patterns (default: finetune/data/*.jsonl)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    files: list[Path] = []
    for pattern in args.paths:
        if "*" in pattern:
            files.extend(repo_root.glob(pattern))
        else:
            files.append(repo_root / pattern)

    files = [p for p in files if p.exists()]
    if not files:
        print("No files found to validate.")
        return 1

    total_lines = 0
    total_errors = 0
    for path in sorted(files):
        lines, errors = validate_file(path)
        total_lines += lines
        total_errors += errors
        status = "OK" if errors == 0 else f"{errors} error(s)"
        print(f"{path}: {lines} lines, {status}")

    if total_errors:
        print(
            f"\nValidation failed: {total_errors} error(s) across {total_lines} lines"
        )
        return 1

    print(f"\nValidation passed: {total_lines} lines checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
