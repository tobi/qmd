#!/usr/bin/env python3
"""Score JSONL datasets with the reward function."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import (
    normalize_output_items,
    output_items_to_text,
    parse_output_text,
)
from reward import score_expansion_detailed


def score_file(path: Path) -> tuple[int, int, list[float], dict]:
    total = 0
    errors = 0
    scores: list[float] = []
    ratings: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            query = obj.get("query") or obj.get("input")
            output = obj.get("output")
            if not isinstance(query, str) or not query.strip():
                errors += 1
                continue
            if output is None:
                errors += 1
                continue

            if isinstance(output, str):
                output_items = normalize_output_items(parse_output_text(output))
            else:
                output_items = normalize_output_items(output)

            output_text = output_items_to_text(output_items)
            if not output_text:
                errors += 1
                continue

            detail = score_expansion_detailed(query, output_text)
            score = detail["percentage"]
            scores.append(score)
            rating = detail["rating"]
            ratings[rating] = ratings.get(rating, 0) + 1

    return total, errors, scores, ratings


def main() -> int:
    parser = argparse.ArgumentParser(description="Score QMD datasets")
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
        print("No files found to score.")
        return 1

    for path in sorted(files):
        total, errors, scores, ratings = score_file(path)
        if scores:
            avg = statistics.mean(scores)
            median = statistics.median(scores)
            min_score = min(scores)
            max_score = max(scores)
            above_70 = sum(1 for s in scores if s >= 70.0)
            pct_70 = above_70 / len(scores) * 100
            print(
                f"{path}: {len(scores)} scored, {errors} errors, "
                f"avg {avg:.1f}, median {median:.1f}, min {min_score:.1f}, "
                f"max {max_score:.1f}, >=70 {pct_70:.1f}%"
            )
        else:
            print(f"{path}: 0 scored, {errors} errors")

        if ratings:
            rating_parts = [f"{k}:{v}" for k, v in sorted(ratings.items())]
            print(f"  ratings: {', '.join(rating_parts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
