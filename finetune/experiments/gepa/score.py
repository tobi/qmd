#!/usr/bin/env python3
"""Score GEPA JSONL outputs using reward.py."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from example import example_from_json

from reward import score_expansion_detailed
from dataset.schema import output_items_to_text


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
                example = example_from_json(obj)
            except Exception:
                errors += 1
                continue

            output_text = output_items_to_text([item.to_pair() for item in example.output])
            if not output_text:
                errors += 1
                continue

            detail = score_expansion_detailed(example.query, output_text)
            score = detail["percentage"]
            scores.append(score)
            rating = detail["rating"]
            ratings[rating] = ratings.get(rating, 0) + 1

    return total, errors, scores, ratings


def main() -> int:
    parser = argparse.ArgumentParser(description="Score GEPA JSONL outputs")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input not found: {path}")
        return 1

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
        rating_parts = [f\"{k}:{v}\" for k, v in sorted(ratings.items())]
        print(f\"  ratings: {', '.join(rating_parts)}\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
