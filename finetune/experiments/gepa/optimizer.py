#!/usr/bin/env python3
"""Write model.json prompt config for generating high-quality examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from example import SearchType, SEARCH_TYPE_TO_PREFIX


def build_prompt() -> str:
    lex = SEARCH_TYPE_TO_PREFIX[SearchType.LexSearch]
    vec = SEARCH_TYPE_TO_PREFIX[SearchType.VecSearch]
    hyde = SEARCH_TYPE_TO_PREFIX[SearchType.HydeSearch]

    return (
        "You are a query expansion expert. Given a user query, output a single JSON object "
        "that matches the training JSONL schema:\n"
        '{"query": "...", "output": [["lex", "..."], ["vec", "..."], ["hyde", "..."]]}\n'
        "Rules:\n"
        f"- output is a list of pairs, where the first element is one of: "
        f"\"{lex}\", \"{vec}\", \"{hyde}\".\n"
        "- Include 2-3 lex lines, 2-3 vec lines, and 0-1 hyde line.\n"
        "- lex lines are short keyword phrases; never equal or near-echo the query.\n"
        "- vec lines are natural language search phrases.\n"
        "- hyde is a concise hypothetical passage (50-200 chars), single line.\n"
        "- Preserve key terms and named entities in lex lines.\n"
        "- No extra text outside the JSON object.\n"
    )


def write_model_json(path: Path) -> None:
    payload = {
        "name": "qmd-gepa-example-generator",
        "model": "grok-4-1-fast-reasoning",
        "schema_version": 1,
        "prompt": build_prompt(),
        "output_schema": {
            "query": "string",
            "output": [["lex|vec|hyde", "string"]],
        },
        "notes": [
            "LexSearch/VecSearch/HydeSearch are represented as lex/vec/hyde in output.",
            "Do not echo the query in lex lines.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write model.json for GEPA generation")
    parser.add_argument(
        "--output",
        type=str,
        default="gepa/model.json",
        help="Path to write model.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_model_json(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
