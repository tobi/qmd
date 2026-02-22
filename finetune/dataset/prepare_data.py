#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "pydantic>=2.0",
#     "jinja2",
# ]
# ///
"""Prepare QMD query expansion data for training.

Loads all data/*.jsonl via the strict Pydantic schema, applies the Qwen3
chat template, deduplicates by query, and writes train/val splits.

The prepared train files are ephemeral build artifacts — the canonical
data lives in data/*.jsonl and is always loaded through the schema.
"""

import argparse
import json
import random
import os
from pathlib import Path

from dataset.schema import (
    TrainingExample,
    load_examples,
    output_items_to_text,
)

from transformers import AutoTokenizer

_tokenizer = None
_tokenizer_model = None


def get_tokenizer():
    global _tokenizer, _tokenizer_model
    model_name = os.environ.get("QMD_BASE_MODEL", "Qwen/Qwen3-1.7B")
    if _tokenizer is None or _tokenizer_model != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model = model_name
    return _tokenizer


def format_for_training(ex: TrainingExample) -> dict:
    """Format a validated TrainingExample for SFT training."""
    tokenizer = get_tokenizer()
    output_text = output_items_to_text(ex.output)

    messages = [
        {
            "role": "user",
            "content": f"/no_think Expand this search query: {ex.query}",
        },
        {"role": "assistant", "content": output_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Strip empty <think> tags — /no_think should suppress them
    text = text.replace("<think>\n\n</think>\n\n", "")

    return {
        "text": text,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--input",
        type=str,
        default="data/*.jsonl",
        help="Input JSONL file(s) - supports glob patterns",
    )
    parser.add_argument(
        "--output", type=str, default="data/train", help="Output directory"
    )
    parser.add_argument(
        "--split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input files
    import glob as globmod

    if "*" in args.input:
        input_files = sorted(globmod.glob(args.input))
        if not input_files:
            print(f"Error: No files found matching: {args.input}")
            exit(1)
        print(f"Found {len(input_files)} input files")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            exit(1)
        input_files = [str(input_path)]

    # Load all examples through strict Pydantic schema
    all_examples: list[TrainingExample] = []
    for input_file in input_files:
        examples = load_examples(input_file)
        print(f"  {Path(input_file).name}: {len(examples)} examples")
        all_examples.extend(examples)

    print(f"Loaded {len(all_examples)} examples total")

    # Deduplicate by query (case-insensitive)
    seen: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in all_examples:
        key = ex.query.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    if len(deduped) < len(all_examples):
        print(f"Deduplicated: {len(all_examples)} -> {len(deduped)}")
    all_examples = deduped

    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Format each example using the Pydantic model
    formatted = [format_for_training(ex) for ex in all_examples]

    # Split
    split_idx = int(len(formatted) * (1 - args.split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # Write (these are ephemeral build artifacts)
    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with open(output_dir / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    with open(output_dir / "train_chat.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")

    # Stats
    short_final = sum(1 for ex in all_examples if len(ex.query.split()) <= 2)
    print(f"\n=== Summary ===")
    print(f"Total examples: {len(all_examples)}")
    print(f"Short queries: {short_final} ({100 * short_final / len(all_examples):.1f}%)")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")

    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "short_query_pct": round(100 * short_final / len(all_examples), 1),
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
