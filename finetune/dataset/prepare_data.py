#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "jinja2",
# ]
# ///
"""Prepare QMD query expansion data for training.

See PROMPT_FORMAT.md for format specification.
"""

import argparse
import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import (
    normalize_output_items,
    output_items_to_text,
    parse_output_text,
    has_type,
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



def format_for_training(query_text: str, output_items: list[list[str]]) -> dict:
    """Format a single example for SFT training using Qwen chat format."""
    tokenizer = get_tokenizer()
    output_text = output_items_to_text(output_items)

    # Use /no_think to disable thinking mode - we want direct output
    messages = [
        {
            "role": "user",
            "content": f"/no_think Expand this search query: {query_text}",
        },
        {"role": "assistant", "content": output_text},
    ]

    # Use tokenizer to generate proper chat format with special tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Strip empty <think> tags - we don't want thinking mode
    # The template adds "<think>\n\n</think>\n\n" which we remove
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
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support glob patterns for input
    import glob

    if "*" in args.input:
        input_files = sorted(glob.glob(args.input))
        if not input_files:
            print(f"Error: No files found matching: {args.input}")
            exit(1)
        print(
            f"Found {len(input_files)} input files: {[Path(f).name for f in input_files]}"
        )
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            exit(1)
        input_files = [str(input_path)]

    # Load all examples from all input files
    examples = []

    for input_file in input_files:
        file_count = 0
        with open(input_file) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)

                    # Normalize legacy format
                    if "query" not in ex and "input" in ex:
                        ex["query"] = ex.pop("input")
                    if isinstance(ex.get("output"), str):
                        ex["output"] = parse_output_text(ex["output"])
                    ex["output"] = normalize_output_items(ex.get("output", []))

                    examples.append(ex)
                    file_count += 1
        print(f"  {Path(input_file).name}: {file_count} examples")

    print(f"Loaded {len(examples)} examples total")

    # Combine and shuffle
    all_examples = examples
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Format for training
    formatted = [format_for_training(ex["query"], ex["output"]) for ex in all_examples]

    # Split into train/val
    split_idx = int(len(formatted) * (1 - args.split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # Write train set
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Write validation set
    val_path = output_dir / "val.jsonl"
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Write chat format (for TRL)
    chat_path = output_dir / "train_chat.jsonl"
    with open(chat_path, "w") as f:
        for item in train_data:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")

    # Stats
    short_final = sum(1 for ex in all_examples if len(ex["query"].split()) <= 2)

    print(f"\n=== Summary ===")
    print(f"Total examples: {len(all_examples)}")
    print(
        f"Short queries: {short_final} ({100 * short_final / len(all_examples):.1f}%)"
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")

    # Dataset info
    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "short_query_pct": round(100 * short_final / len(all_examples), 1),
        "columns": ["prompt", "completion", "text", "messages"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
