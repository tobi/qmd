#!/usr/bin/env python3
"""
Prepare generated data for MLX training.

Takes the JSONL output from generate_from_notes.py and creates
train.jsonl and valid.jsonl files in the format expected by mlx-lm.

Usage:
    python dataset/prepare_data.py --input data/custom/expansions.jsonl --output data/custom
"""

import argparse
import json
import random
from pathlib import Path


def load_examples(input_path: Path) -> list[dict]:
    """Load and validate examples from JSONL."""
    examples = []
    invalid = 0

    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                # Validate required fields
                if "text" in example and len(example["text"]) > 50:
                    examples.append(example)
                else:
                    invalid += 1
            except json.JSONDecodeError:
                invalid += 1

    print(f"Loaded {len(examples)} valid examples ({invalid} invalid)")
    return examples


def split_data(examples: list[dict], eval_ratio: float = 0.1) -> tuple[list, list]:
    """Split into train and validation sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - eval_ratio))
    return examples[:split_idx], examples[split_idx:]


def save_for_mlx(examples: list[dict], output_path: Path, name: str):
    """Save in MLX-lm format (JSONL with 'text' field)."""
    filepath = output_path / f"{name}.jsonl"
    with open(filepath, "w") as f:
        for ex in examples:
            # MLX expects just {"text": "..."} format
            f.write(json.dumps({"text": ex["text"]}) + "\n")
    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for MLX training")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL from generate_from_notes.py")
    parser.add_argument("--output", "-o", type=str, default="data/custom",
                        help="Output directory for train/valid files")
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and split
    examples = load_examples(input_path)
    train_examples, valid_examples = split_data(examples, args.eval_ratio)

    print(f"\nSplit: {len(train_examples)} train, {len(valid_examples)} valid")

    # Save for MLX
    save_for_mlx(train_examples, output_path, "train")
    save_for_mlx(valid_examples, output_path, "valid")

    # Save summary
    summary = {
        "total": len(examples),
        "train": len(train_examples),
        "valid": len(valid_examples),
        "eval_ratio": args.eval_ratio,
    }
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Data prepared in {output_path}")
    print(f"   Ready for: python train.py sft --data {output_path}")


if __name__ == "__main__":
    main()
