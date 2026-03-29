#!/usr/bin/env python3
"""Prepare QMD query expansion data for LFM2.5-1.2B-Instruct training.

LFM2.5 uses ChatML format:
  <|startoftext|><|im_start|>user
  Expand this search query: {query}<|im_end|>
  <|im_start|>assistant
  {output}<|im_end|>

No /no_think needed (that's Qwen3-specific).
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import normalize_output_items, output_items_to_text

from transformers import AutoTokenizer


def format_for_training(query_text: str, output_items: list[list[str]], tokenizer) -> dict:
    """Format a single example for SFT training using LFM2.5 chat format."""
    output_text = output_items_to_text(output_items)

    messages = [
        {"role": "user", "content": f"Expand this search query: {query_text}"},
        {"role": "assistant", "content": output_text},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return {"text": text}


def main():
    input_path = Path("data/qmd_expansion_v2.jsonl")
    output_dir = Path("data/train-lfm2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LFM2.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", trust_remote_code=True
    )

    examples = []
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            items = normalize_output_items(row["output"])
            example = format_for_training(row["query"], items, tokenizer)
            examples.append(example)

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:]

    # Write as JSONL
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(train)} train, {len(val)} val examples to {output_dir}")
    print(f"\nSample formatted text:")
    print(train[0]["text"][:500])


if __name__ == "__main__":
    main()
