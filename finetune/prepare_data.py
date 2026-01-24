#!/usr/bin/env python3
"""Prepare QMD query expansion data for training."""

import argparse
import json
from pathlib import Path

# Prompt template matching QMD's llm.ts format (simplified for training)
PROMPT_TEMPLATE = """You are a search query optimization expert. Transform the query into retrieval-optimized outputs.

Query: {query}

Output format:
lex: {{keyword variation}}
vec: {{semantic reformulation}}
hyde: {{hypothetical document passage}}

Output:"""


def format_for_training(input_text: str, output_text: str) -> dict:
    """Format a single example for SFT training."""
    prompt = PROMPT_TEMPLATE.format(query=input_text)
    return {
        "prompt": prompt,
        "completion": output_text,
        # Alternative format for some trainers
        "text": f"{prompt}\n{output_text}",
        # Chat format
        "messages": [
            {"role": "user", "content": f"Expand this search query:\n\n{input_text}"},
            {"role": "assistant", "content": output_text}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input", type=str, default="data/qmd_expansion.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/train", help="Output directory")
    parser.add_argument("--split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    # Load examples
    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Format for training
    formatted = [format_for_training(ex["input"], ex["output"]) for ex in examples]

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

    # Write chat format (for TRL/Unsloth)
    chat_path = output_dir / "train_chat.jsonl"
    with open(chat_path, "w") as f:
        for item in train_data:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")

    print(f"Written {len(train_data)} train examples to {train_path}")
    print(f"Written {len(val_data)} validation examples to {val_path}")
    print(f"Written chat format to {chat_path}")

    # Also save as HuggingFace datasets format info
    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "columns": ["prompt", "completion", "text", "messages"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
