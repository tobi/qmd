# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare v4 dataset: high-quality expansions + /only: variants."""

import json
import random
from pathlib import Path

def to_chat_format(query: str, output: str) -> dict:
    """Convert input/output to chat format with /no_think."""
    # For /only: queries, keep the suffix in the prompt
    prompt = f"/no_think Expand this search query: {query}"
    
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{output}<|im_end|>\n"
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output}
    ]
    
    return {"text": text, "messages": messages}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    data_dir = Path("data")
    
    # High-quality sources
    sources = [
        ("qmd_expansion_v2.jsonl", "v2"),
        ("qmd_expansion_handcrafted.jsonl", "handcrafted"),
        ("qmd_only_variants.jsonl", "only"),
    ]
    
    all_examples = []
    stats = {}
    
    for filename, label in sources:
        path = data_dir / filename
        if not path.exists():
            print(f"  Skipping {filename} (not found)")
            continue
        
        raw = load_jsonl(path)
        converted = []
        
        for item in raw:
            query = item.get("input", "")
            output = item.get("output", "")
            if query and output:
                converted.append(to_chat_format(query, output))
        
        all_examples.extend(converted)
        stats[label] = len(converted)
        print(f"  {label}: {len(converted)} examples")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)
    
    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    val = all_examples[split_idx:]
    
    # Write output
    out_dir = data_dir / "train_v4"
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    
    with open(out_dir / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")
    
    # Dataset info
    info = {
        "dataset_name": "qmd-query-expansion-v4",
        "train_samples": len(train),
        "val_samples": len(val),
        "sources": stats,
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Dataset prepared in {out_dir}/")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Total: {len(all_examples)}")


if __name__ == "__main__":
    main()
