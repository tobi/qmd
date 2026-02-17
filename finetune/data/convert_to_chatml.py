#!/usr/bin/env python3
"""
Convert QMD expansion v3 JSONL to ChatML format for LFM2.5 training.
"""
import json
import random
from pathlib import Path


def reorder_hyde_first(output_items):
    """Reorder output items to put hyde first, then lex, then vec."""
    hyde_items = [item for item in output_items if item[0] == "hyde"]
    lex_items = [item for item in output_items if item[0] == "lex"]
    vec_items = [item for item in output_items if item[0] == "vec"]
    return hyde_items + lex_items + vec_items


def convert_entry(entry):
    """Convert a single QMD entry to ChatML format."""
    query = entry["query"]
    output_items = entry["output"]
    
    # Reorder: hyde first, then lex, then vec
    output_items = reorder_hyde_first(output_items)
    
    # Build the assistant response
    assistant_lines = []
    
    for item_type, content in output_items:
        assistant_lines.append(f"{item_type}: {content}")
    
    assistant_response = "\n".join(assistant_lines)
    
    # Create ChatML formatted text
    chatml_text = (
        "<|startoftext|>"
        "<|im_start|>user\n"
        f"Expand this search query: {query}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_response}"
        "<|im_end|>\n"
    )
    
    return {"text": chatml_text}


def main():
    # Use paths relative to this script's location
    script_dir = Path(__file__).parent
    input_file = script_dir / "qmd_expansion_v3.jsonl"
    output_dir = script_dir / "train-lfm2"
    
    # Load all data
    print(f"Loading data from {input_file}...")
    all_entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                converted = convert_entry(entry)
                all_entries.append(converted)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
    
    print(f"Successfully converted {len(all_entries)} entries")
    
    # Shuffle for better training
    random.seed(42)  # For reproducibility
    random.shuffle(all_entries)
    
    # Split into train (90%) and validation (10%)
    split_idx = int(len(all_entries) * 0.9)
    train_entries = all_entries[:split_idx]
    val_entries = all_entries[split_idx:]
    
    print(f"Train set: {len(train_entries)} entries")
    print(f"Validation set: {len(val_entries)} entries")
    
    # Write train set
    train_file = output_dir / "train.jsonl"
    print(f"Writing train set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write validation set
    val_file = output_dir / "val.jsonl"
    print(f"Writing validation set to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("Conversion complete!")
    
    # Show some sample entries
    print("\nSample train entries:")
    for i, entry in enumerate(train_entries[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(entry["text"][:300] + "..." if len(entry["text"]) > 300 else entry["text"])

if __name__ == "__main__":
    main()