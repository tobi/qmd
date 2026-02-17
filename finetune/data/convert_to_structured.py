#!/usr/bin/env python3
"""
Convert QMD expansion JSONL to structured format with type/query objects.
Also applies hyde-first ordering.
"""
import json
from pathlib import Path


def reorder_hyde_first(output_items):
    """Reorder output items to put hyde first, then lex, then vec."""
    hyde_items = [item for item in output_items if item[0] == "hyde"]
    lex_items = [item for item in output_items if item[0] == "lex"]
    vec_items = [item for item in output_items if item[0] == "vec"]
    return hyde_items + lex_items + vec_items


def convert_to_structured(entry):
    """Convert flat output format to structured searches array."""
    query = entry["query"]
    output_items = entry.get("output", [])
    
    # Apply hyde-first ordering
    output_items = reorder_hyde_first(output_items)
    
    # Convert to structured format
    searches = []
    for item_type, content in output_items:
        searches.append({
            "type": item_type,
            "query": content
        })
    
    return {
        "query": query,
        "searches": searches
    }


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "qmd_expansion_v3.jsonl"
    output_file = script_dir / "qmd_expansion_v3_structured.jsonl"
    
    print(f"Converting {input_file} to structured format...")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.strip():
                entry = json.loads(line)
                structured = convert_to_structured(entry)
                f_out.write(json.dumps(structured, ensure_ascii=False) + '\n')
                count += 1
    
    print(f"Converted {count} entries to {output_file}")
    
    # Show sample
    print("\nSample entry:")
    with open(output_file, 'r') as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
