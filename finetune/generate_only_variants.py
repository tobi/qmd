# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Generate 'only:' variant training data from high-quality expansions.

Takes existing training data and creates derivative examples where the query
ends with 'only: lex', 'only: hyde', or 'only: vec', and the output contains
ONLY that component type.

Usage:
    uv run generate_only_variants.py data/qmd_expansion_handcrafted.jsonl
    uv run generate_only_variants.py data/qmd_expansion_handcrafted.jsonl -o data/qmd_only_variants.jsonl
    uv run generate_only_variants.py data/*.jsonl --combine  # combine all inputs
"""

import argparse
import json
import sys
from pathlib import Path


def parse_expansion(text: str) -> dict:
    """Parse a multi-line expansion into {lex, vec, hyde} lists."""
    result = {"lex": [], "vec": [], "hyde": []}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())
    return result


# Templated patterns to filter out from hyde (low quality)
TEMPLATED_PATTERNS = [
    "This comprehensive guide covers",
    "This comprehensive guide to",
    "requires practice and patience",
    "This resource provides",
    "Follow the steps carefully",
    "covers all the essential information",
    "includes practical examples, best practices",
]


def is_templated_hyde(hyde_text: str) -> bool:
    """Check if a hyde output is a low-quality templated response."""
    return any(pattern in hyde_text for pattern in TEMPLATED_PATTERNS)


def format_output(parsed: dict, only_type: str) -> str | None:
    """Format output for a single type. Returns None if type is empty or low quality."""
    items = parsed.get(only_type, [])
    if not items:
        return None
    
    # Filter out templated hyde outputs
    if only_type == "hyde":
        filtered = [item for item in items if not is_templated_hyde(item)]
        if not filtered:
            return None
        items = filtered
    
    lines = []
    for item in items:
        lines.append(f"{only_type}: {item}")
    return "\n".join(lines)


def generate_only_variants(input_query: str, output: str) -> list[dict]:
    """Generate all valid 'only:' variants from a single example."""
    variants = []
    parsed = parse_expansion(output)
    
    for only_type in ["lex", "vec", "hyde"]:
        formatted = format_output(parsed, only_type)
        if formatted:
            # Add the '/only:' suffix to the query (slash prefix)
            new_query = f"{input_query} /only:{only_type}"
            variants.append({
                "input": new_query,
                "output": formatted,
                "_source_type": only_type,
                "_source_query": input_query,
            })
    
    return variants


def process_file(input_path: Path) -> list[dict]:
    """Process a single JSONL file and return all 'only:' variants."""
    variants = []
    seen_queries = set()
    
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num} (invalid JSON): {e}", file=sys.stderr)
                continue
            
            # Skip metadata lines
            if data.get("_meta"):
                continue
            
            input_query = data.get("input", "")
            output = data.get("output", "")
            
            if not input_query or not output:
                continue
            
            # Skip if query already has '/only:' suffix
            if " /only:" in input_query.lower():
                continue
            
            # Skip duplicates
            if input_query in seen_queries:
                continue
            seen_queries.add(input_query)
            
            # Generate variants
            for variant in generate_only_variants(input_query, output):
                variants.append(variant)
    
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Generate 'only:' variant training data from high-quality expansions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSONL files with training data",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/qmd_only_variants.jsonl",
        help="Output JSONL file (default: data/qmd_only_variants.jsonl)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all input files into one output",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about generated variants",
    )
    
    args = parser.parse_args()
    
    all_variants = []
    stats = {"lex": 0, "vec": 0, "hyde": 0}
    
    for input_file in args.input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping", file=sys.stderr)
            continue
        
        print(f"Processing {input_path.name}...", file=sys.stderr)
        variants = process_file(input_path)
        
        for v in variants:
            stats[v["_source_type"]] += 1
        
        if args.combine:
            all_variants.extend(variants)
        else:
            # Write to separate output files per input
            output_path = input_path.parent / f"{input_path.stem}_only.jsonl"
            with open(output_path, "w") as f:
                for variant in variants:
                    # Remove internal fields before writing
                    clean = {"input": variant["input"], "output": variant["output"]}
                    f.write(json.dumps(clean) + "\n")
            print(f"  -> {len(variants)} variants written to {output_path}", file=sys.stderr)
    
    if args.combine and all_variants:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for variant in all_variants:
                clean = {"input": variant["input"], "output": variant["output"]}
                f.write(json.dumps(clean) + "\n")
        
        print(f"\nTotal: {len(all_variants)} variants written to {output_path}", file=sys.stderr)
    
    if args.stats or args.combine:
        print(f"\nStats:", file=sys.stderr)
        print(f"  lex:  {stats['lex']}", file=sys.stderr)
        print(f"  vec:  {stats['vec']}", file=sys.stderr)
        print(f"  hyde: {stats['hyde']}", file=sys.stderr)
        print(f"  total: {sum(stats.values())}", file=sys.stderr)


if __name__ == "__main__":
    main()
