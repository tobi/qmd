#!/usr/bin/env python3
"""
Fix lex entries in QMD training data by removing filler words that were
inserted as padding rather than being genuine search intent.

Filler words to remove (case insensitive):
- overview
- tutorial  
- guide
- examples
- documentation
- best practices

Keep these words when they're genuinely part of the query intent.
"""

import json
import re
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "qmd_expansion_v2.jsonl"
OUTPUT_FILE = Path(__file__).parent / "qmd_expansion_v3_lex_fixed.jsonl"

# Filler words/phrases to remove when they're just padding
FILLER_WORDS = [
    'overview',
    'tutorial',
    'guide',
    'examples',
    'documentation',
    'best practices',
]


def count_word(text: str, word: str) -> int:
    """Count occurrences of a word in text (case insensitive, whole word)."""
    if word == 'best practices':
        return len(re.findall(r'\bbest practices\b', text, re.IGNORECASE))
    return len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))


def clean_lex_entry(lex: str, query: str) -> str:
    """
    Remove filler words from a lex entry.
    
    Logic:
    - If filler word appears in lex but NOT in query: remove all occurrences
    - If filler word appears in both: remove excess occurrences (keep same count as in query)
    """
    query_lower = query.lower()
    result = lex
    
    for filler in FILLER_WORDS:
        query_count = count_word(query_lower, filler)
        lex_count = count_word(result, filler)
        
        # Remove excess occurrences
        if lex_count > query_count:
            # Remove (lex_count - query_count) occurrences
            for _ in range(lex_count - query_count):
                if filler == 'best practices':
                    result = re.sub(r'\bbest practices\b', '', result, count=1, flags=re.IGNORECASE)
                else:
                    result = re.sub(r'\b' + re.escape(filler) + r'\b', '', result, count=1, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    result = ' '.join(result.split())
    return result.strip()


def has_filler_to_clean(lex: str, query: str) -> bool:
    """Check if lex entry has filler words that need cleaning."""
    query_lower = query.lower()
    
    for filler in FILLER_WORDS:
        query_count = count_word(query_lower, filler)
        lex_count = count_word(lex, filler)
        if lex_count > query_count:
            return True
    return False


def process_entry(entry: dict) -> tuple[dict, bool]:
    """
    Process a single entry, cleaning lex entries if needed.
    Returns (processed_entry, was_modified)
    """
    query = entry.get("query", "")
    output = entry.get("output", [])
    modified = False
    new_output = []
    
    for item in output:
        if item[0] == "lex":
            original_lex = item[1]
            if has_filler_to_clean(original_lex, query):
                cleaned_lex = clean_lex_entry(original_lex, query)
                if cleaned_lex != original_lex:
                    new_output.append(["lex", cleaned_lex])
                    modified = True
                else:
                    new_output.append(item)
            else:
                new_output.append(item)
        else:
            new_output.append(item)
    
    new_entry = entry.copy()
    new_entry["output"] = new_output
    return new_entry, modified


def main():
    entries = []
    modified_count = 0
    total_lex_modified = 0
    
    print(f"Reading {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            processed, modified = process_entry(entry)
            entries.append(processed)
            if modified:
                modified_count += 1
                # Count how many lex entries were modified
                orig_output = entry.get("output", [])
                new_output = processed.get("output", [])
                for i, item in enumerate(orig_output):
                    if item[0] == "lex" and item[1] != new_output[i][1]:
                        total_lex_modified += 1
    
    print(f"Total entries: {len(entries)}")
    print(f"Entries modified: {modified_count}")
    print(f"Total lex entries cleaned: {total_lex_modified}")
    
    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print("Done!")
    
    # Show some examples of modifications
    print("\n--- Sample modifications ---")
    sample_count = 0
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            processed, modified = process_entry(entry)
            if modified and sample_count < 15:
                query = entry.get("query", "")
                print(f"\nQuery: {query}")
                orig_lex = [item[1] for item in entry.get("output", []) if item[0] == "lex"]
                new_lex = [item[1] for item in processed.get("output", []) if item[0] == "lex"]
                for orig, new in zip(orig_lex, new_lex):
                    if orig != new:
                        print(f"  - \"{orig}\"")
                        print(f"  + \"{new}\"")
                sample_count += 1


if __name__ == "__main__":
    main()
