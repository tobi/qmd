#!/usr/bin/env python3
"""
Verify the quality and correctness of the converted ChatML data.
"""
import json
import re
from pathlib import Path


def verify_chatml_format(text):
    """Verify that the text follows proper ChatML format."""
    issues = []
    
    # Check start token
    if not text.startswith("<|startoftext|>"):
        issues.append("Missing <|startoftext|> at beginning")
    
    # Check user section
    user_pattern = r"<\|im_start\|>user\n.*?<\|im_end\|>"
    if not re.search(user_pattern, text, re.DOTALL):
        issues.append("Missing or malformed user section")
    
    # Check assistant section
    assistant_pattern = r"<\|im_start\|>assistant\n.*?<\|im_end\|>"
    if not re.search(assistant_pattern, text, re.DOTALL):
        issues.append("Missing or malformed assistant section")
    
    # Check for proper query format
    if "Expand this search query:" not in text:
        issues.append("Missing 'Expand this search query:' prompt")
    
    # Check for required output types
    assistant_content = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if assistant_content:
        content = assistant_content.group(1)
        has_lex = "lex:" in content
        has_vec = "vec:" in content
        has_hyde = "hyde:" in content
        
        if not has_lex:
            issues.append("Missing lex: entries")
        if not has_vec:
            issues.append("Missing vec: entries")
        if not has_hyde:
            issues.append("Missing hyde: entries")
        
        # Validate hyde-first ordering
        lines = content.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if not first_line.startswith("hyde:"):
                issues.append("Hyde not first (expected hyde-first ordering)")
    
    return issues

def analyze_file(filepath):
    """Analyze a JSONL file for quality and issues."""
    print(f"\nAnalyzing {filepath}...")
    
    total_entries = 0
    total_issues = 0
    issue_counts = {}
    query_lengths = []
    assistant_lengths = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                total_entries += 1
                
                text = entry["text"]
                issues = verify_chatml_format(text)
                
                if issues:
                    total_issues += 1
                    for issue in issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
                # Extract query and assistant response for length analysis
                user_match = re.search(r"Expand this search query: (.*?)<\|im_end\|>", text, re.DOTALL)
                assistant_match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
                
                if user_match:
                    query_lengths.append(len(user_match.group(1).strip()))
                if assistant_match:
                    assistant_lengths.append(len(assistant_match.group(1)))
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print(f"Total entries: {total_entries}")
    print(f"Entries with issues: {total_issues}")
    print(f"Success rate: {((total_entries - total_issues) / total_entries * 100):.1f}%")
    
    if issue_counts:
        print("\nIssue breakdown:")
        for issue, count in sorted(issue_counts.items()):
            print(f"  {issue}: {count}")
    
    if query_lengths:
        print(f"\nQuery length stats:")
        print(f"  Min: {min(query_lengths)} chars")
        print(f"  Max: {max(query_lengths)} chars")
        print(f"  Avg: {sum(query_lengths) / len(query_lengths):.1f} chars")
    
    if assistant_lengths:
        print(f"\nAssistant response length stats:")
        print(f"  Min: {min(assistant_lengths)} chars")
        print(f"  Max: {max(assistant_lengths)} chars") 
        print(f"  Avg: {sum(assistant_lengths) / len(assistant_lengths):.1f} chars")

def main():
    # Use paths relative to this script's location
    script_dir = Path(__file__).parent
    data_dir = script_dir / "train-lfm2"
    
    # Analyze both train and validation sets
    analyze_file(data_dir / "train.jsonl")
    analyze_file(data_dir / "val.jsonl")
    
    print("\n" + "="*50)
    print("DATA PREPARATION VERIFICATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()