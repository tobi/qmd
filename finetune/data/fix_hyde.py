#!/usr/bin/env python3
"""
Fix template hyde entries in qmd_expansion_v2.jsonl
Replaces generic "comprehensive guide covers everything" hydes with query-specific ones.
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI

# Configuration
INPUT_FILE = Path("qmd_expansion_v2.jsonl")
OUTPUT_FILE = Path("qmd_expansion_v3.jsonl")
CHECKPOINT_FILE = Path("fix_hyde_checkpoint.json")
BAD_PATTERN = "comprehensive guide covers everything"
BATCH_SIZE = 25  # Process 25 queries per API call

def load_checkpoint():
    """Load progress checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_queries": {}, "completed_indices": []}

def save_checkpoint(checkpoint):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def load_examples():
    """Load all examples from input file."""
    examples = []
    with open(INPUT_FILE) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def is_bad_hyde(example):
    """Check if example has the bad template hyde."""
    for item in example.get("output", []):
        if item[0] == "hyde" and BAD_PATTERN in item[1]:
            return True
    return False

def get_hyde_from_example(example):
    """Extract the hyde value from an example."""
    for item in example.get("output", []):
        if item[0] == "hyde":
            return item[1]
    return None

def set_hyde_in_example(example, new_hyde):
    """Set the hyde value in an example."""
    for i, item in enumerate(example.get("output", [])):
        if item[0] == "hyde":
            example["output"][i] = ["hyde", new_hyde]
            return
    # If no hyde found, append it
    example["output"].append(["hyde", new_hyde])

def generate_hydes_batch(client, queries):
    """Generate hydes for a batch of queries using GPT-4o-mini."""
    queries_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    
    prompt = f"""Generate hypothetical document snippets (hyde) for each query below.

Requirements:
- 100-180 characters each
- Query-specific factual information
- Written as if from an actual document that would answer the query
- NO generic phrases like "comprehensive guide" or "everything you need to know"
- Include actual facts, numbers, names, or specifics

Example:
Query: "kubernetes pod networking"
Hyde: "Pods communicate via cluster IP. Use CNI plugins like Calico or Flannel. Service discovery through DNS. NetworkPolicy controls traffic between namespaces."

Queries to process:
{queries_text}

Output ONLY valid JSON - a single object mapping query numbers to hyde texts:
{{"1": "hyde text for query 1", "2": "hyde text for query 2", ...}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the response
    text = response.choices[0].message.content.strip()
    # Handle potential markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    
    try:
        result = json.loads(text)
        # Convert keys to int
        return {int(k): v for k, v in result.items()}
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response text: {text[:500]}...")
        return {}

def main():
    print("Loading examples...")
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")
    
    # Find bad examples
    bad_indices = []
    for i, ex in enumerate(examples):
        if is_bad_hyde(ex):
            bad_indices.append(i)
    
    print(f"Found {len(bad_indices)} examples with bad hyde")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_indices", []))
    processed_queries = checkpoint.get("processed_queries", {})
    
    # Filter to only unprocessed
    to_process = [i for i in bad_indices if i not in completed]
    print(f"Already processed: {len(completed)}, remaining: {len(to_process)}")
    
    if not to_process:
        print("All examples already processed!")
    else:
        # Initialize OpenAI client
        client = OpenAI()
        
        # Process in batches
        for batch_start in range(0, len(to_process), BATCH_SIZE):
            batch_indices = to_process[batch_start:batch_start + BATCH_SIZE]
            queries = [examples[i]["query"] for i in batch_indices]
            
            print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(to_process) + BATCH_SIZE - 1)//BATCH_SIZE}")
            print(f"Queries: {queries[:3]}...")
            
            try:
                hydes = generate_hydes_batch(client, queries)
                
                # Apply the generated hydes
                for j, idx in enumerate(batch_indices):
                    query_num = j + 1
                    if query_num in hydes:
                        new_hyde = hydes[query_num]
                        processed_queries[str(idx)] = new_hyde
                        completed.add(idx)
                        print(f"  [{idx}] {examples[idx]['query'][:40]}... -> {new_hyde[:50]}...")
                    else:
                        print(f"  [{idx}] MISSING hyde for: {examples[idx]['query']}")
                
                # Save checkpoint after each batch
                checkpoint = {
                    "processed_queries": processed_queries,
                    "completed_indices": list(completed)
                }
                save_checkpoint(checkpoint)
                print(f"  Checkpoint saved: {len(completed)}/{len(bad_indices)} complete")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                # Save checkpoint before exiting
                checkpoint = {
                    "processed_queries": processed_queries,
                    "completed_indices": list(completed)
                }
                save_checkpoint(checkpoint)
                raise
    
    # Apply all fixes and write output
    print(f"\nApplying {len(processed_queries)} fixes...")
    for idx_str, new_hyde in processed_queries.items():
        idx = int(idx_str)
        set_hyde_in_example(examples[idx], new_hyde)
    
    # Write output file
    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    # Verify
    with open(OUTPUT_FILE) as f:
        bad_count = sum(1 for line in f if BAD_PATTERN in line)
    
    print(f"\nDone! Bad hydes remaining: {bad_count}")
    print(f"Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
