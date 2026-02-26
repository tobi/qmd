#!/usr/bin/env python3
"""
Evaluate QMD Query Expansion model.

Usage:
    python eval.py                           # Use default adapter
    python eval.py --adapter adapters/sft    # Specify adapter
    python eval.py --query "auth config"     # Test single query
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate


# Test queries for evaluation
TEST_QUERIES = [
    "auth config",
    "how to deploy",
    "rate limiting",
    "database connection",
    "api keys",
    "error handling",
    "caching strategy",
    "user permissions",
]

PROMPT_TEMPLATE = """<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
"""


def score_expansion(text: str) -> dict:
    """Score an expansion based on format and quality."""
    scores = {
        "has_lex": 0,
        "has_vec": 0,
        "has_hyde": 0,
        "lex_count": 0,
        "vec_count": 0,
        "format_valid": 0,
        "total": 0,
    }
    
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if line.startswith("lex:"):
            scores["has_lex"] = 1
            scores["lex_count"] += 1
        elif line.startswith("vec:"):
            scores["has_vec"] = 1
            scores["vec_count"] += 1
        elif line.startswith("hyde:"):
            scores["has_hyde"] = 1
    
    # Format is valid if we have at least one of each type
    if scores["has_lex"] and scores["has_vec"] and scores["has_hyde"]:
        scores["format_valid"] = 1
    
    # Total score (0-100)
    scores["total"] = (
        scores["format_valid"] * 40 +
        min(scores["lex_count"], 3) * 10 +
        min(scores["vec_count"], 3) * 10 +
        scores["has_hyde"] * 20
    )
    
    return scores


def expand_query(model, tokenizer, query: str, max_tokens: int = 256) -> str:
    """Generate expansion for a query."""
    prompt = PROMPT_TEMPLATE.format(query=query)
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    
    # Extract just the assistant response
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate QMD model")
    parser.add_argument("--model", default="models/Qwen_Qwen2.5-1.5B/mlx", help="Base model path")
    parser.add_argument("--adapter", default="adapters/sft", help="LoRA adapter path")
    parser.add_argument("--query", "-q", help="Single query to test")
    parser.add_argument("--no-adapter", action="store_true", help="Run without adapter (baseline)")
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    adapter_path = Path(args.adapter) if not args.no_adapter else None
    
    print(f"📦 Loading model: {model_path}")
    if adapter_path and adapter_path.exists():
        print(f"🔌 Loading adapter: {adapter_path}")
        model, tokenizer = load(str(model_path), adapter_path=str(adapter_path))
    else:
        if adapter_path:
            print(f"⚠️  Adapter not found: {adapter_path}, using base model")
        model, tokenizer = load(str(model_path))
    
    # Single query mode
    if args.query:
        print(f"\n🔍 Query: {args.query}\n")
        expansion = expand_query(model, tokenizer, args.query)
        print(expansion)
        print(f"\n📊 Score: {score_expansion(expansion)}")
        return
    
    # Batch evaluation
    print(f"\n📝 Evaluating {len(TEST_QUERIES)} test queries...\n")
    
    results = []
    total_score = 0
    
    for query in TEST_QUERIES:
        print(f"🔍 {query}")
        expansion = expand_query(model, tokenizer, query)
        scores = score_expansion(expansion)
        total_score += scores["total"]
        
        results.append({
            "query": query,
            "expansion": expansion,
            "scores": scores,
        })
        
        # Print summary
        status = "✅" if scores["format_valid"] else "❌"
        print(f"   {status} Score: {scores['total']}/100")
        print(f"   lex:{scores['lex_count']} vec:{scores['vec_count']} hyde:{scores['has_hyde']}")
        print()
    
    # Summary
    avg_score = total_score / len(TEST_QUERIES)
    print(f"\n{'='*50}")
    print(f"📊 Average Score: {avg_score:.1f}/100")
    print(f"{'='*50}")
    
    # Save results
    output_file = Path("eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
