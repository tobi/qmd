#!/usr/bin/env python3
"""
Quick demo: Use the trained QMD query expansion model.

Usage:
    python demo.py "your search query"
    python demo.py --interactive
"""

import argparse
import sys

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROMPT_TEMPLATE = """<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
"""


def expand_query(model, tokenizer, query: str, temp: float = 0.0) -> str:
    """Generate query expansion."""
    prompt = PROMPT_TEMPLATE.format(query=query)
    sampler = make_sampler(temp=temp)
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=200,
        sampler=sampler,
        verbose=False
    )
    return response.replace('<|im_end|>', '').strip()


def main():
    parser = argparse.ArgumentParser(description="QMD Query Expansion Demo")
    parser.add_argument("query", nargs="?", help="Query to expand")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model", default="models/Qwen_Qwen2.5-1.5B/mlx", help="Base model path")
    parser.add_argument("--adapter", default="adapters/sft", help="LoRA adapter path")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature (0=deterministic)")
    args = parser.parse_args()

    print("Loading model...")
    try:
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying without adapter (base model)...")
        model, tokenizer = load(args.model)

    print("Ready!\n")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query> ").strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                if not query:
                    continue
                print("\nExpansion:")
                print(expand_query(model, tokenizer, query, args.temp))
                print()
            except (KeyboardInterrupt, EOFError):
                break
    elif args.query:
        print(f"Query: {args.query}\n")
        print("Expansion:")
        print(expand_query(model, tokenizer, args.query, args.temp))
    else:
        # Demo queries
        demos = [
            "auth config",
            "docker networking",
            "how to deploy",
            "kubernetes pod",
        ]
        for q in demos:
            print(f"Query: {q}")
            print("-" * 40)
            print(expand_query(model, tokenizer, q, args.temp))
            print()


if __name__ == "__main__":
    main()
