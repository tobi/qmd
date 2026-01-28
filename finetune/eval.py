# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub",
#     "accelerate",
# ]
# ///
"""
Evaluate a QMD query expansion model.

Generates expansions from a model and scores them using the reward function.
Replaces the old two-step evals/run.py + evals/score.py workflow.

Usage:
    uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft
    uv run eval.py --model tobil/qmd-query-expansion-1.7B-grpo \
                   --sft-model tobil/qmd-query-expansion-1.7B-sft \
                   --base-model Qwen/Qwen3-1.7B
    uv run eval.py --model ./local-checkpoint --verbose
    uv run eval.py --score-only results.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Import reward module
sys.path.insert(0, os.path.dirname(__file__))
from reward import score_expansion_detailed


def load_queries(path: str) -> list[str]:
    """Load queries from file, one per line, ignoring comments and blanks."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def load_model(model_path: str, base_model: str = None, sft_model: str = None):
    """Load model with optional SFT stacking for GRPO models."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_local = Path(model_path).exists()

    # Auto-detect adapter vs full model
    is_adapter = True
    if is_local:
        is_adapter = (Path(model_path) / "adapter_config.json").exists()
        if is_adapter and not base_model:
            with open(Path(model_path) / "adapter_config.json") as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path", "Qwen/Qwen3-1.7B")

    if not base_model:
        base_model = "Qwen/Qwen3-1.7B"

    print(f"Loading tokenizer from {base_model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {base_model}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    if sft_model:
        print(f"Loading and merging SFT adapter {sft_model}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, sft_model)
        model = model.merge_and_unload()

    if is_adapter:
        print(f"Loading adapter from {model_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, model_path)

    model.eval()
    return model, tokenizer


def generate_expansion(model, tokenizer, query: str, max_new_tokens: int = 200) -> str:
    """Generate a query expansion using Qwen3 chat template with /no_think."""
    import torch

    messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "\nassistant\n" in full_output:
        expansion = full_output.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in full_output:
        expansion = full_output.split("assistant\n")[-1].strip()
    else:
        expansion = full_output[len(prompt):].strip()

    # Strip leftover <think> blocks
    import re
    if "<think>" in expansion:
        expansion = re.sub(r'<think>.*?</think>', '', expansion, flags=re.DOTALL).strip()

    return expansion


def print_result(query: str, expansion: str, scores: dict, verbose: bool = False):
    """Print a single scored result."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'~'*60}")
    print(expansion)
    print(f"{'~'*60}")
    print(f"Score: {scores['percentage']:.0f}% ({scores['rating']})")
    print(f"  Format: {scores['format']}/30  Diversity: {scores['diversity']}/30  "
          f"Hyde: {scores['hyde']}/20  Quality: {scores['quality']}/20  "
          f"Entity: {scores['entity']}/20  Think: {scores['think_bonus']}/20")
    if verbose and scores["deductions"]:
        print(f"  Issues: {', '.join(scores['deductions'][:5])}")
    if verbose and scores["entities_detected"]:
        print(f"  Entities: {scores['entities_detected']}")


def print_summary(scored_results: list):
    """Print aggregate summary."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    avg_score = sum(r["scores"]["percentage"] for r in scored_results) / len(scored_results)
    ratings = Counter(r["scores"]["rating"] for r in scored_results)

    print(f"  Total queries: {len(scored_results)}")
    print(f"  Average score: {avg_score:.1f}%")
    print(f"  Ratings:")
    for rating in ["Excellent", "Good", "Acceptable", "Poor", "Failed"]:
        count = ratings.get(rating, 0)
        print(f"    {rating:10s}: {count:2d} {'#' * count}")


def cmd_generate_and_score(args):
    """Generate expansions from a model and score them."""
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}", file=sys.stderr)

    model, tokenizer = load_model(args.model, args.base_model, args.sft_model)

    scored_results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}", file=sys.stderr)
        expansion = generate_expansion(model, tokenizer, query, args.max_tokens)
        scores = score_expansion_detailed(query, expansion)

        if not args.summary_only:
            print_result(query, expansion, scores, args.verbose)

        scored_results.append({
            "query": query,
            "expansion": expansion,
            "scores": {k: v for k, v in scores.items() if k not in ("parsed", "deductions", "entities_detected")},
            "deductions": scores["deductions"],
            "entities_detected": scores["entities_detected"],
        })

    print_summary(scored_results)

    if args.output:
        output_data = {
            "metadata": {"model": args.model, "timestamp": datetime.now().isoformat()},
            "summary": {
                "total": len(scored_results),
                "average_score": round(sum(r["scores"]["percentage"] for r in scored_results) / len(scored_results), 1),
            },
            "results": scored_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nScores saved to: {args.output}")


def cmd_score_only(args):
    """Score an existing JSONL file (from the old run.py format)."""
    results = []
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            if not data.get("_meta"):
                results.append(data)

    scored_results = []
    for result in results:
        query = result["query"]
        expansion = result["expansion"]
        scores = score_expansion_detailed(query, expansion)

        if not args.summary_only:
            print_result(query, expansion, scores, args.verbose)

        scored_results.append({
            "query": query,
            "expansion": expansion,
            "scores": {k: v for k, v in scores.items() if k not in ("parsed", "deductions", "entities_detected")},
            "deductions": scores["deductions"],
            "entities_detected": scores["entities_detected"],
        })

    print_summary(scored_results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QMD query expansion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft
  uv run eval.py --model tobil/qmd-query-expansion-1.7B-grpo \\
                 --sft-model tobil/qmd-query-expansion-1.7B-sft
  uv run eval.py --score-only evals/results.jsonl
        """,
    )

    # Model evaluation mode
    parser.add_argument("--model", help="Model path (HF Hub or local)")
    parser.add_argument("--base-model", default=None, help="Base model for tokenizer (default: Qwen/Qwen3-1.7B)")
    parser.add_argument("--sft-model", default=None, help="SFT adapter to merge first (for GRPO models)")
    parser.add_argument("--queries", default="evals/queries.txt", help="Queries file")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per generation")

    # Score-only mode
    parser.add_argument("--score-only", metavar="JSONL", help="Score existing JSONL file instead of generating")

    # Output options
    parser.add_argument("--output", "-o", help="Save detailed scores to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--summary-only", action="store_true")

    args = parser.parse_args()

    if args.score_only:
        args.input = args.score_only
        cmd_score_only(args)
    elif args.model:
        cmd_generate_and_score(args)
    else:
        parser.error("Either --model or --score-only is required")


if __name__ == "__main__":
    main()
