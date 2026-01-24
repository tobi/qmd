# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub",
# ]
# ///
"""
Evaluate QMD query expansion model quality.

Generates expansions for test queries and outputs results for review.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Test queries covering different QMD use cases
TEST_QUERIES = [
    # Technical documentation
    "how to configure authentication",
    "typescript async await",
    "docker compose networking",
    "git rebase vs merge",
    "react useEffect cleanup",

    # Short/ambiguous queries
    "auth",
    "config",
    "setup",
    "api",

    # Personal notes / journals style
    "meeting notes project kickoff",
    "ideas for new feature",
    "todo list app architecture",

    # Research / learning
    "what is dependency injection",
    "difference between sql and nosql",
    "kubernetes vs docker swarm",

    # Error/debugging
    "connection timeout error",
    "memory leak debugging",
    "cors error fix",

    # Complex queries
    "how to implement caching with redis in nodejs",
    "best practices for api rate limiting",
    "setting up ci cd pipeline with github actions",
]

PROMPT_TEMPLATE = """Expand this search query:

{query}"""


def load_model(model_name: str, base_model: str = "Qwen/Qwen3-0.6B"):
    """Load the finetuned model."""
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from {model_name}...")
    model = PeftModel.from_pretrained(base, model_name)
    model.eval()

    return model, tokenizer


def generate_expansion(model, tokenizer, query: str, max_new_tokens: int = 200) -> str:
    """Generate query expansion."""
    prompt = PROMPT_TEMPLATE.format(query=query)

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

    # Decode and extract just the generated part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt to get just the expansion
    if "Output:" in full_output:
        expansion = full_output.split("Output:")[-1].strip()
    else:
        expansion = full_output[len(prompt):].strip()

    return expansion


def evaluate_expansion(query: str, expansion: str) -> dict:
    """Basic automatic evaluation metrics."""
    lines = expansion.strip().split("\n")

    has_lex = any(l.strip().startswith("lex:") for l in lines)
    has_vec = any(l.strip().startswith("vec:") for l in lines)
    has_hyde = any(l.strip().startswith("hyde:") for l in lines)

    # Count valid lines
    valid_lines = sum(1 for l in lines if l.strip().startswith(("lex:", "vec:", "hyde:")))

    # Check for repetition
    contents = []
    for l in lines:
        if ":" in l:
            contents.append(l.split(":", 1)[1].strip().lower())
    unique_contents = len(set(contents))

    return {
        "has_lex": has_lex,
        "has_vec": has_vec,
        "has_hyde": has_hyde,
        "valid_lines": valid_lines,
        "total_lines": len(lines),
        "unique_contents": unique_contents,
        "format_score": (has_lex + has_vec + has_hyde) / 3,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tobil/qmd-query-expansion-0.6B",
                        help="Model to evaluate")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B",
                        help="Base model")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--queries", type=str, help="Custom queries file (one per line)")
    args = parser.parse_args()

    # Load custom queries if provided
    queries = TEST_QUERIES
    if args.queries:
        with open(args.queries) as f:
            queries = [l.strip() for l in f if l.strip()]

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Run evaluation
    results = []
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query}")
        print("-" * 50)

        expansion = generate_expansion(model, tokenizer, query)
        metrics = evaluate_expansion(query, expansion)

        print(expansion)
        print(f"\n  Format: {'✓' if metrics['format_score'] == 1.0 else '⚠'} "
              f"(lex:{metrics['has_lex']}, vec:{metrics['has_vec']}, hyde:{metrics['has_hyde']})")
        print(f"  Lines: {metrics['valid_lines']}/{metrics['total_lines']} valid, "
              f"{metrics['unique_contents']} unique")
        print()

        results.append({
            "query": query,
            "expansion": expansion,
            "metrics": metrics,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_format = sum(r["metrics"]["format_score"] for r in results) / len(results)
    full_format = sum(1 for r in results if r["metrics"]["format_score"] == 1.0)

    print(f"  Total queries: {len(results)}")
    print(f"  Average format score: {avg_format:.2%}")
    print(f"  Full format compliance: {full_format}/{len(results)} ({full_format/len(results):.0%})")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
