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
Generate query expansions from a model and save to JSONL.

Usage:
    uv run evals/run.py --model tobil/qmd-query-expansion-0.6B-v4
    uv run evals/run.py --model ./local-model --queries evals/queries.txt
    uv run evals/run.py --model tobil/qmd-query-expansion-0.6B-v4 --output results.jsonl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_queries(path: str) -> list[str]:
    """Load queries from file, one per line, ignoring comments."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def load_model(model_path: str, base_model: str = None):
    """
    Load the model (supports Hub URLs, local adapters, and merged models).

    Args:
        model_path: HF Hub ID (tobil/model), local adapter dir, or merged model dir
        base_model: Base model for tokenizer (auto-detected if not provided)
    """
    model_path = Path(model_path) if not model_path.startswith(("http", "hf://")) and "/" not in model_path[:20] else model_path
    is_local = isinstance(model_path, Path) and model_path.exists()

    # Check if it's an adapter or full model
    is_adapter = False
    if is_local:
        adapter_config = model_path / "adapter_config.json"
        is_adapter = adapter_config.exists()
        if is_adapter and not base_model:
            # Read base model from adapter config
            import json
            with open(adapter_config) as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path", "Qwen/Qwen3-0.6B")
    else:
        # For Hub models, assume adapter unless --base-model not provided
        is_adapter = base_model is not None

    # Default base model
    if not base_model:
        base_model = "Qwen/Qwen3-0.6B"

    print(f"Loading tokenizer from {base_model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        print(f"Loading base model {base_model}...", file=sys.stderr)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loading adapter from {model_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        print(f"Loading model from {model_path}...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def generate_expansion(model, tokenizer, query: str, max_new_tokens: int = 200) -> str:
    """Generate query expansion using Qwen3 chat template with /no_think."""
    messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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

    # Remove any <think> tags
    if expansion.startswith("<think>"):
        think_end = expansion.find("</think>")
        if think_end != -1:
            expansion = expansion[think_end + 8:].strip()

    return expansion


def main():
    parser = argparse.ArgumentParser(description="Generate query expansions")
    parser.add_argument("--model", required=True, help="Model path (Hub or local)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Base model for tokenizer")
    parser.add_argument("--queries", default="evals/queries.txt", help="Queries file")
    parser.add_argument("--output", help="Output JSONL file (default: evals/results_{model_name}.jsonl)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = args.model.replace("/", "_").replace(".", "_")
        output_path = Path(f"evals/results_{model_name}.jsonl")

    # Load queries
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}", file=sys.stderr)

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Generate expansions
    print(f"Generating expansions...", file=sys.stderr)
    results = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}", file=sys.stderr)
        expansion = generate_expansion(model, tokenizer, query, args.max_tokens)
        results.append({
            "query": query,
            "expansion": expansion,
        })

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Write metadata as first line
        metadata = {
            "_meta": True,
            "model": args.model,
            "base_model": args.base_model,
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries),
        }
        f.write(json.dumps(metadata) + "\n")

        # Write results
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to: {output_path}", file=sys.stderr)
    print(str(output_path))  # Print path to stdout for piping


if __name__ == "__main__":
    main()
