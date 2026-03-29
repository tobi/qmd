# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "accelerate",
# ]
# ///
"""
QMD Retrieval-Based Evaluation with Precision & Recall

Evaluates model outputs against golden data (training set).
Measures how well the model reproduces the expected expansions.

Metrics:
- Precision: Of model-generated expansions, how many match golden?
- Recall: Of golden expansions, how many did the model generate?
- F1: Harmonic mean of precision and recall

Matching is done via token overlap (Jaccard similarity) with a threshold.

Usage:
    uv run eval_retrieval.py ./outputs/sft
    uv run eval_retrieval.py tobil/qmd-query-expansion-1.7B --golden data/qmd_expansion_v3_structured.jsonl
    uv run eval_retrieval.py ./outputs/sft --threshold 0.5 --sample 100
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# =============================================================================
# Matching Functions
# =============================================================================

def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set, removing stopwords."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 
                 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
                 'how', 'what', 'do', 'does', 'can', 'you', 'your', 'i'}
    words = re.findall(r'\b\w+\b', text.lower())
    return {w for w in words if w not in stopwords and len(w) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings based on token overlap."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def find_best_match(pred: str, golden_list: list[str], threshold: float) -> tuple[str | None, float]:
    """Find best matching golden expansion for a prediction."""
    best_match = None
    best_score = 0.0
    for golden in golden_list:
        score = jaccard_similarity(pred, golden)
        if score > best_score:
            best_score = score
            best_match = golden
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


# =============================================================================
# Parsing
# =============================================================================

def parse_model_output(text: str) -> dict[str, list[str]]:
    """Parse model output into {lex: [...], vec: [...], hyde: [...]}."""
    # Clean thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<|im_end|>', '').strip()
    
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


def parse_golden_data(searches: list[dict] | str) -> dict[str, list[str]]:
    """Parse golden data format into {lex: [...], vec: [...], hyde: [...]}."""
    # If it's a string (from messages format), parse it
    if isinstance(searches, str):
        return parse_model_output(searches)
    
    # Otherwise it's the structured format [{type, query}, ...]
    result = {"lex": [], "vec": [], "hyde": []}
    for item in searches:
        exp_type = item.get("type", "")
        value = item.get("query", "") or item.get("value", "")
        if exp_type in result:
            result[exp_type].append(value)
    return result


def load_golden_data(filepath: Path) -> list[dict]:
    """Load golden data from JSONL, supporting both structured and messages formats."""
    data = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Structured format: {query, searches}
            if "query" in item and "searches" in item:
                data.append({
                    "query": item["query"],
                    "searches": item["searches"]
                })
            # Messages format: {messages: [{role, content}, ...]}
            elif "messages" in item:
                messages = item["messages"]
                query = None
                searches = None
                for msg in messages:
                    if msg["role"] == "user":
                        # Extract query from "/no_think Expand this search query: ..."
                        content = msg["content"]
                        if "Expand this search query:" in content:
                            query = content.split("Expand this search query:")[-1].strip()
                        else:
                            query = content.strip()
                    elif msg["role"] == "assistant":
                        # The assistant content IS the expected output
                        searches = msg["content"]
                if query and searches:
                    data.append({
                        "query": query,
                        "searches": searches  # Will be parsed as string
                    })
    return data


# =============================================================================
# Metrics Calculation
# =============================================================================

# Different thresholds by type - lex needs strict matching, hyde is more flexible
DEFAULT_THRESHOLDS = {
    "lex": 0.5,   # Keywords should overlap well
    "vec": 0.35,  # Semantic sentences have more variation
    "hyde": 0.25, # Passages have the most variation
}


def calculate_metrics(
    predictions: dict[str, list[str]], 
    golden: dict[str, list[str]], 
    threshold: float | dict[str, float] = 0.4,
    return_mismatches: bool = False
) -> dict:
    """Calculate precision, recall, F1 per type and overall.
    
    Args:
        threshold: Either a single float, or dict mapping type -> threshold
        return_mismatches: If True, include lists of unmatched predictions/golden
    """
    if isinstance(threshold, (int, float)):
        thresholds = {"lex": threshold, "vec": threshold, "hyde": threshold}
    else:
        thresholds = threshold
    
    metrics = {}
    mismatches = {}
    total_tp = 0
    total_pred = 0
    total_golden = 0
    
    for exp_type in ["lex", "vec", "hyde"]:
        preds = predictions.get(exp_type, [])
        golds = golden.get(exp_type, [])
        type_threshold = thresholds.get(exp_type, 0.4)
        
        if not preds and not golds:
            continue
        
        # Track which golden items were matched
        matched_golden = set()
        unmatched_preds = []
        tp = 0
        
        for pred in preds:
            match, score = find_best_match(pred, golds, type_threshold)
            if match is not None:
                tp += 1
                matched_golden.add(match)
            else:
                unmatched_preds.append((pred, score))
        
        unmatched_golden = [g for g in golds if g not in matched_golden]
        
        precision = tp / len(preds) if preds else 0.0
        recall = len(matched_golden) / len(golds) if golds else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[exp_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_count": len(preds),
            "golden_count": len(golds),
            "matched": tp,
        }
        
        if return_mismatches:
            mismatches[exp_type] = {
                "unmatched_preds": unmatched_preds,
                "unmatched_golden": unmatched_golden,
            }
        
        total_tp += tp
        total_pred += len(preds)
        total_golden += len(golds)
    
    # Overall metrics (micro-averaged)
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_golden if total_golden > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "pred_count": total_pred,
        "golden_count": total_golden,
        "matched": total_tp,
    }
    
    if return_mismatches:
        metrics["_mismatches"] = mismatches
    
    return metrics


# =============================================================================
# Model Loading and Generation
# =============================================================================

def load_model(model_path: str):
    """Load model (adapter or merged)."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"

    # Get base model from adapter config or default
    base_model = "Qwen/Qwen3-1.7B"
    if adapter_config.exists():
        with open(adapter_config) as f:
            cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", base_model)

    print(f"Loading base: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(base_model)
    config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map={"": 0}, config=config
    )
    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    # Load adapter if present
    if adapter_config.exists():
        print(f"Loading adapter: {model_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, str(model_path))

    model.eval()
    return model, tokenizer


def generate_expansion(model, tokenizer, query: str, max_new_tokens: int = 400) -> str:
    """Generate expansion for a single query."""
    import torch
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"/no_think Expand this search query: {query}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen_tokens = out[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QMD Retrieval-Based Evaluation")
    parser.add_argument("model", help="Model path (local or HF)")
    parser.add_argument("--golden", default="data/qmd_expansion_v3_structured.jsonl",
                        help="Golden data JSONL file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Jaccard similarity threshold for all types (overrides --type-thresholds)")
    parser.add_argument("--type-thresholds", action="store_true",
                        help="Use type-specific thresholds (lex=0.5, vec=0.35, hyde=0.25)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N queries (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=400,
                        help="Max new tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-query details")
    parser.add_argument("--show-mismatches", action="store_true",
                        help="Show examples of mismatched predictions")
    args = parser.parse_args()
    
    # Determine thresholds
    if args.threshold is not None:
        thresholds = args.threshold
    elif args.type_thresholds:
        thresholds = DEFAULT_THRESHOLDS.copy()
    else:
        thresholds = 0.4  # Default single threshold

    # Load golden data
    golden_path = Path(args.golden)
    if not golden_path.exists():
        # Try relative to script directory
        golden_path = Path(__file__).parent / args.golden
    
    if not golden_path.exists():
        print(f"Error: Golden data file not found: {args.golden}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading golden data from {golden_path}...", file=sys.stderr)
    golden_data = load_golden_data(golden_path)
    print(f"Loaded {len(golden_data)} golden examples", file=sys.stderr)

    # Sample if requested
    if args.sample > 0 and args.sample < len(golden_data):
        random.seed(args.seed)
        golden_data = random.sample(golden_data, args.sample)
        print(f"Sampled {len(golden_data)} examples", file=sys.stderr)

    # Load model
    model, tokenizer = load_model(args.model)

    # Evaluate
    all_metrics = []
    all_mismatches = []
    type_aggregates = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
    
    threshold_desc = thresholds if isinstance(thresholds, (int, float)) else f"lex={thresholds['lex']}, vec={thresholds['vec']}, hyde={thresholds['hyde']}"
    print(f"\nEvaluating {len(golden_data)} queries (thresholds: {threshold_desc})...\n")
    
    for i, item in enumerate(golden_data, 1):
        query = item["query"]
        golden_parsed = parse_golden_data(item["searches"])
        
        # Generate model output
        output = generate_expansion(model, tokenizer, query, args.max_new_tokens)
        pred_parsed = parse_model_output(output)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_parsed, golden_parsed, thresholds, return_mismatches=args.show_mismatches)
        all_metrics.append({"query": query, "metrics": metrics, "pred": pred_parsed, "golden": golden_parsed})
        
        if args.show_mismatches and "_mismatches" in metrics:
            all_mismatches.append({"query": query, "mismatches": metrics.pop("_mismatches")})
        
        # Aggregate by type
        for exp_type in ["lex", "vec", "hyde", "overall"]:
            if exp_type in metrics:
                type_aggregates[exp_type]["precision"].append(metrics[exp_type]["precision"])
                type_aggregates[exp_type]["recall"].append(metrics[exp_type]["recall"])
                type_aggregates[exp_type]["f1"].append(metrics[exp_type]["f1"])
        
        # Progress
        overall = metrics.get("overall", {})
        p = overall.get("precision", 0) * 100
        r = overall.get("recall", 0) * 100
        f = overall.get("f1", 0) * 100
        
        if args.verbose:
            print(f"[{i:3d}/{len(golden_data)}] P={p:5.1f}% R={r:5.1f}% F1={f:5.1f}%  {query[:50]}")
        elif i % 50 == 0 or i == len(golden_data):
            print(f"  Processed {i}/{len(golden_data)}...", file=sys.stderr)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Threshold: {args.threshold} | Samples: {len(golden_data)}")
    print()
    
    print(f"{'Type':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    
    for exp_type in ["lex", "vec", "hyde", "overall"]:
        if exp_type in type_aggregates:
            agg = type_aggregates[exp_type]
            avg_p = sum(agg["precision"]) / len(agg["precision"]) * 100 if agg["precision"] else 0
            avg_r = sum(agg["recall"]) / len(agg["recall"]) * 100 if agg["recall"] else 0
            avg_f = sum(agg["f1"]) / len(agg["f1"]) * 100 if agg["f1"] else 0
            label = exp_type.upper() if exp_type != "overall" else "OVERALL"
            print(f"{label:<10} {avg_p:>9.1f}% {avg_r:>9.1f}% {avg_f:>9.1f}%")
    
    print(f"{'='*60}")
    
    # Show worst examples
    print("\nBottom 5 by F1:")
    sorted_by_f1 = sorted(all_metrics, key=lambda x: x["metrics"].get("overall", {}).get("f1", 0))
    for item in sorted_by_f1[:5]:
        f1 = item["metrics"].get("overall", {}).get("f1", 0) * 100
        print(f"  {f1:5.1f}%  {item['query'][:60]}")
    
    # Show mismatches if requested
    if args.show_mismatches and all_mismatches:
        print(f"\n{'='*60}")
        print("MISMATCH EXAMPLES")
        print(f"{'='*60}")
        
        # Group by type and show up to 3 examples per type
        for exp_type in ["lex", "vec", "hyde"]:
            type_mismatches = []
            for item in all_mismatches:
                if exp_type in item["mismatches"]:
                    mm = item["mismatches"][exp_type]
                    if mm["unmatched_preds"] or mm["unmatched_golden"]:
                        type_mismatches.append({
                            "query": item["query"],
                            **mm
                        })
            
            if type_mismatches:
                print(f"\n--- {exp_type.upper()} mismatches ({len(type_mismatches)} queries) ---")
                for example in type_mismatches[:3]:
                    print(f"\nQuery: {example['query'][:60]}")
                    if example["unmatched_preds"]:
                        print(f"  Unmatched predictions:")
                        for pred, score in example["unmatched_preds"][:2]:
                            print(f"    - [{score:.2f}] {pred[:80]}{'...' if len(pred) > 80 else ''}")
                    if example["unmatched_golden"]:
                        print(f"  Missing golden:")
                        for g in example["unmatched_golden"][:2]:
                            print(f"    - {g[:80]}{'...' if len(g) > 80 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
