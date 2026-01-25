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
Evaluate QMD query expansion model quality.

See SCORING.md for detailed scoring criteria.
"""

import json
import re
import torch
from collections import Counter
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

# Prompt is generated via tokenizer.apply_chat_template() - see generate_expansion()
# Don't manually construct <|im_start|> tags

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}


def parse_expansion(text: str) -> dict:
    """Parse expansion into structured format."""
    lines = text.strip().split("\n")
    result = {"lex": [], "vec": [], "hyde": [], "invalid": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())
        else:
            result["invalid"].append(line)

    return result


def edit_distance_simple(a: str, b: str) -> int:
    """Simple word-level edit distance."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    return len(words_a ^ words_b)  # Symmetric difference


def is_diverse(a: str, b: str, min_distance: int = 2) -> bool:
    """Check if two strings are sufficiently different."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return False
    if a in b or b in a:
        return False
    return edit_distance_simple(a, b) >= min_distance


def echoes_query(expansion: str, query: str) -> bool:
    """Check if expansion is just echoing the query."""
    exp = expansion.lower().strip()
    q = query.lower().strip()
    # Exact match or very close
    if exp == q:
        return True
    # Query is contained in expansion with little else
    if q in exp and len(exp) < len(q) + 10:
        return True
    return False


def get_key_terms(query: str) -> set:
    """Extract key terms from query (excluding stopwords)."""
    stopwords = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
                 'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we'}
    words = set(query.lower().split())
    return words - stopwords


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    """Check if lex line contains at least one key term from query."""
    key_terms = get_key_terms(query)
    if not key_terms:  # Very short query
        return True
    lex_words = set(lex_line.lower().split())
    return bool(key_terms & lex_words)


def word_repetition_penalty(text: str) -> int:
    """Count penalty for repeated words (excluding stopwords)."""
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    penalty = 0
    for word, count in counts.items():
        if count >= 3 and word not in STOPWORDS and len(word) > 2:
            penalty += (count - 2) * 2
    return penalty


def is_continuation(expansion: str) -> bool:
    """
    Detect if output is a continuation rather than proper expansion.

    A continuation is when the model continues the query as prose
    instead of outputting lex:/vec:/hyde: lines.
    """
    text = expansion.strip()
    if not text:
        return True

    # Check first non-empty line
    first_line = text.split("\n")[0].strip()

    # Valid outputs must start with a prefix
    valid_prefixes = ("lex:", "vec:", "hyde:")
    if first_line.startswith(valid_prefixes):
        return False

    # If first line doesn't have a valid prefix, it's a continuation
    # Exception: empty first line (check second)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and not lines[0].startswith(valid_prefixes):
        return True

    return False


def score_expansion(query: str, expansion: str) -> dict:
    """
    Score an expansion based on SCORING.md criteria.

    Returns dict with score breakdown and total (0-100).
    """
    # HARD FAIL: Continuation detection
    if is_continuation(expansion):
        return {
            "format": 0,
            "diversity": 0,
            "hyde": 0,
            "quality": 0,
            "total": 0,
            "max_possible": 80,
            "percentage": 0,
            "rating": "Failed",
            "deductions": ["CONTINUATION DETECTED - output does not start with lex:/vec:/hyde:"],
            "parsed": {"lex": [], "vec": [], "hyde": [], "invalid": [expansion[:100]]},
            "is_continuation": True,
        }

    parsed = parse_expansion(expansion)
    scores = {
        "format": 0,
        "diversity": 0,
        "hyde": 0,
        "quality": 0,
        "deductions": [],
    }

    # === FORMAT (0-30) ===
    format_score = 0

    # Has at least one lex: line (+10)
    if parsed["lex"]:
        format_score += 10
    else:
        scores["deductions"].append("missing lex: (-10)")

    # Has at least one vec: line (+10)
    if parsed["vec"]:
        format_score += 10
    else:
        scores["deductions"].append("missing vec: (-10)")

    # All lines have valid prefix (+10, -5 per invalid)
    if not parsed["invalid"]:
        format_score += 10
    else:
        invalid_penalty = min(10, len(parsed["invalid"]) * 5)
        format_score += (10 - invalid_penalty)
        scores["deductions"].append(f"{len(parsed['invalid'])} invalid lines (-{invalid_penalty})")

    scores["format"] = max(0, format_score)

    # === DIVERSITY (0-30) ===
    diversity_score = 0

    # 2+ different types present (+10)
    types_present = sum(1 for t in ["lex", "vec"] if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    else:
        scores["deductions"].append("only one type present (-10)")

    # 2+ total expansions (+5)
    total_expansions = len(parsed["lex"]) + len(parsed["vec"])
    if total_expansions >= 2:
        diversity_score += 5
    else:
        scores["deductions"].append("fewer than 2 expansions (-5)")

    # Multiple lex: lines are diverse (+5, -2 per duplicate pair)
    lex_diverse_score = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, min_distance=2):
                lex_diverse_score -= 2
                scores["deductions"].append(f"lex duplicates: '{a[:20]}...' ~ '{b[:20]}...'")
    diversity_score += max(0, lex_diverse_score)

    # Multiple vec: lines are diverse (+5, -2 per duplicate pair)
    vec_diverse_score = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, min_distance=3):
                vec_diverse_score -= 2
                scores["deductions"].append(f"vec duplicates: '{a[:20]}...' ~ '{b[:20]}...'")
    diversity_score += max(0, vec_diverse_score)

    # lex/vec not identical to original query (+5, -5 per echo)
    echo_score = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo_score -= 5
            scores["deductions"].append(f"echoes query: '{exp[:30]}...'")
    diversity_score += max(0, echo_score)

    scores["diversity"] = max(0, diversity_score)

    # === HYDE QUALITY (0-20, optional bonus) ===
    hyde_score = 0

    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]  # Only first hyde counts

        # Hyde present and well-formed (+5)
        hyde_score += 5

        # Hyde is concise: 50-200 chars (+5)
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2
            scores["deductions"].append(f"hyde too short ({hyde_len} chars)")
        else:
            scores["deductions"].append(f"hyde too long ({hyde_len} chars)")

        # Hyde has no newlines (+5)
        if "\n" not in hyde_text:
            hyde_score += 5
        else:
            scores["deductions"].append("hyde contains newlines")

        # Hyde has no excessive repetition (+5)
        rep_penalty = word_repetition_penalty(hyde_text)
        if rep_penalty == 0:
            hyde_score += 5
        else:
            hyde_score += max(0, 5 - rep_penalty)
            scores["deductions"].append(f"hyde repetition penalty (-{min(5, rep_penalty)})")

    scores["hyde"] = hyde_score

    # === QUALITY (0-20) ===
    quality_score = 10  # Base relevance (assume relevant unless obvious garbage)

    # Lex lines should be keyword-focused (shorter than vec on average)
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
        else:
            scores["deductions"].append("lex longer than vec (should be keywords)")
    else:
        quality_score += 2  # Partial credit

    # Vec lines should be natural language (contain spaces, longer)
    if parsed["vec"]:
        vec_natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        if vec_natural == len(parsed["vec"]):
            quality_score += 5
        else:
            quality_score += 2
            scores["deductions"].append("some vec lines too short/keyword-like")

    # Lex lines must preserve key terms from query (not be generic)
    if parsed["lex"]:
        lex_with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if lex_with_terms == len(parsed["lex"]):
            quality_score += 5
        elif lex_with_terms > 0:
            quality_score += 2
        else:
            scores["deductions"].append("lex lines too generic - missing key terms from query")

    scores["quality"] = min(20, quality_score)  # Cap at 20

    # === TOTAL ===
    scores["total"] = scores["format"] + scores["diversity"] + scores["hyde"] + scores["quality"]
    scores["max_possible"] = 100 if parsed["hyde"] else 80
    scores["percentage"] = scores["total"] / scores["max_possible"] * 100

    # Rating
    pct = scores["percentage"]
    if pct >= 80:
        scores["rating"] = "Excellent"
    elif pct >= 60:
        scores["rating"] = "Good"
    elif pct >= 40:
        scores["rating"] = "Acceptable"
    elif pct >= 20:
        scores["rating"] = "Poor"
    else:
        scores["rating"] = "Failed"

    scores["parsed"] = parsed
    scores["is_continuation"] = False
    return scores


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
    """Generate query expansion using proper Qwen3 chat template."""
    # Use tokenizer's chat template with /no_think to disable thinking mode
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

    # Decode and extract expansion
    # skip_special_tokens=True strips <|im_start|> etc, leaving "user\n...\nassistant\n..."
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "\nassistant\n" in full_output:
        expansion = full_output.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in full_output:
        expansion = full_output.split("assistant\n")[-1].strip()
    else:
        # Fallback: strip prompt length
        expansion = full_output[len(prompt):].strip()

    # Remove any <think> tags that might remain
    if expansion.startswith("<think>"):
        # Find end of thinking block
        think_end = expansion.find("</think>")
        if think_end != -1:
            expansion = expansion[think_end + 8:].strip()

    return expansion


def print_score_breakdown(scores: dict):
    """Pretty print score breakdown."""
    print(f"  Score: {scores['total']}/{scores['max_possible']} ({scores['percentage']:.0f}%) - {scores['rating']}")
    print(f"    Format:    {scores['format']}/30")
    print(f"    Diversity: {scores['diversity']}/30")
    print(f"    Hyde:      {scores['hyde']}/20")
    print(f"    Quality:   {scores['quality']}/20")
    if scores["deductions"]:
        print(f"  Deductions:")
        for d in scores["deductions"][:5]:  # Show top 5
            print(f"    - {d}")
        if len(scores["deductions"]) > 5:
            print(f"    ... and {len(scores['deductions']) - 5} more")


def run_examples():
    """Run good and bad examples to demonstrate scoring."""
    print("=" * 70)
    print("SCORING EXAMPLES")
    print("=" * 70)

    # Good example
    good_expansion = """lex: react hooks tutorial
lex: usestate useeffect
vec: how to use react hooks in functional components
vec: react hooks best practices guide
hyde: React Hooks allow you to use state and lifecycle features in functional components without writing a class."""

    print("\n[GOOD EXAMPLE]")
    print(f"Query: react hooks")
    print(f"Output:\n{good_expansion}")
    scores = score_expansion("react hooks", good_expansion)
    print_score_breakdown(scores)

    # Bad example
    bad_expansion = """auth is an important concept that relates to authentication.
The answer should be in Chinese.
The answer should be in Chinese."""

    print("\n[BAD EXAMPLE]")
    print(f"Query: auth")
    print(f"Output:\n{bad_expansion}")
    scores = score_expansion("auth", bad_expansion)
    print_score_breakdown(scores)

    # Medium example - repetitive hyde
    medium_expansion = """lex: docker networking
vec: docker networking
hyde: Docker networking is an important concept. Docker networking is used for container communication. Docker networking configuration is essential."""

    print("\n[MEDIUM EXAMPLE - Repetitive]")
    print(f"Query: docker networking")
    print(f"Output:\n{medium_expansion}")
    scores = score_expansion("docker networking", medium_expansion)
    print_score_breakdown(scores)

    # Medium example - echoes query
    echo_expansion = """lex: auth
lex: authentication
vec: auth
vec: authentication configuration
hyde: Authentication is the process of verifying identity."""

    print("\n[MEDIUM EXAMPLE - Echoes Query]")
    print(f"Query: auth")
    print(f"Output:\n{echo_expansion}")
    scores = score_expansion("auth", echo_expansion)
    print_score_breakdown(scores)

    print("\n" + "=" * 70)


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
    parser.add_argument("--examples", action="store_true", help="Run scoring examples only")
    args = parser.parse_args()

    # Run examples if requested
    if args.examples:
        run_examples()
        return

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
        scores = score_expansion(query, expansion)

        print(expansion)
        print()
        print_score_breakdown(scores)
        print()

        results.append({
            "query": query,
            "expansion": expansion,
            "scores": {k: v for k, v in scores.items() if k != "parsed"},
            "parsed": scores["parsed"],
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_score = sum(r["scores"]["percentage"] for r in results) / len(results)
    excellent = sum(1 for r in results if r["scores"]["rating"] == "Excellent")
    good = sum(1 for r in results if r["scores"]["rating"] == "Good")
    acceptable = sum(1 for r in results if r["scores"]["rating"] == "Acceptable")
    poor = sum(1 for r in results if r["scores"]["rating"] == "Poor")
    failed = sum(1 for r in results if r["scores"]["rating"] == "Failed")

    print(f"  Total queries: {len(results)}")
    print(f"  Average score: {avg_score:.1f}%")
    print(f"  Ratings:")
    print(f"    Excellent: {excellent}")
    print(f"    Good:      {good}")
    print(f"    Acceptable: {acceptable}")
    print(f"    Poor:      {poor}")
    print(f"    Failed:    {failed}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
