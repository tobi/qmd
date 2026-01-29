# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub>=0.20.0",
#     "accelerate",
# ]
# ///
"""
Evaluate QMD query expansion models on HuggingFace Jobs.

Self-contained script — inlines the reward function and test queries.

    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 30m jobs/eval.py
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 30m jobs/eval.py -- --sft-only
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
GRPO_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"

# --- Test queries (inlined from evals/queries.txt) ---
QUERIES = [
    # Technical documentation
    "how to configure authentication",
    "typescript async await",
    "docker compose networking",
    "git rebase vs merge",
    "react useEffect cleanup",
    # Short/ambiguous
    "auth",
    "config",
    "setup",
    "api",
    # Named entities
    "who is TDS motorsports",
    "React hooks tutorial",
    "Docker container networking",
    "Kubernetes pod deployment",
    "AWS Lambda functions",
    # Personal notes / journals
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
    # Temporal / recency
    "recent news about Shopify",
    "latest AI developments",
    "best laptops right now",
    "what changed in kubernetes latest version",
    # Complex
    "how to implement caching with redis in nodejs",
    "best practices for api rate limiting",
    "setting up ci cd pipeline with github actions",
]

# =============================================================================
# Reward function (inlined from reward.py)
# =============================================================================

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in',
    'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
})

KEY_TERM_STOPWORDS = frozenset({
    'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
    'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
    'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell',
})

GENERIC_LEX_PHRASES = frozenset({
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with',
})

CHAT_TEMPLATE_TOKENS = frozenset({
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\nassistant\n', '\nuser\n',
})


def parse_expansion(text):
    result = {"lex": [], "vec": [], "hyde": [], "invalid": []}
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
        else:
            result["invalid"].append(line)
    return result


def clean_model_output(text):
    text = text.replace('<|im_end|>', '').strip()
    used_thinking = '<think>' in text and '</think>' in text
    if used_thinking:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text, used_thinking


def extract_named_entities(query):
    entities = set()
    words = query.split()
    prev_was_entity = False
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue
        is_entity = False
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower()); is_entity = True
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower()); is_entity = True
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower()); is_entity = True
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower()); is_entity = True
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower()); is_entity = True
        prev_was_entity = is_entity
    return entities


def get_key_terms(query):
    return set(query.lower().split()) - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line, query):
    key_terms = get_key_terms(query)
    return not key_terms or bool(key_terms & set(lex_line.lower().split()))


def lex_preserves_entities(line, entities):
    if not entities: return True
    return any(e in line.lower() for e in entities)


def lex_is_generic(lex_line):
    lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lower or lower.startswith(phrase.split()[0]):
            remaining = lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


def word_set_distance(a, b):
    return len(set(a.lower().split()) ^ set(b.lower().split()))


def is_diverse(a, b, min_distance=2):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a: return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion, query):
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text):
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


def score_expansion_detailed(query, expansion):
    text, used_thinking = clean_model_output(expansion.strip())
    deductions = []

    def _fail(reason):
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "think_bonus": 0, "total": 0, "max_possible": 100,
            "percentage": 0.0, "rating": "Failed", "deductions": [reason],
        }

    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return _fail("CHAT TEMPLATE LEAKAGE")
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return _fail(f"INVALID LINE: {line[:50]}")

    parsed = parse_expansion(text)

    format_score = 10
    if parsed["lex"]: format_score += 10
    else: deductions.append("missing lex:")
    if parsed["vec"]: format_score += 10
    else: deductions.append("missing vec:")

    diversity_score = 0
    types_present = sum(1 for t in ("lex", "vec") if parsed[t])
    if types_present >= 2: diversity_score += 10
    if len(parsed["lex"]) + len(parsed["vec"]) >= 2: diversity_score += 5
    lex_div = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2): lex_div -= 2
    diversity_score += max(0, lex_div)
    vec_div = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3): vec_div -= 2
    diversity_score += max(0, vec_div)
    echo = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query): echo -= 3
    diversity_score += max(0, echo)

    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200: hyde_score += 5
        elif hyde_len < 50: hyde_score += 2
        if "\n" not in hyde_text: hyde_score += 5
        hyde_score += max(0, 5 - word_repetition_penalty(hyde_text))

    quality_score = 5
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec: quality_score += 5
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        quality_score += 5 if natural == len(parsed["vec"]) else 2
    if parsed["lex"]:
        with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if with_terms == len(parsed["lex"]): quality_score += 5
        elif with_terms > 0: quality_score += 2

    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if with_entities == len(parsed["lex"]): entity_score += 15
        elif with_entities > 0: entity_score += 5
        else: entity_score -= 30
        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count: entity_score -= generic_count * 15
        if parsed["vec"]:
            vec_with = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with > 0: entity_score += 5
    elif not entities:
        entity_score = 10

    think_bonus = 0 if used_thinking else 20
    total = format_score + diversity_score + hyde_score + quality_score + entity_score + think_bonus
    max_possible = 140 if parsed["hyde"] else 120
    percentage = max(0.0, min(100.0, total / max_possible * 100))

    if percentage >= 80: rating = "Excellent"
    elif percentage >= 60: rating = "Good"
    elif percentage >= 40: rating = "Acceptable"
    elif percentage >= 20: rating = "Poor"
    else: rating = "Failed"

    return {
        "format": format_score, "diversity": diversity_score, "hyde": hyde_score,
        "quality": quality_score, "entity": max(0, entity_score),
        "think_bonus": think_bonus, "total": max(0, total),
        "max_possible": max_possible, "percentage": round(percentage, 1),
        "rating": rating, "deductions": deductions,
        "entities_detected": list(entities) if entities else [],
    }


# =============================================================================
# Model loading and generation
# =============================================================================

def load_model(base, sft=None, grpo=None):
    print(f"Loading tokenizer from {base}...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {base}...")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
    )

    if sft:
        print(f"Loading and merging SFT adapter {sft}...")
        model = PeftModel.from_pretrained(model, sft)
        model = model.merge_and_unload()

    if grpo:
        print(f"Loading GRPO adapter {grpo}...")
        model = PeftModel.from_pretrained(model, grpo)

    model.eval()
    return model, tokenizer


def generate_expansion(model, tokenizer, query, max_new_tokens=200):
    messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "\nassistant\n" in full_output:
        expansion = full_output.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in full_output:
        expansion = full_output.split("assistant\n")[-1].strip()
    else:
        expansion = full_output[len(prompt):].strip()

    if "<think>" in expansion:
        expansion = re.sub(r'<think>.*?</think>', '', expansion, flags=re.DOTALL).strip()
    return expansion


# =============================================================================
# Main
# =============================================================================

def results_to_csv(results, label):
    """Convert eval results to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "model", "query", "expansion", "score_pct", "rating",
        "format", "diversity", "hyde", "quality", "entity", "think_bonus",
        "total", "max_possible", "deductions",
    ])
    for r in results:
        s = r["scores"]
        writer.writerow([
            label, r["query"], r["expansion"], s["percentage"], s["rating"],
            s["format"], s["diversity"], s["hyde"], s["quality"], s["entity"],
            s["think_bonus"], s["total"], s["max_possible"],
            "; ".join(s.get("deductions", [])),
        ])
    return buf.getvalue()


def upload_csv(results, label, repo_id, api):
    """Upload eval results CSV to HuggingFace Hub."""
    csv_data = results_to_csv(results, label)
    tag = label.split("/")[-1].replace(" ", "_").lower()
    filename = f"eval_{tag}.csv"
    print(f"  Uploading {filename} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=csv_data.encode("utf-8"),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploaded: https://huggingface.co/{repo_id}/blob/main/{filename}")


def evaluate_model(model, tokenizer, label):
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {label}")
    print(f"{'='*70}")

    results = []
    for i, query in enumerate(QUERIES, 1):
        expansion = generate_expansion(model, tokenizer, query)
        scores = score_expansion_detailed(query, expansion)
        results.append({"query": query, "expansion": expansion, "scores": scores})

        marker = "+" if scores["percentage"] >= 80 else "-" if scores["percentage"] < 60 else "~"
        print(f"  [{marker}] {i:2d}/{len(QUERIES)} {scores['percentage']:5.1f}% {scores['rating']:10s}  {query}")

    avg = sum(r["scores"]["percentage"] for r in results) / len(results)
    ratings = Counter(r["scores"]["rating"] for r in results)

    print(f"\n  {'─'*50}")
    print(f"  Average score: {avg:.1f}%")
    print(f"  Ratings:")
    for rating in ["Excellent", "Good", "Acceptable", "Poor", "Failed"]:
        count = ratings.get(rating, 0)
        if count > 0:
            print(f"    {rating:10s}: {count:2d}  {'█' * count}")

    # Show worst queries
    worst = sorted(results, key=lambda r: r["scores"]["percentage"])[:5]
    print(f"\n  Bottom 5:")
    for r in worst:
        print(f"    {r['scores']['percentage']:5.1f}%  {r['query']}")
        if r["scores"]["deductions"]:
            print(f"           {', '.join(r['scores']['deductions'][:3])}")

    return results, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-only", action="store_true", help="Only evaluate SFT model")
    parser.add_argument("--upload-repo", default="tobil/qmd-query-expansion-evals",
                        help="HF repo to upload CSV results")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    api = HfApi()
    api.create_repo(repo_id=args.upload_repo, repo_type="model", exist_ok=True)

    # Evaluate SFT
    model, tokenizer = load_model(BASE_MODEL, sft=SFT_MODEL)
    sft_results, sft_avg = evaluate_model(model, tokenizer, f"SFT: {SFT_MODEL}")
    upload_csv(sft_results, "sft", args.upload_repo, api)

    if not args.sft_only:
        # For GRPO: reload base, merge SFT, then load GRPO adapter
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_model(BASE_MODEL, sft=SFT_MODEL, grpo=GRPO_MODEL)
        grpo_results, grpo_avg = evaluate_model(model, tokenizer, f"GRPO: {GRPO_MODEL}")
        upload_csv(grpo_results, "grpo", args.upload_repo, api)

        # Upload combined comparison CSV
        combined = results_to_csv(sft_results, "sft") + results_to_csv(grpo_results, "grpo").split("\n", 1)[1]
        api.upload_file(
            path_or_fileobj=combined.encode("utf-8"),
            path_in_repo="eval_comparison.csv",
            repo_id=args.upload_repo,
            repo_type="model",
        )
        print(f"  Uploaded: eval_comparison.csv")

        # Comparison
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"  SFT  average: {sft_avg:.1f}%")
        print(f"  GRPO average: {grpo_avg:.1f}%")
        print(f"  Delta:        {grpo_avg - sft_avg:+.1f}%")

        improved = sum(1 for s, g in zip(sft_results, grpo_results)
                       if g["scores"]["percentage"] > s["scores"]["percentage"])
        regressed = sum(1 for s, g in zip(sft_results, grpo_results)
                        if g["scores"]["percentage"] < s["scores"]["percentage"])
        print(f"  Improved: {improved}/{len(QUERIES)}, Regressed: {regressed}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
