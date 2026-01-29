"""
Common evaluation and reward scoring for QMD query expansion models.

Shared by sft.py and grpo.py for post-training evaluation.
"""

import csv
import io
import re
from collections import Counter

import torch
from huggingface_hub import HfApi

# =============================================================================
# Reward function (single source of truth)
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
    if not entities:
        return True
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
    if a == b or a in b or b in a:
        return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion, query):
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text):
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


def score_expansion(query, expansion):
    """Score expansion as float in [0.0, 1.0] for RL reward."""
    text, used_thinking = clean_model_output(expansion.strip())

    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return 0.0
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return 0.0

    parsed = parse_expansion(text)

    format_score = 10
    if parsed["lex"]: format_score += 10
    if parsed["vec"]: format_score += 10

    diversity_score = 0
    if sum(1 for t in ("lex", "vec") if parsed[t]) >= 2: diversity_score += 10
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
        if 50 <= len(hyde_text) <= 200: hyde_score += 5
        elif len(hyde_text) < 50: hyde_score += 2
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
    return max(0.0, min(1.0, total / max_possible))


def extract_query_from_prompt(prompt):
    """Extract the search query from a formatted prompt string."""
    if "Expand this search query:" in prompt:
        query = prompt.split("Expand this search query:")[-1].strip()
        if "<|im_end|>" in query:
            query = query.split("<|im_end|>")[0].strip()
        return query
    return prompt.strip()


class QMDRewardFunction:
    """Reward function wrapper for TRL's GRPOTrainer."""
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])
            rewards.append(score_expansion(query, completion))
        return rewards


# =============================================================================
# Evaluation
# =============================================================================

EVAL_QUERIES = [
    # Technical documentation
    "how to configure authentication",
    "typescript async await",
    "docker compose networking",
    "git rebase vs merge",
    "react useEffect cleanup",
    # Short/ambiguous
    "auth", "config", "setup", "api",
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


def generate_expansion(model, tokenizer, query, max_new_tokens=200):
    """Generate a query expansion using the model."""
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
        return full_output.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in full_output:
        return full_output.split("assistant\n")[-1].strip()
    return full_output[len(prompt):].strip()


def run_eval(model, tokenizer, label, upload_repo="tobil/qmd-query-expansion-evals"):
    """Evaluate model on EVAL_QUERIES, print results, upload CSV."""
    api = HfApi()
    api.create_repo(repo_id=upload_repo, repo_type="model", exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EVALUATING: {label}")
    print(f"{'='*70}")

    results = []
    for i, query in enumerate(EVAL_QUERIES, 1):
        expansion = generate_expansion(model, tokenizer, query)
        score = score_expansion(query, expansion)
        pct = round(score * 100, 1)
        rating = ("Excellent" if pct >= 80 else "Good" if pct >= 60
                  else "Acceptable" if pct >= 40 else "Poor" if pct >= 20 else "Failed")
        marker = "+" if pct >= 80 else "-" if pct < 60 else "~"
        print(f"  [{marker}] {i:2d}/{len(EVAL_QUERIES)} {pct:5.1f}% {rating:10s}  {query}")
        results.append({"query": query, "expansion": expansion, "score": pct, "rating": rating})

    avg = sum(r["score"] for r in results) / len(results)
    ratings = Counter(r["rating"] for r in results)

    print(f"\n  {'─'*50}")
    print(f"  Average score: {avg:.1f}%")
    for r in ["Excellent", "Good", "Acceptable", "Poor", "Failed"]:
        c = ratings.get(r, 0)
        if c:
            print(f"    {r:10s}: {c:2d}  {'█' * c}")

    worst = sorted(results, key=lambda r: r["score"])[:5]
    print(f"\n  Bottom 5:")
    for r in worst:
        print(f"    {r['score']:5.1f}%  {r['query']}")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["model", "query", "expansion", "score_pct", "rating"])
    for r in results:
        writer.writerow([label, r["query"], r["expansion"], r["score"], r["rating"]])

    filename = f"eval_{label}.csv"
    print(f"\n  Uploading {filename} to {upload_repo}...")
    api.upload_file(
        path_or_fileobj=buf.getvalue().encode("utf-8"),
        path_in_repo=filename,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"  Done: https://huggingface.co/{upload_repo}/blob/main/{filename}")
