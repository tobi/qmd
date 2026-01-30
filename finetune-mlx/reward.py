# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
QMD Query Expansion Reward Function

Single source of truth for scoring query expansions. Used by:
- GRPO training (as the RL reward signal)
- Evaluation scripts (for scoring model outputs)

Scores expansions on five dimensions:
  Format (30)   - Has lex/vec lines, no invalid lines
  Diversity (30) - Multiple types, diverse content, no echoes
  HyDE (20)      - Optional bonus for hypothetical document passage
  Quality (20)   - Lex shorter than vec, natural language, key terms
  Entity (20)    - Named entity preservation in lex/vec lines

Returns 0.0-1.0 for RL rewards, or a detailed breakdown dict for evaluation.
"""

import re
from collections import Counter

# =============================================================================
# Constants
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

# Chat template tokens that indicate a broken output
CHAT_TEMPLATE_TOKENS = frozenset({
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\nassistant\n', '\nuser\n',
})


# =============================================================================
# Parsing
# =============================================================================

def parse_expansion(text: str) -> dict:
    """Parse a multi-line expansion into {lex, vec, hyde, invalid} lists."""
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


def clean_model_output(text: str) -> tuple[str, bool]:
    """Strip chat template artifacts from model output.

    Returns (cleaned_text, used_thinking) where used_thinking is True
    if the model emitted <think>...</think> blocks.
    """
    text = text.replace('<|im_end|>', '').strip()

    used_thinking = '<think>' in text and '</think>' in text
    if used_thinking:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    return text, used_thinking


# =============================================================================
# Helpers
# =============================================================================

def extract_named_entities(query: str) -> set:
    """Extract named entities using heuristics.

    Detects: ALL-CAPS acronyms (TDS, API), capitalized proper nouns (React),
    technical terms with special chars (node.js, C++), CamelCase (JavaScript),
    and compound names (TDS motorsports -> both words).
    """
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
            entities.add(clean.lower())
            is_entity = True
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_entity = True
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        prev_was_entity = is_entity

    return entities


def get_key_terms(query: str) -> set:
    """Get non-stopword terms from a query."""
    return set(query.lower().split()) - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    """Does the lex line contain at least one key term from the query?"""
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    return bool(key_terms & set(lex_line.lower().split()))


def lex_preserves_entities(line: str, entities: set) -> bool:
    """Does the line contain at least one named entity?"""
    if not entities:
        return True
    lower = line.lower()
    return any(e in lower for e in entities)


def lex_is_generic(lex_line: str) -> bool:
    """Is this lex line a useless generic filler phrase?"""
    lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lower or lower.startswith(phrase.split()[0]):
            remaining = lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


def word_set_distance(a: str, b: str) -> int:
    """Symmetric difference of word sets (how many words are unique to one)."""
    return len(set(a.lower().split()) ^ set(b.lower().split()))


def is_diverse(a: str, b: str, min_distance: int = 2) -> bool:
    """Are two strings sufficiently different?"""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion: str, query: str) -> bool:
    """Is this expansion just echoing the original query?"""
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text: str) -> int:
    """Penalty for words repeated 3+ times (excluding stopwords)."""
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


# =============================================================================
# Scoring
# =============================================================================

def score_expansion_detailed(query: str, expansion: str) -> dict:
    """Score an expansion with full breakdown. Returns dict with all dimensions."""
    text, used_thinking = clean_model_output(expansion.strip())
    deductions = []

    def _fail(reason):
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "think_bonus": 0, "total": 0, "max_possible": 100,
            "percentage": 0.0, "rating": "Failed",
            "deductions": [reason],
            "parsed": parse_expansion(expansion),
            "entities_detected": [],
        }

    # Hard fail: remaining chat template tokens
    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return _fail("CHAT TEMPLATE LEAKAGE")

    # Hard fail: every non-empty line must have a valid prefix
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return _fail(f"INVALID LINE: {line[:50]}")

    parsed = parse_expansion(text)

    # --- Format (0-30) ---
    format_score = 10  # no invalid lines (guaranteed by hard fail)
    if parsed["lex"]:
        format_score += 10
    else:
        deductions.append("missing lex:")
    if parsed["vec"]:
        format_score += 10
    else:
        deductions.append("missing vec:")

    # --- Diversity (0-30) ---
    diversity_score = 0

    types_present = sum(1 for t in ("lex", "vec") if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    else:
        deductions.append("only one type")

    if len(parsed["lex"]) + len(parsed["vec"]) >= 2:
        diversity_score += 5

    lex_div = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2):
                lex_div -= 2
                deductions.append(f"lex duplicate: {a[:20]}...")
    diversity_score += max(0, lex_div)

    vec_div = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3):
                vec_div -= 2
                deductions.append(f"vec duplicate: {a[:20]}...")
    diversity_score += max(0, vec_div)

    echo = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo -= 3
            deductions.append(f"echoes query: {exp[:20]}...")
    diversity_score += max(0, echo)

    # --- HyDE (0-20, optional bonus) ---
    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2
            deductions.append(f"hyde too short ({hyde_len})")
        else:
            deductions.append(f"hyde too long ({hyde_len})")
        if "\n" not in hyde_text:
            hyde_score += 5
        hyde_score += max(0, 5 - word_repetition_penalty(hyde_text))

    # --- Quality (0-20) ---
    quality_score = 5  # base relevance
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
        else:
            deductions.append("lex longer than vec")
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        quality_score += 5 if natural == len(parsed["vec"]) else 2
    if parsed["lex"]:
        with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if with_terms == len(parsed["lex"]):
            quality_score += 5
        elif with_terms > 0:
            quality_score += 2
        else:
            deductions.append("lex missing key terms")

    # --- Entity Preservation (-45 to +20) ---
    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if with_entities == len(parsed["lex"]):
            entity_score += 15
        elif with_entities > 0:
            entity_score += 5
        else:
            entity_score -= 30
            deductions.append(f"lex missing entities: {entities}")

        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count:
            entity_score -= generic_count * 15
            deductions.append(f"{generic_count} generic lex phrases")

        if parsed["vec"]:
            vec_with = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with > 0:
                entity_score += 5
    elif not entities:
        entity_score = 10

    # --- Think bonus (0-20): reward NOT using thinking mode ---
    think_bonus = 0 if used_thinking else 20

    # --- Total ---
    total = format_score + diversity_score + hyde_score + quality_score + entity_score + think_bonus
    max_possible = 140 if parsed["hyde"] else 120
    percentage = max(0.0, min(100.0, total / max_possible * 100))

    if percentage >= 80:
        rating = "Excellent"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Acceptable"
    elif percentage >= 20:
        rating = "Poor"
    else:
        rating = "Failed"

    return {
        "format": format_score,
        "diversity": diversity_score,
        "hyde": hyde_score,
        "quality": quality_score,
        "entity": max(0, entity_score),
        "think_bonus": think_bonus,
        "total": max(0, total),
        "max_possible": max_possible,
        "percentage": round(percentage, 1),
        "rating": rating,
        "deductions": deductions,
        "parsed": parsed,
        "entities_detected": list(entities) if entities else [],
    }


def score_expansion(query: str, expansion: str) -> float:
    """Score expansion as a float in [0.0, 1.0] for use as RL reward."""
    result = score_expansion_detailed(query, expansion)
    return max(0.0, min(1.0, result["total"] / result["max_possible"]))


def extract_query_from_prompt(prompt: str) -> str:
    """Extract the query string from a chat-formatted prompt."""
    if "Expand this search query:" in prompt:
        query = prompt.split("Expand this search query:")[-1].strip()
        if "<|im_end|>" in query:
            query = query.split("<|im_end|>")[0].strip()
        return query
    return prompt.strip()


# =============================================================================
# TRL-compatible reward class
# =============================================================================

class QMDRewardFunction:
    """Reward function compatible with TRL's GRPOTrainer."""
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])
            rewards.append(score_expansion(query, completion))
        return rewards


# =============================================================================
# CLI: run standalone to test the reward function
# =============================================================================

if __name__ == "__main__":
    print("QMD Reward Function Self-Test")
    print("=" * 60)

    tests = [
        ("auth", "lex: auth setup\nlex: authentication config\nvec: how to configure authentication\nhyde: Configure auth by setting AUTH_SECRET."),
        ("auth", "auth is important for security"),
        ("who is TDS motorsports", "lex: TDS motorsports history\nlex: TDS motorsports founders\nvec: information about TDS motorsports company"),
        ("who is TDS motorsports", "lex: find information about\nlex: company details\nvec: who is this company"),
        ("how to use React hooks", "lex: React hooks tutorial\nlex: useEffect useState\nvec: how to use React hooks in functional components"),
        ("auth", "<think>Let me think...</think>\nlex: auth"),
        ("auth", "lex: auth\nThis is some explanation\nvec: more"),
    ]

    for query, expansion in tests:
        score = score_expansion(query, expansion)
        detail = score_expansion_detailed(query, expansion)
        print(f"\n  Query: '{query}'")
        print(f"  Score: {score:.2f} ({detail['rating']})")
        if detail["deductions"]:
            print(f"  Issues: {', '.join(detail['deductions'][:3])}")
