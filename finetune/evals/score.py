# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Score query expansion results from a JSONL file.

Usage:
    uv run evals/score.py evals/results_model.jsonl
    uv run evals/score.py evals/results_model.jsonl --output scores.json
    uv run evals/score.py evals/results_model.jsonl --verbose
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ============== SCORING CONSTANTS ==============
STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}
KEY_TERM_STOPWORDS = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
                      'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
                      'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell'}
GENERIC_LEX_PHRASES = {
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with'
}


# ============== HELPER FUNCTIONS ==============
def extract_named_entities(query: str) -> set:
    """Extract named entities from query using simple heuristics."""
    entities = set()
    words = query.split()
    prev_was_entity = False

    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue

        is_entity = False

        # All-caps words (acronyms): TDS, API, GPU
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        # Capitalized words (not first word)
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True
        # Technical terms: node.js, C++
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        # CamelCase: JavaScript
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_entity = True
        # Word following an entity (compound names)
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        prev_was_entity = is_entity

    return entities


def get_key_terms(query: str) -> set:
    """Get key terms (non-stopwords) from query."""
    words = set(query.lower().split())
    return words - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    """Check if lex line preserves key terms from query."""
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    lex_words = set(lex_line.lower().split())
    return bool(key_terms & lex_words)


def lex_preserves_entities(lex_line: str, entities: set) -> bool:
    """Check if lex line contains at least one named entity."""
    if not entities:
        return True
    lex_lower = lex_line.lower()
    return any(entity in lex_lower for entity in entities)


def lex_is_generic(lex_line: str) -> bool:
    """Check if lex line is a generic filler phrase."""
    lex_lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lex_lower or lex_lower.startswith(phrase.split()[0]):
            remaining = lex_lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


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
    return len(words_a ^ words_b)


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
    if exp == q:
        return True
    if q in exp and len(exp) < len(q) + 10:
        return True
    return False


def word_repetition_penalty(text: str) -> int:
    """Count penalty for repeated words."""
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    penalty = 0
    for word, count in counts.items():
        if count >= 3 and word not in STOPWORDS and len(word) > 2:
            penalty += (count - 2) * 2
    return penalty


# ============== MAIN SCORING FUNCTION ==============
def score_expansion(query: str, expansion: str) -> dict:
    """Score an expansion. Returns detailed breakdown."""
    text = expansion.strip()
    deductions = []

    # HARD FAIL: Chat template artifacts
    if any(token in text for token in ['<|im_start|>', '<|im_end|>', '<think>', '</think>',
                                        '\nassistant\n', '\nuser\n', '<|endoftext|>']):
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "total": 0, "max_possible": 100, "percentage": 0, "rating": "Failed",
            "deductions": ["CHAT TEMPLATE LEAKAGE"],
            "parsed": {"lex": [], "vec": [], "hyde": [], "invalid": [text[:100]]},
            "entities_detected": [],
        }

    # HARD FAIL: Every line must start with lex:, vec:, or hyde:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith(("lex:", "vec:", "hyde:")):
            return {
                "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
                "total": 0, "max_possible": 100, "percentage": 0, "rating": "Failed",
                "deductions": [f"INVALID LINE: {line[:50]}"],
                "parsed": parse_expansion(expansion),
                "entities_detected": [],
            }

    parsed = parse_expansion(expansion)

    # FORMAT (0-30)
    format_score = 0
    if parsed["lex"]:
        format_score += 10
    else:
        deductions.append("missing lex:")
    if parsed["vec"]:
        format_score += 10
    else:
        deductions.append("missing vec:")
    format_score += 10  # No invalid lines (guaranteed by hard fail)

    # DIVERSITY (0-30)
    diversity_score = 0
    types_present = sum(1 for t in ["lex", "vec"] if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    else:
        deductions.append("only one type")

    total_expansions = len(parsed["lex"]) + len(parsed["vec"])
    if total_expansions >= 2:
        diversity_score += 5

    lex_score = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2):
                lex_score -= 2
                deductions.append(f"lex duplicate: {a[:20]}...")
    diversity_score += max(0, lex_score)

    vec_score = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3):
                vec_score -= 2
                deductions.append(f"vec duplicate: {a[:20]}...")
    diversity_score += max(0, vec_score)

    echo_score = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo_score -= 3
            deductions.append(f"echoes query: {exp[:20]}...")
    diversity_score += max(0, echo_score)

    # HYDE (0-20)
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
        rep_penalty = word_repetition_penalty(hyde_text)
        hyde_score += max(0, 5 - rep_penalty)

    # QUALITY (0-20)
    quality_score = 5
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
        else:
            deductions.append("lex longer than vec")
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        if natural == len(parsed["vec"]):
            quality_score += 5
        else:
            quality_score += 2
    if parsed["lex"]:
        lex_with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if lex_with_terms == len(parsed["lex"]):
            quality_score += 5
        elif lex_with_terms > 0:
            quality_score += 2
        else:
            deductions.append("lex missing key terms")

    # NAMED ENTITY PRESERVATION (0-20, can go negative)
    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        lex_with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if lex_with_entities == len(parsed["lex"]):
            entity_score += 15
        elif lex_with_entities > 0:
            entity_score += 5
        else:
            entity_score -= 30
            deductions.append(f"lex missing entities: {entities}")

        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count > 0:
            entity_score -= generic_count * 15
            deductions.append(f"{generic_count} generic lex phrases")

        if parsed["vec"]:
            vec_with_entities = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with_entities > 0:
                entity_score += 5
    elif not entities:
        entity_score = 10

    # TOTAL
    total = format_score + diversity_score + hyde_score + quality_score + entity_score
    max_possible = 120 if parsed["hyde"] else 100
    percentage = max(0.0, min(100.0, total / max_possible * 100))

    # Rating
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
        "total": max(0, total),
        "max_possible": max_possible,
        "percentage": round(percentage, 1),
        "rating": rating,
        "deductions": deductions,
        "parsed": parsed,
        "entities_detected": list(entities) if entities else [],
    }


def print_result(query: str, expansion: str, scores: dict, verbose: bool = False):
    """Print a single result."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'─'*60}")
    print(expansion)
    print(f"{'─'*60}")
    print(f"Score: {scores['percentage']:.0f}% ({scores['rating']})")
    print(f"  Format: {scores['format']}/30  Diversity: {scores['diversity']}/30  "
          f"Hyde: {scores['hyde']}/20  Quality: {scores['quality']}/20  Entity: {scores['entity']}/20")

    if verbose and scores["deductions"]:
        print(f"  Deductions: {', '.join(scores['deductions'][:5])}")
    if verbose and scores["entities_detected"]:
        print(f"  Entities: {scores['entities_detected']}")


def main():
    parser = argparse.ArgumentParser(description="Score query expansion results")
    parser.add_argument("input", help="Input JSONL file from run.py")
    parser.add_argument("--output", help="Output JSON file with scores")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary")
    args = parser.parse_args()

    # Load results
    results = []
    metadata = None
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            if data.get("_meta"):
                metadata = data
            else:
                results.append(data)

    print(f"Scoring {len(results)} results from {args.input}", file=sys.stderr)
    if metadata:
        print(f"Model: {metadata.get('model', 'unknown')}", file=sys.stderr)

    # Score each result
    scored_results = []
    for result in results:
        query = result["query"]
        expansion = result["expansion"]
        scores = score_expansion(query, expansion)

        if not args.summary_only:
            print_result(query, expansion, scores, args.verbose)

        scored_results.append({
            "query": query,
            "expansion": expansion,
            "scores": {k: v for k, v in scores.items() if k not in ["parsed", "deductions", "entities_detected"]},
            "deductions": scores["deductions"],
            "entities_detected": scores["entities_detected"],
        })

    # Summary
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
        bar = "█" * count
        print(f"    {rating:10s}: {count:2d} {bar}")

    # Save output
    if args.output:
        output_data = {
            "metadata": metadata,
            "summary": {
                "total": len(scored_results),
                "average_score": round(avg_score, 1),
                "ratings": dict(ratings),
            },
            "results": scored_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nScores saved to: {args.output}")


if __name__ == "__main__":
    main()
