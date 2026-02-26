#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic>=2.0"]
# ///
"""
Dataset Analysis and Quality Report Generator

Analyzes training data loaded through the strict Pydantic schema for:
1. Query length distribution
2. Category diversity
3. Named entity coverage
4. Output format coverage
5. Duplicate detection
"""

import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import TrainingExample, OutputType, load_examples


@dataclass
class DatasetStats:
    total_examples: int = 0
    short_queries: int = 0
    medium_queries: int = 0
    long_queries: int = 0
    has_lex: int = 0
    has_vec: int = 0
    has_hyde: int = 0
    long_hyde_count: int = 0
    duplicate_queries: int = 0
    named_entity_queries: int = 0
    temporal_queries: int = 0
    short_keyword_queries: int = 0


def categorize_query(query: str) -> str:
    query_lower = query.lower()
    words = query_lower.split()
    word_count = len(words)

    if word_count <= 2:
        return "short_keyword"
    if any(w[0].isupper() for w in query.split() if w):
        return "named_entity"

    temporal_keywords = [
        "latest", "recent", "new", "update", "changelog",
        "changed", "version", "release", "news", "2024", "2025",
    ]
    if any(kw in query_lower for kw in temporal_keywords):
        return "temporal"
    if query_lower.startswith("how "):
        return "how_to"
    if query_lower.startswith("what "):
        return "what_is"
    if any(kw in query_lower for kw in ["difference", "vs", "versus", "compare"]):
        return "comparison"
    if any(kw in query_lower for kw in ["meeting", "notes", "journal", "ideas", "thoughts"]):
        return "personal"

    return "other"


def extract_named_entities(query: str) -> list:
    entities = []
    stopwords = {"the", "a", "an", "is", "are", "to", "for", "of", "in", "and", "or"}
    for word in query.split():
        if word.lower() in stopwords:
            continue
        if word and word[0].isupper() and len(word) > 1:
            entities.append(word)
        if any(c in word for c in ".+-0123456789") and len(word) > 1:
            entities.append(word)
    return entities


def analyze_examples(examples: list[TrainingExample]) -> tuple[DatasetStats, dict, dict]:
    stats = DatasetStats()
    categories: Counter = Counter()
    seen_queries: set[str] = set()
    category_examples: dict[str, list[str]] = defaultdict(list)

    for ex in examples:
        stats.total_examples += 1

        query_lower = ex.query.lower()
        if query_lower in seen_queries:
            stats.duplicate_queries += 1
        else:
            seen_queries.add(query_lower)

        word_count = len(ex.query.split())
        if word_count <= 2:
            stats.short_queries += 1
        elif word_count <= 5:
            stats.medium_queries += 1
        else:
            stats.long_queries += 1

        category = categorize_query(ex.query)
        categories[category] += 1
        category_examples[category].append(ex.query)

        if extract_named_entities(ex.query):
            stats.named_entity_queries += 1

        # Use the typed OutputPair model
        types_present = {p.type for p in ex.output}
        if OutputType.lex in types_present:
            stats.has_lex += 1
        if OutputType.vec in types_present:
            stats.has_vec += 1
        if OutputType.hyde in types_present:
            stats.has_hyde += 1
            for p in ex.output:
                if p.type == OutputType.hyde and len(p.text) > 200:
                    stats.long_hyde_count += 1

    stats.temporal_queries = categories.get("temporal", 0)
    stats.short_keyword_queries = categories.get("short_keyword", 0)
    return stats, dict(categories), dict(category_examples)


def print_report(stats: DatasetStats, categories: dict, category_examples: dict):
    print("=" * 70)
    print("QMD TRAINING DATA ANALYSIS REPORT")
    print("=" * 70)
    print()

    total = stats.total_examples
    print("BASIC STATISTICS")
    print("-" * 40)
    print(f"Total examples:     {total:>6}")
    print(f"Duplicates found:   {stats.duplicate_queries:>6}")
    print()

    print("QUERY LENGTH DISTRIBUTION")
    print("-" * 40)
    print(f"Short (1-2 words):  {stats.short_queries:>6} ({100 * stats.short_queries / total:5.1f}%)")
    print(f"Medium (3-5 words): {stats.medium_queries:>6} ({100 * stats.medium_queries / total:5.1f}%)")
    print(f"Long (6+ words):    {stats.long_queries:>6} ({100 * stats.long_queries / total:5.1f}%)")
    print()

    print("CATEGORY DISTRIBUTION")
    print("-" * 40)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"{cat:20} {count:>6} ({pct:5.1f}%) {bar}")
    print()

    print("OUTPUT FORMAT COVERAGE")
    print("-" * 40)
    print(f"Has lex:            {stats.has_lex:>6} ({100 * stats.has_lex / total:5.1f}%)")
    print(f"Has vec:            {stats.has_vec:>6} ({100 * stats.has_vec / total:5.1f}%)")
    print(f"Has hyde:           {stats.has_hyde:>6} ({100 * stats.has_hyde / total:5.1f}%)")
    print(f"Long hyde (>200ch): {stats.long_hyde_count:>6}")
    print()

    print("EVALUATION ALIGNMENT")
    print("-" * 40)
    print(f"Named entity queries:   {stats.named_entity_queries:>6} ({100 * stats.named_entity_queries / total:5.1f}%)")
    print(f"Temporal/recency:       {stats.temporal_queries:>6} ({100 * stats.temporal_queries / total:5.1f}%)")
    print(f"Short keyword queries:  {stats.short_keyword_queries:>6} ({100 * stats.short_keyword_queries / total:5.1f}%)")
    print()

    print("RECOMMENDATIONS")
    print("-" * 40)
    recommendations = []
    if stats.short_queries / total < 0.15:
        recommendations.append("Short queries below 15% - add more 1-2 word keyword queries")
    if stats.named_entity_queries / total < 0.10:
        recommendations.append("Named entity queries below 10% - add more capitalized tech term queries")
    if stats.temporal_queries / total < 0.05:
        recommendations.append("Temporal queries below 5% - add more 'latest', 'recent' queries")
    if stats.long_hyde_count > 50:
        recommendations.append(f"{stats.long_hyde_count} long hyde sections - consider truncating")
    if stats.duplicate_queries > 0:
        recommendations.append(f"{stats.duplicate_queries} duplicate queries - consider deduplication")
    if not recommendations:
        print("Dataset looks good! No major issues detected.")
    else:
        for rec in recommendations:
            print(f"  - {rec}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze QMD training dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/qmd_expansion_v3_structured.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of example queries to show per category",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        script_dir = Path(__file__).parent.parent
        input_path = script_dir / args.input

    if not input_path.exists():
        print(f"Error: Could not find dataset at {input_path}")
        return 1

    print(f"Analyzing: {input_path}")
    print()

    examples = load_examples(input_path)
    stats, categories, category_examples = analyze_examples(examples)
    print_report(stats, categories, category_examples)

    if args.show_examples > 0:
        print("SAMPLE QUERIES BY CATEGORY")
        print("-" * 40)
        for cat in sorted(categories.keys()):
            exs = category_examples.get(cat, [])
            if exs:
                print(f"\n{cat.upper()}:")
                for ex in exs[:args.show_examples]:
                    print(f"  - {ex}")
        print()

    return 0


if __name__ == "__main__":
    exit(main())
