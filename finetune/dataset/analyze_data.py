#!/usr/bin/env python3
"""
Dataset Analysis and Quality Report Generator

Analyzes the training data for:
1. Query length distribution
2. Category diversity
3. Named entity coverage
4. Temporal query coverage
5. Short query coverage (important for ambiguous queries)
6. Duplicate detection
7. Quality issues (long hyde, missing fields, etc.)
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import normalize_output_items, parse_output_text


@dataclass
class DatasetStats:
    total_examples: int = 0
    short_queries: int = 0  # 1-2 words
    medium_queries: int = 0  # 3-5 words
    long_queries: int = 0  # 6+ words
    has_lex: int = 0
    has_vec: int = 0
    has_hyde: int = 0
    long_hyde_count: int = 0
    duplicate_queries: int = 0
    named_entity_queries: int = 0
    temporal_queries: int = 0
    short_keyword_queries: int = 0


def categorize_query(query: str) -> str:
    """Categorize a query by type."""
    query_lower = query.lower()
    words = query_lower.split()
    word_count = len(words)

    # Short keyword queries
    if word_count <= 2:
        return "short_keyword"

    # Named entity queries (capitalized words or tech terms)
    if any(w[0].isupper() for w in words if w):
        return "named_entity"

    # Temporal/recency queries
    temporal_keywords = [
        "latest",
        "recent",
        "new",
        "update",
        "changelog",
        "changed",
        "version",
        "release",
        "news",
        "2024",
        "2025",
    ]
    if any(kw in query_lower for kw in temporal_keywords):
        return "temporal"

    # How-to queries
    if query_lower.startswith("how "):
        return "how_to"

    # What is queries
    if query_lower.startswith("what "):
        return "what_is"

    # Difference/comparison queries
    if any(kw in query_lower for kw in ["difference", "vs", "versus", "compare"]):
        return "comparison"

    # Personal/journal style
    if any(
        kw in query_lower for kw in ["meeting", "notes", "journal", "ideas", "thoughts"]
    ):
        return "personal"

    return "other"


def extract_named_entities(query: str) -> list:
    """Extract potential named entities from query."""
    entities = []
    words = query.split()

    for word in words:
        # Skip stopwords
        if word.lower() in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "to",
            "for",
            "of",
            "in",
            "and",
            "or",
        }:
            continue

        # Capitalized words (potential named entities)
        if word and word[0].isupper() and len(word) > 1:
            entities.append(word)

        # Technology terms with version numbers or special chars
        if any(c in word for c in ".+-0123456789") and len(word) > 1:
            entities.append(word)

    return entities


def analyze_dataset(filepath: Path) -> tuple[DatasetStats, dict, dict]:
    """Analyze the dataset and return statistics."""
    stats = DatasetStats()
    categories = Counter()
    seen_queries = set()
    duplicate_count = 0
    category_examples = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                query = example.get("query", "") or example.get("input", "")
                output = example.get("output", [])
                if isinstance(output, str):
                    output = parse_output_text(output)
                output = normalize_output_items(output)

                stats.total_examples += 1

                # Check for duplicates
                query_lower = query.lower()
                if query_lower in seen_queries:
                    duplicate_count += 1
                else:
                    seen_queries.add(query_lower)

                # Query length categorization
                word_count = len(query.split())
                if word_count <= 2:
                    stats.short_queries += 1
                elif word_count <= 5:
                    stats.medium_queries += 1
                else:
                    stats.long_queries += 1

                # Category detection
                category = categorize_query(query)
                categories[category] += 1
                category_examples[category].append(query)

                # Named entity detection
                if extract_named_entities(query):
                    stats.named_entity_queries += 1

                # Output analysis
                has_lex = any(o[0] == "lex" for o in output)
                has_vec = any(o[0] == "vec" for o in output)
                has_hyde = any(o[0] == "hyde" for o in output)

                if has_lex:
                    stats.has_lex += 1
                if has_vec:
                    stats.has_vec += 1
                if has_hyde:
                    stats.has_hyde += 1

                    # Check hyde length
                    for kind, text in output:
                        if kind == "hyde" and len(text) > 200:
                            stats.long_hyde_count += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")

    stats.duplicate_queries = duplicate_count
    stats.temporal_queries = categories.get("temporal", 0)
    stats.short_keyword_queries = categories.get("short_keyword", 0)

    return stats, dict(categories), dict(category_examples)


def print_report(stats: DatasetStats, categories: dict, category_examples: dict):
    """Print a comprehensive analysis report."""
    print("=" * 70)
    print("QMD TRAINING DATA ANALYSIS REPORT")
    print("=" * 70)
    print()

    # Basic statistics
    print("📊 BASIC STATISTICS")
    print("-" * 40)
    print(f"Total examples:     {stats.total_examples:>6}")
    print(f"Duplicates found:   {stats.duplicate_queries:>6}")
    print()

    # Query length distribution
    print("📝 QUERY LENGTH DISTRIBUTION")
    print("-" * 40)
    total = stats.total_examples
    print(
        f"Short (1-2 words):  {stats.short_queries:>6} ({100 * stats.short_queries / total:5.1f}%)"
    )
    print(
        f"Medium (3-5 words): {stats.medium_queries:>6} ({100 * stats.medium_queries / total:5.1f}%)"
    )
    print(
        f"Long (6+ words):    {stats.long_queries:>6} ({100 * stats.long_queries / total:5.1f}%)"
    )
    print()

    # Category distribution
    print("🏷️  CATEGORY DISTRIBUTION")
    print("-" * 40)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"{cat:20} {count:>6} ({pct:5.1f}%) {bar}")
    print()

    # Output format coverage
    print("✅ OUTPUT FORMAT COVERAGE")
    print("-" * 40)
    print(
        f"Has lex:            {stats.has_lex:>6} ({100 * stats.has_lex / total:5.1f}%)"
    )
    print(
        f"Has vec:            {stats.has_vec:>6} ({100 * stats.has_vec / total:5.1f}%)"
    )
    print(
        f"Has hyde:           {stats.has_hyde:>6} ({100 * stats.has_hyde / total:5.1f}%)"
    )
    print(f"Long hyde (>200ch): {stats.long_hyde_count:>6}")
    print()

    # Critical metrics for evals
    print("🎯 EVALUATION ALIGNMENT")
    print("-" * 40)
    print(
        f"Named entity queries:   {stats.named_entity_queries:>6} ({100 * stats.named_entity_queries / total:5.1f}%)"
    )
    print(
        f"Temporal/recency:       {stats.temporal_queries:>6} ({100 * stats.temporal_queries / total:5.1f}%)"
    )
    print(
        f"Short keyword queries:  {stats.short_keyword_queries:>6} ({100 * stats.short_keyword_queries / total:5.1f}%)"
    )
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 40)

    recommendations = []

    if stats.short_queries / total < 0.15:
        recommendations.append(
            "⚠️  Short queries below 15% - add more 1-2 word keyword queries"
        )

    if stats.named_entity_queries / total < 0.10:
        recommendations.append(
            "⚠️  Named entity queries below 10% - add more capitalized tech term queries"
        )

    if stats.temporal_queries / total < 0.05:
        recommendations.append(
            "⚠️  Temporal queries below 5% - add more 'latest', 'recent' queries"
        )

    if stats.long_hyde_count > 50:
        recommendations.append(
            f"⚠️  {stats.long_hyde_count} long hyde sections - consider truncating"
        )

    if stats.duplicate_queries > 0:
        recommendations.append(
            f"⚠️  {stats.duplicate_queries} duplicate queries - consider deduplication"
        )

    if categories.get("short_keyword", 0) < 100:
        recommendations.append(
            "⚠️  Need more short keyword examples for ambiguous query training"
        )

    if not recommendations:
        print("✅ Dataset looks good! No major issues detected.")
    else:
        for rec in recommendations:
            print(rec)

    print()
    print("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze QMD training dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/qmd_expansion_v2.jsonl",
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
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent
        input_path = script_dir / args.input

    if not input_path.exists():
        print(f"Error: Could not find dataset at {input_path}")
        print("Please run from finetune directory or specify correct path")
        return 1

    print(f"Analyzing: {input_path}")
    print()

    stats, categories, category_examples = analyze_dataset(input_path)
    print_report(stats, categories, category_examples)

    # Show examples if requested
    if args.show_examples > 0:
        print("📋 SAMPLE QUERIES BY CATEGORY")
        print("-" * 40)
        for cat in sorted(categories.keys()):
            examples = category_examples.get(cat, [])
            if examples:
                print(f"\n{cat.upper()}:")
                for ex in examples[: args.show_examples]:
                    print(f"  • {ex}")
        print()

    return 0


if __name__ == "__main__":
    exit(main())
