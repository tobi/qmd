#!/usr/bin/env python3
"""Generate synthetic training data for QMD query expansion using Claude API."""

import argparse
import json
import os
import random
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    exit(1)

# Sample query templates for diverse training data
QUERY_TEMPLATES = [
    # Technical documentation
    "how to {action} {technology}",
    "{technology} {concept} example",
    "configure {technology} for {use_case}",
    "{error_type} error in {technology}",
    "best practices for {concept}",

    # Personal notes / journals
    "meeting notes {topic}",
    "ideas for {project}",
    "{date} journal entry",
    "thoughts on {topic}",

    # Research / learning
    "what is {concept}",
    "difference between {thing1} and {thing2}",
    "{topic} tutorial",
    "learn {skill}",

    # Short queries
    "{keyword}",
    "{keyword} {modifier}",
]

ACTIONS = ["install", "configure", "setup", "debug", "deploy", "test", "optimize", "migrate"]
TECHNOLOGIES = ["python", "typescript", "react", "docker", "kubernetes", "postgres", "redis", "nginx", "git", "linux"]
CONCEPTS = ["authentication", "caching", "logging", "testing", "deployment", "API", "database", "security"]
USE_CASES = ["production", "development", "CI/CD", "local", "cloud"]
ERROR_TYPES = ["connection", "timeout", "permission", "memory", "syntax"]
TOPICS = ["productivity", "workflow", "architecture", "design", "performance"]
KEYWORDS = ["auth", "config", "setup", "api", "data", "cache", "log", "test"]
MODIFIERS = ["best", "fast", "simple", "advanced", "secure"]

SYSTEM_PROMPT = """You are a search query optimization expert for a markdown document search system called QMD.

Your task is to transform user queries into retrieval-optimized outputs with THREE distinct types:

1. **lex** lines: Keyword variations optimized for BM25 full-text search
   - Short, keyword-focused
   - Good for exact term matching
   - 1-3 lines

2. **vec** lines: Semantic reformulations for vector/embedding search
   - Complete phrases or questions
   - Capture semantic meaning
   - 1-3 lines

3. **hyde** line: A hypothetical document passage (HyDE technique)
   - A realistic passage that would answer the query
   - Contains domain-specific terminology
   - Written as if it's FROM a document, not ABOUT the query
   - MAX 1 line

Output format (STRICT - follow exactly):
```
lex: keyword1
lex: keyword2
vec: semantic query reformulation
hyde: A passage that would appear in a document answering this query.
```

Rules:
- Each line must start with "lex:", "vec:", or "hyde:"
- No blank lines
- No repetition between lines
- hyde should be a realistic document excerpt, not a question
- Stay focused on the original query intent"""

USER_PROMPT_TEMPLATE = """Generate query expansion outputs for this search query:

Query: {query}

Respond with ONLY the lex/vec/hyde lines, nothing else."""


def generate_random_query() -> str:
    """Generate a random query from templates."""
    template = random.choice(QUERY_TEMPLATES)

    replacements = {
        "{action}": random.choice(ACTIONS),
        "{technology}": random.choice(TECHNOLOGIES),
        "{concept}": random.choice(CONCEPTS),
        "{use_case}": random.choice(USE_CASES),
        "{error_type}": random.choice(ERROR_TYPES),
        "{topic}": random.choice(TOPICS),
        "{project}": random.choice(["website", "app", "CLI tool", "API", "library"]),
        "{date}": random.choice(["2024-01", "2024-06", "yesterday", "today"]),
        "{thing1}": random.choice(CONCEPTS[:4]),
        "{thing2}": random.choice(CONCEPTS[4:]),
        "{skill}": random.choice(TECHNOLOGIES),
        "{keyword}": random.choice(KEYWORDS),
        "{modifier}": random.choice(MODIFIERS),
    }

    query = template
    for key, value in replacements.items():
        query = query.replace(key, value)

    return query


def generate_expansion(client: anthropic.Anthropic, query: str) -> str | None:
    """Generate expansion using Claude API."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error generating expansion for '{query}': {e}")
        return None


def validate_output(output: str) -> bool:
    """Validate that output follows the expected format."""
    lines = output.strip().split("\n")
    if not lines:
        return False

    has_lex = False
    has_vec = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            has_lex = True
        elif line.startswith("vec:"):
            has_vec = True
        elif line.startswith("hyde:"):
            pass
        else:
            return False  # Invalid line type

    return has_lex and has_vec


def main():
    parser = argparse.ArgumentParser(description="Generate QMD query expansion training data")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/qmd_expansion.jsonl", help="Output file path")
    parser.add_argument("--queries", type=str, help="Optional file with custom queries (one per line)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load custom queries if provided
    custom_queries = []
    if args.queries and Path(args.queries).exists():
        custom_queries = Path(args.queries).read_text().strip().split("\n")
        print(f"Loaded {len(custom_queries)} custom queries")

    examples = []
    seen_queries = set()

    print(f"Generating {args.count} examples...")

    i = 0
    while len(examples) < args.count:
        # Use custom query or generate random one
        if custom_queries and i < len(custom_queries):
            query = custom_queries[i].strip()
        else:
            query = generate_random_query()

        i += 1

        # Skip duplicates
        if query in seen_queries:
            continue
        seen_queries.add(query)

        # Generate expansion
        output = generate_expansion(client, query)
        if output and validate_output(output):
            examples.append({"input": query, "output": output})
            print(f"[{len(examples)}/{args.count}] {query[:50]}...")
        else:
            print(f"  Skipped invalid output for: {query[:50]}...")

    # Write output
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nGenerated {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
