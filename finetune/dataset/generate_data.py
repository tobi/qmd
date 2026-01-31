#!/usr/bin/env python3
"""Generate synthetic training data for QMD query expansion using Claude API."""

import argparse
import json
import os
import random
from pathlib import Path

from dataset.schema import normalize_output_items, parse_output_text

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    exit(1)

# Sample query templates for diverse training data - organized by category
QUERY_TEMPLATES = [
    # === Technical documentation (35% of queries) ===
    "how to {action} {technology}",
    "{technology} {concept} example",
    "configure {technology} for {use_case}",
    "{error_type} error in {technology}",
    "best practices for {concept}",
    "{technology} vs {technology2}",
    "{action} {technology} {use_case}",
    "setup {technology} {use_case}",
    "{technology} tutorial for beginners",
    "{technology} documentation",
    "{technology} {error_type} troubleshooting",
    "{concept} in {technology}",
    "migrate from {technology} to {technology2}",
    "{action} {concept} {technology}",
    # === Personal notes / journals (15% of queries) ===
    "meeting notes {topic}",
    "ideas for {project}",
    "{date} journal entry",
    "thoughts on {topic}",
    "{project} {topic} notes",
    "{topic} meeting {date}",
    "reflect on {topic}",
    "brainstorm {project}",
    # === Research / learning (20% of queries) ===
    "what is {concept}",
    "difference between {thing1} and {thing2}",
    "{topic} tutorial",
    "learn {skill}",
    "understand {concept}",
    "explain {concept}",
    "{topic} fundamentals",
    "intro to {skill}",
    "{thing1} or {thing2}",
    "when to use {concept}",
    # === Short / keyword queries (15% of queries) ===
    "{keyword}",
    "{keyword} {modifier}",
    "{keyword} {action}",
    "{keyword} {use_case}",
    "{technology} {keyword}",
    "{concept} {keyword}",
    # === Temporal / recency queries (10% of queries) ===
    "latest {topic}",
    "recent {concept} changes",
    "new {technology} features",
    "{topic} update {date}",
    "what changed in {technology}",
    "{technology} changelog {date}",
    "{topic} news {date}",
    # === Named entities / specific topics (5% of queries) ===
    "{named_entity} {topic}",
    "{person} {concept}",
    "{organization} {use_case}",
    "{product} {action}",
]

# Category weights for balanced sampling
TEMPLATE_CATEGORIES = {
    "technical": list(range(0, 14)),  # 0-13
    "personal": list(range(14, 22)),  # 14-21
    "research": list(range(22, 31)),  # 22-30
    "short": list(range(31, 36)),  # 31-35
    "temporal": list(range(36, 42)),  # 36-41
    "entities": list(range(42, 46)),  # 42-45
}

ACTIONS = [
    "install",
    "configure",
    "setup",
    "debug",
    "deploy",
    "test",
    "optimize",
    "migrate",
    "build",
    "run",
    "lint",
    "format",
    "backup",
    "restore",
    "update",
    "rollback",
    "monitor",
    "scale",
    "secure",
    "integrate",
    "automate",
    "refactor",
    "initialize",
]

TECHNOLOGIES = [
    # Languages
    "python",
    "typescript",
    "javascript",
    "rust",
    "golang",
    "java",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "cpp",
    "c",
    "elixir",
    "scala",
    "clojure",
    "dart",
    # Frameworks/Frontend
    "react",
    "vue",
    "angular",
    "svelte",
    "solid",
    "htmx",
    "alpine",
    "nextjs",
    "nuxt",
    # Backend
    "django",
    "flask",
    "fastapi",
    "express",
    "rails",
    "spring",
    "laravel",
    # Infrastructure
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "github-actions",
    # Databases
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "sqlite",
    "dynamodb",
    "cassandra",
    "cockroachdb",
    "supabase",
    "firebase",
    # Tools
    "git",
    "nginx",
    "apache",
    "linux",
    "aws",
    "gcp",
    "azure",
    "vercel",
    "netlify",
    # Data/ML
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "jupyter",
    "spark",
    "kafka",
    "airflow",
    "dbt",
]

TECHNOLOGIES_2 = [
    "docker",
    "kubernetes",
    "postgres",
    "mysql",
    "redis",
    "mongodb",
    "aws",
    "gcp",
    "react",
    "vue",
    "angular",
    "python",
    "javascript",
    "typescript",
    "github-actions",
    "gitlab-ci",
    "jenkins",
    "terraform",
    "ansible",
]

CONCEPTS = [
    "authentication",
    "caching",
    "logging",
    "testing",
    "deployment",
    "API",
    "database",
    "security",
    "monitoring",
    "performance",
    "scalability",
    "reliability",
    "observability",
    "microservices",
    "serverless",
    "virtualization",
    "containerization",
    "orchestration",
    "CI/CD",
    "version control",
    "dependency injection",
    "event sourcing",
    "CQRS",
    "load balancing",
    "rate limiting",
    "circuit breaker",
    "retry logic",
    "idempotency",
]

USE_CASES = [
    "production",
    "development",
    "CI/CD",
    "local",
    "cloud",
    "staging",
    "testing",
    "microservices",
    "serverless",
    "hybrid",
    "multi-tenant",
    "high-availability",
    "real-time",
    "batch processing",
    "stream processing",
    "data pipeline",
]

ERROR_TYPES = [
    "connection",
    "timeout",
    "permission",
    "memory",
    "syntax",
    "runtime",
    "configuration",
    "dependency",
    "network",
    "authentication",
    "authorization",
    "validation",
    "concurrency",
    "deadlock",
    "resource",
    "quota",
]

TOPICS = [
    "productivity",
    "workflow",
    "architecture",
    "design",
    "performance",
    "security",
    "scalability",
    "reliability",
    "observability",
    "maintainability",
    "testing",
    "documentation",
    "refactoring",
    "debugging",
    "optimization",
    "best practices",
    "patterns",
    "anti-patterns",
    "trade-offs",
    "decision making",
]

KEYWORDS = [
    "auth",
    "config",
    "setup",
    "api",
    "cache",
    "log",
    "test",
    "debug",
    "env",
    "vars",
    "secrets",
    "tokens",
    "headers",
    "params",
    "query",
    "body",
    "route",
    "middleware",
    "handler",
    "controller",
    "model",
    "view",
    "template",
    "migration",
    "seed",
    "fixture",
    "mock",
    "stub",
    "spy",
    "fake",
    "build",
    "bundle",
    "compile",
    "transpile",
    "minify",
    "optimize",
    "deploy",
    "release",
    "rollback",
    "promote",
    "freeze",
    "thaw",
    "pull",
    "push",
    "commit",
    "merge",
    "rebase",
    "cherry-pick",
    "stash",
    "up",
    "down",
    "scale",
    "restart",
    "reload",
    "refresh",
    "flush",
    "cron",
    "queue",
    "job",
    "worker",
    "scheduler",
    "trigger",
    "webhook",
    "alert",
    "metric",
    "trace",
    "span",
    "event",
    "incident",
    "oncall",
]

MODIFIERS = [
    "best",
    "fast",
    "simple",
    "advanced",
    "secure",
    "quick",
    "easy",
    "proper",
    "correct",
    "safe",
    "efficient",
    "reliable",
    "robust",
    "latest",
    "recent",
    "new",
    "old",
    "legacy",
    "modern",
    "local",
    "remote",
    "global",
    "shared",
    "private",
    "public",
]

NAMED_ENTITIES = [
    "React",
    "Vue",
    "Angular",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "GitHub",
    "GitLab",
    "Vercel",
    "Netlify",
    "Supabase",
    "Firebase",
    "Stripe",
    "Twilio",
    "SendGrid",
    "Datadog",
    "PagerDuty",
    "Sentry",
    "Terraform",
    "Ansible",
    "Jenkins",
    "CircleCI",
    "TravisCI",
]

PERSONS = [
    "Kent Beck",
    "Martin Fowler",
    "Robert Martin",
    "Dave Thomas",
    "Guido van Rossum",
    "Brendan Eich",
    "Ryan Dahl",
    "Anders Hejlsberg",
    "Linus Torvalds",
    "DHH",
    "Yukihiro Matsumoto",
    "Rich Hickey",
]

ORGANIZATIONS = [
    "Google",
    "Microsoft",
    "Amazon",
    "Meta",
    "Apple",
    "Netflix",
    "Spotify",
    "Stripe",
    "Shopify",
    "Airbnb",
    "Uber",
    "Lyft",
    "Slack",
    "Discord",
]

PRODUCTS = [
    "VS Code",
    "IntelliJ",
    "PyCharm",
    "WebStorm",
    "DataGrip",
    "Postman",
    "Insomnia",
    "TablePlus",
    "Docker Desktop",
    "Lens",
    "Figma",
    "Sketch",
    "Notion",
    "Linear",
    "Jira",
    "Trello",
]

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
hyde: A passage that would appear in a document answering this query.
lex: keyword1
lex: keyword2
vec: semantic query reformulation
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


# Category weights - BALANCED approach
# Tech at 15% (reasonable for QMD's technical document use case)
CATEGORY_WEIGHTS = {
    "technical": 0.15,  # 15% - Technical documentation
    "personal": 0.10,  # 10% - Personal notes, journals
    "research": 0.10,  # 10% - Research and learning
    "short": 0.15,  # 15% - Short keyword queries
    "temporal": 0.10,  # 10% - Temporal/recency queries (2025/2026)
    "entities": 0.05,  # 5% - Named entity queries
    "health": 0.10,  # 10% - Health & wellness
    "finance": 0.10,  # 10% - Finance & business
    "lifestyle": 0.10,  # 10% - Home, food, hobbies, travel
    "education": 0.05,  # 5% - Education & arts
}


def generate_random_query() -> str:
    """Generate a random query from templates with category-weighted sampling."""
    # Select category based on weights
    categories = list(CATEGORY_WEIGHTS.keys())
    weights = list(CATEGORY_WEIGHTS.values())
    selected_category = random.choices(categories, weights=weights, k=1)[0]

    # Select template from that category
    template_idx = random.choice(TEMPLATE_CATEGORIES[selected_category])
    template = QUERY_TEMPLATES[template_idx]

    # Build replacements based on template type
    replacements = {
        "{action}": random.choice(ACTIONS),
        "{technology}": random.choice(TECHNOLOGIES),
        "{technology2}": random.choice(TECHNOLOGIES_2),
        "{concept}": random.choice(CONCEPTS),
        "{use_case}": random.choice(USE_CASES),
        "{error_type}": random.choice(ERROR_TYPES),
        "{topic}": random.choice(TOPICS),
        "{project}": random.choice(
            ["website", "app", "CLI tool", "API", "library", "service", "platform"]
        ),
        "{date}": random.choice(
            # Emphasize 2025/2026 for recency queries (current era)
            [
                "2026",
                "2026",
                "2025",
                "2025",
                "January 2026",
                "February 2026",
                "March 2026",
                "last month",
                "this week",
                "yesterday",
                "today",
                "recently",
                "latest",
            ]
        ),
        "{thing1}": random.choice(CONCEPTS[:10]),
        "{thing2}": random.choice(CONCEPTS[10:] if len(CONCEPTS) > 10 else CONCEPTS),
        "{skill}": random.choice(TECHNOLOGIES),
        "{keyword}": random.choice(KEYWORDS),
        "{modifier}": random.choice(MODIFIERS),
        "{named_entity}": random.choice(NAMED_ENTITIES),
        "{person}": random.choice(PERSONS),
        "{organization}": random.choice(ORGANIZATIONS),
        "{product}": random.choice(PRODUCTS),
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
            ],
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
    parser = argparse.ArgumentParser(
        description="Generate QMD query expansion training data"
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of examples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qmd_expansion.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--queries", type=str, help="Optional file with custom queries (one per line)"
    )
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
            output_items = normalize_output_items(parse_output_text(output))
            examples.append({"query": query, "output": output_items})
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
