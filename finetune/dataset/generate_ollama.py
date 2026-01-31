#!/usr/bin/env python3
"""Generate synthetic training data for QMD query expansion using local Ollama."""

import argparse
import json
import random
import sys
import time

from dataset.schema import normalize_output_items, parse_output_text
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    exit(1)

# Diverse query seeds across many domains
QUERY_SEEDS = [
    # Programming & Tech
    "async await javascript",
    "rust ownership borrow checker",
    "kubernetes pod networking",
    "docker compose volumes",
    "nginx reverse proxy",
    "postgresql index optimization",
    "redis caching strategies",
    "graphql mutations",
    "websocket authentication",
    "terraform state management",
    "ansible playbook variables",
    "prometheus alerting rules",
    "elasticsearch aggregations",
    "kafka consumer groups",
    "grpc streaming",
    "oauth2 refresh tokens",
    "jwt token expiration",
    "cors preflight requests",
    "css grid layout",
    "react hooks useEffect",
    "vue composition api",
    "svelte stores",
    "nextjs middleware",
    "webpack code splitting",
    "typescript generics constraints",
    "python asyncio gather",
    "go goroutines channels",
    "java streams filter map",
    "c++ smart pointers",
    "swift optionals unwrapping",
    # DevOps & Infrastructure
    "ci cd pipeline best practices",
    "blue green deployment",
    "canary release strategy",
    "infrastructure as code",
    "secrets management vault",
    "load balancer health checks",
    "ssl certificate renewal",
    "dns propagation time",
    "cdn cache invalidation",
    "container orchestration",
    "service mesh istio",
    "observability tracing",
    "log aggregation elk",
    "metrics dashboards grafana",
    "incident response runbook",
    # Data & ML
    "pandas dataframe groupby",
    "numpy array broadcasting",
    "scikit learn pipeline",
    "pytorch autograd",
    "tensorflow keras layers",
    "huggingface transformers",
    "feature engineering techniques",
    "hyperparameter tuning",
    "model evaluation metrics",
    "data preprocessing normalization",
    "time series forecasting",
    "anomaly detection",
    "recommendation systems",
    "natural language processing",
    "computer vision cnn",
    "reinforcement learning",
    "transfer learning",
    "model deployment mlops",
    # Databases
    "sql join types explained",
    "database normalization forms",
    "acid transactions",
    "database sharding strategies",
    "read replicas setup",
    "connection pooling",
    "query optimization explain",
    "stored procedures triggers",
    "database migrations",
    "nosql document model",
    "graph database queries",
    "vector database similarity",
    # Security
    "xss prevention sanitization",
    "sql injection prepared statements",
    "csrf tokens",
    "content security policy",
    "rate limiting api",
    "input validation patterns",
    "password hashing bcrypt",
    "two factor authentication",
    "penetration testing",
    "security headers http",
    "vulnerability scanning",
    "audit logging",
    # System Administration
    "linux file permissions",
    "systemd service unit",
    "cron job scheduling",
    "ssh key management",
    "firewall rules iptables",
    "process monitoring",
    "disk space management",
    "memory leak debugging",
    "network troubleshooting",
    "backup restore strategies",
    "log rotation configuration",
    "performance profiling",
    # General Knowledge
    "climate change effects",
    "renewable energy sources",
    "electric vehicles",
    "artificial intelligence ethics",
    "blockchain technology",
    "quantum computing basics",
    "space exploration mars",
    "gene editing crispr",
    "vaccine development",
    "economic indicators gdp",
    "stock market investing",
    "cryptocurrency trading",
    "mental health awareness",
    "nutrition diet tips",
    "exercise fitness routine",
    "meditation mindfulness",
    "sleep hygiene habits",
    "stress management",
    "time management productivity",
    "remote work tips",
    "team collaboration",
    "project management agile",
    "design thinking process",
    "user experience research",
    # Short/Ambiguous Queries (important for training)
    "cache",
    "proxy",
    "queue",
    "mutex",
    "semaphore",
    "deadlock",
    "heap",
    "stack",
    "tree",
    "graph",
    "hash",
    "sort",
    "api",
    "sdk",
    "cli",
    "gui",
    "orm",
    "cdn",
    "auth",
    "cors",
    "csrf",
    "xss",
    "jwt",
    "ssh",
]

PROMPT_TEMPLATE = """Generate search query expansions for: {query}

Output EXACTLY this format (3 lex, 2 vec, 1 hyde):
lex: keyword phrase 1
lex: keyword phrase 2
lex: keyword phrase 3
vec: natural language search query
vec: alternative semantic query
hyde: A specific 2-sentence document passage answering this query.

Output:"""


def generate_with_ollama(
    query: str, model: str = "gemma3:4b", base_url: str = "http://localhost:11434"
) -> str | None:
    """Generate query expansion using Ollama API."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT_TEMPLATE.format(query=query),
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 800,  # More tokens for thinking models
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error generating for '{query}': {e}", file=sys.stderr)
        return None


def parse_expansion(output: str) -> list[list[str]] | None:
    """Parse the model output into structured format."""
    items = normalize_output_items(parse_output_text(output))
    lex_count = sum(1 for kind, _ in items if kind == "lex")
    vec_count = sum(1 for kind, _ in items if kind == "vec")
    hyde_count = sum(1 for kind, _ in items if kind == "hyde")
    if lex_count >= 2 and vec_count >= 1 and hyde_count >= 1:
        return items
    return None


def generate_query_variations(seed: str) -> list[str]:
    """Generate variations of a seed query."""
    variations = [seed]

    # Add question forms
    if not seed.startswith(("how", "what", "why", "when", "where")):
        variations.append(f"how to {seed}")
        variations.append(f"what is {seed}")

    # Add context
    variations.append(f"{seed} tutorial")
    variations.append(f"{seed} best practices")
    variations.append(f"{seed} examples")

    return variations


def main():
    parser = argparse.ArgumentParser(description="Generate training data using Ollama")
    parser.add_argument(
        "--output", "-o", default="data/qmd_expansion_ollama.jsonl", help="Output file"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=1000, help="Number of examples to generate"
    )
    parser.add_argument("--model", "-m", default="gemma3:4b", help="Ollama model name")
    parser.add_argument(
        "--base-url", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing file"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if resuming
    existing_queries = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                existing_queries.add(obj.get("query", obj.get("input", "")).lower())
        print(
            f"Resuming with {len(existing_queries)} existing examples", file=sys.stderr
        )

    # Generate query pool
    all_queries = []
    for seed in QUERY_SEEDS:
        all_queries.extend(generate_query_variations(seed))

    # Shuffle and filter
    random.shuffle(all_queries)
    queries_to_process = [q for q in all_queries if q.lower() not in existing_queries]

    print(
        f"Processing {min(args.count, len(queries_to_process))} queries with {args.model}...",
        file=sys.stderr,
    )

    generated = 0
    errors = 0

    mode = "a" if args.resume else "w"
    with open(output_path, mode) as f:
        for i, query in enumerate(queries_to_process):
            if generated >= args.count:
                break

            output = generate_with_ollama(query, args.model, args.base_url)
            if output:
                parsed = parse_expansion(output)
                if parsed:
                    example = {"query": query, "output": parsed}
                    f.write(json.dumps(example) + "\n")
                    f.flush()
                    generated += 1

                    if generated % 10 == 0:
                        print(
                            f"Generated {generated}/{args.count} ({errors} errors)",
                            file=sys.stderr,
                        )
                else:
                    errors += 1
            else:
                errors += 1

            # Small delay to avoid overwhelming the API
            time.sleep(0.1)

    print(f"\nDone! Generated {generated} examples, {errors} errors", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
