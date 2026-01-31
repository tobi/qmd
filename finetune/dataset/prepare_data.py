#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "jinja2",
# ]
# ///
"""Prepare QMD query expansion data for training.

See PROMPT_FORMAT.md for format specification.
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _tokenizer


# Short single-word queries that need proper expansion examples - organized by category
SHORT_QUERIES = [
    # === Programming Languages & Runtimes ===
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
    "r",
    "node",
    "deno",
    "bun",
    # === Frontend Frameworks ===
    "react",
    "vue",
    "angular",
    "svelte",
    "solid",
    "htmx",
    "alpine",
    "nextjs",
    "nuxt",
    "jquery",
    "backbone",
    "ember",
    # === Backend Frameworks ===
    "django",
    "flask",
    "fastapi",
    "express",
    "rails",
    "spring",
    "laravel",
    "gin",
    # === Databases ===
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "sqlite",
    "dynamodb",
    "cassandra",
    "cockroachdb",
    "neo4j",
    "couchdb",
    # === Infrastructure & DevOps ===
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "vagrant",
    "packer",
    "jenkins",
    "gitlab-ci",
    "github-actions",
    "circleci",
    "travis",
    "argo",
    "nginx",
    "apache",
    "caddy",
    "traefik",
    "haproxy",
    "envoy",
    # === Cloud Platforms ===
    "aws",
    "gcp",
    "azure",
    "vercel",
    "netlify",
    "heroku",
    "digitalocean",
    "cloudflare",
    "flyio",
    "render",
    # === Tools & Utilities ===
    "git",
    "linux",
    "bash",
    "zsh",
    "vim",
    "tmux",
    "curl",
    "wget",
    "ssh",
    "npm",
    "pip",
    "brew",
    "apt",
    "yum",
    "cargo",
    "gem",
    "composer",
    "maven",
    # === Security & Auth ===
    "auth",
    "oauth",
    "jwt",
    "saml",
    "ldap",
    "rbac",
    "cors",
    "csrf",
    "xss",
    "ssl",
    "tls",
    "cert",
    "encrypt",
    "hash",
    "cipher",
    # === Web Technologies ===
    "rest",
    "graphql",
    "grpc",
    "websocket",
    "sse",
    "http",
    "https",
    "html",
    "css",
    "sass",
    "less",
    "styled-components",
    "tailwind",
    # === Data & ML ===
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "sklearn",
    "jupyter",
    "spark",
    "kafka",
    "airflow",
    "dbt",
    "hadoop",
    "hive",
    "presto",
    # === Testing ===
    "jest",
    "vitest",
    "pytest",
    "mocha",
    "cypress",
    "playwright",
    "selenium",
    "rspec",
    "junit",
    "testng",
    # === Build Tools ===
    "webpack",
    "vite",
    "esbuild",
    "rollup",
    "parcel",
    "turbopack",
    "babel",
    # === Monitoring & Observability ===
    "prometheus",
    "grafana",
    "datadog",
    "newrelic",
    "sentry",
    "jaeger",
    "logging",
    "metrics",
    "tracing",
    "observability",
    # === API & Integration ===
    "swagger",
    "openapi",
    "postman",
    "api",
    "webhook",
    "sdk",
    "cli",
    # === Architecture Patterns ===
    "microservices",
    "serverless",
    "monolith",
    "event-driven",
    "cqrs",
    "event-sourcing",
    "saga",
    "circuit-breaker",
    "retry",
    "idempotency",
    # === Development Concepts ===
    "config",
    "setup",
    "cache",
    "log",
    "debug",
    "deploy",
    "build",
    "lint",
    "format",
    "migrate",
    "backup",
    "restore",
    "env",
    "vars",
    "secrets",
    "rate-limit",
    "load-balance",
    "scale",
    "replicate",
    "shard",
    # === General Knowledge: Trivia ===
    "trivia",
    "quiz",
    "facts",
    "did-you-know",
    "random-facts",
    "world-records",
    # === General Knowledge: Geography ===
    "countries",
    "capitals",
    "continents",
    "oceans",
    "rivers",
    "mountains",
    "deserts",
    "islands",
    "climate",
    "population",
    "maps",
    "coordinates",
    # === General Knowledge: Philosophy ===
    "ethics",
    "metaphysics",
    "epistemology",
    "logic",
    "stoicism",
    "existentialism",
    "nihilism",
    "utilitarianism",
    "deontology",
    "virtue-ethics",
    "free-will",
    # === General Knowledge: History ===
    "ancient",
    "medieval",
    "renaissance",
    "industrial",
    "world-war",
    "cold-war",
    "revolution",
    "empire",
    "civilization",
    "archaeology",
    "timeline",
    # === General Knowledge: Science ===
    "physics",
    "chemistry",
    "biology",
    "astronomy",
    "geology",
    "ecology",
    "evolution",
    "genetics",
    "quantum",
    "relativity",
    "thermodynamics",
    # === General Knowledge: Arts & Culture ===
    "art",
    "music",
    "literature",
    "film",
    "theater",
    "dance",
    "sculpture",
    "painting",
    "photography",
    "architecture",
    "poetry",
    "novel",
    # === Common Short Phrases ===
    "env vars",
    "api keys",
    "error handling",
    "rate limiting",
    "file upload",
    "user auth",
    "db connection",
    "query params",
    "hot reload",
    "code split",
    "tree shake",
    "lazy load",
    "dependency injection",
    "event listener",
    "middleware chain",
    "route handler",
    "controller logic",
    "service layer",
    "repository pattern",
    "unit of work",
    "domain model",
    "value object",
    "aggregate root",
    "event bus",
    "message queue",
    "job scheduler",
    "web server",
    "app server",
    "proxy server",
    "load balancer",
    "cdn",
    "dns",
    "ssl-cert",
    "firewall",
    "subnet",
    "vpc",
    "gateway",
]

# Templates for generating short query expansions
# IMPORTANT: All lex lines MUST include {q} to preserve key terms
SHORT_TEMPLATES = [
    # Configuration/Setup templates
    {
        "lex": ["{q} configuration", "{q} settings", "{q} setup guide"],
        "vec": [
            "how to configure {q} in my project",
            "{q} setup and configuration tutorial",
        ],
        "hyde": "To set up {q}, first install the required dependencies. Then configure the settings in your project configuration file.",
    },
    # Tutorial/Learning templates
    {
        "lex": ["{q} tutorial", "{q} guide", "{q} basics"],
        "vec": ["beginner guide to {q}", "how to get started with {q}"],
        "hyde": "This guide covers the basics of {q}. Follow the steps below to get started with your first implementation.",
    },
    # Best practices templates
    {
        "lex": ["{q} best practices", "{q} patterns", "{q} tips"],
        "vec": ["best practices for using {q}", "recommended patterns for {q}"],
        "hyde": "When working with {q}, follow these best practices: use consistent naming, handle errors properly, and document your code.",
    },
    # Troubleshooting templates
    {
        "lex": ["{q} troubleshooting", "{q} fix", "{q} errors"],
        "vec": ["how to fix {q} errors", "troubleshooting common {q} problems"],
        "hyde": "If you encounter {q} issues, check your configuration first. Common problems include missing dependencies and incorrect settings.",
    },
    # Examples/Code templates
    {
        "lex": ["{q} examples", "{q} code samples", "{q} usage"],
        "vec": ["code examples for {q}", "practical {q} implementation examples"],
        "hyde": "Here are some practical examples of {q} in action. Each example demonstrates a common use case with working code.",
    },
    # Documentation/Reference templates
    {
        "lex": ["{q} documentation", "{q} reference", "{q} manual"],
        "vec": ["official {q} documentation", "{q} API reference guide"],
        "hyde": "The official documentation for {q} provides comprehensive information about features, configuration options, and usage examples.",
    },
    # Installation templates
    {
        "lex": ["{q} install", "{q} setup", "{q} getting started"],
        "vec": ["how to install {q} on my system", "{q} installation guide"],
        "hyde": "To install {q}, run the appropriate package manager command for your system. Verify the installation by checking the version.",
    },
    # Comparison templates
    {
        "lex": ["{q} comparison", "{q} vs alternatives", "{q} differences"],
        "vec": ["how does {q} compare to alternatives", "{q} pros and cons"],
        "hyde": "When comparing {q} to similar tools, consider factors like performance, ease of use, community support, and ecosystem compatibility.",
    },
    # Performance templates
    {
        "lex": ["{q} performance", "{q} optimization", "{q} speed"],
        "vec": ["how to optimize {q} performance", "{q} performance tuning tips"],
        "hyde": "To improve {q} performance, profile your application to identify bottlenecks. Common optimizations include caching, lazy loading, and query optimization.",
    },
    # Security templates
    {
        "lex": ["{q} security", "{q} hardening", "{q} vulnerabilities"],
        "vec": ["how to secure {q} configuration", "{q} security best practices"],
        "hyde": "Security considerations for {q} include input validation, authentication, authorization, and keeping dependencies up to date with security patches.",
    },
    # Testing templates
    {
        "lex": ["{q} testing", "{q} test suite", "{q} unit tests"],
        "vec": ["how to test {q} code", "{q} testing strategies and frameworks"],
        "hyde": "Testing {q} involves writing unit tests, integration tests, and end-to-end tests. Use appropriate testing frameworks for your language and platform.",
    },
    # Deployment templates
    {
        "lex": ["{q} deployment", "{q} production", "{q} release"],
        "vec": ["how to deploy {q} to production", "{q} production deployment guide"],
        "hyde": "Deploying {q} to production requires proper configuration, environment variables, monitoring, and rollback procedures for reliability.",
    },
    # Debugging templates
    {
        "lex": ["{q} debugging", "{q} troubleshooting", "{q} error handling"],
        "vec": ["how to debug {q} issues", "{q} debugging techniques and tools"],
        "hyde": "Debugging {q} involves using logging, breakpoints, stack traces, and specialized debugging tools to identify and fix issues efficiently.",
    },
    # Integration templates
    {
        "lex": ["{q} integration", "{q} connect", "{q} interoperability"],
        "vec": ["how to integrate {q} with other systems", "{q} integration patterns"],
        "hyde": "Integrating {q} with other systems requires understanding APIs, data formats, authentication mechanisms, and error handling strategies.",
    },
    # Migration templates
    {
        "lex": ["{q} migration", "{q} upgrade", "{q} versioning"],
        "vec": ["how to migrate to {q}", "{q} upgrade guide and breaking changes"],
        "hyde": "Migrating to {q} involves planning, testing compatibility, addressing breaking changes, and validating functionality before production deployment.",
    },
]


def truncate_hyde(hyde_text: str, max_len: int = 150) -> str:
    """Truncate hyde to max length, ending at sentence boundary."""
    if len(hyde_text) <= max_len:
        return hyde_text

    truncated = hyde_text[:max_len]
    last_period = truncated.rfind(". ")
    if last_period > max_len // 2:
        return truncated[: last_period + 1]

    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space] + "."

    return truncated[: max_len - 1] + "."


def clean_output(output: str) -> str:
    """Clean output: truncate hyde, remove invalid lines."""
    lines = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("hyde:"):
            hyde_text = line[5:].strip()
            hyde_text = truncate_hyde(hyde_text)
            lines.append(f"hyde: {hyde_text}")
        elif line.startswith(("lex:", "vec:")):
            lines.append(line)
    return "\n".join(lines)


def generate_short_example(query: str) -> dict:
    """Generate a training example for a short query."""
    template = random.choice(SHORT_TEMPLATES)

    lex_lines = random.sample(template["lex"], 2)
    vec_lines = random.sample(template["vec"], 2)
    hyde_line = template["hyde"]

    output_lines = []
    for lex in lex_lines:
        output_lines.append(f"lex: {lex.format(q=query)}")
    for vec in vec_lines:
        output_lines.append(f"vec: {vec.format(q=query)}")
    output_lines.append(f"hyde: {hyde_line.format(q=query)}")

    return {"input": query, "output": "\n".join(output_lines)}


def format_for_training(input_text: str, output_text: str) -> dict:
    """Format a single example for SFT training using Qwen chat format."""
    tokenizer = get_tokenizer()

    # Use /no_think to disable thinking mode - we want direct output
    messages = [
        {
            "role": "user",
            "content": f"/no_think Expand this search query: {input_text}",
        },
        {"role": "assistant", "content": output_text},
    ]

    # Use tokenizer to generate proper chat format with special tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Strip empty   tags - we don't want thinking mode
    # The template adds " \n\n\u5df4\u6bd4\n\n" which we remove
    text = text.replace(" \n\n\u5df4\u6bd4\n\n", "")

    return {
        "text": text,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--input", type=str, default="data/qmd_expansion.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--output", type=str, default="data/train", help="Output directory"
    )
    parser.add_argument(
        "--split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--add-short", type=int, default=3, help="Variations per short query to add"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    # Load and clean existing examples
    examples = []
    seen_queries = set()
    long_hyde_count = 0

    with open(input_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)

                # Clean output (truncate hyde, remove invalid lines)
                original_output = ex["output"]
                ex["output"] = clean_output(ex["output"])

                # Track hyde truncation
                if "hyde:" in original_output:
                    for orig_line in original_output.split("\n"):
                        if orig_line.strip().startswith("hyde:"):
                            if len(orig_line) > 160:
                                long_hyde_count += 1

                # Validate cleaned output
                has_lex = "lex:" in ex["output"]
                has_vec = "vec:" in ex["output"]

                if has_lex and has_vec:
                    examples.append(ex)
                    seen_queries.add(ex["input"].lower())

    print(f"Loaded and cleaned {len(examples)} examples")
    print(f"Truncated {long_hyde_count} long hyde sections")

    # Count existing short queries
    short_existing = sum(1 for ex in examples if len(ex["input"].split()) <= 2)
    print(f"Existing short queries (1-2 words): {short_existing}")

    # Generate additional short query examples
    new_short = []
    for query in SHORT_QUERIES:
        if query.lower() not in seen_queries:
            for _ in range(args.add_short):
                new_short.append(generate_short_example(query))
            seen_queries.add(query.lower())

    print(f"Generated {len(new_short)} new short query examples")

    # Combine and shuffle
    all_examples = examples + new_short
    random.shuffle(all_examples)

    # Format for training
    formatted = [format_for_training(ex["input"], ex["output"]) for ex in all_examples]

    # Split into train/val
    split_idx = int(len(formatted) * (1 - args.split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # Write train set
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Write validation set
    val_path = output_dir / "val.jsonl"
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Write chat format (for TRL)
    chat_path = output_dir / "train_chat.jsonl"
    with open(chat_path, "w") as f:
        for item in train_data:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")

    # Stats
    short_final = sum(1 for ex in all_examples if len(ex["input"].split()) <= 2)

    print(f"\n=== Summary ===")
    print(f"Total examples: {len(all_examples)}")
    print(
        f"Short queries: {short_final} ({100 * short_final / len(all_examples):.1f}%)"
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")

    # Dataset info
    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "short_query_pct": round(100 * short_final / len(all_examples), 1),
        "columns": ["prompt", "completion", "text", "messages"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
