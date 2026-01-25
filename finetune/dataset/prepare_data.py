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

# Short single-word queries that need proper expansion examples
SHORT_QUERIES = [
    # Technical keywords
    "auth", "config", "setup", "api", "cache", "log", "test", "debug",
    "deploy", "build", "lint", "format", "migrate", "backup", "restore",
    "docker", "git", "npm", "pip", "brew", "curl", "ssh", "ssl", "tls",
    "cors", "csrf", "jwt", "oauth", "saml", "ldap", "rbac", "acl",
    "crud", "rest", "graphql", "grpc", "websocket", "sse", "http",
    "redis", "mongo", "postgres", "mysql", "sqlite", "elastic", "kafka",
    "nginx", "apache", "caddy", "traefik", "haproxy", "envoy",
    "react", "vue", "angular", "svelte", "solid", "htmx", "alpine",
    "node", "deno", "bun", "python", "rust", "golang", "java", "kotlin",
    "webpack", "vite", "esbuild", "rollup", "parcel", "turbopack",
    "jest", "vitest", "pytest", "mocha", "cypress", "playwright",
    # Common short phrases
    "env vars", "api keys", "error handling", "rate limiting",
    "file upload", "user auth", "db connection", "query params",
    "hot reload", "code split", "tree shake", "lazy load",
]

# Templates for generating short query expansions
# IMPORTANT: All lex lines MUST include {q} to preserve key terms
SHORT_TEMPLATES = [
    {
        "lex": ["{q} configuration", "{q} settings", "{q} setup"],
        "vec": ["how to configure {q} in my project", "{q} setup and configuration tutorial"],
        "hyde": "To set up {q}, first install the required dependencies. Then configure the settings in your project configuration file.",
    },
    {
        "lex": ["{q} tutorial", "{q} guide", "{q} basics"],
        "vec": ["beginner guide to {q}", "how to get started with {q}"],
        "hyde": "This guide covers the basics of {q}. Follow the steps below to get started with your first implementation.",
    },
    {
        "lex": ["{q} best practices", "{q} patterns", "{q} tips"],
        "vec": ["best practices for using {q}", "recommended patterns for {q}"],
        "hyde": "When working with {q}, follow these best practices: use consistent naming, handle errors properly, and document your code.",
    },
    {
        "lex": ["{q} troubleshooting", "{q} fix", "{q} errors"],
        "vec": ["how to fix {q} errors", "troubleshooting common {q} problems"],
        "hyde": "If you encounter {q} issues, check your configuration first. Common problems include missing dependencies and incorrect settings.",
    },
    {
        "lex": ["{q} examples", "{q} code", "{q} usage"],
        "vec": ["code examples for {q}", "practical {q} implementation examples"],
        "hyde": "Here are some practical examples of {q} in action. Each example demonstrates a common use case with working code.",
    },
]


def truncate_hyde(hyde_text: str, max_len: int = 150) -> str:
    """Truncate hyde to max length, ending at sentence boundary."""
    if len(hyde_text) <= max_len:
        return hyde_text

    truncated = hyde_text[:max_len]
    last_period = truncated.rfind(". ")
    if last_period > max_len // 2:
        return truncated[:last_period + 1]

    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space] + "."

    return truncated[:max_len-1] + "."


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
        {"role": "user", "content": f"/no_think Expand this search query: {input_text}"},
        {"role": "assistant", "content": output_text}
    ]

    # Use tokenizer to generate proper chat format with special tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Strip empty <think> tags - we don't want thinking mode
    # The template adds "<think>\n\n</think>\n\n" which we remove
    text = text.replace("<think>\n\n</think>\n\n", "")

    return {
        "text": text,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input", type=str, default="data/qmd_expansion.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/train", help="Output directory")
    parser.add_argument("--split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--add-short", type=int, default=3, help="Variations per short query to add")
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
    print(f"Short queries: {short_final} ({100*short_final/len(all_examples):.1f}%)")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")

    # Dataset info
    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "short_query_pct": round(100*short_final/len(all_examples), 1),
        "columns": ["prompt", "completion", "text", "messages"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
