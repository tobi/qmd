#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
# ]
# ///
"""
Generate QMD training data by transforming s-emanuilov/query-expansion dataset
and adding synthetic hyde passages. No API calls needed.
"""

import json
import random
from pathlib import Path

# HyDE passage templates for different query types
HYDE_TEMPLATES = {
    "how_to": [
        "To {action}, you need to {steps}. This can be done by {method}.",
        "The recommended way to {action} is to first {step1}, then {step2}.",
        "{Topic} can be achieved by {method}. Make sure to {consideration}.",
    ],
    "what_is": [
        "{Topic} is a {category} that {description}. It is commonly used for {use_case}.",
        "{Topic} refers to {definition}. Key features include {features}.",
    ],
    "config": [
        "To configure {topic}, set the {setting} option to {value}. You can also customize {other}.",
        "Configuration for {topic} is done in the {file} file. Key settings include {settings}.",
    ],
    "error": [
        "The {error} error occurs when {cause}. To fix this, {solution}.",
        "If you encounter {error}, check that {check}. Common solutions include {solutions}.",
    ],
    "general": [
        "{Topic} provides {benefit} for {use_case}. It works by {mechanism}.",
        "When working with {topic}, consider {considerations}. Best practices include {practices}.",
    ],
}

def classify_query(query: str) -> str:
    """Classify query type for hyde template selection."""
    q = query.lower()
    if any(w in q for w in ["how to", "how do", "setup", "install", "configure", "create"]):
        return "how_to"
    if any(w in q for w in ["what is", "what are", "definition", "meaning"]):
        return "what_is"
    if any(w in q for w in ["config", "setting", "option"]):
        return "config"
    if any(w in q for w in ["error", "issue", "problem", "fix", "debug"]):
        return "error"
    return "general"


def extract_topic(query: str) -> str:
    """Extract main topic from query."""
    # Remove common prefixes
    for prefix in ["how to ", "how do i ", "what is ", "what are ", "configure ", "setup "]:
        if query.lower().startswith(prefix):
            return query[len(prefix):].strip()
    return query


def generate_hyde(query: str, expansions: list[str]) -> str:
    """Generate a hypothetical document passage by combining expansions naturally."""
    topic = extract_topic(query)
    query_type = classify_query(query)

    # Use the longest, most descriptive expansion as the base
    sorted_exp = sorted(expansions, key=len, reverse=True)
    main_exp = sorted_exp[0] if sorted_exp else topic

    # Build a natural passage based on query type
    if query_type == "how_to":
        templates = [
            f"To {topic}, start by reviewing the requirements and dependencies. {main_exp.capitalize()} is the recommended approach. Make sure all prerequisites are met before proceeding.",
            f"The process of {topic} involves several steps. First, {main_exp}. Follow the official documentation for detailed instructions.",
            f"When you need to {topic}, the most effective method is to {main_exp}. This ensures compatibility and follows best practices.",
        ]
    elif query_type == "what_is":
        templates = [
            f"{topic.capitalize()} refers to {main_exp}. It is widely used in various applications and provides significant benefits.",
            f"The concept of {topic} encompasses {main_exp}. Understanding this is essential for effective implementation.",
            f"{topic.capitalize()} is defined as {main_exp}. This plays a crucial role in modern development practices.",
        ]
    elif query_type == "config":
        templates = [
            f"Configuration for {topic} requires setting the appropriate parameters. {main_exp.capitalize()} should be adjusted based on your specific requirements.",
            f"To configure {topic}, modify the settings in your configuration file. Key options include those related to {main_exp}.",
            f"The {topic} configuration can be customized by {main_exp}. Default values work for most use cases.",
        ]
    elif query_type == "error":
        templates = [
            f"The {topic} issue typically occurs when dependencies are misconfigured. To resolve this, {main_exp}. Check your environment settings.",
            f"If you encounter problems with {topic}, verify that {main_exp}. Common solutions include updating dependencies and checking permissions.",
            f"Debugging {topic} requires understanding the root cause. Often, {main_exp} resolves the issue. Review logs for details.",
        ]
    else:
        templates = [
            f"{topic.capitalize()} is an important concept that relates to {main_exp}. It provides functionality for various use cases in software development.",
            f"Understanding {topic} is essential for modern development. Key aspects include {main_exp}. This knowledge helps in building robust applications.",
            f"The topic of {topic} covers {main_exp}. Proper implementation follows established patterns and best practices.",
        ]

    return random.choice(templates)


def transform_to_qmd_format(query: str, expansions: list[str]) -> str:
    """Transform s-emanuilov format to QMD lex/vec/hyde format."""
    lines = []

    # Generate lex lines (keyword-focused, shorter)
    lex_candidates = []
    for exp in expansions:
        # Shorter versions for lex
        words = exp.split()
        if len(words) <= 4:
            lex_candidates.append(exp)
        else:
            # Take key phrases
            lex_candidates.append(" ".join(words[:3]))

    # Add 1-2 lex lines
    for lex in lex_candidates[:2]:
        if lex.lower() != query.lower():
            lines.append(f"lex: {lex}")

    # Generate vec lines (semantic, complete phrases)
    vec_candidates = [exp for exp in expansions if len(exp.split()) >= 3]
    if not vec_candidates:
        vec_candidates = expansions

    # Add 1-2 vec lines
    for vec in vec_candidates[:2]:
        if vec.lower() != query.lower():
            lines.append(f"vec: {vec}")

    # Generate hyde line
    hyde = generate_hyde(query, expansions)
    lines.append(f"hyde: {hyde}")

    return "\n".join(lines)


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        import subprocess
        subprocess.run(["uv", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    print("Loading s-emanuilov/query-expansion dataset...")
    dataset = load_dataset("s-emanuilov/query-expansion", split="train")

    print(f"Loaded {len(dataset)} examples")

    # Transform each example
    output_path = Path("data/qmd_expansion.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for item in dataset:
        query = item["query"]
        expansions = item["expansions"]

        output = transform_to_qmd_format(query, expansions)
        examples.append({"input": query, "output": output})

    # Shuffle
    random.seed(42)
    random.shuffle(examples)

    # Write output
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} examples to {output_path}")

    # Show sample
    print("\nSample output:")
    print("-" * 50)
    sample = examples[0]
    print(f"Input: {sample['input']}")
    print(f"Output:\n{sample['output']}")


if __name__ == "__main__":
    main()
