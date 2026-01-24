# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "trackio",
#     "datasets",
#     "bitsandbytes",
# ]
# ///
"""
GRPO (Group Relative Policy Optimization) training for QMD query expansion.

Uses the comprehensive scoring system from SCORING.md:
- Format (30%): Must have lex: and vec: prefixes
- Diversity (30%): No echoing query, diverse expansions
- Hyde (20%): Concise, no newlines, no repetition
- Quality (20%): lex=keywords, vec=natural language

Usage:
    uv run train_grpo.py --sft-model tobil/qmd-query-expansion-0.6B
"""

import os
import re
import torch
import trackio
from collections import Counter
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}

# ============================================================================
# Scoring Functions (from SCORING.md)
# ============================================================================

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
    """Count penalty for repeated words (excluding stopwords)."""
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    penalty = 0
    for word, count in counts.items():
        if count >= 3 and word not in STOPWORDS and len(word) > 2:
            penalty += (count - 2) * 2
    return penalty


def score_expansion(query: str, expansion: str) -> float:
    """
    Score an expansion based on SCORING.md criteria.
    Returns normalized score 0.0-1.0 for RL reward.
    """
    parsed = parse_expansion(expansion)

    # === FORMAT (0-30) ===
    format_score = 0
    if parsed["lex"]:
        format_score += 10
    if parsed["vec"]:
        format_score += 10
    if not parsed["invalid"]:
        format_score += 10
    else:
        format_score += max(0, 10 - len(parsed["invalid"]) * 5)

    # === DIVERSITY (0-30) ===
    diversity_score = 0

    # 2+ different types
    types_present = sum(1 for t in ["lex", "vec"] if parsed[t])
    if types_present >= 2:
        diversity_score += 10

    # 2+ total expansions
    total_expansions = len(parsed["lex"]) + len(parsed["vec"])
    if total_expansions >= 2:
        diversity_score += 5

    # Lex diversity
    lex_score = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2):
                lex_score -= 2
    diversity_score += max(0, lex_score)

    # Vec diversity
    vec_score = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3):
                vec_score -= 2
    diversity_score += max(0, vec_score)

    # Don't echo query
    echo_score = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo_score -= 3  # Heavier penalty for echoing
    diversity_score += max(0, echo_score)

    # === HYDE (0-20) ===
    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5  # Present

        # Length check (50-200 chars ideal)
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2

        # No newlines
        if "\n" not in hyde_text:
            hyde_score += 5

        # No repetition
        rep_penalty = word_repetition_penalty(hyde_text)
        hyde_score += max(0, 5 - rep_penalty)

    # === QUALITY (0-20) ===
    quality_score = 10  # Base

    # Lex should be shorter than vec
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5

    # Vec should be natural language
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        if natural == len(parsed["vec"]):
            quality_score += 5
        else:
            quality_score += 2

    # === TOTAL ===
    total = format_score + diversity_score + hyde_score + quality_score
    max_possible = 100 if parsed["hyde"] else 80

    # Normalize to 0-1
    return total / max_possible


def extract_query_from_prompt(prompt: str) -> str:
    """Extract the query from the prompt template."""
    # Prompt format: "Expand this search query:\n\n{query}"
    if "Expand this search query:" in prompt:
        return prompt.split("Expand this search query:")[-1].strip()
    return prompt.strip()


class QMDRewardFunction:
    """Reward function using comprehensive SCORING.md criteria."""
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        """Compute rewards for a batch of completions."""
        rewards = []

        for i, completion in enumerate(completions):
            # Get the query from prompt if available
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])

            # Score using comprehensive system
            score = score_expansion(query, completion)
            rewards.append(score)

        return rewards


# ============================================================================
# Main Training
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", default="tobil/qmd-query-expansion-0.6B",
                        help="SFT model to use as starting point")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B",
                        help="Base model (for loading tokenizer)")
    parser.add_argument("--output", default="tobil/qmd-query-expansion-0.6B-grpo-v2",
                        help="Output model name on Hub")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate (lower for stability)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("GRPO Training Config:")
        print(f"  SFT Model: {args.sft_model}")
        print(f"  Base Model: {args.base_model}")
        print(f"  Output: {args.output}")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        return

    # Login to HuggingFace Hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set, will try cached login")

    # Load dataset (just prompts needed for GRPO)
    print("Loading dataset...")
    dataset = load_dataset("tobil/qmd-query-expansion-train", split="train")

    # Extract just the queries as prompts
    def extract_prompt(example):
        return {"prompt": example["messages"][0]["content"]}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(2000, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model with LoRA adapter
    print(f"Loading SFT model from {args.sft_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.sft_model)
    model = model.merge_and_unload()
    print("Model loaded and LoRA merged.")

    # Add new LoRA adapter for GRPO training (smaller rank for stability)
    grpo_lora_config = LoraConfig(
        r=4,  # Smaller rank for more stable RL
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Fewer modules for stability
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()
    print("Added new LoRA adapter for GRPO.")

    # Initialize reward function
    reward_fn = QMDRewardFunction()

    # Test reward function
    print("\nTesting reward function...")
    test_good = "lex: auth setup\nlex: authentication config\nvec: how to configure authentication\nhyde: Configure auth by setting AUTH_SECRET."
    test_bad = "auth is important for security"
    print(f"  Good output score: {score_expansion('auth', test_good):.2f}")
    print(f"  Bad output score: {score_expansion('auth', test_bad):.2f}")

    # GRPO config with conservative settings
    config = GRPOConfig(
        output_dir="qmd-expansion-grpo-v2",
        push_to_hub=True,
        hub_model_id=args.output,

        # GRPO specific - conservative
        num_generations=4,
        max_completion_length=200,  # Shorter to avoid rambling

        # Training - very conservative
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        max_grad_norm=0.5,  # Clip gradients more aggressively

        # Logging
        logging_steps=10,
        save_strategy="epoch",

        # Monitoring
        report_to="trackio",
        project="qmd-query-expansion-grpo-v2",
        run_name="grpo-scoring-v2",
    )

    # Create trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save
    print("Pushing to Hub...")
    trainer.push_to_hub()

    trackio.finish()
    print(f"Done! Model at: https://huggingface.co/{args.output}")


if __name__ == "__main__":
    main()
