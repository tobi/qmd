# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "trackio",
#     "datasets",
#     "bitsandbytes",
#     "sentence-transformers",
# ]
# ///
"""
GRPO (Group Relative Policy Optimization) training for QMD query expansion.

Reward Type 2: Format + Diversity
- Rewards correct lex/vec/hyde format
- Penalizes repetition between lines
- Rewards semantic diversity of expansions

Usage:
    uv run train_grpo.py --sft-model tobil/qmd-query-expansion-0.6B
"""

import re
import torch
import trackio
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer

# ============================================================================
# Reward Function: Format + Diversity
# ============================================================================

def parse_expansion(text: str) -> dict:
    """Parse expansion output into lex/vec/hyde components."""
    result = {"lex": [], "vec": [], "hyde": []}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())

    return result


def compute_format_reward(text: str) -> float:
    """
    Reward for correct format:
    - Has at least 1 lex line: +0.2
    - Has at least 1 vec line: +0.2
    - Has hyde line: +0.1
    - Correct line format (type: content): +0.1 per line (max 0.3)
    - No garbage/malformed lines: +0.2
    """
    reward = 0.0
    parsed = parse_expansion(text)

    # Check required components
    if parsed["lex"]:
        reward += 0.2
    if parsed["vec"]:
        reward += 0.2
    if parsed["hyde"]:
        reward += 0.1

    # Check line format
    lines = text.strip().split("\n")
    valid_lines = 0
    for line in lines:
        if re.match(r'^(lex|vec|hyde):\s*.+', line.strip()):
            valid_lines += 1

    reward += min(0.3, valid_lines * 0.1)

    # Penalize malformed lines
    malformed = len(lines) - valid_lines
    if malformed == 0:
        reward += 0.2
    else:
        reward -= malformed * 0.1

    return max(0.0, min(1.0, reward))


def compute_diversity_reward(text: str, embedder) -> float:
    """
    Reward for diverse expansions:
    - Penalize exact duplicates
    - Reward semantic distance between expansions
    """
    parsed = parse_expansion(text)
    all_expansions = parsed["lex"] + parsed["vec"] + parsed["hyde"]

    if len(all_expansions) < 2:
        return 0.0

    # Penalize exact duplicates
    unique = set(e.lower() for e in all_expansions)
    duplicate_penalty = (len(all_expansions) - len(unique)) * 0.2

    # Compute semantic diversity
    if len(unique) >= 2:
        try:
            embeddings = embedder.encode(list(unique))
            # Compute pairwise cosine similarities
            from torch.nn.functional import cosine_similarity
            emb_tensor = torch.tensor(embeddings)

            similarities = []
            for i in range(len(emb_tensor)):
                for j in range(i + 1, len(emb_tensor)):
                    sim = cosine_similarity(
                        emb_tensor[i].unsqueeze(0),
                        emb_tensor[j].unsqueeze(0)
                    ).item()
                    similarities.append(sim)

            # Lower similarity = higher diversity = higher reward
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            diversity_reward = 1.0 - avg_similarity  # 0 = identical, 1 = orthogonal
        except Exception:
            diversity_reward = 0.0
    else:
        diversity_reward = 0.0

    return max(0.0, diversity_reward - duplicate_penalty)


def compute_length_reward(text: str) -> float:
    """Reward appropriate length (not too short, not too long)."""
    lines = [l for l in text.strip().split("\n") if l.strip()]

    # Ideal: 3-6 lines
    if 3 <= len(lines) <= 6:
        return 0.2
    elif 2 <= len(lines) <= 7:
        return 0.1
    else:
        return 0.0


class QMDRewardFunction:
    """Combined reward function for QMD query expansion."""
    __name__ = "qmd_format_diversity_reward"

    def __init__(self):
        # Load a small embedding model for diversity computation
        print("Loading embedding model for diversity reward...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        """Compute rewards for a batch of completions."""
        rewards = []

        for completion in completions:
            # Extract just the generated part (after prompt)
            text = completion

            # Compute component rewards
            format_r = compute_format_reward(text)
            diversity_r = compute_diversity_reward(text, self.embedder)
            length_r = compute_length_reward(text)

            # Weighted combination
            total = (
                0.5 * format_r +      # Format is most important
                0.35 * diversity_r +  # Diversity is second
                0.15 * length_r       # Length is minor
            )

            rewards.append(total)

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
    parser.add_argument("--output", default="tobil/qmd-query-expansion-0.6B-grpo",
                        help="Output model name on Hub")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("GRPO Training Config:")
        print(f"  SFT Model: {args.sft_model}")
        print(f"  Base Model: {args.base_model}")
        print(f"  Output: {args.output}")
        print(f"  Epochs: {args.epochs}")
        return

    # Load dataset (just prompts needed for GRPO)
    print("Loading dataset...")
    dataset = load_dataset("tobil/qmd-query-expansion-train", split="train")

    # Extract just the queries as prompts
    def extract_prompt(example):
        return {"prompt": example["messages"][0]["content"]}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(2000, len(dataset))))  # Use subset for GRPO
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
    model = model.merge_and_unload()  # Merge LoRA weights
    print("Model loaded and LoRA merged.")

    # Add new LoRA adapter for GRPO training
    from peft import get_peft_model
    grpo_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()
    print("Added new LoRA adapter for GRPO.")

    # Initialize reward function
    reward_fn = QMDRewardFunction()

    # GRPO config
    config = GRPOConfig(
        output_dir="qmd-expansion-grpo",
        push_to_hub=True,
        hub_model_id=args.output,

        # GRPO specific
        num_generations=4,  # Generate 4 completions per prompt
        max_completion_length=256,

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,  # Lower LR for RL

        # Logging
        logging_steps=10,
        save_strategy="epoch",

        # Monitoring
        report_to="trackio",
        project="qmd-query-expansion-grpo",
        run_name="grpo-format-diversity",
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
