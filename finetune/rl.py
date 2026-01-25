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
#     "pyyaml",
# ]
# ///
"""
GRPO (Group Relative Policy Optimization) training for QMD query expansion.

Uses the scoring system from SCORING.md as the reward function.

Usage:
    uv run rl.py --config configs/grpo_v4.yaml
    uv run rl.py --config configs/grpo_v4.yaml --dry-run
"""

import os
import re
import argparse
import yaml

import torch
import trackio
from collections import Counter
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}
KEY_TERM_STOPWORDS = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
                      'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we'}


def get_key_terms(query: str) -> set:
    words = set(query.lower().split())
    return words - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    lex_words = set(lex_line.lower().split())
    return bool(key_terms & lex_words)


def parse_expansion(text: str) -> dict:
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
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    return len(words_a ^ words_b)


def is_diverse(a: str, b: str, min_distance: int = 2) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return False
    if a in b or b in a:
        return False
    return edit_distance_simple(a, b) >= min_distance


def echoes_query(expansion: str, query: str) -> bool:
    exp = expansion.lower().strip()
    q = query.lower().strip()
    if exp == q:
        return True
    if q in exp and len(exp) < len(q) + 10:
        return True
    return False


def word_repetition_penalty(text: str) -> int:
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    penalty = 0
    for word, count in counts.items():
        if count >= 3 and word not in STOPWORDS and len(word) > 2:
            penalty += (count - 2) * 2
    return penalty


def score_expansion(query: str, expansion: str) -> float:
    """Score expansion. Returns 0.0-1.0 for RL reward."""
    text = expansion.strip()

    # HARD FAIL: Must start with valid prefix (prevents verbose explanations)
    first_line = text.split("\n")[0].strip() if text else ""
    if not first_line.startswith(("lex:", "vec:", "hyde:")):
        return 0.0  # Zero reward for wrong format

    parsed = parse_expansion(expansion)

    # FORMAT (0-30)
    format_score = 0
    if parsed["lex"]:
        format_score += 10
    if parsed["vec"]:
        format_score += 10
    if not parsed["invalid"]:
        format_score += 10
    else:
        format_score += max(0, 10 - len(parsed["invalid"]) * 5)

    # DIVERSITY (0-30)
    diversity_score = 0
    types_present = sum(1 for t in ["lex", "vec"] if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    total_expansions = len(parsed["lex"]) + len(parsed["vec"])
    if total_expansions >= 2:
        diversity_score += 5

    lex_score = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2):
                lex_score -= 2
    diversity_score += max(0, lex_score)

    vec_score = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3):
                vec_score -= 2
    diversity_score += max(0, vec_score)

    echo_score = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo_score -= 3
    diversity_score += max(0, echo_score)

    # HYDE (0-20)
    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2
        if "\n" not in hyde_text:
            hyde_score += 5
        rep_penalty = word_repetition_penalty(hyde_text)
        hyde_score += max(0, 5 - rep_penalty)

    # QUALITY (0-20)
    quality_score = 5
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        if natural == len(parsed["vec"]):
            quality_score += 5
        else:
            quality_score += 2
    if parsed["lex"]:
        lex_with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if lex_with_terms == len(parsed["lex"]):
            quality_score += 5
        elif lex_with_terms > 0:
            quality_score += 2

    total = format_score + diversity_score + hyde_score + quality_score
    max_possible = 100 if parsed["hyde"] else 80
    return total / max_possible


def extract_query_from_prompt(prompt: str) -> str:
    if "Expand this search query:" in prompt:
        return prompt.split("Expand this search query:")[-1].strip()
    return prompt.strip()


class QMDRewardFunction:
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])
            score = score_expansion(query, completion)
            rewards.append(score)
        return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print("GRPO Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        print("\nTesting reward function...")
        test_good = "lex: auth setup\nlex: authentication config\nvec: how to configure authentication\nhyde: Configure auth by setting AUTH_SECRET."
        test_bad = "auth is important for security"
        print(f"  Good output score: {score_expansion('auth', test_good):.2f}")
        print(f"  Bad output score: {score_expansion('auth', test_bad):.2f}")
        return

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(cfg["dataset"]["name"], split="train")

    def extract_prompt(example):
        return {"prompt": example[cfg["dataset"]["prompt_field"]][0]["content"]}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    max_samples = cfg["dataset"].get("max_samples", len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load tokenizer and model
    print(f"Loading tokenizer from {cfg['model']['base']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading SFT model from {cfg['model']['sft']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, cfg["model"]["sft"])
    model = model.merge_and_unload()
    print("Model loaded and LoRA merged.")

    # Add LoRA for GRPO
    grpo_lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()

    # Reward function
    reward_fn = QMDRewardFunction()

    # GRPO config
    config = GRPOConfig(
        output_dir=cfg["model"]["output"].split("/")[-1],
        push_to_hub=True,
        hub_model_id=cfg["model"]["output"],

        num_generations=cfg["grpo"]["num_generations"],
        max_completion_length=cfg["grpo"]["max_completion_length"],

        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        max_grad_norm=cfg["training"]["max_grad_norm"],

        logging_steps=10,
        save_strategy="epoch",

        report_to="trackio",
        project=cfg["tracking"]["project"],
        run_name=cfg["tracking"]["run_name"],
    )

    # Train
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()

    trackio.finish()
    print(f"Done! Model at: https://huggingface.co/{cfg['model']['output']}")


if __name__ == "__main__":
    main()
