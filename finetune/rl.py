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
                      'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
                      'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell'}

# Generic filler phrases that should never be in lex queries
GENERIC_LEX_PHRASES = {
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with'
}


def extract_named_entities(query: str) -> set:
    """Extract named entities from query using simple heuristics.

    Named entities are:
    - Capitalized words (except first word which may just be sentence start)
    - All-caps words/acronyms (TDS, API, GPU)
    - Technical terms with special chars (node.js, C++, .NET)
    - Words following acronyms/proper nouns (TDS motorsports -> both words)
    """
    entities = set()
    words = query.split()
    prev_was_entity = False

    for i, word in enumerate(words):
        # Clean punctuation but keep internal special chars
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue

        is_entity = False

        # All-caps words (acronyms): TDS, API, GPU, etc.
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True

        # Capitalized words (not first word, not common words)
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        # Technical terms with special chars: node.js, C++, .NET
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True

        # CamelCase: JavaScript, TypeScript, etc.
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_entity = True

        # Word following an entity is likely part of compound name (TDS motorsports)
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        prev_was_entity = is_entity

    return entities


def get_key_terms(query: str) -> set:
    """Get key terms (non-stopwords) from query."""
    words = set(query.lower().split())
    return words - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    """Check if lex line preserves key terms from query."""
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    lex_words = set(lex_line.lower().split())
    return bool(key_terms & lex_words)


def lex_preserves_entities(lex_line: str, entities: set) -> bool:
    """Check if lex line contains at least one named entity."""
    if not entities:
        return True  # No entities to preserve
    lex_lower = lex_line.lower()
    return any(entity in lex_lower for entity in entities)


def lex_is_generic(lex_line: str) -> bool:
    """Check if lex line is a generic filler phrase."""
    lex_lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lex_lower or lex_lower.startswith(phrase.split()[0]):
            # Also check if it's ONLY the generic phrase with no specifics
            remaining = lex_lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:  # Nothing specific left
                return True
    return False


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

    # NAMED ENTITY PRESERVATION (critical for quality)
    # This score can go heavily negative to punish missing entities
    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        # Count lex lines that preserve at least one entity
        lex_with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if lex_with_entities == len(parsed["lex"]):
            entity_score += 15  # All lex lines have entities - great!
        elif lex_with_entities > 0:
            entity_score += 5   # Some have entities
        else:
            entity_score -= 30  # NO lex lines have entities - HEAVY penalty!

        # Penalize generic filler phrases in lex (these are useless for BM25)
        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        entity_score -= generic_count * 15  # -15 per generic phrase

        # Bonus for entities in vec too (less critical but nice)
        if parsed["vec"]:
            vec_with_entities = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with_entities > 0:
                entity_score += 5
    elif not entities:
        # No entities in query - give base score
        entity_score = 10

    # Entity score CAN go negative to heavily penalize missing entities
    total = format_score + diversity_score + hyde_score + quality_score + entity_score
    max_possible = 120 if parsed["hyde"] else 100
    return max(0.0, min(1.0, total / max_possible))  # Clamp to 0.0-1.0


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

        # Test 1: Basic query
        test_good = "lex: auth setup\nlex: authentication config\nvec: how to configure authentication\nhyde: Configure auth by setting AUTH_SECRET."
        test_bad = "auth is important for security"
        print(f"\n  Query: 'auth'")
        print(f"    Good output score: {score_expansion('auth', test_good):.2f}")
        print(f"    Bad output score: {score_expansion('auth', test_bad):.2f}")

        # Test 2: Named entity query (the critical case!)
        query_entity = "who is TDS motorsports"
        entities = extract_named_entities(query_entity)
        print(f"\n  Query: '{query_entity}'")
        print(f"    Extracted entities: {entities}")

        good_entity = "lex: TDS motorsports history\nlex: TDS motorsports founders\nvec: information about TDS motorsports company"
        bad_entity = "lex: find information about\nlex: company details\nvec: who is this company"
        print(f"    Good (preserves entity): {score_expansion(query_entity, good_entity):.2f}")
        print(f"    Bad (generic phrases): {score_expansion(query_entity, bad_entity):.2f}")

        # Test 3: Technical term
        query_tech = "how to use React hooks"
        entities_tech = extract_named_entities(query_tech)
        print(f"\n  Query: '{query_tech}'")
        print(f"    Extracted entities: {entities_tech}")

        good_tech = "lex: React hooks tutorial\nlex: useEffect useState\nvec: how to use React hooks in functional components"
        bad_tech = "lex: programming tutorial\nlex: how to code\nvec: learn web development"
        print(f"    Good (preserves React): {score_expansion(query_tech, good_tech):.2f}")
        print(f"    Bad (generic): {score_expansion(query_tech, bad_tech):.2f}")

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
