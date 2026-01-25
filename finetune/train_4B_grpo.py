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
GRPO training for Qwen3-4B query expansion model.
Trains on top of merged SFT weights with reward function.
"""

import os
import re
from collections import Counter

import torch
import trackio
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# ==================== REWARD FUNCTION ====================

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}
KEY_TERM_STOPWORDS = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
                      'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
                      'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell'}

GENERIC_LEX_PHRASES = {
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with'
}


def extract_named_entities(query: str) -> set:
    """Extract named entities from query using simple heuristics."""
    entities = set()
    words = query.split()
    prev_was_entity = False

    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue

        is_entity = False

        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_entity = True
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        prev_was_entity = is_entity

    return entities


def get_key_terms(query: str) -> set:
    words = set(query.lower().split())
    return words - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    lex_words = set(lex_line.lower().split())
    return bool(key_terms & lex_words)


def lex_preserves_entities(lex_line: str, entities: set) -> bool:
    if not entities:
        return True
    lex_lower = lex_line.lower()
    return any(entity in lex_lower for entity in entities)


def lex_is_generic(lex_line: str) -> bool:
    lex_lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lex_lower or lex_lower.startswith(phrase.split()[0]):
            remaining = lex_lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
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

    # HARD FAIL: Chat template artifacts
    if any(token in text for token in ['<|im_start|>', '<|im_end|>', '<think>', '</think>',
                                        '\nassistant\n', '\nuser\n', '<|endoftext|>']):
        return 0.0

    # HARD FAIL: EVERY line must start with lex:, vec:, or hyde:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith(("lex:", "vec:", "hyde:")):
            return 0.0

    parsed = parse_expansion(expansion)

    # FORMAT (0-30)
    format_score = 0
    if parsed["lex"]:
        format_score += 10
    if parsed["vec"]:
        format_score += 10
    format_score += 10

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

    # NAMED ENTITY PRESERVATION
    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        lex_with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if lex_with_entities == len(parsed["lex"]):
            entity_score += 15
        elif lex_with_entities > 0:
            entity_score += 5
        else:
            entity_score -= 30

        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        entity_score -= generic_count * 15

        if parsed["vec"]:
            vec_with_entities = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with_entities > 0:
                entity_score += 5
    elif not entities:
        entity_score = 10

    total = format_score + diversity_score + hyde_score + quality_score + entity_score
    max_possible = 120 if parsed["hyde"] else 100
    return max(0.0, min(1.0, total / max_possible))


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


# ==================== MAIN ====================

def main():
    # Config
    SFT_MODEL = "tobil/qmd-query-expansion-4B-sft"
    BASE_MODEL = "Qwen/Qwen3-4B"
    OUTPUT_MODEL = "tobil/qmd-query-expansion-4B-grpo"
    DATASET = "tobil/qmd-query-expansion-train-v2"

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET, split="train")

    def extract_prompt(example):
        return {"prompt": example["messages"][0]["content"]}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load tokenizer and model
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading SFT model from {SFT_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, SFT_MODEL)
    model = model.merge_and_unload()
    print("Model loaded and LoRA merged.")

    # Add LoRA for GRPO
    grpo_lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()

    # GRPO config
    config = GRPOConfig(
        output_dir="qmd-query-expansion-4B-grpo",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,

        num_generations=4,
        max_completion_length=200,

        num_train_epochs=1,
        per_device_train_batch_size=1,  # Smaller for 4B model
        gradient_accumulation_steps=16,  # Compensate with more accumulation
        learning_rate=5e-7,
        max_grad_norm=0.5,
        max_steps=200,

        logging_steps=10,
        save_strategy="epoch",

        report_to="trackio",
        project="qmd-query-expansion",
        run_name="qwen3-4b-grpo",
    )

    # Train
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[QMDRewardFunction()],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()

    trackio.finish()
    print(f"Complete! Model at: https://huggingface.co/{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
