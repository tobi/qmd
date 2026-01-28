# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "datasets",
#     "bitsandbytes",
#     "torch",
# ]
# ///
"""
GRPO training for QMD query expansion (Qwen3-1.7B).

Runs on top of merged SFT weights. Self-contained for HuggingFace Jobs:
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h jobs/grpo.py
"""

import os
import re
from collections import Counter

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# --- Config (inlined from configs/grpo.yaml) ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
OUTPUT_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"
DATASET = "tobil/qmd-query-expansion-train-v2"

# =============================================================================
# Reward function (inlined from reward.py — single source of truth)
# =============================================================================

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in',
    'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
})

KEY_TERM_STOPWORDS = frozenset({
    'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
    'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
    'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell',
})

GENERIC_LEX_PHRASES = frozenset({
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with',
})

CHAT_TEMPLATE_TOKENS = frozenset({
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\nassistant\n', '\nuser\n',
})


def parse_expansion(text: str) -> dict:
    result = {"lex": [], "vec": [], "hyde": [], "invalid": []}
    for line in text.strip().split("\n"):
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


def clean_model_output(text: str) -> tuple[str, bool]:
    text = text.replace('<|im_end|>', '').strip()
    used_thinking = '<think>' in text and '</think>' in text
    if used_thinking:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text, used_thinking


def extract_named_entities(query: str) -> set:
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
    return set(query.lower().split()) - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    return bool(key_terms & set(lex_line.lower().split()))


def lex_preserves_entities(line: str, entities: set) -> bool:
    if not entities:
        return True
    lower = line.lower()
    return any(e in lower for e in entities)


def lex_is_generic(lex_line: str) -> bool:
    lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lower or lower.startswith(phrase.split()[0]):
            remaining = lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


def word_set_distance(a: str, b: str) -> int:
    return len(set(a.lower().split()) ^ set(b.lower().split()))


def is_diverse(a: str, b: str, min_distance: int = 2) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion: str, query: str) -> bool:
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text: str) -> int:
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


def score_expansion(query: str, expansion: str) -> float:
    """Score expansion as float in [0.0, 1.0] for RL reward."""
    text, used_thinking = clean_model_output(expansion.strip())

    # Hard fail: chat template leakage
    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return 0.0

    # Hard fail: invalid lines
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return 0.0

    parsed = parse_expansion(text)

    # Format (0-30)
    format_score = 10  # no invalid lines
    if parsed["lex"]:
        format_score += 10
    if parsed["vec"]:
        format_score += 10

    # Diversity (0-30)
    diversity_score = 0
    types_present = sum(1 for t in ("lex", "vec") if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    if len(parsed["lex"]) + len(parsed["vec"]) >= 2:
        diversity_score += 5
    lex_div = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2):
                lex_div -= 2
    diversity_score += max(0, lex_div)
    vec_div = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3):
                vec_div -= 2
    diversity_score += max(0, vec_div)
    echo = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query):
            echo -= 3
    diversity_score += max(0, echo)

    # HyDE (0-20)
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
        hyde_score += max(0, 5 - word_repetition_penalty(hyde_text))

    # Quality (0-20)
    quality_score = 5
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        quality_score += 5 if natural == len(parsed["vec"]) else 2
    if parsed["lex"]:
        with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if with_terms == len(parsed["lex"]):
            quality_score += 5
        elif with_terms > 0:
            quality_score += 2

    # Entity (-45 to +20)
    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if with_entities == len(parsed["lex"]):
            entity_score += 15
        elif with_entities > 0:
            entity_score += 5
        else:
            entity_score -= 30
        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count:
            entity_score -= generic_count * 15
        if parsed["vec"]:
            vec_with = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with > 0:
                entity_score += 5
    elif not entities:
        entity_score = 10

    # Think bonus (0-20)
    think_bonus = 0 if used_thinking else 20

    total = format_score + diversity_score + hyde_score + quality_score + entity_score + think_bonus
    max_possible = 140 if parsed["hyde"] else 120
    return max(0.0, min(1.0, total / max_possible))


def extract_query_from_prompt(prompt: str) -> str:
    if "Expand this search query:" in prompt:
        query = prompt.split("Expand this search query:")[-1].strip()
        if "<|im_end|>" in query:
            query = query.split("<|im_end|>")[0].strip()
        return query
    return prompt.strip()


class QMDRewardFunction:
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])
            rewards.append(score_expansion(query, completion))
        return rewards


# =============================================================================
# Main training
# =============================================================================

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    print(f"Loading dataset: {DATASET}...")
    dataset = load_dataset(DATASET, split="train")

    def extract_prompt(example):
        content = example["messages"][0]["content"]
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base model, merge SFT adapter
    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Merging SFT adapter {SFT_MODEL}...")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Fresh LoRA for GRPO (small: rank 4, q/v only)
    grpo_lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, grpo_lora)
    model.print_trainable_parameters()

    config = GRPOConfig(
        output_dir="qmd-query-expansion-1.7B-grpo",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,

        num_generations=4,
        max_completion_length=200,
        beta=0.04,  # KL regularization — prevents drift from SFT checkpoint

        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        max_grad_norm=0.5,
        max_steps=200,

        logging_steps=10,
        save_strategy="epoch",
        bf16=True,

        report_to="none",
    )

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
    print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
