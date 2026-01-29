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
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import QMDRewardFunction, run_eval

# --- Config (inlined from configs/grpo.yaml) ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
OUTPUT_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"
DATASET = "tobil/qmd-query-expansion-train-v2"


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

    # --- Automatic evaluation ---
    print("\nStarting automatic evaluation...")
    trainer.model.eval()
    run_eval(trainer.model, tokenizer, "grpo")


if __name__ == "__main__":
    main()
