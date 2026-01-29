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
SFT training for QMD query expansion (Qwen3-1.7B).

Self-contained script for HuggingFace Jobs:
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py
"""

import os
import sys
from huggingface_hub import login

# --- Config (inlined from configs/sft.yaml) ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
OUTPUT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
DATASET = "tobil/qmd-query-expansion-train-v2"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load and split dataset
print(f"Loading dataset: {DATASET}...")
dataset = load_dataset(DATASET, split="train")
print(f"Dataset loaded: {len(dataset)} examples")

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# SFT config
config = SFTConfig(
    output_dir="qmd-query-expansion-1.7B-sft",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_strategy="every_save",

    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,

    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,

    report_to="none",
)

# LoRA: rank 16, all projection layers
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print("Initializing SFT trainer...")
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting SFT training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()
print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")

# --- Automatic evaluation ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import run_eval

print("\nStarting automatic evaluation...")
eval_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if eval_tokenizer.pad_token is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
trainer.model.eval()
run_eval(trainer.model, eval_tokenizer, "sft")
