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
# ]
# ///
"""
Improved Qwen3-1.7B training with best practices for larger models:
- Lower learning rate (1e-4 instead of 2e-4)
- Higher LoRA rank (32 instead of 16)
- More epochs (5 instead of 3)
- Weight decay for regularization
"""

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Load dataset from Hub
print("Loading dataset...")
dataset = load_dataset("tobil/qmd-query-expansion-train", split="train")
print(f"Loaded {len(dataset)} examples")

# Create train/eval split
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Training configuration - optimized for larger model
config = SFTConfig(
    output_dir="qmd-query-expansion-1.7B-v2",
    push_to_hub=True,
    hub_model_id="tobil/qmd-query-expansion-1.7B-v2",
    hub_strategy="every_save",

    # Training parameters - lower LR, more epochs for larger model
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # Lowered from 2e-4
    weight_decay=0.01,   # Added regularization
    max_length=512,

    # Logging & checkpointing
    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=200,

    # Optimization
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Monitoring
    report_to="trackio",
    project="qmd-query-expansion",
    run_name="qwen3-1.7B-lora-v2",
)

# LoRA configuration - higher rank for better learning
peft_config = LoraConfig(
    r=32,           # Increased from 16
    lora_alpha=64,  # Increased from 32 (2x rank)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Initialize trainer
print("Initializing trainer with Qwen/Qwen3-1.7B...")
trainer = SFTTrainer(
    model="Qwen/Qwen3-1.7B",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()

trackio.finish()
print("Done! Model at: https://huggingface.co/tobil/qmd-query-expansion-1.7B-v2")
