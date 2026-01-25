# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "datasets>=2.14.0",
#     "trackio",
#     "pyyaml",
# ]
# ///
"""
SFT Training for QMD Query Expansion.

Usage:
    uv run train.py --config configs/sft_v4.yaml
    uv run train.py --config configs/sft_v4.yaml --dry-run
"""

import argparse
import yaml

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Train QMD query expansion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print("SFT Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    print(f"Loading dataset: {cfg['dataset']['name']}...")
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    print(f"Dataset loaded: {len(dataset)} examples")

    # Create train/eval split
    print("Creating train/eval split...")
    split = dataset.train_test_split(test_size=cfg["dataset"]["eval_split"], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Eval: {len(eval_dataset)} examples")

    # Training configuration
    config = SFTConfig(
        output_dir=cfg["model"]["output"].split("/")[-1],
        push_to_hub=True,
        hub_model_id=cfg["model"]["output"],
        hub_strategy="every_save",

        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        max_length=cfg["training"]["max_length"],

        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        eval_strategy="steps",
        eval_steps=200,

        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler"],

        report_to="trackio",
        project=cfg["tracking"]["project"],
        run_name=cfg["tracking"]["run_name"],
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )

    # Initialize and train
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=cfg["model"]["base"],
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
    print(f"Complete! Model at: https://huggingface.co/{cfg['model']['output']}")


if __name__ == "__main__":
    main()
