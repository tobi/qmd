#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "transformers>=4.45.0",
#     "datasets",
#     "trl>=0.12.0",
#     "torch",
#     "huggingface_hub",
# ]
# ///
"""
Train QMD query expansion model using LoRA on HuggingFace Jobs.

This script is designed to run on HuggingFace Jobs infrastructure.
Uses Unsloth for efficient LoRA finetuning.

Usage:
    # Local test
    python train_hf_job.py --model Qwen/Qwen3-0.6B --data data/train --dry-run

    # HuggingFace Jobs (via huggingface-skills)
    # See hugging-face-model-trainer skill for deployment
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train QMD query expansion model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("--data", type=str, default="data/train", help="Training data directory")
    parser.add_argument("--output", type=str, default="models/qmd-expansion", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--push-to-hub", type=str, help="Push to HuggingFace Hub repo")
    args = parser.parse_args()

    config = {
        "model": args.model,
        "data": args.data,
        "output": args.output,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "max_seq_length": args.max_seq_length,
    }

    if args.dry_run:
        print("Training configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        return

    # Import heavy dependencies only when needed
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    print(f"Loading base model: {args.model}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    data_path = Path(args.data)
    if (data_path / "train_chat.jsonl").exists():
        dataset = load_dataset("json", data_files=str(data_path / "train_chat.jsonl"))["train"]
        print(f"Loaded {len(dataset)} training examples (chat format)")
    else:
        dataset = load_dataset("json", data_files=str(data_path / "train.jsonl"))["train"]
        print(f"Loaded {len(dataset)} training examples")

    # Format function for chat template
    def format_chat(example):
        messages = example.get("messages", [])
        if messages:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = example.get("text", "")
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Training config
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("Training complete!")


if __name__ == "__main__":
    main()
