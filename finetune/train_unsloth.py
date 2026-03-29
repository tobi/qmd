#!/usr/bin/env python3
"""
QMD Query Expansion fine-tuning with Unsloth (Qwen3.5 support).

Usage:
    python train_unsloth.py --model 0.8B
    python train_unsloth.py --model 2B
    python train_unsloth.py --model 4B --epochs 3

Requires: pip install unsloth unsloth_zoo
"""

import argparse
import json
import sys
from pathlib import Path

MODEL_MAP = {
    "0.8B": "unsloth/Qwen3.5-0.8B",
    "2B":   "unsloth/Qwen3.5-2B",
    "4B":   "unsloth/Qwen3.5-4B",
    "9B":   "unsloth/Qwen3.5-9B",
    "27B":  "unsloth/Qwen3.5-27B",
}

def main():
    parser = argparse.ArgumentParser(description="QMD fine-tuning with Unsloth")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                        help="Model size to train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--data", type=str, default="data/train/train.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/qwen3.5-{size})")
    parser.add_argument("--push-hub", type=str, default=None,
                        help="Push to HF hub (e.g. tobil/qmd-query-expansion-qwen3.5-0.8B)")
    parser.add_argument("--no-gguf", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    output_dir = args.output or f"outputs/qwen3.5-{args.model}"

    print(f"{'='*60}")
    print(f"QMD Query Expansion — Unsloth SFT")
    print(f"  Base model:  {model_name}")
    print(f"  Output:      {output_dir}")
    print(f"  Data:        {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch:       {args.batch_size} x {args.grad_accum} accum")
    print(f"  LR:          {args.lr}")
    print(f"  LoRA rank:   {args.lora_rank}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"{'='*60}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # --- Imports (heavy) ---
    import os
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # --- Load model ---
    print(f"\nLoading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # --- LoRA ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_len,
    )

    # --- Dataset ---
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- Tracking ---
    report_to = "none"
    if os.environ.get("HF_TOKEN"):
        try:
            import trackio
            report_to = "trackio"
            os.environ.setdefault("TRACKIO_PROJECT", "qmd-query-expansion")
        except ImportError:
            pass

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=output_dir,
            max_seq_length=args.max_seq_len,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=200,
            bf16=True,
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=4,
            report_to=report_to,
            run_name=f"sft-qwen3.5-{args.model}",
        ),
    )

    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Final loss:  {stats.training_loss:.4f}")

    # --- Save ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

    # --- GGUF export ---
    if not args.no_gguf:
        print("\nExporting GGUF quantizations...")
        gguf_dir = f"{output_dir}/gguf"
        for quant in ["q4_k_m", "q8_0"]:
            print(f"  {quant}...")
            try:
                model.save_pretrained_gguf(
                    gguf_dir, tokenizer, quantization_method=quant
                )
                print(f"  ✓ {quant} saved")
            except Exception as e:
                print(f"  ✗ {quant} failed: {e}")

    # --- Push to Hub ---
    if args.push_hub:
        print(f"\nPushing to {args.push_hub}...")
        model.push_to_hub_merged(args.push_hub, tokenizer, save_method="lora")
        if not args.no_gguf:
            for quant in ["q4_k_m", "q8_0"]:
                try:
                    model.push_to_hub_gguf(args.push_hub, tokenizer, quantization_method=quant)
                except Exception as e:
                    print(f"  GGUF push {quant} failed: {e}")

    # --- Eval ---
    if not args.no_eval:
        print("\nRunning evaluation...")
        import subprocess
        subprocess.run(
            [sys.executable, "eval.py", output_dir],
            cwd=str(Path(__file__).parent),
        )

    print(f"\n{'='*60}")
    print(f"Done! Model at: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
