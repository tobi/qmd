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
Unified training script for QMD query expansion models.

Supports two stages:
  sft  - Supervised fine-tuning on labeled examples
  grpo - Group Relative Policy Optimization (RL) on top of merged SFT weights

Usage:
    uv run train.py sft  --config configs/sft.yaml
    uv run train.py grpo --config configs/grpo.yaml
    uv run train.py grpo --config configs/grpo.yaml --dry-run
"""

import argparse
import os
import sys

import yaml


def cmd_sft(args):
    """Run supervised fine-tuning."""
    import trackio
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print("SFT Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    print(f"Loading dataset: {cfg['dataset']['name']}...")
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    print(f"Dataset loaded: {len(dataset)} examples")

    split = dataset.train_test_split(test_size=cfg["dataset"]["eval_split"], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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

    peft_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )

    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=cfg["model"]["base"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        peft_config=peft_config,
    )

    print("Starting SFT training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()
    trackio.finish()
    print(f"Done! Model: https://huggingface.co/{cfg['model']['output']}")


def cmd_grpo(args):
    """Run GRPO reinforcement learning on top of merged SFT weights."""
    import torch
    import trackio
    from datasets import load_dataset
    from huggingface_hub import login
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    # Import reward from the shared module
    sys.path.insert(0, os.path.dirname(__file__))
    from reward import QMDRewardFunction, score_expansion, extract_named_entities

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print("GRPO Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        print("\nTesting reward function...")

        tests = [
            ("auth", "lex: auth setup\nlex: authentication config\nvec: how to configure authentication\nhyde: Configure auth by setting AUTH_SECRET."),
            ("auth", "auth is important for security"),
            ("who is TDS motorsports", "lex: TDS motorsports history\nlex: TDS motorsports founders\nvec: information about TDS motorsports company"),
            ("who is TDS motorsports", "lex: find information about\nlex: company details\nvec: who is this company"),
        ]
        for query, expansion in tests:
            score = score_expansion(query, expansion)
            print(f"  '{query}' -> {score:.2f}")
        return

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)

    # Load tokenizer
    base_model_name = cfg["model"]["base"]
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    print("Loading dataset...")
    dataset = load_dataset(cfg["dataset"]["name"], split="train")

    def extract_prompt(example):
        content = example[cfg["dataset"]["prompt_field"]][0]["content"]
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    max_samples = cfg["dataset"].get("max_samples", len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base model, merge SFT adapter
    sft_model_name = cfg["model"]["sft"]
    print(f"Loading SFT model from {sft_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, sft_model_name)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Add fresh LoRA for GRPO
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

    # Build GRPO config, including beta and temperature
    grpo_cfg = cfg.get("grpo", {})
    config = GRPOConfig(
        output_dir=cfg["model"]["output"].split("/")[-1],
        push_to_hub=True,
        hub_model_id=cfg["model"]["output"],

        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 200),
        beta=grpo_cfg.get("beta", 0.04),

        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        max_steps=cfg["training"].get("max_steps", -1),

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
        reward_funcs=[QMDRewardFunction()],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()
    trackio.finish()
    print(f"Done! Model: https://huggingface.co/{cfg['model']['output']}")


def main():
    parser = argparse.ArgumentParser(
        description="QMD Query Expansion Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run train.py sft  --config configs/sft.yaml
  uv run train.py grpo --config configs/grpo.yaml
  uv run train.py grpo --config configs/grpo.yaml --dry-run
        """,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    sft_parser = sub.add_parser("sft", help="Supervised fine-tuning")
    sft_parser.add_argument("--config", required=True, help="Path to SFT config YAML")
    sft_parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    grpo_parser = sub.add_parser("grpo", help="GRPO reinforcement learning")
    grpo_parser.add_argument("--config", required=True, help="Path to GRPO config YAML")
    grpo_parser.add_argument("--dry-run", action="store_true", help="Print config, test reward, and exit")

    args = parser.parse_args()

    if args.stage == "sft":
        cmd_sft(args)
    elif args.stage == "grpo":
        cmd_grpo(args)


if __name__ == "__main__":
    main()
