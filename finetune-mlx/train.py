#!/usr/bin/env python3
"""
QMD Query Expansion Training - Apple Silicon Edition

Uses MLX for efficient fine-tuning on Metal GPUs.

Usage:
    python train.py sft              # Supervised fine-tuning
    python train.py grpo             # RL refinement (after SFT)
    python train.py sft --config configs/sft.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download


def prepare_dataset(config: dict, stage: str) -> Path:
    """Download and prepare dataset in MLX format.

    Supports both:
    - Local path: config.dataset.path = "data/custom"
    - HuggingFace: config.dataset.name = "tobil/qmd-query-expansion-train-v2"
    """
    dataset_config = config.get("dataset", {})

    # Check for local path first
    local_path = dataset_config.get("path")
    if local_path:
        local_dir = Path(local_path)
        train_file = local_dir / "train.jsonl"
        valid_file = local_dir / "valid.jsonl"

        if train_file.exists() and valid_file.exists():
            # Count examples
            train_count = sum(1 for _ in open(train_file))
            valid_count = sum(1 for _ in open(valid_file))
            print(f"📦 Using local dataset: {local_dir}")
            print(f"✅ Dataset ready: {train_count} train, {valid_count} valid")
            return local_dir
        else:
            print(f"❌ Local dataset not found at {local_dir}")
            print(f"   Expected: {train_file} and {valid_file}")
            print(f"   Run: python dataset/generate_from_notes.py && python dataset/prepare_data.py")
            sys.exit(1)

    # Fall back to HuggingFace dataset
    dataset_name = dataset_config.get("name", "tobil/qmd-query-expansion-train-v2")
    output_dir = Path("data") / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Loading dataset from HuggingFace: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    # Split into train/valid
    eval_ratio = dataset_config.get("eval_split", 0.1)
    split = ds.train_test_split(test_size=eval_ratio, seed=42)

    # Convert to MLX format (JSONL with "text" field)
    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    text_field = dataset_config.get("text_field", "text")

    train_count = 0
    with open(train_file, "w") as f:
        for item in split["train"]:
            text = item[text_field]
            if text and isinstance(text, str) and len(text) > 10:
                f.write(json.dumps({"text": text}) + "\n")
                train_count += 1

    valid_count = 0
    with open(valid_file, "w") as f:
        for item in split["test"]:
            text = item[text_field]
            if text and isinstance(text, str) and len(text) > 10:
                f.write(json.dumps({"text": text}) + "\n")
                valid_count += 1

    print(f"✅ Dataset prepared: {train_count} train, {valid_count} valid (filtered nulls)")
    return output_dir


def download_model(model_name: str) -> Path:
    """Download model and convert to MLX format if needed."""
    print(f"📥 Downloading model: {model_name}")
    
    # Check if already converted to MLX
    mlx_model_dir = Path("models") / model_name.replace("/", "_") / "mlx"
    
    if mlx_model_dir.exists():
        print(f"✅ Model already exists: {mlx_model_dir}")
        return mlx_model_dir
    
    # Download and convert
    mlx_model_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Use mlx_lm.convert to download and convert
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", model_name,
        "--mlx-path", str(mlx_model_dir),
        "-q"  # Quantize for memory efficiency
    ]
    
    print(f"🔄 Converting to MLX format...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        # Try without quantization
        cmd.remove("-q")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Model conversion failed: {result.stderr}")
    
    print(f"✅ Model ready: {mlx_model_dir}")
    return mlx_model_dir


def train_sft(config: dict):
    """Run supervised fine-tuning with LoRA."""
    print("\n🚀 Starting SFT Training\n")
    
    model_name = config.get("model", {}).get("base", "Qwen/Qwen2.5-1.5B")
    output_name = config.get("model", {}).get("output", "qmd-expansion-sft")
    
    # Prepare data
    data_dir = prepare_dataset(config, "sft")
    
    # Download/convert model
    model_dir = download_model(model_name)
    
    # Training params
    training = config.get("training", {})
    lora = config.get("lora", {})
    
    # Use output name for adapter path (allows multiple trained models)
    adapter_name = output_name.replace("/", "_").replace("-", "_")
    adapter_path = Path("adapters") / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(model_dir),
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_path),
        "--batch-size", str(training.get("batch_size", 4)),
        "--iters", str(training.get("iters", 1000)),
        "--learning-rate", str(training.get("learning_rate", 1e-4)),
        "--num-layers", str(lora.get("num_layers", 16)),
        "--steps-per-report", "10",
        "--steps-per-eval", "100",
        "--save-every", "200",
    ]
    
    max_length = training.get("max_length", 512)
    if max_length:
        cmd.extend(["--max-seq-length", str(max_length)])
    
    print(f"📝 Command: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Training failed")
        sys.exit(1)
    
    print(f"\n✅ SFT complete! Adapter saved to: {adapter_path}")
    return adapter_path


def train_grpo(config: dict):
    """Run GRPO (RL refinement) on top of SFT model."""
    print("\n🚀 Starting GRPO Training\n")
    
    # Check SFT adapter exists
    sft_adapter = Path("adapters") / "sft"
    if not sft_adapter.exists():
        print("❌ SFT adapter not found. Run 'python train.py sft' first.")
        sys.exit(1)
    
    # For GRPO, we need to implement reward-based training
    # MLX doesn't have built-in GRPO, so we'll use a simpler approach:
    # Continue SFT with filtered high-quality examples
    
    print("⚠️  GRPO not yet implemented for MLX.")
    print("   The SFT model should work well for most use cases.")
    print("   For GRPO, use HuggingFace Jobs with the original scripts.")
    
    # TODO: Implement reward model + PPO-style training
    # This would require:
    # 1. Generate expansions from SFT model
    # 2. Score with reward function
    # 3. Filter top examples
    # 4. Continue training on high-reward examples
    
    return sft_adapter


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="QMD Query Expansion Training")
    parser.add_argument("stage", choices=["sft", "grpo"], help="Training stage")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Default config
        config_path = Path("configs") / f"{args.stage}.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = {}
    
    # Run training
    if args.stage == "sft":
        train_sft(config)
    elif args.stage == "grpo":
        train_grpo(config)


if __name__ == "__main__":
    main()
