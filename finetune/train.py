# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "trackio",
#     "nvidia-ml-py",
#     "datasets",
#     "bitsandbytes",
#     "pyyaml",
#     "gguf",
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
import subprocess
import sys
from pathlib import Path

import yaml


def export_gguf(model, tokenizer, output_dir: str, model_name: str):
    """Export model to GGUF at Q4_K_M, Q6_K, Q8_0 quantizations."""
    import shutil
    import tempfile

    output_path = Path(output_dir)
    gguf_dir = output_path / "gguf"
    gguf_dir.mkdir(exist_ok=True)

    # Save merged model to temp dir
    print("Saving merged model for GGUF conversion...")
    with tempfile.TemporaryDirectory() as tmp:
        merged_path = Path(tmp) / "merged"
        model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

        # Setup llama.cpp
        llama_cpp = Path("/tmp/llama.cpp")
        if not llama_cpp.exists():
            print("Cloning llama.cpp...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(llama_cpp),
                ],
                capture_output=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "-r",
                    str(llama_cpp / "requirements.txt"),
                ],
                capture_output=True,
            )

        # Build quantize tool if needed
        quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            print("Building llama-quantize...")
            build_dir = llama_cpp / "build"
            build_dir.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "cmake",
                    "-B",
                    str(build_dir),
                    "-S",
                    str(llama_cpp),
                    "-DGGML_CUDA=OFF",
                ],
                capture_output=True,
            )
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--target",
                    "llama-quantize",
                    "-j",
                    "4",
                ],
                capture_output=True,
            )

        # Convert to FP16 first
        fp16_file = gguf_dir / f"{model_name}-f16.gguf"
        print(f"Converting to FP16: {fp16_file}")
        log_out = Path("/tmp/qmd-gguf-convert.log")
        log_err = Path("/tmp/qmd-gguf-convert.err")
        with log_out.open("w") as out_f, log_err.open("w") as err_f:
            result = subprocess.run(
                [
                    sys.executable,
                    str(llama_cpp / "convert_hf_to_gguf.py"),
                    str(merged_path),
                    "--outfile",
                    str(fp16_file),
                    "--outtype",
                    "f16",
                ],
                stdout=out_f,
                stderr=err_f,
                text=True,
            )
        if result.returncode != 0:
            print("GGUF conversion failed.")
            print(f"stdout: {log_out}")
            print(f"stderr: {log_err}")
            return

        # Quantize to 4, 6, 8 bit
        for quant_type in ["Q4_K_M", "Q6_K", "Q8_0"]:
            out_file = gguf_dir / f"{model_name}-{quant_type.lower()}.gguf"
            print(f"Quantizing {quant_type}: {out_file}")
            subprocess.run(
                [str(quantize_bin), str(fp16_file), str(out_file), quant_type],
                capture_output=True,
            )
            if out_file.exists():
                size_mb = out_file.stat().st_size / (1024 * 1024)
                print(f"  {quant_type}: {size_mb:.1f} MB")

        # Remove FP16 to save space
        if fp16_file.exists():
            fp16_file.unlink()

    print(f"GGUF files saved to: {gguf_dir}")


def run_eval(model_path: str) -> float | None:
    """Run eval.py on the trained model and return average score."""
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    eval_script = Path(__file__).parent / "eval.py"
    result = subprocess.run(
        [sys.executable, str(eval_script), model_path],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    avg = None
    for line in (result.stdout or "").splitlines():
        if line.strip().startswith("Average:"):
            try:
                avg = float(line.split("Average:", 1)[1].split("%", 1)[0].strip())
            except ValueError:
                pass
            break
    return avg


def cmd_sft(args):
    """Run supervised fine-tuning."""
    import torch
    import os
    from datasets import load_dataset
    import torch
    import torch.distributed as dist
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from trl import SFTTrainer, SFTConfig

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_LOG_CUDA_MEMORY", "0")

    if args.dry_run:
        print("SFT Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    dataset_name = cfg["dataset"]["name"]
    print(f"Loading dataset: {dataset_name}...")

    # Support local JSONL files and glob patterns
    if dataset_name.startswith("data/") or dataset_name.endswith(".jsonl"):
        from pathlib import Path
        import glob

        # Handle glob patterns like "data/*.jsonl"
        if "*" in dataset_name:
            jsonl_files = sorted(glob.glob(dataset_name))
            if not jsonl_files:
                raise ValueError(f"No files found matching: {dataset_name}")
            print(
                f"  Found {len(jsonl_files)} JSONL files: {[Path(f).name for f in jsonl_files]}"
            )
            dataset = load_dataset("json", data_files=jsonl_files, split="train")
        else:
            data_path = Path(dataset_name)
            if data_path.is_dir():
                train_file = data_path / "train.jsonl"
                dataset = load_dataset(
                    "json", data_files=str(train_file), split="train"
                )
            else:
                dataset = load_dataset("json", data_files=dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_name, split=cfg["dataset"]["split"])
    print(f"Dataset loaded: {len(dataset)} examples")

    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=cfg["dataset"]["eval_split"], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Check if output looks like a HF Hub path (contains /)
    output_name = cfg["model"]["output"]
    push_to_hub = "/" in output_name and not output_name.startswith("outputs/")
    if "push_to_hub" in cfg["model"]:
        push_to_hub = bool(cfg["model"]["push_to_hub"])
    output_dir = output_name.split("/")[-1] if push_to_hub else output_name

    report_to = "none"
    if os.environ.get("HF_TOKEN"):
        try:
            import trackio  # noqa: F401

            report_to = "trackio"
        except Exception:
            print("Trackio not installed; disabling tracking.")

    tracking = cfg.get("tracking", {})
    if report_to == "trackio":
        project = tracking.get("project")
        if project:
            os.environ.setdefault("TRACKIO_PROJECT", project)

    run_name = tracking.get("run_name")
    if run_name and "{" in run_name:
        from datetime import datetime

        now = datetime.now()
        run_name = run_name.replace("{day}", now.strftime("%b %d")).replace(
            "{time}", now.strftime("%H:%M")
        )

    config = SFTConfig(
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=output_name if push_to_hub else None,
        hub_strategy="every_save" if push_to_hub else "end",
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
        ddp_find_unused_parameters=cfg["training"].get(
            "ddp_find_unused_parameters", False
        ),
        bf16=True,
        report_to=report_to,
        run_name=run_name if report_to == "trackio" else None,
    )

    # LoRA config with modules_to_save for embedding layers
    # This prevents token ID mismatches during inference
    peft_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
        modules_to_save=["embed_tokens", "lm_head"],  # Critical for special tokens
        ensure_weight_tying=True,
    )

    print("Loading tokenizer...")
    base_model = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    is_main = os.environ.get("RANK", "0") == "0"
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main:
        return

    if push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()
        print(f"Done! Model: https://huggingface.co/{output_name}")
    else:
        trainer.save_model()
        print(f"Done! Model saved to: {output_dir}")

    # Export GGUF
    print("\nExporting to GGUF...")
    # Need to get the merged model for GGUF
    print("Loading model for GGUF export...")
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base.config.tie_word_embeddings = False
    model = PeftModel.from_pretrained(base, output_dir, local_files_only=True)
    model = model.merge_and_unload()
    export_gguf(model, tokenizer, output_dir, Path(output_dir).name)

    # Run eval
    eval_avg = run_eval(output_dir)
    if report_to == "trackio":
        try:
            import trackio

            if eval_avg is not None:
                trackio.log({"eval.avg": eval_avg})
        except Exception:
            pass


def cmd_grpo(args):
    """Run GRPO reinforcement learning on top of merged SFT weights."""
    import torch
    import os
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from trl import GRPOTrainer, GRPOConfig

    # Import reward from the shared module
    sys.path.insert(0, os.path.dirname(__file__))
    from reward import QMDRewardFunction

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_LOG_CUDA_MEMORY", "0")

    if args.dry_run:
        print("GRPO Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    # Tracking
    report_to = "none"
    if os.environ.get("HF_TOKEN"):
        try:
            import trackio  # noqa: F401

            report_to = "trackio"
        except Exception:
            print("Trackio not installed; disabling tracking.")

    tracking = cfg.get("tracking", {})
    if report_to == "trackio":
        project = tracking.get("project")
        if project:
            os.environ.setdefault("TRACKIO_PROJECT", project)

    run_name = tracking.get("run_name")
    if run_name and "{" in run_name:
        from datetime import datetime

        now = datetime.now()
        run_name = run_name.replace("{day}", now.strftime("%b %d")).replace(
            "{time}", now.strftime("%H:%M")
        )

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
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    max_samples = cfg["dataset"].get("max_samples", len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base model, merge SFT adapter
    sft_model_name = cfg["model"]["sft"]
    print(f"Loading SFT model from {sft_model_name}...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if available == 0:
            raise RuntimeError("CUDA is available but no devices were detected.")
        if local_rank >= available:
            print(
                f"Warning: LOCAL_RANK={local_rank} but only {available} CUDA device(s) visible. "
                "Falling back to the last available device."
            )
            local_rank = available - 1
        torch.cuda.set_device(local_rank)
    dtype_name = cfg["model"].get("torch_dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": {"": local_rank} if torch.cuda.is_available() else "auto",
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, sft_model_name)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Add fresh LoRA for GRPO with modules_to_save
    grpo_lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
        modules_to_save=["embed_tokens", "lm_head"],  # Critical for special tokens
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()

    # Build GRPO config
    output_name = cfg["model"]["output"]
    push_to_hub = "/" in output_name and not output_name.startswith("outputs/")
    if "push_to_hub" in cfg["model"]:
        push_to_hub = bool(cfg["model"]["push_to_hub"])
    output_dir = output_name.split("/")[-1] if push_to_hub else output_name

    grpo_cfg = cfg.get("grpo", {})
    learning_rate = cfg["training"]["learning_rate"]
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)

    config = GRPOConfig(
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=output_name if push_to_hub else None,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 200),
        beta=grpo_cfg.get("beta", 0.04),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        max_steps=cfg["training"].get("max_steps", -1),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        skip_memory_metrics=True,
        report_to=report_to,
        run_name=run_name if report_to == "trackio" else None,
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

    if push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()

    trainer.save_model()
    if report_to == "trackio":
        try:
            import trackio

            trackio.finish()
        except Exception:
            pass
    print(f"Done! Model saved to: {output_dir}")

    # Export GGUF
    print("\nExporting to GGUF...")
    merged = model.merge_and_unload()
    export_gguf(merged, tokenizer, output_dir, Path(output_dir).name)

    # Run eval
    eval_avg = run_eval(output_dir)
    if report_to == "trackio" and eval_avg is not None:
        try:
            import trackio

            trackio.log({"eval.avg": eval_avg})
        except Exception:
            pass


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
    sft_parser.add_argument(
        "--dry-run", action="store_true", help="Print config and exit"
    )

    grpo_parser = sub.add_parser("grpo", help="GRPO reinforcement learning")
    grpo_parser.add_argument("--config", required=True, help="Path to GRPO config YAML")
    grpo_parser.add_argument(
        "--dry-run", action="store_true", help="Print config, test reward, and exit"
    )

    args = parser.parse_args()

    if args.stage == "sft":
        cmd_sft(args)
    elif args.stage == "grpo":
        cmd_grpo(args)


if __name__ == "__main__":
    main()
