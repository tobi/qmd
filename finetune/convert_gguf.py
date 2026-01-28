#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///
"""
Convert QMD query expansion model to GGUF format.

Loads the base model, merges SFT and GRPO adapters, then converts to
GGUF with multiple quantizations for use with Ollama/llama.cpp/LM Studio.

Usage:
    uv run convert_gguf.py --size 1.7B
    uv run convert_gguf.py --size 4B --skip-quantize
    uv run convert_gguf.py --base Qwen/Qwen3-1.7B \
                           --sft tobil/qmd-query-expansion-1.7B-sft \
                           --grpo tobil/qmd-query-expansion-1.7B-grpo \
                           --output tobil/qmd-query-expansion-1.7B-gguf
"""

import argparse
import os
import subprocess
import sys

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Preset configurations for each model size
PRESETS = {
    "1.7B": {
        "base": "Qwen/Qwen3-1.7B",
        "sft": "tobil/qmd-query-expansion-1.7B-sft",
        "grpo": "tobil/qmd-query-expansion-1.7B-grpo",
        "output": "tobil/qmd-query-expansion-1.7B-gguf",
        "ollama_name": "qmd-expand",
    },
    "4B": {
        "base": "Qwen/Qwen3-4B",
        "sft": "tobil/qmd-query-expansion-4B-sft",
        "grpo": "tobil/qmd-query-expansion-4B-grpo",
        "output": "tobil/qmd-query-expansion-4B-gguf",
        "ollama_name": "qmd-expand-4b",
    },
}


def run_cmd(cmd, description):
    """Run a shell command with error handling."""
    print(f"  {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {' '.join(cmd)}")
        if e.stderr:
            print(f"  {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print(f"  Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert QMD model to GGUF")
    parser.add_argument("--size", choices=PRESETS.keys(), help="Use preset config for model size")
    parser.add_argument("--base", help="Base model (overrides preset)")
    parser.add_argument("--sft", help="SFT adapter (overrides preset)")
    parser.add_argument("--grpo", help="GRPO adapter (overrides preset)")
    parser.add_argument("--output", help="Output HF repo (overrides preset)")
    parser.add_argument("--skip-quantize", action="store_true", help="Only produce FP16 GGUF")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload to HF Hub")
    args = parser.parse_args()

    # Resolve config
    if args.size:
        preset = PRESETS[args.size]
        base_model = args.base or preset["base"]
        sft_model = args.sft or preset["sft"]
        grpo_model = args.grpo or preset["grpo"]
        output_repo = args.output or preset["output"]
    elif args.base and args.sft and args.grpo and args.output:
        base_model = args.base
        sft_model = args.sft
        grpo_model = args.grpo
        output_repo = args.output
    else:
        parser.error("Either --size or all of --base/--sft/--grpo/--output are required")

    model_name = output_repo.split("/")[-1].replace("-gguf", "")
    print(f"QMD GGUF Conversion: {model_name}")
    print("=" * 60)

    # Install build tools (for Colab/cloud environments)
    print("\nInstalling build dependencies...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "git"], capture_output=True)

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace...")
        login(token=hf_token)

    # Step 1: Load and merge
    print(f"\nStep 1: Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    print(f"Step 2: Merging SFT adapter {sft_model}...")
    model = PeftModel.from_pretrained(model, sft_model)
    model = model.merge_and_unload()

    print(f"Step 3: Merging GRPO adapter {grpo_model}...")
    model = PeftModel.from_pretrained(model, grpo_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Step 2: Save merged model
    merged_dir = "/tmp/merged_model"
    print(f"\nStep 4: Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Step 3: Setup llama.cpp
    print("\nStep 5: Setting up llama.cpp...")
    if not os.path.exists("/tmp/llama.cpp"):
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
                "Cloning llama.cpp")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/llama.cpp/requirements.txt"],
                   capture_output=True)

    # Step 4: Convert to FP16 GGUF
    gguf_dir = "/tmp/gguf_output"
    os.makedirs(gguf_dir, exist_ok=True)
    gguf_file = f"{gguf_dir}/{model_name}-f16.gguf"

    print(f"\nStep 6: Converting to FP16 GGUF...")
    if not run_cmd([sys.executable, "/tmp/llama.cpp/convert_hf_to_gguf.py",
                    merged_dir, "--outfile", gguf_file, "--outtype", "f16"],
                   "Converting"):
        sys.exit(1)

    size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
    print(f"  FP16: {size_mb:.1f} MB")

    # Step 5: Quantize
    quantized_files = []
    if not args.skip_quantize:
        print("\nStep 7: Building quantize tool...")
        os.makedirs("/tmp/llama.cpp/build", exist_ok=True)
        run_cmd(["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"],
                "CMake configure")
        run_cmd(["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
                "Building llama-quantize")
        quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"

        print("\nStep 8: Quantizing...")
        for quant_type, desc in [("Q4_K_M", "4-bit"), ("Q5_K_M", "5-bit"), ("Q8_0", "8-bit")]:
            qfile = f"{gguf_dir}/{model_name}-{quant_type.lower()}.gguf"
            if run_cmd([quantize_bin, gguf_file, qfile, quant_type], f"{quant_type} ({desc})"):
                qsize = os.path.getsize(qfile) / (1024 * 1024)
                print(f"  {quant_type}: {qsize:.1f} MB")
                quantized_files.append((qfile, quant_type))

    # Step 6: Upload
    if not args.no_upload:
        print(f"\nStep 9: Uploading to {output_repo}...")
        api = HfApi()
        api.create_repo(repo_id=output_repo, repo_type="model", exist_ok=True)

        api.upload_file(path_or_fileobj=gguf_file,
                        path_in_repo=f"{model_name}-f16.gguf", repo_id=output_repo)
        for qfile, qtype in quantized_files:
            api.upload_file(path_or_fileobj=qfile,
                            path_in_repo=f"{model_name}-{qtype.lower()}.gguf", repo_id=output_repo)

        # Upload README
        readme = f"""---
base_model: {base_model}
tags: [gguf, llama.cpp, quantized, query-expansion, qmd]
---
# {model_name} (GGUF)

GGUF conversion of the QMD Query Expansion model.

## Details
- **Base:** {base_model}
- **SFT:** {sft_model}
- **GRPO:** {grpo_model}
- **Task:** Query expansion (lex/vec/hyde format)

## Prompt Format
```
<|im_start|>user
/no_think Expand this search query: your query here<|im_end|>
<|im_start|>assistant
```
"""
        api.upload_file(path_or_fileobj=readme.encode(),
                        path_in_repo="README.md", repo_id=output_repo)

    print(f"\nDone! Repository: https://huggingface.co/{output_repo}")


if __name__ == "__main__":
    main()
