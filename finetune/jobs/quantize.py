# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub>=0.20.0",
#     "accelerate",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///
"""
Merge SFT + GRPO adapters and convert to GGUF with multiple quantizations.

Uploads each quantization to HuggingFace Hub as it's produced, so partial
results are available even if the job times out.

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/quantize.py
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/quantize.py -- --size 4B
"""

import argparse
import os
import subprocess
import sys

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PRESETS = {
    "1.7B": {
        "base": "Qwen/Qwen3-1.7B",
        "sft": "tobil/qmd-query-expansion-1.7B-sft",
        "grpo": "tobil/qmd-query-expansion-1.7B-grpo",
        "output": "tobil/qmd-query-expansion-1.7B-gguf",
    },
    "4B": {
        "base": "Qwen/Qwen3-4B",
        "sft": "tobil/qmd-query-expansion-4B-sft",
        "grpo": "tobil/qmd-query-expansion-4B-grpo",
        "output": "tobil/qmd-query-expansion-4B-gguf",
    },
}

QUANT_TYPES = [
    ("Q4_K_M", "4-bit (recommended for most use)"),
    ("Q5_K_M", "5-bit (balanced quality/size)"),
    ("Q8_0", "8-bit (highest quality)"),
]


def run_cmd(cmd, description):
    print(f"  {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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
    parser.add_argument("--size", default="1.7B", choices=PRESETS.keys(), help="Model size preset")
    args = parser.parse_args()

    preset = PRESETS[args.size]
    base_model = preset["base"]
    sft_model = preset["sft"]
    grpo_model = preset["grpo"]
    output_repo = preset["output"]
    model_name = output_repo.split("/")[-1].replace("-gguf", "")

    print(f"QMD GGUF Conversion: {model_name}")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    api = HfApi()
    api.create_repo(repo_id=output_repo, repo_type="model", exist_ok=True)

    # Step 1: Install build tools
    print("\nStep 1: Installing build dependencies...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "git"], capture_output=True)

    # Step 2: Load and merge
    print(f"\nStep 2: Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    print(f"Step 3: Merging SFT adapter {sft_model}...")
    model = PeftModel.from_pretrained(model, sft_model)
    model = model.merge_and_unload()

    print(f"Step 4: Merging GRPO adapter {grpo_model}...")
    model = PeftModel.from_pretrained(model, grpo_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Step 3: Save merged model
    merged_dir = "/tmp/merged_model"
    print(f"\nStep 5: Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del model
    torch.cuda.empty_cache()

    # Step 4: Setup llama.cpp
    print("\nStep 6: Setting up llama.cpp...")
    if not os.path.exists("/tmp/llama.cpp"):
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
                "Cloning llama.cpp")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/llama.cpp/requirements.txt"],
                   capture_output=True)

    # Step 5: Convert to FP16 GGUF
    gguf_dir = "/tmp/gguf_output"
    os.makedirs(gguf_dir, exist_ok=True)
    fp16_file = f"{gguf_dir}/{model_name}-f16.gguf"

    print(f"\nStep 7: Converting to FP16 GGUF...")
    if not run_cmd([sys.executable, "/tmp/llama.cpp/convert_hf_to_gguf.py",
                    merged_dir, "--outfile", fp16_file, "--outtype", "f16"],
                   "Converting to FP16"):
        sys.exit(1)

    size_mb = os.path.getsize(fp16_file) / (1024 * 1024)
    print(f"  FP16: {size_mb:.1f} MB")

    # Upload FP16 immediately
    print(f"  Uploading FP16 to {output_repo}...")
    api.upload_file(path_or_fileobj=fp16_file,
                    path_in_repo=f"{model_name}-f16.gguf", repo_id=output_repo)
    print(f"  Uploaded: {model_name}-f16.gguf")

    # Step 6: Build quantize tool
    print("\nStep 8: Building quantize tool...")
    os.makedirs("/tmp/llama.cpp/build", exist_ok=True)
    run_cmd(["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"],
            "CMake configure")
    run_cmd(["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
            "Building llama-quantize")
    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"

    # Step 7: Quantize and upload each one immediately
    print("\nStep 9: Quantizing and uploading...")
    for quant_type, desc in QUANT_TYPES:
        qfile = f"{gguf_dir}/{model_name}-{quant_type.lower()}.gguf"
        if run_cmd([quantize_bin, fp16_file, qfile, quant_type], f"{quant_type} ({desc})"):
            qsize = os.path.getsize(qfile) / (1024 * 1024)
            print(f"  {quant_type}: {qsize:.1f} MB")

            print(f"  Uploading {quant_type} to {output_repo}...")
            api.upload_file(path_or_fileobj=qfile,
                            path_in_repo=f"{model_name}-{quant_type.lower()}.gguf", repo_id=output_repo)
            print(f"  Uploaded: {model_name}-{quant_type.lower()}.gguf")

            # Remove to save disk
            os.remove(qfile)

    # Step 8: Upload README
    ollama_name = "qmd-expand" if args.size == "1.7B" else f"qmd-expand-{args.size.lower()}"
    readme = f"""---
base_model: {base_model}
tags: [gguf, llama.cpp, quantized, query-expansion, qmd]
---
# {model_name} (GGUF)

GGUF quantizations of the QMD Query Expansion model for use with
[Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp),
or [LM Studio](https://lmstudio.ai).

## Available Quantizations

| File | Quant | Description |
|------|-------|-------------|
| `{model_name}-q4_k_m.gguf` | Q4_K_M | 4-bit — smallest, recommended for most use |
| `{model_name}-q5_k_m.gguf` | Q5_K_M | 5-bit — balanced quality/size |
| `{model_name}-q8_0.gguf` | Q8_0 | 8-bit — highest quality |
| `{model_name}-f16.gguf` | FP16 | Full precision (large) |

## Details

- **Base:** {base_model}
- **SFT:** {sft_model}
- **GRPO:** {grpo_model}
- **Task:** Query expansion for hybrid search (lex/vec/hyde format)
- **Eval score:** 90.7% average (29/30 Excellent)

## Quick Start with Ollama

```bash
huggingface-cli download {output_repo} \\
    {model_name}-q4_k_m.gguf --local-dir .

echo 'FROM ./{model_name}-q4_k_m.gguf' > Modelfile
ollama create {ollama_name} -f Modelfile
ollama run {ollama_name}
```

## Prompt Format

```
<|im_start|>user
/no_think Expand this search query: your query here<|im_end|>
<|im_start|>assistant
```

The model produces structured output:
```
lex: keyword expansion for BM25 search
lex: another keyword variant
vec: natural language expansion for vector search
vec: another semantic expansion
hyde: A hypothetical document passage that might match this query.
```
"""
    api.upload_file(path_or_fileobj=readme.encode(),
                    path_in_repo="README.md", repo_id=output_repo)

    print(f"\nDone! Repository: https://huggingface.co/{output_repo}")
    print(f"\nTo use with Ollama:")
    print(f"  huggingface-cli download {output_repo} {model_name}-q4_k_m.gguf --local-dir .")
    print(f"  echo 'FROM ./{model_name}-q4_k_m.gguf' > Modelfile")
    print(f"  ollama create {ollama_name} -f Modelfile")


if __name__ == "__main__":
    main()
