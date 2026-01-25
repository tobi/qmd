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
GGUF Conversion for QMD Query Expansion 4B Model

Loads base model, applies SFT adapter, then GRPO adapter, merges all,
and converts to GGUF format for use with Ollama/llama.cpp/LM Studio.
"""

import os
import sys
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

# Configuration
BASE_MODEL = "Qwen/Qwen3-4B"
SFT_MODEL = "tobil/qmd-query-expansion-4B-sft"
GRPO_MODEL = "tobil/qmd-query-expansion-4B-grpo"
OUTPUT_REPO = "tobil/qmd-query-expansion-4B-gguf"

def run_command(cmd, description):
    """Run a command with error handling."""
    print(f"   {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"   STDERR: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False


print("🔄 QMD Query Expansion 4B GGUF Conversion")
print("=" * 60)

# Install build tools
print("\n📦 Installing build dependencies...")
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "git"], capture_output=True)
print("   ✅ Build tools ready")

# Login to HuggingFace
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("\n🔐 Logging in to HuggingFace...")
    login(token=hf_token)
    print("   ✅ Logged in")

# Step 1: Load base model
print(f"\n🔧 Step 1: Loading base model {BASE_MODEL}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("   ✅ Base model loaded")

# Step 2: Load and merge SFT adapter
print(f"\n🔧 Step 2: Loading SFT adapter {SFT_MODEL}...")
model = PeftModel.from_pretrained(base_model, SFT_MODEL)
print("   Merging SFT adapter...")
model = model.merge_and_unload()
print("   ✅ SFT merged")

# Step 3: Load and merge GRPO adapter
print(f"\n🔧 Step 3: Loading GRPO adapter {GRPO_MODEL}...")
model = PeftModel.from_pretrained(model, GRPO_MODEL)
print("   Merging GRPO adapter...")
merged_model = model.merge_and_unload()
print("   ✅ GRPO merged - final model ready")

# Load tokenizer
print("\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
print("   ✅ Tokenizer loaded")

# Step 4: Save merged model
print("\n💾 Step 4: Saving merged model to disk...")
merged_dir = "/tmp/merged_model"
merged_model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print(f"   ✅ Saved to {merged_dir}")

# Step 5: Setup llama.cpp
print("\n📥 Step 5: Setting up llama.cpp...")
if not os.path.exists("/tmp/llama.cpp"):
    run_command(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
        "Cloning llama.cpp"
    )

# Install Python deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/llama.cpp/requirements.txt"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"], capture_output=True)
print("   ✅ llama.cpp ready")

# Step 6: Convert to GGUF (FP16)
print("\n🔄 Step 6: Converting to GGUF format (FP16)...")
gguf_output_dir = "/tmp/gguf_output"
os.makedirs(gguf_output_dir, exist_ok=True)

model_name = "qmd-query-expansion-4B"
gguf_file = f"{gguf_output_dir}/{model_name}-f16.gguf"

convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
if not run_command(
    [sys.executable, convert_script, merged_dir, "--outfile", gguf_file, "--outtype", "f16"],
    "Converting to FP16 GGUF"
):
    print("   ❌ Conversion failed!")
    sys.exit(1)

size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
print(f"   ✅ FP16 GGUF created: {size_mb:.1f} MB")

# Step 7: Build quantize tool
print("\n⚙️  Step 7: Building quantize tool...")
os.makedirs("/tmp/llama.cpp/build", exist_ok=True)

run_command(
    ["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"],
    "Configuring with CMake"
)
run_command(
    ["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
    "Building llama-quantize"
)

quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
print("   ✅ Quantize tool built")

# Step 8: Create quantized versions
print("\n⚙️  Step 8: Creating quantized versions...")
quant_formats = [
    ("Q4_K_M", "4-bit medium (recommended)"),
    ("Q5_K_M", "5-bit medium"),
    ("Q8_0", "8-bit"),
]

quantized_files = []
for quant_type, description in quant_formats:
    print(f"   Creating {quant_type} ({description})...")
    quant_file = f"{gguf_output_dir}/{model_name}-{quant_type.lower()}.gguf"

    if run_command([quantize_bin, gguf_file, quant_file, quant_type], f"Quantizing to {quant_type}"):
        size_mb = os.path.getsize(quant_file) / (1024 * 1024)
        print(f"   ✅ {quant_type}: {size_mb:.1f} MB")
        quantized_files.append((quant_file, quant_type))
    else:
        print(f"   ⚠️  Skipping {quant_type}")

# Step 9: Upload to Hub
print("\n☁️  Step 9: Uploading to Hugging Face Hub...")
api = HfApi()

print(f"   Creating repository: {OUTPUT_REPO}")
api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True)

# Upload F16
print("   Uploading FP16...")
api.upload_file(
    path_or_fileobj=gguf_file,
    path_in_repo=f"{model_name}-f16.gguf",
    repo_id=OUTPUT_REPO,
)
print("   ✅ FP16 uploaded")

# Upload quantized versions
for quant_file, quant_type in quantized_files:
    print(f"   Uploading {quant_type}...")
    api.upload_file(
        path_or_fileobj=quant_file,
        path_in_repo=f"{model_name}-{quant_type.lower()}.gguf",
        repo_id=OUTPUT_REPO,
    )
    print(f"   ✅ {quant_type} uploaded")

# Create README
print("\n📝 Creating README...")
readme_content = f"""---
base_model: {BASE_MODEL}
tags:
- gguf
- llama.cpp
- quantized
- query-expansion
- qmd
---

# QMD Query Expansion 4B (GGUF)

GGUF conversion of the QMD Query Expansion model for use with Ollama, llama.cpp, and LM Studio.

## Model Details

- **Base Model:** {BASE_MODEL}
- **SFT Adapter:** {SFT_MODEL}
- **GRPO Adapter:** {GRPO_MODEL}
- **Task:** Query expansion for hybrid search (lex/vec/hyde format)

## Available Quantizations

| File | Quant | Description |
|------|-------|-------------|
| {model_name}-f16.gguf | F16 | Full precision |
| {model_name}-q8_0.gguf | Q8_0 | 8-bit |
| {model_name}-q5_k_m.gguf | Q5_K_M | 5-bit medium |
| {model_name}-q4_k_m.gguf | Q4_K_M | 4-bit medium (recommended) |

## Usage

### With Ollama

```bash
# Download
huggingface-cli download {OUTPUT_REPO} {model_name}-q4_k_m.gguf --local-dir .

# Create Modelfile
echo 'FROM ./{model_name}-q4_k_m.gguf' > Modelfile

# Create and run
ollama create qmd-expand-4b -f Modelfile
ollama run qmd-expand-4b
```

### Prompt Format

Use Qwen3 chat format with `/no_think`:

```
<|im_start|>user
/no_think Expand this search query: your query here<|im_end|>
<|im_start|>assistant
```

### Expected Output

```
lex: keyword variation 1
lex: keyword variation 2
vec: natural language reformulation
hyde: Hypothetical document passage answering the query.
```

## License

Apache 2.0 (inherited from Qwen3)
"""

api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id=OUTPUT_REPO,
)
print("   ✅ README uploaded")

print("\n" + "=" * 60)
print("✅ GGUF Conversion Complete!")
print(f"📦 Repository: https://huggingface.co/{OUTPUT_REPO}")
print("=" * 60)
