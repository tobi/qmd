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
#     "optimum>=1.17.0",
#     "onnx>=1.15.0",
#     "onnxruntime>=1.17.0",
# ]
# ///
"""
Convert QMD query expansion model to ONNX format for Transformers.js.

Loads the base model, merges SFT and GRPO adapters, then exports to ONNX
with quantization for browser deployment via Transformers.js + WebGPU.

Usage:
    uv run convert_onnx.py --size 1.7B
    uv run convert_onnx.py --size 1.7B --no-upload
    uv run convert_onnx.py --base Qwen/Qwen3-1.7B \
                           --sft tobil/qmd-query-expansion-1.7B-sft \
                           --grpo tobil/qmd-query-expansion-1.7B-grpo \
                           --output tobil/qmd-query-expansion-1.7B-ONNX

Quantization options:
    --quantize q4    MatMulNBits 4-bit (default, smallest)
    --quantize q8    8-bit dynamic quantization
    --quantize fp16  FP16 (requires GPU export)
    --quantize none  No quantization (FP32, ~7GB)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PRESETS = {
    "1.7B": {
        "base": "Qwen/Qwen3-1.7B",
        "sft": "tobil/qmd-query-expansion-1.7B-sft",
        "grpo": "tobil/qmd-query-expansion-1.7B-grpo",
        "output": "tobil/qmd-query-expansion-1.7B-ONNX",
    },
    "4B": {
        "base": "Qwen/Qwen3-4B",
        "sft": "tobil/qmd-query-expansion-4B-sft",
        "grpo": "tobil/qmd-query-expansion-4B-grpo",
        "output": "tobil/qmd-query-expansion-4B-ONNX",
    },
}


def merge_adapters(base_model: str, sft_model: str, grpo_model: str) -> tuple:
    """Load base model, merge SFT + GRPO adapters, return (model, tokenizer)."""
    print(f"\nStep 1: Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=True,
    )

    print(f"Step 2: Merging SFT adapter {sft_model}...")
    model = PeftModel.from_pretrained(model, sft_model)
    model = model.merge_and_unload()

    print(f"Step 3: Merging GRPO adapter {grpo_model}...")
    model = PeftModel.from_pretrained(model, grpo_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return model, tokenizer


def export_onnx(model, tokenizer, output_dir: str):
    """Export merged model to ONNX using Optimum."""
    from optimum.exporters.onnx import main_export

    # Save merged model to temp dir first (Optimum needs HF format on disk)
    merged_dir = "/tmp/merged_model_onnx"
    print(f"\nStep 4: Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nStep 5: Exporting to ONNX at {output_dir}...")
    main_export(
        model_name_or_path=merged_dir,
        output=output_dir,
        task="text-generation-with-past",
        device="cpu",
        fp16=False,
    )

    # Clean up temp merged dir
    shutil.rmtree(merged_dir, ignore_errors=True)


def quantize_onnx(onnx_dir: str, quantize_type: str):
    """Quantize the exported ONNX model."""
    if quantize_type == "none":
        print("\nSkipping quantization (FP32).")
        return

    model_path = Path(onnx_dir) / "model.onnx"
    if not model_path.exists():
        # Optimum may produce decoder_model.onnx for text-generation-with-past
        candidates = list(Path(onnx_dir).glob("*.onnx"))
        if not candidates:
            print("  WARNING: No .onnx files found to quantize.")
            return
        model_path = candidates[0]

    print(f"\nStep 6: Quantizing {model_path.name} ({quantize_type})...")

    if quantize_type == "q4":
        try:
            from onnxruntime.quantization import matmul_nbits_quantizer
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model=str(model_path),
                block_size=32,
                is_symmetric=True,
                bits=4,
            )
            quant.process()
            q_path = model_path.with_name(
                model_path.stem + "_q4" + model_path.suffix,
            )
            quant.model.save(str(q_path))
            size_mb = q_path.stat().st_size / (1024 * 1024)
            print(f"  Q4: {size_mb:.1f} MB -> {q_path.name}")
        except ImportError:
            print("  WARNING: onnxruntime quantization not available, trying alternative...")
            _quantize_dynamic(model_path, quantize_type)

    elif quantize_type == "q8":
        _quantize_dynamic(model_path, quantize_type)

    elif quantize_type == "fp16":
        _convert_fp16(model_path)


def _quantize_dynamic(model_path: Path, qtype: str):
    """Dynamic quantization fallback."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    weight_type = QuantType.QUInt8 if qtype == "q8" else QuantType.QInt8
    q_path = model_path.with_name(
        model_path.stem + f"_{qtype}" + model_path.suffix,
    )
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(q_path),
        weight_type=weight_type,
    )
    size_mb = q_path.stat().st_size / (1024 * 1024)
    print(f"  {qtype.upper()}: {size_mb:.1f} MB -> {q_path.name}")


def _convert_fp16(model_path: Path):
    """Convert ONNX model to FP16."""
    import onnx
    from onnx import numpy_helper

    print("  Converting to FP16...")
    model = onnx.load(str(model_path))
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            np_data = numpy_helper.to_array(initializer)
            initializer.CopyFrom(
                numpy_helper.from_array(np_data.astype("float16"), initializer.name),
            )
    fp16_path = model_path.with_name(
        model_path.stem + "_fp16" + model_path.suffix,
    )
    onnx.save(model, str(fp16_path))
    size_mb = fp16_path.stat().st_size / (1024 * 1024)
    print(f"  FP16: {size_mb:.1f} MB -> {fp16_path.name}")


def write_transformers_js_config(onnx_dir: str):
    """Write Transformers.js compatibility config."""
    config_path = Path(onnx_dir) / "transformers_js_config.json"
    config = {
        "model_type": "text-generation",
        "quantized": True,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Wrote {config_path.name}")


def upload_to_hub(
    onnx_dir: str,
    output_repo: str,
    base_model: str,
    sft_model: str,
    grpo_model: str,
):
    """Upload ONNX model to HuggingFace Hub."""
    print(f"\nStep 7: Uploading to {output_repo}...")
    api = HfApi()
    api.create_repo(repo_id=output_repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=onnx_dir,
        repo_id=output_repo,
        commit_message="Upload ONNX model",
    )

    readme = f"""---
base_model: {base_model}
tags: [onnx, transformers.js, webgpu, query-expansion, qmd]
library_name: transformers.js
---
# {output_repo.split("/")[-1]}

ONNX conversion of the QMD Query Expansion model for use with
[Transformers.js](https://huggingface.co/docs/transformers.js) and WebGPU.

## Details
- **Base:** {base_model}
- **SFT:** {sft_model}
- **GRPO:** {grpo_model}
- **Task:** Query expansion (lex/vec/hyde format)
- **Format:** ONNX with Q4 quantization

## Usage with Transformers.js

```javascript
import {{ AutoTokenizer, AutoModelForCausalLM }} from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("{output_repo}");
const model = await AutoModelForCausalLM.from_pretrained("{output_repo}", {{
  dtype: "q4",
  device: "webgpu",
}});
```

## Prompt Format
```
<|im_start|>user
/no_think Expand this search query: your query here<|im_end|>
<|im_start|>assistant
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=output_repo,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert QMD model to ONNX")
    parser.add_argument(
        "--size", choices=PRESETS.keys(), help="Use preset config for model size",
    )
    parser.add_argument("--base", help="Base model (overrides preset)")
    parser.add_argument("--sft", help="SFT adapter (overrides preset)")
    parser.add_argument("--grpo", help="GRPO adapter (overrides preset)")
    parser.add_argument("--output", help="Output HF repo (overrides preset)")
    parser.add_argument(
        "--quantize",
        choices=["q4", "q8", "fp16", "none"],
        default="q4",
        help="Quantization type (default: q4)",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Don't upload to HF Hub",
    )
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
        parser.error(
            "Either --size or all of --base/--sft/--grpo/--output are required",
        )

    model_name = output_repo.split("/")[-1]
    print(f"QMD ONNX Conversion: {model_name}")
    print("=" * 60)

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace...")
        login(token=hf_token)

    # Merge adapters
    model, tokenizer = merge_adapters(base_model, sft_model, grpo_model)

    # Export to ONNX
    onnx_dir = f"/tmp/onnx_output/{model_name}"
    os.makedirs(onnx_dir, exist_ok=True)
    export_onnx(model, tokenizer, onnx_dir)

    # Quantize
    quantize_onnx(onnx_dir, args.quantize)

    # Write Transformers.js config
    write_transformers_js_config(onnx_dir)

    # Upload
    if not args.no_upload:
        upload_to_hub(onnx_dir, output_repo, base_model, sft_model, grpo_model)

    print(f"\nDone! ONNX files at: {onnx_dir}")
    if not args.no_upload:
        print(f"Repository: https://huggingface.co/{output_repo}")


if __name__ == "__main__":
    main()
