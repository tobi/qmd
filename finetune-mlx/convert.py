#!/usr/bin/env python3
"""
Convert trained MLX model to GGUF for Ollama/llama.cpp.

Full pipeline:
1. Merge LoRA adapter into base model
2. Dequantize if needed (MLX 4-bit → FP16)
3. Convert to GGUF (requires llama.cpp)
4. Quantize to Q4_K_M (optional, requires llama-quantize)

Usage:
    python convert.py                           # Full pipeline with defaults
    python convert.py --output qmd-expand       # Custom output name
    python convert.py --quantize q4_k_m         # Include quantization step
    python convert.py --skip-merge              # Use already-merged model

Requirements:
    pip install mlx_lm gguf
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && mkdir build && cd build && cmake .. && make llama-quantize
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_llama_cpp():
    """Find llama.cpp installation."""
    # Check common locations
    locations = [
        Path.home() / "src" / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/usr/local/llama.cpp"),
        Path("../llama.cpp"),
    ]
    
    for loc in locations:
        convert_script = loc / "convert_hf_to_gguf.py"
        if convert_script.exists():
            return loc
    
    # Check if in PATH
    if shutil.which("llama-quantize"):
        return Path(shutil.which("llama-quantize")).parent.parent
    
    return None


def merge_adapter(model_path: Path, adapter_path: Path, output_path: Path):
    """Merge LoRA adapter into base model using mlx_lm fuse."""
    print(f"🔄 Merging adapter into base model...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(model_path),
        "--adapter-path", str(adapter_path),
        "--save-path", str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Merge failed: {result.stderr}")
        sys.exit(1)
    
    print(f"✅ Merged model saved to: {output_path}")
    return output_path


def dequantize_model(model_path: Path, output_path: Path):
    """Dequantize MLX model from 4-bit to FP16."""
    print(f"🔄 Dequantizing model to FP16...")
    
    # Check if model is quantized
    config_path = model_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        if "quantization" not in config and "quantization_config" not in config:
            print(f"ℹ️  Model is not quantized, skipping dequantization")
            return model_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use mlx_lm convert with dequantize flag
    cmd = [
        sys.executable, "-c",
        f"""
from mlx_lm import convert
convert(
    hf_path='{model_path}',
    mlx_path='{output_path}',
    dequantize=True,
    dtype='float16'
)
"""
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Dequantization failed: {result.stderr}")
        sys.exit(1)
    
    print(f"✅ Dequantized model saved to: {output_path}")
    return output_path


def convert_to_gguf(model_path: Path, output_path: Path, llama_cpp_path: Path):
    """Convert HF/MLX model to GGUF format."""
    print(f"🔄 Converting to GGUF...")
    
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ GGUF conversion failed")
        sys.exit(1)
    
    print(f"✅ GGUF saved to: {output_path}")
    return output_path


def quantize_gguf(input_path: Path, output_path: Path, quant_type: str, llama_cpp_path: Path):
    """Quantize GGUF model."""
    print(f"🔄 Quantizing to {quant_type}...")
    
    # Find llama-quantize
    quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = shutil.which("llama-quantize")
    
    if not quantize_bin:
        print(f"⚠️  llama-quantize not found, skipping quantization")
        print(f"   Build it: cd llama.cpp && mkdir build && cd build && cmake .. && make llama-quantize")
        return input_path
    
    cmd = [str(quantize_bin), str(input_path), str(output_path), quant_type.upper()]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Quantization failed")
        return input_path
    
    print(f"✅ Quantized GGUF saved to: {output_path}")
    return output_path


def create_modelfile(output_name: str, gguf_filename: str):
    """Create Ollama Modelfile."""
    modelfile = Path("exports") / f"{output_name}.Modelfile"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    
    content = f'''# Modelfile for {output_name}
# Usage: ollama create {output_name} -f exports/{output_name}.Modelfile

FROM ./{gguf_filename}

TEMPLATE """<|im_start|>user
/no_think Expand this search query: {{{{.Prompt}}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
'''
    
    with open(modelfile, "w") as f:
        f.write(content)
    
    print(f"✅ Modelfile saved to: {modelfile}")
    return modelfile


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLX model to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert.py
    python convert.py --quantize q4_k_m
    python convert.py --model models/Qwen_Qwen3-1.7B/mlx --adapter adapters/qmd_query_expansion_1.7B_sft
"""
    )
    parser.add_argument("--model", default="models/Qwen_Qwen3-1.7B/mlx", help="Base model path")
    parser.add_argument("--adapter", default="adapters/qmd_query_expansion_1.7B_sft", help="LoRA adapter path")
    parser.add_argument("--output", "-o", default="qmd-query-expand", help="Output model name")
    parser.add_argument("--quantize", "-q", help="Quantization type (e.g., q4_k_m, q8_0)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip adapter merge (use pre-merged model)")
    parser.add_argument("--llama-cpp", help="Path to llama.cpp directory")
    
    args = parser.parse_args()
    
    # Find llama.cpp
    llama_cpp_path = Path(args.llama_cpp) if args.llama_cpp else find_llama_cpp()
    if not llama_cpp_path:
        print("❌ llama.cpp not found. Clone it first:")
        print("   git clone https://github.com/ggerganov/llama.cpp ~/src/llama.cpp")
        sys.exit(1)
    
    print(f"📍 Using llama.cpp at: {llama_cpp_path}")
    
    model_path = Path(args.model)
    adapter_path = Path(args.adapter)
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    # Step 1: Merge adapter
    if not args.skip_merge and adapter_path.exists():
        merged_path = exports_dir / "merged" / args.output
        model_path = merge_adapter(model_path, adapter_path, merged_path)
    
    # Step 2: Dequantize if needed
    fp16_path = exports_dir / "merged" / f"{args.output}-fp16"
    model_path = dequantize_model(model_path, fp16_path)
    
    # Step 3: Convert to GGUF
    gguf_path = exports_dir / f"{args.output}-f16.gguf"
    convert_to_gguf(model_path, gguf_path, llama_cpp_path)
    
    final_gguf = gguf_path
    
    # Step 4: Quantize if requested
    if args.quantize:
        quant_path = exports_dir / f"{args.output}-{args.quantize}.gguf"
        final_gguf = quantize_gguf(gguf_path, quant_path, args.quantize, llama_cpp_path)
        
        # Clean up F16 if quantization succeeded
        if final_gguf != gguf_path and final_gguf.exists():
            print(f"🧹 Removing intermediate F16 GGUF...")
            gguf_path.unlink()
    
    # Step 5: Create Modelfile
    create_modelfile(args.output, final_gguf.name)
    
    print(f"\n🎉 Done! Create Ollama model with:")
    print(f"   cd finetune-mlx && ollama create {args.output} -f exports/{args.output}.Modelfile")


if __name__ == "__main__":
    main()
