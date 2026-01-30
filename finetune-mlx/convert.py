#!/usr/bin/env python3
"""
Convert trained model to GGUF for Ollama/llama.cpp.

Usage:
    python convert.py                    # Convert with default paths
    python convert.py --output my-model  # Custom output name
"""

import argparse
import subprocess
import sys
from pathlib import Path


def merge_adapter(model_path: Path, adapter_path: Path, output_path: Path):
    """Merge LoRA adapter into base model."""
    print(f"🔄 Merging adapter into base model...")
    
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


def convert_to_gguf(model_path: Path, output_name: str):
    """Convert MLX model to GGUF format."""
    print(f"\n🔄 Converting to GGUF...")
    
    # Check if llama.cpp convert script exists
    # We'll use the huggingface approach via mlx_lm
    
    # First, export to HuggingFace format
    hf_path = Path("exports") / "hf" / output_name
    hf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # MLX models can be uploaded to HF and converted there
    # Or we can use llama.cpp's convert script
    
    print(f"📦 Model at: {model_path}")
    print(f"\nTo convert to GGUF, you have two options:")
    print(f"\n1. Upload to HuggingFace and use their GGUF converter:")
    print(f"   huggingface-cli upload {output_name} {model_path}")
    print(f"   Then use HF's GGUF conversion in the model settings")
    print(f"\n2. Use llama.cpp's convert script:")
    print(f"   git clone https://github.com/ggerganov/llama.cpp")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {output_name}.gguf")
    print(f"\n3. Create Ollama Modelfile:")
    
    # Create Modelfile template
    modelfile = Path("exports") / f"{output_name}.Modelfile"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    
    modelfile_content = f'''# Modelfile for {output_name}
# After converting to GGUF, run: ollama create {output_name} -f {modelfile}

FROM ./{output_name}.gguf

TEMPLATE """<|im_start|>user
/no_think Expand this search query: {{{{.Prompt}}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
'''
    
    with open(modelfile, "w") as f:
        f.write(modelfile_content)
    
    print(f"\n✅ Modelfile template saved to: {modelfile}")
    print(f"\nAfter GGUF conversion, create Ollama model with:")
    print(f"   ollama create {output_name} -f {modelfile}")


def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF")
    parser.add_argument("--model", default="models/Qwen_Qwen2.5-1.5B/mlx", help="Base model path")
    parser.add_argument("--adapter", default="adapters/sft", help="LoRA adapter path")
    parser.add_argument("--output", "-o", default="qmd-expansion", help="Output model name")
    parser.add_argument("--skip-merge", action="store_true", help="Skip adapter merge")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    adapter_path = Path(args.adapter)
    
    # Merge adapter if exists
    if not args.skip_merge and adapter_path.exists():
        merged_path = Path("exports") / "merged" / args.output
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merge_adapter(model_path, adapter_path, merged_path)
        model_path = merged_path
    
    # Convert to GGUF
    convert_to_gguf(model_path, args.output)


if __name__ == "__main__":
    main()
