#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "transformers>=4.45.0",
#     "torch",
# ]
# ///
"""
Export finetuned model to GGUF format for use with node-llama-cpp.

Usage:
    python export_gguf.py --model models/qmd-expansion --quantization Q8_0
    python export_gguf.py --model models/qmd-expansion --quantization Q4_K_M
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument("--model", type=str, required=True, help="Path to finetuned model")
    parser.add_argument("--output", type=str, help="Output GGUF file path")
    parser.add_argument("--quantization", type=str, default="Q8_0",
                        choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
                        help="Quantization method")
    parser.add_argument("--push-to-hub", type=str, help="Push GGUF to HuggingFace Hub repo")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

    # Default output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(model_path / f"qmd-expansion-{args.quantization}.gguf")

    print(f"Loading model from {model_path}")

    # Load the finetuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Exporting to GGUF with {args.quantization} quantization...")

    # Export to GGUF
    model.save_pretrained_gguf(
        output_path.replace(".gguf", ""),  # Unsloth adds .gguf
        tokenizer,
        quantization_method=args.quantization.lower().replace("_", "-"),
    )

    print(f"Exported to {output_path}")

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing GGUF to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub_gguf(
            args.push_to_hub,
            tokenizer,
            quantization_method=args.quantization.lower().replace("_", "-"),
        )

    print("Export complete!")
    print(f"\nTo use in QMD, update src/llm.ts:")
    print(f'  const DEFAULT_GENERATE_MODEL = "{output_path}";')


if __name__ == "__main__":
    main()
