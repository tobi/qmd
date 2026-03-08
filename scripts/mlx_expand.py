#!/usr/bin/env python3
"""QMD MLX Query Expansion (sidecar).

Outputs query expansions in the exact line-oriented format expected by QMD:
  lex: ...
  vec: ...
  hyde: ...

Environment variables:
  QMD_MLX_MODEL   Base MLX model directory (default: finetune-mlx/models/Qwen_Qwen3-1.7B/mlx)
  QMD_MLX_ADAPTER LoRA adapter directory (default: finetune-mlx/adapters/qmd_query_expansion_1.7B_sft)
  QMD_MLX_TEMP    Temperature (default: 1.0)
  QMD_MLX_MAX_TOKENS Max tokens (default: 512)

Usage:
  ./scripts/mlx_expand.py "auth config"
  echo "auth config" | ./scripts/mlx_expand.py
"""

import os
import sys

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROMPT_TEMPLATE = """<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
"""


def _read_query(argv):
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()
    data = sys.stdin.read().strip()
    return data


def main():
    query = _read_query(sys.argv)
    if not query:
        print("", end="")
        return 0

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model = os.environ.get(
        "QMD_MLX_MODEL",
        os.path.join(repo_root, "finetune-mlx", "models", "Qwen_Qwen3-1.7B", "mlx"),
    )
    adapter = os.environ.get(
        "QMD_MLX_ADAPTER",
        os.path.join(repo_root, "finetune-mlx", "adapters", "qmd_query_expansion_1.7B_sft"),
    )

    temp = float(os.environ.get("QMD_MLX_TEMP", "1.0"))
    max_tokens = int(os.environ.get("QMD_MLX_MAX_TOKENS", "512"))

    prompt = PROMPT_TEMPLATE.format(query=query)

    # Load base model + adapter if available.
    try:
        m, tok = load(model, adapter_path=adapter)
    except Exception:
        m, tok = load(model)

    sampler = make_sampler(temp=temp)
    out = generate(
        m,
        tok,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    # Clean special tokens and keep only line-based output.
    out = out.replace("<|im_end|>", "").strip()
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]

    # Keep only lex/vec/hyde lines (defensive)
    filtered = []
    for ln in lines:
        if ln.startswith("lex:") or ln.startswith("vec:") or ln.startswith("hyde:"):
            filtered.append(ln)

    sys.stdout.write("\n".join(filtered).strip())
    if filtered:
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
