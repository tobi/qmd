# QMD-MLX

Apple Silicon acceleration for [QMD](https://github.com/tobi/qmd) — the on-device hybrid search engine for your notes, docs, and knowledge bases.

## What This Is

Fork of [tobi/qmd](https://github.com/tobi/qmd) that swaps the LLM backend from GGUF/node-llama-cpp to MLX. Same CLI, same SQLite index (`~/.cache/qmd/index.sqlite`), same commands. Embedding, reranking, and query expansion run 2-4x faster via the Metal GPU on M-series chips. On non-Apple-Silicon hardware, `selectBackend()` falls back to the GGUF path automatically.

![QMD Architecture](assets/qmd-architecture.png)

## When to Use This

- **Apple Silicon Mac (M1/M2/M3/M4)** -- use this fork
- **Everything else** -- use [upstream qmd](https://github.com/tobi/qmd)

## Setup

```sh
# 1. Clone
git clone https://github.com/ComputelessComputer/qmd-mlx
cd qmd-mlx

# 2. Node/Bun dependencies
bun install

# 3. Python dependencies (use a venv if you prefer)
pip install -r requirements.txt

# 4. Link globally -- replaces the system `qmd` command
bun link

# 5. Verify the MLX backend
python3 test_mlx_backend.py

# 6. Verify the CLI picks it up
qmd status   # should show "Using MLX backend"
```

## MLX Models

| Model | Purpose | Size |
|-------|---------|------|
| `mlx-community/embeddinggemma-300m-4bit` | Embeddings | ~300MB |
| `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | Reranking | ~640MB |
| `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` | Query expansion | ~1GB |

Models download from HuggingFace on first use and cache to `~/.cache/huggingface/`.

## How It Works

`mlx_backend.py` runs as a Python subprocess. Node.js communicates with it over stdin/stdout using newline-delimited JSON. The `LlamaMlx` class in `src/llm.ts` implements the same `LLM` interface as the GGUF backend, so the rest of qmd is unaware of the swap. `selectBackend()` checks `process.arch === "arm64" && process.platform === "darwin"` to decide which path to take.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.3+
- Python 3.9+
- Node.js >= 22 or Bun >= 1.0
- Homebrew SQLite: `brew install sqlite`

## Development

```sh
bun src/qmd.ts <command>    # Run from source
bun test                    # Run tests
python3 test_mlx_backend.py # Test MLX backend directly
```

## Staying in Sync with Upstream

This fork tracks [tobi/qmd](https://github.com/tobi/qmd) main. Core search logic, indexing, MCP server, CLI commands -- all from upstream. The MLX additions are isolated in `mlx_backend.py` and the `LlamaMlx` class in `src/llm.ts`.

## Full Documentation

For complete usage -- collections, context, search commands, MCP server, architecture -- see the [QMD README](https://github.com/tobi/qmd#readme).

## License

MIT
