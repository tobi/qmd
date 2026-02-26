# QMD Query Expansion - Apple Silicon (MLX)

Apple Silicon alternative to the CUDA-based [`finetune/`](../finetune/) directory.

Port of QMD's query expansion fine-tuning to Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

Train small language models locally on M1/M2/M3/M4 Macs to expand search queries for hybrid retrieval.

## Features

- **SFT Training**: Supervised fine-tuning with LoRA
- **GRPO Training**: Group Relative Policy Optimization (reinforcement learning)
- **100% Local**: No cloud GPU needed, runs on Apple Silicon
- **MLX Optimized**: Native Metal acceleration

## Results

Comparison with original NVIDIA A10G implementation:

| Metric | NVIDIA (SFT+GRPO) | Apple Silicon (SFT) | Apple Silicon (GRPO) |
|--------|-------------------|---------------------|----------------------|
| Avg Score | 92% | 99.6% | 100.4% |
| Perfect Queries | 30/30 | 28/30 | 28/30 |
| Hardware | A10G 24GB | Mac Mini M4 | Mac Mini M4 |
| Cost | ~$2/run | $0 | $0 |

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download and convert base model
python -c "from mlx_lm import load; load('Qwen/Qwen3-1.7B')"

# Train SFT (supervised fine-tuning)
python train.py sft --iters 3500

# Train GRPO (reinforcement learning refinement)
python grpo.py --steps 200

# Evaluate
python grpo.py --eval-only --adapter adapters/qwen3-grpo
```

## What It Does

Given a query like `"auth config"`, the model produces structured expansions:

```
lex: authentication configuration
lex: auth settings setup
vec: how to configure authentication settings
hyde: Authentication can be configured by setting AUTH_SECRET...
```

These feed into QMD's hybrid retrieval:
- `lex:` → BM25 full-text search
- `vec:` → Vector similarity search  
- `hyde:` → Hypothetical document embedding

## File Structure

```
├── train.py          # SFT training script
├── grpo.py           # GRPO (RL) training script
├── eval.py           # Evaluation utilities
├── reward.py         # Scoring/reward function
├── convert.py        # GGUF conversion for Ollama
├── configs/
│   └── sft.yaml      # SFT hyperparameters
├── evals/
│   └── queries.txt   # Test queries (31 total)
└── tests/            # Unit tests
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~8GB RAM for training
- ~4GB disk for models

## Training Details

### SFT (Supervised Fine-Tuning)
- Base model: Qwen3-1.7B
- LoRA rank: 8, layers: 8
- Learning rate: 1e-4
- Steps: 3500
- Time: ~60 min on M4

### GRPO (Group Relative Policy Optimization)
- Starts from SFT checkpoint
- 4 completions per query
- KL regularization (β=0.04)
- Steps: 200
- Time: ~30 min on M4

## Credits

- Original QMD: [tobi/qmd](https://github.com/tobi/qmd)
- MLX framework: [ml-explore/mlx](https://github.com/ml-explore/mlx)
- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

## Contributors

- [@sujito00](https://github.com/sujito00)
- [@dgilperez](https://github.com/dgilperez)

## License

MIT
