# QMD Query Expansion Fine-Tuning

Train small language models to expand search queries for [QMD](https://github.com/tobi/qmd)'s hybrid retrieval pipeline.

## What This Does

Given a raw search query like `"auth config"`, the trained model produces structured expansions:

```
lex: authentication configuration
lex: auth settings setup
vec: how to configure authentication settings
vec: authentication configuration options
hyde: Authentication can be configured by setting the AUTH_SECRET environment variable.
```

These feed into QMD's three search backends:
- **`lex:`** lines go to BM25 full-text search (short, keyword-focused)
- **`vec:`** lines go to vector similarity search (natural language phrases)
- **`hyde:`** is a hypothetical document passage for embedding-based retrieval ([HyDE](https://arxiv.org/abs/2212.10496) technique)

## Quick Start

### Cloud training via HuggingFace Jobs (no GPU needed)

```bash
# 1. SFT: teach the model the output format (~45 min on A10G, ~$1.50)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py

# 2. GRPO: RL refinement on top of SFT (~20 min on A10G, ~$0.50)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h jobs/grpo.py

# 3. Evaluate against test queries (needs local GPU or use eval job)
uv run eval.py --model tobil/qmd-query-expansion-1.7B-grpo \
               --sft-model tobil/qmd-query-expansion-1.7B-sft

# 4. Convert to GGUF for local deployment (Ollama, llama.cpp)
uv run convert_gguf.py --size 1.7B
```

### Local training (if you have a GPU)

```bash
uv run train.py sft  --config configs/sft.yaml
uv run train.py grpo --config configs/grpo.yaml
```

### Monitoring HF Jobs

```bash
hf jobs ps                           # list running jobs
hf jobs inspect <job-id>             # check status
hf jobs logs <job-id>                # stream logs
hf jobs cancel <job-id>              # cancel a job
```

## Prompt Format

All tools use the same prompt — **Qwen3 chat template with `/no_think`**:

```
<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
```

The `/no_think` directive suppresses Qwen3's chain-of-thought mode, producing
direct `lex:/vec:/hyde:` output without `<think>` blocks.

## File Structure

```
finetune/
├── reward.py          # Scoring/reward function (single source of truth)
├── train.py           # Unified SFT + GRPO training (two subcommands)
├── eval.py            # Generate expansions and score them
├── convert_gguf.py    # GGUF conversion for Ollama/llama.cpp
├── jobs/
│   ├── sft.py         # Self-contained SFT for HuggingFace Jobs
│   ├── grpo.py        # Self-contained GRPO for HuggingFace Jobs
│   ├── eval.py        # Self-contained eval for HuggingFace Jobs
│   ├── eval_common.py # Shared eval utilities
│   └── quantize.py    # GGUF quantization for HuggingFace Jobs
├── configs/
│   ├── sft.yaml       # SFT hyperparameters for Qwen3-1.7B
│   └── grpo.yaml      # GRPO hyperparameters for Qwen3-1.7B
├── evals/
│   └── queries.txt    # 31 test queries across 8 categories
├── data/
│   └── qmd_expansion_v2.jsonl  # Source training data (1,000 high-quality examples)
├── dataset/
│   ├── generate_data.py         # Generate data via Claude API
│   ├── generate_data_offline.py # Generate from existing HF dataset
│   ├── prepare_data.py          # Format for Qwen3 chat template
│   └── clean_data.py            # Detect technical term misinterpretations
├── SCORING.md         # Detailed scoring rubric reference
└── README.md          # This file
```

## Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)

Teaches the model the `lex:/vec:/hyde:` output format from labeled examples.

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3-1.7B` |
| Method | LoRA (rank 16, alpha 32) |
| Target modules | All projection layers (q/k/v/o/gate/up/down) |
| Dataset | ~2,290 examples (train split) |
| Effective batch size | 16 (4 × 4 gradient accumulation) |
| Epochs | 5 |
| Learning rate | 2e-4 (cosine schedule) |

```bash
uv run train.py sft --config configs/sft.yaml
uv run train.py sft --config configs/sft.yaml --dry-run  # preview config
```

### Stage 2: GRPO (Group Relative Policy Optimization)

Reinforcement learning on top of the merged SFT weights. The model generates
multiple expansions per query, they are scored by the reward function, and the
model is updated to prefer higher-scoring outputs.

| Parameter | Value |
|-----------|-------|
| Base | Merged SFT checkpoint |
| Method | LoRA (rank 4, alpha 8) — smaller for RL stability |
| Target modules | q_proj, v_proj only |
| Reward | `reward.py` (rule-based, 5 dimensions) |
| KL beta | 0.04 — prevents drift from SFT checkpoint |
| Generations per prompt | 4 |
| Max steps | 200 |
| Learning rate | 5e-7 |

**Important:** `beta > 0` is critical. With `beta=0` the model experiences
catastrophic drift and scores drop to 0%.

```bash
uv run train.py grpo --config configs/grpo.yaml
uv run train.py grpo --config configs/grpo.yaml --dry-run  # test reward function
```

## Evaluation

`eval.py` generates expansions from a model and scores them against test queries:

```bash
# Evaluate an SFT model
uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft

# Evaluate a GRPO model (needs SFT adapter merged first)
uv run eval.py --model tobil/qmd-query-expansion-1.7B-grpo \
               --sft-model tobil/qmd-query-expansion-1.7B-sft

# Verbose output with deduction details
uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft -v

# Save detailed scores to JSON
uv run eval.py --model tobil/qmd-query-expansion-1.7B-sft -o scores.json

# Score an existing JSONL file (backwards compat with old run.py output)
uv run eval.py --score-only evals/results_old.jsonl
```

## Reward Function

`reward.py` is the single source of truth for scoring. It is used both as the
GRPO reward signal during training and for evaluation.

Five scoring dimensions (max 120 without hyde, 140 with):

| Dimension | Points | What It Measures |
|-----------|--------|------------------|
| **Format** | 0-30 | Has lex/vec lines, no invalid lines |
| **Diversity** | 0-30 | Multiple expansion types, diverse content, no query echoes |
| **HyDE** | 0-20 | Present, 50-200 chars, single line, not repetitive |
| **Quality** | 0-20 | Lex shorter than vec, natural language, preserves key terms |
| **Entity** | -45 to +20 | Named entities preserved in lex and vec lines |
| **Think bonus** | 0-20 | Reward for NOT using `<think>` mode |

**Hard failures** (instant 0.0):
- Chat template leakage (`<|im_start|>`, `<|im_end|>`, etc.)
- Any line without a valid `lex:`, `vec:`, or `hyde:` prefix

```bash
# Self-test the reward function
uv run reward.py
```

## GGUF Conversion

Merges base + SFT + GRPO adapters into a single model and produces
quantized GGUF files for deployment:

```bash
# Use preset for 1.7B
uv run convert_gguf.py --size 1.7B

# Use preset for 4B
uv run convert_gguf.py --size 4B

# Custom models
uv run convert_gguf.py --base Qwen/Qwen3-1.7B \
                       --sft tobil/qmd-query-expansion-1.7B-sft \
                       --grpo tobil/qmd-query-expansion-1.7B-grpo \
                       --output tobil/qmd-query-expansion-1.7B-gguf
```

### Using with Ollama

```bash
huggingface-cli download tobil/qmd-query-expansion-1.7B-gguf \
    qmd-query-expansion-1.7B-q4_k_m.gguf --local-dir .

echo 'FROM ./qmd-query-expansion-1.7B-q4_k_m.gguf' > Modelfile
ollama create qmd-expand -f Modelfile
ollama run qmd-expand
```

## Data Pipeline

The training data (1,000 examples in `data/qmd_expansion_v2.jsonl`) was generated
from two sources and cleaned for quality. To regenerate:

```bash
# Generate from existing HuggingFace dataset (bulk, no API needed)
uv run dataset/generate_data_offline.py

# Generate via Claude API (higher quality, needs ANTHROPIC_API_KEY)
uv run dataset/generate_data.py --count 100

# Detect and fix technical term misinterpretations
uv run dataset/clean_data.py

# Format for Qwen3 chat template, add short-query augmentation, split train/val
uv run dataset/prepare_data.py
```

## Architecture Notes

The two-stage training approach (SFT → GRPO) is standard for structured-output models:

1. **SFT** establishes format compliance and basic query understanding. It uses
   a large LoRA (rank 16, all projection layers) because it needs to learn a
   new output format from scratch.

2. **GRPO** refines quality within the learned format. It uses a small LoRA
   (rank 4, q/v only) and KL regularization to make incremental improvements
   without losing what SFT taught.

The reward function is entirely rule-based (no LLM judge) which makes it fast,
deterministic, and suitable as an RL signal. See `SCORING.md` for the full rubric.

## Training Results (Qwen3-1.7B, v2)

### SFT

| Metric | Value |
|--------|-------|
| Final train loss | 0.472 |
| Final eval loss | 0.304 |
| Token accuracy (train) | 97.4% |
| Token accuracy (eval) | 93.8% |
| Epochs | 5 |
| Hardware | A10G (24 GB VRAM) |

### GRPO

| Metric | Value |
|--------|-------|
| Mean reward | 0.757 |
| Final loss | 0.0005 |
| KL divergence | 0.00048 |
| Mean completion length | ~58 tokens |
| Training time | ~19 min (200 steps) |
| Hardware | A10G (24 GB VRAM) |

### Evaluation Scores

| Model | Average Score | Excellent (30) |
|-------|--------------|-----------------|
| SFT | 92.0% | 30/30 |
| GRPO | 91.7% | 30/30 |
