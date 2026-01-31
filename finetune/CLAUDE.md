# QMD Query Expansion Fine-Tuning

## Overview

Train Qwen3-1.7B to expand search queries into structured `hyde:/lex:/vec:` output for QMD's hybrid retrieval pipeline.

## Output Format

```
hyde: A hypothetical document passage that would answer the query.
lex: keyword1
lex: keyword2
vec: semantic query reformulation
vec: another semantic variation
```

- `hyde:` always comes FIRST (one line max)
- `lex:` lines for BM25 keyword search (1-3 lines, short keywords)
- `vec:` lines for vector similarity search (1-3 lines, natural language)

## HuggingFace Repositories

| Repository | Purpose |
|------------|---------|
| `tobil/qmd-query-expansion-1.7B` | Final merged model (SFT + GRPO) |
| `tobil/qmd-query-expansion-1.7B-gguf` | GGUF quantized versions for deployment |
| `tobil/qmd-query-expansion-1.7B-sft` | SFT adapter checkpoint (intermediate) |
| `tobil/qmd-query-expansion-1.7B-grpo` | GRPO adapter checkpoint (intermediate) |
| `tobil/qmd-query-expansion-train` | Prepared training dataset |

**Rules:**
- No versioned repos (`-v1`, `-v2`, `-v4`, etc.) - update in place
- Only push when eval scores improve over current deployed model
- Always include eval results in model card when pushing

## Training Data

All JSONL files in `data/` are training data:

```
data/
├── qmd_expansion_v2.jsonl
├── qmd_expansion_handcrafted_only.jsonl
├── qmd_only_sampled.jsonl
├── qmd_only_variants.jsonl
└── ... any additional .jsonl files
```

**All `.jsonl` files in `data/` should be concatenated for training runs.**

Each JSONL line: `{"input": "query", "output": "hyde:...\nlex:...\nvec:..."}`

## Data Generation Tools

| Script | Purpose |
|--------|---------|
| `dataset/generate_data.py` | Generate via Claude API (high quality) |
| `dataset/generate_data_offline.py` | Transform from HuggingFace datasets |
| `dataset/prepare_data.py` | Format for Qwen3 chat template |
| `dataset/clean_data.py` | Detect and fix technical term issues |
| `generate_only_variants.py` | Generate `/only:lex` and `/only:vec` variants |

## Local Training Output

All training outputs go to `outputs/` (gitignored):

```
outputs/
├── sft/           # SFT checkpoint
└── grpo/          # GRPO checkpoint
```

## Training Pipeline

Always use **Qwen3-1.7B** as the base model unless explicitly stated otherwise.

Training can run **locally** (requires CUDA GPU) or via **HuggingFace Jobs** (cloud GPU, no local hardware needed).

### Stage 0: Prepare Data

Raw data in `data/*.jsonl` must be converted to Qwen3 chat format before training:

```bash
# Process all JSONL files in data/
uv run dataset/prepare_data.py
# Creates: data/train/train.jsonl, data/train/val.jsonl

# Or process a specific file
uv run dataset/prepare_data.py --input data/qmd_expansion_v2.jsonl
```

This applies the Qwen3 chat template, deduplicates, and splits into train/val sets.

### Stage 1: SFT

```bash
# Local (requires CUDA)
uv run train.py sft --config configs/sft.yaml
# Output: outputs/sft/

# Cloud (HuggingFace Jobs - no local GPU needed)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py
```

### Stage 2: GRPO

```bash
# Local (requires CUDA)
uv run train.py grpo --config configs/grpo.yaml
# Output: outputs/grpo/

# Cloud (HuggingFace Jobs - no local GPU needed)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h jobs/grpo.py
```

### HuggingFace Jobs

If no local CUDA device is available, use `hf jobs` to run training in the cloud:

```bash
hf jobs ps                    # List running jobs
hf jobs logs <job-id>         # Stream logs
hf jobs inspect <job-id>      # Check status
hf jobs cancel <job-id>       # Cancel a job
```

The `jobs/` directory contains self-contained scripts that include all dependencies inline.

### Evaluation

```bash
# Eval local model
uv run eval.py --model ./outputs/grpo

# Eval HuggingFace model
uv run eval.py --model tobil/qmd-query-expansion-1.7B

# Save eval results to file
uv run eval.py --model ./outputs/grpo -o eval_results.json
```

## Quality Scoring

`reward.py` is the single source of truth for scoring:

```bash
# Self-test the reward function
uv run reward.py
```

See `SCORING.md` for the full rubric.

## Deployment Rules

**Never upload without eval.** Every model push must include eval results.

### Checklist

1. Train SFT on all `data/*.jsonl` → `outputs/sft/`
2. Train GRPO on top of SFT → `outputs/grpo/`
3. **Run eval on local model**: `uv run eval.py --model ./outputs/grpo -o eval_results.json`
4. Compare against current deployed model's eval
5. If eval improves:
   - Push to `tobil/qmd-query-expansion-1.7B`
   - **Include eval output in the model card / commit message**
6. Convert to GGUF and update `tobil/qmd-query-expansion-1.7B-gguf`
7. Update `src/llm.ts` DEFAULT_GENERATE_MODEL if repo name changed

## Key Files

```
finetune/
├── reward.py          # Scoring function (single source of truth)
├── train.py           # Unified SFT + GRPO training
├── eval.py            # Generate and score expansions
├── convert_gguf.py    # GGUF conversion
├── SCORING.md         # Detailed scoring rubric
├── CLAUDE.md          # This file
├── data/              # All training JSONL files
├── outputs/           # Local training outputs (gitignored)
├── dataset/           # Data generation scripts
├── jobs/              # Self-contained HuggingFace Jobs scripts
├── configs/           # Training configs (sft.yaml, grpo.yaml)
└── evals/             # Test queries and results
```
