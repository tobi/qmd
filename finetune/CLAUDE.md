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

## Training Data Format

**There is exactly one JSONL format.** Every file in `data/*.jsonl` must match the strict Pydantic schema in `dataset/schema.py`:

```json
{"query": "auth config", "output": [["hyde", "..."], ["lex", "..."], ["vec", "..."]]}
```

- `query`: non-empty string
- `output`: list of `[type, text]` pairs where type is `"lex"`, `"vec"`, or `"hyde"`
- Extra metadata fields (`category`, `intent`, `is_short`) are allowed but ignored

The schema is enforced by `dataset/schema.py:TrainingExample` (Pydantic model). All data loading goes through `load_examples()` which fails loudly on invalid data. No format alternatives, no legacy fallbacks.

**All `.jsonl` files in `data/` are concatenated and deduplicated for training runs.** The prepared train/val files in `data/train/` are ephemeral build artifacts.

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

## Dataset Tools

| Script | Purpose |
|--------|---------|
| `dataset/schema.py` | Pydantic `TrainingExample` model + `load_examples()` |
| `dataset/prepare_data.py` | Load via schema, apply Qwen3 chat template, dedup, split |
| `dataset/validate_schema.py` | Validate all JSONL files against schema |
| `dataset/score_data.py` | Score all examples using reward.py |
| `dataset/analyze_data.py` | Analyze distribution and quality |

## Training Pipeline

Always use **Qwen3-1.7B** as the base model unless explicitly stated otherwise.

### Stage 0: Prepare Data

```bash
uv run dataset/prepare_data.py
# Creates: data/train/train.jsonl, data/train/val.jsonl (ephemeral)
```

### Stage 1: SFT

```bash
# Local (requires CUDA)
uv run train.py sft --config configs/sft.yaml

# Cloud (HuggingFace Jobs)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py
```

### Stage 2: GRPO

```bash
# Local (requires CUDA)
uv run train.py grpo --config configs/grpo.yaml

# Cloud (HuggingFace Jobs)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h jobs/grpo.py
```

### HuggingFace Jobs

```bash
hf jobs ps                    # List running jobs
hf jobs logs <job-id>         # Stream logs
hf jobs inspect <job-id>      # Check status
hf jobs cancel <job-id>       # Cancel a job
```

### Evaluation

```bash
uv run eval.py --model ./outputs/grpo
uv run eval.py --model tobil/qmd-query-expansion-1.7B
uv run eval.py --model ./outputs/grpo -o eval_results.json
```

## Quality Scoring

`reward.py` is the single source of truth for scoring:

```bash
uv run reward.py   # Self-test
```

See `SCORING.md` for the full rubric.

## Experiments

Experimental training configurations live in `experiments/`:

```
experiments/
├── lfm2/          # LiquidAI LFM2-1.2B (hybrid architecture, faster inference)
│   ├── sft_lfm2.yaml
│   └── sft_lfm2.py
└── gepa/          # DSPy-based prompt optimization (GEPA)
    ├── dspy_gepa.py
    └── ...
```

These are not part of the main training pipeline.

## Key Files

```
finetune/
├── reward.py          # Scoring function (single source of truth)
├── train.py           # Unified SFT + GRPO training
├── eval.py            # Generate and score expansions
├── convert_gguf.py    # GGUF conversion
├── SCORING.md         # Detailed scoring rubric
├── CLAUDE.md          # This file
├── Justfile           # Common commands
├── data/              # All training JSONL files (strict schema)
├── dataset/           # Schema + data tools (Pydantic-based)
├── jobs/              # Self-contained HuggingFace Jobs scripts
├── configs/           # Training configs (sft.yaml, grpo.yaml)
├── evals/             # Test queries
├── experiments/       # Experimental configs (LFM2, GEPA)
└── outputs/           # Local training outputs (gitignored)
```
