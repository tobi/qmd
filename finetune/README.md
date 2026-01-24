# QMD Query Expansion Model Finetuning

Finetune small Qwen models for QMD's query expansion task.

## Goal

Train models that convert user queries into retrieval-optimized outputs:

```
Input: "how to configure authentication"

Output:
lex: authentication setup
lex: auth configuration
vec: how to set up user authentication in the application
hyde: To configure authentication, set the AUTH_SECRET environment variable and enable the auth middleware in your application config.
```

## Output Format

| Type | Purpose | Count |
|------|---------|-------|
| `lex` | BM25 keyword variations | 1-3 |
| `vec` | Semantic reformulations | 1-3 |
| `hyde` | Hypothetical document passage | 0-1 |

## Trained Models

| Model | HuggingFace | Format Compliance | Status |
|-------|-------------|-------------------|--------|
| **Qwen3-0.6B (finetuned)** | [tobil/qmd-query-expansion-0.6B](https://huggingface.co/tobil/qmd-query-expansion-0.6B) | **95%** | Recommended |
| Qwen3-1.7B (finetuned) | [tobil/qmd-query-expansion-1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | 0% | Training issues |
| Qwen3-0.6B (baseline) | - | 0% | Untrained |

## Training Dataset

- **Dataset**: [tobil/qmd-query-expansion-train](https://huggingface.co/datasets/tobil/qmd-query-expansion-train)
- **Source**: Transformed from [s-emanuilov/query-expansion](https://huggingface.co/datasets/s-emanuilov/query-expansion) (CC BY 4.0)
- **Size**: 5,157 examples (train: 4,641, eval: 516)
- **Format**: Chat messages with user query and assistant response in lex/vec/hyde format

## Directory Structure

```
finetune/
├── README.md                 # This file
├── DATASETS.md               # Dataset research findings
├── TRAINING_JOBS.md          # HuggingFace Jobs tracking
├── generate_data_offline.py  # Transform s-emanuilov dataset to QMD format
├── prepare_data.py           # Upload to HuggingFace Hub
├── train_0.6B.py             # Training script for 0.6B model
├── train_1.7B.py             # Training script for 1.7B model
├── train_grpo.py             # GRPO RL training (optional)
├── evaluate_model.py         # Evaluate finetuned models
├── evaluate_baseline.py      # Evaluate base models
├── data/
│   ├── qmd_expansion.jsonl   # Generated training data
│   └── train/                # Prepared chat format
└── evaluation_*.json         # Evaluation results
```

## Quick Start

### 1. Generate Training Data

```bash
# Transform s-emanuilov dataset to QMD format (no API needed)
uv run generate_data_offline.py
```

### 2. Prepare and Upload Dataset

```bash
# Convert to chat format and upload to HuggingFace Hub
uv run prepare_data.py
```

### 3. Train on HuggingFace Jobs

```bash
# Train Qwen3-0.6B (recommended)
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
  "https://huggingface.co/tobil/qmd-training-scripts/resolve/main/train_0.6B.py"
```

### 4. Evaluate

```bash
# Evaluate finetuned model
uv run evaluate_model.py --model tobil/qmd-query-expansion-0.6B --base-model Qwen/Qwen3-0.6B

# Compare to baseline
uv run evaluate_baseline.py --model Qwen/Qwen3-0.6B --num-queries 10
```

### 5. Export to GGUF

```bash
# Convert to GGUF for node-llama-cpp (TODO)
uv run export_gguf.py --model tobil/qmd-query-expansion-0.6B --quantization Q8_0
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Method | LoRA (rank 16, alpha 32) |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size | 4 (with 4x gradient accumulation) |
| Max Seq Length | 512 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## Prompt Format

The models are trained on this simple prompt format:

```
Expand this search query:

{query}
```

The model responds with lex/vec/hyde lines directly.

## Evaluation Results

### 0.6B Finetuned Model (95% format compliance)

Sample outputs:

| Query | Output |
|-------|--------|
| `how to configure authentication` | lex: steps for setting up authentication<br>vec: steps for setting up authentication in cloud services<br>hyde: The process of configure authentication... |
| `kubernetes vs docker swarm` | lex: kubernetes and docker swarm<br>vec: kubernetes vs docker swarm<br>hyde: Kubernetes vs docker swarm is an important concept... |
| `cors error fix` | lex: how to fix cors<br>vec: how to fix cors issues in web apps<br>hyde: The topic of cors error fix guide... |

### Baseline Model (0% format compliance)

The untrained model generates random prose, code blocks, or repetitive text with no understanding of the lex/vec/hyde format.

## Future Work

- [ ] Export to GGUF for local inference
- [ ] Integrate into QMD as default query expansion model
- [ ] GRPO training for improved diversity (optional)
- [ ] Fix 1.7B training issues
