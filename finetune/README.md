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
| `lex:` | BM25 keyword variations (short, keyword-focused) | 1-3 |
| `vec:` | Semantic reformulations (natural language) | 1-3 |
| `hyde:` | Hypothetical document passage (50-150 chars) | 0-1 |

## Trained Models

| Model | HuggingFace | Score | Status |
|-------|-------------|-------|--------|
| **Qwen3-0.6B v4 (SFT)** | [tobil/qmd-query-expansion-0.6B-v4](https://huggingface.co/tobil/qmd-query-expansion-0.6B-v4) | **98.8%** | Recommended |
| Qwen3-0.6B v4 (GRPO) | [tobil/qmd-query-expansion-0.6B-v4-grpo](https://huggingface.co/tobil/qmd-query-expansion-0.6B-v4-grpo) | 89.7% | Requires SFT base (see note) |

**Note on GRPO model**: The GRPO adapter was trained on top of the merged SFT model, so you must load SFT first:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base → merge SFT → apply GRPO
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(model, "tobil/qmd-query-expansion-0.6B-v4")
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, "tobil/qmd-query-expansion-0.6B-v4-grpo")
```

## Prompt Format

The models use **Qwen3 chat template** with `/no_think` to disable thinking mode.

### Inference (Python)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# CRITICAL: Use /no_think to disable Qwen3's thinking mode
messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate and decode
output = tokenizer.decode(tokens, skip_special_tokens=True)

# Extract assistant response (skip_special_tokens converts to "user\n...\nassistant\n...")
if "\nassistant\n" in output:
    expansion = output.split("\nassistant\n")[-1].strip()
```

### Raw Format

```
<|im_start|>user
/no_think Expand this search query: auth<|im_end|>
<|im_start|>assistant
lex: authentication configuration
lex: auth settings
vec: how to configure authentication
vec: authentication setup guide
hyde: To configure authentication, set AUTH_SECRET in your environment.<|im_end|>
```

See `PROMPT_FORMAT.md` for complete specification.

## Directory Structure

```
finetune/
├── train.py              # SFT training (uses YAML config)
├── rl.py                 # GRPO/RL training (uses YAML config)
├── tui.py                # Interactive testing interface
├── configs/
│   ├── sft_v4.yaml       # SFT training config
│   └── grpo_v4.yaml      # GRPO training config
├── evals/
│   ├── run.py            # Generate model outputs to JSONL
│   ├── score.py          # Score outputs from JSONL
│   └── queries.txt       # Test queries
├── dataset/
│   ├── prepare_data.py   # Prepare training data
│   ├── clean_data.py     # Data quality improvements
│   └── generate_data*.py # Generate from source datasets
├── PROMPT_FORMAT.md      # Prompt format specification
├── SCORING.md            # Scoring criteria
└── data/
    └── train/            # Prepared training data
```

## Quick Start

### 1. Prepare Training Data

```bash
cd dataset
uv run prepare_data.py --add-short 5
```

### 2. Train with YAML Config

```bash
# Local training
uv run train.py --config configs/sft_v4.yaml

# Or on HuggingFace Jobs
hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN \
  "https://huggingface.co/datasets/tobil/qmd-query-expansion-train-v2/resolve/main/train_sft_v4.py"
```

### 3. Evaluate

```bash
# Generate outputs
uv run evals/run.py --model tobil/qmd-query-expansion-0.6B-v4

# Score them
uv run evals/score.py evals/results_tobil_qmd-query-expansion-0.6B-v4.jsonl
```

### 4. Interactive Testing

```bash
uv run tui.py
```

## Training Configuration

Default SFT config (`configs/sft_v4.yaml`):

| Parameter | Value |
|-----------|-------|
| Method | LoRA (rank 16, alpha 32) |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size | 4 (with 4x gradient accumulation) |
| Max Seq Length | 512 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## Training Dataset

- **Dataset**: [tobil/qmd-query-expansion-train-v2](https://huggingface.co/datasets/tobil/qmd-query-expansion-train-v2)
- **Size**: 6,180 examples (26.5% short queries)
- **Format**: Qwen3 chat messages with `/no_think` directive

Key improvements in v2:
- Short query examples with proper expansions
- Hyde passages truncated to 150 chars
- Key term preservation in lex lines

## Evaluation Results

### SFT v4 (98.8% average score)

All 21 test queries rated "Excellent":

| Query | Score | Rating |
|-------|-------|--------|
| `how to configure authentication` | 99% | Excellent |
| `auth` | 95% | Excellent |
| `git rebase vs merge` | 100% | Excellent |
| `react useEffect cleanup` | 100% | Excellent |

### GRPO v4 (89.7% - with SFT base)

All 26 test queries rated "Excellent" when loaded correctly (SFT first, then GRPO adapter).

| Query | Score | Rating |
|-------|-------|--------|
| `AWS Lambda functions` | 96% | Excellent |
| `typescript async await` | 92% | Excellent |
| `kubernetes vs docker swarm` | 92% | Excellent |
| `who is TDS motorsports` | 89% | Excellent |

**Important**: Loading GRPO directly on base model results in 0% (catastrophic drift) because GRPO was trained on merged SFT weights.

## Known Issues

- **GRPO loading**: Requires SFT adapter loaded first before GRPO adapter (see model card note above)
- **Key term preservation**: Some lex lines still too generic (missing query key terms)
- **Entity scoring**: Named entity detection is heuristic-based, may miss some cases
