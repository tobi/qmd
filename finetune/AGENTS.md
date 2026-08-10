# finetune/ — Python ML Pipeline

Fine-tunes Qwen3-1.7B to expand QMD search queries into `hyde:` / `lex:` / `vec:` structured output. **Isolated from `src/`** — no Python imports the TS runtime; the bridge is a HuggingFace-hosted GGUF file consumed by `src/llm.ts`.

## STRUCTURE

```
finetune/
├── train.py                # 670 lines — SFT entrypoint (LoRA on Qwen3-1.7B)
├── reward.py               # 698 lines — SCORING SOURCE OF TRUTH (5 dimensions, max 140 pts)
├── eval.py                 # 194 lines — generate expansions + score them
├── eval_retrieval.py       # 488 lines — retrieval-quality eval (separate from reward.py)
├── benchmark.py            # 182 lines — model comparison harness
├── convert_gguf.py         # 221 lines — base + adapter merge → quantized GGUF
├── convert_onnx.py         # 461 lines — ONNX export path (alternative to GGUF)
├── train_unsloth.py        # 198 lines — Unsloth-optimized SFT variant
├── SCORING.md              # full rubric (read this before touching reward.py)
├── CLAUDE.md               # canonical contributor doc (read first)
├── README.md               # HuggingFace model card
├── Justfile                # common commands (validate, eval, etc.)
├── Modelfile               # Ollama import template
├── pyproject.toml          # uv-managed deps (torch, trl, peft, transformers, accelerate)
├── dataset/
│   ├── schema.py           # 244 lines — Pydantic `TrainingExample` + `load_examples()`
│   ├── prepare_data.py     # 184 lines — chat-template format + dedup + train/val split
│   ├── validate_schema.py  # 88 lines — JSONL validator (CI gate)
│   ├── score_data.py       # 96 lines — score all examples via reward.py
│   ├── analyze_data.py     # 243 lines — distribution + quality analysis
│   └── prepare_data_lfm2.py # 85 lines — LFM2-specific prep (experimental)
├── data/                   # 13 JSONL files (~2,290 examples) + train/ (ephemeral build artifacts)
├── configs/                # 4 YAML: sft.yaml, sft_local.yaml, sft-lfm2.yaml, accelerate_multi_gpu.yaml
├── evals/queries.txt       # 80 test queries (8 categories)
├── jobs/                   # HuggingFace Jobs scripts (sft.py, eval.py, eval_common.py)
└── experiments/            # NOT PRODUCTION — gepa/, grpo/, lfm2/
```

## WHERE TO LOOK

| Task | Start here |
|------|-----------|
| Change scoring rubric | `reward.py` (then update `SCORING.md`) |
| Change training recipe | `train.py` + `configs/sft.yaml` |
| Add new training data | Drop `.jsonl` into `data/` (must pass `dataset/schema.py`) |
| Validate data | `uv run dataset/validate_schema.py` |
| Prepare train/val split | `uv run dataset/prepare_data.py` (writes to `data/train/`, ephemeral) |
| Run SFT locally | `uv run train.py sft --config configs/sft_local.yaml` (needs CUDA) |
| Run SFT on HF Jobs | `hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py` |
| Evaluate model | `uv run eval.py tobil/qmd-query-expansion-1.7B` |
| Convert to GGUF | `uv run convert_gguf.py --size 1.7B` |
| Inspect dataset stats | `uv run dataset/analyze_data.py` |

## CONVENTIONS

- **`uv` only — never `pip`.** `uv run <script>` resolves deps from `pyproject.toml` + `uv.lock`.
- **Python ≥3.10.** Use modern syntax (`match`, PEP 604 unions, `from __future__ import annotations` not needed).
- **Strict Pydantic schema** — every JSONL row must validate against `dataset/schema.py:TrainingExample`. No legacy fallbacks. Run `validate_schema.py` before committing data.
- **All `.jsonl` in `data/` are concatenated + deduplicated** for training. The `data/train/*.jsonl` files are ephemeral outputs of `prepare_data.py` — do not edit them directly.
- **Base model is always `Qwen/Qwen3-1.7B`.** Don't substitute without a separate experiment branch.
- **Prompt format: Qwen3 chat template + `/no_think`.** Suppresses CoT mode; produces direct `lex:/vec:/hyde:` output without `<think>` blocks.
- **Output order: `hyde:` FIRST.** Then `lex:` lines, then `vec:` lines. `reward.py` enforces this.
- **HuggingFace repos are unversioned.** Update in place — never push `*-v1`, `*-v2` suffixes. See `CLAUDE.md:48`.
- **Justfile** is the canonical command surface — `just validate`, `just eval`, etc. Mirror new commands there.

## ANTI-PATTERNS

Inherits all root rules. Plus:

| Rule | Why |
|------|-----|
| **NEVER** use `pip install` | Use `uv` — `uv.lock` is the source of truth |
| **NEVER** create versioned HF repos (`-v2`, `-v3`) | Update `tobil/qmd-query-expansion-1.7B` in place. Past `-v2`, `-v3` repos exist as historical mistakes. |
| **NEVER** add a new JSONL field without updating `dataset/schema.py` | Pydantic fails loudly. Extra metadata fields are allowed but ignored — don't depend on them downstream. |
| **NEVER** reorder the output format (`lex:` before `hyde:`) | Reward function deducts heavily; downstream QMD parser expects `hyde:` first |
| **NEVER** push a model without eval results in the model card | Blocker for deployment |
| **NEVER** run `train.py grpo` expecting production output | GRPO lives in `experiments/grpo/` — see "Production path is SFT-only" note in `README.md` |
| **NEVER** use `<think>` mode in Qwen3 prompts | The `/no_think` directive is mandatory. `<think>` triggers CoT which breaks the output format. |

## UNIQUE STYLES

- **`reward.py` is the single source of truth** for what "good expansion" means. Five dimensions:
  - Format (0-30): has `lex/vec` lines, no invalid lines
  - Diversity (0-30): multiple types, no query echoes
  - HyDE (0-20): 50-200 chars, single line, not repetitive
  - Quality (0-20): `lex` shorter than `vec`, natural language
  - Entity (-45 to +20): named entities preserved (negative scoring for loss)
  - Think bonus (0-20): reward for NOT using `<think>` mode
  - Max: 140 with hyde, 120 without
- **Hard failures (instant 0.0):** chat template leakage (`<|im_start|>`, `<|im_end|>`), any line without valid `lex:`/`vec:`/`hyde:` prefix.
- **`experiments/` is a sandbox** — three independent tracks:
  - `gepa/` — DSPy-based prompt optimization
  - `grpo/` — experimental RL recipe (NOT in production path as of v2.6.3)
  - `lfm2/` — LiquidAI LFM2-1.2B alternative (hybrid arch, faster inference)
- **`jobs/` scripts are self-contained** — they re-import shared code from `jobs/eval_common.py` but don't depend on the parent `finetune/` package being installed. Each can run standalone on HuggingFace Jobs hardware.
- **`convert_onnx.py` is an alternative export path** (not GGUF). Used for browser-deployable runtimes (Transformers.js, ONNX Runtime Web). Not currently in production.

## NOTES

- **Cross-stack contract:** Output of this pipeline → `tobil/qmd-query-expansion-1.7B-gguf`. `src/llm.ts:DEFAULT_GENERATE_MODEL` references that repo. Changing the output format (e.g. adding a new prefix beyond `lex/vec/hyde`) requires coordinated updates in both repos.
- **Eval scores are the deploy gate.** Only push to `tobil/qmd-query-expansion-1.7B-gguf` if eval improves over current deployed model. Always attach eval JSON to the HF model card.
- **`SCORING.md` is the rubric reference.** Read it before editing `reward.py` — the deductions are subtle and interlocking.
- **`eval_retrieval.py`** is a separate retrieval-quality eval (end-to-end search quality), distinct from `eval.py` (which scores format compliance via `reward.py`). Both matter; they measure different things.
- **`train_unsloth.py`** uses Unsloth for faster LoRA training on consumer GPUs. Falls back to standard TRL `SFTTrainer` if Unsloth is unavailable.
- **See also:** `CLAUDE.md` (canonical contributor guide), `SCORING.md` (rubric), `../AGENTS.md` (root).
