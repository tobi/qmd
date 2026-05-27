# clinical-qmd

A clinical fork of [`qmd`](https://github.com/tobi/qmd) — on-device hybrid retrieval over a physician's local notes, protocols, lab results, and references.

## Why fork

Upstream qmd is already an excellent on-device retrieval engine: SQLite FTS5 + sqlite-vec + local LLM query expansion and reranking via `node-llama-cpp`, exposed as a CLI and an MCP server. Everything runs locally on the device, which is exactly the property a clinical tool needs: **PHI must never leave the laptop**.

The fork adds clinical-specific layers on top of upstream's primitives:

| Layer | What it adds | Where it lives |
| --- | --- | --- |
| Schema | Clinical front-matter (note type, encounter date, problem list, ICD/SNOMED, meds, labs) parsed into queryable sidecar table | `src/clinical/schema.ts`, `src/clinical/parse.ts` |
| Model defaults | Clinical-friendly embedder + reranker + query-expansion model | `src/clinical/defaults.ts`, config preset |
| Query expansion | Clinical synonym and abbreviation expansion (MI ↔ myocardial infarction, BID ↔ twice daily, etc.) | `src/clinical/expansion.ts` |
| MCP tools | `find_recent_labs`, `summarize_problem_list`, `pull_similar_cases`, `cite_with_source` | `src/clinical/mcp/*` |
| Training data | Clinical query → `hyde:/lex:/vec:` examples for fine-tuning the query-expansion model | `finetune/data/clinical/*.jsonl` |
| Eval harness | Clinical retrieval tasks scored against ground truth (right note, right lab, right protocol) | `finetune/evals/clinical/` |
| PHI guardrails | Optional de-id pass on import, audit log of every query, no-network mode | `src/clinical/phi/*` |

## Non-goals

- Not an EHR. Notes are read-only inputs.
- Not a clinical decision support tool. Retrieval only; the physician interprets.
- Not a SaaS. Everything runs on-device. There is no server to hit.

## Staying in sync with upstream

Upstream is added as the `upstream` git remote and pulled periodically.

```bash
git fetch upstream
git merge upstream/main          # or: git rebase upstream/main on feature branches
```

Clinical-specific code lives under `src/clinical/`, `finetune/data/clinical/`, `finetune/evals/clinical/`, and `clinical/` (docs). This keeps merge conflicts isolated to the layers we own.

## Model swap

Upstream is already model-agnostic — any GGUF works via env vars:

```bash
QMD_EMBED_MODEL=hf:org/your-clinical-embedder/model.gguf
QMD_RERANK_MODEL=hf:org/your-clinical-reranker/model.gguf
QMD_GENERATE_MODEL=hf:org/your-clinical-expansion-model/model.gguf
```

The clinical defaults will be set in `src/clinical/defaults.ts` and can be a clinical embedder (e.g. a Med-E5 GGUF), a clinical reranker, and a query-expansion model fine-tuned on clinical queries using the existing `finetune/` pipeline.

## Roadmap

1. **M1 — Scaffold + schema** (current): fork, dev loop, clinical front-matter sidecar table, schema spec. ← we are here
2. **M2 — Clinical front-matter parsing in indexer**: parse the schema fields out of YAML front-matter into the sidecar table, expose them in query results.
3. **M3 — Clinical defaults + config preset**: clinical-tuned embedder + reranker + expansion model selected automatically when `QMD_PROFILE=clinical`.
4. **M4 — Autoresearch training loop for clinical queries**: clinical training data under `finetune/data/clinical/`, eval set under `finetune/evals/clinical/`, fine-tuned query-expansion model published to HF.
5. **M5 — Clinical MCP tools**: `find_recent_labs`, `summarize_problem_list`, etc.
6. **M6 — PHI guardrails + audit log**: optional de-id pass on import, queryable audit log.

## Compliance posture

clinical-qmd is a local tool — no network egress for indexing or querying. This means:

- It does not transmit PHI off-device.
- It does not require a BAA for itself (there is no service).
- Models are downloaded once from Hugging Face and then run locally.
- Audit logs (M6) record query text and timestamps locally so the physician can produce an access log if asked.

This is not the same as being "HIPAA-certified" — HIPAA is a regulation that applies to covered entities, not software. clinical-qmd is designed to be **compatible with a HIPAA-compliant workflow**: a covered-entity physician can use it on a managed device with full-disk encryption and an organizational BAA covering the device, and clinical-qmd itself never widens the trust boundary.
