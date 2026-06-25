# QMD Remote-LLM Port — Specification (SDD)

| Field | Value |
| --- | --- |
| Spec ID | `qmd-remote-llm-port` |
| Status | Ready for implementation (spec contract; open questions resolved) |
| Author | Coda (analysis of `lauerprojects/qmd-remote-llm.git` `@6b1336c`) |
| Target | `alauer/qmd.git` mainline (HEAD `e428df7`, v2.6.3 at time of analysis) |
| Source feature branch | `lauerprojects/qmd-remote-llm.git` `remote-llm` (HEAD `6b1336c`) |
| Estimated effort | 12–14 hours (one focused day, mostly mechanical port + verification) |
| Risk | Medium (interface shape divergence; one non-obvious behaviour gap on rerank) |

## SPEC

### Problem statement

QMD's local stack (`node-llama-cpp` + GGUF models) is great for desktop users with
modern GPUs/CPUs but is the wrong default for several real-world deployment shapes:

1. **CI / containerized indexers** — no GPU, no Metal, cold-start model downloads blow
   up cache budgets. Most CI just needs embeddings.
2. **Apple Silicon users who want speed** — downloading a 1.2B reranker takes minutes;
   a remote API rerank call costs ~150ms and zero local RAM.
3. **Fleet deployments (corporate, regulated, or managed)** — operators want a single
   API key, not GGUF model management per device.
4. **Users on low-RAM machines** — the 1.7B query-expansion model is the largest
   allocation QMD makes; offloading it removes the dominant memory cost.

The `lauerprojects/qmd-remote-llm.git` fork (`remote-llm` branch, HEAD `6b1336c`)
already solves all four. The work here is to port that fork back into mainline without
regressing anything that works today, and without widening the public surface more
than necessary.

### Goals

- **G1.** Allow embeddings, query expansion, and reranking to be routed independently
  to any OpenAI-compatible API (`/embeddings`, `/chat/completions`).
- **G2.** Fall back to local `node-llama-cpp` automatically when no API key is set —
  zero-config users see no behaviour change.
- **G3.** Per-operation backend selection (`QMD_EMBED_BACKEND`, `QMD_GENERATE_BACKEND`,
  `QMD_RERANK_BACKEND`, `QMD_TOKENIZE_BACKEND`) so a user can mix local reranking
  with remote embeddings if they want.
- **G4.** Listwise LLM reranking on the remote backend that scores 15 documents per
  prompt with min-max normalization per batch, cached in SQLite exactly like the
  local path.
- **G5.** Cache SQLite compatibility — a remote-embedding index must round-trip with
  the local cache schema (`llm_cache`, `vectors_vec`) without migration.
- **G6.** Bring the existing fork's test suite (`remote-llm.test.ts`,
  `hybrid-llm.test.ts`) over so the new code paths are exercised in CI.
- **G7.** Widen the `LLM` interface from 6 to 9 methods (add `embedBatch`,
  `tokenize`, `detokenize`). This is the smallest widening that lets
  `chunkDocumentByTokens`, the embed batch path, and the chunked rerank call site
  all go through the same `getDefaultLLM()` factory. `getDeviceInfo()` stays a
  concrete method on `LlamaCpp`/`HybridLLM` (CLI status only).

### Non-goals

- **N1.** Not adding new vector stores or backends (LanceDB, Qdrant, etc.).
- **N2.** Not changing the RRF fusion math or rerank-blend weights.
- **N3.** Not porting the reward-design / GRPO training harness (out of scope, not
  in the `remote-llm` branch's net diff vs main).
- **N4.** Not adding streaming responses. RemoteLLM uses `complete()` from
  `@mariozechner/pi-ai` which is non-streaming; the local path is also non-streaming.
- **N5.** Not switching the default local reranker. The fork changed
  `DEFAULT_RERANK_MODEL` from `Qwen3-Reranker-0.6B` to `ms-marco-MiniLM-L6-v2-F16`
  in the fork only; mainline keeps Qwen3 as the local default.
- **N6.** Not changing the `LLMSession` lifecycle or session manager.
- **N7.** Not bumping mainline's version number — that's a release concern; this
  port is code-only.

### User impact

- **Existing users with no env vars:** zero change. Local `node-llama-cpp` path
  unchanged.
- **CI / container users:** set `QMD_REMOTE_API_KEY` + `QMD_REMOTE_BASE_URL`, skip
  the 1.7B GGUF download.
- **Power users:** mix backends per-operation; rerank-quality-sensitive workflows
  keep local Qwen3, indexing pipelines go remote for throughput.
- **MCP users (`qmd mcp`):** no surface change — the MCP server already runs the
  same `searchVec` / `rerank` / `expandQuery` codepaths that the new
  HybridLLM threads through.

### Success criteria

- **S1.** All current mainline tests still pass after the port (`vitest` run against
  `test/`).
- **S2.** The 612 lines of fork tests (`remote-llm.test.ts` 414 + `hybrid-llm.test.ts`
  198) pass against the ported code.
- **S3.** `qmd query "test"` with no env vars produces identical results to current
  mainline HEAD on the test corpora (the `eval.test.ts` suite).
- **S4.** `QMD_REMOTE_API_KEY=sk-… QMD_REMOTE_BASE_URL=http://localhost:11434/v1`
  with an Ollama container running makes `qmd embed` and `qmd query` succeed end
  to end.
- **S5.** `QMD_DEBUG_RERANK=1 qmd query "..."` prints pre/post rerank order with
  parse diagnostics.
- **S6.** Switching `QMD_EMBED_BACKEND` from `remote` back to `local` requires
  `qmd embed -f` and a model swap, and the README documents that explicitly.

## DESIGN

### Architecture

The fork uses **Strategy + Decorator** via `HybridLLM`. That is the right call and
stays:

```
                   ┌───────────────────────────────────────┐
                   │ getDefaultLLM(): LLM (interface)      │
                   └───────────────┬───────────────────────┘
                                   │ returns
                                   ▼
                   ┌───────────────────────────────────────┐
                   │ HybridLLM (routes per operation)      │
                   │   - embedBackend  → getBackend()      │
                   │   - generateBackend                  │
                   │   - rerankBackend                    │
                   │   - tokenizeBackend                  │
                   └───────┬───────────────────────────────┘
                           │ delegates
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────────┐   ┌────────────────────────────┐
   │ LlamaCpp            │   │ RemoteLLM (if API key set) │
   │  (node-llama-cpp)   │   │  (fetch + pi-ai complete)  │
   │  DEFAULT            │   │  OPTIONAL                  │
   └─────────────────────┘   └────────────────────────────┘
```

Three concrete classes, one interface. Each backend is independently testable via
`vi.mock`. The wrapper is the only place that knows about routing.

### Interface shape — final decision

Mainline's current `LLM` interface (in `src/llm.ts` line 521) has **6 methods**:

```typescript
embed, generate, modelExists, expandQuery, rerank, dispose
```

The fork's `llm-types.ts` widened this to **10 methods** by adding:

```typescript
embedBatch, tokenize, detokenize, getDeviceInfo
```

**Final decision (locked): the mainline `LLM` interface becomes 9 methods.**
Concretely: current 6 + `embedBatch` + `tokenize` + `detokenize`.
`getDeviceInfo()` stays a concrete method on `LlamaCpp` and `HybridLLM` (CLI
status calls it directly on the concrete singleton; it is not part of the
`LLM` contract).

Rationale:

- `embedBatch` is already implemented on `LlamaCpp` in mainline
  (`src/llm.ts:214`); promoting it to the interface is a no-op for the local
  path and is required for `HybridLLM` to expose it without `instanceof` checks.
- `tokenize` / `detokenize` are called from `chunkDocumentByTokens`
  (`src/store.ts:2795, 2832`). Today that call site works because it uses
  `getDefaultLlamaCpp()` (concrete type) and a strict cast; the port moves
  those call sites onto `getDefaultLLM()` and adds `tokenize`/`detokenize` to
  the interface so `HybridLLM` is a true `LLM` rather than a partial duck-type.
- `getDeviceInfo` has a single call site (`qmd status`). Adding it to the
  interface would force `RemoteLLM` to fake GPU/Metal/VRAM values it has no
  way to know. Keep it concrete-class-only and have `qmd status` cast
  through `getDefaultLlamaCpp()` (or read it from the `HybridLLM.local`
  field when only the wrapper is available).

This is the only public-surface change in the port. All other fork methods
are either already in mainline or reach the LLM via concrete calls that
`HybridLLM` proxies cleanly.

### Files to create (6)

| File | Purpose | LOC (fork) |
| --- | --- | --- |
| `src/llm-types.ts` | Shared types — `LLM` interface (9 methods), `EmbeddingResult`, `GenerateResult`, `RerankResult`, `ModelInfo`, options, etc. | 171 |
| `src/remote-llm.ts` | `RemoteLLM implements LLM`. fetch-based `/embeddings` and `@mariozechner/pi-ai` `complete()` for chat; listwise rerank with batch-15 prompt. | 423 |
| `src/hybrid-llm.ts` | `HybridLLM implements LLM`. Routes per operation, falls back to local when remote is missing. | 109 |
| `scripts/debug-config.ts` | `npx tsx scripts/debug-config.ts` prints backend env state. | 32 |
| `test/remote-llm.test.ts` | Mocks `fetch` and `pi-ai`. Covers all 9 public methods, batch rerank, JSON parse fallbacks, listwise scoring math. | 414 |
| `test/hybrid-llm.test.ts` | Mocks the `LlamaCpp` constructor. Asserts routing to local / remote / fallback, `modelExists` checks both backends, `getDeviceInfo` returns local info. | 198 |

### Files to modify (7)

| File | Change |
| --- | --- |
| `src/llm.ts` | Extract types into `llm-types.ts` (re-export from `llm.ts` for back-compat). Add `getDefaultLLM()` factory that returns `HybridLLM`. Keep `getDefaultLlamaCpp()` as a strict cast (throws if remote-only). Update `withLLMSession()` to detect `HybridLLM`/`RemoteLLM` and route to a thin proxy session. Add `dotenv/config` import at top. |
| `src/store.ts` | Switch call sites from `getDefaultLlamaCpp()` to `getDefaultLLM()`. Pass `model: undefined` instead of `DEFAULT_EMBED_MODEL` so each backend uses its own configured model. Stop passing `{ model }` to `rerank()` for the same reason — see `DESIGN` note about rerank-model passing below. |
| `src/qmd.ts` | `disposeDefaultLlamaCpp` → `disposeDefaultLLM`. `getDefaultLlamaCpp()` → `getDefaultLLM()` in the `status` command for `getDeviceInfo()`. Add `import "dotenv/config"` at top. Move the "Model: …" log line out of the `vectorIndex` pre-call (the fork moves it to be emitted after the first successful batch, which is the right fix anyway). |
| `src/mcp/server.ts` | `disposeDefaultLlamaCpp` → `disposeDefaultLLM` (only one occurrence, in the SIGTERM handler). |
| `src/test-preload.ts` | `disposeDefaultLlamaCpp` → `disposeDefaultLLM`. |
| `package.json` | Add deps: `@mariozechner/pi-ai` (^0.55.4), `dotenv` (^17.3.1). Do **not** bump version. |
| `.gitignore` | Append `.env`. |

### Configuration surface (env vars)

All new; no defaults changed for existing users.

| Variable | Default | Purpose |
| --- | --- | --- |
| `QMD_REMOTE_API_KEY` | — | Bearer token. When unset, `RemoteLLM` is not constructed; everything stays local. |
| `QMD_REMOTE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base. OpenRouter, OpenAI, Ollama, vLLM, LiteLLM all work. |
| `QMD_REMOTE_EMBED_MODEL` | `text-embedding-3-small` | Embedding model id on the remote API. |
| `QMD_REMOTE_GENERATE_MODEL` | `openai/gpt-3.5-turbo` | Chat model used for query expansion. |
| `QMD_REMOTE_RERANK_MODEL` | — | Optional. If set, `QMD_RERANK_BACKEND` defaults to `remote`; otherwise local Qwen3. |
| `QMD_REMOTE_TIMEOUT` | `60000` | Per-request timeout in ms. |
| `QMD_EMBED_BACKEND` | `remote` if API key set, else `local` | Routes `embed`/`embedBatch`. |
| `QMD_GENERATE_BACKEND` | `remote` if API key set, else `local` | Routes `generate`/`expandQuery`. |
| `QMD_RERANK_BACKEND` | `remote` if `QMD_REMOTE_RERANK_MODEL` set, else `local` | Routes `rerank`. |
| `QMD_TOKENIZE_BACKEND` | `local` | Routes `tokenize`/`detokenize`. Default local because tokenization drives chunking heuristics that depend on the model's actual tokenizer. |
| `QMD_RERANK_CHUNK_CHARS` | `1200` | Max chars per chunk sent to remote reranker. |
| `QMD_DEBUG_RERANK` | — | Set `1` to print pre/post rerank order, prompt, raw response, parse errors. |

`.env` files in CWD are auto-loaded via `import "dotenv/config"` at the top of
`src/llm.ts` and `src/qmd.ts`.

### Compatibility with current mainline interfaces

| Public surface | Before port | After port |
| --- | --- | --- |
| `LLM` interface | 6 methods | 9 methods (adds `embedBatch`, `tokenize`, `detokenize`) |
| `ILLMSession` interface | unchanged | unchanged |
| `getDefaultLlamaCpp()` | returns `LlamaCpp` | returns `LlamaCpp` (throws if HybridLLM's local isn't `LlamaCpp`); aliased to `disposeDefaultLLM` |
| `getDefaultLLM()` | does not exist | new; returns `HybridLLM` |
| `disposeDefaultLlamaCpp()` | exists | alias for `disposeDefaultLLM()` |
| `LlamaCpp` class | unchanged | unchanged |
| `RemoteLLM` class | does not exist | new |
| `HybridLLM` class | does not exist | new |
| MCP server entrypoint | `bin/qmd mcp` runs `src/mcp/server.ts` | unchanged; transparent |
| CLI exit | `disposeDefaultLlamaCpp` | `disposeDefaultLLM` |
| Default models (`DEFAULT_EMBED_MODEL`, `DEFAULT_RERANK_MODEL`, `DEFAULT_GENERATE_MODEL`) | Qwen3 / embeddinggemma / qmd-query-expansion | unchanged |

### Rerank model passing — non-obvious gotcha

Mainline currently calls:

```typescript
const llm = getDefaultLlamaCpp();
const rerankResult = await llm.rerank(query, uncachedDocs, { model });  // model = DEFAULT_RERANK_MODEL
```

The fork correctly changes this to:

```typescript
const llm = getDefaultLLM();
const rerankResult = await llm.rerank(query, uncachedDocs, {});  // no model
```

**Why this matters:** if you pass `DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-..."`
to a `RemoteLLM` instance, `RemoteLLM.resolvePiModel()` will hand that HuggingFace
URI to `pi-ai`'s OpenAI registry, which doesn't know it. The `if (this.baseURL !==
"https://api.openai.com/v1")` branch in `resolvePiModel` *does* catch this and
synthesizes an OpenAI-compatible model object — but with a contextWindow and
maxTokens of 128000 / 4096, which is just wrong. The clean fix is to not pass a
model and let each backend use its own configured one. **Port carries this fix.**

### Re-rank debug logs

Fork adds `QMD_DEBUG_RERANK` instrumentation in `src/store.ts`'s `rerank()` and
`src/remote-llm.ts`'s `rerank()`. Both are ported verbatim. Useful for triaging
"why did my remote rerank move this doc to the top?".

### Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Existing tests reference `getDefaultLlamaCpp()` and rely on it returning `LlamaCpp` for setup | Low | Low | Alias kept; tests continue to work via `setDefaultLLM(new LlamaCpp())`. |
| Vector cache schema drift between local and remote embeddings | High | High | README warns user to `qmd embed -f` after backend switch; no schema change attempted. |
| `@mariozechner/pi-ai` adds a heavy dep | Medium | Low | `pi-ai` is the standard abstraction used across the ecosystem; tree-shakes well; only loaded when `RemoteLLM` is constructed. |
| `RemoteLLM` JSON parse on listwise rerank occasionally fails | Medium | Low | Already handled in the fork: per-batch parse errors leave the batch at `0.5` (neutral) and continue. `QMD_DEBUG_RERANK=1` exposes parse diagnostics. |
| Widening `LLM` interface breaks external consumers that `implements LLM` | Low | Medium | In-tree: only `LlamaCpp` implements it. Out-of-tree: the change is additive (3 methods, all with default-able signatures). Documented in CHANGELOG. |
| `dotenv/config` overwrites process env from a stale `.env` | Low | Medium | `dotenv` defaults to non-overriding mode; explicitly set no override. README says `.env` is for setup convenience, real config via shell. |
| `chunkDocumentByTokens` regression when `HybridLLM.tokenize` proxies to remote | Low | Low | `QMD_TOKENIZE_BACKEND` defaults to `local`; `RemoteLLM.tokenize` is a regex approximation but is unreachable unless the user explicitly sets it. |
| `RemoteLLM` base URL validation: OpenRouter-specific headers | Low | Low | Fork already includes `HTTP-Referer` and `X-Title` which OpenRouter recommends; harmless on OpenAI / Ollama. |
| `pi-ai`'s `complete()` returns content blocks with `type: 'thinking'` for reasoning models | Low | Low | Fork already filters: `.filter(b => b.type === 'text').map(b => b.text).join('')`. |

## IMPLEMENTATION

### Step-by-step port plan

1. **Branch & scaffold** (10 min)
   - Branch: `feat/qmd-remote-llm-port` off mainline `main` (this is an
     implementation branch, not a docs branch — the spec already lives on
     `docs/qmd-remote-llm-sdd-spec`).
   - Add `docs/qmd-remote-llm-port.md` (this file, from the docs branch — bring
     it across with `git checkout docs/qmd-remote-llm-sdd-spec -- docs/qmd-remote-llm-port.md`
     or copy contents).
   - `mkdir -p scripts/`.

2. **Extract types into `src/llm-types.ts`** (45 min)
   - Copy fork's `src/llm-types.ts` (171 lines).
   - From `src/llm.ts`, remove the inline type definitions (lines 123–267 roughly)
     and `export type { ... } from "./llm-types.js"` so existing imports
     (`type LLM`, `type EmbedOptions`, etc.) keep working without churn.
   - Verify: `grep -rn "from \"./llm.js\"" src/` — no callsite change required.

3. **Add `src/remote-llm.ts`** (90 min, mostly straight copy)
   - Copy fork's `src/remote-llm.ts` (423 lines) verbatim.
   - Re-export `RemoteLLMConfig` and `DEFAULT_BASE_URL`.
   - Note: the file imports from `@mariozechner/pi-ai` — that dep is added in
     step 7.

4. **Add `src/hybrid-llm.ts`** (45 min)
   - Copy fork's `src/hybrid-llm.ts` (109 lines) verbatim.
   - Update `LLMBackend` import — `LLM` interface lives in `llm-types.ts` now.

5. **Refactor `src/llm.ts`** (90 min — non-trivial)
   - Add imports: `RemoteLLM` from `./remote-llm.js`, `HybridLLM` and `LLMBackend`
     from `./hybrid-llm.js`.
   - Add `import "dotenv/config"` at the very top of the file (so env vars are
     loaded before the singleton factory runs).
   - Add `getDefaultLLM()` factory (mirror fork lines 1316–1352): reads
     `QMD_REMOTE_API_KEY` and friends, constructs `LlamaCpp` always, conditionally
     constructs `RemoteLLM`, wraps in `HybridLLM` with the four backend env vars.
   - Add `setDefaultLLM(llm: LLM | null)` for tests.
   - Add `disposeDefaultLLM()` (calls `defaultLLM?.dispose()` and clears the
     singleton).
   - Keep `getDefaultLlamaCpp()` as a strict cast (fork lines 1374–1380). It throws
     if `defaultLLM` isn't a `LlamaCpp`, which is the correct safety net for
     `chunkDocumentByTokens`.
   - Keep `disposeDefaultLlamaCpp` as `export const disposeDefaultLlamaCpp = disposeDefaultLLM`
     for back-compat (alias, fork line 1382).
   - Update `withLLMSession()` to dispatch on type (fork lines 1256–1295):
     - If `LlamaCpp`: existing session manager.
     - If `HybridLLM`: thin ILLMSession proxy.
     - If `RemoteLLM`: thin ILLMSession proxy (with `embedBatch` falling back to
       `Promise.all(texts.map(t => llm.embed(t)))` since fork's `RemoteLLM`
       already has `embedBatch` but the narrow `LLM` interface doesn't).
   - Widen `LLM` interface by 3 methods: add `embedBatch`, `tokenize`,
     `detokenize` (mirroring fork's `llm-types.ts`). `getDeviceInfo` stays
     concrete-class-only.

6. **Update `src/store.ts`** (45 min)
   - Change import from `getDefaultLlamaCpp` to `getDefaultLLM`.
   - 5 call sites change from `getDefaultLlamaCpp()` to `getDefaultLLM()`:
     `chunkDocumentByTokens` (line 1429 in fork), `getEmbedding` (line 2242),
     `expandQuery` (2310), `rerank` (2352), `hybridQuery` (2984),
     `structuredSearch` (3273).
   - In `searchVec` (line 2152), change parameter `model: string` → `model: string | undefined`.
   - In `hybridQuery` (line 3025) and `structuredSearch` (line 3315), pass
     `undefined` instead of `DEFAULT_EMBED_MODEL` so the backend uses its own
     configured embed model.
   - In `rerank` (line 2369), pass `{}` instead of `{ model }` to `llm.rerank(...)`.
   - Add `QMD_DEBUG_RERANK` logging around `rerank()` (fork lines 2354–2363 and
     2386–2396). Pure addition.

7. **Update `src/qmd.ts`** (15 min)
   - Add `import "dotenv/config"` at top.
   - Change import: `disposeDefaultLlamaCpp, getDefaultLlamaCpp` →
     `disposeDefaultLLM, getDefaultLLM`.
   - 2 call sites: `showStatus()` `getDeviceInfo()` (line 401), final cleanup
     (line 2897).
   - In `vectorIndex` (line 1596 onwards): move the `Model: ${model}` log to
     *after* the first successful batch (fork lines 1603–1665). This is a real
     bugfix in the fork — pre-fork, the model string is the local default but
     the actual model used may be the remote one. The fork's fix is correct.

8. **Update `src/mcp/server.ts`** (5 min)
   - Change import: `disposeDefaultLlamaCpp` → `disposeDefaultLLM`.
   - One occurrence: SIGTERM handler (fork of `mcp.ts` line 717 maps to
     `src/mcp/server.ts` ~equivalent line; search for the exact line with grep).

9. **Update `src/test-preload.ts`** (5 min)
   - `disposeDefaultLlamaCpp` → `disposeDefaultLLM`.

10. **`package.json`** (5 min)
    - Add to `dependencies`: `"@mariozechner/pi-ai": "^0.55.4"`, `"dotenv": "^17.3.1"`.
    - Do **not** bump version.

11. **`.gitignore`** (1 min)
    - Append: `.env`.

12. **Add `scripts/debug-config.ts`** (5 min)
    - Copy fork's 32-line file verbatim.

13. **Bring over tests** (45 min)
    - Copy `test/remote-llm.test.ts` (414 lines) and `test/hybrid-llm.test.ts`
      (198 lines) verbatim. Both mock their deps, so they don't hit the network.
    - Update test imports if necessary (most reference `../src/remote-llm` and
      `../src/hybrid-llm` which both exist after step 3/4).

14. **Run the test matrix** (60 min)
    - `npm install` (or `bun install`).
    - `npm run test:types` — strict tsc.
    - `npm run test:unit` — vitest on `test/`.
    - `npm run test:bun` — bun test on `test/`.
    - `npm run test:package` — smoke test the built package.

15. **Update README** (30 min)
    - Section "Remote LLM & Embeddings" (155 lines in fork). Place between the
      existing "Database Schema" and "Environment Variables" sections.
    - Update the lead paragraph: "By default everything runs locally via
      node-llama-cpp with GGUF models; embeddings, query expansion, and
      reranking can optionally be routed to any OpenAI-compatible API."
    - Update the architecture diagram's reranker row.
    - Update the score-range table to add the remote rerank row.
    - Append 10 rows to the Environment Variables table.

16. **Commit, push, open MR** (10 min)
    - One commit per logical layer (types, remote-llm, hybrid-llm, llm.ts refactor,
      store.ts, qmd.ts, mcp+preload, tests, debug script, README) — or one squash
      commit if the MR is small enough to review as a unit. Recommendation:
      **squash** because the layers have non-trivial interdependencies that read
      better as a single change.
    - Branch: `feat/qmd-remote-llm-port` (per step 1).
    - Push to `alauer/qmd.git`. Open MR against `main` with this spec document
      linked in the description.

Total: roughly 8 hours focused + 4 hours slack for test failures and review fixes.

### Tests to add / run

**New (ported from fork):**
- `test/remote-llm.test.ts` — 414 lines, ~25 `it()` cases. Covers:
  - embed single + batch
  - generate happy path + thinking-block filtering + error path
  - modelExists always-true stub
  - expandQuery with JSON parse fallback
  - rerank listwise batching, JSON parse errors, min-max normalization
  - tokenize/detokenize regex
  - dispose no-op
- `test/hybrid-llm.test.ts` — 198 lines, ~12 `it()` cases. Covers:
  - routing to local when `backend: 'local'`
  - routing to remote when `backend: 'remote'`
  - warn-and-fall-back to local when remote is undefined
  - `modelExists` checks both backends
  - `getDeviceInfo` returns local info

**Existing — must remain green:**
- `test/llm.test.ts` (1550+ lines) — LlamaCpp + session manager; should not be
  touched, but the `setDefaultLlamaCpp` helper still works via the alias.
- `test/store.test.ts` (4148 lines) — exercises `chunkDocumentByTokens` and
  `rerank` which are the most-likely-to-regress call sites.
- `test/mcp.test.ts` — exercises MCP server lifecycle including SIGTERM.
- `test/eval.test.ts` — quality regression on BM25 / vector / hybrid corpora.
- `test/cli.test.ts`, `test/cli-lazy-llm-import.test.ts` — CLI behaviour.

### Verification commands

```sh
# 1. Type check
npm run test:types

# 2. Unit + bun tests
npm run test:unit

# 3. Package smoke
npm run test:package

# 4. With NO env vars (default local path) — must behave identically to pre-port
unset QMD_REMOTE_API_KEY
npm run test:unit       # vitest + bun on test/ (covers eval-style assertions)
npm run test:package    # smoke test the built package
# `npm run test:eval` does not exist in mainline; the closest equivalent is
# `test:unit` (which runs every vitest + bun test under test/, including any
# eval-style corpora). Behavioural parity on the test corpora is enforced by
# S1 + S2 in the success criteria.

# 5. With an Ollama container running (full remote path)
docker run -d -p 11434:11434 --name ollama-test ollama/ollama
docker exec ollama-test ollama pull nomic-embed-text
docker exec ollama-test ollama pull qwen2.5:1.5b

export QMD_REMOTE_API_KEY="ollama"
export QMD_REMOTE_BASE_URL="http://localhost:11434/v1"
export QMD_REMOTE_EMBED_MODEL="nomic-embed-text"
export QMD_REMOTE_GENERATE_MODEL="qwen2.5:1.5b"
export QMD_REMOTE_RERANK_MODEL="qwen2.5:1.5b"
export QMD_EMBED_BACKEND=remote
export QMD_GENERATE_BACKEND=remote
export QMD_RERANK_BACKEND=remote
export QMD_DEBUG_RERANK=1

# 6. Index a small corpus
mkdir -p /tmp/qmd-corpus && cd /tmp/qmd-corpus
for i in {1..20}; do echo "# Doc $i\n\nSome content about topic $i." > "doc-$i.md"; done
cd /workspace/mainline
node bin/qmd embed -c /tmp/qmd-corpus
node bin/qmd query "topic 5" -c /tmp/qmd-corpus
# Expect: doc-5.md at the top, debug log on stderr.

# 7. MCP smoke
node bin/qmd mcp &
PID=$!
sleep 2
kill -TERM $PID
# Expect: clean shutdown, no GGML_ASSERT (since local path not exercised).
```

### Rollout / fallback plan

**Phase 1 — internal preview (this MR):**
- Land on a feature branch.
- Land the port + tests + README.
- Run the verification commands above locally and on CI.

**Phase 2 — canary (optional):**
- If the maintainer wants a canary, tag the merge commit `v2.7.0-rc.1` and ship
  to npm under a `next` dist-tag for ~1 week.
- Collect feedback on whether the `RemoteLLM` listwise rerank quality matches
  Qwen3 local.

**Phase 3 — stable release:**
- Merge to `main`.
- Bump version (`v2.7.0` suggested) in a follow-up commit; release notes call
  out: "Optional remote-LLM backend via `QMD_REMOTE_API_KEY`. Local behaviour
  unchanged."

**Rollback:**
- Revert the merge commit. `getDefaultLLM()` was new, so removing it restores
  the pre-port `getDefaultLlamaCpp()`-only world.
- If only the HybridLLM facade has a bug, it can be disabled by setting
  `QMD_EMBED_BACKEND=local`, `QMD_GENERATE_BACKEND=local`,
  `QMD_RERANK_BACKEND=local`, and the code path through `HybridLLM.getBackend()`
  short-circuits to local for every operation. No redeploy needed.

**No-data-loss guarantee:**
- The `llm_cache` SQLite table is unchanged.
- The `vectors_vec` schema is unchanged.
- Vector dimension changes (between embedgemma 768-d and OpenAI text-embedding-3-small
  1536-d) require `qmd embed -f`; this is the same constraint that exists today
  if you swap local models. Documented in README.

### Decisions (locked before merging)

- **D1. `LLM` interface width.** **9 methods** — current 6 + `embedBatch` +
  `tokenize` + `detokenize`. `getDeviceInfo` stays concrete-only on
  `LlamaCpp` and `HybridLLM`. See the *Interface shape* section above for
  the full rationale. The interface in `src/llm-types.ts` must be exactly
  these 9 signatures; the implementation worker should not introduce
  additional methods (no `complete`, no `stream`, no `countTokens`).
- **D2. Public exports.** `RemoteLLM` and `HybridLLM` are exported from
  `src/index.ts` so advanced users can construct their own compositions.
  Cost of exporting is zero; benefit is real (lets users wire remote + local
  per call without going through env vars).
- **D3. Rerank default.** When `QMD_REMOTE_RERANK_MODEL` is unset but a
  remote API key is present, rerank defaults to **local Qwen3**. Listwise
  LLM rerank is a quality regression on most small chat models; this is
  the right conservative default.
- **D4. Debug-config script wiring.** Add a `package.json` script
  `debug-config` that runs `tsx scripts/debug-config.ts` so users can
  `npm run debug-config` (or `pnpm run debug-config`). Cosmetic, but
  cheap.

### Implementation contract & guardrail

This document is the **binding contract** for the implementation worker. The
implementation branch (`feat/qmd-remote-llm-port`) MUST satisfy every section
above as written. Specifically:

1. **No invention beyond the spec.** If the implementation worker needs a new
   file, a new env var, a new `LLM` method, a new dep, or a schema change
   that is not listed in *Files to create (6)*, *Files to modify (7)*,
   *Configuration surface*, or *Compatibility with current mainline interfaces*,
   it must **block** the task and surface the discrepancy here, not patch the
   spec in passing.
2. **Repo evidence wins on conflict.** If mainline has moved between this
   spec's reference HEAD (`e428df7`, v2.6.3) and the implementation branch's
   tip — e.g. a line number referenced in *Step-by-step port plan* no longer
   matches, a symbol was renamed, or a function was extracted — the worker
   must verify by `grep` / `rg` against the live tree, then either:
   - apply the change against the live code (and call out the diff in the MR
     description), or
   - **block** the task if the divergence is non-trivial (e.g. the LLM
     interface has already been widened, a method is gone, or a call site
     has been removed entirely).
3. **Ambiguity is a blocker, not a guess.** Any wording in this spec that the
   worker cannot resolve in 5 minutes of code reading must trigger
   `kanban_block`, not silent interpretation. Examples that should block:
   "does the rerank path use a separate `ILLMSession`?" (not specified —
   block), "should `withLLMSession` close over a `RemoteLLM`? (yes for
   `embed`, but `rerank` may not need one — block).
4. **One final interface.** Do not introduce a temporary 6-method `LLM`
   interface with the intent of widening to 9 in a follow-up. Land the 9-method
   interface on the first commit. The diff is small enough.
5. **Tests are part of the contract.** The two test files listed in *Files to
   create (6)* are not optional. If a ported test cannot be made to pass
   against the live mainline tree, the worker must block — do not delete or
   `.skip` failing cases to make CI green.
6. **No drive-by edits.** Do not reformat unrelated code, bump deps beyond
   the two listed in *Files to modify (7)*, or change the mainline version
   number. If a cleanup is genuinely needed, do it in a separate commit on a
   separate branch.
7. **Spec path stays at `docs/qmd-remote-llm-port.md`.** Do not move it to
   `docs/specs/`. The mainline `.gitignore` is permissive about `docs/*.md`
   and the existing location is discoverable from the repo root.

### Acceptance checklist (for MR review)

- [ ] All 6 new files present and reviewed.
- [ ] All 7 modified files compile under `tsc --noEmit`.
- [ ] `vitest` green on `test/` (existing + 612 lines of new tests).
- [ ] `bun test` green on `test/`.
- [ ] `package-smoke.mjs` exits 0.
- [ ] Default (no env vars) `qmd query` produces same results as `main` HEAD on
      `test/eval.test.ts` corpora.
- [ ] With Ollama + `QMD_DEBUG_RERANK=1`, `qmd query` prints expected debug log.
- [ ] `qmd status` prints `LLM Class: HybridLLM` and backend preferences when
      `QMD_REMOTE_API_KEY` is set; prints `LLM Class: LlamaCpp` otherwise.
- [ ] README updated; environment-variable table has all 10 new rows.
- [ ] MR description links to `docs/qmd-remote-llm-port.md`.
- [ ] No version bump in this MR.