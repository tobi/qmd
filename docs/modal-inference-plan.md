# Modal Inference Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional remote GPU inference via Modal.com so users without a local GPU can run QMD's three GGUF models on a cheap cloud T4.

**Architecture:** Python Modal service (`modal/serve.py`) exposes `embed()`, `generate()`, `rerank()`, `ping()` methods on a T4 GPU with memory snapshots. JS `ModalBackend` class (`src/modal.ts`) calls it via the `modal` npm SDK. All prompt formatting stays in JS. Controlled by `modal.*` config keys in QMD's YAML config.

**Tech Stack:** Modal Python SDK (deploy), `modal` npm SDK v0.6+ (runtime), `llama-cpp-python` (inference), vitest (tests)

**Spec:** `docs/modal-inference-backend-design.md`

---

## Task 1: Config Keys for Modal

Add `modal.*` config keys to the existing YAML config system.

**Files:**
- Modify: `src/collections.ts` — extend `CollectionConfig` interface and load/save logic
- Test: `test/modal-config.test.ts`

- [ ] **Step 1: Write failing test**

Test that modal config keys can be read/written via the existing config system.

```typescript
// test/modal-config.test.ts
// Test: loadConfig returns modal defaults when no modal keys present
// Test: saveConfig persists modal.inference, modal.gpu, modal.scaledown_window
// Test: round-trip — save then load preserves modal values
// Use setConfigSource({ config: {...} }) for in-memory testing (existing pattern)
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `npx vitest run test/modal-config.test.ts`

- [ ] **Step 3: Implement config changes**

In `src/collections.ts`:
- Add `modal?` field to `CollectionConfig` interface:
  ```typescript
  modal?: {
    inference?: boolean;      // default false
    gpu?: string;             // default "T4"
    scaledown_window?: number; // default 15
  }
  ```
- The YAML round-trip handles new keys automatically, but the TypeScript interface must be extended for type safety.
- Add helper functions:
  ```typescript
  // getModalConfig() — reads config, returns modal block with defaults applied
  // setModalConfig(partial) — loads config, merges partial into modal block, saves
  ```

- [ ] **Step 4: Run test, confirm it passes**

Run: `npx vitest run test/modal-config.test.ts`

- [ ] **Step 5: Commit**

```bash
git add src/collections.ts test/modal-config.test.ts
git commit -m "feat: add modal config keys to CollectionConfig"
```

---

## Task 2: Python Modal Service (`modal/serve.py`)

Create the Modal app definition that loads all 3 GGUF models and exposes raw inference.

**Files:**
- Create: `modal/serve.py`
- Create: `modal/requirements.txt`

- [ ] **Step 1: Create `modal/requirements.txt`**

Just a reference file (deps installed in Modal image, not locally):
```
modal
llama-cpp-python
huggingface-hub
```

- [ ] **Step 2: Write `modal/serve.py`**

Structure the file in three sections:

**Section A — Image definition:**
- Use `modal.Image.debian_slim(python_version="3.11")`
- pip install `llama-cpp-python` with CUDA (use the `--extra-index-url` for CUDA wheels)
- pip install `huggingface-hub` for model download
- Use `.run_commands()` to download all 3 GGUF models from HuggingFace into `/models/` during image build
  - Use `huggingface_hub.hf_hub_download(repo_id, filename, local_dir="/models/")`
  - Three calls: embeddinggemma, qwen3-reranker, qmd-query-expansion

**Section B — QMDInference class:**
- `@app.cls(gpu=gpu_config, scaledown_window=idle_timeout, allow_concurrent_inputs=4, enable_memory_snapshot=True)`
- gpu_config and idle_timeout read from CLI args with defaults "T4" and 15
- `@modal.enter(snap=True) load_models()`:
  - Load all 3 Llama instances from `/models/` paths
  - Warmup each model (embed warmup text, generate 1 token from each)
- `@modal.method() embed(texts: list[str]) -> list[list[float]]`:
  - Iterate texts, call `self.embed_model.embed(text)`
  - Normalize: if result[0] is list, take result[0], else use result directly
  - Return list of vectors
- `@modal.method() generate(prompt, grammar_str, max_tokens, model="expand") -> str`:
  - Select model by name ("rerank" → rerank_model, else expand_model)
  - If grammar_str provided, create LlamaGrammar.from_string(grammar_str)
  - Call model(prompt=prompt, max_tokens=max_tokens, grammar=grammar_if_any)
  - Return choices[0]["text"]
- `@modal.method() ping() -> bool`: return True

**Section C — CLI entry point:**
- `if __name__ == "__main__":` with argparse
- Subcommands: `deploy`, `status`, `destroy`
- `deploy`: accept `--gpu` (default "T4") and `--scaledown-window` (default 15), run `modal deploy` programmatically
- `status`: use modal SDK to check if app "qmd-inference" exists and has deployed functions
- `destroy`: use `modal app stop` or equivalent to tear down

- [ ] **Step 3: Manual smoke test instructions**

Cannot run automated tests for Modal deploy (requires credentials + GPU). Leave a note:
```
# To test: modal token set, then python modal/serve.py deploy
# Verify: python modal/serve.py status
# Cleanup: python modal/serve.py destroy
```

- [ ] **Step 4: Commit**

```bash
git add modal/
git commit -m "feat: add Modal inference service (serve.py)"
```

---

## Task 3: ModalBackend JS Client (`src/modal.ts`)

JS class that calls the deployed Modal function via the `modal` npm package.

**Files:**
- Create: `src/modal.ts`
- Test: `test/modal-backend.test.ts`

- [ ] **Step 1: Install modal npm dependency**

```bash
bun add --optional modal
```

Add `modal` to `optionalDependencies` in `package.json` (not `dependencies`). In `src/modal.ts`, use a dynamic import with try/catch:

```typescript
let modalModule: typeof import("modal");
try {
  modalModule = await import("modal");
} catch {
  throw new Error("Modal npm package not installed. Run: bun add modal");
}
```

This keeps `modal` from being required for users who never use the Modal backend.

- [ ] **Step 2: Write failing tests**

```typescript
// test/modal-backend.test.ts
// These test the ModalBackend class with a mock Modal client.
//
// Test: constructor throws if ~/.modal.toml missing (mock fs.existsSync)
// Test: embed() calls fn.embed.remote() with correct args, returns vectors
// Test: generate() calls fn.generate.remote() with prompt, grammar, maxTokens, model
// Test: rerank() calls fn.rerank.remote() with query and texts, returns scores
// Test: ping() calls fn.ping.remote()
// Test: retry logic — connection error on first attempt, success on second
// Test: retry logic — 3 connection errors → throws with clear message
// Test: non-connection errors → immediate throw, no retry
// Test: lazy initialization — client not created until first call
```

- [ ] **Step 3: Run tests, confirm they fail**

Run: `npx vitest run test/modal-backend.test.ts`

- [ ] **Step 4: Implement `src/modal.ts`**

Structure:

```typescript
// src/modal.ts

// --- Types ---
// ModalConfig: { inference: boolean, gpu: string, scaledown_window: number }

// --- Retry helper ---
// withRetry(fn, maxAttempts=3, retryDelayMs=100):
//   attempt 1 → catch connection error → attempt 2 immediately
//   → catch connection error → wait retryDelayMs → attempt 3
//   → throw with full context
//   Non-connection errors: throw immediately
//   Connection error detection: check error message/code for ECONNREFUSED,
//   ETIMEDOUT, ECONNRESET, "unavailable", "deadline exceeded"

// --- ModalBackend class ---
// Implements a subset of the LLM interface (embed, generate)
// but NOT the full LLM interface yet — that's Task 5.
//
// Private state:
//   _client: modal.Client | null (lazy)
//   _cls: reference to deployed QMDInference class | null (lazy)
//
// Private ensureConnected():
//   - Check ~/.modal.toml exists → hard fail with message if missing
//   - Create modal.Client() if not yet created
//   - Resolve modal.Function.from_name("qmd-inference", "QMDInference")
//   - Store references for reuse
//
// Public methods:
//   embed(texts: string[]): Promise<number[][]>
//     - ensureConnected()
//     - withRetry(() => cls.embed.remote(texts))
//
//   generate(prompt: string, grammar: string | null, maxTokens: number,
//            model?: "expand" | "rerank"): Promise<string>
//     - ensureConnected()
//     - withRetry(() => cls.generate.remote(prompt, grammar, maxTokens, model))
//
//   rerank(query: string, texts: string[]): Promise<number[]>
//     - ensureConnected()
//     - withRetry(() => cls.rerank.remote(query, texts))
//
//   ping(): Promise<boolean>
//     - ensureConnected()
//     - withRetry(() => cls.ping.remote())
//
//   dispose(): no-op (nothing local to clean up)
```

Note: The exact API for calling Modal functions from JS SDK needs to be verified against `modal` npm docs. The pattern is roughly:
```typescript
import { Client, Function as ModalFunction } from "modal";
const client = new Client();
const fn = ModalFunction.lookup("qmd-inference", "QMDInference", client);
const result = await fn.call("embed", [texts]);
```
Consult the `modal` npm package README and types for the exact API. Use Context7 docs tool if needed.

- [ ] **Step 5: Run tests, confirm they pass**

Run: `npx vitest run test/modal-backend.test.ts`

- [ ] **Step 6: Commit**

```bash
git add src/modal.ts test/modal-backend.test.ts package.json bun.lockb
git commit -m "feat: add ModalBackend JS client with retry logic"
```

---

## Task 4: CLI Commands (`qmd modal deploy|status|destroy|test`)

Add the `modal` subcommand group to the CLI.

**Files:**
- Modify: `src/cli/qmd.ts` — add `case "modal":` block
- Test: `test/modal-cli.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// test/modal-cli.test.ts
// Test the pre-flight checks and error messages (mock child_process.execSync).
//
// Test: "qmd modal deploy" — when python3 not found → specific error message
// Test: "qmd modal deploy" — when modal module missing → specific error message
// Test: "qmd modal deploy" — when ~/.modal.toml missing → specific error message
// Test: "qmd modal deploy" — success → sets modal.inference=true in config
// Test: "qmd modal destroy" — sets modal.inference=false in config
// Test: "qmd modal" with no subcommand → usage help
// Test: "qmd modal unknown" → error with available subcommands
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `npx vitest run test/modal-cli.test.ts`

- [ ] **Step 3: Implement modal CLI commands**

In `src/cli/qmd.ts`, add a new case in the main switch:

```typescript
case "modal": {
  const subcommand = cli.args[0];
  // If no subcommand → print usage and exit
  // Available: deploy, status, destroy, test

  switch (subcommand) {
    case "deploy": {
      // Step 1: Check python3 exists
      //   try { execSync("python3 --version", { stdio: "pipe" }) }
      //   catch → print error: "python3 not found on PATH..."
      //
      // Step 2: Check modal pip package
      //   try { execSync('python3 -c "import modal"', { stdio: "pipe" }) }
      //   catch → print error: "Python 'modal' package not found..."
      //
      // Step 3: Check ~/.modal.toml
      //   if (!existsSync(join(homedir(), ".modal.toml"))) → print error
      //
      // Step 4: Read modal config (gpu, scaledown_window) with defaults
      //   const modalConfig = getModalConfig();
      //
      // Step 5: Resolve serve.py path
      //   Relative to package root: find it via import.meta.url
      //   const servePy = join(dirname(fileURLToPath(import.meta.url)),
      //                        "../../modal/serve.py")
      //   If running from src: adjust path accordingly
      //
      // Step 6: Shell out to python3
      //   execSync(`python3 "${servePy}" deploy --gpu ${gpu} --scaledown-window ${timeout}`,
      //            { stdio: "inherit" })
      //
      // Step 7: On success → setModalConfig({ inference: true })
      // Step 8: Print cost note
      break;
    }

    case "status": {
      // Create ModalBackend, call ping(), print result
      // Catch errors → print them with context
      break;
    }

    case "destroy": {
      // Check python3 + modal (same pre-flight)
      // Shell out: python3 serve.py destroy
      // setModalConfig({ inference: false })
      break;
    }

    case "test": {
      // Create ModalBackend
      // Call embed(["test"]) → verify array of floats
      // Call generate("Hello", null, 5) → verify non-empty string
      // Print pass/fail for each
      break;
    }

    default:
      // Unknown subcommand → error + available list
  }
  break;
}
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `npx vitest run test/modal-cli.test.ts`

- [ ] **Step 5: Commit**

```bash
git add src/cli/qmd.ts test/modal-cli.test.ts
git commit -m "feat: add qmd modal deploy/status/destroy/test CLI commands"
```

---

## Task 5: Integrate ModalBackend into LLM Layer

Wire the ModalBackend into `src/llm.ts` so that when `modal.inference=true`, all inference routes through Modal.

**Files:**
- Modify: `src/llm.ts` — add modal backend swap logic
- Test: `test/modal-integration.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// test/modal-integration.test.ts
// Test the backend swap logic in isolation.
//
// Test: when modal.inference=false → withLLMSession uses LlamaCpp (default)
// Test: when modal.inference=true → withLLMSession uses ModalBackend
// Test: modal embed() call formats prompt locally then sends raw text to backend
// Test: modal expandQuery() constructs chat-templated prompt locally,
//       sends to generate() with grammar string
// Test: modal rerank() deduplicates texts, calls modalBackend.rerank(),
//       maps scores back to original document order
// Test: startup validation — ping() called, failure → hard error
//
// Mock ModalBackend for these tests (don't need real Modal connection).
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `npx vitest run test/modal-integration.test.ts`

- [ ] **Step 3: Implement the integration**

Key changes to `src/llm.ts`:

**A) Create a ModalLLM class that wraps ModalBackend and implements the full `LLM` interface:**

```typescript
// Implements LLM interface using ModalBackend for remote inference.
// All prompt formatting happens HERE (same as LlamaCpp), only raw
// inference calls go to Modal.
//
// embed(text, options):
//   Format text using formatQueryForEmbedding / formatDocForEmbedding
//   (existing functions, already in llm.ts)
//   Call modalBackend.embed([formattedText])
//   Return EmbeddingResult with the vector
//
// generate(prompt, options):
//   Call modalBackend.generate(prompt, grammarString, maxTokens)
//   Return GenerateResult
//
// expandQuery(query, options):
//   Build the full chat-templated prompt with special tokens
//   (<|im_start|>system\n...<|im_end|>, <|im_start|>user\n...<|im_end|>)
//   exactly as LlamaCpp.expandQuery does, but as a raw string.
//
//   The Qwen3 chat template wraps the user prompt as:
//   <|im_start|>system
//   You are a helpful assistant.<|im_end|>
//   <|im_start|>user
//   /no_think Expand this search query: {query}<|im_end|>
//   <|im_start|>assistant
//
//   The GBNF grammar string is:
//   root ::= line+
//   line ::= type ": " content "\n"
//   type ::= "lex" | "vec" | "hyde"
//   content ::= [^\n]+
//
//   Generation params: temperature=0.7, top_k=20, top_p=0.8, max_tokens=600
//
//   Note: Verify during implementation that this GBNF grammar parses
//   identically in both node-llama-cpp and llama-cpp-python. Both use
//   llama.cpp's grammar parser, but test with a concrete input.
//
//   Call modalBackend.generate(fullPrompt, grammarStr, maxTokens, "expand")
//   Parse the response same as existing code
//
// rerank(query, documents, options):
//   Deduplicate texts (same as existing LlamaCpp.rerank)
//   Truncate documents that exceed context size
//   Call modalBackend.rerank(query, uniqueTexts)
//   → Python side uses create_chat_completion() with Qwen3-Reranker native template
//   Map scores back to documents in original order
//   Return RerankResult
//
// modelExists(): return stub info (models always exist on Modal)
// dispose(): call modalBackend.dispose() (no-op)
```

**B) Modify `getDefaultLlamaCpp()` (or add a parallel `getDefaultModalLLM()`):**

The existing `getDefaultLlamaCpp()` singleton pattern needs a modal-aware wrapper:

```typescript
// getDefaultLLM(): LLM
//   if (getModalConfig().inference) return getDefaultModalLLM()
//   else return getDefaultLlamaCpp()
```

**C) Modify `withLLMSession()` to use `getDefaultLLM()`** instead of always using `getDefaultLlamaCpp()`.

**D) Startup validation:**
- Add a `validateModalConnection()` function that calls `ping()`
- Call it from MCP server startup and CLI entry point when `modal.inference=true`
- On failure → throw with clear error message, no fallback

**E) Session management bypass:**
When Modal is active, `withLLMSession` should NOT use `LLMSessionManager` (which is coupled to LlamaCpp). Instead, create a thin wrapper that:
- Wraps `ModalLLM` directly
- Makes `canUnload()` always return `false` (nothing to unload)
- Makes inactivity timeout a no-op
- Implements `ILLMSession` by delegating to `ModalLLM` methods

**Important:** The chat template construction for query expansion and reranking is the trickiest part. Study the existing `LlamaCpp.expandQuery()` and `LlamaCpp.rerank()` methods carefully to extract:
- The exact system/user prompt text
- The special token wrapping (`<|im_start|>`, `<|im_end|>`, etc.)
- The GBNF grammar string (for expansion)
- How rerank scores are extracted from model output

These are currently handled by node-llama-cpp's `LlamaChatSession` and `createRankingContext()`. For Modal, we need to construct these as raw text strings. Read the Qwen3 model documentation if needed.

- [ ] **Step 4: Run tests, confirm they pass**

Run: `npx vitest run test/modal-integration.test.ts`

- [ ] **Step 5: Run full test suite**

Run: `npx vitest run --reporter=verbose test/`

Ensure no regressions in existing tests.

- [ ] **Step 6: Commit**

```bash
git add src/llm.ts test/modal-integration.test.ts
git commit -m "feat: integrate ModalBackend into LLM layer with backend swap"
```

---

## Task 6: Package & Docs Finalization

Update package.json and add the `modal/` directory to the distributed files.

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Update package.json**

Add `"modal/"` to the `files` array so `serve.py` and `requirements.txt` ship with the npm package:

```json
"files": [
  "bin/",
  "dist/",
  "modal/",
  "LICENSE",
  "CHANGELOG.md"
]
```

Add `modal` to `optionalDependencies` (not `dependencies`):
```json
"optionalDependencies": {
  "modal": "^0.6.0"
}
```

- [ ] **Step 2: Verify build still works**

```bash
bun run build
```

- [ ] **Step 3: Verify modal/serve.py is resolvable from dist**

```bash
# Check that the path resolution in the CLI works
bun -e "
  import { fileURLToPath } from 'url';
  import { dirname, join } from 'path';
  const __dirname = dirname(fileURLToPath(import.meta.url));
  console.log(join(__dirname, '../modal/serve.py'));
" --input-type=module
```

- [ ] **Step 4: Commit**

```bash
git add package.json
git commit -m "feat: add modal package to files array and optionalDependencies"
```

- [ ] **Step 5: Update CHANGELOG.md**

Add under `## [Unreleased]`:
```
- Added: Optional Modal.com inference backend for remote GPU inference (`qmd modal deploy`)
```

---

## Task 7: End-to-End Smoke Test (Manual)

This task is a manual verification checklist, not automated tests.

- [ ] **Step 1: Verify local mode still works**

```bash
qmd search "test query" --collection <some-collection>
```

Confirm: no Modal references, works as before.

- [ ] **Step 2: Deploy Modal function**

```bash
qmd modal deploy
```

Confirm: python3 found, modal found, ~/.modal.toml found, deploy succeeds, config updated.

- [ ] **Step 3: Run Modal smoke test**

```bash
qmd modal test
```

Confirm: embed and generate both pass.

- [ ] **Step 4: Run a real query via Modal**

```bash
qmd query "test query" --collection <some-collection>
```

Confirm: results come back, no errors.

- [ ] **Step 5: Tear down**

```bash
qmd modal destroy
```

Confirm: function torn down, config reset.

- [ ] **Step 6: Verify local mode resumes**

```bash
qmd query "test query" --collection <some-collection>
```

Confirm: works locally again.
