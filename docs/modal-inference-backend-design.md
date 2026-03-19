# Modal Inference Backend for QMD

## Summary

Add optional remote GPU inference via Modal.com, allowing users without a local GPU to run QMD's three GGUF models (embedding, reranking, query expansion) on a cheap cloud GPU. The Modal function is a dumb model server exposing raw inference primitives (`embed()`, `generate()`, `rerank()`, `ping()`) — all prompt formatting (including chat templates and special tokens), grammar construction, and search logic stays in QMD's JS codebase. Reranking uses a dedicated `rerank()` method that leverages `create_chat_completion()` for model-native template handling (analogous to node-llama-cpp's `rankAll()`).

## Architecture Overview

```
User machine (JS/TS)                          Modal (Python)
┌─────────────────────────┐                   ┌──────────────────────────┐
│  src/llm.ts             │                   │  modal/serve.py          │
│  ┌───────────────────┐  │                   │  ┌────────────────────┐  │
│  │ Prompt formatting  │  │                   │  │ QMDInference class │  │
│  │ Grammar logic      │  │   JS SDK (gRPC)   │  │                    │  │
│  │ Result parsing     │──┼──────────────────►│  │ embed()   - raw    │  │
│  │                    │  │                   │  │ generate() - raw   │  │
│  │                    │  │                   │  │ rerank()  - scores │  │
│  │ ModalBackend       │  │                   │  │ ping()    - health │  │
│  └───────────────────┘  │                   │  └────────────────────┘  │
│                         │                   │                          │
│  src/store.ts           │                   │  T4 GPU (default)        │
│  (unchanged logic)      │                   │  3 GGUF models in image  │
└─────────────────────────┘                   └──────────────────────────┘
```

## Python Modal Service (`modal/serve.py`)

A single Python file shipped in the QMD repo. Defines and deploys the Modal function.

### Image Build

- Base image with `llama-cpp-python` installed with CUDA support
- Downloads all 3 GGUF models from HuggingFace at image build time:
  - `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf` (~300MB)
  - `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf` (~640MB)
  - `hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf` (~1.1GB)
- Models baked into image — no runtime downloads

### Class Definition

```python
import modal

app = modal.App("qmd-inference")

# gpu and scaledown_window are passed as CLI args during deploy
@app.cls(
    gpu=gpu_config,                     # Configurable, default "T4"
    scaledown_window=idle_timeout,      # Configurable, default 15 seconds
    allow_concurrent_inputs=4,          # See Design Decisions for concurrency rationale
    enable_memory_snapshot=True,        # Snapshot model state after loading
)
class QMDInference:
    @modal.enter(snap=True)
    def load_models(self):
        """Load all 3 models + warmup. Captured in memory snapshot."""
        from llama_cpp import Llama

        self.embed_model = Llama(
            model_path="/models/embeddinggemma-300M-Q8_0.gguf",
            embedding=True,
            n_ctx=2048,
        )
        self.rerank_model = Llama(
            model_path="/models/qwen3-reranker-0.6b-q8_0.gguf",
            n_ctx=2048,
        )
        self.expand_model = Llama(
            model_path="/models/qmd-query-expansion-1.7B-q4_k_m.gguf",
            n_ctx=2048,
        )
        # Warmup passes to pre-fill caches before snapshot
        self.embed_model.embed("warmup")
        self.rerank_model("warmup", max_tokens=1)
        self.expand_model("warmup", max_tokens=1)

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raw embedding — no prompt formatting.
        Returns shape [n_texts, embed_dim].
        Note: llama-cpp-python's Llama.embed() returns List[List[float]].
        We normalize to always return one embedding vector per input text.
        """
        result = []
        for text in texts:
            vec = self.embed_model.embed(text)
            # Llama.embed() returns List[List[float]] — take first element
            # to normalize to a single vector per input text
            if isinstance(vec[0], list):
                result.append(vec[0])
            else:
                result.append(vec)
        return result

    @modal.method()
    def generate(self, prompt: str, grammar: str | None, max_tokens: int,
                 model: str = "expand") -> str:
        """Raw completion with optional GBNF grammar constraint.
        No chat template application — the caller must construct the full
        prompt including any special tokens (e.g. <|im_start|>, <|im_end|>).
        Used for both query expansion (model="expand") and reranking
        (model="rerank") since both go through raw completion.
        """
        llm = self.rerank_model if model == "rerank" else self.expand_model
        kwargs = {"prompt": prompt, "max_tokens": max_tokens}
        if grammar:
            from llama_cpp import LlamaGrammar
            kwargs["grammar"] = LlamaGrammar.from_string(grammar)
        result = llm(**kwargs)
        return result["choices"][0]["text"]

    @modal.method()
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder scoring using Qwen3-Reranker.
        Uses create_chat_completion() which applies the model's native
        chat template automatically (same as node-llama-cpp's rankAll).
        """
        scores = []
        for text in texts:
            response = self.rerank_model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Judge whether the Document meets the requirements of the Query. Note that the answer can only be \"yes\" or \"no\"."},
                    {"role": "user", "content": f"<Query>{query}</Query>\n<Document>{text}</Document>"}
                ],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
            )
            # Extract the "yes" probability as the relevance score
            # (standard Qwen3-Reranker scoring approach)
            logprobs_data = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            yes_prob = 0.0
            for lp in logprobs_data:
                if lp["token"].lower().strip() == "yes":
                    import math
                    yes_prob = math.exp(lp["logprob"])
                    break
            scores.append(yes_prob)
        return scores

    @modal.method()
    def ping(self) -> bool:
        """Health check — verifies function is reachable and models loaded."""
        return True
```

### Container Behavior

- `scaledown_window=15` — container shuts down 15s after last request (cost-efficient for burst patterns)
- `allow_concurrent_inputs=4` — single container handles bursts without triggering scale-up. llama-cpp-python serializes GPU inference internally, so concurrent requests queue at the CUDA level rather than causing OOM, but lower concurrency reduces latency variance
- `enable_memory_snapshot=True` + `@modal.enter(snap=True)` — after first deploy, subsequent cold starts restore from memory snapshot instead of re-loading ~2GB of models
- No `min_containers` / `keep_warm` — scales to zero when idle
- Single container by design (no max_containers param exists, but concurrency + low traffic pattern means no scale-up)

### Deploy CLI

`modal/serve.py` also acts as a CLI entry point:

```sh
python modal/serve.py deploy --gpu T4 --scaledown-window 15
python modal/serve.py status
python modal/serve.py destroy
```

## QMD Config

The `modal.*` config keys live in QMD's existing config file at `~/.config/qmd/index.yml` (managed by `src/collections.ts`). These are global settings, not per-collection.

### New Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `modal.inference` | boolean | `false` | Enable Modal-based inference |
| `modal.gpu` | string | `"T4"` | GPU type for the Modal container |
| `modal.scaledown_window` | number | `15` | Seconds before idle container shuts down |

### Auth

Reads from `~/.modal.toml` (Modal's native auth file, created by user running `modal token set`). QMD does not store Modal credentials — it relies on Modal's own config.

## CLI Commands

### `qmd modal deploy`

1. Check `python3` is available on PATH
   - If missing: `Error: python3 not found. Install Python 3.10+ to deploy Modal functions. See https://python.org`
2. Check `modal` Python package is installed (`python3 -c "import modal"`)
   - If missing: `Error: Modal Python package not found. Run: pip install modal`
3. Check `~/.modal.toml` exists
   - If missing: `Error: Modal not authenticated. Run: modal token set`
4. Read `modal.gpu` and `modal.scaledown_window` from QMD config
5. Shell out to `python3 modal/serve.py deploy --gpu <gpu> --scaledown-window <timeout>`. The JS CLI resolves `modal/serve.py` relative to the QMD package installation directory (via `import.meta.url` resolution from the compiled JS in `dist/`). The `modal/` directory must be included in `package.json`'s `files` array.
6. On success: auto-set `modal.inference = true` in QMD config
7. Print confirmation with deployed function name and a brief cost note: "GPU: T4 (~$0.59/hr, billed per second, scales to zero when idle)"

### `qmd modal status`

Check if the Modal function is deployed and reachable. Calls `ping()` method.

### `qmd modal destroy`

Tears down the deployed Modal function. Sets `modal.inference = false`.

### `qmd modal test`

Runs a small end-to-end smoke test to verify the deployed function works correctly:

1. Calls `embed()` with a short test string, verifies the response is a well-formed embedding vector (array of floats with expected dimensionality)
2. Calls `generate()` with a short prompt, verifies the response is a non-empty string of generated tokens
3. Prints pass/fail for each check

This catches model loading or CUDA issues that `ping()` alone would not detect (e.g., models loaded but producing garbage output, CUDA out-of-memory during inference).

## JS Runtime Integration (`src/modal.ts`)

### ModalBackend Class

```typescript
class ModalBackend {
  private client: modal.Client
  private fn: modal.Function  // Reference to deployed QMDInference

  /**
   * Raw embedding — caller is responsible for prompt formatting.
   */
  async embed(texts: string[]): Promise<number[][]>

  /**
   * Raw text generation with optional GBNF grammar.
   * Caller passes the fully formatted prompt (including all special tokens)
   * and grammar string. Used for both query expansion and reranking.
   */
  async generate(prompt: string, grammar: string | null, maxTokens: number,
                 model?: "expand" | "rerank"): Promise<string>

  /**
   * Cross-encoder reranking using Qwen3-Reranker.
   * Python side uses create_chat_completion() for model-native template handling.
   * Returns relevance scores (0-1) for each text.
   */
  async rerank(query: string, texts: string[]): Promise<number[]>

  /**
   * Health check.
   */
  async ping(): Promise<boolean>
}
```

`ModalBackend` should implement the existing `LLM` interface from `src/llm.ts` so it can be swapped in transparently. Session management (`withLLMSession`, `canUnload`, inactivity timeouts) becomes a no-op in Modal mode since there are no local models to manage.

**Chat template responsibility:** The JS side must construct full chat-templated prompts (including `<|im_start|>`, `<|im_end|>` special tokens) for query expansion before sending to `generate()`. The Python `generate()` does raw completion only — it does not apply any chat template. Reranking uses the dedicated `rerank()` method, which handles the Qwen3-Reranker chat template via `create_chat_completion()` on the Python side.

### Initialization

- Created lazily on first inference call
- Reads `~/.modal.toml` for auth (via `modal` npm package)
- Resolves the deployed `qmd-inference` function reference
- If function not found or auth missing: immediate hard fail with clear error message

### Retry Logic (Connection Errors Only)

3 total attempts: initial call, 1 immediate retry, 1 retry after 100ms.

```
attempt 1 (initial)   → connection error → immediate retry
attempt 2 (retry)     → connection error → wait 100ms → retry
attempt 3 (final)     → connection error → throw with full stacktrace and reason

Non-connection errors (auth, not found, etc.) → immediate throw, no retry
```

### Integration in `src/llm.ts`

The existing inference call sites in `llm.ts` get a conditional swap:

```typescript
// At each inference call site (embed, generate, rerank)
if (config.get("modal.inference")) {
  return modalBackend.embed(texts)
} else {
  return localLlama.embed(texts)
}
```

`src/store.ts` functions (`generateEmbeddings`, `searchVec`, `expandQuery`, `rerank`) remain unchanged — they call `llm.ts` which handles the backend swap transparently.

### Startup Validation

When `modal.inference = true`:
- On MCP server startup: call `ping()` to verify Modal function is reachable
- On CLI command: same `ping()` check before executing
- Failure: hard fail with full error, no fallback to local

## Error Messages

### Deploy-Time Errors

```
Error: python3 not found on PATH.
Modal deployment requires Python 3.10+.
Install it from https://python.org or via your package manager.

Error: Python 'modal' package not found.
Install it with: pip install modal
Then authenticate with: modal token set

Error: Modal not authenticated. No ~/.modal.toml found.
Run: modal token set
to authenticate with your Modal account.

Error: Modal deployment failed.
<full stderr from python process>
```

### Runtime Errors

```
Error: Modal inference is enabled but the deployed function is not reachable.
Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.
<full error details / stacktrace>

Error: Modal inference is enabled but ~/.modal.toml is missing or invalid.
Run: modal token set
```

## File Layout

### New Files

```
modal/
  serve.py            # Modal app definition + deploy CLI
  requirements.txt    # llama-cpp-python, modal (for reference — installed in Modal image)
src/
  modal.ts            # ModalBackend class (JS SDK client)
```

### Modified Files

```
src/llm.ts            # Conditional: local vs modal backend at inference call sites
src/cli/qmd.ts        # New 'qmd modal deploy|status|destroy|test' subcommands
src/collections.ts    # New config keys: modal.inference, modal.gpu, modal.scaledown_window
package.json          # Add 'modal' npm dependency; add "modal/" to `files` array so serve.py ships with the npm package
```

## Dependencies

### Deploy-Time (user's machine)

- `python3` (3.10+)
- `modal` pip package (for `modal deploy` CLI)

### Runtime (user's machine)

- `modal` npm package (v0.6+) — JS SDK for calling deployed function

### Modal Image (cloud)

- `llama-cpp-python` with CUDA support
- 3 GGUF model files (baked into image)

## Design Decisions

1. **Dumb model server** — Modal function exposes raw `embed()`, `generate()`, `rerank()`, `ping()`. All prompt formatting (including chat templates with special tokens), grammar construction, and result parsing stays in JS. The `rerank()` method is the one exception to the "dumb server" principle: it uses `create_chat_completion()` for model-native template handling, analogous to how node-llama-cpp's `rankAll()` handles the Qwen3-Reranker chat template internally. This is necessary because `rankAll()` is a high-level API with no visible prompt template to reverse-engineer on the JS side. Zero duplication, zero drift risk.

2. **Memory snapshots** — `enable_memory_snapshot=True` with `@modal.enter(snap=True)` captures loaded models in memory. Subsequent cold starts restore from snapshot instead of re-loading ~2GB of model weights.

3. **Single container** — `allow_concurrent_inputs=4` handles burst patterns on one container. llama-cpp-python serializes GPU inference internally, so concurrent requests queue at the CUDA level rather than causing OOM, but lower concurrency (4 instead of 10) reduces latency variance. 15s idle timeout scales to zero quickly. No `keep_warm` to minimize cost.

4. **JS SDK for runtime** — The `modal` npm package calls deployed functions directly via gRPC. No HTTP endpoints, no URL management. Auth handled by Modal tokens.

5. **Python only for deploy** — Users need Python + `modal` pip package only for the one-time `qmd modal deploy`. Runtime is pure JS.

6. **No fallback** — When `modal.inference = true`, local models are never used. Clear failure modes prevent confusing mixed results. Connection errors get 3 total attempts (initial call, 1 immediate retry, 1 retry after 100ms), then hard fail.

7. **JS SDK risk** — The `modal` npm SDK is beta (v0.6). If it becomes unavailable, a fallback to HTTP web endpoints is straightforward since the Python side can expose the same methods via `@modal.web_endpoint`.

8. **GBNF grammar compatibility** — GBNF grammar syntax compatibility between node-llama-cpp and llama-cpp-python must be verified during implementation. Both use llama.cpp's grammar parser under the hood so they should be identical, but edge cases should be tested.
