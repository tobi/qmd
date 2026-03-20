# Design: Full Modal Embedding Support for `qmd embed`

## Status

Proposed — awaiting implementation.

## Background

PR #444 adds Modal inference for query expansion, reranking, and vector search. However, the `qmd embed` command (used to generate vector embeddings for indexing) still requires a local GPU — it always uses `LlamaCpp` via `node-llama-cpp`, even when Modal inference is enabled.

This design addresses that gap: when `modal.inference=true` in config, the `embed` command should use Modal for both tokenization (during chunking) and embedding generation.

## Current State

The `embed` command flow:

```
qmd embed → vectorIndex() → generateEmbeddings()
```

Inside `generateEmbeddings` (`src/store.ts:1305`):
- **Chunking**: `chunkDocumentByTokens()` calls `getDefaultLlamaCpp().tokenize()` — always local
- **Embedding**: `withLLMSessionForLlm(llm, ...)` where `llm = store.llm ?? getDefaultLlamaCpp()` — always local LlamaCpp

The session management (`withLLMSessionForLlm`) only accepts `LlamaCpp`, not the broader `LLM` interface, preventing Modal routing.

## Goal

When `modal.inference=true`, `qmd embed` should:
1. Tokenize documents via Modal's `embeddinggemma` endpoint
2. Generate embeddings via Modal's `embeddinggemma` endpoint
3. No local GPU required for indexing

## Architecture

### Modal: Add `tokenize` endpoint

Add a `tokenize()` method to `QMDInference` in `modal/serve.py` that proxies to llama-server's `/tokenize` endpoint on port 8081 (the same `embeddinggemma` model already deployed):

```python
@modal.method()
def tokenize(self, texts: list[str]) -> list[list[int]]:
    resp = requests.post(
        f"http://127.0.0.1:{EMBED_SERVER.port}/tokenize",
        json={"content": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
```

### TypeScript: Update `LLM` interface

Add `tokenize()` and `countTokens()` to the `LLM` interface (`src/llm.ts`) so both `LlamaCpp` and `ModalLLM` are typed consistently:

```typescript
export interface LLM {
  // ...existing methods...
  tokenize(text: string): Promise<readonly LlamaToken[]>;
  countTokens(text: string): Promise<number>;
}
```

### TypeScript: `ModalLLM` implements `tokenize`

Add to `ModalLLM` class (`src/llm.ts`):

```typescript
async tokenize(text: string): Promise<readonly LlamaToken[]> {
  const tokens = await this.backend.tokenize([text]);
  return tokens[0] ?? [];
}

async countTokens(text: string): Promise<number> {
  return (await this.tokenize(text)).length;
}
```

### TypeScript: `ModalBackend` calls Modal `tokenize`

Add to `ModalBackend` class (`src/modal.ts`):

```typescript
async tokenize(texts: string[]): Promise<number[][]> {
  const fn = this.getMethod("tokenize");
  return fn.remote(texts);
}
```

### TypeScript: `generateEmbeddings` routes to Modal

In `generateEmbeddings` (`src/store.ts:1330`), replace:
```typescript
const llm = store.llm ?? getDefaultLlamaCpp();
const result = await withLLMSessionForLlm(llm, async (session) => {
  // session.embed() / session.embedBatch()
});
```

With:
```typescript
const llm = store.llm ?? getDefaultLLM();
// Call embed/embedBatch directly — both LlamaCpp and ModalLLM implement LLM interface
// Remove withLLMSessionForLlm wrapper (LlamaCpp.embed handles its own session internally)
// For ModalLLM, there's no session management needed
```

Note: `getDefaultLLM()` already returns `ModalLLM` when `modal.inference=true` (see `src/llm.ts:1797`).

### TypeScript: `chunkDocumentByTokens` routes to Modal

In `chunkDocumentByTokens` (`src/store.ts:2102`), replace:
```typescript
const llm = getDefaultLlamaCpp();
```

With:
```typescript
const llm = getDefaultLLM();
```

Both `LlamaCpp` and `ModalLLM` now implement `tokenize()` on the `LLM` interface.

### TypeScript: `Store` type accepts `LLM`

In the `Store` type definition (`src/store.ts:989`), change:
```typescript
llm?: LlamaCpp;
```

To:
```typescript
llm?: LLM;
```

This allows `createStore()` to accept either `LlamaCpp` or `ModalLLM`.

## Data Flow (Modal-enabled)

```
qmd embed
  → vectorIndex()
    → generateEmbeddings(store)
      → getDefaultLLM() → ModalLLM
        → chunkDocumentByTokens()
          → ModalLLM.tokenize() → Modal.tokenize() → llama-server /tokenize
        → llm.embed() / llm.embedBatch() → ModalLLM → Modal embed() → llama-server /embedding
        → insertEmbedding() → SQLite vectors_vec
```

## Error Handling

- **Modal not deployed**: `ModalBackend.tokenize()`/`embed()` throws with a message pointing to `qmd modal deploy` (existing behavior via `withRetry`)
- **Modal request fails**: errors are counted and reported in the progress output (existing behavior)
- **Local fallback**: not implemented — if `modal.inference=true`, Modal must be deployed; for local-only users, config stays unchanged

## Testing

1. **Unit**: `modal-backend.test.ts` — add test for `ModalBackend.tokenize()`
2. **Unit**: `modal-integration.test.ts` — add test for `ModalLLM.tokenize()` and `countTokens()` (this file already tests ModalLLM with mocked ModalBackend)
3. **Integration**: existing `store.test.ts` `generateEmbeddings` tests verify routing (they use `store.llm` mock — no changes needed)
4. **CLI smoke**: `modal-cli.test.ts` — add `qmd embed --force` test with Modal enabled

## Files Changed

| File | Change |
|------|--------|
| `modal/serve.py` | Add `tokenize()` method to `QMDInference` |
| `src/modal.ts` | Add `tokenize()` to `ModalBackend` |
| `src/llm.ts` | Add `tokenize`/`countTokens` to `LLM` interface; implement in `ModalLLM` |
| `src/store.ts` | `generateEmbeddings`: use `getDefaultLLM()`, call embed directly; `chunkDocumentByTokens`: use `getDefaultLLM()`; `Store` type: `llm?: LLM` |
| `test/modal-backend.test.ts` | Test `ModalBackend.tokenize()` |
| `test/modal-backend.test.ts` | Test `ModalBackend.tokenize()` |
| `test/modal-integration.test.ts` | Test `ModalLLM.tokenize()` and `countTokens()` |

## Out of Scope

- **Local fallback when Modal fails**: if `modal.inference=true`, Modal must be deployed. Local LlamaCpp fallback would be a follow-up.
- **tiktoken as JS tokenizer**: could be added later as a third option, but not needed for this PR.
- **Changes to `qmd query`, `qmd search`, `qmd vsearch`**: these already route to Modal when enabled.
