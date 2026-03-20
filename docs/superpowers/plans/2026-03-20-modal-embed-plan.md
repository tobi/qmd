# Full Modal Embedding Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `qmd embed` through Modal when `modal.inference=true`, enabling GPU-free indexing. Adds `tokenize` endpoint to Modal, updates the `LLM` interface, and refactors `generateEmbeddings` to call `llm.embed()`/`embedBatch()` directly.

**Architecture:** Modal hosts `embeddinggemma` via llama-server on port 8081. This same server exposes `/tokenize`. We add a JS-side `tokenize()` method to `ModalBackend` and `ModalLLM`, update the `LLM` interface, and change `generateEmbeddings`/`chunkDocumentByTokens` to use `getDefaultLLM()` (which returns `ModalLLM` when Modal is enabled) instead of hardcoded `getDefaultLlamaCpp()`.

**Tech Stack:** TypeScript, Bun, Modal SDK (Python), node-llama-cpp, vitest

---

## File Map

| File | Responsibility |
|------|----------------|
| `modal/serve.py` | Modal Python inference — add `tokenize()` endpoint |
| `src/modal.ts` | JS Modal client — add `tokenize()` to `ModalBackend` |
| `src/llm.ts` | `LLM` interface, `ModalLLM`, `LlamaCpp` — add `tokenize`/`countTokens` to interface, update `ModalSession` |
| `src/store.ts` | `generateEmbeddings`, `chunkDocumentByTokens`, `Store` type |
| `test/modal-backend.test.ts` | Unit test `ModalBackend.tokenize()` |
| `test/modal-integration.test.ts` | Unit test `ModalLLM.tokenize()`, `countTokens()` |
| `test/store.test.ts` | Update existing `generateEmbeddings` tests |

---

## Task 1: Modal Python — Add `tokenize()` endpoint

**Files:**
- Modify: `modal/serve.py:210`

- [ ] **Step 1: Add `tokenize()` method to `QMDInference` class**

In `QMDInference` class (after `embed()` method, around line 210), add:

```python
@modal.method()
def tokenize(self, texts: list[str]) -> list[list[int]]:
    """Tokenize texts using the embedding model's tokenizer.

    Proxies to llama-server's /tokenize endpoint on port 8081
    (the embeddinggemma model). Returns token IDs as nested int lists.
    """
    import requests

    resp = requests.post(
        f"http://127.0.0.1:{EMBED_SERVER.port}/tokenize",
        json={"content": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
```

- [ ] **Step 2: Commit**

```bash
git add modal/serve.py
git commit -m "feat(modal): add tokenize endpoint to QMDInference

Proxies /tokenize from llama-server embeddinggemma (port 8081) to
enable tokenization for Modal-only indexing."
```

---

## Task 2: TypeScript `LLM` Interface — Add `tokenize`/`countTokens`

**Files:**
- Modify: `src/llm.ts:316-353`

- [ ] **Step 1: Add `tokenize` and `countTokens` to `LLM` interface**

After the `rerank` method (line 347) and before `dispose` (line 352), add:

```typescript
  /**
   * Tokenize text using the embedding model's tokenizer.
   * Returns tokenizer tokens (opaque type from the underlying implementation).
   */
  tokenize(text: string): Promise<readonly LlamaToken[]>;

  /**
   * Count tokens in text using the embedding model's tokenizer.
   */
  countTokens(text: string): Promise<number>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
```

Note: `LlamaToken` is already defined/exported by node-llama-cpp types. `ModalLLM.tokenize()` will return `number[]` cast to `readonly LlamaToken[]` (they are both just integer arrays at runtime).

- [ ] **Step 2: Verify `LlamaToken` is exported**

Check that `LlamaToken` is exported from `src/llm.ts` (it comes from node-llama-cpp types). If not exported, add:
```typescript
export type { LlamaToken } from "./types.js";
```
or wherever it's defined.

- [ ] **Step 3: Commit**

```bash
git add src/llm.ts
git commit -m "feat(llm): add tokenize/countTokens to LLM interface

Both LlamaCpp and ModalLLM already implement these methods; adding them
to the interface enables generateEmbeddings to route through getDefaultLLM."
```

---

## Task 3: TypeScript `ModalBackend` — Add `tokenize()`

**Files:**
- Modify: `src/modal.ts:165-179` (after `embed()` method)

- [ ] **Step 1: Add `tokenize()` method to `ModalBackend`**

After the `embed()` method (after line 179), add:

```typescript
  /**
   * Tokenize texts using the embedding model's tokenizer on Modal.
   * Returns token IDs as nested number arrays.
   */
  async tokenize(texts: string[]): Promise<number[][]> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("tokenize");
      return fn.remote([texts]);
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }
```

- [ ] **Step 2: Commit**

```bash
git add src/modal.ts
git commit -m "feat(modal): add tokenize() to ModalBackend

Calls QMDInference.tokenize() on Modal, which proxies to llama-server's
/tokenize endpoint for the embeddinggemma model."
```

---

## Task 4: TypeScript `ModalLLM` — Implement `tokenize`/`countTokens`

**Files:**
- Modify: `src/llm.ts:1612-1618` (after `embedBatch()` in `ModalLLM`)

- [ ] **Step 1: Add `tokenize()` and `countTokens()` to `ModalLLM`**

After `embedBatch()` (around line 1618) and before `generate()` (line 1620), add:

```typescript
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    const tokens = await this.backend.tokenize([text]);
    return tokens[0] ?? [];
  }

  async countTokens(text: string): Promise<number> {
    return (await this.tokenize(text)).length;
  }
```

`LlamaToken` is imported from node-llama-cpp types (line 10-13 of llm.ts). `number[]` from `backend.tokenize()` is compatible at runtime.

- [ ] **Step 2: Also add `tokenize`/`countTokens` to `ModalSession`**

In `ModalSession` class (around line 1882, after `embedBatch()`), add:

```typescript
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.modalLLM.tokenize(text);
  }

  async countTokens(text: string): Promise<number> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.modalLLM.countTokens(text);
  }
```

- [ ] **Step 3: Add `LlamaToken` import if needed**

If `LlamaToken` isn't already importable in `llm.ts`, check the node-llama-cpp import at the top of the file:
```typescript
import { fileURLToPath } from "node:url";
```
`LlamaToken` is likely from `node-llama-cpp`'s type exports. Verify it's accessible. If not, check the import chain.

- [ ] **Step 4: Commit**

```bash
git add src/llm.ts
git commit -m "feat(llm): implement tokenize/countTokens in ModalLLM and ModalSession

ModalLLM delegates to ModalBackend.tokenize() which calls the
QMDInference.tokenize() endpoint on Modal. ModalSession also exposes
these for consistency with LlamaCpp's session interface."
```

---

## Task 5: TypeScript `src/store.ts` — Refactor `generateEmbeddings` and `chunkDocumentByTokens`

**Files:**
- Modify: `src/store.ts:1328-1333` (`generateEmbeddings`)
- Modify: `src/store.ts:2100-2102` (`chunkDocumentByTokens`)
- Modify: `src/store.ts:989` (`Store` type)

- [ ] **Step 1: Update `Store` type — `llm?: LlamaCpp` → `llm?: LLM`**

In the `Store` type definition (line 989), change:
```typescript
  /** Optional LlamaCpp instance for this store (overrides the global singleton) */
  llm?: LlamaCpp;
```
To:
```typescript
  /** Optional LLM instance for this store (overrides the global singleton) */
  llm?: LLM;
```

- [ ] **Step 2: Refactor `generateEmbeddings` — remove `withLLMSessionForLlm`, call `llm.embed()` directly**

In `generateEmbeddings` (line 1330), replace:
```typescript
  const llm = store.llm ?? getDefaultLlamaCpp();
  const result = await withLLMSessionForLlm(llm, async (session) => {
```
With:
```typescript
  const llm = store.llm ?? getDefaultLLM();
```

Then replace the session callback body with direct `llm` calls. The session wrapper was:
```typescript
    const result = await withLLMSessionForLlm(llm, async (session) => {
      // ... session.embed() and session.embedBatch() calls ...
    });
```

Replace with:
```typescript
    const result = await (async () => {
      // ... llm.embed() and llm.embedBatch() calls instead ...
    })();
```

Specifically, in the session callback (lines 1377-1404), replace:
- `session.embed(firstText)` → `llm.embed(firstText, { model, isQuery: true })`
- `session.embedBatch(texts)` → `llm.embedBatch(texts)`

Note: `isQuery: true` is used for the init-embedding dimension probe. For actual doc chunks, use `isQuery: false` (default).

- [ ] **Step 3: Refactor `chunkDocumentByTokens` — use `getDefaultLLM()` instead of `getDefaultLlamaCpp()`**

At line 2102, replace:
```typescript
  const llm = getDefaultLlamaCpp();
```
With:
```typescript
  const llm = getDefaultLLM();
```

- [ ] **Step 4: Commit**

```bash
git add src/store.ts
git commit -m "refactor(store): route embed through getDefaultLLM() for Modal support

generateEmbeddings now calls llm.embed()/embedBatch() directly on the
LLM interface (LlamaCpp or ModalLLM) instead of through
withLLMSessionForLlm which was LlamaCpp-only. chunkDocumentByTokens
uses getDefaultLLM() for tokenization. Store.llm type widened to LLM."
```

---

## Task 6: Tests

**Files:**
- Modify: `test/modal-backend.test.ts`
- Modify: `test/modal-integration.test.ts`
- Modify: `test/store.test.ts`

### 6A: `modal-backend.test.ts` — Test `ModalBackend.tokenize()`

- [ ] **Step 1: Add test for `tokenize()` in the `ModalBackend` describe block**

After the `embed` describe block (around line 202), add:

```typescript
  describe("tokenize", () => {
    test("calls remote with texts and returns token IDs", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      const tokenIds = [[1, 2, 3], [4, 5, 6, 7]];
      mockRemote.mockResolvedValue(tokenIds);

      const backend = createBackend();
      const result = await backend.tokenize(["hello world", "goodbye"]);

      expect(mockMethod).toHaveBeenCalledWith("tokenize");
      expect(mockRemote).toHaveBeenCalledWith([["hello world", "goodbye"]]);
      expect(result).toEqual(tokenIds);
    });

    test("throws on connection error after retries", async () => {
      const { mockRemote } = await getMocks();
      mockRemote.mockRejectedValue(new Error("connect ECONNREFUSED"));

      const backend = createBackend();
      await expect(backend.tokenize(["test"])).rejects.toThrow(
        /Modal .* not reachable/,
      );
    });
  });
```

- [ ] **Step 2: Commit**

```bash
git add test/modal-backend.test.ts
git commit -m "test(modal): add ModalBackend.tokenize() unit tests"
```

### 6B: `modal-integration.test.ts` — Test `ModalLLM.tokenize()` and `countTokens()`

- [ ] **Step 1: Add `mockTokenize` to ModalBackend mock**

In the mock setup (around line 28-42), add:
```typescript
const mockTokenize = vi.fn();
```

And in the `ModalBackend` mock:
```typescript
  ModalBackend: vi.fn(() => ({
    embed: mockEmbed,
    generate: mockGenerate,
    rerank: mockRerank,
    ping: mockPing,
    tokenize: mockTokenize,
    dispose: mockDispose,
  })),
```

- [ ] **Step 2: Add tests for `tokenize()` and `countTokens()`**

After the `embed` describe block in `ModalLLM` (around line 131), add:

```typescript
  describe("tokenize", () => {
    test("calls backend.tokenize with text and returns tokens", async () => {
      const modalLLM = new ModalLLM();
      mockTokenize.mockResolvedValue([[1, 2, 3, 4]]);

      const result = await modalLLM.tokenize("hello world");

      expect(mockTokenize).toHaveBeenCalledWith([["hello world"]]);
      expect(result).toEqual([1, 2, 3, 4]);
    });

    test("returns empty array when backend returns no tokens", async () => {
      const modalLLM = new ModalLLM();
      mockTokenize.mockResolvedValue([[]]);

      const result = await modalLLM.tokenize("test");
      expect(result).toEqual([]);
    });
  });

  describe("countTokens", () => {
    test("returns length of token array from tokenize", async () => {
      const modalLLM = new ModalLLM();
      mockTokenize.mockResolvedValue([[1, 2, 3, 4, 5]]);

      const count = await modalLLM.countTokens("hello world");
      expect(count).toBe(5);
    });
  });
```

- [ ] **Step 3: Commit**

```bash
git add test/modal-integration.test.ts
git commit -m "test(llm): add ModalLLM tokenize/countTokens unit tests"
```

### 6C: `store.test.ts` — Verify existing `generateEmbeddings` tests still pass

- [ ] **Step 1: Run existing tests to verify no regressions**

```bash
npx vitest run test/store.test.ts --reporter=verbose 2>&1 | head -50
```

The existing tests use `store.llm = fakeEmbedLlm` and `setDefaultLlamaCpp(fakeTokenizer)`. After our changes:
- `store.llm` is used directly (good — fake embed LLM still works)
- `getDefaultLLM()` is called when `store.llm` is falsy — but the test sets `store.llm`, so it's used

- [ ] **Step 2: Commit (if changes needed, otherwise skip)**

Only commit if the test behavior changed.

---

## Task 7: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
npx vitest run --reporter=verbose 2>&1 | tail -30
```

Expected: All tests pass, including new ones. If any fail, fix the issue.

- [ ] **Step 2: Run typecheck**

```bash
bun run typecheck 2>&1
```
Or find the typecheck command from `package.json` scripts.

- [ ] **Step 3: Commit if tests pass**

```bash
git add -A && git commit -m "test: full suite passes after Modal embed routing

- modal-backend.test.ts: ModalBackend.tokenize() tests
- modal-integration.test.ts: ModalLLM tokenize/countTokens tests
- store.test.ts: verified existing generateEmbeddings tests still pass
"
```

---

## Task 8: Deploy Modal and smoke test

- [ ] **Step 1: Deploy Modal (user runs manually)**

```sh
python modal/serve.py deploy
```

- [ ] **Step 2: Smoke test embed via Modal (user runs manually)**

```sh
# Set modal.inference=true in ~/.config/qmd/index.yml first
qmd embed --force
```

- [ ] **Step 3: Verify no commit needed here** (Modal deploy is a runtime action)

---

## Summary of Commits

| # | Message |
|---|---------|
| 1 | `feat(modal): add tokenize endpoint to QMDInference` |
| 2 | `feat(llm): add tokenize/countTokens to LLM interface` |
| 3 | `feat(modal): add tokenize() to ModalBackend` |
| 4 | `feat(llm): implement tokenize/countTokens in ModalLLM and ModalSession` |
| 5 | `refactor(store): route embed through getDefaultLLM() for Modal support` |
| 6 | `test(modal): add ModalBackend.tokenize() unit tests` |
| 7 | `test(llm): add ModalLLM tokenize/countTokens unit tests` |
| 8 | `test: full suite passes after Modal embed routing` |
