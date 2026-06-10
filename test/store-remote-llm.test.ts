/**
 * store-remote-llm.test.ts - Store layer routes through the polymorphic LLM
 * default (not the local-only LlamaCpp singleton), so QMD_REMOTE_URL actually
 * reaches RemoteLLM.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  setDefaultLLM,
  setDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  type LLM,
  type EmbedOptions,
  type EmbeddingResult,
  type GenerateOptions,
  type GenerateResult,
  type Queryable,
  type RerankDocument,
  type RerankResult,
  type ModelInfo,
} from "../src/llm.js";
import { RemoteLLM } from "../src/llm-remote.js";
import { expandQuery, rerank, createStore, generateEmbeddings, chunkDocumentByTokens } from "../src/store.js";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

class CountingLLM implements LLM {
  calls = {
    embed: 0,
    embedBatch: 0,
    expandQuery: 0,
    rerank: 0,
    tokenize: 0,
    dispose: 0,
  };
  readonly embedModelName = "fake-embed-model";
  readonly generateModelName = "fake-generate-model";
  readonly rerankModelName = "fake-rerank-model";

  async embed(_text: string, _options?: EmbedOptions): Promise<EmbeddingResult | null> {
    this.calls.embed++;
    return { embedding: [0.1, 0.2, 0.3] };
  }
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    this.calls.embedBatch++;
    return texts.map(() => ({ embedding: [0.1, 0.2, 0.3] }));
  }
  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    return null;
  }
  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }
  async expandQuery(
    _query: string,
    _options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    this.calls.expandQuery++;
    return [{ type: "lex", text: "expanded-lex" }, { type: "vec", text: "expanded-vec" }];
  }
  async rerank(
    _query: string,
    documents: RerankDocument[],
    _options?: { model?: string },
  ): Promise<RerankResult> {
    this.calls.rerank++;
    return {
      results: documents.map((d, i) => ({ file: d.file, score: documents.length - i })),
    };
  }
  async tokenize(text: string): Promise<readonly unknown[]> {
    this.calls.tokenize++;
    return new Array(Math.ceil(text.length / 4)).fill(0);
  }
  async dispose(): Promise<void> {
    this.calls.dispose++;
  }
}

describe("store layer routes through the polymorphic LLM default", () => {
  let testDir: string;
  let store: ReturnType<typeof createStore>;
  let originalRemoteUrl: string | undefined;

  beforeEach(async () => {
    originalRemoteUrl = process.env.QMD_REMOTE_URL;
    delete process.env.QMD_REMOTE_URL;
    testDir = await mkdtemp(join(tmpdir(), "qmd-remote-llm-"));
    store = createStore(join(testDir, "store.sqlite"));
    setDefaultLlamaCpp(null);
    setDefaultLLM(null);
  });

  afterEach(async () => {
    store.close();
    await rm(testDir, { recursive: true, force: true });
    setDefaultLlamaCpp(null);
    setDefaultLLM(null);
    await disposeDefaultLlamaCpp();
    if (originalRemoteUrl === undefined) {
      delete process.env.QMD_REMOTE_URL;
    } else {
      process.env.QMD_REMOTE_URL = originalRemoteUrl;
    }
  });

  test("expandQuery routes through llmOverride when a non-LlamaCpp LLM is passed", async () => {
    const fake = new CountingLLM();
    const results = await expandQuery("rclone bisync", "ignored-model", store.db, "test intent", fake);
    expect(fake.calls.expandQuery).toBe(1);
    expect(results.map(r => r.query)).toContain("expanded-lex");
  });

  test("rerank routes through llmOverride when a non-LlamaCpp LLM is passed", async () => {
    const fake = new CountingLLM();
    const docs = [
      { file: "a.md", text: "alpha" },
      { file: "b.md", text: "beta" },
    ];
    const results = await rerank("rclone bisync", docs, "ignored-model", store.db, undefined, fake);
    expect(fake.calls.rerank).toBe(1);
    expect(results.map(r => r.file).sort()).toEqual(["a.md", "b.md"]);
  });

  test("expandQuery routes through the global default LLM when no override is passed", async () => {
    const fake = new CountingLLM();
    setDefaultLLM(fake);
    await expandQuery("rclone bisync uncached", "ignored-model", store.db);
    expect(fake.calls.expandQuery).toBe(1);
  });

  test("rerank routes through the global default LLM when no override is passed", async () => {
    const fake = new CountingLLM();
    setDefaultLLM(fake);
    const docs = [{ file: "c.md", text: "gamma" }];
    await rerank("query for global default", docs, "ignored-model", store.db);
    expect(fake.calls.rerank).toBe(1);
  });

  test("Store.expandQuery (adapter) routes through Store.llm when set to a non-LlamaCpp LLM", async () => {
    const fake = new CountingLLM();
    store.llm = fake;
    await store.expandQuery("hybrid query path");
    expect(fake.calls.expandQuery).toBe(1);
  });

  test("Store.rerank (adapter) routes through Store.llm when set to a non-LlamaCpp LLM", async () => {
    const fake = new CountingLLM();
    store.llm = fake;
    await store.rerank("hybrid query path", [{ file: "d.md", text: "delta" }]);
    expect(fake.calls.rerank).toBe(1);
  });

  test("generateEmbeddings accepts a real RemoteLLM (no requireLlamaCpp gate)", async () => {
    // Spin a minimal /health responder so RemoteLLM can warm up.
    const server = createServer((req, res) => {
      if (req.url === "/health") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          ok: true,
          version: "2",
          backend: "local",
          models: { embed: "test-embed", rerank: "test-rerank", generate: "test-generate" },
        }));
        return;
      }
      res.writeHead(404);
      res.end();
    });
    await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
    const port = (server.address() as AddressInfo).port;

    try {
      const remote = new RemoteLLM({ serverUrl: `http://127.0.0.1:${port}` });
      store.llm = remote;
      // No documents inserted → early return path. Pre-fix this still threw
      // "requires a local LlamaCpp backend" because the gate ran before the
      // empty-pending check. Post-fix it returns cleanly.
      const result = await generateEmbeddings(store);
      expect(result.docsProcessed).toBe(0);
      expect(result.chunksEmbedded).toBe(0);
      await remote.dispose();
    } finally {
      await new Promise<void>(r => server.close(() => r()));
    }
  });

  test("chunkDocumentByTokens falls back to char truncation when llm has no detokenize", async () => {
    // Fake LLM with tokenize but no detokenize — exercises the new fallback path.
    class TokenOnlyLLM implements LLM {
      readonly embedModelName = "fake-embed";
      async embed(): Promise<EmbeddingResult | null> { return null; }
      async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
        return texts.map(() => null);
      }
      async generate(): Promise<GenerateResult | null> { return null; }
      async modelExists(model: string): Promise<ModelInfo> { return { name: model, exists: true }; }
      async expandQuery(): Promise<Queryable[]> { return []; }
      async rerank(_q: string, docs: RerankDocument[]): Promise<RerankResult> {
        return { results: docs.map(d => ({ file: d.file, score: 0 })) };
      }
      async tokenize(text: string): Promise<readonly unknown[]> {
        // 2 chars per "token" — exaggerates so the maxTokens limit triggers fallback.
        return new Array(Math.ceil(text.length / 2)).fill(0);
      }
      async dispose(): Promise<void> {}
    }
    setDefaultLLM(new TokenOnlyLLM());
    // Pathological single-line blob: no whitespace, length forces re-split, but
    // every sub-split returns the same text → fallback char-truncate kicks in.
    const blob = "x".repeat(4000);
    const chunks = await chunkDocumentByTokens(blob, 50, 5, 10);
    expect(chunks.length).toBeGreaterThan(0);
    // Every chunk must be a non-empty string (no thrown detokenize errors).
    for (const c of chunks) {
      expect(typeof c.text).toBe("string");
      expect(c.text.length).toBeGreaterThan(0);
    }
  });
});
