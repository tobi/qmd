/**
 * rerank-timeout.test.ts - A hung native rerank (e.g. a GPU fault leaving
 * ggml-vulkan waiting on a fence forever) must not hang the caller.
 * Mocks only: no model is loaded and no native code runs.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import { runBenchmark } from "../src/bench/bench.js";
import { createStore as createIndexStore } from "../src/index.js";
import * as llmModule from "../src/llm.js";
import { LlamaCpp, RerankTimeoutError, resolveRerankTimeoutMs, isLlamaPoisoned, resetLlamaPoisonedForTests } from "../src/llm.js";
import {
  createStore,
  structuredSearch,
  insertContent,
  insertDocument,
  hashContent,
  resolveRerankMaxDocChars,
  type Store,
} from "../src/store.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-rerank-timeout-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  store = createStore(join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
  process.env.QMD_RERANK_TIMEOUT_MS = "50";
});

afterEach(() => {
  delete process.env.QMD_RERANK_TIMEOUT_MS;
  delete process.env.QMD_RERANK_MAX_DOC_CHARS;
  resetLlamaPoisonedForTests();
  vi.restoreAllMocks();
  store.close();
});

async function insertDoc(path: string, body: string): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(body);
  insertContent(store.db, hash, body, now);
  insertDocument(store.db, "notes", path, path, hash, now, now);
}

function hangingLlm() {
  return {
    rerank: vi.fn(() => new Promise<never>(() => {})),
    rerankModelName: "hf:example/rerank/hang.gguf",
    poison: vi.fn(),
  };
}

describe("rerank timeout", () => {
  test("resolveRerankTimeoutMs defaults to 60s and honours QMD_RERANK_TIMEOUT_MS", () => {
    delete process.env.QMD_RERANK_TIMEOUT_MS;
    expect(resolveRerankTimeoutMs()).toBe(60_000);
    process.env.QMD_RERANK_TIMEOUT_MS = "1500";
    expect(resolveRerankTimeoutMs()).toBe(1500);
    process.env.QMD_RERANK_TIMEOUT_MS = "garbage";
    expect(resolveRerankTimeoutMs()).toBe(60_000);
    // setTimeout would clamp this to 1 ms and time out every rerank instantly.
    process.env.QMD_RERANK_TIMEOUT_MS = "9999999999";
    expect(resolveRerankTimeoutMs()).toBe(2_147_483_647);
  });

  test("store.rerank rejects with RerankTimeoutError within budget and poisons the LLM", async () => {
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const started = Date.now();
    await expect(store.rerank("q", [{ file: "a.md", text: "alpha" }])).rejects.toBeInstanceOf(RerankTimeoutError);
    expect(Date.now() - started).toBeLessThan(2000);
    expect(llm.poison).toHaveBeenCalledTimes(1);
  });

  test("a real LlamaCpp is poisoned after a timeout and the process-wide flag is set", async () => {
    const real = new LlamaCpp({});
    vi.spyOn(real, "prepareRerank").mockResolvedValue(undefined); // no native load
    vi.spyOn(real, "rerank").mockImplementation(() => new Promise<never>(() => {}));
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(real);
    expect(isLlamaPoisoned()).toBe(false);

    await expect(store.rerank("q", [{ file: "a.md", text: "alpha" }])).rejects.toBeInstanceOf(RerankTimeoutError);
    expect(real.poisoned).toBe(true);
    expect(isLlamaPoisoned()).toBe(true);
  });

  test("structuredSearch falls back to RRF results flagged rerankTimedOut", async () => {
    await insertDoc("a.md", "# A\n\nfallback keyword");
    await insertDoc("b.md", "# B\n\nfallback keyword");
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const started = Date.now();
    const results = await structuredSearch(store, [{ type: "lex", query: "fallback keyword" }], {});
    expect(Date.now() - started).toBeLessThan(2000);

    expect(results.length).toBe(2);
    expect(results.every(r => r.rerankTimedOut === true)).toBe(true);
    // Pure RRF scores: 1/rank, so the order is preserved and rank 1 scores 1.
    expect(results[0]!.score).toBe(1);
    expect(results[1]!.score).toBe(0.5);

    const skipped = await structuredSearch(store, [{ type: "lex", query: "fallback keyword" }], { skipRerank: true });
    expect(results.map(r => r.file)).toEqual(skipped.map(r => r.file));
    expect(skipped.every(r => r.rerankTimedOut === undefined)).toBe(true);
  });
});

describe("rerank budget", () => {
  test("cold model load (prepareRerank) is outside the timeout budget", async () => {
    const llm = {
      rerankModelName: "hf:example/rerank/slow-load.gguf",
      poison: vi.fn(),
      prepareRerank: vi.fn(() => new Promise<void>(r => setTimeout(r, 200))),
      rerank: vi.fn(async (_q: string, docs: { file: string; text: string }[]) => ({
        results: docs.map((d, index) => ({ file: d.file, score: 0.7, index })), model: "m",
      })),
    };
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const out = await store.rerank("q", [{ file: "a.md", text: "alpha" }]); // budget is 50 ms
    expect(out[0]!.score).toBe(0.7);
    expect(llm.poison).not.toHaveBeenCalled();
  });
});

describe("poisoned LlamaCpp", () => {
  test("dispose leaves a poisoned instance alone (a native call is still running on it)", async () => {
    const llm = new LlamaCpp({});
    llm.poison("test");
    await llm.dispose();
    await llm.unloadIdleResources();
    expect((llm as any).disposed).toBe(false);
  });

  test("refuses every native entry point without touching native code", async () => {
    const llm = new LlamaCpp({});
    const ensure = vi.spyOn(llm as any, "ensureLlama");
    llm.poison("test");

    expect(llm.poisoned).toBe(true);
    await expect(llm.rerank("q", [{ file: "a", text: "t" }])).rejects.toThrow(/poisoned/);
    await expect(llm.embed("t")).rejects.toThrow(/poisoned/);
    await expect(llm.embedBatch(["t"])).rejects.toThrow(/poisoned/);
    await expect(llm.generate("p")).rejects.toThrow(/poisoned/);
    await expect(llm.expandQuery("q")).rejects.toThrow(/poisoned/);
    expect(ensure).not.toHaveBeenCalled();
  });
});

describe("per-document rerank truncation", () => {
  test("store.rerank sends at most QMD_RERANK_MAX_DOC_CHARS per document", async () => {
    process.env.QMD_RERANK_MAX_DOC_CHARS = "100";
    expect(resolveRerankMaxDocChars()).toBe(100);
    const sent: string[] = [];
    const llm = {
      rerankModelName: "hf:example/rerank/trunc.gguf",
      poison: vi.fn(),
      rerank: vi.fn(async (_q: string, docs: { file: string; text: string }[]) => {
        sent.push(...docs.map(d => d.text));
        return { results: docs.map((d, index) => ({ file: d.file, score: 0.5, index })), model: "m" };
      }),
    };
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const out = await store.rerank("q", [
      { file: "long.md", text: "x".repeat(5000) },
      { file: "short.md", text: "tiny" },
    ]);
    expect(sent.map(t => t.length).sort((a, b) => a - b)).toEqual([4, 100]);
    expect(out.map(r => r.file).sort()).toEqual(["long.md", "short.md"]);
    // Cache stays keyed on the full chunk, so the second call is fully cached.
    await store.rerank("q", [{ file: "long.md", text: "x".repeat(5000) }]);
    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  test("a different QMD_RERANK_MAX_DOC_CHARS does not replay cached scores", async () => {
    const llm = {
      rerankModelName: "hf:example/rerank/keyed.gguf",
      poison: vi.fn(),
      rerank: vi.fn(async (_q: string, docs: { file: string; text: string }[]) => ({
        results: docs.map((d, index) => ({ file: d.file, score: 0.5, index })),
        model: "m",
      })),
    };
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);
    const docs = [{ file: "long.md", text: "x".repeat(5000) }];

    process.env.QMD_RERANK_MAX_DOC_CHARS = "100";
    await store.rerank("q", docs);
    await store.rerank("q", docs);
    expect(llm.rerank).toHaveBeenCalledTimes(1);

    process.env.QMD_RERANK_MAX_DOC_CHARS = "200";
    await store.rerank("q", docs);
    expect(llm.rerank).toHaveBeenCalledTimes(2);
  });

  test("default budget is generous (6000 chars)", () => {
    delete process.env.QMD_RERANK_MAX_DOC_CHARS;
    expect(resolveRerankMaxDocChars()).toBe(6000);
  });
});

describe("bench records a timed-out rerank row", () => {
  test("the full backend is flagged rerank_timed_out instead of aborting", async () => {
    const docs = join(testDir, "bench-docs");
    await mkdir(docs, { recursive: true });
    await writeFile(join(docs, "api.md"), "# API versioning\n\nUse /v1 and /v2 endpoints.\n");
    await writeFile(join(docs, "api2.md"), "# API versioning notes\n\nMore on versioning.\n");
    const dbPath = join(testDir, `bench-${Date.now()}.sqlite`);
    const config = { collections: { docs: { path: docs, pattern: "**/*.md" } } };
    const setup = await createIndexStore({ dbPath, config });
    await setup.update();
    await setup.close();

    const fixturePath = join(testDir, "bench-fixture.json");
    await writeFile(fixturePath, JSON.stringify({
      description: "timeout fixture",
      version: 1,
      collection: "docs",
      queries: [{
        id: "q1",
        query: "lex: API versioning",
        type: "exact",
        description: "keyword",
        expected_files: ["api.md"],
        expected_in_top_k: 1,
      }],
    }));
    // The bench store builds its own LlamaCpp, so stub the instance methods.
    vi.spyOn(LlamaCpp.prototype, "prepareRerank").mockResolvedValue(undefined);
    vi.spyOn(LlamaCpp.prototype, "rerank").mockImplementation(() => new Promise<never>(() => {}));
    vi.spyOn(LlamaCpp.prototype, "poison").mockImplementation(() => {});

    const result = await runBenchmark(fixturePath, { json: true, dbPath, backends: ["full"], config });
    const row = result.results[0]!.backends.full!;
    expect(row.rerank_timed_out).toBe(true);
    expect(row.top_files.length).toBeGreaterThan(0);
  });
});
