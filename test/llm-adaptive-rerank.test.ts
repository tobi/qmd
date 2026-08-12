/**
 * llm-adaptive-rerank.test.ts — Adaptive rerank-context sizing.
 *
 * `resolveRerankContextSize` picks 4096 or 2048 based on free VRAM, with
 * the `QMD_RERANK_CONTEXT_SIZE` env override always winning. These tests
 * exercise the resolver and its callers (`ensureRerankContexts`,
 * `rerankImpl`) without loading real models by monkey-patching `ensureLlama`
 * + private methods on a real LlamaCpp instance.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { LlamaCpp } from "../src/llm.js";

type FakeLlama = {
  gpu: false | string;
  getVramState: () => Promise<{ total: number; used: number; free: number }>;
};

function patchLlama(llm: LlamaCpp, llama: FakeLlama): void {
  (llm as unknown as { ensureLlama: () => Promise<FakeLlama> }).ensureLlama = async () => llama;
  (llm as unknown as { isCpuOffloadForced: () => boolean }).isCpuOffloadForced = () => false;
}

function gb(n: number): number {
  return n * 1024 * 1024 * 1024;
}

describe("LlamaCpp resolveRerankContextSize — env override", () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env.QMD_RERANK_CONTEXT_SIZE;
    delete process.env.QMD_RERANK_CONTEXT_SIZE;
  });

  afterEach(() => {
    if (originalEnv === undefined) delete process.env.QMD_RERANK_CONTEXT_SIZE;
    else process.env.QMD_RERANK_CONTEXT_SIZE = originalEnv;
  });

  test("env override returns the user-supplied size regardless of VRAM state", async () => {
    process.env.QMD_RERANK_CONTEXT_SIZE = "8192";
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // 0.5 GB free — way below the tight threshold; adaptive would pick 2048.
      getVramState: async () => ({ total: gb(24), used: gb(23.5), free: gb(0.5) }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(8192);
  });

  test("env override is respected even when VRAM is plentiful", async () => {
    process.env.QMD_RERANK_CONTEXT_SIZE = "1024";
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      getVramState: async () => ({ total: gb(24), used: gb(0), free: gb(24) }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(1024);
  });

  test.each([
    ["0", false],
    ["-1", false],
    ["not-a-number", false],
    ["", false],
  ])("env override %p (invalid) falls back to adaptive logic", async (raw, _) => {
    process.env.QMD_RERANK_CONTEXT_SIZE = raw;
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      getVramState: async () => ({ total: gb(24), used: gb(0), free: gb(24) }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
  });
});

describe("LlamaCpp resolveRerankContextSize — adaptive VRAM logic", () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env.QMD_RERANK_CONTEXT_SIZE;
    delete process.env.QMD_RERANK_CONTEXT_SIZE;
  });

  afterEach(() => {
    if (originalEnv === undefined) delete process.env.QMD_RERANK_CONTEXT_SIZE;
    else process.env.QMD_RERANK_CONTEXT_SIZE = originalEnv;
  });

  test("CPU mode (no GPU) returns the default size without probing VRAM", async () => {
    const llm = new LlamaCpp({});
    let probed = false;
    patchLlama(llm, {
      gpu: false,
      getVramState: async () => { probed = true; return { total: 0, used: 0, free: 0 }; },
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
    expect(probed).toBe(false);
  });

  test("CPU offload forced returns the default size without probing", async () => {
    const llm = new LlamaCpp({});
    let probed = false;
    (llm as unknown as { ensureLlama: () => Promise<FakeLlama> }).ensureLlama = async () => ({
      gpu: "cuda",
      getVramState: async () => { probed = true; return { total: 0, used: 0, free: 0 }; },
    });
    (llm as unknown as { isCpuOffloadForced: () => boolean }).isCpuOffloadForced = () => true;

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
    expect(probed).toBe(false);
  });

  test("plentiful free VRAM (24 GB) returns the default size", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      getVramState: async () => ({ total: gb(24), used: gb(0), free: gb(24) }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
  });

  test("free VRAM exactly at threshold (1500 MB) returns the default size", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // Exactly 1500 MB free. Boundary: < threshold shrinks, >= threshold stays.
      getVramState: async () => ({ total: gb(24), used: gb(23) - 1500 * 1024 * 1024, free: 1500 * 1024 * 1024 }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
  });

  test("free VRAM just below threshold returns the tight size", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // 1499 MB free — just under the 1500 MB threshold.
      getVramState: async () => ({ total: gb(24), used: gb(24) - 1499 * 1024 * 1024, free: 1499 * 1024 * 1024 }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(2048);
  });

  test("Ollama-hogged GPU (~1.4 GB free) returns the tight size", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // Reproduces the actual failure mode: Ollama has 21.6 GB pinned of 24 GB.
      getVramState: async () => ({ total: gb(24), used: gb(22.6), free: gb(1.4) }),
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(2048);
  });

  test("getVramState throwing falls back to the default size (conservative)", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      getVramState: async () => { throw new Error("VRAM state unavailable"); },
    });

    const size = await (llm as unknown as { resolveRerankContextSize: () => Promise<number> }).resolveRerankContextSize();
    expect(size).toBe(4096);
  });
});

describe("LlamaCpp ensureRerankContexts — wiring", () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env.QMD_RERANK_CONTEXT_SIZE;
    delete process.env.QMD_RERANK_CONTEXT_SIZE;
  });

  afterEach(() => {
    if (originalEnv === undefined) delete process.env.QMD_RERANK_CONTEXT_SIZE;
    else process.env.QMD_RERANK_CONTEXT_SIZE = originalEnv;
  });

  test("stores the resolved size on rerankContextSize for rerankImpl to read", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // 1.4 GB free → tight (2048).
      getVramState: async () => ({ total: gb(24), used: gb(22.6), free: gb(1.4) }),
    });
    const created: number[] = [];
    (llm as unknown as { ensureRerankModel: () => Promise<unknown> }).ensureRerankModel = async () => ({
      createRankingContext: async ({ contextSize }: { contextSize: number }) => {
        created.push(contextSize);
        return { dispose: async () => undefined };
      },
    });
    (llm as unknown as { computeParallelism: (mb: number) => Promise<number> }).computeParallelism = async () => 1;
    (llm as unknown as { threadsPerContext: (n: number) => Promise<number> }).threadsPerContext = async () => 0;

    await (llm as unknown as { ensureRerankContexts: () => Promise<unknown> }).ensureRerankContexts();

    expect(created).toEqual([2048]);
    expect((llm as unknown as { rerankContextSize: number }).rerankContextSize).toBe(2048);
  });

  test("passes perContextMB derived from the actual chosen size, not a hardcoded 1000", async () => {
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // Plentiful — picks 4096.
      getVramState: async () => ({ total: gb(24), used: gb(0), free: gb(24) }),
    });
    (llm as unknown as { ensureRerankModel: () => Promise<unknown> }).ensureRerankModel = async () => ({
      createRankingContext: async () => ({ dispose: async () => undefined }),
    });
    let observedMB = -1;
    (llm as unknown as { computeParallelism: (mb: number) => Promise<number> }).computeParallelism = async (mb: number) => {
      observedMB = mb;
      return 1;
    };
    (llm as unknown as { threadsPerContext: (n: number) => Promise<number> }).threadsPerContext = async () => 0;

    await (llm as unknown as { ensureRerankContexts: () => Promise<unknown> }).ensureRerankContexts();

    // 4096 * 0.28 ≈ 1146 MB. The old hardcode was 1000 MB.
    expect(observedMB).toBeGreaterThan(1000);
    expect(observedMB).toBeLessThanOrEqual(1200);
  });

  test("uses the env-override size when it disagrees with the adaptive choice", async () => {
    process.env.QMD_RERANK_CONTEXT_SIZE = "1024";
    const llm = new LlamaCpp({});
    patchLlama(llm, {
      gpu: "cuda",
      // Adaptive would say 4096; env override 1024 must win.
      getVramState: async () => ({ total: gb(24), used: gb(0), free: gb(24) }),
    });
    const created: number[] = [];
    (llm as unknown as { ensureRerankModel: () => Promise<unknown> }).ensureRerankModel = async () => ({
      createRankingContext: async ({ contextSize }: { contextSize: number }) => {
        created.push(contextSize);
        return { dispose: async () => undefined };
      },
    });
    let observedMB = -1;
    (llm as unknown as { computeParallelism: (mb: number) => Promise<number> }).computeParallelism = async (mb: number) => {
      observedMB = mb;
      return 1;
    };
    (llm as unknown as { threadsPerContext: (n: number) => Promise<number> }).threadsPerContext = async () => 0;

    await (llm as unknown as { ensureRerankContexts: () => Promise<unknown> }).ensureRerankContexts();

    expect(created).toEqual([1024]);
    // 1024 * 0.28 ≈ 287 MB. Per-context budget must reflect the smaller size.
    expect(observedMB).toBeGreaterThanOrEqual(286);
    expect(observedMB).toBeLessThanOrEqual(310);
  });
});

describe("LlamaCpp rerankContextSize lifecycle — reset on teardown", () => {
  test("disposeRerankModel clears rerankContextSize so next call re-probes", async () => {
    const llm = new LlamaCpp({});
    (llm as unknown as { rerankContextSize: number | null }).rerankContextSize = 2048;
    (llm as unknown as { rerankContexts: unknown[] }).rerankContexts = [];

    await (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel();

    expect((llm as unknown as { rerankContextSize: number | null }).rerankContextSize).toBeNull();
  });

  test("dispose clears rerankContextSize", async () => {
    const llm = new LlamaCpp({});
    (llm as unknown as { rerankContextSize: number | null }).rerankContextSize = 2048;
    (llm as unknown as { rerankContexts: unknown[] }).rerankContexts = [];

    await llm.dispose();

    expect((llm as unknown as { rerankContextSize: number | null }).rerankContextSize).toBeNull();
  });
});
