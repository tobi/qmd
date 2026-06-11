/**
 * Unit tests for the VRAM precondition helper that gates the GPU-dependent
 * integration suites. Exercises the helper with an injected fake `getLlama`
 * — no real node-llama-cpp runtime is loaded.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import {
  hasSufficientFreeVramForIntegration,
  _resetVramPreconditionCacheForTests,
  type LlamaLike,
} from "./_helpers/vram-precondition.js";

function gb(n: number): number {
  return n * 1024 * 1024 * 1024;
}

function fakeLlama(opts: { gpu: false | string; free?: number; throwOnVram?: boolean }): LlamaLike {
  return {
    gpu: opts.gpu,
    getVramState: async () => {
      if (opts.throwOnVram) throw new Error("simulated probe failure");
      return { total: gb(24), used: gb(24) - (opts.free ?? 0), free: opts.free ?? 0 };
    },
  };
}

describe("hasSufficientFreeVramForIntegration", () => {
  let originalSkip: string | undefined;

  beforeEach(() => {
    _resetVramPreconditionCacheForTests();
    originalSkip = process.env.QMD_SKIP_GPU_INTEGRATION;
    delete process.env.QMD_SKIP_GPU_INTEGRATION;
  });

  afterEach(() => {
    if (originalSkip === undefined) delete process.env.QMD_SKIP_GPU_INTEGRATION;
    else process.env.QMD_SKIP_GPU_INTEGRATION = originalSkip;
  });

  test("returns true on CPU mode (no GPU) without checking VRAM", async () => {
    let vramChecked = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => ({
        gpu: false,
        getVramState: async () => { vramChecked = true; return { total: 0, used: 0, free: 0 }; },
      }),
      log: () => {},
    });
    expect(result).toBe(true);
    expect(vramChecked).toBe(false);
  });

  test("returns true when GPU has plenty of free VRAM", async () => {
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(20) }),
      log: () => {},
    });
    expect(result).toBe(true);
  });

  test("returns true at exactly the threshold", async () => {
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(4) }),
      minFreeGB: 4,
      log: () => {},
    });
    expect(result).toBe(true);
  });

  test("returns false just below the threshold", async () => {
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(4) - 1 }),
      minFreeGB: 4,
      log: () => {},
    });
    expect(result).toBe(false);
  });

  test("returns false on the Ollama-hogged scenario (~1.4 GB free)", async () => {
    const messages: string[] = [];
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(1.4) }),
      log: (m) => messages.push(m),
    });
    expect(result).toBe(false);
    expect(messages.some((m) => m.includes("1.40 GB free") && m.includes("QMD_SKIP_GPU_INTEGRATION"))).toBe(true);
  });

  test("QMD_SKIP_GPU_INTEGRATION=1 forces a skip without probing", async () => {
    process.env.QMD_SKIP_GPU_INTEGRATION = "1";
    let probed = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => { probed = true; return fakeLlama({ gpu: "cuda", free: gb(20) }); },
      log: () => {},
    });
    expect(result).toBe(false);
    expect(probed).toBe(false);
  });

  test("getLlama throwing fails open (returns true) so a real failure surfaces in the test", async () => {
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => { throw new Error("node-llama-cpp not built for this platform"); },
      log: () => {},
    });
    expect(result).toBe(true);
  });

  test("getVramState throwing fails open", async () => {
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", throwOnVram: true }),
      log: () => {},
    });
    expect(result).toBe(true);
  });

  test("caches the result across calls (probe runs once)", async () => {
    let probeCalls = 0;
    const opts = {
      getLlama: async () => { probeCalls++; return fakeLlama({ gpu: "cuda", free: gb(20) }); },
      log: () => {},
    };
    await hasSufficientFreeVramForIntegration(opts);
    await hasSufficientFreeVramForIntegration(opts);
    await hasSufficientFreeVramForIntegration(opts);
    expect(probeCalls).toBe(1);
  });

  test("only logs the skip message once across calls", async () => {
    const messages: string[] = [];
    const opts = {
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(1) }),
      log: (m: string) => messages.push(m),
    };
    await hasSufficientFreeVramForIntegration(opts);
    await hasSufficientFreeVramForIntegration(opts);
    expect(messages).toHaveLength(1);
  });

  test("concurrent first calls do not over-probe (cache wins after first resolve)", async () => {
    let probeCalls = 0;
    const opts = {
      getLlama: async () => {
        probeCalls++;
        await new Promise<void>((r) => setImmediate(r));
        return fakeLlama({ gpu: "cuda", free: gb(20) });
      },
      log: () => {},
    };

    // Two simultaneous awaits.  The current implementation does not coalesce
    // racing probes; this test documents that observed behavior so a future
    // optimization (in-flight promise sharing) is a deliberate change rather
    // than an accidental regression.
    const [a, b] = await Promise.all([
      hasSufficientFreeVramForIntegration(opts),
      hasSufficientFreeVramForIntegration(opts),
    ]);
    expect(a).toBe(true);
    expect(b).toBe(true);
    // After both resolve, subsequent calls hit the cache and do not probe again.
    await hasSufficientFreeVramForIntegration(opts);
    await hasSufficientFreeVramForIntegration(opts);
    // The first batch may double-probe; the second batch must not.
    expect(probeCalls).toBeLessThanOrEqual(2);
  });

  test("QMD_SKIP_GPU_INTEGRATION values other than '1' do not force skip (precise env match)", async () => {
    // Red-team: prevent any truthy-string leak (e.g. "0", "false", "no") from
    // accidentally forcing a skip.  Only the exact string "1" triggers.
    for (const raw of ["0", "false", "no", "true", "yes", ""]) {
      _resetVramPreconditionCacheForTests();
      process.env.QMD_SKIP_GPU_INTEGRATION = raw;
      const result = await hasSufficientFreeVramForIntegration({
        getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(20) }),
        log: () => {},
      });
      expect(result, `value ${JSON.stringify(raw)} should not force skip`).toBe(true);
    }
  });
});
