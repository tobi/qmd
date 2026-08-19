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
  let originalEvacuate: string | undefined;
  let originalOllamaHost: string | undefined;

  beforeEach(() => {
    _resetVramPreconditionCacheForTests();
    // Snapshot AND clear every env var this helper reads.  A leaked
    // `QMD_TEST_EVACUATE_VRAM=1` from the surrounding process would otherwise
    // activate the eviction path inside tests that expect a clean skip,
    // turning false-returning probes into true-returning re-probes against
    // a real Ollama.  Same for `OLLAMA_HOST`: a leaked override could send
    // the default eviction step to the wrong endpoint.
    originalSkip = process.env.QMD_SKIP_GPU_INTEGRATION;
    originalEvacuate = process.env.QMD_TEST_EVACUATE_VRAM;
    originalOllamaHost = process.env.OLLAMA_HOST;
    delete process.env.QMD_SKIP_GPU_INTEGRATION;
    delete process.env.QMD_TEST_EVACUATE_VRAM;
    delete process.env.OLLAMA_HOST;
  });

  afterEach(() => {
    if (originalSkip === undefined) delete process.env.QMD_SKIP_GPU_INTEGRATION;
    else process.env.QMD_SKIP_GPU_INTEGRATION = originalSkip;
    if (originalEvacuate === undefined) delete process.env.QMD_TEST_EVACUATE_VRAM;
    else process.env.QMD_TEST_EVACUATE_VRAM = originalEvacuate;
    if (originalOllamaHost === undefined) delete process.env.OLLAMA_HOST;
    else process.env.OLLAMA_HOST = originalOllamaHost;
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

describe("hasSufficientFreeVramForIntegration — QMD_TEST_EVACUATE_VRAM opt-in", () => {
  let originalEvacuate: string | undefined;
  let originalSkip: string | undefined;
  let originalOllamaHost: string | undefined;

  beforeEach(() => {
    _resetVramPreconditionCacheForTests();
    originalEvacuate = process.env.QMD_TEST_EVACUATE_VRAM;
    originalSkip = process.env.QMD_SKIP_GPU_INTEGRATION;
    originalOllamaHost = process.env.OLLAMA_HOST;
    delete process.env.QMD_TEST_EVACUATE_VRAM;
    delete process.env.QMD_SKIP_GPU_INTEGRATION;
    delete process.env.OLLAMA_HOST;
  });

  afterEach(() => {
    if (originalEvacuate === undefined) delete process.env.QMD_TEST_EVACUATE_VRAM;
    else process.env.QMD_TEST_EVACUATE_VRAM = originalEvacuate;
    if (originalSkip === undefined) delete process.env.QMD_SKIP_GPU_INTEGRATION;
    else process.env.QMD_SKIP_GPU_INTEGRATION = originalSkip;
    if (originalOllamaHost === undefined) delete process.env.OLLAMA_HOST;
    else process.env.OLLAMA_HOST = originalOllamaHost;
  });

  test("does NOT run evacuate when env unset (default behavior preserved)", async () => {
    let evacuateCalled = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(1) }),
      evacuate: async () => { evacuateCalled = true; },
      log: () => {},
    });
    expect(result).toBe(false);
    expect(evacuateCalled).toBe(false);
  });

  test("runs evacuate when env=1 and first probe insufficient, then re-probes", async () => {
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    let probeCalls = 0;
    let evacuateCalled = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => {
        probeCalls++;
        // First probe: tight.  Second probe (after evacuate): roomy.
        return fakeLlama({ gpu: "cuda", free: probeCalls === 1 ? gb(1) : gb(20) });
      },
      evacuate: async () => { evacuateCalled = true; },
      log: () => {},
    });
    expect(result).toBe(true);
    expect(evacuateCalled).toBe(true);
    expect(probeCalls).toBe(2);
  });

  test("evacuate runs but does not free enough → still returns false", async () => {
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    let evacuateCalled = false;
    const messages: string[] = [];
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(1) }),  // stays tight
      evacuate: async () => { evacuateCalled = true; },
      log: (m) => messages.push(m),
    });
    expect(result).toBe(false);
    expect(evacuateCalled).toBe(true);
    expect(messages.some((m) => m.includes("post-eviction"))).toBe(true);
  });

  test("evacuate throws → still re-probes (best-effort), logs the failure", async () => {
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    let probeCalls = 0;
    const messages: string[] = [];
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => {
        probeCalls++;
        return fakeLlama({ gpu: "cuda", free: probeCalls === 1 ? gb(1) : gb(20) });
      },
      evacuate: async () => { throw new Error("simulated eviction failure"); },
      log: (m) => messages.push(m),
    });
    expect(result).toBe(true);
    expect(probeCalls).toBe(2);
    expect(messages.some((m) => m.includes("Eviction step threw"))).toBe(true);
  });

  test("first probe already sufficient → evacuate is NOT called even with env=1", async () => {
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    let evacuateCalled = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(20) }),
      evacuate: async () => { evacuateCalled = true; },
      log: () => {},
    });
    expect(result).toBe(true);
    expect(evacuateCalled).toBe(false);
  });

  test("QMD_SKIP_GPU_INTEGRATION=1 wins over QMD_TEST_EVACUATE_VRAM=1 (no probe, no evacuate)", async () => {
    process.env.QMD_SKIP_GPU_INTEGRATION = "1";
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    let probeCalls = 0;
    let evacuateCalled = false;
    const result = await hasSufficientFreeVramForIntegration({
      getLlama: async () => { probeCalls++; return fakeLlama({ gpu: "cuda", free: gb(1) }); },
      evacuate: async () => { evacuateCalled = true; },
      log: () => {},
    });
    expect(result).toBe(false);
    expect(probeCalls).toBe(0);
    expect(evacuateCalled).toBe(false);
  });

  test("QMD_TEST_EVACUATE_VRAM values other than '1' do not trigger eviction (precise env match)", async () => {
    for (const raw of ["0", "false", "no", "true", "yes", ""]) {
      _resetVramPreconditionCacheForTests();
      process.env.QMD_TEST_EVACUATE_VRAM = raw;
      let evacuateCalled = false;
      const result = await hasSufficientFreeVramForIntegration({
        getLlama: async () => fakeLlama({ gpu: "cuda", free: gb(1) }),
        evacuate: async () => { evacuateCalled = true; },
        log: () => {},
      });
      expect(result, `value ${JSON.stringify(raw)} should not trigger eviction`).toBe(false);
      expect(evacuateCalled, `value ${JSON.stringify(raw)} should not call evacuate`).toBe(false);
    }
  });

  test("default evacuate path: Ollama unreachable does not throw (best-effort)", async () => {
    // Red-team: when the default evacuate runs against a host with no Ollama,
    // it must swallow the connection error and let the re-probe proceed.
    // We exercise the default evacuate (no injection) by aiming at a port
    // that nothing is listening on.
    process.env.QMD_TEST_EVACUATE_VRAM = "1";
    const originalHost = process.env.OLLAMA_HOST;
    process.env.OLLAMA_HOST = "http://127.0.0.1:1";  // reserved/unused port
    try {
      let probeCalls = 0;
      const messages: string[] = [];
      const result = await hasSufficientFreeVramForIntegration({
        getLlama: async () => { probeCalls++; return fakeLlama({ gpu: "cuda", free: gb(1) }); },
        log: (m) => messages.push(m),
        // No `evacuate` injected — exercises the real defaultEvacuate path.
      });
      expect(result).toBe(false);
      expect(probeCalls).toBe(2);  // initial + post-eviction re-probe
      expect(messages.some((m) => m.includes("Ollama unreachable"))).toBe(true);
    } finally {
      if (originalHost === undefined) delete process.env.OLLAMA_HOST;
      else process.env.OLLAMA_HOST = originalHost;
    }
  });
});
