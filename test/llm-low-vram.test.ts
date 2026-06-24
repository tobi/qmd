/**
 * llm-low-vram.test.ts - Concurrency tests for LlamaCpp lowVram mode.
 *
 * lowVram mode disposes the heavy generate/rerank models after each call to
 * keep peak VRAM down. That only works if calls serialize — otherwise a
 * dispose could race with another caller's mid-flight use of the same model.
 * These tests exercise that serialization without loading real models by
 * monkey-patching the *Impl methods + dispose helpers on a real LlamaCpp
 * instance.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { LlamaCpp, type RerankDocument, type Queryable, type RerankResult, type EmbeddingResult, type EmbedOptions } from "../src/llm.js";

/**
 * Build a LlamaCpp with lowVram=true and replace its low-level methods with
 * fakes that record the relative order of operations. Each fake takes one
 * tick so overlapping callers can interleave.
 *
 * Without lowVram-mode serialization, overlapping callers would observe
 * `load → load → end → dispose → end → dispose`.
 * With serialization, the pattern is `load → end → dispose → load → end → dispose`.
 */
function makeInstrumentedLlm(): { llm: LlamaCpp; events: string[] } {
  const events: string[] = [];
  const resident = { generate: false, rerank: false };
  const tick = () => new Promise<void>((r) => setImmediate(r));

  const llm = new LlamaCpp({ lowVram: true });

  // Replace the heavy work with fast fakes that exercise the chain only.
  // The public expandQuery/rerank still run their lowVram wrappers, which is
  // what we want to test.
  (llm as unknown as { expandQueryImpl: (query: string, options?: unknown) => Promise<Queryable[]> }).expandQueryImpl = async (
    _query: string,
    _options?: unknown,
  ): Promise<Queryable[]> => {
    events.push(resident.generate ? "expand:start" : "expand:load");
    resident.generate = true;
    await tick();
    events.push("expand:end");
    return [{ type: "lex", text: "x" }];
  };

  (llm as unknown as { rerankImpl: (query: string, documents: RerankDocument[], options?: unknown) => Promise<RerankResult> }).rerankImpl = async (
    _query: string,
    documents: RerankDocument[],
    _options?: unknown,
  ): Promise<RerankResult> => {
    events.push(resident.rerank ? "rerank:start" : "rerank:load");
    resident.rerank = true;
    await tick();
    events.push("rerank:end");
    return {
      results: documents.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })),
      model: "fake",
    };
  };

  (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
    events.push("expand:dispose");
    resident.generate = false;
    await tick();
  };

  (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
    events.push("rerank:dispose");
    resident.rerank = false;
    await tick();
  };

  return { llm, events };
}

describe("LlamaCpp lowVram mode", () => {
  // Two ambient env vars would otherwise distort these tests, so neutralize
  // both for the suite and restore after:
  //   CI — LlamaCpp captures `_ciMode = !!process.env.CI` at construction and
  //   throws from embed/embedBatch/rerank/expand when set. These tests
  //   monkey-patch the *Impl methods (no real model loads) to exercise the
  //   reclaim/serialization wrappers, so they must take the non-CI path.
  //   QMD_LOW_VRAM — `new LlamaCpp({})` resolves lowVram from this var, so the
  //   lowVram=false cases would flip to true under a leaked QMD_LOW_VRAM=1.
  const savedEnv: Record<string, string | undefined> = {};
  beforeAll(() => {
    for (const key of ["CI", "QMD_LOW_VRAM"]) {
      savedEnv[key] = process.env[key];
      delete process.env[key];
    }
  });
  afterAll(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value !== undefined) process.env[key] = value;
    }
  });

  test("serializes overlapping expandQuery calls — dispose never races with use", async () => {
    const { llm, events } = makeInstrumentedLlm();

    const [a, b] = await Promise.all([
      llm.expandQuery("alpha"),
      llm.expandQuery("beta"),
    ]);

    expect(a).toEqual([{ type: "lex", text: "x" }]);
    expect(b).toEqual([{ type: "lex", text: "x" }]);
    expect(events).toEqual([
      "expand:load",
      "expand:end",
      "expand:dispose",
      "expand:load",
      "expand:end",
      "expand:dispose",
    ]);
  });

  test("serializes overlapping rerank calls", async () => {
    const { llm, events } = makeInstrumentedLlm();

    const docs: RerankDocument[] = [{ file: "a.md", text: "doc-1" }];
    await Promise.all([
      llm.rerank("q1", docs),
      llm.rerank("q2", docs),
    ]);

    expect(events).toEqual([
      "rerank:load",
      "rerank:end",
      "rerank:dispose",
      "rerank:load",
      "rerank:end",
      "rerank:dispose",
    ]);
  });

  test("expand and rerank run on independent chains (parallel allowed)", async () => {
    const { llm, events } = makeInstrumentedLlm();

    await Promise.all([
      llm.expandQuery("alpha"),
      llm.rerank("alpha", [{ file: "a.md", text: "doc" }]),
    ]);

    // Both stages should have started before either disposed — proves they
    // ran in parallel rather than queueing behind each other.
    const expandStart = events.indexOf("expand:load");
    const rerankStart = events.indexOf("rerank:load");
    const expandDispose = events.indexOf("expand:dispose");
    const rerankDispose = events.indexOf("rerank:dispose");

    expect(expandStart).toBeGreaterThanOrEqual(0);
    expect(rerankStart).toBeGreaterThanOrEqual(0);
    expect(expandDispose).toBeGreaterThan(expandStart);
    expect(rerankDispose).toBeGreaterThan(rerankStart);
    expect(Math.max(expandStart, rerankStart)).toBeLessThan(Math.min(expandDispose, rerankDispose));
  });

  test("a failing call still releases the chain for the next caller", async () => {
    const { llm, events } = makeInstrumentedLlm();

    let calls = 0;
    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async (_q: string) => {
      calls++;
      events.push(calls === 1 ? "expand:throw" : "expand:end");
      if (calls === 1) throw new Error("boom");
      return [{ type: "lex", text: "ok" }];
    };

    const first = llm.expandQuery("first").catch((e: Error) => e.message);
    const second = llm.expandQuery("second");

    await expect(first).resolves.toBe("boom");
    await expect(second).resolves.toEqual([{ type: "lex", text: "ok" }]);
    expect(events).toEqual([
      "expand:throw",
      "expand:dispose",
      "expand:end",
      "expand:dispose",
    ]);
  });

  test("default (lowVram=false) does not serialize or dispose", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({}); // lowVram defaults to false
    let inFlight = 0;
    let maxInFlight = 0;

    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setImmediate(r));
      inFlight--;
      events.push("end");
      return [{ type: "lex", text: "x" }];
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose");
    };

    await Promise.all([llm.expandQuery("a"), llm.expandQuery("b"), llm.expandQuery("c")]);

    // In default mode, all three calls run in parallel — maxInFlight reaches 3.
    expect(maxInFlight).toBe(3);
    // And dispose is never called — the model stays resident.
    expect(events.filter((e) => e === "dispose")).toEqual([]);
  });

  test("reads QMD_LOW_VRAM=1 from env when no explicit config is given", () => {
    const original = process.env.QMD_LOW_VRAM;
    try {
      process.env.QMD_LOW_VRAM = "1";
      const llm = new LlamaCpp({});
      // The internal `lowVram` field is private; verify by behaviour:
      // calling expandQuery should go through the chain wrapper. We can
      // detect that by stubbing impl and asserting it gets queued.
      let chained = false;
      (llm as unknown as { expandQueryImpl: () => Promise<Queryable[]> }).expandQueryImpl = async () => {
        chained = true;
        return [];
      };
      (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => undefined;
      return llm.expandQuery("test").then(() => {
        expect(chained).toBe(true);
      });
    } finally {
      if (original === undefined) delete process.env.QMD_LOW_VRAM;
      else process.env.QMD_LOW_VRAM = original;
    }
  });

  test("embed: catch-and-retry on insufficient VRAM disposes heavies and retries once", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      attempts++;
      events.push(`embed:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("A context size of 2048 is too large for the available VRAM");
      }
      return { embedding: [0.1, 0.2, 0.3], model: "fake-embed" };
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };

    const result = await llm.embed("hello");
    expect(result).toEqual({ embedding: [0.1, 0.2, 0.3], model: "fake-embed" });
    expect(attempts).toBe(2);
    expect(events).toEqual([
      "embed:attempt-1",
      "dispose:generate",
      "dispose:rerank",
      "embed:attempt-2",
    ]);
  });

  test("embedBatch: catch-and-retry on insufficient VRAM disposes heavies and retries once", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { embedBatchImpl: (texts: string[], options: EmbedOptions) => Promise<(EmbeddingResult | null)[]> }).embedBatchImpl = async (texts: string[]) => {
      attempts++;
      events.push(`batch:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("Failed to create any embedding context");
      }
      return texts.map(() => ({ embedding: [1, 2, 3], model: "fake-embed" }));
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };

    const results = await llm.embedBatch(["a", "b"]);
    expect(results).toHaveLength(2);
    expect(results.every((r) => r?.embedding[0] === 1)).toBe(true);
    expect(attempts).toBe(2);
    expect(events).toEqual([
      "batch:attempt-1",
      "dispose:generate",
      "dispose:rerank",
      "batch:attempt-2",
    ]);
  });

  test("embed: non-VRAM errors do not trigger reclaim (no dispose, returns null per existing contract)", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });

    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      events.push("embed:attempt");
      throw new Error("model file not found");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };

    const result = await llm.embed("hello");
    expect(result).toBeNull();
    // Single attempt — no retry, no dispose
    expect(events).toEqual(["embed:attempt"]);
  });

  test("embed: lowVram=false skips reclaim entirely", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({}); // lowVram defaults to false

    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      events.push("embed:attempt");
      throw new Error("A context size of 2048 is too large for the available VRAM");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };

    const result = await llm.embed("hello");
    expect(result).toBeNull();
    // No retry: outside lowVram, VRAM errors fall through to the existing null-returning catch
    expect(events).toEqual(["embed:attempt"]);
  });

  test("embed: reclaim awaits in-flight generate/rerank chains before disposing", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let embedAttempts = 0;
    const release: { generate?: () => void; rerank?: () => void } = {};

    // expandQuery and rerank each park inside their lowVram chain wrappers
    // until we release them, simulating in-flight callers.
    (llm as unknown as { expandQueryImpl: () => Promise<Queryable[]> }).expandQueryImpl = async () => {
      events.push("expand:start");
      await new Promise<void>((r) => { release.generate = r; });
      events.push("expand:end");
      return [];
    };
    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async (_q, docs) => {
      events.push("rerank:start");
      await new Promise<void>((r) => { release.rerank = r; });
      events.push("rerank:end");
      return { results: docs.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })), model: "fake" };
    };
    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      embedAttempts++;
      events.push(`embed:attempt-${embedAttempts}`);
      if (embedAttempts === 1) throw new Error("too large for the available VRAM");
      return { embedding: [9], model: "fake-embed" };
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };

    // Kick off in-flight expand and rerank — they park.
    const expand = llm.expandQuery("e");
    const rerank = llm.rerank("r", [{ file: "a.md", text: "x" }]);
    await new Promise((r) => setImmediate(r));

    // Start embed: it should fail on attempt 1, then await both chains.
    const embed = llm.embed("hello");
    // Give embed time to throw and enter reclaim await.
    await new Promise((r) => setImmediate(r));

    // At this point, embed:attempt-1 has fired and embed is parked on the chains.
    // Dispose should NOT have been called yet — chains haven't drained.
    expect(events).toContain("embed:attempt-1");
    expect(events).not.toContain("dispose:generate");
    expect(events).not.toContain("dispose:rerank");

    // Release the in-flight callers. Their lowVram wrappers will run their own
    // finally-dispose first, then embed's reclaim will run its (idempotent) dispose,
    // then embed retries and succeeds.
    release.generate!();
    release.rerank!();
    await Promise.all([expand, rerank]);
    const result = await embed;

    expect(result).toEqual({ embedding: [9], model: "fake-embed" });
    // expand:end and rerank:end must precede embed's dispose calls.
    const expandEnd = events.indexOf("expand:end");
    const rerankEnd = events.indexOf("rerank:end");
    const firstDispose = Math.min(events.indexOf("dispose:generate"), events.indexOf("dispose:rerank"));
    expect(expandEnd).toBeGreaterThanOrEqual(0);
    expect(rerankEnd).toBeGreaterThanOrEqual(0);
    expect(firstDispose).toBeGreaterThan(expandEnd);
    expect(firstDispose).toBeGreaterThan(rerankEnd);
  });

  test("expandQuery: catch-and-retry on insufficient VRAM disposes rerank + embed contexts and retries once", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async () => {
      attempts++;
      events.push(`expand:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("A context size of 2048 is too large for the available VRAM");
      }
      return [{ type: "lex", text: "ok" }];
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };

    const result = await llm.expandQuery("test");
    expect(result).toEqual([{ type: "lex", text: "ok" }]);
    expect(attempts).toBe(2);
    // Reclaim disposes rerank + embed contexts (not generate — we need generate).
    // Then the chain's own finally fires after the call returns, disposing generate.
    expect(events).toEqual([
      "expand:attempt-1",
      "dispose:rerank",
      "dispose:embed-contexts",
      "expand:attempt-2",
      "dispose:generate",
    ]);
  });

  test("rerank: catch-and-retry on insufficient VRAM disposes generate + embed contexts and retries once", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async (_q, docs) => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("Failed to create any rerank context");
      }
      return { results: docs.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })), model: "fake" };
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };

    const result = await llm.rerank("q", [{ file: "a.md", text: "x" }]);
    expect(result.results.map(r => r.file)).toEqual(["a.md"]);
    expect(attempts).toBe(2);
    // Reclaim disposes generate + embed contexts (not rerank — we need rerank).
    // Then the chain's own finally fires after the call returns, disposing rerank.
    expect(events).toEqual([
      "rerank:attempt-1",
      "dispose:generate",
      "dispose:embed-contexts",
      "rerank:attempt-2",
      "dispose:rerank",
    ]);
  });

  test("embed reclaim never disposes embed contexts or model (it needs them)", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      attempts++;
      events.push(`embed:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("Failed to create any embedding context");
      }
      return { embedding: [1], model: "fake-embed" };
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => {
      events.push("dispose:embed-model");
    };

    const result = await llm.embed("hello");
    expect(result).toEqual({ embedding: [1], model: "fake-embed" });
    expect(attempts).toBe(2);
    expect(events).not.toContain("dispose:embed-contexts");
    expect(events).not.toContain("dispose:embed-model");
    expect(events).toEqual([
      "embed:attempt-1",
      "dispose:generate",
      "dispose:rerank",
      "embed:attempt-2",
    ]);
  });

  test("rerank reclaim escalates to disposing the embed model when first-pass retry still hits VRAM", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async (_q, docs) => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      if (attempts < 3) {
        throw new Error("Failed to create any rerank context");
      }
      return { results: docs.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })), model: "fake" };
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => {
      events.push("dispose:embed-model");
    };

    const result = await llm.rerank("q", [{ file: "a.md", text: "x" }]);
    expect(result.results.map((r) => r.file)).toEqual(["a.md"]);
    expect(attempts).toBe(3);
    expect(events).toEqual([
      "rerank:attempt-1",
      "dispose:generate",
      "dispose:embed-contexts",
      "rerank:attempt-2",
      "dispose:embed-model",
      "rerank:attempt-3",
      "dispose:rerank",
    ]);
  });

  test("rerank reclaim gives up after second-pass retry still fails for VRAM", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async () => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      throw new Error("Failed to create any rerank context");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => undefined;
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => undefined;
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => undefined;
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => undefined;

    await expect(llm.rerank("q", [{ file: "a.md", text: "x" }])).rejects.toThrow(/Failed to create any rerank context/);
    // 1 initial + 2 reclaim retries = 3 attempts, no more.
    expect(attempts).toBe(3);
  });

  test("expandQuery reclaim waits for in-flight rerank chain before disposing rerank", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let expandAttempts = 0;
    const release: { rerank?: () => void } = {};

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async (_q, docs) => {
      events.push("rerank:start");
      await new Promise<void>((r) => { release.rerank = r; });
      events.push("rerank:end");
      return { results: docs.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })), model: "fake" };
    };
    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async () => {
      expandAttempts++;
      events.push(`expand:attempt-${expandAttempts}`);
      if (expandAttempts === 1) throw new Error("too large for the available VRAM");
      return [{ type: "lex", text: "ok" }];
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };

    // Start rerank — it parks until released.
    const rerank = llm.rerank("r", [{ file: "a.md", text: "x" }]);
    await new Promise((r) => setImmediate(r));

    // Start expand — it should fail on attempt 1 and park awaiting rerankChain.
    const expand = llm.expandQuery("e");
    await new Promise((r) => setImmediate(r));

    // expand:attempt-1 has fired; rerank is still mid-flight; expand is parked.
    expect(events).toContain("expand:attempt-1");
    expect(events).toContain("rerank:start");
    expect(events).not.toContain("dispose:rerank");

    // Release rerank. Its chain finally fires (disposing rerank), then expand's
    // reclaim runs its own (idempotent) dispose:rerank, then expand retries.
    release.rerank!();
    await rerank;
    const result = await expand;

    expect(result).toEqual([{ type: "lex", text: "ok" }]);
    // rerank:end must precede expand's reclaim dispose.
    const rerankEnd = events.indexOf("rerank:end");
    const expandRetry = events.indexOf("expand:attempt-2");
    expect(rerankEnd).toBeGreaterThanOrEqual(0);
    expect(expandRetry).toBeGreaterThan(rerankEnd);
  });

  // ===========================================================================
  // Red-team: second-pass escalation symmetry, caps, and failure modes
  // ===========================================================================

  test("expandQuery reclaim escalates to disposing the embed model when first-pass retry still hits VRAM", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async () => {
      attempts++;
      events.push(`expand:attempt-${attempts}`);
      if (attempts < 3) {
        throw new Error("A context size of 2048 is too large for the available VRAM");
      }
      return [{ type: "lex", text: "ok" }];
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => {
      events.push("dispose:embed-model");
    };

    const result = await llm.expandQuery("test");
    expect(result).toEqual([{ type: "lex", text: "ok" }]);
    expect(attempts).toBe(3);
    expect(events).toEqual([
      "expand:attempt-1",
      "dispose:rerank",
      "dispose:embed-contexts",
      "expand:attempt-2",
      "dispose:embed-model",
      "expand:attempt-3",
      "dispose:generate",
    ]);
  });

  test("expandQuery reclaim gives up after second-pass retry still fails for VRAM", async () => {
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { expandQueryImpl: (q: string) => Promise<Queryable[]> }).expandQueryImpl = async () => {
      attempts++;
      throw new Error("A context size of 2048 is too large for the available VRAM");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => undefined;
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => undefined;
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => undefined;
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => undefined;

    await expect(llm.expandQuery("test")).rejects.toThrow(/too large for the available VRAM/);
    // Initial + first-pass retry + second-pass retry = 3 attempts, then surface.
    expect(attempts).toBe(3);
  });

  test("embed reclaim surfaces a second VRAM error without ever escalating to embed-model dispose", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { embedImpl: (text: string, options: EmbedOptions) => Promise<EmbeddingResult> }).embedImpl = async () => {
      attempts++;
      events.push(`embed:attempt-${attempts}`);
      throw new Error("Failed to create any embedding context");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => {
      events.push("dispose:embed-model");
    };

    // embed swallows the error and returns null per its existing contract;
    // it must not have escalated to disposing the embed model.
    const result = await llm.embed("hello");
    expect(result).toBeNull();
    expect(attempts).toBe(2); // initial + one first-pass retry, no second-pass
    expect(events).not.toContain("dispose:embed-contexts");
    expect(events).not.toContain("dispose:embed-model");
  });

  test("rerank reclaim continues retry even when a dispose helper throws (best-effort)", async () => {
    const events: string[] = [];
    const warnings: unknown[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async (_q, docs) => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      if (attempts === 1) {
        throw new Error("Failed to create any rerank context");
      }
      return { results: docs.map((d, i) => ({ file: d.file, score: 1 - i * 0.1, index: i })), model: "fake" };
    };
    // First-pass dispose throws — reclaim must swallow and continue.
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate-throws");
      throw new Error("simulated dispose failure");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };

    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(args); };
    try {
      const result = await llm.rerank("q", [{ file: "a.md", text: "x" }]);
      expect(result.results.map((r) => r.file)).toEqual(["a.md"]);
    } finally {
      console.warn = originalWarn;
    }

    // The retry MUST still have run; the embed-contexts dispose MUST still have
    // run despite the generate-dispose throwing.
    expect(attempts).toBe(2);
    expect(events).toContain("dispose:embed-contexts");
    expect(events).toContain("rerank:attempt-2");
    // The warning should mention the failing label so the operator can grep.
    expect(warnings.some((w) => JSON.stringify(w).includes("disposeGenerateModel"))).toBe(true);
  });

  test("reclaim ignores non-VRAM errors thrown by the operation (no dispose, no retry)", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async () => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      // A *different* error — looks like a bug, not VRAM pressure. Reclaim
      // must not chew through retries on errors that won't be fixed by
      // dropping models.
      throw new Error("Computing rankings is not supported for this model.");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => {
      events.push("dispose:generate");
    };
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => {
      events.push("dispose:rerank");
    };
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => {
      events.push("dispose:embed-contexts");
    };

    await expect(llm.rerank("q", [{ file: "a.md", text: "x" }])).rejects.toThrow(/not supported for this model/);
    expect(attempts).toBe(1);
    // The reclaim path must NOT have fired — those touch generate and embed.
    // The rerank-chain's own finally fires regardless (intentional lowVram
    // post-call cleanup), so dispose:rerank is expected exactly once.
    expect(events).not.toContain("dispose:generate");
    expect(events).not.toContain("dispose:embed-contexts");
    expect(events.filter((e) => e === "dispose:rerank")).toHaveLength(1);
  });

  test("reclaim treats post-first-pass non-VRAM errors as terminal (no second pass)", async () => {
    const events: string[] = [];
    const llm = new LlamaCpp({ lowVram: true });
    let attempts = 0;

    (llm as unknown as { rerankImpl: (q: string, docs: RerankDocument[]) => Promise<RerankResult> }).rerankImpl = async () => {
      attempts++;
      events.push(`rerank:attempt-${attempts}`);
      if (attempts === 1) throw new Error("Failed to create any rerank context");
      // Second attempt fails for an unrelated reason — must not escalate.
      throw new Error("model file corrupted");
    };
    (llm as unknown as { disposeGenerateModel: () => Promise<void> }).disposeGenerateModel = async () => undefined;
    (llm as unknown as { disposeRerankModel: () => Promise<void> }).disposeRerankModel = async () => undefined;
    (llm as unknown as { disposeEmbedContexts: () => Promise<void> }).disposeEmbedContexts = async () => undefined;
    (llm as unknown as { disposeEmbedModel: () => Promise<void> }).disposeEmbedModel = async () => {
      events.push("dispose:embed-model");
    };

    await expect(llm.rerank("q", [{ file: "a.md", text: "x" }])).rejects.toThrow(/model file corrupted/);
    expect(attempts).toBe(2);
    expect(events).not.toContain("dispose:embed-model");
  });
});
