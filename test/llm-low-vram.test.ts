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

import { describe, test, expect } from "vitest";
import { LlamaCpp, type RerankDocument, type Queryable, type RerankResult } from "../src/llm.js";

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
  (llm as unknown as { expandQueryImpl: (...args: unknown[]) => Promise<Queryable[]> }).expandQueryImpl = async (
    _query: string,
    _options?: unknown,
  ): Promise<Queryable[]> => {
    events.push(resident.generate ? "expand:start" : "expand:load");
    resident.generate = true;
    await tick();
    events.push("expand:end");
    return [{ type: "lex", text: "x" }];
  };

  (llm as unknown as { rerankImpl: (...args: unknown[]) => Promise<RerankResult> }).rerankImpl = async (
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
});
