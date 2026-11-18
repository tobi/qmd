/**
 * llm-lifecycle.test.ts - Regression tests for the LLM resource lifecycle.
 *
 * The two race paths these cover were reproduced against the real LlamaCpp
 * methods with a controlled asynchronous context:
 *
 * 1. A session opened with withLLMSessionForLlm was invisible to the global
 *    canUnloadLLM() check, so the instance's inactivity timer disposed the
 *    context the session was using.
 * 2. A session admitted after the timer had started an async ctx.dispose()
 *    still saw the old context array and borrowed the context being disposed.
 *
 * Fake contexts/models and fake remote clients only: no GGUF files and no
 * network. The inactivity timer's decision is driven through
 * `handleInactivityTimeout()` — the exact callback the timer runs — and
 * readiness is observed by awaiting real events or by flushing the microtask
 * queue. No platform fake-timer API is used, so the file runs under both the
 * Bun and Vitest runners.
 */

import { describe, test, expect, afterEach } from "vitest";
import {
  LlamaCpp,
  withLLMSession,
  withLLMSessionForLlm,
  setDefaultLlamaCpp,
} from "../src/llm.js";

type FakeEmbedContext = {
  getEmbeddingFor: (text: string) => Promise<{ vector: Float32Array }>;
  dispose: () => Promise<void>;
};

type LlmInternals = {
  embedContexts: FakeEmbedContext[];
  embedModel: unknown;
  remoteEmbedder: unknown;
  jevReranker: unknown;
  touchActivity: () => void;
  handleInactivityTimeout: () => void;
  computeParallelism: () => Promise<number>;
  threadsPerContext: () => Promise<number>;
};

/**
 * Runtime check for the in-process test seam: the private lifecycle fields and
 * the methods these tests drive. LlamaCpp keeps them private, so the shape is
 * verified here instead of asserted; a rename in src/llm.ts then fails this
 * check with a named error instead of silently writing a property nothing reads.
 */
function hasLlmTestSeam(value: unknown): value is LlmInternals {
  if (typeof value !== "object" || value === null) return false;
  return (
    "embedContexts" in value &&
    Array.isArray(value.embedContexts) &&
    "embedModel" in value &&
    "remoteEmbedder" in value &&
    "jevReranker" in value &&
    "touchActivity" in value &&
    typeof value.touchActivity === "function" &&
    "handleInactivityTimeout" in value &&
    typeof value.handleInactivityTimeout === "function" &&
    "computeParallelism" in value &&
    typeof value.computeParallelism === "function" &&
    "threadsPerContext" in value &&
    typeof value.threadsPerContext === "function"
  );
}

function internals(llm: LlamaCpp): LlmInternals {
  if (!hasLlmTestSeam(llm)) {
    throw new Error(
      "LlamaCpp test seam is missing: expected embedContexts/embedModel/remoteEmbedder/jevReranker fields and touchActivity/handleInactivityTimeout/computeParallelism/threadsPerContext methods (see src/llm.ts)",
    );
  }
  return llm;
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  return Promise.withResolvers<void>();
}

/**
 * Drain the microtask queue a few turns. Everything the lifecycle parks on is
 * promise-based, so nothing can resolve here unless the awaited event happened.
 */
async function flushMicrotasks(turns = 32): Promise<void> {
  for (let i = 0; i < turns; i++) await Promise.resolve();
}

/** Embed model stub: enough of the API for embed() and pool creation. */
function fakeEmbedModel(createContext: () => FakeEmbedContext) {
  return {
    trainContextSize: 2048,
    tokenize: (text: string) => Array.from({ length: text.length }, () => 1),
    detokenize: (tokens: readonly number[]) => "x".repeat(tokens.length),
    createEmbeddingContext: async () => createContext(),
    dispose: async () => {},
  };
}

/** Pooled embedding context that traces borrows and disposals. */
function fakeContext(vector: number[], trace: string[], name: string): FakeEmbedContext {
  return {
    getEmbeddingFor: async () => {
      trace.push(`embed:${name}`);
      return { vector: new Float32Array(vector) };
    },
    dispose: async () => {
      trace.push(`dispose:${name}`);
    },
  };
}

/**
 * An instance whose pool holds a context whose dispose() parks, so tests can
 * observe the "unloading" window deterministically.
 */
function createUnloadingInstance() {
  const trace: string[] = [];
  const disposalStarted = deferred();
  const finishDisposal = deferred();

  const staleContext: FakeEmbedContext = {
    getEmbeddingFor: async () => {
      trace.push("embed:stale");
      throw new Error("borrowed a context that was being disposed");
    },
    dispose: async () => {
      disposalStarted.resolve();
      await finishDisposal.promise;
      trace.push("dispose:stale");
    },
  };
  const freshContext: FakeEmbedContext = fakeContext([7], trace, "fresh");

  const llm = new LlamaCpp({ inactivityTimeoutMs: 10 });
  const state = internals(llm);
  state.embedContexts = [staleContext];
  state.embedModel = fakeEmbedModel(() => freshContext);
  state.computeParallelism = async () => 1;
  state.threadsPerContext = async () => 0;

  return { llm, trace, disposalStarted, finishDisposal };
}

afterEach(() => {
  setDefaultLlamaCpp(null);
});

describe("LlamaCpp resource lifecycle", () => {
  test("a withLLMSessionForLlm session blocks that instance's own idle unload", async () => {
    const trace: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [fakeContext([1], trace, "pool")];

    const opened = deferred();
    const close = deferred();
    const run = withLLMSessionForLlm(llm, async () => {
      opened.resolve();
      await close.promise;
    });

    await opened.promise;
    // Run the exact decision the inactivity timer runs: it must consult THIS
    // instance's lifecycle, which holds the lease for the per-instance session.
    internals(llm).handleInactivityTimeout();
    await flushMicrotasks();
    expect(trace).toEqual([]);
    expect(llm.lifecycleState).toBe("ready");

    close.resolve();
    await run;

    // Released: the same callback now unloads, without any further activity.
    internals(llm).handleInactivityTimeout();
    await flushMicrotasks();
    expect(trace).toEqual(["dispose:pool"]);
    expect(llm.lifecycleState).toBe("ready");
    await llm.dispose();
  });

  test("a session admitted while idle disposal is running waits for it and borrows a fresh context", async () => {
    const { llm, trace, disposalStarted, finishDisposal } = createUnloadingInstance();
    setDefaultLlamaCpp(llm);

    // Arm the instance's real inactivity timer. The disposal signal is the
    // event that proves the timer fired and the unload reached ctx.dispose().
    internals(llm).touchActivity();
    await disposalStarted.promise;
    expect(llm.lifecycleState).toBe("unloading");

    let settled = false;
    const run = withLLMSession(async (session) => session.embed("after idle unload"));
    void run.then(
      () => { settled = true; },
      () => { settled = true; },
    );

    await flushMicrotasks();
    // Admission is parked on the unload completion: no context is borrowed
    // while disposal is in progress.
    expect(settled).toBe(false);
    expect(trace).toEqual([]);

    // The gate is the actual disposal promise, not a timeout: further progress
    // of the microtask queue must not reopen admission.
    await flushMicrotasks();
    expect(settled).toBe(false);
    expect(trace).toEqual([]);
    expect(llm.lifecycleState).toBe("unloading");

    finishDisposal.resolve();
    const result = await run;

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([7]);
    expect(trace).toEqual(["dispose:stale", "embed:fresh"]);
    await llm.dispose();
  });

  test("a direct LlamaCpp operation admitted during idle disposal waits for it", async () => {
    const { llm, trace, disposalStarted, finishDisposal } = createUnloadingInstance();

    const unload = llm.unloadIdleResources();
    await disposalStarted.promise;

    let settled = false;
    const run = llm.embed("direct call during unload");
    void run.then(
      () => { settled = true; },
      () => { settled = true; },
    );

    await flushMicrotasks();
    expect(settled).toBe(false);
    expect(trace).toEqual([]);

    finishDisposal.resolve();
    const result = await run;
    await unload;

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([7]);
    expect(trace).toEqual(["dispose:stale", "embed:fresh"]);
    await llm.dispose();
  });

  test("remote embedding holds an operation lease until its request settles", async () => {
    const requestStarted = deferred();
    const finishRequest = deferred();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).remoteEmbedder = {
      modelName: "remote-openai:test",
      embedBatch: async () => {
        requestStarted.resolve();
        await finishRequest.promise;
        return [[0.25, 0.75]];
      },
    };

    let embedDone = false;
    const embed = llm.embed("remote text").then((result) => {
      embedDone = true;
      return result;
    });
    await requestStarted.promise;

    let disposalDone = false;
    const disposal = llm.dispose();
    void disposal.then(() => { disposalDone = true; });

    await flushMicrotasks();
    // The HTTP request is still pending, so teardown (and a store close that
    // awaited it) must not proceed past the search's continuation.
    expect(llm.lifecycleState).toBe("disposing");
    expect(disposalDone).toBe(false);
    expect(embedDone).toBe(false);

    finishRequest.resolve();
    const result = await embed;
    await disposal;

    expect(result!.embedding).toEqual([0.25, 0.75]);
    expect(llm.lifecycleState).toBe("disposed");
  });

  test("remote TypeSafe rerank holds an operation lease until its request settles", async () => {
    const requestStarted = deferred();
    const finishRequest = deferred();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).jevReranker = {
      modelName: "remote-typesafe:test",
      rerank: async () => {
        requestStarted.resolve();
        await finishRequest.promise;
        return { results: [{ file: "a.md", score: 0.9, index: 0 }], model: "remote-typesafe:test" };
      },
    };

    let rerankDone = false;
    const rerank = llm.rerank("query", [{ file: "a.md", text: "doc" }]).then((result) => {
      rerankDone = true;
      return result;
    });
    await requestStarted.promise;

    let disposalDone = false;
    const disposal = llm.dispose();
    void disposal.then(() => { disposalDone = true; });

    await flushMicrotasks();
    expect(llm.lifecycleState).toBe("disposing");
    expect(disposalDone).toBe(false);
    expect(rerankDone).toBe(false);

    finishRequest.resolve();
    const result = await rerank;
    await disposal;

    expect(result.results).toEqual([{ file: "a.md", score: 0.9, index: 0 }]);
    expect(llm.lifecycleState).toBe("disposed");
  });

  test("dispose waits for an in-flight operation before tearing down contexts", async () => {
    const trace: string[] = [];
    const embeddingStarted = deferred();
    const finishEmbedding = deferred();

    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [
      {
        getEmbeddingFor: async () => {
          embeddingStarted.resolve();
          await finishEmbedding.promise;
          trace.push("embed:done");
          return { vector: new Float32Array([3]) };
        },
        dispose: async () => {
          trace.push("dispose:pool");
        },
      },
    ];
    internals(llm).embedModel = fakeEmbedModel(() => fakeContext([3], trace, "unused"));

    const operation = llm.embed("in flight");
    await embeddingStarted.promise;

    let disposalDone = false;
    const disposal = llm.dispose();
    void disposal.then(() => { disposalDone = true; });

    await flushMicrotasks();
    // Terminal admission is already in effect, but the borrowed context is not
    // destroyed under the operation: teardown waits for actual completion.
    expect(llm.lifecycleState).toBe("disposing");
    expect(disposalDone).toBe(false);
    expect(trace).toEqual([]);

    finishEmbedding.resolve();
    const result = await operation;
    expect(result!.embedding).toEqual([3]);

    await disposal;
    expect(trace).toEqual(["embed:done", "dispose:pool"]);
    expect(llm.lifecycleState).toBe("disposed");
  });

  test("concurrent dispose callers await the same teardown completion", async () => {
    const trace: string[] = [];
    const embeddingStarted = deferred();
    const finishEmbedding = deferred();

    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [
      {
        getEmbeddingFor: async () => {
          embeddingStarted.resolve();
          await finishEmbedding.promise;
          trace.push("embed:done");
          return { vector: new Float32Array([4]) };
        },
        dispose: async () => {
          trace.push("dispose:pool");
        },
      },
    ];
    internals(llm).embedModel = fakeEmbedModel(() => fakeContext([4], trace, "unused"));

    const operation = llm.embed("in flight");
    await embeddingStarted.promise;

    const settled = { first: false, second: false };
    const first = llm.dispose().then(() => { settled.first = true; });
    const second = llm.dispose().then(() => { settled.second = true; });

    await flushMicrotasks();
    // Neither caller may resolve while the first teardown is still draining the
    // operation: a second close() (createStore().close()) must not proceed —
    // closing SQLite, say — underneath an in-flight search.
    expect(llm.lifecycleState).toBe("disposing");
    expect(settled).toEqual({ first: false, second: false });
    expect(trace).toEqual([]);

    finishEmbedding.resolve();
    await operation;
    await Promise.all([first, second]);

    expect(settled).toEqual({ first: true, second: true });
    expect(trace).toEqual(["embed:done", "dispose:pool"]);
    expect(llm.lifecycleState).toBe("disposed");

    // Calls after "disposed" resolve immediately.
    await expect(llm.dispose()).resolves.toBeUndefined();
  });

  test("a session callback may dispose the instance after its awaited operation", async () => {
    const trace: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [fakeContext([5], trace, "pool")];
    internals(llm).embedModel = fakeEmbedModel(() => fakeContext([5], trace, "fresh"));

    const result = await withLLMSessionForLlm(llm, async (session) => {
      const embedding = await session.embed("before close");
      expect(embedding!.embedding).toEqual([5]);

      // Must not wait for this session's own lease, or the callback deadlocks.
      await llm.dispose();
      return embedding;
    });

    expect(result!.embedding).toEqual([5]);
    expect(llm.lifecycleState).toBe("disposed");
    expect(trace).toEqual(["embed:pool", "dispose:pool"]);
  });

  test("nested sessions keep the instance unloadable only after the outermost release", async () => {
    const trace: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [fakeContext([1], trace, "pool")];

    await withLLMSessionForLlm(llm, async () => {
      await withLLMSessionForLlm(llm, async () => {
        await llm.unloadIdleResources();
        expect(trace).toEqual([]);
      });
      // Inner session released, outer lease still held.
      await llm.unloadIdleResources();
      expect(trace).toEqual([]);
    });

    await llm.unloadIdleResources();
    expect(trace).toEqual(["dispose:pool"]);
    expect(llm.lifecycleState).toBe("ready");
    await llm.dispose();
  });

  test("dispose during an idle unload completes and rejects new work", async () => {
    const disposalStarted = deferred();
    const finishDisposal = deferred();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    internals(llm).embedContexts = [
      {
        getEmbeddingFor: async () => ({ vector: new Float32Array([1]) }),
        dispose: async () => {
          disposalStarted.resolve();
          await finishDisposal.promise;
        },
      },
    ];

    const unload = llm.unloadIdleResources();
    await disposalStarted.promise;

    let disposalDone = false;
    const disposal = llm.dispose();
    void disposal.then(() => { disposalDone = true; });
    expect(llm.lifecycleState).toBe("disposing");

    // Terminal admission: new sessions and direct operations fail fast instead
    // of parking on a completion that belongs to a dying instance.
    await expect(withLLMSessionForLlm(llm, async () => "never")).rejects.toThrow(
      "LLM has been disposed",
    );
    await expect(llm.embed("after dispose")).rejects.toThrow("LLM has been disposed");

    // Teardown must not proceed while the idle context disposal is still
    // running: dispose waits for that promise, not for a timer.
    await flushMicrotasks();
    expect(disposalDone).toBe(false);

    finishDisposal.resolve();
    await unload;
    await disposal;

    expect(llm.lifecycleState).toBe("disposed");
  });
});
