import { describe, expect, test } from "vitest";
import { LlamaCpp, SessionReleasedError, setNodeLlamaCppModuleForTest, withLLMSessionForLlm } from "../src/llm.ts";

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

describe("LLM shutdown admission and abort", () => {
  test("an existing session receives the manager abort broadcast", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const aborted = deferred();

    const running = withLLMSessionForLlm(llm, async (session) => {
      session.signal.addEventListener("abort", () => aborted.resolve(), { once: true });
      started.resolve();
      await aborted.promise;
    });

    await started.promise;
    llm.requestSessionAbort(new Error("stop"));
    await running;
  });

  test("a session acquired after permanent admission closure rejects", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    llm.closeSessionAdmission();
    await expect(withLLMSessionForLlm(llm, async () => "ok")).rejects.toThrow(SessionReleasedError);
  });

  test("a direct embed call after shutdown begins rejects instead of returning null", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    llm.closeSessionAdmission();
    await expect(llm.embed("hello")).rejects.toThrow(SessionReleasedError);
  });

  test("a direct expandQuery call after shutdown begins rejects instead of falling back", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    llm.closeSessionAdmission();
    await expect(llm.expandQuery("hello")).rejects.toThrow(SessionReleasedError);
  });

  test("an AbortError from embed is not converted into a null fallback", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as unknown as Record<string, unknown>, {
      ensureEmbedContext: async () => {
        throw Object.assign(new Error("aborted"), { name: "AbortError" });
      },
    });
    await expect(llm.embed("hello")).rejects.toMatchObject({ name: "AbortError" });
  });

  test("an AbortError from expandQuery is not converted into fallback query text", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as unknown as Record<string, unknown>, {
      _ciMode: false,
      ensureLlama: async () => {
        throw Object.assign(new Error("aborted"), { name: "AbortError" });
      },
    });
    await expect(llm.expandQuery("hello")).rejects.toMatchObject({ name: "AbortError" });
  });

  test("a non-shutdown expandQuery failure still falls back to the original query", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as unknown as Record<string, unknown>, {
      _ciMode: false,
      ensureLlama: async () => ({ createGrammar: async () => { throw new Error("oom"); } }),
      ensureGenerateModel: async () => undefined,
      generateModel: { createContext: async () => { throw new Error("oom"); } },
    });
    const result = await llm.expandQuery("hello");
    expect(result.some((q) => q.text === "hello")).toBe(true);
  });

  test("external abort listeners are removed when the session is released", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const external = new AbortController();
    let adds = 0;
    let removes = 0;
    const signal = external.signal;
    const add = signal.addEventListener.bind(signal);
    const rem = signal.removeEventListener.bind(signal);
    signal.addEventListener = ((...args: Parameters<AbortSignal["addEventListener"]>) => {
      adds += 1;
      return add(...args);
    }) as AbortSignal["addEventListener"];
    signal.removeEventListener = ((...args: Parameters<AbortSignal["removeEventListener"]>) => {
      removes += 1;
      return rem(...args);
    }) as AbortSignal["removeEventListener"];

    await withLLMSessionForLlm(llm, async () => undefined, { signal });
    expect(adds).toBe(1);
    expect(removes).toBe(1);
  });

  test("idle unload cannot begin concurrently with a newly acquired session", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    Object.assign(llm as unknown as Record<string, unknown>, {
      embedContexts: [{
        dispose: async () => {
          started.resolve();
          await gate.promise;
        },
      }],
    });

    const unloading = llm.unloadIdleResources();
    await started.promise;

    let sessionBegan = false;
    const session = withLLMSessionForLlm(llm, async () => {
      sessionBegan = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(sessionBegan).toBe(false);

    gate.resolve();
    await unloading;
    await session;
    expect(sessionBegan).toBe(true);
  });

  test("concurrent dispose() still returns the same promise", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const first = llm.dispose();
    const second = llm.dispose();
    expect(first).toBe(second);
    await Promise.all([first, second]);
  });

  test("active tokenization blocks disposal and rejects after admission close", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    Object.assign(llm as unknown as Record<string, unknown>, {
      ensureEmbedContext: async () => {
        started.resolve();
        await gate.promise;
      },
      embedModel: {
        tokenize: () => [1, 2, 3],
        detokenize: () => "hi",
        dispose: async () => { calls.push("embed-model"); },
      },
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    const tokenizing = llm.tokenize("hello");
    await started.promise;
    const disposing = llm.dispose();
    await Promise.resolve();
    expect(calls).toEqual([]);

    llm.closeSessionAdmission();
    await expect(llm.detokenize([1] as never)).rejects.toThrow(SessionReleasedError);
    await expect(llm.countTokens("later")).rejects.toThrow(SessionReleasedError);

    gate.resolve();
    await tokenizing;
    await disposing;
    expect(calls[0]).toBe("embed-model");
    expect(calls).toContain("llama");
  });
});

describe("LLM phase shutdown", () => {
  test("expansion observes abort and awaits sequence-before-context cleanup", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    const seenSignals: AbortSignal[] = [];
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: async () => "unused",
      getLlama: async () => ({ gpu: false, dispose: async () => {} } as never),
      LlamaChatSession: class {
        async prompt(_prompt: string, options?: Record<string, unknown>) {
          const signal = options?.signal as AbortSignal | undefined;
          if (signal) seenSignals.push(signal);
          started.resolve();
          await gate.promise;
          if (signal?.aborted) {
            throw Object.assign(new Error("aborted"), { name: "AbortError" });
          }
          return "lex: q\n";
        }
      },
    });
    Object.assign(llm as unknown as Record<string, unknown>, {
      _ciMode: false,
      ensureLlama: async () => ({ createGrammar: async () => ({}) }),
      ensureGenerateModel: async () => undefined,
      generateModel: {
        createContext: async () => ({
          getSequence: () => ({ dispose: async () => { calls.push("seq"); } }),
          dispose: async () => { calls.push("ctx"); },
        }),
        dispose: async () => { calls.push("model"); },
      },
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    try {
      const running = llm.expandQuery("q");
      await started.promise;
      llm.requestSessionAbort(new Error("Shutdown requested by SIGINT"));
      const disposing = llm.dispose();
      await Promise.resolve();
      expect(calls).toEqual([]);
      gate.resolve();
      await expect(running).rejects.toMatchObject({ name: "AbortError" });
      await disposing;
      expect(seenSignals.length).toBeGreaterThan(0);
      expect(seenSignals[0]?.aborted).toBe(true);
      expect(calls).toEqual(["seq", "ctx", "model", "llama"]);
    } finally {
      setNodeLlamaCppModuleForTest(null);
    }
  });

  test("real embed() returns from a blocked native call before disposal starts", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    Object.assign(llm as unknown as Record<string, unknown>, {
      ensureEmbedContext: async () => ({
        getEmbeddingFor: async () => {
          started.resolve();
          await gate.promise;
          return { vector: [0.1] };
        },
      }),
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    const running = llm.embed("hello");
    await started.promise;
    const disposing = llm.dispose();
    await Promise.resolve();
    expect(calls).toEqual([]);
    gate.resolve();
    await running;
    await disposing;
    expect(calls).toEqual(["llama"]);
  });

  test("real rerank() returns before context/model/runtime disposal starts", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    const ctx = {
      rankAll: async () => {
        started.resolve();
        await gate.promise;
        return [0.9];
      },
      dispose: async () => { calls.push("context"); },
    };
    Object.assign(llm as unknown as Record<string, unknown>, {
      _ciMode: false,
      rerankContexts: [ctx],
      ensureRerankContexts: async () => [ctx],
      ensureRerankModel: async () => ({
        tokenize: (text: string) => Array.from(text, (_, i) => i),
        detokenize: () => "x",
      }),
      rerankModel: { dispose: async () => { calls.push("model"); } },
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    const running = llm.rerank("q", [{ file: "a", text: "b" }]);
    await started.promise;
    const disposing = llm.dispose();
    await Promise.resolve();
    expect(calls).toEqual([]);
    gate.resolve();
    await running;
    await disposing;
    expect(calls).toEqual(["context", "model", "llama"]);
  });
});
