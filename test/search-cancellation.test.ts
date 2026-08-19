/**
 * Cooperative search cancellation.
 *
 * The shutdown signal these tests drive is the one PR 1 already publishes
 * (`LlamaCpp.sessionAbortSignal` / `requestSessionAbort`). Search adds no
 * cancellation machinery of its own — it only observes that signal at the
 * boundaries before expensive native or sqlite-vec work.
 */

import { describe, test, expect, vi, afterEach } from "vitest";
import { mkdtemp, unlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  hybridQuery,
  structuredSearch,
  vectorSearchQuery,
  type Store,
  type SearchResult,
} from "../src/store.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../src/llm.js";
import { createShutdownCoordinator } from "../src/shutdown.js";
import { createInflightGate, registerStdioEofShutdown } from "../src/mcp/server.js";
import { EventEmitter } from "node:events";

class FakeStdin extends EventEmitter {
  readableEnded = false;
  destroyed = false;
}

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

const MODEL = "hf:test/embed/model.gguf";
const SHUTDOWN = /Shutdown requested by stdin-eof/;

const vecHit: SearchResult = {
  filepath: "qmd://docs/a.md",
  displayPath: "docs/a.md",
  title: "Alpha",
  body: "hello world authentication content for rerank",
  score: 0.9,
  docid: "abc123",
  source: "vec",
  context: null,
  hash: "abc123ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  collectionName: "docs",
  modifiedAt: "",
  bodyLength: 45,
};

async function createCancellationStore(): Promise<Store> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-search-cancel-"));
  const store = createStore(join(dir, "index.sqlite"));
  store.db.exec(`CREATE TABLE IF NOT EXISTS vectors_vec (hash_seq TEXT PRIMARY KEY, embedding BLOB)`);
  (store as Store & { _tmpdir?: string })._tmpdir = dir;
  return store;
}

async function cleanupCancellationStore(store: Store): Promise<void> {
  const dir = (store as Store & { _tmpdir?: string })._tmpdir;
  store.close();
  try { await unlink(store.dbPath); } catch { /* ignore */ }
  if (dir) await rm(dir, { recursive: true, force: true });
}

/** A store LLM stub exposing exactly the surface search reads. */
function stubLlm(signal: AbortSignal | undefined, embedBatch: (texts: string[]) => Promise<unknown>) {
  return {
    embedModelName: MODEL,
    generateModelName: "fake-generate-model",
    embedBatch,
    ...(signal ? { sessionAbortSignal: signal } : {}),
  } as Store["llm"];
}

const embedOk = async (texts: string[]) =>
  texts.map(() => ({ embedding: [1, 2, 3], model: MODEL }));

describe("hybridQuery cooperative cancellation", () => {
  test("an already-aborted session never reaches query expansion", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      controller.abort(new Error("Shutdown requested by stdin-eof"));

      const expandQuery = vi.fn(async () => [{ type: "vec", query: "expanded" }]);
      store.llm = stubLlm(controller.signal, embedOk);
      store.expandQuery = expandQuery as never;
      store.searchFTS = vi.fn(() => []) as never;
      store.searchVec = vi.fn(async () => [vecHit]) as never;
      store.rerank = vi.fn(async () => []) as never;

      await expect(hybridQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(expandQuery).not.toHaveBeenCalled();
      expect(store.searchVec).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("abort during expansion prevents embedding and every vector lookup", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const embedBatch = vi.fn(embedOk);
      const searchVec = vi.fn(async () => [vecHit]);

      store.llm = stubLlm(controller.signal, embedBatch);
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = (async () => {
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return [{ type: "vec", query: "expanded semantic" }];
      }) as never;
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => []) as never;

      await expect(hybridQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(embedBatch).not.toHaveBeenCalled();
      expect(searchVec).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("abort while awaiting embedBatch prevents every vector lookup and reranking", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const embedStarted = deferred();
      const releaseEmbed = deferred();
      const searchVec = vi.fn(async () => [vecHit]);
      const rerank = vi.fn(async () => [{ file: vecHit.filepath, score: 0.8 }]);

      store.llm = stubLlm(controller.signal, async (texts) => {
        embedStarted.resolve();
        await releaseEmbed.promise;
        return embedOk(texts);
      });
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => [{ type: "vec", query: "expanded semantic" }]) as never;
      store.searchVec = searchVec as never;
      store.rerank = rerank as never;

      const running = hybridQuery(store, "user query");
      await embedStarted.promise;
      controller.abort(new Error("Shutdown requested by stdin-eof"));
      releaseEmbed.resolve();

      await expect(running).rejects.toThrow(SHUTDOWN);
      expect(searchVec).not.toHaveBeenCalled();
      expect(rerank).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("shutdown mid-fanout stops the next vector lookup", async () => {
    const store = await createCancellationStore();
    try {
      const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
      llm.embedBatch = (async (texts: string[]) =>
        texts.map(() => ({ embedding: [1, 2, 3], model: llm.embedModelName }))
      ) as typeof llm.embedBatch;

      const searchVec = vi.fn(async () => {
        llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      });

      store.llm = llm;
      store.searchFTS = vi.fn(() => []) as never;
      // Original + two expansions = three lookups if nothing stops the fan-out.
      store.expandQuery = vi.fn(async () => [
        { type: "vec", query: "first expansion" },
        { type: "hyde", query: "second expansion document" },
      ]) as never;
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => []) as never;

      await expect(hybridQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(searchVec).toHaveBeenCalledTimes(1);
      await llm.dispose();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("abort after the last lookup prevents reranking", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const rerank = vi.fn(async () => [{ file: vecHit.filepath, score: 0.8 }]);

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => []) as never;
      store.searchVec = (async () => {
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      }) as never;
      store.rerank = rerank as never;

      await expect(hybridQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(rerank).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("rerank aborting mid-call rejects instead of blending its scores", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const rerank = vi.fn(async (_q: string, docs: { file: string; text: string }[]) => {
        // The reranker can be aborted and still resolve with scores.
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return docs.map((doc) => ({ file: doc.file, score: 0.8 }));
      });

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => []) as never;
      store.searchVec = vi.fn(async () => [vecHit]) as never;
      store.rerank = rerank as never;

      await expect(hybridQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(rerank).toHaveBeenCalledTimes(1);
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("skipRerank rejects instead of reporting an aborted search as a result set", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => []) as never;
      store.searchVec = (async () => {
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return [];
      }) as never;
      store.rerank = vi.fn(async () => []) as never;

      // An empty candidate set would otherwise short-circuit to a successful
      // `[]` — indistinguishable from "nothing matched".
      await expect(hybridQuery(store, "user query", { skipRerank: true })).rejects.toThrow(SHUTDOWN);
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("a live caller signal does not hide requestSessionAbort", async () => {
    const store = await createCancellationStore();
    try {
      const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
      const external = new AbortController();
      const embedStarted = deferred();
      const releaseEmbed = deferred();
      llm.embedBatch = (async (texts: string[]) => {
        embedStarted.resolve();
        await releaseEmbed.promise;
        return texts.map(() => ({ embedding: [1, 2, 3], model: llm.embedModelName }));
      }) as typeof llm.embedBatch;

      const searchVec = vi.fn(async () => [vecHit]);
      store.llm = llm;
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => []) as never;
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => []) as never;

      const running = hybridQuery(store, "user query", { signal: external.signal });
      await embedStarted.promise;
      llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));
      releaseEmbed.resolve();

      await expect(running).rejects.toThrow(SHUTDOWN);
      expect(external.signal.aborted).toBe(false);
      expect(searchVec).not.toHaveBeenCalled();
      await llm.dispose();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("the normal non-aborted path still embeds, looks up, and reranks", async () => {
    const store = await createCancellationStore();
    try {
      const embedBatch = vi.fn(embedOk);
      const searchVec = vi.fn(async () => [vecHit]);
      const rerank = vi.fn(async (_q: string, docs: { file: string; text: string }[]) =>
        docs.map((doc) => ({ file: doc.file, score: 0.75 })),
      );

      store.llm = stubLlm(undefined, embedBatch);
      store.searchFTS = vi.fn(() => []) as never;
      store.expandQuery = vi.fn(async () => [{ type: "vec", query: "expanded semantic" }]) as never;
      store.searchVec = searchVec as never;
      store.rerank = rerank as never;

      const results = await hybridQuery(store, "user query");

      expect(embedBatch).toHaveBeenCalledTimes(1);
      expect(searchVec).toHaveBeenCalledTimes(2);
      expect(rerank).toHaveBeenCalledTimes(1);
      expect(results[0]!.file).toBe(vecHit.filepath);
    } finally {
      await cleanupCancellationStore(store);
    }
  });
});

describe("structuredSearch cooperative cancellation", () => {
  test("abort while awaiting embedBatch prevents every searchVec call and reranking", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const embedStarted = deferred();
      const releaseEmbed = deferred();
      const searchVec = vi.fn(async () => [vecHit]);
      const rerank = vi.fn(async () => [{ file: vecHit.filepath, score: 0.8 }]);

      store.llm = stubLlm(controller.signal, async (texts) => {
        embedStarted.resolve();
        await releaseEmbed.promise;
        return embedOk(texts);
      });
      store.searchVec = searchVec as never;
      store.rerank = rerank as never;

      const running = structuredSearch(store, [
        { type: "vec", query: "first semantic query" },
        { type: "hyde", query: "second hypothetical document" },
      ], { signal: controller.signal });

      await embedStarted.promise;
      controller.abort(new Error("Shutdown requested by stdin-eof"));
      releaseEmbed.resolve();

      await expect(running).rejects.toThrow(SHUTDOWN);
      expect(searchVec).not.toHaveBeenCalled();
      expect(rerank).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("abort between vector lookups prevents later lookups", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const searchVec = vi.fn(async () => {
        if (searchVec.mock.calls.length === 1) {
          controller.abort(new Error("Shutdown requested by stdin-eof"));
        }
        return [vecHit];
      });

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => []) as never;

      await expect(structuredSearch(store, [
        { type: "vec", query: "first semantic query" },
        { type: "vec", query: "second semantic query" },
      ], { signal: controller.signal })).rejects.toThrow(SHUTDOWN);

      expect(searchVec).toHaveBeenCalledTimes(1);
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("abort before reranking prevents reranking", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const rerank = vi.fn(async () => [{ file: vecHit.filepath, score: 0.8 }]);
      const searchVec = vi.fn(async () => {
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      });

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchVec = searchVec as never;
      store.rerank = rerank as never;

      await expect(structuredSearch(store, [
        { type: "vec", query: "semantic query" },
      ], { signal: controller.signal })).rejects.toThrow(SHUTDOWN);

      expect(searchVec).toHaveBeenCalledTimes(1);
      expect(rerank).not.toHaveBeenCalled();
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("rerank aborting mid-call rejects instead of blending its scores", async () => {
    const store = await createCancellationStore();
    try {
      const controller = new AbortController();
      const rerank = vi.fn(async (_q: string, docs: { file: string; text: string }[]) => {
        controller.abort(new Error("Shutdown requested by stdin-eof"));
        return docs.map((doc) => ({ file: doc.file, score: 0.8 }));
      });

      store.llm = stubLlm(controller.signal, embedOk);
      store.searchVec = vi.fn(async () => [vecHit]) as never;
      store.rerank = rerank as never;

      await expect(structuredSearch(store, [
        { type: "vec", query: "semantic query" },
      ], { signal: controller.signal })).rejects.toThrow(SHUTDOWN);

      expect(rerank).toHaveBeenCalledTimes(1);
    } finally {
      await cleanupCancellationStore(store);
    }
  });

  test("skipRerank still rejects when the only searchVec aborts and returns a hit", async () => {
    const store = await createCancellationStore();
    try {
      const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
      llm.embedBatch = (async (texts: string[]) =>
        texts.map(() => ({ embedding: [1, 2, 3], model: llm.embedModelName }))
      ) as typeof llm.embedBatch;

      store.llm = llm;
      const searchVec = vi.fn(async () => {
        llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      });
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => [{ file: vecHit.filepath, score: 0.8 }]) as never;

      await expect(structuredSearch(store, [
        { type: "vec", query: "semantic query" },
      ], { skipRerank: true })).rejects.toThrow(SHUTDOWN);

      expect(searchVec).toHaveBeenCalledTimes(1);
      expect(store.rerank).not.toHaveBeenCalled();
      await llm.dispose();
    } finally {
      await cleanupCancellationStore(store);
    }
  });
});

describe("vectorSearchQuery cooperative cancellation", () => {
  afterEach(() => {
    setDefaultLlamaCpp(null);
  });

  test("a store without an attached LLM observes the default LLM shutdown signal", async () => {
    const store = await createCancellationStore();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    try {
      store.llm = null;
      setDefaultLlamaCpp(llm);

      const searchVec = vi.fn(async () => {
        llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      });
      store.expandQuery = vi.fn(async () => [
        { type: "vec", query: "first expansion" },
        { type: "hyde", query: "second expansion document" },
      ]) as never;
      store.searchVec = searchVec as never;

      await expect(vectorSearchQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      // Without the effective-LLM lookup this would run all three queries.
      expect(searchVec).toHaveBeenCalledTimes(1);
    } finally {
      setDefaultLlamaCpp(null);
      await llm.dispose();
      await cleanupCancellationStore(store);
    }
  });

  test("the final lookup aborting with a hit rejects instead of returning it", async () => {
    const store = await createCancellationStore();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    try {
      store.llm = null;
      setDefaultLlamaCpp(llm);

      // The only lookup returns a usable hit and aborts on the way out.
      const searchVec = vi.fn(async () => {
        llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));
        return [vecHit];
      });
      store.expandQuery = vi.fn(async () => []) as never;
      store.searchVec = searchVec as never;

      await expect(vectorSearchQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(searchVec).toHaveBeenCalledTimes(1);
    } finally {
      setDefaultLlamaCpp(null);
      await llm.dispose();
      await cleanupCancellationStore(store);
    }
  });

  test("an already-aborted default LLM never reaches query expansion", async () => {
    const store = await createCancellationStore();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    try {
      store.llm = null;
      setDefaultLlamaCpp(llm);
      llm.requestSessionAbort(new Error("Shutdown requested by stdin-eof"));

      const expandQuery = vi.fn(async () => []);
      const searchVec = vi.fn(async () => [vecHit]);
      store.expandQuery = expandQuery as never;
      store.searchVec = searchVec as never;

      await expect(vectorSearchQuery(store, "user query")).rejects.toThrow(SHUTDOWN);
      expect(expandQuery).not.toHaveBeenCalled();
      expect(searchVec).not.toHaveBeenCalled();
    } finally {
      setDefaultLlamaCpp(null);
      await llm.dispose();
      await cleanupCancellationStore(store);
    }
  });
});

describe("stdio EOF during embedding", () => {
  test("the cancelled search releases its session so teardown runs in order", async () => {
    const store = await createCancellationStore();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const embedStarted = deferred();
    const releaseEmbed = deferred();
    const hardStops: string[] = [];
    const teardown: string[] = [];

    try {
      llm.embedBatch = (async (texts: string[]) => {
        embedStarted.resolve();
        await releaseEmbed.promise;
        return texts.map(() => ({ embedding: [1, 2, 3], model: llm.embedModelName }));
      }) as typeof llm.embedBatch;

      store.llm = llm;
      const searchVec = vi.fn(async () => [vecHit]);
      store.searchVec = searchVec as never;
      store.rerank = vi.fn(async () => []) as never;

      const inflight = createInflightGate();
      const aborted = deferred();
      const search = inflight.track(() =>
        structuredSearch(store, [{ type: "vec", query: "typed vector query" }], { skipRerank: true }),
      );

      const running = search();
      await embedStarted.promise;

      const coordinator = createShutdownCoordinator({
        closeAdmission() {
          inflight.closeAdmission();
          llm.closeSessionAdmission();
          teardown.push("close-admission");
        },
        stopServing() { teardown.push("stop-serving"); },
        requestAbort(reason) {
          llm.requestSessionAbort(reason);
          teardown.push("request-abort");
          aborted.resolve();
        },
        waitForInflight: () => inflight.waitForIdle(),
        waitForLlmIdle: () => llm.waitForSessionIdle(),
        async disposeLlm() {
          teardown.push("llm-dispose");
          await llm.dispose();
        },
        closeStore() { teardown.push("store-close"); },
        setExitCode(code) { teardown.push(`exit:${code}`); },
        logError() { teardown.push("log-error"); },
        hardStop() { hardStops.push("coordinator-hard-stop"); },
      });

      const stdin = new FakeStdin();
      const shutdown = registerStdioEofShutdown({
        stdin,
        shutdown: (trigger) => coordinator.shutdown(trigger),
        stderr: { write: () => true },
      });

      stdin.emit("end");
      const shutting = shutdown();
      await aborted.promise;
      releaseEmbed.resolve();

      await expect(running).rejects.toThrow(/Shutdown requested/);
      await shutting;

      // The in-flight search released before disposal, so teardown never had
      // to hard-stop and never disposed the LLM under active work.
      expect(searchVec).not.toHaveBeenCalled();
      expect(hardStops).toEqual([]);
      expect(teardown).toEqual([
        "close-admission",
        "stop-serving",
        "request-abort",
        "llm-dispose",
        "store-close",
        "exit:0",
      ]);
      expect(inflight.getActiveCount()).toBe(0);
    } finally {
      await cleanupCancellationStore(store);
    }
  });
});
