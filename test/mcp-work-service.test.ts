/** Covers shared search admission, maintenance, and shutdown behavior. */
import { describe, expect, test, vi } from "vitest";
import type {
  HybridQueryResult,
  QMDStore,
  SearchResult,
} from "../src/index.js";
import {
  INTERACTIVE_QUEUE_LIMIT,
  INTERACTIVE_QUEUE_TIMEOUT_MS,
  LEXICAL_FALLBACK_CONCURRENCY_LIMIT,
  MAINTENANCE_QUEUE_LIMIT,
  QMDWorkService,
} from "../src/mcp/work-service.js";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

async function waitForCall(mock: {
  mock: { calls: unknown[][] };
}): Promise<void> {
  for (let attempt = 0; attempt < 20 && mock.mock.calls.length === 0; attempt++)
    await flush();
  expect(mock.mock.calls.length).toBeGreaterThan(0);
}

function lexicalResult(score = 0.8): SearchResult {
  return {
    filepath: "/tmp/docs/one.md",
    displayPath: "docs/one.md",
    title: "One",
    context: null,
    hash: "abcdef123456",
    docid: "abcdef",
    collectionName: "docs",
    modifiedAt: "2026-01-01T00:00:00.000Z",
    bodyLength: 12,
    score,
    source: "fts",
  };
}

function semanticResult(): HybridQueryResult {
  return {
    file: "qmd://docs/one.md",
    displayPath: "docs/one.md",
    title: "One",
    body: "one document",
    bestChunk: "one document",
    bestChunkPos: 0,
    score: 0.9,
    context: null,
    docid: "abcdef",
  };
}

function createFakeStore() {
  const search = vi.fn<
    (options: Record<string, unknown>) => Promise<HybridQueryResult[]>
  >(async () => [semanticResult()]);
  const searchLex = vi.fn(async () => [lexicalResult()]);
  const getDefaultCollectionNames = vi.fn(async () => ["docs"]);
  const update = vi.fn(async () => ({
    collections: 1,
    indexed: 1,
    updated: 0,
    unchanged: 0,
    removed: 0,
    skipped: 0,
    needsEmbedding: 1,
  }));
  const embed = vi.fn(async () => ({
    docsProcessed: 1,
    chunksEmbedded: 1,
    errors: 0,
    durationMs: 1,
  }));
  const applyCollectionMutations = vi.fn(async () => undefined);
  const listCollections = vi.fn(async () => [{ name: "docs" }]);
  const close = vi.fn(async () => undefined);
  const store = {
    search,
    searchLex,
    getDefaultCollectionNames,
    update,
    embed,
    applyCollectionMutations,
    listCollections,
    close,
  } as unknown as QMDStore;
  return {
    store,
    search,
    searchLex,
    getDefaultCollectionNames,
    update,
    embed,
    applyCollectionMutations,
    listCollections,
    close,
  };
}

describe("QMDWorkService", () => {
  test("deduplicates heavy searches and allows explicit lexical searches through", async () => {
    const fake = createFakeStore();
    const gate = deferred<HybridQueryResult[]>();
    fake.search.mockReturnValue(gate.promise);
    const service = new QMDWorkService(fake.store);

    const first = service.search({ query: "same request" });
    await waitForCall(fake.search);
    const second = service.search({ query: "same request" });
    await vi.waitFor(() => expect(service.metrics.deduplicated).toBe(1));

    expect(service.metrics.activeHeavy).toBe(1);
    gate.resolve([semanticResult()]);

    await expect(first).resolves.toMatchObject({
      status: "ok",
      mode: "semantic",
    });
    await expect(second).resolves.toMatchObject({
      status: "ok",
      mode: "semantic",
    });
    expect(fake.search).toHaveBeenCalledTimes(1);

    const lexical = await service.search({
      searches: [{ type: "lex", query: "exact term" }],
      rerank: false,
    });
    expect(lexical).toMatchObject({ status: "ok", mode: "lexical" });
    expect(fake.searchLex).toHaveBeenCalledTimes(1);
    expect(fake.search).toHaveBeenCalledTimes(1);

    await service.close();
  });

  test("treats explicit and default empty collection scopes as authoritative", async () => {
    const fake = createFakeStore();
    const service = new QMDWorkService(fake.store);

    await expect(
      service.search({
        searches: [{ type: "lex", query: "nothing" }],
        collections: [],
        rerank: false,
      }),
    ).resolves.toMatchObject({
      status: "ok",
      mode: "lexical",
      authoritativeEmpty: true,
      results: [],
    });
    fake.getDefaultCollectionNames.mockResolvedValue([]);
    await expect(
      service.search({
        searches: [{ type: "lex", query: "nothing" }],
        rerank: false,
      }),
    ).resolves.toMatchObject({
      status: "ok",
      mode: "lexical",
      authoritativeEmpty: true,
      results: [],
    });
    expect(fake.searchLex).not.toHaveBeenCalled();
    expect(fake.search).not.toHaveBeenCalled();
    await service.close();
  });

  test("bounds the heavy queue and degrades queued work after its timeout", async () => {
    const fake = createFakeStore();
    const activeGate = deferred<HybridQueryResult[]>();
    fake.search.mockReturnValue(activeGate.promise);
    const service = new QMDWorkService(fake.store);

    const active = service.search({ query: "active" });
    await waitForCall(fake.search);

    const controllers = Array.from(
      { length: INTERACTIVE_QUEUE_LIMIT },
      () => new AbortController(),
    );
    const queued = controllers.map((controller, index) =>
      service.search({ query: `queued-${index}` }, controller.signal),
    );
    await vi.waitFor(() =>
      expect(service.metrics.queuedInteractive).toBe(INTERACTIVE_QUEUE_LIMIT),
    );

    const overflow = await service.search({ query: "overflow" });
    expect(overflow).toMatchObject({
      status: "ok",
      mode: "lexical",
      reason: "queue_full",
      authoritativeEmpty: false,
    });
    expect(service.metrics.queueFull).toBe(1);

    for (const controller of controllers) controller.abort();
    await Promise.allSettled(queued);

    activeGate.resolve([semanticResult()]);
    await active;
    await service.close();

    const timeoutFake = createFakeStore();
    const timeoutActiveGate = deferred<HybridQueryResult[]>();
    timeoutFake.search.mockReturnValue(timeoutActiveGate.promise);
    const timeoutService = new QMDWorkService(timeoutFake.store);
    const timeoutActive = timeoutService.search({ query: "timeout-active" });
    await waitForCall(timeoutFake.search);
    const timedOut = timeoutService.search({ query: "timed-out" });

    await new Promise((resolve) =>
      setTimeout(resolve, INTERACTIVE_QUEUE_TIMEOUT_MS + 50),
    );
    await expect(timedOut).resolves.toMatchObject({
      status: "ok",
      mode: "lexical",
      reason: "queue_timeout",
      authoritativeEmpty: false,
    });
    expect(timeoutService.metrics.queueTimeout).toBe(1);

    timeoutActiveGate.resolve([semanticResult()]);
    await timeoutActive;
    await timeoutService.close();
  }, 10000);

  test("bounds concurrent lexical degradation", async () => {
    const fake = createFakeStore();
    const activeGate = deferred<HybridQueryResult[]>();
    fake.search.mockReturnValue(activeGate.promise);
    const fallbackGates = Array.from(
      { length: LEXICAL_FALLBACK_CONCURRENCY_LIMIT },
      () => deferred<SearchResult[]>(),
    );
    let fallbackIndex = 0;
    fake.searchLex.mockImplementation(
      async () => fallbackGates[fallbackIndex++]!.promise,
    );
    const service = new QMDWorkService(fake.store);

    const active = service.search({ query: "active" });
    await waitForCall(fake.search);
    const controllers = Array.from(
      { length: INTERACTIVE_QUEUE_LIMIT },
      () => new AbortController(),
    );
    const queued = controllers.map((controller, index) =>
      service.search({ query: `queued-${index}` }, controller.signal),
    );
    await vi.waitFor(() =>
      expect(service.metrics.queuedInteractive).toBe(INTERACTIVE_QUEUE_LIMIT),
    );

    const fallbacks = fallbackGates.map((_, index) =>
      service.search({ query: `fallback-${index}` }),
    );
    await vi.waitFor(() =>
      expect(fake.searchLex).toHaveBeenCalledTimes(
        LEXICAL_FALLBACK_CONCURRENCY_LIMIT,
      ),
    );
    await expect(
      service.search({ query: "fallback-overflow" }),
    ).resolves.toMatchObject({
      status: "unavailable",
      reason: "queue_full",
    });

    for (const gate of fallbackGates) gate.resolve([lexicalResult()]);
    await Promise.all(fallbacks);
    for (const controller of controllers) controller.abort();
    await Promise.allSettled(queued);
    activeGate.resolve([semanticResult()]);
    await active;
    await service.close();
  });

  test("prioritizes queued interactive work over maintenance", async () => {
    const fake = createFakeStore();
    const activeGate = deferred<HybridQueryResult[]>();
    const queuedGate = deferred<HybridQueryResult[]>();
    fake.search
      .mockReturnValueOnce(activeGate.promise)
      .mockReturnValueOnce(queuedGate.promise);
    const service = new QMDWorkService(fake.store);

    const active = service.search({ query: "active" });
    await waitForCall(fake.search);
    const update = service.scheduleUpdate({ collections: ["docs"] });
    const queued = service.search({ query: "queued" });
    await vi.waitFor(() => expect(service.metrics.queuedInteractive).toBe(1));

    activeGate.resolve([semanticResult()]);
    await vi.waitFor(() => expect(fake.search).toHaveBeenCalledTimes(2));
    expect(fake.update).not.toHaveBeenCalled();

    queuedGate.resolve([semanticResult()]);
    await active;
    await queued;
    await vi.waitFor(() =>
      expect(service.getOperation(update.operationId)?.state).toBe("completed"),
    );
    await service.close();
  });

  test("keeps maintenance exclusive and coalesces queued updates", async () => {
    const fake = createFakeStore();
    const firstEmbedGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    const updateGate = deferred<Awaited<ReturnType<typeof fake.update>>>();
    const secondEmbedGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    fake.embed
      .mockReturnValueOnce(firstEmbedGate.promise)
      .mockReturnValueOnce(secondEmbedGate.promise);
    fake.update.mockReturnValue(updateGate.promise);
    const service = new QMDWorkService(fake.store);

    const firstEmbed = service.scheduleEmbed();
    await waitForCall(fake.embed);
    const update = service.scheduleUpdate({ collections: ["one"] });
    const coalesced = service.scheduleUpdate({ collections: ["two"] });
    const secondEmbed = service.scheduleEmbed({ force: true });
    expect(secondEmbed.state).toBe("queued");

    expect(coalesced).toMatchObject({
      operationId: update.operationId,
      coalesced: true,
    });
    expect(service.metrics.maintenanceActive).toBe(true);
    const blocked = await service.search({
      searches: [{ type: "lex", query: "during maintenance" }],
      rerank: false,
    });
    expect(blocked).toMatchObject({
      status: "unavailable",
      reason: "maintenance_busy",
    });
    expect(fake.searchLex).not.toHaveBeenCalled();

    firstEmbedGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await vi.waitFor(() => expect(fake.update).toHaveBeenCalledTimes(1));
    expect(fake.update).toHaveBeenCalledWith({
      collections: ["one", "two"],
    });

    updateGate.resolve({
      collections: 1,
      indexed: 1,
      updated: 0,
      unchanged: 0,
      removed: 0,
      skipped: 0,
      needsEmbedding: 0,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(update.operationId)?.state).toBe("completed"),
    );
    expect(service.getOperation(firstEmbed.operationId)?.state).toBe(
      "completed",
    );

    await vi.waitFor(() => expect(fake.embed).toHaveBeenCalledTimes(2));
    expect(service.getOperation(secondEmbed.operationId)?.state).toBe(
      "running",
    );
    secondEmbedGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(secondEmbed.operationId)?.state).toBe(
        "completed",
      ),
    );
    expect(service.metrics.maintenanceActive).toBe(false);

    const after = await service.search({
      searches: [{ type: "lex", query: "after maintenance" }],
      rerank: false,
    });
    expect(after).toMatchObject({ status: "ok", mode: "lexical" });
    await service.close();
  });

  test("ensures collections through the exclusive service and closes after active work", async () => {
    const fake = createFakeStore();
    const service = new QMDWorkService(fake.store);

    const ensure = service.scheduleEnsure({
      adds: [
        {
          name: "notes",
          path: "/tmp/notes",
          pattern: "**/*.md",
        },
      ],
      contexts: [
        {
          collection: "notes",
          path: "README.md",
          context: "Personal notes",
        },
      ],
      markDirty: false,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(ensure.operationId)?.state).toBe("completed"),
    );
    expect(fake.applyCollectionMutations).toHaveBeenCalledWith([
      {
        kind: "upsert",
        name: "notes",
        path: "/tmp/notes",
        pattern: "**/*.md",
      },
      {
        kind: "context",
        collection: "notes",
        path: "README.md",
        context: "Personal notes",
      },
    ]);
    expect(fake.update).not.toHaveBeenCalled();

    const gate = deferred<HybridQueryResult[]>();
    fake.search.mockReturnValue(gate.promise);
    const active = service.search({ query: "close waits" });
    await waitForCall(fake.search);
    const closing = service.close();
    await flush();
    expect(fake.close).not.toHaveBeenCalled();
    await expect(
      service.search({ query: "after close" }),
    ).rejects.toMatchObject({
      reason: "closed",
    });
    gate.resolve([semanticResult()]);
    await active;
    await closing;
    expect(fake.close).toHaveBeenCalledTimes(1);
  });

  test("cancels queued and active searches without cancelling shared work", async () => {
    const fake = createFakeStore();
    const activeGate = deferred<HybridQueryResult[]>();
    fake.search.mockReturnValue(activeGate.promise);
    const service = new QMDWorkService(fake.store);

    const activeController = new AbortController();
    const active = service.search({ query: "active" }, activeController.signal);
    await waitForCall(fake.search);
    const queuedController = new AbortController();
    const queued = service.search({ query: "queued" }, queuedController.signal);
    await vi.waitFor(() => expect(service.metrics.queuedInteractive).toBe(1));

    queuedController.abort();
    await expect(queued).rejects.toMatchObject({ name: "AbortError" });
    expect(service.metrics.queuedInteractive).toBe(0);
    expect(fake.search).toHaveBeenCalledTimes(1);

    activeController.abort();
    await expect(active).rejects.toMatchObject({ name: "AbortError" });
    activeGate.resolve([semanticResult()]);
    await flush();
    expect(fake.search).toHaveBeenCalledTimes(1);
    await service.close();
  });

  test("coalesces identical maintenance and queues distinct work", async () => {
    const fake = createFakeStore();
    const embedGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    fake.embed.mockReturnValue(embedGate.promise);
    const service = new QMDWorkService(fake.store);

    const firstEmbed = service.scheduleEmbed();
    await waitForCall(fake.embed);
    expect(
      service.scheduleEmbed({
        force: false,
        maxDocsPerBatch: 64,
        maxBatchBytes: 64 * 1024 * 1024,
        chunkStrategy: "regex",
      }),
    ).toMatchObject({
      operationId: firstEmbed.operationId,
      coalesced: true,
    });
    const distinct = service.scheduleEmbed({ force: true });
    expect(distinct.operationId).not.toBe(firstEmbed.operationId);
    expect(distinct.state).toBe("queued");

    embedGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(firstEmbed.operationId)?.state).toBe(
        "completed",
      ),
    );
    await vi.waitFor(() =>
      expect(service.getOperation(distinct.operationId)?.state).toBe(
        "completed",
      ),
    );
    expect(fake.embed).toHaveBeenCalledTimes(2);
    await service.close();
  });

  test("queues an identical embed after intervening maintenance", async () => {
    const fake = createFakeStore();
    const firstEmbedGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    const updateGate = deferred<Awaited<ReturnType<typeof fake.update>>>();
    const secondEmbedGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    fake.embed
      .mockReturnValueOnce(firstEmbedGate.promise)
      .mockReturnValueOnce(secondEmbedGate.promise);
    fake.update.mockReturnValue(updateGate.promise);
    const service = new QMDWorkService(fake.store);

    const firstEmbed = service.scheduleEmbed();
    await waitForCall(fake.embed);
    service.scheduleUpdate({ collections: ["docs"] });
    const secondEmbed = service.scheduleEmbed();
    expect(secondEmbed).toMatchObject({ state: "queued", coalesced: false });
    expect(secondEmbed.operationId).not.toBe(firstEmbed.operationId);

    firstEmbedGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await waitForCall(fake.update);
    updateGate.resolve({
      collections: 1,
      indexed: 1,
      updated: 0,
      unchanged: 0,
      removed: 0,
      skipped: 0,
      needsEmbedding: 1,
    });
    await vi.waitFor(() => expect(fake.embed).toHaveBeenCalledTimes(2));
    secondEmbedGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(secondEmbed.operationId)?.state).toBe(
        "completed",
      ),
    );
    await service.close();
  });

  test("coalesces identical collection ensures", async () => {
    const fake = createFakeStore();
    const applyGate =
      deferred<Awaited<ReturnType<typeof fake.applyCollectionMutations>>>();
    fake.applyCollectionMutations.mockReturnValue(applyGate.promise);
    const service = new QMDWorkService(fake.store);
    const request = {
      adds: [{ name: "notes", path: "/tmp/notes" }],
      markDirty: false,
    };

    const first = service.scheduleEnsure(request);
    await waitForCall(fake.applyCollectionMutations);
    expect(service.scheduleEnsure(request)).toMatchObject({
      operationId: first.operationId,
      coalesced: true,
    });
    const distinct = service.scheduleEnsure({
      adds: [{ name: "other", path: "/tmp/other" }],
      markDirty: false,
    });
    expect(distinct.operationId).not.toBe(first.operationId);
    expect(distinct.state).toBe("queued");

    applyGate.resolve(undefined);
    await vi.waitFor(() =>
      expect(service.getOperation(first.operationId)?.state).toBe("completed"),
    );
    await vi.waitFor(() =>
      expect(service.getOperation(distinct.operationId)?.state).toBe(
        "completed",
      ),
    );
    expect(fake.applyCollectionMutations).toHaveBeenCalledTimes(2);
    await service.close();
  });

  test("preflights collection mutations and removes empty contexts", async () => {
    const fake = createFakeStore();
    fake.listCollections.mockResolvedValue([
      { name: "old" },
      { name: "existing" },
    ]);
    const service = new QMDWorkService(fake.store);

    const operation = service.scheduleEnsure({
      adds: [{ name: "notes", path: "/tmp/notes" }],
      updates: [{ name: "renamed", path: "/tmp/renamed" }],
      renames: [{ from: "old", to: "renamed" }],
      contexts: [{ collection: "renamed", path: "/", context: "" }],
      markDirty: false,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(operation.operationId)?.state).toBe(
        "completed",
      ),
    );
    expect(fake.applyCollectionMutations).toHaveBeenCalledWith([
      { kind: "rename", from: "old", to: "renamed" },
      { kind: "upsert", name: "notes", path: "/tmp/notes" },
      { kind: "upsert", name: "renamed", path: "/tmp/renamed" },
      {
        kind: "context",
        collection: "renamed",
        path: "/",
        context: "",
      },
    ]);

    expect(() =>
      service.scheduleEnsure({
        renames: [
          { from: "a", to: "b" },
          { from: "b", to: "c" },
        ],
      }),
    ).toThrow(expect.objectContaining({ reason: "malformed" }));

    fake.listCollections.mockResolvedValue([{ name: "docs" }]);
    const collision = service.scheduleEnsure({
      adds: [{ name: "docs", path: "/tmp/docs" }],
      markDirty: false,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(collision.operationId)).toMatchObject({
        state: "failed",
        error: { reason: "maintenance_failed" },
      }),
    );
    expect(fake.applyCollectionMutations).toHaveBeenCalledTimes(1);
    await service.close();
  });

  test("schedules bounded dirty updates without blocking distinct maintenance", async () => {
    const fake = createFakeStore();
    const activeGate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    fake.embed.mockReturnValue(activeGate.promise);
    const service = new QMDWorkService(fake.store);

    const active = service.scheduleEmbed({ model: "active" });
    await waitForCall(fake.embed);
    const update = service.scheduleUpdate({ collections: ["docs"] });
    const ensure = service.scheduleEnsure({
      adds: [{ name: "notes", path: "/tmp/notes" }],
      markDirty: true,
    });
    const queuedEmbed = service.scheduleEmbed({ model: "queued" });
    expect(service.metrics.queuedMaintenance).toBe(3);
    expect(ensure.operationId).not.toBe(update.operationId);
    expect(queuedEmbed.state).toBe("queued");

    activeGate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await vi.waitFor(() =>
      expect(service.getOperation(active.operationId)?.state).toBe("completed"),
    );
    await vi.waitFor(() =>
      expect(service.getOperation(ensure.operationId)?.state).toBe("completed"),
    );
    await vi.waitFor(() =>
      expect(service.getOperation(queuedEmbed.operationId)?.state).toBe(
        "completed",
      ),
    );
    expect(fake.update).toHaveBeenCalledWith({
      collections: ["docs", "notes"],
    });
    await service.close();
  });

  test("bounds distinct maintenance requests", async () => {
    const fake = createFakeStore();
    const gate = deferred<Awaited<ReturnType<typeof fake.embed>>>();
    fake.embed.mockReturnValue(gate.promise);
    const service = new QMDWorkService(fake.store);

    service.scheduleEmbed({ model: "active" });
    await waitForCall(fake.embed);
    for (let index = 0; index < MAINTENANCE_QUEUE_LIMIT; index += 1)
      service.scheduleEmbed({ model: `queued-${index}` });
    expect(service.metrics.queuedMaintenance).toBe(MAINTENANCE_QUEUE_LIMIT);
    expect(() => service.scheduleEmbed({ model: "overflow" })).toThrow(
      expect.objectContaining({ reason: "maintenance_busy" }),
    );

    gate.resolve({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    });
    await service.close();
  });

  test("rejects malformed search requests instead of invoking the store", async () => {
    const fake = createFakeStore();
    const service = new QMDWorkService(fake.store);

    await expect(service.search({ query: "" })).rejects.toMatchObject({
      reason: "malformed",
    });
    await expect(
      service.search({ query: "large", limit: 501 }),
    ).rejects.toMatchObject({ reason: "malformed" });
    await expect(
      service.search({ query: "large", candidateLimit: 501 }),
    ).rejects.toMatchObject({ reason: "malformed" });
    expect(fake.getDefaultCollectionNames).not.toHaveBeenCalled();
    await service.close();
  });

  test("reports lexical fallback as non-authoritative when the store fails", async () => {
    const fake = createFakeStore();
    fake.searchLex.mockRejectedValue(new Error("temporary store failure"));
    const service = new QMDWorkService(fake.store);

    const result = await service.search({
      searches: [{ type: "lex", query: "missing" }],
      rerank: false,
    });
    expect(result).toMatchObject({
      status: "unavailable",
      reason: "store_error",
      authoritativeEmpty: false,
    });
    await service.close();
  });
});
