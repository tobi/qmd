/**
 * Regression tests for the SDK store operation lifecycle.
 *
 * close() used to dispose the store's LlamaCpp instance and then close SQLite
 * without regard for operations still using either resource: the LLM only
 * leases resources per model call, so an SDK operation was free to keep
 * querying SQLite after its last LLM call returned and after close() had closed
 * the connection underneath it. These tests pin the contract that replaces that
 * race:
 *
 *   open     → operations are admitted
 *   closing  → new operations are rejected, admitted ones are drained
 *   closed   → terminal; llm and SQLite are already released
 *
 * Nothing here loads a model: the in-flight operations are reindex scans whose
 * registered processor holds a gate the test controls, which makes "close()
 * while an operation is mid-flight" fully deterministic.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  StoreCloseDeadlockError,
  StoreClosedError,
  type DocumentProcessor,
  type ProcessorDocument,
  type ProcessorInput,
  type QMDStore,
} from "../src/index.js";

let testDir: string;
let docsDir: string;

beforeEach(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-store-lifecycle-"));
  docsDir = join(testDir, "docs");
  await mkdir(docsDir, { recursive: true });
  await writeFile(join(docsDir, "data.json"), JSON.stringify([{ id: "a", text: "orchard" }]));
});

afterEach(async () => {
  await rm(testDir, { recursive: true, force: true });
});

function dbPath(name: string): string {
  return join(testDir, `${name}.sqlite`);
}

/** Index one source document so every test starts from a real reindex scan. */
async function openStore(path: string): Promise<QMDStore> {
  return createStore({
    dbPath: path,
    config: { collections: { docs: { path: docsDir, pattern: "**/*.json" } } },
  });
}

function render(input: ProcessorInput): ProcessorDocument[] {
  const entries = JSON.parse(Buffer.from(input.bytes).toString()) as Array<{ id: string; text: string }>;
  return entries.map(entry => ({ key: `${entry.id}.md`, markdown: `# ${entry.id}\n${entry.text}\n` }));
}

/** A processor that signals entry, then parks until the test opens the gate. */
function gatedProcessor(onEnter: () => void, gate: Promise<void>): DocumentProcessor {
  return {
    fingerprint: "gated-1",
    process: async input => {
      onEnter();
      await gate;
      return render(input);
    },
  };
}

describe("store operation lifecycle", () => {
  test("close() drains an admitted operation before releasing LLM and SQLite", async () => {
    const path = dbPath("drain");
    const store = await openStore(path);
    const { promise: gate, resolve: openGate } = Promise.withResolvers<void>();
    const { promise: entered, resolve: signalEntered } = Promise.withResolvers<void>();

    try {
      // update() is admitted, then parks inside its processor: an operation that
      // is in flight without holding any LLM lease.
      const update = store.update({ processors: { docs: gatedProcessor(signalEntered, gate) } });
      await entered;

      const closing = store.close();

      // The transition is synchronous: work started after close() begins is
      // rejected instead of being admitted into a teardown.
      await expect(store.getStatus()).rejects.toBeInstanceOf(StoreClosedError);
      await expect(store.searchLex("orchard")).rejects.toBeInstanceOf(StoreClosedError);

      // SQLite is still open and the drain is still waiting, because the
      // admitted update() has not returned yet. The drain has no deadline by
      // design: resource safety wins over shutdown latency, so a caller that
      // wants a bound has to bound the operation itself.
      expect(store.internal.db.prepare("SELECT 1 AS ok").get()).toEqual({ ok: 1 });
      expect(await Promise.race([closing.then(() => "closed"), Promise.resolve("pending")])).toBe("pending");

      openGate();
      const result = await update;
      expect(result.indexed).toBe(1);

      await closing;

      // The store is terminal, and closing again is a resolved no-op.
      await expect(store.getStatus()).rejects.toBeInstanceOf(StoreClosedError);
      await expect(store.close()).resolves.toBeUndefined();

      // The drained update committed before the connection was closed: a fresh
      // store on the same file reads its rows.
      const reopened = await createStore({ dbPath: path });
      try {
        expect(reopened.internal.searchFTS("orchard")).toHaveLength(1);
      } finally {
        await reopened.close();
      }
    } finally {
      openGate();
      await store.close();
    }
  });

  test("concurrent close() calls share one completion and settle after the drain", async () => {
    const store = await openStore(dbPath("concurrent"));
    const { promise: gate, resolve: openGate } = Promise.withResolvers<void>();
    const { promise: entered, resolve: signalEntered } = Promise.withResolvers<void>();

    try {
      const update = store.update({ processors: { docs: gatedProcessor(signalEntered, gate) } });
      await entered;

      const first = store.close();
      const second = store.close();
      const third = store.close();

      // One teardown, one completion: no caller can observe a half-closed store
      // while another caller's teardown is still draining.
      expect(second).toBe(first);
      expect(third).toBe(first);
      expect(await Promise.race([first.then(() => "closed"), Promise.resolve("pending")])).toBe("pending");

      openGate();
      await update;
      await Promise.all([first, second, third]);

      await expect(store.listCollections()).rejects.toBeInstanceOf(StoreClosedError);
      // Late callers join the same, already resolved completion.
      await expect(store.close()).resolves.toBeUndefined();
    } finally {
      openGate();
      await store.close();
    }
  });

  test("a failing admitted operation still releases the drain", async () => {
    const store = await openStore(dbPath("failing"));
    const { promise: gate, resolve: openGate } = Promise.withResolvers<void>();
    const { promise: entered, resolve: signalEntered } = Promise.withResolvers<void>();

    try {
      // A processor failure is published as a skipped file, so update() would
      // resolve. Progress is emitted only after a source is published, which
      // makes a throwing progress callback a deterministic rejection of the
      // admitted operation — and it fires after the gate, i.e. while close() is
      // already draining.
      const update = store.update({
        processors: { docs: gatedProcessor(signalEntered, gate) },
        onProgress: () => { throw new Error("progress exploded"); },
      });
      await entered;
      const closing = store.close();

      openGate();
      // The operation's own error surfaces unchanged through the rollback.
      await expect(update).rejects.toThrow("progress exploded");

      // The lease is released in a finally block, so close() must not hang on an
      // operation that failed after admission.
      await closing;
      await expect(store.getStatus()).rejects.toBeInstanceOf(StoreClosedError);
    } finally {
      openGate();
      await store.close();
    }
  });

  test("close() awaited inside an admitted operation fails fast instead of deadlocking", async () => {
    const store = await openStore(dbPath("deadlock"));
    let closeError: unknown;

    // A processor awaiting close() waits for the operation close() is draining.
    const processor: DocumentProcessor = {
      fingerprint: "close-inside-1",
      process: async input => {
        closeError = await store.close().then(() => null, (error: unknown) => error);
        return render(input);
      },
    };

    try {
      const result = await store.update({ processors: { docs: processor } });

      expect(closeError).toBeInstanceOf(StoreCloseDeadlockError);
      expect(result.indexed).toBe(1);

      // The rejected attempt changed nothing: the store is still open and usable.
      expect(store.internal.db.prepare("SELECT 1 AS ok").get()).toEqual({ ok: 1 });
      expect((await store.listCollections()).map(collection => collection.name)).toEqual(["docs"]);

      // A progress callback cannot await (its signature is synchronous), so the
      // fire-and-forget case must fail the same way rather than tear the store
      // down mid-scan or wait forever.
      const progressErrors: unknown[] = [];
      await store.update({
        processors: { docs: { fingerprint: "close-inside-2", process: async input => render(input) } },
        onProgress: () => {
          void store.close().catch((error: unknown) => progressErrors.push(error));
        },
      });
      expect(progressErrors.length).toBeGreaterThan(0);
      for (const error of progressErrors) expect(error).toBeInstanceOf(StoreCloseDeadlockError);

      // Closing from outside the operation still works.
      await store.close();
      await expect(store.getStatus()).rejects.toBeInstanceOf(StoreClosedError);
    } finally {
      await store.close();
    }
  });

  test("closing one store leaves another store's lifecycle untouched", async () => {
    const first = await openStore(dbPath("first"));
    const second = await openStore(dbPath("second"));

    try {
      await first.addContext("docs", "/", "first only");
      await first.close();

      // Lifecycles are per store: the closed one rejects, the other keeps
      // admitting operations instead of inheriting its neighbour's state.
      await expect(first.listContexts()).rejects.toBeInstanceOf(StoreClosedError);
      expect((await second.listCollections()).map(collection => collection.name)).toEqual(["docs"]);
      await expect(second.getStatus()).resolves.toBeDefined();
    } finally {
      await first.close();
      await second.close();
    }
  });
});
