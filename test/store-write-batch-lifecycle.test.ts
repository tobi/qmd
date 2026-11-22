/**
 * store-write-batch-lifecycle.test.ts - who owns a scan's write transaction.
 *
 * A `scanWriteBatch` transaction is a lease on the store's single connection:
 * a second writer that begins inside it either fails (`cannot start a
 * transaction within a transaction`) or, worse, has its rows swept into a
 * transaction somebody else can roll back. These tests pin the ownership
 * guards, the reindex serialization, and the invariant that a scan never
 * suspends while it holds the transaction.
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, reindexCollection, scanWriteBatch, type Store } from "../src/store.js";
import type { DocumentProcessor } from "../src/processors.js";

const FILE_COUNT = 3;

let testDir: string;
let docsDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-write-batch-lifecycle-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(async () => {
  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  store = createStore(join(testDir, `index-${stamp}.sqlite`));
  docsDir = join(testDir, `docs-${stamp}`);
  await mkdir(docsDir);
  for (let i = 0; i < FILE_COUNT; i++) {
    await writeFile(join(docsDir, `doc-${i}.md`), `# Doc ${i}\n\nbody ${i}\n`);
  }
});

afterEach(() => {
  store.close();
});

function activeDocumentCount(collection: string): number {
  const row = store.db
    .prepare(`SELECT COUNT(*) AS c FROM documents WHERE collection = ? AND active = 1`)
    .get(collection) as { c: number };
  return row.c;
}

describe("scanWriteBatch transaction ownership", () => {
  test("a second batch cannot begin inside an open batch, and cannot close it", () => {
    const owner = scanWriteBatch(store.db);
    owner.next();
    const intruder = scanWriteBatch(store.db);

    expect(() => intruder.next()).toThrow(/already has an open transaction/);

    // commit/rollback stay no-ops when the batch holds no transaction: they
    // must never commit or roll back the rows a different writer owns.
    intruder.commit();
    intruder.rollback();
    expect(store.db.inTransaction).toBe(true);

    owner.commit();
    expect(store.db.inTransaction).toBe(false);
  });

  test("a batch never rolls back a transaction it does not own", () => {
    store.db.exec("BEGIN");
    store.db
      .prepare(`INSERT INTO store_config (key, value) VALUES ('foreign', 'kept')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value`)
      .run();

    const batch = scanWriteBatch(store.db);
    expect(() => batch.next()).toThrow(/already has an open transaction/);
    batch.rollback();

    store.db.exec("COMMIT");
    expect(store.db.prepare(`SELECT value FROM store_config WHERE key = 'foreign'`).get())
      .toEqual({ value: "kept" });
  });

  test("committing after the transaction vanished fails instead of claiming writes", () => {
    const batch = scanWriteBatch(store.db);
    batch.next();
    store.db.exec("COMMIT");

    expect(() => batch.commit()).toThrow(/closed outside the batch/);
    expect(store.db.inTransaction).toBe(false);

    // The batch released the connection on the way out, so a fresh scan starts.
    const next = scanWriteBatch(store.db);
    next.next();
    expect(store.db.inTransaction).toBe(true);
    next.rollback();
    expect(store.db.inTransaction).toBe(false);
  });

  test("rolling back after SQLite already closed the transaction clears ownership", () => {
    const batch = scanWriteBatch(store.db);
    batch.next();
    // SQLite can abort a transaction on its own (constraint, IO, disk-full).
    store.db.exec("ROLLBACK");

    expect(() => batch.rollback()).not.toThrow();
    expect(store.db.inTransaction).toBe(false);

    // The connection is usable again: the aborted batch owns nothing.
    const next = scanWriteBatch(store.db);
    next.next();
    expect(store.db.inTransaction).toBe(true);
    next.rollback();
  });

  test("batches on separate connections do not block each other", () => {
    const other = createStore(join(testDir, `other-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
    try {
      const first = scanWriteBatch(store.db);
      const second = scanWriteBatch(other.db);
      first.next();
      second.next();
      expect(store.db.inTransaction).toBe(true);
      expect(other.db.inTransaction).toBe(true);
      first.commit();
      second.commit();
      expect(store.db.inTransaction).toBe(false);
      expect(other.db.inTransaction).toBe(false);
    } finally {
      other.close();
    }
  });
});

describe("reindexCollection serialization", () => {
  test("simultaneous scans queue instead of nesting BEGIN", async () => {
    const first = reindexCollection(store, docsDir, "**/*.md", "notes");
    const second = reindexCollection(store, docsDir, "**/*.md", "notes");
    const [firstResult, secondResult] = await Promise.all([first, second]);

    // The first scan indexes every file; the second starts after it committed
    // and therefore sees them all unchanged.
    expect(firstResult.indexed).toBe(FILE_COUNT);
    expect(firstResult.unchanged).toBe(0);
    expect(secondResult.indexed).toBe(0);
    expect(secondResult.updated).toBe(0);
    expect(secondResult.unchanged).toBe(FILE_COUNT);
    expect(activeDocumentCount("notes")).toBe(FILE_COUNT);
    expect(store.db.inTransaction).toBe(false);
  });

  test("a failing scan rolls back and releases the queue for the next scan", async () => {
    const failing = reindexCollection(store, docsDir, "**/*.md", "notes", {
      onProgress: () => {
        throw new Error("progress callback failed");
      },
    });
    const queued = reindexCollection(store, docsDir, "**/*.md", "notes");

    await expect(failing).rejects.toThrow("progress callback failed");
    expect(store.db.inTransaction).toBe(false);

    const result = await queued;
    expect(result.indexed).toBe(FILE_COUNT);
    expect(activeDocumentCount("notes")).toBe(FILE_COUNT);
  });

  test("an aborted transaction surfaces the original error, not a rollback error", async () => {
    const boom = new Error("scan aborted");
    const scan = reindexCollection(store, docsDir, "**/*.md", "notes", {
      onProgress: () => {
        // SQLite drops the batch transaction underneath the scan; the batch's
        // rollback must stay a no-op so the real failure is what the caller sees.
        store.db.exec("ROLLBACK");
        throw boom;
      },
    });

    await expect(scan).rejects.toBe(boom);
    expect(store.db.inTransaction).toBe(false);

    expect((await reindexCollection(store, docsDir, "**/*.md", "notes")).indexed).toBe(FILE_COUNT);
    expect(activeDocumentCount("notes")).toBe(FILE_COUNT);
  });

  test("a processor cannot start a nested scan of the same store", async () => {
    let nested: unknown;
    const processor: DocumentProcessor = {
      fingerprint: "v1",
      process: async () => {
        try {
          // The same processor, so the call is valid on its own terms: it gets
          // past requireProcessor (the running scan registered a processor mode
          // for this collection) and is stopped only by the reentrancy guard.
          await reindexCollection(store, docsDir, "**/*.md", "notes", { processor });
        } catch (error) {
          nested = error;
        }
        return [{ key: "out.md", markdown: "# out\n\nbody\n" }];
      },
    };

    const result = await reindexCollection(store, docsDir, "**/*.md", "notes", { processor });

    expect(nested).toBeInstanceOf(Error);
    expect((nested as Error).message).toMatch(/cannot start a scan from inside a processor/);
    expect(result.indexed).toBe(FILE_COUNT);
  });

  test("an independent scan queues behind a processor-driven scan instead of failing", async () => {
    const entered = Promise.withResolvers<void>();
    const gate = Promise.withResolvers<void>();
    const processor: DocumentProcessor = {
      fingerprint: "v1",
      process: async () => {
        entered.resolve();
        await gate.promise;
        return [{ key: "out.md", markdown: "# out\n\nbody\n" }];
      },
    };

    const processed = reindexCollection(store, docsDir, "**/*.md", "notes", { processor });
    await entered.promise; // the scan now holds the reindex slot inside its processor

    // This caller is not inside the processor's async context, so it must be
    // queued rather than rejected as a nested scan.
    const independent = reindexCollection(store, docsDir, "**/*.md", "other");
    gate.resolve();

    await expect(processed).resolves.toBeDefined();
    expect((await independent).indexed).toBe(FILE_COUNT);
    expect(activeDocumentCount("other")).toBe(FILE_COUNT);
  });

  test("a callback that ends the batch transaction fails the scan at the next file", async () => {
    const scan = reindexCollection(store, docsDir, "**/*.md", "notes", {
      onProgress: () => {
        // User code running inside the batch: committing it would leave the
        // scan autocommitting each file.
        store.db.exec("COMMIT");
      },
    });

    await expect(scan).rejects.toThrow(/closed while the batch was open/);
    expect(store.db.inTransaction).toBe(false);
  });

  test("the scan loop holds no transaction across an await", async () => {
    const collisions: string[] = [];
    await reindexCollection(store, docsDir, "**/*.md", "notes", {
      onProgress: () => {
        // Queued for the next event-loop turn. If the scan suspends while its
        // batch transaction is open, this runs inside that transaction and the
        // second BEGIN throws.
        queueMicrotask(() => {
          if (!store.db.inTransaction) return;
          try {
            store.db.exec("BEGIN");
            store.db.exec("ROLLBACK");
          } catch (error) {
            collisions.push(error instanceof Error ? error.message : String(error));
          }
        });
      },
    });

    expect(collisions).toEqual([]);
    expect(store.db.inTransaction).toBe(false);
    expect(activeDocumentCount("notes")).toBe(FILE_COUNT);
  });
});
