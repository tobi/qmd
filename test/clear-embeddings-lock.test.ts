/**
 * `clearAllEmbeddings()` joins the index's exclusive embed domain.
 *
 * A clear and an embedding run both delete vector rows. A whole-index clear
 * drops the vec0 table under a run that is publishing chunks into it, so that
 * run fails spuriously; a scoped clear that commits between a run's
 * pending-vector read and its writes can resurrect vectors the user asked to
 * delete. The clear therefore takes the same crash-safe sidecar lock as the run
 * and fails with `EmbedLockBusyError` instead of interleaving.
 *
 * An embedding run clearing its own vectors for `force` must not try to acquire
 * the lock it already owns: it goes through the lock-held body, which these
 * tests pin by driving a forced run end to end without a model.
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Database } from "../src/db.js";
import {
  clearAllEmbeddings,
  createStore,
  generateEmbeddings,
  EmbedLockBusyError,
} from "../src/store.js";
import { acquireEmbedLock, embedLockPathForDb } from "../src/cli/embed-lock.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "qmd-clear-lock-"));
  tempDirs.push(dir);
  return dir;
}

function countContentVectors(db: Database): number {
  const row = db.prepare(`SELECT COUNT(*) AS n FROM content_vectors`).get() as { n: number };
  return Number(row.n);
}

/** Seed one chunk row, so a clear has something to remove without a model. */
function seedVectorRow(db: Database): void {
  db.prepare(
    `INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
     VALUES ('seeded', 0, 0, 'test-model', '', 1, '2026-01-01T00:00:00.000Z')`,
  ).run();
}

describe("clearAllEmbeddings — exclusive per index", () => {
  test("a whole-index clear is refused while a run owns the index lock", () => {
    const dir = makeTempDir();
    const dbPath = join(dir, "index.sqlite");
    const store = createStore(dbPath);
    try {
      seedVectorRow(store.db);

      // Another process is embedding this index right now.
      const owner = acquireEmbedLock(embedLockPathForDb(realpathSync(dbPath)));
      expect(owner.status).toBe("acquired");
      if (owner.status !== "acquired") throw new Error("unreachable");

      try {
        expect(() => clearAllEmbeddings(store.db)).toThrow(EmbedLockBusyError);
        // Refused before it dropped the vec0 table or deleted a row.
        expect(countContentVectors(store.db)).toBe(1);
      } finally {
        owner.handle.release();
      }

      // With the lock free the same clear runs to completion.
      clearAllEmbeddings(store.db);
      expect(countContentVectors(store.db)).toBe(0);
    } finally {
      store.close();
    }
  });

  test("a scoped clear takes the same lock", () => {
    const dir = makeTempDir();
    const dbPath = join(dir, "index.sqlite");
    const store = createStore(dbPath);
    try {
      seedVectorRow(store.db);
      const owner = acquireEmbedLock(embedLockPathForDb(realpathSync(dbPath)));
      expect(owner.status).toBe("acquired");
      if (owner.status !== "acquired") throw new Error("unreachable");

      try {
        expect(() => clearAllEmbeddings(store.db, "docs")).toThrow(EmbedLockBusyError);
        expect(countContentVectors(store.db)).toBe(1);
      } finally {
        owner.handle.release();
      }

      // Free again: a scoped clear of a collection with no documents still
      // commits (it has nothing exclusive to remove).
      expect(() => clearAllEmbeddings(store.db, "docs")).not.toThrow();
      expect(countContentVectors(store.db)).toBe(1);
    } finally {
      store.close();
    }
  });

  test("a forced embedding run clears through the lock it already owns", async () => {
    const dir = makeTempDir();
    const dbPath = join(dir, "index.sqlite");
    const store = createStore(dbPath);
    try {
      seedVectorRow(store.db);

      // No documents, so the run clears and reports nothing to embed, all
      // before any model call. Re-acquiring the lock it holds would throw.
      const result = await generateEmbeddings(store, { force: true });
      expect(result.nothingToDo).toBe(true);
      expect(countContentVectors(store.db)).toBe(0);
    } finally {
      store.close();
    }
  });
});
