/**
 * qmd cleanup must drop content pinned only by inactive documents and compact
 * FTS5 so a wrong-directory update can actually shrink the index (#550).
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  deactivateDocument,
  deleteInactiveDocuments,
  cleanupOrphanedContent,
  countOrphanedContent,
  previewCleanup,
  repackVectors,
  runCleanup,
  vectorTableLayout,
  type Store,
} from "../src/store.js";
import type { Database } from "../src/db.js";

let store: Store | null = null;

async function openStore(): Promise<Store> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-cleanup-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  if (!store) return;
  const path = store.dbPath;
  store.close();
  store = null;
  try { await unlink(path); } catch { /* ignore */ }
  try { await unlink(`${path}-wal`); } catch { /* ignore */ }
  try { await unlink(`${path}-shm`); } catch { /* ignore */ }
});

async function seedDocs(s: Store, keepBody: string, dropBody: string) {
  const now = new Date().toISOString();
  const keepHash = await hashContent(keepBody);
  const dropHash = await hashContent(dropBody);
  insertContent(s.db, keepHash, keepBody, now);
  insertContent(s.db, dropHash, dropBody, now);
  insertDocument(s.db, "docs", "keep.md", "Keep", keepHash, now, now);
  insertDocument(s.db, "docs", "drop.md", "Drop", dropHash, now, now);
  return { keepHash, dropHash };
}

describe("qmd cleanup reclaim (#550)", () => {
  test("deactivating a document removes it from FTS but leaves content until cleanup", async () => {
    const s = await openStore();
    const { dropHash } = await seedDocs(s, "keepuniquealpha body", "dropuniqueomega body");

    expect(s.db.prepare(`SELECT count(*) as c FROM documents_fts`).get()).toEqual({ c: 2 });

    deactivateDocument(s.db, "docs", "drop.md");
    expect(s.db.prepare(`SELECT count(*) as c FROM documents_fts`).get()).toEqual({ c: 1 });
    expect(countOrphanedContent(s.db)).toBe(1);
    expect(previewCleanup(s.db)).toMatchObject({ inactiveDocs: 1, orphanedContent: 1 });

    // Historical bug: deleting the tombstone did not drop the content row.
    deleteInactiveDocuments(s.db);
    expect(s.db.prepare(`SELECT count(*) as c FROM documents`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT count(*) as c FROM content`).get()).toEqual({ c: 2 });
    expect(s.db.prepare(`SELECT count(*) as c FROM content WHERE hash = ?`).get(dropHash)).toEqual({ c: 1 });

    expect(cleanupOrphanedContent(s.db)).toBe(1);
    expect(s.db.prepare(`SELECT count(*) as c FROM content`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT count(*) as c FROM content WHERE hash = ?`).get(dropHash)).toEqual({ c: 0 });
  });

  test("runCleanup drops inactive docs, orphaned content, and leftover FTS rows", async () => {
    const s = await openStore();
    await seedDocs(s, "keepuniquealpha body", "dropuniqueomega body");
    deactivateDocument(s.db, "docs", "drop.md");

    const stats = runCleanup(s.db);
    expect(stats.inactiveDocs).toBe(1);
    expect(stats.orphanedContent).toBe(1);
    expect(s.db.prepare(`SELECT count(*) as c FROM documents`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT count(*) as c FROM content`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT count(*) as c FROM documents_fts`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT path FROM documents WHERE active = 1`).get()).toEqual({ path: "keep.md" });

    const ftsHit = s.db.prepare(
      `SELECT count(*) as c FROM documents_fts WHERE documents_fts MATCH 'keepuniquealpha'`
    ).get() as { c: number };
    expect(ftsHit.c).toBe(1);
  });

  test("runCleanup keeps content still referenced by an active document", async () => {
    const s = await openStore();
    const now = new Date().toISOString();
    const shared = await hashContent("shareduniquebody");
    insertContent(s.db, shared, "shareduniquebody", now);
    insertDocument(s.db, "docs", "keep.md", "Keep", shared, now, now);
    insertDocument(s.db, "docs", "drop.md", "Drop", shared, now, now);
    deactivateDocument(s.db, "docs", "drop.md");

    const stats = runCleanup(s.db);
    expect(stats.inactiveDocs).toBe(1);
    expect(stats.orphanedContent).toBe(0);
    expect(s.db.prepare(`SELECT count(*) as c FROM content`).get()).toEqual({ c: 1 });
    expect(s.db.prepare(`SELECT count(*) as c FROM documents`).get()).toEqual({ c: 1 });
  });
});

describe("qmd cleanup vector repack", () => {
  const DIMS = 3;

  /**
   * Inserts `total` vectors, then deletes every one except the indexes in
   * `keep` (which also get active documents). vec0 drops a chunk only when it
   * is completely empty, so keeping one row in each chunk leaves the holes.
   */
  async function seedVectors(s: Store, total: number, keep: readonly number[]): Promise<void> {
    const now = new Date().toISOString();
    s.ensureVecTable(DIMS);
    s.db.transaction(() => {
      for (let i = 0; i < total; i++) {
        const hash = `vec${String(i).padStart(5, "0")}`;
        if (keep.includes(i)) {
          insertContent(s.db, hash, `body ${hash}`, now);
          insertDocument(s.db, "docs", `${hash}.md`, hash, hash, now, now);
        }
        s.insertEmbedding(hash, 0, 0, new Float32Array([1, i * 0.001, 0]), "test-model", now, 1);
      }
      const drop = s.db.prepare(`DELETE FROM vectors_vec WHERE hash_seq = ?`);
      for (let i = 0; i < total; i++) if (!keep.includes(i)) drop.run(`vec${String(i).padStart(5, "0")}_0`);
    })();
  }

  /** Two chunks of 1024 slots holding three live rows: two in the first chunk, one in the second. */
  const HOLEY = { total: 1100, keep: [0, 1, 1099] } as const;

  function nearest(s: Store): string {
    const row = s.db.prepare(`SELECT hash_seq FROM vectors_vec WHERE embedding MATCH ? AND k = 1`).get(new Float32Array([1, 0, 0])) as { hash_seq: string };
    return row.hash_seq;
  }

  test("layout counts the chunks a packed table would need against the chunks in use", async () => {
    const s = await openStore();
    await seedVectors(s, HOLEY.total, HOLEY.keep);

    expect(vectorTableLayout(s.db)).toEqual({ rows: 3, chunks: 2, neededChunks: 1, occupancy: 0.5 });
  });

  test("layout is null without a vector table", async () => {
    const s = await openStore();
    expect(vectorTableLayout(s.db)).toBeNull();
    expect(previewCleanup(s.db)).toMatchObject({ vectorLayout: null, vectorsRepacked: false });
  });

  test("runCleanup repacks a table that is mostly holes and keeps every live row", async () => {
    const s = await openStore();
    await seedVectors(s, HOLEY.total, HOLEY.keep);

    const stats = runCleanup(s.db);
    expect(stats.vectorsRepacked).toBe(true);
    expect(stats.vectorLayout).toMatchObject({ chunks: 2, neededChunks: 1 });
    expect(vectorTableLayout(s.db)).toEqual({ rows: 3, chunks: 1, neededChunks: 1, occupancy: 1 });
    expect(s.db.prepare(`SELECT hash_seq FROM vectors_vec ORDER BY hash_seq`).all()).toEqual([
      { hash_seq: "vec00000_0" }, { hash_seq: "vec00001_0" }, { hash_seq: "vec01099_0" },
    ]);
    expect(nearest(s)).toBe("vec00000_0");
  });

  test("runCleanup leaves a packed table alone", async () => {
    const s = await openStore();
    await seedVectors(s, 3, [0, 1, 2]);

    const stats = runCleanup(s.db);
    expect(stats.vectorsRepacked).toBe(false);
    expect(stats.vectorLayout).toEqual({ rows: 3, chunks: 1, neededChunks: 1, occupancy: 1 });
  });

  test("previewCleanup reports the repack without rebuilding", async () => {
    const s = await openStore();
    await seedVectors(s, HOLEY.total, HOLEY.keep);

    expect(previewCleanup(s.db)).toMatchObject({ vectorsRepacked: true, vectorLayout: { chunks: 2, neededChunks: 1 } });
    expect(vectorTableLayout(s.db)).toMatchObject({ chunks: 2 });
  });

  test("a chunk move that fails midway leaves that chunk's rows in place", async () => {
    const s = await openStore();
    await seedVectors(s, HOLEY.total, HOLEY.keep);
    const failing: Database = {
      prepare: (sql: string) => {
        const real = s.db.prepare(sql);
        if (!sql.startsWith("INSERT INTO vectors_vec (")) return real;
        return { ...real, get: real.get.bind(real), all: real.all.bind(real), iterate: real.iterate.bind(real), run: () => { throw new Error("injected failure during re-insert"); } };
      },
      transaction: (fn) => s.db.transaction(fn),
      exec: (sql: string) => s.db.exec(sql),
      loadExtension: (path: string) => s.db.loadExtension(path),
      close: () => s.db.close(),
    };

    expect(() => repackVectors(failing)).toThrow("injected failure during re-insert");
    expect(vectorTableLayout(s.db)).toEqual({ rows: 3, chunks: 2, neededChunks: 1, occupancy: 0.5 });
    expect(nearest(s)).toBe("vec00000_0");
    expect(s.db.prepare(`SELECT hash_seq FROM vectors_vec ORDER BY hash_seq`).all()).toEqual([
      { hash_seq: "vec00000_0" }, { hash_seq: "vec00001_0" }, { hash_seq: "vec01099_0" },
    ]);
  });

  test("previewCleanup projects the layout after orphan removal, matching what runCleanup does", async () => {
    const s = await openStore();
    const now = new Date().toISOString();
    s.ensureVecTable(DIMS);
    s.db.transaction(() => {
      for (let i = 0; i < 2048; i++) {
        const hash = `orphan${String(i).padStart(5, "0")}`;
        insertContent(s.db, hash, `body ${hash}`, now);
        insertDocument(s.db, "docs", `${hash}.md`, hash, hash, now, now);
        s.insertEmbedding(hash, 0, 0, new Float32Array([1, i * 0.001, 0]), "test-model", now, 1);
      }
    })();
    for (let i = 0; i < 2048; i += 2) deactivateDocument(s.db, "docs", `orphan${String(i).padStart(5, "0")}.md`);

    expect(vectorTableLayout(s.db)).toEqual({ rows: 2048, chunks: 2, neededChunks: 2, occupancy: 1 });
    expect(previewCleanup(s.db)).toMatchObject({ orphanedVectors: 1024, vectorsRepacked: true, vectorLayout: { rows: 1024, chunks: 2, neededChunks: 1 } });

    const stats = runCleanup(s.db);
    expect(stats.vectorsRepacked).toBe(true);
    expect(vectorTableLayout(s.db)).toEqual({ rows: 1024, chunks: 1, neededChunks: 1, occupancy: 1 });
  });

  test("a legacy vector table without hash_seq is left alone", async () => {
    const s = await openStore();
    s.ensureVecTable(DIMS);
    s.db.exec(`DROP TABLE vectors_vec`);
    s.db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash TEXT PRIMARY KEY, embedding float[${DIMS}] distance_metric=cosine)`);
    s.db.transaction(() => {
      for (let i = 0; i < 1100; i++) s.db.prepare(`INSERT INTO vectors_vec (hash, embedding) VALUES (?, ?)`).run(`legacy${i}`, new Float32Array([1, i * 0.001, 0]));
      for (let i = 2; i < 1099; i++) s.db.prepare(`DELETE FROM vectors_vec WHERE hash = ?`).run(`legacy${i}`);
    })();

    expect(repackVectors(s.db)).toEqual({ rows: 3, chunks: 2, neededChunks: 1, occupancy: 0.5 });
  });
});
