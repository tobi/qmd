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
  searchVec,
  vectorTableLayout,
  type Store,
} from "../src/store.js";
import type { Database } from "../src/db.js";
import { VEC_ROWS_TABLE, VEC_TABLE, deletePartitionRows, partitionRowKey, resolveCollectionId, vecInteger } from "../src/vec-layout.js";

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

  const hashOf = (prefix: string, i: number) => `${prefix}${String(i).padStart(5, "0")}`;

  /**
   * Inserts `total` embedded documents into `collection`, then deletes every
   * vector row except the indexes in `keep`. vec0 drops a chunk only when it
   * is completely empty, so keeping one row in each chunk leaves the holes.
   */
  async function seedVectors(s: Store, total: number, keep: readonly number[], collection = "docs", prefix = "vec"): Promise<void> {
    const now = new Date().toISOString();
    s.ensureVecTable(DIMS);
    s.db.transaction(() => {
      for (let i = 0; i < total; i++) {
        const hash = hashOf(prefix, i);
        insertContent(s.db, hash, `body ${hash}`, now);
        insertDocument(s.db, collection, `${hash}.md`, hash, hash, now, now);
        s.insertEmbedding(hash, 0, 0, new Float32Array([1, i * 0.001, 0]), "test-model", now, 1);
      }
      const kept = new Set(keep.map((i) => hashOf(prefix, i)));
      const rows = s.db.prepare(`SELECT vr.id, vr.hash FROM ${VEC_ROWS_TABLE} vr JOIN vector_collection_ids ci ON ci.id = vr.collection_id WHERE ci.name = ?`).all(collection) as { id: number; hash: string }[];
      deletePartitionRows(s.db, rows.filter((row) => !kept.has(row.hash)).map((row) => row.id));
    })();
  }

  /** Two chunks of 1024 slots holding three live rows: two in the first chunk, one in the second. */
  const HOLEY = { total: 1100, keep: [0, 1, 1099] } as const;

  function nearest(s: Store): string {
    const row = s.db.prepare(`SELECT rowid FROM ${VEC_TABLE} WHERE embedding MATCH ? AND k = 1`).get(new Float32Array([1, 0, 0])) as { rowid: number };
    return partitionRowKey(s.db, row.rowid)!.hash;
  }

  /** Hashes whose vector rows are present in the vec0 table, in hash order. */
  function liveHashes(s: Store): string[] {
    return (s.db.prepare(`SELECT hash FROM ${VEC_ROWS_TABLE} WHERE id IN (SELECT rowid FROM ${VEC_TABLE}) ORDER BY hash`).all() as { hash: string }[]).map((row) => row.hash);
  }

  function partitionCount(s: Store, collection: string): number {
    return (s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_TABLE} WHERE collection_id = ?`).get(vecInteger(resolveCollectionId(s.db, collection)!)) as { c: number }).c;
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
    expect(liveHashes(s)).toEqual(["vec00000", "vec00001", "vec01099"]);
    expect(nearest(s)).toBe("vec00000");
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
        if (!sql.startsWith(`INSERT INTO ${VEC_TABLE} (`)) return real;
        return { ...real, get: real.get.bind(real), all: real.all.bind(real), iterate: real.iterate.bind(real), run: () => { throw new Error("injected failure during re-insert"); } };
      },
      transaction: (fn) => s.db.transaction(fn),
      exec: (sql: string) => s.db.exec(sql),
      loadExtension: (path: string) => s.db.loadExtension(path),
      close: () => s.db.close(),
    };

    expect(() => repackVectors(failing)).toThrow("injected failure during re-insert");
    expect(vectorTableLayout(s.db)).toEqual({ rows: 3, chunks: 2, neededChunks: 1, occupancy: 0.5 });
    expect(nearest(s)).toBe("vec00000");
    expect(liveHashes(s)).toEqual(["vec00000", "vec00001", "vec01099"]);
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

  test("repack moves rows within their own partition and leaves each partition's newest chunk alone", async () => {
    const s = await openStore();
    await seedVectors(s, HOLEY.total, HOLEY.keep, "alpha", "alpha");
    await seedVectors(s, HOLEY.total, HOLEY.keep, "beta", "beta");
    // Rows of different partitions never share a chunk: a packed copy needs one chunk each.
    expect(vectorTableLayout(s.db)).toEqual({ rows: 6, chunks: 4, neededChunks: 2, occupancy: 0.5 });

    const stats = runCleanup(s.db);

    expect(stats.vectorsRepacked).toBe(true);
    expect(vectorTableLayout(s.db)).toEqual({ rows: 6, chunks: 2, neededChunks: 2, occupancy: 1 });
    expect(runCleanup(s.db).vectorsRepacked).toBe(false);
    expect(partitionCount(s, "alpha")).toBe(3);
    expect(partitionCount(s, "beta")).toBe(3);
    const alpha = await searchVec(s.db, "ignored", "test-model", 5, "alpha", undefined, [1, 0, 0]);
    expect(alpha.map((r) => r.hash).sort()).toEqual(["alpha00000", "alpha00001", "alpha01099"]);
    const beta = await searchVec(s.db, "ignored", "test-model", 5, "beta", undefined, [1, 0, 0]);
    expect(beta.map((r) => r.hash).sort()).toEqual(["beta00000", "beta00001", "beta01099"]);
  });

  test("a table below the occupancy trigger with no chunk to move is not repacked", async () => {
    const s = await openStore();
    // Two chunks at 973 of 1024 slots (above the per-chunk fill) and a newest chunk of 51.
    const keep = Array.from({ length: 2099 }, (_, i) => i).filter((i) => !(i <= 50 || (i >= 1024 && i <= 1074)));
    await seedVectors(s, 2099, keep);
    expect(vectorTableLayout(s.db)).toEqual({ rows: 1997, chunks: 3, neededChunks: 2, occupancy: 2 / 3 });

    expect(previewCleanup(s.db).vectorsRepacked).toBe(false);
    expect(runCleanup(s.db).vectorsRepacked).toBe(false);
  });

  test("previewCleanup leaves out a chunk that orphan removal empties, as runCleanup finds it", async () => {
    const s = await openStore();
    await seedVectors(s, 1100, Array.from({ length: 1100 }, (_, i) => i));
    s.db.transaction(() => {
      for (let i = 0; i < 1024; i++) deactivateDocument(s.db, "docs", `${hashOf("vec", i)}.md`);
    })();

    const preview = previewCleanup(s.db);
    expect(preview).toMatchObject({ orphanedVectors: 1024, vectorsRepacked: false, vectorLayout: { rows: 76, chunks: 1, neededChunks: 1, occupancy: 1 } });

    const stats = runCleanup(s.db);
    expect(stats.vectorsRepacked).toBe(false);
    expect(stats.vectorLayout).toEqual(preview.vectorLayout);
  });

  test("a repack reads each chunk again, so a rowid reused by another partition stays there", async () => {
    const s = await openStore();
    // Alpha holds two sparse chunks and a newest one; beta is small and packed.
    await seedVectors(s, 2100, [0, 1, 1024, 1025, 2099], "alpha", "alpha");
    await seedVectors(s, 3, [0, 1, 2], "beta", "beta");
    const betaId = resolveCollectionId(s.db, "beta")!;
    const reused = (s.db.prepare(`SELECT id FROM ${VEC_ROWS_TABLE} WHERE hash = ?`).get(hashOf("alpha", 1024)) as { id: number }).id;

    repackVectors(s.db, (done) => {
      if (done !== 1) return;
      // Between two chunk transactions, another writer drops a row of the second
      // sparse chunk and a beta embed takes over its rowid.
      deletePartitionRows(s.db, [reused]);
      s.db.prepare(`INSERT INTO ${VEC_ROWS_TABLE} (id, hash, seq, collection_id) VALUES (?, ?, 0, ?)`).run(reused, "betanew", betaId);
      s.db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`).run(vecInteger(reused), vecInteger(betaId), new Float32Array([0, 1, 0]));
    });

    const row = s.db.prepare(`SELECT collection_id AS collectionId FROM ${VEC_TABLE} WHERE rowid = ?`).get(vecInteger(reused)) as { collectionId: number };
    expect(row.collectionId).toBe(betaId);
    expect(partitionCount(s, "alpha")).toBe(4);
    expect(partitionCount(s, "beta")).toBe(4);
  });

  test("a failed move of a chunk's last row rolls back the emptied chunk", async () => {
    const s = await openStore();
    // The first chunk keeps one row, so moving it empties the chunk inside the transaction.
    await seedVectors(s, 1100, [0, 1099]);
    const failing: Database = {
      prepare: (sql: string) => {
        const real = s.db.prepare(sql);
        if (!sql.startsWith(`INSERT INTO ${VEC_TABLE} (`)) return real;
        return { ...real, get: real.get.bind(real), all: real.all.bind(real), iterate: real.iterate.bind(real), run: () => { throw new Error("injected failure during re-insert"); } };
      },
      transaction: (fn) => s.db.transaction(fn),
      exec: (sql: string) => s.db.exec(sql),
      loadExtension: (path: string) => s.db.loadExtension(path),
      close: () => s.db.close(),
    };

    expect(() => repackVectors(failing)).toThrow("injected failure during re-insert");
    expect(vectorTableLayout(s.db)).toEqual({ rows: 2, chunks: 2, neededChunks: 1, occupancy: 0.5 });
    expect(liveHashes(s)).toEqual(["vec00000", "vec01099"]);
    expect(nearest(s)).toBe("vec00000");
  });

  test("previewCleanup counts a shared hash's row as orphaned only in the collection that dropped it", async () => {
    const s = await openStore();
    const now = new Date().toISOString();
    s.ensureVecTable(DIMS);
    insertContent(s.db, "sharedhash", "body sharedhash", now);
    insertDocument(s.db, "alpha", "shared.md", "shared", "sharedhash", now, now);
    insertDocument(s.db, "beta", "shared.md", "shared", "sharedhash", now, now);
    s.insertEmbedding("sharedhash", 0, 0, new Float32Array([1, 0, 0]), "test-model", now, 1);
    expect(partitionCount(s, "alpha")).toBe(1);
    expect(partitionCount(s, "beta")).toBe(1);
    deactivateDocument(s.db, "beta", "shared.md");

    expect(previewCleanup(s.db).vectorLayout).toMatchObject({ rows: 1, chunks: 1 });
    runCleanup(s.db);
    expect(partitionCount(s, "alpha")).toBe(1);
    expect(partitionCount(s, "beta")).toBe(0);
  });
});
