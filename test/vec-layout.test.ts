/**
 * The vector layout resolver owns the names of the sqlite-vec tables and the
 * integer collection ids that partition the vec0 table.
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, type Store } from "../src/store.js";
import {
  LEGACY_VEC_TABLE,
  VEC_COLLECTION_IDS_TABLE,
  VEC_ROWS_TABLE,
  VEC_TABLE,
  allocateCollectionId,
  createPartitionedVecTable,
  hasVectorIndex,
  renameCollectionId,
  resolveCollectionId,
  resolveCollectionIds,
  rowidList,
  vecLayout,
  vecTableReadable,
} from "../src/vec-layout.js";

let store: Store | null = null;
let dir: string | null = null;

async function openStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-vec-layout-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = null;
});

function tableExists(s: Store, name: string): boolean {
  return Boolean(s.db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`).get(name));
}

describe("vecLayout", () => {
  test("resolves none on a store without a vector table", async () => {
    const s = await openStore();
    expect(vecLayout(s.db)).toEqual({ kind: "none" });
    expect(hasVectorIndex(s.db)).toBe(false);
  });

  test("creates the collection id and row mapping tables at open", async () => {
    const s = await openStore();
    expect(tableExists(s, VEC_COLLECTION_IDS_TABLE)).toBe(true);
    expect(tableExists(s, VEC_ROWS_TABLE)).toBe(true);
  });

  test("resolves the partitioned table with its dimensions and shadow tables", async () => {
    const s = await openStore();
    createPartitionedVecTable(s.db, 3);

    const layout = vecLayout(s.db);
    expect(layout.kind).toBe("partitioned");
    if (layout.kind !== "partitioned") return;
    expect(layout.table).toBe(VEC_TABLE);
    expect(layout.dimensions).toBe(3);
    expect(layout.shadow).toEqual({
      chunks: `${VEC_TABLE}_chunks`,
      rowids: `${VEC_TABLE}_rowids`,
      vectorChunks: `${VEC_TABLE}_vector_chunks00`,
      info: `${VEC_TABLE}_info`,
    });
    expect(tableExists(s, layout.shadow.chunks)).toBe(true);
    expect(tableExists(s, layout.shadow.rowids)).toBe(true);
    expect(hasVectorIndex(s.db)).toBe(true);
    expect(vecTableReadable(s.db, layout)).toBe(true);
  });

  test("resolves a legacy hash_seq table, and keeps resolving it while both tables exist", async () => {
    const s = await openStore();
    s.db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)`);

    const legacy = vecLayout(s.db);
    expect(legacy).toMatchObject({ kind: "legacy", table: LEGACY_VEC_TABLE, dimensions: 4, keyedByHashSeq: true });
    expect(hasVectorIndex(s.db)).toBe(false);

    createPartitionedVecTable(s.db, 4);
    expect(vecLayout(s.db).kind).toBe("legacy");
  });

  test("marks a legacy table keyed by hash alone", async () => {
    const s = await openStore();
    s.db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash TEXT PRIMARY KEY, embedding float[2] distance_metric=cosine)`);
    expect(vecLayout(s.db)).toMatchObject({ kind: "legacy", dimensions: 2, keyedByHashSeq: false });
  });

  test("reports a schema entry whose module is not loaded as unreadable", async () => {
    const s = await openStore();
    s.db.exec(`CREATE TABLE ${VEC_TABLE} (collection_id INTEGER, embedding BLOB)`);
    const layout = vecLayout(s.db);
    expect(layout.kind).toBe("partitioned");
    if (layout.kind !== "partitioned") return;
    expect(layout.dimensions).toBeNull();
    expect(vecTableReadable(s.db, layout)).toBe(true);
    s.db.exec(`DROP TABLE ${VEC_TABLE}`);
    expect(vecTableReadable(s.db, layout)).toBe(false);
  });
});

describe("collection ids", () => {
  test("allocates a stable integer id per name and resolves it back", async () => {
    const s = await openStore();
    const docs = allocateCollectionId(s.db, "docs");
    const notes = allocateCollectionId(s.db, "notes");
    expect(docs).not.toBe(notes);
    expect(allocateCollectionId(s.db, "docs")).toBe(docs);
    expect(resolveCollectionId(s.db, "docs")).toBe(docs);
    expect(resolveCollectionId(s.db, "unknown")).toBeUndefined();
    expect(resolveCollectionIds(s.db, ["notes", "unknown", "docs"])).toEqual(new Map([["docs", docs], ["notes", notes]]));
  });

  test("rename keeps the id so partition rows stay reachable", async () => {
    const s = await openStore();
    const id = allocateCollectionId(s.db, "old");
    renameCollectionId(s.db, "old", "new");
    expect(resolveCollectionId(s.db, "old")).toBeUndefined();
    expect(resolveCollectionId(s.db, "new")).toBe(id);
  });

  test("rename of a name without an id is a no-op", async () => {
    const s = await openStore();
    expect(() => renameCollectionId(s.db, "missing", "other")).not.toThrow();
    expect(resolveCollectionId(s.db, "other")).toBeUndefined();
  });

  test("the row mapping refuses a null collection id", async () => {
    const s = await openStore();
    expect(() => s.db.prepare(`INSERT INTO ${VEC_ROWS_TABLE} (hash, seq, collection_id) VALUES ('h', 0, NULL)`).run()).toThrow(/NOT NULL/);
  });

  test("the row mapping is unique per (hash, seq, collection)", async () => {
    const s = await openStore();
    const id = allocateCollectionId(s.db, "docs");
    s.db.prepare(`INSERT INTO ${VEC_ROWS_TABLE} (hash, seq, collection_id) VALUES ('h', 0, ?)`).run(id);
    expect(() => s.db.prepare(`INSERT INTO ${VEC_ROWS_TABLE} (hash, seq, collection_id) VALUES ('h', 0, ?)`).run(id)).toThrow(/UNIQUE/);
  });
});

describe("rowid lists", () => {
  test("json_each expands a rowid list on the loaded SQLite build", async () => {
    const s = await openStore();
    const ids = Array.from({ length: 60_000 }, (_, i) => i + 1);
    const row = s.db.prepare(`SELECT COUNT(*) AS n, MAX(value) AS top FROM json_each(?)`).get(rowidList(ids)) as { n: number; top: number };
    expect(row).toEqual({ n: 60_000, top: 60_000 });
  });
});
