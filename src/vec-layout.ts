/**
 * vec-layout.ts - Names and resolution of the sqlite-vec vector layout.
 *
 * Vectors live in one vec0 table partitioned by an integer collection id, one
 * row per (chunk, active collection). `vector_rows` maps each vec0 rowid back
 * to its (hash, seq, collection_id), and `vector_collection_ids` maps names to
 * ids. The integer key is what makes `qmd collection rename` a one-row update:
 * vec0 rejects UPDATE on a partition key, so a name-keyed partition would have
 * to rewrite every row of the collection.
 *
 * A layout change needs a new permanent table name: `ALTER TABLE RENAME` on a
 * vec0 table leaves its shadow tables under the old name (sqlite-vec 0.1.9),
 * and SQLite refuses to rename shadow tables by hand. `vecLayout` is the one
 * place that knows which table is live, so no other module spells the names.
 */

import type { Database } from "./db.js";

export const VEC_TABLE = "vectors_by_collection";
export const VEC_ROWS_TABLE = "vector_rows";
export const VEC_COLLECTION_IDS_TABLE = "vector_collection_ids";
export const LEGACY_VEC_TABLE = "vectors_vec";

/** vec0's shadow tables for a virtual table name. */
export type VecShadowTables = {
  chunks: string;
  rowids: string;
  vectorChunks: string;
  info: string;
};

export type VecLayout =
  | { kind: "none" }
  | { kind: "legacy"; table: typeof LEGACY_VEC_TABLE; dimensions: number | null; keyedByHashSeq: boolean; shadow: VecShadowTables }
  | { kind: "partitioned"; table: typeof VEC_TABLE; dimensions: number | null; shadow: VecShadowTables };

export type ReadableVecLayout = Exclude<VecLayout, { kind: "none" }>;

function shadowTables(table: string): VecShadowTables {
  return {
    chunks: `${table}_chunks`,
    rowids: `${table}_rowids`,
    vectorChunks: `${table}_vector_chunks00`,
    info: `${table}_info`,
  };
}

export function parseDimensions(sql: string | null): number | null {
  const match = sql?.match(/float\[(\d+)\]/);
  return match?.[1] ? parseInt(match[1], 10) : null;
}

/**
 * Which vector layout the database holds. A legacy table wins while it
 * exists: the migration drops it as its last step, so that drop is the flip.
 */
export function vecLayout(db: Database): VecLayout {
  const rows = db.prepare(
    `SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)`
  ).all(LEGACY_VEC_TABLE, VEC_TABLE) as { name: string; sql: string | null }[];

  const legacy = rows.find((row) => row.name === LEGACY_VEC_TABLE);
  if (legacy) {
    return {
      kind: "legacy",
      table: LEGACY_VEC_TABLE,
      dimensions: parseDimensions(legacy.sql),
      keyedByHashSeq: (legacy.sql ?? "").includes("hash_seq"),
      shadow: shadowTables(LEGACY_VEC_TABLE),
    };
  }
  const partitioned = rows.find((row) => row.name === VEC_TABLE);
  if (partitioned) {
    return {
      kind: "partitioned",
      table: VEC_TABLE,
      dimensions: parseDimensions(partitioned.sql),
      shadow: shadowTables(VEC_TABLE),
    };
  }
  return { kind: "none" };
}

/** True when the partitioned vector index exists; vector search reads nothing else. */
export function hasVectorIndex(db: Database): boolean {
  return vecLayout(db).kind === "partitioned";
}

/**
 * A schema entry can outlive the vec0 module (reopening without sqlite-vec
 * loaded), in which case touching the table throws "no such module".
 */
export function vecTableReadable(db: Database, layout: ReadableVecLayout): boolean {
  try {
    db.prepare(`SELECT 1 FROM ${layout.table} LIMIT 0`).get();
    return true;
  } catch {
    return false;
  }
}

export function createVectorMetadataTables(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${VEC_COLLECTION_IDS_TABLE} (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE
    )
  `);
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${VEC_ROWS_TABLE} (
      id INTEGER PRIMARY KEY,
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL,
      collection_id INTEGER NOT NULL,
      UNIQUE(hash, seq, collection_id)
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_vector_rows_collection ON ${VEC_ROWS_TABLE}(collection_id)`);
}

export function createPartitionedVecTable(db: Database, dimensions: number): void {
  db.exec(
    `CREATE VIRTUAL TABLE IF NOT EXISTS ${VEC_TABLE} USING vec0(collection_id INTEGER PARTITION KEY, embedding float[${dimensions}] distance_metric=cosine)`
  );
}

export function resolveCollectionId(db: Database, name: string): number | undefined {
  const row = db.prepare(`SELECT id FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`).get(name) as { id: number } | undefined;
  return row?.id;
}

/** Ids of the named collections that have one; names without an id are absent. */
export function resolveCollectionIds(db: Database, names: readonly string[]): Map<string, number> {
  const ids = new Map<string, number>();
  const stmt = db.prepare(`SELECT id FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`);
  for (const name of names) {
    const row = stmt.get(name) as { id: number } | undefined;
    if (row) ids.set(name, row.id);
  }
  return ids;
}

export function allocateCollectionId(db: Database, name: string): number {
  db.prepare(`INSERT OR IGNORE INTO ${VEC_COLLECTION_IDS_TABLE} (name) VALUES (?)`).run(name);
  const id = resolveCollectionId(db, name);
  if (id === undefined || id === null) {
    throw new Error(`Could not allocate a vector collection id for '${name}'`);
  }
  return id;
}

export function renameCollectionId(db: Database, oldName: string, newName: string): void {
  db.prepare(`UPDATE ${VEC_COLLECTION_IDS_TABLE} SET name = ? WHERE name = ?`).run(newName, oldName);
}

export function deleteCollectionId(db: Database, name: string): void {
  db.prepare(`DELETE FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`).run(name);
}

/**
 * Integer parameter for the vec0 table. better-sqlite3 binds every JS number
 * as a double, and vec0 checks the bound type of its partition key and rowid
 * instead of applying column affinity, so those parameters go in as bigint.
 */
export function vecInteger(value: number | bigint): bigint {
  return BigInt(value);
}

/**
 * One bound parameter for `json_each(?)`: a scoped search over many
 * partitions can return more rowids than SQLite allows as separate
 * parameters (32,766).
 */
export function rowidList(rowids: readonly number[]): string {
  return JSON.stringify(rowids);
}

/** Deletes vector rows by rowid from the vec0 table (when it exists) and the mapping. */
export function deletePartitionRows(db: Database, rowids: readonly number[]): void {
  if (rowids.length === 0) return;
  const deleteVec = hasVectorIndex(db) ? db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`) : null;
  const deleteRow = db.prepare(`DELETE FROM ${VEC_ROWS_TABLE} WHERE id = ?`);
  for (const id of rowids) {
    deleteVec?.run(vecInteger(id));
    deleteRow.run(id);
  }
}

/** Deletes every partition row of a hash, in every collection. */
export function deletePartitionRowsOfHash(db: Database, hash: string): void {
  const rows = db.prepare(`SELECT id FROM ${VEC_ROWS_TABLE} WHERE hash = ?`).all(hash) as { id: number }[];
  deletePartitionRows(db, rows.map((row) => row.id));
}

const ACTIVE_COLLECTIONS_OF_HASH_SQL = `SELECT DISTINCT collection FROM documents WHERE hash = ? AND active = 1`;

/** Names of the collections with an active document for the hash: the partitions that must hold its vectors. */
export function activeCollectionsOfHash(db: Database, hash: string): string[] {
  return (db.prepare(ACTIVE_COLLECTIONS_OF_HASH_SQL).all(hash) as { collection: string }[]).map((row) => row.collection);
}

/**
 * Replace or insert the vector of (hash, seq) in one collection's partition.
 * vec0 ignores OR REPLACE, so an existing row is deleted and its rowid reused.
 */
export function upsertPartitionVector(db: Database, hash: string, seq: number, collectionId: number, embedding: Float32Array | Uint8Array): void {
  const existing = db.prepare(`SELECT id FROM ${VEC_ROWS_TABLE} WHERE hash = ? AND seq = ? AND collection_id = ?`)
    .get(hash, seq, collectionId) as { id: number } | undefined;
  let rowid: number;
  if (existing) {
    db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(existing.id));
    rowid = existing.id;
  } else {
    rowid = Number(db.prepare(`INSERT INTO ${VEC_ROWS_TABLE} (hash, seq, collection_id) VALUES (?, ?, ?)`).run(hash, seq, collectionId).lastInsertRowid);
  }
  db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`).run(vecInteger(rowid), vecInteger(collectionId), embedding);
}

/** Bulk writer that inserts a vector under a collection unless that partition row exists. */
export class PartitionWriter {
  private readonly ids = new Map<string, number>();
  private readonly collectionsOf;
  private readonly insertRow;
  private readonly insertVec;

  constructor(private readonly db: Database) {
    this.collectionsOf = db.prepare(ACTIVE_COLLECTIONS_OF_HASH_SQL);
    this.insertRow = db.prepare(`INSERT OR IGNORE INTO ${VEC_ROWS_TABLE} (hash, seq, collection_id) VALUES (?, ?, ?)`);
    this.insertVec = db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`);
  }

  private idOf(collection: string): number {
    let id = this.ids.get(collection);
    if (id === undefined) {
      id = allocateCollectionId(this.db, collection);
      this.ids.set(collection, id);
    }
    return id;
  }

  /** True when a row was written; false when the partition already held (hash, seq). */
  writeToCollection(hash: string, seq: number, collection: string, embedding: Uint8Array | Float32Array): boolean {
    const id = this.idOf(collection);
    const inserted = this.insertRow.run(hash, seq, id);
    if (inserted.changes === 0) return false;
    this.insertVec.run(vecInteger(inserted.lastInsertRowid), vecInteger(id), embedding);
    return true;
  }

  writeToActiveCollections(hash: string, seq: number, embedding: Uint8Array | Float32Array): number {
    let written = 0;
    for (const row of this.collectionsOf.all(hash) as { collection: string }[]) {
      if (this.writeToCollection(hash, seq, row.collection, embedding)) written++;
    }
    return written;
  }
}

/**
 * Lookup of stored vector bytes by (hash, seq) from whichever partition holds
 * the chunk, with its two statements prepared once for callers that loop.
 */
export function storedEmbeddingLookup(db: Database): (hash: string, seq: number) => Uint8Array | undefined {
  const rowOf = db.prepare(`SELECT id FROM ${VEC_ROWS_TABLE} WHERE hash = ? AND seq = ? LIMIT 1`);
  const vectorOf = db.prepare(`SELECT embedding FROM ${VEC_TABLE} WHERE rowid = ?`);
  return (hash, seq) => {
    const row = rowOf.get(hash, seq) as { id: number } | undefined;
    if (!row) return undefined;
    const stored = vectorOf.get(vecInteger(row.id)) as { embedding: Uint8Array } | undefined;
    return stored?.embedding;
  };
}

/** Stored vector bytes of (hash, seq) from whichever partition holds it. */
export function storedEmbedding(db: Database, hash: string, seq: number): Uint8Array | undefined {
  return storedEmbeddingLookup(db)(hash, seq);
}

/** (hash, seq) behind a vec0 rowid. */
export function partitionRowKey(db: Database, rowid: number): { hash: string; seq: number } | undefined {
  return db.prepare(`SELECT hash, seq FROM ${VEC_ROWS_TABLE} WHERE id = ?`).get(rowid) as { hash: string; seq: number } | undefined;
}

export type MissingPartitionRow = { hash: string; seq: number; collection: string };

/**
 * Chunks recorded in content_vectors for an active document that have no row
 * in that document's collection partition. Each one is a vector to copy from
 * another partition (a hash that gained a collection) or, with no partition
 * holding it, a chunk to embed again.
 */
export function missingPartitionRows(db: Database, collection?: string): MissingPartitionRow[] {
  const filter = collection ? `AND collection = ?` : ``;
  return db.prepare(`
    SELECT cv.hash, cv.seq, d.collection
    FROM content_vectors cv
    JOIN (SELECT DISTINCT hash, collection FROM documents WHERE active = 1 ${filter}) d ON d.hash = cv.hash
    LEFT JOIN ${VEC_COLLECTION_IDS_TABLE} ci ON ci.name = d.collection
    LEFT JOIN ${VEC_ROWS_TABLE} vr ON vr.hash = cv.hash AND vr.seq = cv.seq AND vr.collection_id = ci.id
    WHERE vr.id IS NULL
    ORDER BY cv.hash, cv.seq, d.collection
  `).all(...(collection ? [collection] : [])) as MissingPartitionRow[];
}
