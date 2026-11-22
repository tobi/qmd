/**
 * The version 2 step copies the legacy `vectors_vec` rows into the
 * per-collection partitioned layout, chunk by chunk, resumably, and drops the
 * legacy table in the transaction that stamps the version.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openDatabase, loadSqliteVec, type Database } from "../src/db.js";
import { createStore, insertContent, insertDocument, type Store } from "../src/store.js";
import {
  VECTOR_PARTITION_VERSION,
  getUserVersion,
  migrateVectorLayout,
  runStoreMigrations,
  type VectorMigrationPhase,
  type VectorMigrationProgress,
} from "../src/store-migrations.js";
import {
  LEGACY_VEC_TABLE,
  VEC_ROWS_TABLE,
  VEC_TABLE,
  createPartitionedVecTable,
  resolveCollectionId,
  vecInteger,
  vecLayout,
} from "../src/vec-layout.js";

let store: Store | null = null;
let extra: Database[] = [];
let dir: string | null = null;

async function openStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-migration-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

function openSecondConnection(s: Store): Database {
  const db = openDatabase(s.dbPath);
  loadSqliteVec(db);
  extra.push(db);
  return db;
}

afterEach(async () => {
  for (const db of extra) db.close();
  extra = [];
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = null;
});

const DIMS = 3;

function vec(x: number, y: number, z: number): Float32Array {
  return new Float32Array([x, y, z]);
}

function storedVector(db: Database, hash: string, seq: number, collection: string): number[] | undefined {
  const id = resolveCollectionId(db, collection);
  if (id === undefined) return undefined;
  const row = db.prepare(`SELECT id FROM ${VEC_ROWS_TABLE} WHERE hash = ? AND seq = ? AND collection_id = ?`).get(hash, seq, id) as { id: number } | undefined;
  if (!row) return undefined;
  const stored = db.prepare(`SELECT embedding FROM ${VEC_TABLE} WHERE rowid = ?`).get(vecInteger(row.id)) as { embedding: Uint8Array } | undefined;
  if (!stored) return undefined;
  return Array.from(new Float32Array(stored.embedding.buffer, stored.embedding.byteOffset, DIMS));
}

function partitionCount(db: Database, collection: string): number {
  const id = resolveCollectionId(db, collection);
  if (id === undefined) return 0;
  return (db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_TABLE} WHERE collection_id = ?`).get(vecInteger(id)) as { c: number }).c;
}

function tableNames(db: Database, prefix: string): string[] {
  return (db.prepare(`SELECT name FROM sqlite_master WHERE name LIKE ? ORDER BY name`).all(`${prefix}%`) as { name: string }[]).map((r) => r.name);
}

/** A store still at version 1 with a legacy vec0 table of small chunks. */
class LegacyFixture {
  readonly vectors = new Map<string, Float32Array>();

  constructor(readonly db: Database, chunkSize = 8) {
    db.exec(`PRAGMA user_version = 1`);
    db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${DIMS}] distance_metric=cosine, chunk_size=${chunkSize})`);
  }

  doc(collection: string, hash: string, active = 1): void {
    const now = new Date().toISOString();
    insertContent(this.db, hash, `Document ${hash}`, now);
    insertDocument(this.db, collection, `${hash}.md`, hash, hash, now, now);
    if (!active) this.db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND hash = ?`).run(collection, hash);
  }

  chunkRow(hash: string, seq: number, total = 1): void {
    this.db.prepare(`INSERT OR IGNORE INTO content_vectors (hash, seq, pos, model, embedded_at, total_chunks) VALUES (?, ?, ?, 'test', ?, ?)`)
      .run(hash, seq, seq * 100, new Date().toISOString(), total);
  }

  legacyVector(hash: string, seq: number, embedding: Float32Array): void {
    this.db.prepare(`INSERT INTO ${LEGACY_VEC_TABLE} (hash_seq, embedding) VALUES (?, ?)`).run(`${hash}_${seq}`, embedding);
    this.vectors.set(`${hash}_${seq}`, embedding);
  }

  /** Document with content_vectors rows and legacy vectors for every chunk. */
  embedded(collection: string, hash: string, chunks: readonly Float32Array[]): void {
    this.doc(collection, hash);
    chunks.forEach((embedding, seq) => {
      this.chunkRow(hash, seq, chunks.length);
      this.legacyVector(hash, seq, embedding);
    });
  }

  legacyChunkCount(): number {
    return (this.db.prepare(`SELECT COUNT(*) AS c FROM ${LEGACY_VEC_TABLE}_chunks`).get() as { c: number }).c;
  }
}

/**
 * Two collections sharing one hash, a two-chunk document, an inactive-only
 * hash, a legacy row without a content_vectors row, a content_vectors row
 * without a vector, and enough filler to span several legacy chunks.
 */
function seedStandardFixture(db: Database): LegacyFixture {
  const f = new LegacyFixture(db);
  f.embedded("a", "h1", [vec(1, 0, 0), vec(0.9, 0.1, 0)]);
  f.embedded("a", "h2", [vec(0, 1, 0)]);
  f.embedded("a", "hs", [vec(0, 0, 1)]);
  f.doc("b", "hs");
  f.embedded("b", "h3", [vec(0.5, 0.5, 0)]);
  f.doc("a", "h4", 0);
  f.chunkRow("h4", 0);
  f.legacyVector("h4", 0, vec(0.1, 0.2, 0.3));
  f.legacyVector("ghost", 0, vec(0.3, 0.2, 0.1));
  f.doc("a", "h5");
  f.chunkRow("h5", 0);
  for (let i = 0; i < 12; i++) {
    f.embedded("a", `fill${String(i).padStart(2, "0")}`, [vec(0.2, 0.2, i * 0.05)]);
  }
  return f;
}

const STANDARD_A_ROWS = 2 + 1 + 1 + 12;
const STANDARD_B_ROWS = 1 + 1;
/** h1 x2, h2, hs, h3, the inactive h4, the ghost row, and 12 fillers. */
const STANDARD_LEGACY_ROWS = 2 + 1 + 1 + 1 + 1 + 1 + 12;

describe("migrateVectorLayout", () => {
  test("a fresh store has nothing to copy and reaches the partition version at open", async () => {
    const s = await openStore();
    expect(getUserVersion(s.db)).toBeGreaterThanOrEqual(1);
    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toBe("applied");
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db).kind).toBe("none");
  });

  test("copies a hash_seq legacy table into per-collection partitions", async () => {
    const s = await openStore();
    const fixture = seedStandardFixture(s.db);
    expect(fixture.legacyChunkCount()).toBeGreaterThan(1);
    const phases: VectorMigrationProgress[] = [];

    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true, onProgress: (p) => phases.push(p) })).toBe("applied");

    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: DIMS });
    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
    expect(s.db.prepare(`SELECT value FROM store_config WHERE key = 'vector_partition_cursor'`).get()).toBeFalsy();

    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS);
    expect((s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_ROWS_TABLE}`).get() as { c: number }).c).toBe(STANDARD_A_ROWS + STANDARD_B_ROWS);

    expect(storedVector(s.db, "h1", 0, "a")).toEqual([1, 0, 0]);
    expect(storedVector(s.db, "h1", 1, "a")).toEqual(Array.from(vec(0.9, 0.1, 0)));
    expect(storedVector(s.db, "hs", 0, "a")).toEqual([0, 0, 1]);
    expect(storedVector(s.db, "hs", 0, "b")).toEqual([0, 0, 1]);
    expect(storedVector(s.db, "h3", 0, "b")).toEqual([0.5, 0.5, 0]);
    expect(storedVector(s.db, "h3", 0, "a")).toBeUndefined();
    expect(storedVector(s.db, "h4", 0, "a")).toBeUndefined();

    const chunkRows = (s.db.prepare(`SELECT hash FROM content_vectors ORDER BY hash`).all() as { hash: string }[]).map((r) => r.hash);
    expect(chunkRows).not.toContain("h4");
    expect(chunkRows).not.toContain("h5");
    expect(chunkRows).toContain("h1");
    expect(chunkRows).toContain("hs");

    const nearest = s.db.prepare(`SELECT rowid, distance FROM ${VEC_TABLE} WHERE embedding MATCH ? AND k = 1 AND collection_id = ?`)
      .get(vec(0.5, 0.5, 0), vecInteger(resolveCollectionId(s.db, "b")!)) as { rowid: number };
    const nearestRow = s.db.prepare(`SELECT hash FROM ${VEC_ROWS_TABLE} WHERE id = ?`).get(nearest.rowid) as { hash: string };
    expect(nearestRow.hash).toBe("h3");

    expect(phases.map((p) => p.phase)).toEqual([...phases.filter((p) => p.phase === "copy").map(() => "copy"), "verify", "flip", "vacuum", "done"]);
    const lastCopy = phases.filter((p) => p.phase === "copy").at(-1)!;
    expect(lastCopy.copied).toBe(lastCopy.total);
    expect(lastCopy.total).toBe(STANDARD_LEGACY_ROWS);
  });

  test("drops a legacy table keyed by hash alone and stamps the version", async () => {
    const s = await openStore();
    s.db.exec(`PRAGMA user_version = 1`);
    s.db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash TEXT PRIMARY KEY, embedding float[${DIMS}] distance_metric=cosine)`);
    s.db.prepare(`INSERT INTO ${LEGACY_VEC_TABLE} (hash, embedding) VALUES ('h1', ?)`).run(vec(1, 0, 0));

    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toBe("applied");
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db).kind).toBe("none");
    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
  });

  test("defers without sqlite-vec and leaves the legacy table and version alone", async () => {
    const s = await openStore();
    const fixture = seedStandardFixture(s.db);

    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: false })).toBe("deferred");
    expect(getUserVersion(s.db)).toBe(1);
    expect(vecLayout(s.db).kind).toBe("legacy");
    expect((s.db.prepare(`SELECT COUNT(*) AS c FROM ${LEGACY_VEC_TABLE}`).get() as { c: number }).c).toBe(fixture.vectors.size);
    expect((s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_ROWS_TABLE}`).get() as { c: number }).c).toBe(0);
  });

  test("keeps the legacy dimension", async () => {
    const s = await openStore();
    s.db.exec(`PRAGMA user_version = 1`);
    s.db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[5] distance_metric=cosine)`);
    const now = new Date().toISOString();
    insertContent(s.db, "h1", "Document h1", now);
    insertDocument(s.db, "a", "h1.md", "h1", "h1", now, now);
    s.db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES ('h1', 0, 0, 'test', ?)`).run(now);
    s.db.prepare(`INSERT INTO ${LEGACY_VEC_TABLE} (hash_seq, embedding) VALUES ('h1_0', ?)`).run(new Float32Array([1, 2, 3, 4, 5]));

    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toBe("applied");
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: 5 });
    expect(partitionCount(s.db, "a")).toBe(1);
  });

  test("resumes from the cursor after a failure between chunks", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    let copies = 0;

    expect(() => migrateVectorLayout(s.db, {
      sqliteVecAvailable: true,
      onProgress: (p) => {
        if (p.phase === "copy" && ++copies === 1) throw new Error("killed after the first chunk");
      },
    })).toThrow("killed after the first chunk");

    expect(getUserVersion(s.db)).toBe(1);
    expect(vecLayout(s.db).kind).toBe("legacy");
    const cursor = s.db.prepare(`SELECT value FROM store_config WHERE key = 'vector_partition_cursor'`).get() as { value: string };
    expect(Number(cursor.value)).toBeGreaterThanOrEqual(1);
    const partial = (s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_ROWS_TABLE}`).get() as { c: number }).c;
    expect(partial).toBeGreaterThan(0);
    expect(partial).toBeLessThan(STANDARD_A_ROWS + STANDARD_B_ROWS);

    const resumed: number[] = [];
    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true, onProgress: (p) => { if (p.phase === "copy") resumed.push(p.copied); } })).toBe("applied");
    expect(resumed.length).toBeLessThan(copies + 10);
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS);
    expect((s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_ROWS_TABLE}`).get() as { c: number }).c).toBe(STANDARD_A_ROWS + STANDARD_B_ROWS);
  });

  test("two openers take turns on chunks and both finish with one copy of every row", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    const second = openSecondConnection(s);
    let secondResult: string | null = null;
    let secondCopies = 0;

    const firstResult = migrateVectorLayout(s.db, {
      sqliteVecAvailable: true,
      onProgress: (p) => {
        if (p.phase === "copy" && secondResult === null) {
          secondResult = migrateVectorLayout(second, {
            sqliteVecAvailable: true,
            onProgress: (q) => { if (q.phase === "copy") secondCopies++; },
          });
        }
      },
    });

    expect(firstResult).toBe("applied");
    expect(secondResult).toBe("applied");
    expect(secondCopies).toBeGreaterThan(0);
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db).kind).toBe("partitioned");
    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS);
    expect((s.db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_ROWS_TABLE}`).get() as { c: number }).c).toBe(STANDARD_A_ROWS + STANDARD_B_ROWS);
  });

  test("the verification pass copies a row that landed in an already-walked chunk", async () => {
    const s = await openStore();
    const fixture = seedStandardFixture(s.db);
    let injected = false;

    expect(migrateVectorLayout(s.db, {
      sqliteVecAvailable: true,
      onProgress: (p) => {
        if (p.phase === "copy" && p.copied === p.total && !injected) {
          injected = true;
          fixture.embedded("b", "late", [vec(0.7, 0.7, 0.1)]);
        }
      },
    })).toBe("applied");

    expect(injected).toBe(true);
    expect(storedVector(s.db, "late", 0, "b")).toEqual(Array.from(vec(0.7, 0.7, 0.1)));
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS + 1);
  });

  test("a legacy row written after the copy phase and before the flip ends up in the partition", async () => {
    const s = await openStore();
    const fixture = seedStandardFixture(s.db);
    const phases: VectorMigrationPhase[] = [];
    let injected = false;

    expect(migrateVectorLayout(s.db, {
      sqliteVecAvailable: true,
      onProgress: (p) => {
        phases.push(p.phase);
        if (p.phase === "verify" && !injected) {
          injected = true;
          fixture.embedded("b", "late", [vec(0.7, 0.7, 0.1)]);
        }
      },
    })).toBe("applied");

    expect(injected).toBe(true);
    expect(phases.slice(phases.indexOf("verify"))).toEqual(["verify", "flip", "vacuum", "done"]);
    expect(storedVector(s.db, "late", 0, "b")).toEqual(Array.from(vec(0.7, 0.7, 0.1)));
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS + 1);
    expect(s.db.prepare(`SELECT 1 FROM content_vectors WHERE hash = 'late' AND seq = 0`).get()).toBeTruthy();
    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
  });

  test("the verification pass and the drop hold one write lock", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    const other = openSecondConnection(s);
    other.exec(`PRAGMA busy_timeout = 0`);
    let lockedAtFlip: boolean | null = null;

    expect(migrateVectorLayout(s.db, {
      sqliteVecAvailable: true,
      onProgress: (p) => {
        if (p.phase !== "flip") return;
        try {
          other.exec(`BEGIN IMMEDIATE`);
          other.exec(`ROLLBACK`);
          lockedAtFlip = false;
        } catch {
          lockedAtFlip = true;
        }
      },
    })).toBe("applied");

    expect(lockedAtFlip).toBe(true);
    expect(vecLayout(s.db).kind).toBe("partitioned");
    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
  });

  test("creating the partitioned table a second time is a no-op", async () => {
    const s = await openStore();
    createPartitionedVecTable(s.db, DIMS);
    expect(() => createPartitionedVecTable(s.db, DIMS)).not.toThrow();
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: DIMS });
    expect(tableNames(s.db, VEC_TABLE)).toContain(`${VEC_TABLE}_chunks`);
  });

  test("a second opener finishes the migration with the partitioned table another opener created", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    createPartitionedVecTable(s.db, DIMS);
    expect(vecLayout(s.db).kind).toBe("legacy");
    const second = openSecondConnection(s);

    expect(migrateVectorLayout(second, { sqliteVecAvailable: true })).toBe("applied");

    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: DIMS });
    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS);
    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toBe("applied");
  });

  test("refuses a partitioned table whose dimensions differ from the legacy table's", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    createPartitionedVecTable(s.db, DIMS + 1);

    expect(() => migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toThrow(/dimensions/);
    expect(getUserVersion(s.db)).toBe(1);
    expect(vecLayout(s.db).kind).toBe("legacy");
  });

  test("the flip drops the legacy table and its shadow tables with the version stamp", async () => {
    const s = await openStore();
    seedStandardFixture(s.db);
    expect(tableNames(s.db, LEGACY_VEC_TABLE).length).toBeGreaterThan(1);

    migrateVectorLayout(s.db, { sqliteVecAvailable: true });

    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
    expect(tableNames(s.db, VEC_TABLE)).toContain(`${VEC_TABLE}_chunks`);
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(migrateVectorLayout(s.db, { sqliteVecAvailable: true })).toBe("applied");
    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
  });
});

describe("runStoreMigrations", () => {
  test("migrates a legacy table an older build created on a stamped store", async () => {
    const s = await openStore();
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    seedStandardFixture(s.db);
    s.db.exec(`PRAGMA user_version = ${VECTOR_PARTITION_VERSION}`);
    expect(vecLayout(s.db).kind).toBe("legacy");

    runStoreMigrations(s.db, { installFtsSyncTriggers: () => {}, sqliteVecAvailable: true });

    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: DIMS });
    expect(tableNames(s.db, LEGACY_VEC_TABLE)).toEqual([]);
    expect(partitionCount(s.db, "a")).toBe(STANDARD_A_ROWS);
    expect(partitionCount(s.db, "b")).toBe(STANDARD_B_ROWS);
  });

  test("keeps the legacy table in place when the repair fails on a stamped store", async () => {
    const s = await openStore();
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    seedStandardFixture(s.db);
    s.db.exec(`PRAGMA user_version = ${VECTOR_PARTITION_VERSION}`);
    createPartitionedVecTable(s.db, DIMS + 1);
    expect(vecLayout(s.db).kind).toBe("legacy");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(() => runStoreMigrations(s.db, { installFtsSyncTriggers: () => {}, sqliteVecAvailable: true })).not.toThrow();

    expect(warn).toHaveBeenCalledWith(expect.stringContaining("Legacy vector table left in place"));
    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db).kind).toBe("legacy");
    warn.mockRestore();
  });

  test("leaves a stamped store without a legacy table alone", async () => {
    const s = await openStore();
    createPartitionedVecTable(s.db, DIMS);

    runStoreMigrations(s.db, { installFtsSyncTriggers: () => { throw new Error("must not reinstall"); }, sqliteVecAvailable: true });

    expect(getUserVersion(s.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(vecLayout(s.db)).toMatchObject({ kind: "partitioned", dimensions: DIMS });
  });
});
