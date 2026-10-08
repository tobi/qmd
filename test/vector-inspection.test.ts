/**
 * Deep vector inspection validates recorded embedding coverage and the
 * bidirectional relationship between relational row mappings and sqlite-vec.
 */
import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openDatabase } from "../src/db.js";
import { LlamaCpp, type EmbeddingResult } from "../src/llm.js";
import {
  createStore,
  deactivateDocument,
  generateEmbeddings,
  getEmbeddingFingerprint,
  insertContent,
  insertDocument,
  type Store,
} from "../src/store.js";
import {
  LEGACY_VEC_TABLE,
  VEC_ROWS_TABLE,
  VEC_TABLE,
  allocateCollectionId,
  resolveCollectionId,
  vecInteger,
} from "../src/vec-layout.js";
import { inspectVectorIndex } from "../src/vector-inspection.js";

const MODEL = "hf:test/vector-inspection.gguf";
const OTHER_MODEL = "hf:test/vector-inspection-other.gguf";

let store: Store | null = null;
let dir: string | null = null;

async function openStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-vector-inspection-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = null;
});

function seedDocument(s: Store, collection: string, hash: string, path: string): void {
  const now = new Date().toISOString();
  insertContent(s.db, hash, `# ${hash}\n\nBody for ${hash}.`, now);
  insertDocument(s.db, collection, path, hash, hash, now, now);
}

function seedPartitionVector(s: Store, collection: string, hash: string): void {
  const now = new Date().toISOString();
  s.insertEmbedding(hash, 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 1, getEmbeddingFingerprint(MODEL));
  expect(resolveCollectionId(s.db, collection)).toBeDefined();
}

function inspectStore(s: Store, model: string = MODEL) {
  return inspectVectorIndex(
    s.db,
    model,
    getEmbeddingFingerprint(model),
    () => s.getHashesNeedingEmbedding(model),
  );
}

function requireVectorMapping(s: Store, hash: string): { rowid: number; collectionId: number } {
  const row = s.db.prepare(`
    SELECT id AS rowid, collection_id AS collectionId
    FROM ${VEC_ROWS_TABLE}
    WHERE hash = ? AND seq = 0
  `).get(hash) as { rowid: number; collectionId: number } | undefined;
  if (!row) throw new Error(`missing vector mapping for ${hash}`);
  return row;
}

class FixedEmbeddingLlm extends LlamaCpp {
  constructor() {
    super({ embedModel: MODEL });
  }

  override async tokenize(text: string): ReturnType<LlamaCpp["tokenize"]> {
    return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
  }

  override async detokenize(tokens: Awaited<ReturnType<LlamaCpp["tokenize"]>>): Promise<string> {
    return "x".repeat(tokens.length * 16);
  }

  override async embed(): Promise<EmbeddingResult> {
    return { embedding: [0.1, 0.2, 0.3], model: MODEL };
  }

  override async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model: MODEL }));
  }
}

describe("inspectVectorIndex", () => {
  test("evaluates pending coverage inside the same snapshot and accepts an empty absent index", async () => {
    const s = await openStore();
    let observedTransaction = false;

    const result = inspectVectorIndex(
      s.db,
      MODEL,
      getEmbeddingFingerprint(MODEL),
      () => {
        observedTransaction = s.db.inTransaction;
        return s.getHashesNeedingEmbedding(MODEL);
      },
    );

    expect(observedTransaction).toBe(true);
    expect(result).toMatchObject({
      partitionState: "absent",
      activeDocuments: 0,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 0,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });

  test("reports required active rows missing from an absent index", async () => {
    const s = await openStore();
    seedDocument(s, "docs", "absent-hash", "absent.md");
    const now = new Date().toISOString();
    s.db.prepare(`
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 0, 0, ?, ?, 1, ?)
    `).run("absent-hash", MODEL, getEmbeddingFingerprint(MODEL), now);

    expect(inspectStore(s)).toMatchObject({
      partitionState: "absent",
      activeDocuments: 1,
      needsEmbedding: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports legacy vector storage with unavailable peer counts", async () => {
    const s = await openStore();
    seedDocument(s, "docs", "legacy-hash", "legacy.md");
    const now = new Date().toISOString();
    s.db.prepare(`
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 0, 0, ?, ?, 1, ?)
    `).run("legacy-hash", MODEL, getEmbeddingFingerprint(MODEL), now);
    s.db.exec(`
      CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE}
      USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[3] distance_metric=cosine)
    `);

    expect(inspectStore(s)).toMatchObject({
      partitionState: "legacy",
      activeDocuments: 1,
      needsEmbedding: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: null,
      inconsistentPeerRows: null,
      structurallyReady: false,
    });
  });

  test("reports an index opened without sqlite-vec as unreadable", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "unreadable-hash", "unreadable.md");
    seedPartitionVector(s, "docs", "unreadable-hash");
    const dbPath = s.dbPath;
    s.close();
    store = null;

    const rawDb = openDatabase(dbPath);
    try {
      expect(inspectVectorIndex(
        rawDb,
        MODEL,
        getEmbeddingFingerprint(MODEL),
        () => 0,
      )).toMatchObject({
        partitionState: "unreadable",
        activeDocuments: 1,
        requiredPartitionRows: 1,
        missingRequiredPartitionRows: null,
        inconsistentPeerRows: null,
        structurallyReady: false,
      });
    } finally {
      rawDb.close();
    }
  });

  test("scopes required rows to the selected model and accepts coherent inactive cache rows", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "active-hash", "active.md");
    seedPartitionVector(s, "docs", "active-hash");
    seedDocument(s, "archive", "cached-hash", "cached.md");
    seedPartitionVector(s, "archive", "cached-hash");
    deactivateDocument(s.db, "archive", "cached.md");

    expect(inspectStore(s)).toEqual({
      model: MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(MODEL),
      partitionState: "checked",
      activeDocuments: 1,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });

    const otherModelInspection = inspectStore(s, OTHER_MODEL);
    expect(otherModelInspection).toMatchObject({
      model: OTHER_MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(OTHER_MODEL),
      needsEmbedding: 1,
      inconsistentChunkLayouts: 1,
      requiredPartitionRows: 0,
      structurallyReady: false,
    });
  });

  test("reports excess, non-contiguous, and off-generation chunks as inconsistent", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    const now = new Date().toISOString();
    const fingerprint = getEmbeddingFingerprint(MODEL);

    seedDocument(s, "docs", "excess", "excess.md");
    for (const seq of [0, 1, 2, 3]) {
      s.insertEmbedding("excess", seq, seq * 10, new Float32Array([1, 2, 3]), MODEL, now, 3, fingerprint);
    }

    seedDocument(s, "docs", "gap", "gap.md");
    for (const seq of [0, 2, 3]) {
      s.insertEmbedding("gap", seq, seq * 10, new Float32Array([1, 2, 3]), MODEL, now, 3, fingerprint);
    }

    seedDocument(s, "docs", "off-generation", "off-generation.md");
    s.insertEmbedding("off-generation", 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 1, fingerprint);
    s.insertEmbedding(
      "off-generation",
      1,
      10,
      new Float32Array([1, 2, 3]),
      OTHER_MODEL,
      now,
      2,
      getEmbeddingFingerprint(OTHER_MODEL),
    );

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspectStore(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 3,
      requiredPartitionRows: 8,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports a non-integer chunk sequence as inconsistent", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    const now = new Date().toISOString();
    const fingerprint = getEmbeddingFingerprint(MODEL);

    seedDocument(s, "docs", "fractional-sequence", "fractional-sequence.md");
    for (const [seq, pos] of [[0, 0], [0.5, 10], [2, 20]]) {
      s.insertEmbedding(
        "fractional-sequence",
        seq,
        pos,
        new Float32Array([1, 2, 3]),
        MODEL,
        now,
        3,
        fingerprint,
      );
    }

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspectStore(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 1,
      requiredPartitionRows: 3,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports a null total_chunks layout as inconsistent", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "null-total", "null-total.md");
    seedPartitionVector(s, "docs", "null-total");

    // Rebuild this temp fixture with the same production columns and a
    // nullable total_chunks so the inspection can diagnose stored corruption
    // that the ordinary aggregate comparison treats as SQL NULL.
    s.db.exec(`
      DROP INDEX idx_content_vectors_model_fingerprint;
      ALTER TABLE content_vectors RENAME TO strict_content_vectors;
      CREATE TABLE content_vectors (
        hash TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0,
        pos INTEGER NOT NULL DEFAULT 0,
        model TEXT NOT NULL,
        embed_fingerprint TEXT NOT NULL DEFAULT '',
        total_chunks INTEGER,
        embedded_at TEXT NOT NULL,
        PRIMARY KEY (hash, seq)
      );
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      SELECT hash, seq, pos, model, embed_fingerprint, NULL, embedded_at
      FROM strict_content_vectors;
      DROP TABLE strict_content_vectors;
    `);

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspectStore(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 1,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports invalid recorded chunk positions as inconsistent", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    const now = new Date().toISOString();
    const fingerprint = getEmbeddingFingerprint(MODEL);

    const malformedStarts = [
      { hash: "negative-pos", pos: -1 },
      { hash: "float-pos", pos: 0.5 },
      { hash: "unsafe-pos", pos: 9_007_199_254_740_992 },
      { hash: "nonzero-start", pos: 5 },
    ];
    for (const { hash, pos } of malformedStarts) {
      seedDocument(s, "docs", hash, `${hash}.md`);
      s.insertEmbedding(hash, 0, pos, new Float32Array([1, 2, 3]), MODEL, now, 1, fingerprint);
    }

    seedDocument(s, "docs", "nonmonotone-pos", "nonmonotone-pos.md");
    s.insertEmbedding("nonmonotone-pos", 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 2, fingerprint);
    s.insertEmbedding("nonmonotone-pos", 1, 0, new Float32Array([1, 2, 3]), MODEL, now, 2, fingerprint);

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspectStore(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 5,
      requiredPartitionRows: 6,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports a malformed mapping collection id as an inconsistent peer", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "malformed-map", "malformed-map.md");
    seedPartitionVector(s, "docs", "malformed-map");
    const row = requireVectorMapping(s, "malformed-map");
    s.db.prepare(`UPDATE ${VEC_ROWS_TABLE} SET collection_id = ? WHERE id = ?`)
      .run("oops", row.rowid);

    expect(inspectStore(s)).toMatchObject({
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 1,
      structurallyReady: false,
    });
  });

  test("streams vec0 rows and detects mapping-only, vec-only, and partition-mismatched rowids", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    for (const hash of ["map-only", "partition-mismatch", "healthy"]) {
      seedDocument(s, "docs", hash, `${hash}.md`);
      seedPartitionVector(s, "docs", hash);
    }
    const otherCollectionId = allocateCollectionId(s.db, "other");

    const mapOnly = requireVectorMapping(s, "map-only");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(mapOnly.rowid));

    const mismatch = requireVectorMapping(s, "partition-mismatch");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(mismatch.rowid));
    s.db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`)
      .run(vecInteger(mismatch.rowid), vecInteger(otherCollectionId), new Float32Array([1, 2, 3]));

    const vecOnlyRowid = 90_001;
    s.db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`)
      .run(vecInteger(vecOnlyRowid), vecInteger(otherCollectionId), new Float32Array([1, 2, 3]));

    expect(inspectStore(s)).toMatchObject({
      partitionState: "checked",
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 3,
      missingRequiredPartitionRows: 2,
      inconsistentPeerRows: 3,
      structurallyReady: false,
    });
  });

  test("ordinary embedding copies a missing collection partition", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "first", "shared-hash", "shared.md");
    seedPartitionVector(s, "first", "shared-hash");
    seedDocument(s, "second", "shared-hash", "shared.md");
    s.llm = new FixedEmbeddingLlm();

    expect(inspectStore(s)).toMatchObject({
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });

    const result = await generateEmbeddings(s, { model: MODEL });
    expect(result).toMatchObject({ chunksCopied: 1, chunksEmbedded: 0, errors: 0 });
    expect(inspectStore(s)).toMatchObject({
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });

  test("whole-index forced embedding rebuilds peer corruption", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "corrupt-hash", "corrupt.md");
    seedPartitionVector(s, "docs", "corrupt-hash");
    const row = requireVectorMapping(s, "corrupt-hash");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(row.rowid));
    s.llm = new FixedEmbeddingLlm();

    expect(inspectStore(s)).toMatchObject({
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 1,
      structurallyReady: false,
    });

    const result = await generateEmbeddings(s, { model: MODEL, force: true });
    expect(result).toMatchObject({ chunksEmbedded: 1, errors: 0 });
    expect(inspectStore(s)).toMatchObject({
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });
});
