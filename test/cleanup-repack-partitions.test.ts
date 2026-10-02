/**
 * qmd cleanup repacks a partitioned vector table in place: every surviving
 * vector keeps its bytes, each partition's newest chunk keeps its rows where
 * they are, and the doctor's stored-vector check still passes afterwards.
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, getEmbeddingFingerprint, runCleanup, type Store } from "../src/store.js";
import { VEC_ROWS_TABLE, VEC_TABLE, resolveCollectionId, storedEmbedding, vecInteger } from "../src/vec-layout.js";
import { checkEmbeddingVectorSamples } from "../src/cli/qmd.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../src/llm.js";

const MODEL = "repack-model";
// Two vec0 chunks of 1024 slots per partition: two live rows stay in the old
// chunk and one in the newest, after cleanup removes the rest as orphans.
const TOTAL = 1100;
const KEEP = [0, 1, 1099];
const COLLECTIONS = ["alpha", "beta"];

function vectorFor(i: number): number[] {
  return [1, i / TOTAL, 0];
}

// Re-embeds a chunk as the vector its document was stored with, so the doctor
// check compares like with like.
class StoredVectorLlm extends LlamaCpp {
  async tokenize(text: string) { return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1); }
  async embed(text: string) {
    const index = Number(/\d{5}/.exec(text)?.[0]);
    return { embedding: vectorFor(index), model: MODEL };
  }
}

let dir: string | undefined;
let store: Store | undefined;

afterEach(async () => {
  setDefaultLlamaCpp(null);
  store?.close();
  store = undefined;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = undefined;
});

function hashOf(collection: string, i: number): string {
  return `${collection}${String(i).padStart(5, "0")}`;
}

function seedPartition(s: Store, collection: string): void {
  const now = new Date().toISOString();
  const kept = new Set(KEEP);
  s.db.transaction(() => {
    for (let i = 0; i < TOTAL; i++) {
      const hash = hashOf(collection, i);
      s.insertContent(hash, `# ${hash}\n\nbody ${hash}`, now);
      s.insertDocument(collection, `${hash}.md`, hash, hash, now, now);
      s.insertEmbedding(hash, 0, 0, new Float32Array(vectorFor(i)), MODEL, now, 1);
      if (!kept.has(i)) s.deactivateDocument(collection, `${hash}.md`);
    }
  })();
}

/** "chunk_id:chunk_offset" of a hash's vector row in the vec0 table. */
function rowPosition(s: Store, hash: string): string {
  const row = s.db.prepare(`
    SELECT r.chunk_id AS chunk, r.chunk_offset AS offset
    FROM ${VEC_ROWS_TABLE} vr JOIN ${VEC_TABLE}_rowids r ON r.rowid = vr.id
    WHERE vr.hash = ?
  `).get(hash) as { chunk: number; offset: number };
  return `${row.chunk}:${row.offset}`;
}

function newestChunk(s: Store, collection: string): number {
  const partition = vecInteger(resolveCollectionId(s.db, collection)!);
  const row = s.db.prepare(`SELECT MAX(chunk_id) AS chunk FROM ${VEC_TABLE}_chunks WHERE partition00 = ?`)
    .get(partition) as { chunk: number };
  return row.chunk;
}

describe("qmd cleanup repack on a partitioned vector table", () => {
  test("keeps every surviving vector's bytes and each partition's newest chunk, and the doctor check still passes", async () => {
    dir = await mkdtemp(join(tmpdir(), "qmd-repack-partitions-"));
    const s = createStore(join(dir, "index.sqlite"));
    store = s;
    s.ensureVecTable(3);
    for (const collection of COLLECTIONS) seedPartition(s, collection);
    const survivors = COLLECTIONS.flatMap(collection => KEEP.map(i => hashOf(collection, i)));
    const bytesBefore = new Map(survivors.map(hash => [hash, Array.from(storedEmbedding(s.db, hash, 0)!)]));
    const newestBefore = new Map(COLLECTIONS.map(collection => [collection, rowPosition(s, hashOf(collection, 1099))]));
    for (const collection of COLLECTIONS) {
      expect(rowPosition(s, hashOf(collection, 1099)).split(":")[0]).toBe(String(newestChunk(s, collection)));
    }

    const stats = runCleanup(s.db);

    expect(stats.vectorsRepacked).toBe(true);
    for (const hash of survivors) {
      expect(Array.from(storedEmbedding(s.db, hash, 0)!)).toEqual(bytesBefore.get(hash));
    }
    for (const collection of COLLECTIONS) {
      expect(rowPosition(s, hashOf(collection, 1099))).toBe(newestBefore.get(collection));
      // The old chunk's two rows moved behind the newest one, which emptied it.
      expect(rowPosition(s, hashOf(collection, 0)).split(":")[0]).toBe(String(newestChunk(s, collection)));
    }
    setDefaultLlamaCpp(new StoredVectorLlm());
    expect(await checkEmbeddingVectorSamples(s.db, MODEL, getEmbeddingFingerprint(MODEL), survivors.length))
      .toMatchObject({ ok: true });
  });
});
