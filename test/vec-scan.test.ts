/**
 * In-memory vector scan (vec-scan.ts): the matrix answers exactly what vec0
 * answers, scopes before the top k, and is never used once the index has
 * moved on from the rows it holds.
 */

import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  insertContent,
  insertDocument,
  searchVec,
  type Store,
} from "../src/store.js";
import { VEC_TABLE, resolveCollectionId } from "../src/vec-layout.js";
import { disableVecScan, enableVecScan, vecScanSearch, vecScanSettled } from "../src/vec-scan.js";

const DIMS = 16;
const stores: Store[] = [];
const dirs: string[] = [];

afterEach(async () => {
  for (const s of stores.splice(0)) s.close();
  for (const d of dirs.splice(0)) await rm(d, { recursive: true, force: true });
  delete process.env.QMD_VEC_SCAN;
  delete process.env.QMD_VEC_SCAN_THREADS;
});

async function openStore(path?: string): Promise<Store> {
  if (!path) {
    const dir = await mkdtemp(join(tmpdir(), "qmd-vec-scan-"));
    dirs.push(dir);
    path = join(dir, "index.sqlite");
  }
  const s = createStore(path);
  stores.push(s);
  return s;
}

/** Deterministic pseudo-random vectors, so a failure reproduces. */
function vectors(seed: number) {
  let x = seed;
  const next = () => (x = (x * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
  return () => Float32Array.from({ length: DIMS }, next);
}

const now = new Date().toISOString();

/** `count` one-chunk documents in `collection`, each with its own vector. */
function seed(s: Store, collection: string, count: number, vec: () => Float32Array): Map<string, Float32Array> {
  const out = new Map<string, Float32Array>();
  s.ensureVecTable(DIMS);
  s.db.transaction(() => {
    for (let i = 0; i < count; i++) {
      const hash = `${collection}${String(i).padStart(6, "0")}`;
      insertContent(s.db, hash, `body ${hash}`, now);
      insertDocument(s.db, collection, `${hash}.md`, hash, hash, now, now);
      const v = vec();
      s.insertEmbedding(hash, 0, 0, v, "test-model", now, 1);
      out.set(hash, v);
    }
  })();
  return out;
}

function knn(s: Store, query: Float32Array, k: number, collectionId?: number) {
  const where = collectionId === undefined ? "" : " AND collection_id = ?";
  const params: (Float32Array | number)[] = [query, k];
  if (collectionId !== undefined) params.push(collectionId);
  return s.db.prepare(`SELECT rowid, distance FROM ${VEC_TABLE} WHERE embedding MATCH ? AND k = ?${where}`)
    .all(...params) as { rowid: number; distance: number }[];
}

function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let j = 0; j < a.length; j++) {
    dot += a[j]! * b[j]!;
    na += a[j]! * a[j]!;
    nb += b[j]! * b[j]!;
  }
  return 1 - dot / Math.sqrt(na * nb);
}

async function enabled(s: Store): Promise<void> {
  enableVecScan(s.db, s.dbPath);
  await vecScanSettled(s.db);
}

describe("in-memory vector scan", () => {
  test("returns vec0's exact top k", async () => {
    const s = await openStore();
    const vec = vectors(1);
    seed(s, "docs", 3000, vec);
    await enabled(s);
    for (let q = 0; q < 10; q++) {
      const query = vec();
      const mine = await vecScanSearch(s.db, query, 60);
      const ref = knn(s, query, 60);
      expect(mine).not.toBeNull();
      expect(mine!.map((m) => m.rowid)).toEqual(ref.map((r) => r.rowid));
      mine!.forEach((m, i) => expect(m.distance).toBeCloseTo(ref[i]!.distance, 5));
    }
  });

  test("scans across worker threads the same as inline", async () => {
    process.env.QMD_VEC_SCAN_THREADS = "4";
    const s = await openStore();
    const vec = vectors(2);
    seed(s, "docs", 33_000, vec);
    await enabled(s);
    const query = vec();
    const mine = await vecScanSearch(s.db, query, 25);
    expect(mine!.map((m) => m.rowid)).toEqual(knn(s, query, 25).map((r) => r.rowid));
  }, 120_000);

  test("a small collection beside a big one gets its own nearest rows", async () => {
    const s = await openStore();
    const vec = vectors(3);
    seed(s, "big", 2000, vec);
    const small = seed(s, "small", 30, vec);
    await enabled(s);
    const query = vec();
    const smallId = resolveCollectionId(s.db, "small")!;
    const mine = await vecScanSearch(s.db, query, 10, smallId);
    expect(mine!.map((m) => m.rowid)).toEqual(knn(s, query, 10, smallId).map((r) => r.rowid));

    const results = await searchVec(s.db, "ignored", "test-model", 10, "small", undefined, Array.from(query));
    const brute = [...small.entries()]
      .map(([hash, v]) => ({ hash, d: cosineDistance(v, query) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 10)
      .map((x) => x.hash);
    expect(results.map((r) => r.hash)).toEqual(brute);
  });

  test("searchVec returns the same results with and without the matrix", async () => {
    const s = await openStore();
    const vec = vectors(4);
    seed(s, "alpha", 800, vec);
    seed(s, "beta", 800, vec);
    const query = Array.from(vec());
    const plain = await searchVec(s.db, "ignored", "test-model", 15, ["alpha", "beta"], undefined, query);
    await enabled(s);
    const viaMatrix = await searchVec(s.db, "ignored", "test-model", 15, ["alpha", "beta"], undefined, query);
    expect(viaMatrix.map((r) => [r.filepath, r.chunkPos])).toEqual(plain.map((r) => [r.filepath, r.chunkPos]));
    viaMatrix.forEach((r, i) => expect(r.score).toBeCloseTo(plain[i]!.score, 5));
  });

  test("a write from another connection sends the next search to vec0 and a later one sees it", async () => {
    const s = await openStore();
    const vec = vectors(5);
    seed(s, "docs", 500, vec);
    await enabled(s);
    expect(await vecScanSearch(s.db, vec(), 5)).not.toBeNull();

    const other = await openStore(s.dbPath);
    const fresh = vec();
    insertContent(other.db, "fresh", "fresh body", now);
    insertDocument(other.db, "docs", "fresh.md", "fresh", "fresh", now, now);
    other.insertEmbedding("fresh", 0, 0, fresh, "test-model", now, 1);

    // Stale: no answer from the matrix, and searchVec still finds the new row through vec0.
    expect(await vecScanSearch(s.db, fresh, 1)).toBeNull();
    const viaVec0 = await searchVec(s.db, "ignored", "test-model", 1, "docs", undefined, Array.from(fresh));
    expect(viaVec0[0]?.hash).toBe("fresh");

    await vecScanSettled(s.db);
    const reloaded = await vecScanSearch(s.db, fresh, 1);
    expect(reloaded).not.toBeNull();
    expect(reloaded![0]!.rowid).toBe(knn(s, fresh, 1)[0]!.rowid);
  });

  test("a write on the store's own connection is seen too", async () => {
    const s = await openStore();
    const vec = vectors(6);
    seed(s, "docs", 200, vec);
    await enabled(s);
    const fresh = vec();
    seed(s, "docs2", 1, () => fresh);
    expect(await vecScanSearch(s.db, fresh, 1)).toBeNull();
    await vecScanSettled(s.db);
    expect((await vecScanSearch(s.db, fresh, 1))![0]!.rowid).toBe(knn(s, fresh, 1)[0]!.rowid);
  });

  test("QMD_VEC_SCAN=vec0 leaves search on sqlite-vec", async () => {
    process.env.QMD_VEC_SCAN = "vec0";
    const s = await openStore();
    const vec = vectors(7);
    seed(s, "docs", 50, vec);
    await enabled(s);
    expect(await vecScanSearch(s.db, vec(), 5)).toBeNull();
  });

  test("disabling drops the matrix", async () => {
    const s = await openStore();
    const vec = vectors(8);
    seed(s, "docs", 50, vec);
    await enabled(s);
    expect(await vecScanSearch(s.db, vec(), 5)).not.toBeNull();
    disableVecScan(s.db);
    expect(await vecScanSearch(s.db, vec(), 5)).toBeNull();
  });
});
