/**
 * vec-scan.ts - Exact vector search over an in-memory copy of the vector table.
 *
 * sqlite-vec's vec0 table has no approximate index, so every KNN query reads
 * every stored vector of its partition out of SQLite pages; on a large index
 * that read, not the arithmetic, is what a vector search costs. A long-running
 * server can instead hold the vectors in one contiguous Float32Array and scan
 * it with a pool of worker threads. The scan is exact (the cosine distance
 * vec0 computes), so a search returns the same rows, only sooner.
 *
 * vec0 stays the storage of record. The matrix is a cache of it, read over a
 * second connection, and is used only while it matches the index: before each
 * search the store's connection is asked whether anything was committed since
 * the last check (`PRAGMA data_version` for other connections' commits,
 * `total_changes()` for its own); if so, a fingerprint of the vector tables is
 * compared with the one the matrix was read at. A stale or missing matrix sends
 * the search to vec0 and starts a reload, so a server never answers from
 * vectors the index no longer holds.
 *
 * Each row keeps its partition (collection id), so a collection-scoped scan
 * masks other collections out before it takes the top k, as vec0 does.
 *
 * Costs 4 bytes per dimension per stored row. QMD_VEC_SCAN=vec0 turns it off;
 * QMD_VEC_SCAN_THREADS sets the pool size. If the matrix cannot be allocated,
 * search stays on vec0 and says so on stderr.
 */

import { Worker } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { openDatabase, loadSqliteVec, type Database } from "./db.js";
import { VEC_ROWS_TABLE, VEC_TABLE, vecLayout } from "./vec-layout.js";

/** Below this many rows a scan runs on the calling thread; a worker round trip costs more. */
const INLINE_SCAN_ROWS = 32_768;
const DEFAULT_MAX_THREADS = 16;
/** Rows the loader reads between yields to the event loop. */
const LOAD_YIELD_ROWS = 5_000;

export interface VecScanMatch {
  rowid: number;
  distance: number;
}

interface MatrixBuffers {
  dims: number;
  rows: number;
  vectors: SharedArrayBuffer;
  invNorms: SharedArrayBuffer;
  partitions: SharedArrayBuffer;
}

interface Matrix extends MatrixBuffers {
  fingerprint: string;
  rowids: Float64Array;
  vectorView: Float32Array;
  invNormView: Float32Array;
  partitionView: Int32Array;
}

interface ScanResult {
  rows: Int32Array;
  dists: Float64Array;
}

function log(message: string): void {
  process.stderr.write(`qmd vec-scan: ${message}\n`);
}

/**
 * A fingerprint of everything the matrix is read from. Any embed, re-embed,
 * clear, copy into a new collection or cleanup moves at least one value:
 * row count, largest and summed row id of the partition mapping, and the
 * newest embedding time (a re-embed can reuse a row id with a new vector).
 */
export function vecFingerprint(db: Database): string {
  const v = db.prepare(`SELECT COUNT(*) AS n, MAX(id) AS m, TOTAL(id) AS s FROM ${VEC_ROWS_TABLE}`).get() as { n: number; m: number | null; s: number };
  const c = db.prepare(`SELECT MAX(embedded_at) AS t FROM content_vectors`).get() as { t: string | null };
  return `${v.n}|${v.m}|${v.s}|${c.t}`;
}

/**
 * Exact cosine top-k over rows [from, to) of a matrix. `partition`, when not
 * null, masks out every row of another collection before ranking. Returns row
 * indices (-1 past the end) and distances, nearest first.
 *
 * Kept free of closures and imports: the pool's workers run its source text.
 */
function scanRange(
  vectors: Float32Array, invNorms: Float32Array, partitions: Int32Array, dims: number,
  query: Float32Array, k: number, partition: number | null, from: number, to: number,
): ScanResult {
  let qNorm = 0;
  for (let j = 0; j < dims; j++) qNorm += query[j]! * query[j]!;
  const qInv = qNorm > 0 ? 1 / Math.sqrt(qNorm) : 0;
  const bestDist = new Float64Array(k).fill(Infinity);
  const bestRow = new Int32Array(k).fill(-1);
  let worst = Infinity;
  for (let i = from; i < to; i++) {
    if (partition !== null && partitions[i] !== partition) continue;
    const o = i * dims;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let j = 0;
    for (; j + 3 < dims; j += 4) {
      s0 += vectors[o + j]! * query[j]!;
      s1 += vectors[o + j + 1]! * query[j + 1]!;
      s2 += vectors[o + j + 2]! * query[j + 2]!;
      s3 += vectors[o + j + 3]! * query[j + 3]!;
    }
    for (; j < dims; j++) s0 += vectors[o + j]! * query[j]!;
    const dist = 1 - (s0 + s1 + s2 + s3) * invNorms[i]! * qInv;
    if (dist >= worst) continue;
    let p = k - 1;
    while (p > 0 && bestDist[p - 1]! > dist) {
      bestDist[p] = bestDist[p - 1]!;
      bestRow[p] = bestRow[p - 1]!;
      p--;
    }
    bestDist[p] = dist;
    bestRow[p] = i;
    worst = bestDist[k - 1]!;
  }
  return { rows: bestRow, dists: bestDist };
}

// Evaluated as a worker's source, so it runs the same from src/ (TypeScript,
// under Bun or a test runner) and from the compiled dist/: either way
// `scanRange.toString()` is the function as the runtime holds it, types gone.
const WORKER_SOURCE = `
const { parentPort } = require("node:worker_threads");
const scanRange = ${scanRange.toString()};
let m = null;
parentPort.on("message", (msg) => {
  if (msg.type === "matrix") {
    const b = msg.buffers;
    m = { dims: b.dims, vectors: new Float32Array(b.vectors), invNorms: new Float32Array(b.invNorms), partitions: new Int32Array(b.partitions) };
    return;
  }
  const r = scanRange(m.vectors, m.invNorms, m.partitions, m.dims, msg.query, msg.k, msg.partition, msg.from, msg.to);
  parentPort.postMessage({ id: msg.id, rows: r.rows, dists: r.dists }, [r.rows.buffer, r.dists.buffer]);
});
`;

class ScanPool {
  readonly workers: Worker[] = [];
  private pending = new Map<number, { resolve: (r: ScanResult) => void; reject: (e: unknown) => void }>();
  private nextId = 0;

  constructor(size: number) {
    for (let w = 0; w < size; w++) {
      // An eval worker runs its source as CommonJS, so `require` is there.
      const worker = new Worker(WORKER_SOURCE, { eval: true });
      worker.unref();
      worker.on("message", (msg: { id: number } & ScanResult) => {
        const done = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        done?.resolve(msg);
      });
      worker.on("error", (err) => {
        for (const done of this.pending.values()) done.reject(err);
        this.pending.clear();
      });
      this.workers.push(worker);
    }
  }

  install(buffers: MatrixBuffers): void {
    for (const worker of this.workers) worker.postMessage({ type: "matrix", buffers });
  }

  scan(rows: number, query: Float32Array, k: number, partition: number | null): Promise<ScanResult[]> {
    const size = this.workers.length;
    return Promise.all(this.workers.map((worker, w) => new Promise<ScanResult>((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      worker.postMessage({
        type: "scan", id, query, k, partition,
        from: Math.floor((w * rows) / size),
        to: Math.floor(((w + 1) * rows) / size),
      });
    })));
  }

  close(): void {
    for (const worker of this.workers) void worker.terminate();
  }
}

interface ScanState {
  dbPath: string;
  changeKey: string;
  fingerprint: string;
  matrix: Matrix | null;
  loading: Promise<void> | null;
  failedAt: string | null;
  pool: ScanPool | null;
  poolFailed: boolean;
  closed: boolean;
}

const states = new WeakMap<Database, ScanState>();

function scanThreads(): number {
  const configured = Number(process.env.QMD_VEC_SCAN_THREADS);
  if (Number.isInteger(configured) && configured > 0) return configured;
  return Math.max(1, Math.min(DEFAULT_MAX_THREADS, availableParallelism()));
}

/**
 * Serve vector searches on `db` from an in-memory matrix, loaded now in the
 * background. Meant for long-running servers: a one-shot CLI search is
 * cheaper on vec0 than reading every vector first. Until the matrix is
 * loaded, and whenever it is stale, searches use vec0.
 */
export function enableVecScan(db: Database, dbPath: string): void {
  if (states.has(db)) return;
  if ((process.env.QMD_VEC_SCAN ?? "").toLowerCase() === "vec0") {
    log("disabled by QMD_VEC_SCAN=vec0; vector search uses sqlite-vec");
    return;
  }
  const state: ScanState = {
    dbPath, changeKey: "", fingerprint: "", matrix: null, loading: null,
    failedAt: null, pool: null, poolFailed: false, closed: false,
  };
  states.set(db, state);
  if (vecLayout(db).kind !== "partitioned") return;
  checkFreshness(db, state);
  startLoad(db, state);
}

/** Drop the matrix and its workers; searches on `db` go to vec0 from here on. */
export function disableVecScan(db: Database): void {
  const state = states.get(db);
  if (!state) return;
  state.closed = true;
  state.matrix = null;
  state.pool?.close();
  state.pool = null;
  states.delete(db);
}

/** Resolves once the current load (if any) settles. For tests and warm-up. */
export async function vecScanSettled(db: Database): Promise<void> {
  const state = states.get(db);
  while (state?.loading) await state.loading;
}

function checkFreshness(db: Database, state: ScanState): void {
  const key = db.prepare(`SELECT total_changes() AS c, (SELECT data_version FROM pragma_data_version) AS v`).get() as { c: number; v: number };
  const changeKey = `${key.v}|${key.c}`;
  if (changeKey !== state.changeKey) {
    state.changeKey = changeKey;
    state.fingerprint = vecFingerprint(db);
  }
}

async function readMatrix(dbPath: string): Promise<Matrix | { error: string } | null> {
  const db = openDatabase(dbPath);
  try {
    loadSqliteVec(db);
    db.exec("PRAGMA query_only = 1");
    // One read transaction, so the fingerprint describes exactly the rows read.
    db.exec("BEGIN");
    try {
      const layout = vecLayout(db);
      if (layout.kind !== "partitioned" || !layout.dimensions) return null;
      const dims = layout.dimensions;
      const fingerprint = vecFingerprint(db);
      const capacity = (db.prepare(`SELECT COUNT(*) AS n FROM ${VEC_ROWS_TABLE}`).get() as { n: number }).n;
      let vectors: SharedArrayBuffer, invNorms: SharedArrayBuffer, partitions: SharedArrayBuffer;
      try {
        vectors = new SharedArrayBuffer(capacity * dims * 4);
        invNorms = new SharedArrayBuffer(capacity * 4);
        partitions = new SharedArrayBuffer(capacity * 4);
      } catch (err) {
        return { error: `cannot allocate ${(capacity * dims * 4 / 2 ** 30).toFixed(2)} GiB for ${capacity} vectors: ${err instanceof Error ? err.message : String(err)}` };
      }
      const vectorView = new Float32Array(vectors);
      const invNormView = new Float32Array(invNorms);
      const partitionView = new Int32Array(partitions);
      const rowids = new Float64Array(capacity);
      let n = 0;
      const iter = db.prepare(`SELECT rowid, collection_id, embedding FROM ${VEC_TABLE}`)
        .iterate() as IterableIterator<{ rowid: number; collection_id: number; embedding: Uint8Array }>;
      for (const row of iter) {
        // vector_rows maps every vec0 row; a row it does not count means the
        // mapping and the table disagree, and the matrix would be wrong.
        if (n >= capacity) {
          iter.return?.();
          return { error: `${VEC_TABLE} holds more rows than ${VEC_ROWS_TABLE}` };
        }
        const e = row.embedding;
        const src = new Float32Array(e.buffer, e.byteOffset, e.byteLength / 4);
        if (src.length !== dims) {
          iter.return?.();
          return { error: `row ${row.rowid} has ${src.length} dimensions, the table declares ${dims}` };
        }
        vectorView.set(src, n * dims);
        let sq = 0;
        for (let j = 0; j < dims; j++) sq += src[j]! * src[j]!;
        invNormView[n] = sq > 0 ? 1 / Math.sqrt(sq) : 0;
        partitionView[n] = Number(row.collection_id);
        rowids[n] = Number(row.rowid);
        n++;
        if (n % LOAD_YIELD_ROWS === 0) await new Promise<void>((resolve) => setImmediate(resolve));
      }
      return {
        fingerprint, dims, rows: n, vectors, invNorms, partitions,
        rowids: rowids.subarray(0, n), vectorView, invNormView, partitionView,
      };
    } finally {
      db.exec("COMMIT");
    }
  } finally {
    db.close();
  }
}

function startLoad(db: Database, state: ScanState): void {
  if (state.loading || state.closed || state.failedAt === state.fingerprint) return;
  const started = performance.now();
  state.loading = readMatrix(state.dbPath)
    .catch((err: unknown) => ({ error: err instanceof Error ? err.message : String(err) }))
    .then((result) => {
      state.loading = null;
      if (state.closed) return;
      if (result === null) {
        state.matrix = null;
        return;
      }
      if ("error" in result) {
        state.failedAt = state.fingerprint;
        log(`matrix not loaded, vector search stays on sqlite-vec: ${result.error}`);
        return;
      }
      if (result.rows >= INLINE_SCAN_ROWS && !state.poolFailed) {
        try {
          state.pool ??= new ScanPool(scanThreads());
          state.pool.install(result);
        } catch (err) {
          state.poolFailed = true;
          state.pool = null;
          log(`no scan threads, scanning inline: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      state.matrix = result;
      const mib = (result.rows * result.dims * 4 / 2 ** 20).toFixed(0);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      const how = state.pool && result.rows >= INLINE_SCAN_ROWS ? `${state.pool.workers.length} scan threads` : "scanned inline";
      log(`loaded ${result.rows} vectors x ${result.dims} dims (${mib} MiB) in ${secs} s, ${how}`);
      // The index may have moved on while the loader ran.
      checkFreshness(db, state);
      if (result.fingerprint !== state.fingerprint) startLoad(db, state);
    });
}

/**
 * Nearest `k` vector rows to `embedding`, restricted to one collection's
 * partition when `collectionId` is given, nearest first — the rows a vec0 KNN
 * returns. Null when no current matrix is held for `db`; the caller then
 * queries vec0.
 */
export async function vecScanSearch(db: Database, embedding: Float32Array, k: number, collectionId?: number): Promise<VecScanMatch[] | null> {
  const state = states.get(db);
  if (!state || state.closed) return null;
  checkFreshness(db, state);
  const m = state.matrix;
  if (!m || m.fingerprint !== state.fingerprint) {
    startLoad(db, state);
    return null;
  }
  if (m.rows === 0 || k <= 0) return [];
  const partition = collectionId ?? null;
  const inline = () => [scanRange(m.vectorView, m.invNormView, m.partitionView, m.dims, embedding, k, partition, 0, m.rows)];
  const pool = m.rows >= INLINE_SCAN_ROWS ? state.pool : null;
  let parts: ScanResult[];
  if (pool) {
    try {
      parts = await pool.scan(m.rows, embedding, k, partition);
    } catch (err) {
      log(`scan threads failed, scanning inline from here on: ${err instanceof Error ? err.message : String(err)}`);
      pool.close();
      if (state.pool === pool) state.pool = null;
      state.poolFailed = true;
      parts = inline();
    }
  } else {
    parts = inline();
  }
  const merged: { row: number; distance: number }[] = [];
  for (const { rows, dists } of parts) {
    for (let p = 0; p < rows.length && rows[p]! >= 0; p++) merged.push({ row: rows[p]!, distance: dists[p]! });
  }
  merged.sort((a, b) => a.distance - b.distance || a.row - b.row);
  return merged.slice(0, k).map(({ row, distance }) => ({ rowid: m.rowids[row]!, distance }));
}
