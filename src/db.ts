/**
 * db.ts - Cross-runtime SQLite compatibility layer + backend selection.
 *
 * Provides a unified Database export that works under both Bun (bun:sqlite)
 * and Node.js (better-sqlite3). The APIs are nearly identical — the main
 * difference is the import path.
 *
 * Backend selection (QMD_BACKEND env var):
 *   - 'sqlite' (default): uses SQLite via bun:sqlite or better-sqlite3
 *   - 'postgres': uses PostgreSQL via pg-worker + Atomics sync wrapper
 */

export const isBun = typeof globalThis.Bun !== "undefined";

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

export type Backend = 'sqlite' | 'postgres';

/**
 * Return the active backend. Reads QMD_BACKEND env; defaults to 'sqlite'.
 */
export function getBackend(): Backend {
  const v = process.env.QMD_BACKEND;
  if (v === 'postgres') return 'postgres';
  return 'sqlite';
}

// Loaded eagerly so backend can be switched before createStore() in tests.
let _openPgDatabase: ((url: string) => Database) | null = null;

try {
  const pg = await import('./pg.js');
  _openPgDatabase = pg.openPgDatabase;
} catch (err) {
  if (getBackend() === 'postgres') {
    throw err;
  }
}

/**
 * Open a PostgreSQL database. Returns a Database-compatible object that
 * wraps a worker thread + Atomics for synchronous-looking access.
 * Only available when QMD_BACKEND=postgres.
 */
export function openPgDatabase(url: string): Database {
  if (!_openPgDatabase) {
    throw new Error('PostgreSQL backend is unavailable in this runtime.');
  }
  return _openPgDatabase(url);
}

let _Database: any;
let _sqliteVecLoad: (db: any) => void;

if (isBun) {
  // Dynamic string prevents tsc from resolving bun:sqlite on Node.js builds
  const bunSqlite = "bun:" + "sqlite";
  _Database = (await import(/* @vite-ignore */ bunSqlite)).Database;
  const { getLoadablePath } = await import("sqlite-vec");
  _sqliteVecLoad = (db: any) => db.loadExtension(getLoadablePath());
} else {
  _Database = (await import("better-sqlite3")).default;
  const sqliteVec = await import("sqlite-vec");
  _sqliteVecLoad = (db: any) => sqliteVec.load(db);
}

/**
 * Open a SQLite database. Works with both bun:sqlite and better-sqlite3.
 */
export function openDatabase(path: string): Database {
  return new _Database(path) as Database;
}

/**
 * Common subset of the Database interface used throughout QMD.
 */
export interface Database {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  loadExtension(path: string): void;
  close(): void;
}

export interface Statement {
  run(...params: any[]): { changes: number; lastInsertRowid: number | bigint };
  get(...params: any[]): any;
  all(...params: any[]): any[];
}

/**
 * Load the sqlite-vec extension into a database.
 */
export function loadSqliteVec(db: Database): void {
  _sqliteVecLoad(db);
}
