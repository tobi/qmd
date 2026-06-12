/**
 * db.ts - Cross-runtime SQLite compatibility layer
 *
 * Provides a unified Database export that works under both Bun (bun:sqlite)
 * and Node.js (better-sqlite3). The APIs are nearly identical — the main
 * difference is the import path.
 *
 * On macOS, Apple's system SQLite is compiled with SQLITE_OMIT_LOAD_EXTENSION,
 * which prevents loading native extensions like sqlite-vec. When running under
 * Bun we call Database.setCustomSQLite() to swap in Homebrew's full-featured
 * SQLite build before creating any database instances.
 */

export const isBun = "Bun" in globalThis;

export type SQLiteValue = string | number | bigint | Buffer | Uint8Array | Float32Array | null;
export type SQLiteParams = readonly SQLiteValue[];

type DatabaseConstructor = new (path: string) => Database;
type LoadableSqliteDatabase = Pick<Database, "loadExtension">;

let _Database: DatabaseConstructor;
let _sqliteVecLoad: ((db: LoadableSqliteDatabase) => void) | null;

if (isBun) {
  // Dynamic string prevents tsc from resolving bun:sqlite on Node.js builds
  const bunSqlite = "bun:" + "sqlite";
  const BunDatabase = (await import(/* @vite-ignore */ bunSqlite)).Database;

  // See: https://bun.com/docs/runtime/sqlite#setcustomsqlite
  if (process.platform === "darwin") {
    const homebrewPaths = [
      "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib",  // Apple Silicon
      "/usr/local/opt/sqlite/lib/libsqlite3.dylib",     // Intel
    ];
    for (const p of homebrewPaths) {
      try {
        BunDatabase.setCustomSQLite(p);
        break;
      } catch {}
    }
  }

  _Database = BunDatabase;

  // setCustomSQLite may have silently failed — test that extensions actually work.
  try {
    const { getLoadablePath } = await import("sqlite-vec");
    const vecPath = getLoadablePath();
    const testDb = new BunDatabase(":memory:");
    testDb.loadExtension(vecPath);
    testDb.close();
    _sqliteVecLoad = (db: LoadableSqliteDatabase) => db.loadExtension(vecPath);
  } catch {
    // Vector search won't work, but BM25 and other operations are unaffected.
    _sqliteVecLoad = null;
  }
} else {
  _Database = (await import("better-sqlite3")).default as unknown as DatabaseConstructor;
  const sqliteVec = await import("sqlite-vec");
  _sqliteVecLoad = (db: LoadableSqliteDatabase) => sqliteVec.load(db as Parameters<typeof sqliteVec.load>[0]);
}

/**
 * Open a SQLite database. Works with both bun:sqlite and better-sqlite3.
 *
 * `bun:sqlite` and `better-sqlite3` both default `busy_timeout` to 0, so
 * concurrent writers throw `SQLITE_BUSY` instead of waiting. WAL improves
 * read-while-write concurrency but does not serialise writers. Setting the
 * timeout at connection open makes parallel processes (e.g. an `update` or
 * `query` racing a long `embed`, or a first-open schema migration racing any
 * routine command) queue at batch boundaries instead of failing on contact.
 *
 * Default 120_000 ms outlasts the worst-case batch commit on a multi-GB
 * index. Override with `QMD_SQLITE_BUSY_TIMEOUT` (value in milliseconds; `0`
 * restores the upstream fail-fast behaviour). See
 * https://bun.sh/docs/api/sqlite#busy-timeout.
 */
export function openDatabase(path: string): Database {
  const db = new _Database(path) as Database;
  const raw = process.env.QMD_SQLITE_BUSY_TIMEOUT;
  const parsed = raw !== undefined && raw !== "" ? Number(raw) : Number.NaN;
  const busyTimeoutMs = Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : 120_000;
  db.exec(`PRAGMA busy_timeout = ${busyTimeoutMs}`);
  return db;
}

/**
 * Common subset of the Database interface used throughout QMD.
 */
export interface Database {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  loadExtension(path: string): void;
  transaction<T extends (...args: SQLiteValue[]) => unknown>(fn: T): T;
  close(): void;
}

export interface Statement {
  run(...params: SQLiteValue[]): { changes: number; lastInsertRowid: number | bigint };
  get<T = unknown>(...params: SQLiteValue[]): T | undefined;
  all<T = unknown>(...params: SQLiteValue[]): T[];
}

/**
 * Load the sqlite-vec extension into a database.
 *
 * Throws with platform-specific fix instructions when the extension is
 * unavailable.
 */
export function loadSqliteVec(db: Database): void {
  if (!_sqliteVecLoad) {
    const hint = isBun && process.platform === "darwin"
      ? "On macOS with Bun, install Homebrew SQLite: brew install sqlite\n" +
        "Or install qmd with npm instead: npm install -g @tobilu/qmd"
      : "Ensure the sqlite-vec native module is installed correctly.";
    throw new Error(`sqlite-vec extension is unavailable. ${hint}`);
  }
  _sqliteVecLoad(db);
}
