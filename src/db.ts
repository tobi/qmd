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

import {
  createSqliteVecUnavailableError,
  resolveSqliteVecLoadablePath,
} from "./platform/sqlite-vec.js";

const bunGlobal = globalThis as typeof globalThis & { Bun?: unknown };
export const isBun = typeof bunGlobal.Bun !== "undefined";

let _Database: any;
let _sqliteVecLoad: ((db: any) => void) | null;
let _sqliteVecUnavailableReason: string | null = null;

function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

const sqliteVec = resolveSqliteVecLoadablePath();
const sqliteVecPath = sqliteVec.path;

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
  if (sqliteVecPath) {
    try {
      const testDb = new BunDatabase(":memory:");
      testDb.loadExtension(sqliteVecPath);
      testDb.close();
      _sqliteVecLoad = (db: any) => db.loadExtension(sqliteVecPath);
      _sqliteVecUnavailableReason = null;
    } catch (err) {
      // Vector search won't work, but BM25 and other operations are unaffected.
      _sqliteVecLoad = null;
      _sqliteVecUnavailableReason = `sqlite-vec probe failed (${getErrorMessage(err)})`;
    }
  } else {
    _sqliteVecLoad = null;
    _sqliteVecUnavailableReason = "No loadable sqlite-vec extension was found";
  }
} else {
  _Database = (await import("better-sqlite3")).default;
  if (sqliteVecPath) {
    _sqliteVecLoad = (db: any) => db.loadExtension(sqliteVecPath);
    _sqliteVecUnavailableReason = null;
  } else {
    _sqliteVecLoad = null;
    _sqliteVecUnavailableReason = "No loadable sqlite-vec extension was found";
  }
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
 *
 * Throws with platform-specific fix instructions when the extension is
 * unavailable.
 */
export function loadSqliteVec(db: Database): void {
  if (!_sqliteVecLoad) {
    throw createSqliteVecUnavailableError(
      _sqliteVecUnavailableReason ?? "No loadable sqlite-vec extension was found",
      { isBun },
    );
  }
  try {
    _sqliteVecLoad(db);
  } catch (err) {
    throw createSqliteVecUnavailableError(
      `sqlite-vec load failed (${getErrorMessage(err)})`,
      { isBun },
    );
  }
}
