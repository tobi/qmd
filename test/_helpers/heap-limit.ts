import type { Database } from "../../src/db.js";

/**
 * Whether SQLite enforces the hard heap limit just set on `db`: an allocation
 * past it fails with SQLITE_NOMEM only when SQLite tracks memory. Bun's bundled
 * SQLite on Linux does; better-sqlite3 (DEFAULT_MEMSTATUS=0) does not, so a
 * heap-budget test proves its budget only where this returns true.
 */
export function heapLimitBinds(db: Database, limitBytes: number): boolean {
  try {
    db.prepare(`SELECT length(randomblob(?)) AS n`).get(limitBytes + 8 * 1024 * 1024);
    return false;
  } catch {
    return true;
  }
}
