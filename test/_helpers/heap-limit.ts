import type { Database } from "../../src/db.js";
import { isBun } from "../../src/db.js";

/**
 * Whether SQLite enforces the hard heap limit just set on `db`: an allocation
 * past it fails with SQLITE_NOMEM only when SQLite tracks memory. Bun's bundled
 * SQLite on Linux did before Bun 1.4.4 (see bunSqliteTracksMemory);
 * better-sqlite3 (DEFAULT_MEMSTATUS=0) does not, so a heap-budget test proves
 * its budget only where this returns true.
 */
export function heapLimitBinds(db: Database, limitBytes: number): boolean {
  try {
    db.prepare(`SELECT length(randomblob(?)) AS n`).get(limitBytes + 8 * 1024 * 1024);
    return false;
  } catch {
    return true;
  }
}

/**
 * Whether Bun's SQLite tracks memory, so that heapLimitBinds can return true.
 * Bun 1.4.4 calls sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0) before the first
 * open, for its bundled SQLite and a setCustomSQLite one alike, and without the
 * statistics the hard heap limit never binds. If a later Bun turns them back
 * on, the budget tests' limitBinds check fails, which is the signal to bound
 * this version range. The `-0` makes 1.4.4 prereleases count as 1.4.4.
 */
export const bunSqliteTracksMemory = isBun && !Bun.semver.satisfies(Bun.version, ">=1.4.4-0");
