/**
 * Child helper for cross-process config/SQLite sync regressions.
 *
 * Args: <configPath> <dbPath> <mode>
 *
 * Both modes model one store open against the same index and wait on a barrier,
 * so the parent can land another process's mutation while the snapshot is
 * already frozen:
 *
 *   snapshot  the pre-fix createStore(): read the YAML lock-free, print READY,
 *             wait for GO, then sync that snapshot into store_collections and
 *             stamp its hash. A collection the other process published in
 *             between is deleted by the stale snapshot.
 *   locked    the current createStore(): print READY, wait for GO, then read the
 *             YAML and sync SQLite as one critical section under the config
 *             lock, so whatever the other process published is preserved.
 *
 * Output: READY before the barrier, SYNCED after the sync. Errors go to stderr
 * with a non-zero exit. The helper never calls process.exit(), which could
 * truncate buffered pipe writes.
 */
import { readSync } from "node:fs";
import { CollectionConfigScope } from "../../src/collections.ts";
import { sleepSync } from "../../src/db.ts";
import { createStore, syncConfigToDb } from "../../src/store.ts";

/**
 * Transient stdin read failures: no byte buffered yet (`EAGAIN`, with
 * `EWOULDBLOCK` as its alias), or a signal that interrupted the read.
 */
function isTransientReadError(error: unknown): boolean {
  if (typeof error !== "object" || error === null || !("code" in error)) return false;
  const code = error.code;
  return code === "EAGAIN" || code === "EWOULDBLOCK" || code === "EINTR";
}

/** Block until one line arrives on stdin (or stdin closes). */
function waitForGoSync(): void {
  const byte = Buffer.alloc(1);
  let transient = 0;
  for (;;) {
    let read: number;
    try {
      read = readSync(0, byte, 0, 1, null);
    } catch (error) {
      if (!isTransientReadError(error)) throw error;
      // Node hands the child a non-blocking stdin pipe, so a byte that has not
      // landed yet reports EAGAIN instead of waiting. This barrier runs inside
      // a synchronous callback with no event loop to yield to, so park briefly
      // between attempts instead of spinning (same backoff as db.ts retries).
      sleepSync(Math.min(5 + transient, 25));
      transient++;
      continue;
    }
    transient = 0;
    if (read === 0) return; // stdin closed: the parent hung up.
    if (byte.toString("utf-8") === "\n") return;
  }
}

function main(): void {
  const [configPath, dbPath, mode] = process.argv.slice(2);
  if (!configPath || !dbPath || (mode !== "snapshot" && mode !== "locked")) {
    console.error("usage: config-sync-worker.ts <configPath> <dbPath> snapshot|locked");
    process.exitCode = 2;
    return;
  }

  const scope = new CollectionConfigScope({ configPath });
  // Pre-fix: the store's configuration snapshot is frozen before the barrier.
  const snapshot = mode === "snapshot" ? scope.load() : undefined;
  const store = createStore(dbPath);

  process.stdout.write("READY\n");
  waitForGoSync();

  try {
    if (snapshot) syncConfigToDb(store.db, snapshot);
    else scope.withLockedConfig((config) => syncConfigToDb(store.db, config));
    process.stdout.write("SYNCED\n");
  } catch (error) {
    process.stderr.write(`ERROR ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  } finally {
    store.close();
    scope.close();
  }
}

main();
