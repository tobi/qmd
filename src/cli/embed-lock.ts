/**
 * Process-level exclusive lock for `qmd embed`.
 *
 * Concurrent embed runs against the same index race on vector_rows (UNIQUE
 * constraint on hash, seq, collection_id, #825), so a second process must skip
 * while the first one is embedding.
 *
 * Ownership is an exclusive SQLite write transaction held open for the whole
 * embed body, on a sidecar database next to the index
 * (`.qmd-embed.lock.sqlite` — never the index database itself, whose write
 * lock must stay available to `update`/`query`). SQLite takes the write lock
 * inside `BEGIN EXCLUSIVE` before that statement returns, and the kernel drops
 * it when the process exits or dies, so:
 *
 *   - ownership does not exist in a half-written state. The lock is a single
 *     client-side statement, so there is no window in which a live acquirer
 *     looks unowned and can be reclaimed by a competitor;
 *   - recovery needs neither PID checks nor deletes. A crashed holder's
 *     transaction is rolled back by the next opener, and a recycled PID is
 *     irrelevant because process identity is never part of the protocol;
 *   - nothing unlinks the sidecar, so a reclaiming process cannot delete a
 *     freshly acquired live lock by pathname, and two processes can never end
 *     up holding different inodes for the same lock.
 *
 * The sidecar (plus the `-wal`/`-shm` siblings a WAL database keeps) is created
 * on demand and deliberately left in place; removing it is only safe while no
 * embed runs. It lives in the index directory, which is already a WAL
 * location, so it has the same filesystem requirements as the index.
 *
 * Acquisition is fail-fast: a short busy timeout only absorbs an opener's
 * momentary lock install, so a second embed still skips straight away instead
 * of queueing behind hours of embedding.
 *
 * Legacy pidfile (`.qmd-embed.lock`, qmd <= 2.8.x): older versions wrote their
 * PID with an O_EXCL create and reclaimed the file by unlink-then-retry, which
 * had two races the sidecar makes impossible — the zero-byte window between
 * create and write (a competitor treats the half-written file as stale), and
 * two reclaimers each unlinking by pathname (the second removes a lock that
 * had just been acquired live). This module never creates and never takes
 * ownership from the pidfile, but it still reads it so an older qmd that is
 * embedding right now is respected: a pidfile naming a live qmd process means
 * "busy", a stale one is removed, and an empty/unparsable one — an older
 * version preempted or killed between create and write — is treated as owned
 * for a short settle window before being treated as abandoned. Legacy
 * handling is advisory and fail-safe: it runs only while this process owns the
 * sidecar, every error in it is swallowed, and the worst case is a leftover
 * pidfile — it can neither block nor fail an otherwise valid acquisition.
 *
 * Limits of mixed versions: an older qmd *started* while a newer one embeds
 * does not see the sidecar and never writes it, so it can still run an embed
 * body concurrently with a new version; and a new version skips whenever the
 * pidfile names a live qmd process, even one that is not embedding (command
 * line, not intent, is all a stale pidfile can tell us). New versions never
 * write the pidfile, so old-version detection of a new holder is impossible by
 * construction. Treat "old and new qmd on the same index" as unsupported; the
 * pidfile read only exists so a half-finished upgrade does not immediately
 * double-run.
 */

import { readFileSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { isBusyError, openDatabase, sleepSync, type Database } from "../db.js";
import { isQmdMcpPid, parsePidfileContent } from "./mcp-pid.js";

/**
 * Lock lifecycle (explicit, per handle):
 *
 *   free      → nobody holds the sidecar write transaction
 *   acquired  → this process holds it; `EmbedLockHandle.release` ends it
 *   released  → terminal; the transaction is closed and cannot be re-held
 *
 * `busy` is the other acquisition outcome: some other live holder owns it (or
 * an older qmd still owns the legacy pidfile).
 */
export type EmbedLockHandle = {
  /** Legacy pidfile path the caller asked to lock (API compatibility). */
  lockPath: string;
  /** Sidecar database whose open write transaction *is* the lock. */
  statePath: string;
  /** End ownership. Idempotent; leaves the sidecar file on disk. */
  release: () => void;
};

/** Which artifact made an acquisition attempt lose. */
export type EmbedLockBusyReason =
  /** Another process holds the sidecar write transaction (the normal case). */
  | { kind: "sidecar" }
  /** An older qmd still owns the legacy pidfile; it is deliberately respected. */
  | { kind: "legacy"; pid: number };

/** Outcome of one acquisition attempt. */
export type EmbedLockAcquireResult =
  | { status: "acquired"; handle: EmbedLockHandle }
  | { status: "busy"; reason: EmbedLockBusyReason };

/**
 * Busy budget for taking the sidecar write lock. Long enough to ride out a
 * competing opener's WAL install, short enough that a second embed skips
 * immediately rather than waiting on an hours-long embed.
 */
const ACQUIRE_BUSY_TIMEOUT_MS = 250;

/**
 * How long an empty/unparsable legacy pidfile is treated as "its writer is
 * mid-create". Older qmd wrote the PID microseconds after the O_EXCL create.
 */
const LEGACY_SETTLE_TIMEOUT_MS = 100;
const LEGACY_SETTLE_POLL_MS = 10;

/** Legacy pidfile path sibling to the index database. */
export function embedLockPathForDb(dbPath: string): string {
  return join(dirname(dbPath), ".qmd-embed.lock");
}

/** Sidecar database that carries the real lock for `lockPath`. */
export function embedLockStatePath(lockPath: string): string {
  return `${lockPath}.sqlite`;
}

/** True if `pid` still owns a live embed/qmd process (or is this process). */
export function isLiveEmbedLockHolder(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  // Same-process re-check: we obviously still hold our own lock.
  if (pid === process.pid) return true;
  return isQmdMcpPid(pid);
}

function errCode(err: unknown): string | undefined {
  if (typeof err !== "object" || err === null || !("code" in err)) return undefined;
  const code = err.code;
  return typeof code === "string" ? code : undefined;
}

type LegacySnapshot =
  | { kind: "absent" }
  | { kind: "unreadable" }
  | { kind: "text"; raw: string; pid: number | null };

function snapshotLegacyLock(lockPath: string): LegacySnapshot {
  let raw: string;
  try {
    raw = readFileSync(lockPath, "utf-8");
  } catch (err) {
    return errCode(err) === "ENOENT" ? { kind: "absent" } : { kind: "unreadable" };
  }
  return { kind: "text", raw, pid: parsePidfileContent(raw) };
}

/**
 * Remove a legacy pidfile, but only the exact content that was judged stale: an
 * older qmd may have replaced it in between, and that replacement could name a
 * live holder. Always called while this process owns the sidecar, so no other
 * new-version process can be reclaiming the same file concurrently.
 */
function deleteLegacyLockIfUnchanged(lockPath: string, seen: string): void {
  try {
    if (readFileSync(lockPath, "utf-8") !== seen) return;
    unlinkSync(lockPath);
  } catch {
    // Best effort: a leftover pidfile never constrains a sidecar owner.
  }
}

/**
 * Decide what a legacy pidfile means now that we own the sidecar:
 *   - "live":   an older qmd holds it; the caller must back off.
 *   - "stale":  abandoned (dead/recycled PID, or unreadable past the settle
 *               window); removed when possible and ignored.
 *   - "absent": nothing to migrate.
 *
 * Never throws: the pidfile is advisory, so an unreadable file, a failed
 * unlink, or a vanished path must not fail an acquisition that already owns
 * the sidecar. The worst case is a leftover pidfile that is re-examined on the
 * next acquisition.
 */
function settleLegacyLock(
  lockPath: string,
): { kind: "absent" | "stale" } | { kind: "live"; pid: number } {
  const deadline = Date.now() + LEGACY_SETTLE_TIMEOUT_MS;
  for (;;) {
    const snap = snapshotLegacyLock(lockPath);
    if (snap.kind === "absent") return { kind: "absent" };

    if (snap.kind === "text" && snap.pid !== null) {
      if (isLiveEmbedLockHolder(snap.pid)) return { kind: "live", pid: snap.pid };
      deleteLegacyLockIfUnchanged(lockPath, snap.raw);
      return { kind: "stale" };
    }

    // Empty, unparsable, or unreadable: an older qmd may be between its
    // O_EXCL create and its PID write, or may have died there. Treat it as
    // owned until the settle window expires, then as abandoned.
    if (Date.now() >= deadline) {
      if (snap.kind === "text") deleteLegacyLockIfUnchanged(lockPath, snap.raw);
      return { kind: "stale" };
    }
    sleepSync(LEGACY_SETTLE_POLL_MS);
  }
}

/**
 * Open (creating if needed) the sidecar and pick the lock-acquisition budget.
 *
 * The open path cannot meet a *held* lock in a stopped journal mode: this
 * module installs WAL before it takes ownership, and `PRAGMA journal_mode =
 * WAL` on an already-WAL database is a no-op that does not contend, so
 * `openDatabase`'s generous data-write budget never applies to acquiring a
 * busy lock. It is overridden anyway, so ownership is decided in
 * milliseconds rather than after a data-write timeout.
 */
function openLockState(statePath: string): Database {
  const db = openDatabase(statePath);
  db.exec(`PRAGMA busy_timeout = ${ACQUIRE_BUSY_TIMEOUT_MS}`);
  return db;
}

/** Transfer ownership: the write lock lives exactly as long as this transaction. */
function beginOwnership(db: Database): void {
  db.exec("BEGIN EXCLUSIVE");
}

/** End ownership; releases the write lock that the open transaction holds. */
function endOwnership(db: Database): void {
  if (!db.inTransaction) return;
  db.exec("ROLLBACK");
}

function createHandle(lockPath: string, statePath: string, db: Database): EmbedLockHandle {
  let state: "owned" | "released" = "owned";
  return {
    lockPath,
    statePath,
    release() {
      if (state === "released") return;
      state = "released";
      // Rollback ends ownership; closing drops the connection (and would
      // release the lock anyway, which is also what a crash does). The sidecar
      // file is deliberately left on disk: unlinking it would let a concurrent
      // acquirer create a different inode while a straggler still holds the
      // old one.
      try {
        endOwnership(db);
      } catch {
        // Broken connection: closing below still releases the lock.
      }
      try {
        db.close();
      } catch {
        // Best effort.
      }
    },
  };
}

/**
 * Try to acquire the exclusive embed lock at `lockPath`.
 *
 * Returns `{ status: "acquired", handle }` on success or
 * `{ status: "busy", reason }` if another live holder owns the lock. Throws
 * only on unexpected I/O or database errors.
 */
export function acquireEmbedLock(lockPath: string): EmbedLockAcquireResult {
  const statePath = embedLockStatePath(lockPath);
  let db: Database | null = null;
  let ownsTransaction = false;
  try {
    db = openLockState(statePath);
    beginOwnership(db);
    ownsTransaction = true;

    const legacy = settleLegacyLock(lockPath);
    if (legacy.kind === "live") {
      endOwnership(db);
      ownsTransaction = false;
      db.close();
      return { status: "busy", reason: { kind: "legacy", pid: legacy.pid } };
    }

    return { status: "acquired", handle: createHandle(lockPath, statePath, db) };
  } catch (err) {
    if (db) {
      if (ownsTransaction) {
        try {
          endOwnership(db);
        } catch {
          // Closing below releases whatever is left.
        }
      }
      try {
        db.close();
      } catch {
        // Best effort.
      }
    }
    // Contention while installing the journal mode counts as "someone else is
    // acquiring": skipping this run is the same answer as losing the lock.
    if (isBusyError(err)) return { status: "busy", reason: { kind: "sidecar" } };
    throw err;
  }
}

/**
 * Try to acquire an exclusive embed lock at `lockPath`.
 * Returns a handle with `release()` on success, or `null` if another live
 * qmd process already holds the lock.
 */
export function tryAcquireEmbedLock(lockPath: string): EmbedLockHandle | null {
  const result = acquireEmbedLock(lockPath);
  return result.status === "acquired" ? result.handle : null;
}

/** User-facing message when a second embed is skipped. */
export const EMBED_LOCK_BUSY_MESSAGE =
  "Another embed process is already running. Skipping.";
