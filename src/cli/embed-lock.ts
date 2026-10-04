/**
 * Process-level exclusive lock for `qmd embed`.
 *
 * Concurrent embed runs against the same index can race on vectors_vec
 * (UNIQUE constraint on hash_seq). This lockfile keeps a second process from
 * starting while another embed holds the lock. Stale files left by crashed
 * processes are recovered via PID identity checks (same spirit as mcp-pid.ts).
 *
 * A live PID is not proof of a live embed. The embed session cap
 * (`--timeout`, default 30 minutes) is a JavaScript timer: it cannot fire
 * while the event loop is blocked inside a native node-llama-cpp call, so a
 * process wedged in Metal/CUDA keeps its PID, keeps the lock, and every later
 * `qmd embed` skips itself with "Another embed process is already running"
 * for as long as the zombie lives (observed: 39 hours, #735). The lock
 * therefore also records when it was taken and the cap the holder promised
 * to honour; a holder that has outlived twice its cap is treated as stale
 * even though its PID still answers.
 */

import { existsSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { isQmdMcpPid } from "./mcp-pid.js";

export type EmbedLockHandle = {
  lockPath: string;
  release: () => void;
  /** Set when a live-but-overdue holder was evicted to take this lock. */
  reclaimedFrom?: EmbedLockRecord;
};

/** What the lockfile records about its holder. */
export type EmbedLockRecord = {
  pid: number;
  /** Epoch milliseconds when the holder took the lock. */
  startedAt: number;
  /** The holder's embed session cap in ms; 0 = the holder declared no cap. */
  maxDurationMs: number;
  /** True when the file was written by an older qmd that only stored the PID. */
  legacy: boolean;
};

export type EmbedLockOptions = {
  /**
   * This caller's embed session cap in ms (0 = no cap). Recorded in the lock
   * so later callers can judge staleness by the promise the holder made.
   * Defaults to {@link DEFAULT_EMBED_LOCK_MAX_DURATION_MS}.
   */
  maxDurationMs?: number;
  /** Clock override for tests. */
  now?: () => number;
};

/**
 * Mirrors DEFAULT_EMBED_MAX_DURATION_MS in store.ts. Kept local so the lock
 * module stays free of the store's SQLite/model dependencies.
 */
export const DEFAULT_EMBED_LOCK_MAX_DURATION_MS = 30 * 60 * 1000;

/** A holder that declared no session cap is still evicted after this long. */
export const EMBED_LOCK_UNCAPPED_MAX_AGE_MS = 24 * 60 * 60 * 1000;

/** Lockfile path sibling to the index database. */
export function embedLockPathForDb(dbPath: string): string {
  return join(dirname(dbPath), ".qmd-embed.lock");
}

/**
 * How long a holder may keep the lock before it counts as stale: twice the
 * cap it promised to honour, or {@link EMBED_LOCK_UNCAPPED_MAX_AGE_MS} when it
 * promised none. Twice, so a holder that is merely slow to notice its own
 * abort is not evicted while it is still winding down.
 */
export function embedLockMaxAgeMs(maxDurationMs: number): number {
  if (!Number.isFinite(maxDurationMs) || maxDurationMs <= 0) {
    return EMBED_LOCK_UNCAPPED_MAX_AGE_MS;
  }
  return 2 * maxDurationMs;
}

function parseRecord(raw: string, fileMtimeMs: number, fallbackMaxDurationMs: number): EmbedLockRecord | null {
  const text = raw.trim();
  if (text === "") return null;

  if (text.startsWith("{")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return null;
    }
    if (typeof parsed !== "object" || parsed === null) return null;
    const pid = (parsed as { pid?: unknown }).pid;
    const startedAt = (parsed as { startedAt?: unknown }).startedAt;
    const maxDurationMs = (parsed as { maxDurationMs?: unknown }).maxDurationMs;
    if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return null;
    return {
      pid,
      startedAt: typeof startedAt === "number" && Number.isFinite(startedAt) ? startedAt : fileMtimeMs,
      maxDurationMs:
        typeof maxDurationMs === "number" && Number.isFinite(maxDurationMs) && maxDurationMs >= 0
          ? maxDurationMs
          : fallbackMaxDurationMs,
      legacy: false,
    };
  }

  // Pre-2.9 format: the bare PID. Age comes from the file's mtime and the cap
  // from this caller, since the holder recorded neither.
  const pid = parseInt(text, 10);
  if (!Number.isInteger(pid) || pid <= 0) return null;
  return { pid, startedAt: fileMtimeMs, maxDurationMs: fallbackMaxDurationMs, legacy: true };
}

/** Read and parse the lockfile; null if missing, unreadable, or malformed. */
export function readEmbedLockRecord(
  lockPath: string,
  fallbackMaxDurationMs: number = DEFAULT_EMBED_LOCK_MAX_DURATION_MS,
): EmbedLockRecord | null {
  try {
    const raw = readFileSync(lockPath, "utf-8");
    const mtimeMs = statSync(lockPath).mtimeMs;
    return parseRecord(raw, mtimeMs, fallbackMaxDurationMs);
  } catch {
    return null;
  }
}

/** True if `pid` still owns a live embed/qmd process (or is this process). */
export function isLiveEmbedLockHolder(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  // Same-process re-check: we obviously still hold our own lock.
  if (pid === process.pid) return true;
  return isQmdMcpPid(pid);
}

/** True when the holder has held the lock longer than its cap allows. */
export function isOverdueEmbedLockHolder(record: EmbedLockRecord, nowMs: number): boolean {
  const ageMs = nowMs - record.startedAt;
  return ageMs > embedLockMaxAgeMs(record.maxDurationMs);
}

function createOwnedLock(lockPath: string, maxDurationMs: number, nowMs: number): EmbedLockHandle {
  const record = { pid: process.pid, startedAt: nowMs, maxDurationMs };
  writeFileSync(lockPath, `${JSON.stringify(record)}\n`, { flag: "wx" });
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    try {
      if (!existsSync(lockPath)) return;
      const written = readEmbedLockRecord(lockPath, maxDurationMs);
      if (written?.pid === process.pid) unlinkSync(lockPath);
    } catch {
      // best-effort cleanup
    }
  };
  return { lockPath, release };
}

/**
 * Try to acquire an exclusive embed lock at `lockPath`.
 * Returns a handle with `release()` on success, or `null` if another live
 * qmd process already holds the lock and is within its time budget. A live
 * holder that has outlived twice its declared cap is evicted; the returned
 * handle then carries `reclaimedFrom` so the caller can warn about it.
 */
export function tryAcquireEmbedLock(lockPath: string, options: EmbedLockOptions = {}): EmbedLockHandle | null {
  const maxDurationMs = options.maxDurationMs ?? DEFAULT_EMBED_LOCK_MAX_DURATION_MS;
  const now = options.now ?? Date.now;
  let reclaimedFrom: EmbedLockRecord | undefined;

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const handle = createOwnedLock(lockPath, maxDurationMs, now());
      if (reclaimedFrom) handle.reclaimedFrom = reclaimedFrom;
      return handle;
    } catch (err: unknown) {
      const code =
        typeof err === "object" && err !== null && "code" in err
          ? (err as NodeJS.ErrnoException).code
          : undefined;
      if (code !== "EEXIST") throw err;

      const holder = readEmbedLockRecord(lockPath, maxDurationMs);
      if (holder !== null && isLiveEmbedLockHolder(holder.pid)) {
        if (!isOverdueEmbedLockHolder(holder, now())) {
          return null;
        }
        reclaimedFrom = holder;
      }

      // Stale / unreadable / recycled PID / overdue holder — remove and retry once.
      try {
        unlinkSync(lockPath);
      } catch {
        // Another process may have claimed it; loop and try wx again.
      }
    }
  }

  // Final attempt lost the race to a live holder (or repeated EEXIST).
  return null;
}

/** User-facing message when a second embed is skipped. */
export const EMBED_LOCK_BUSY_MESSAGE =
  "Another embed process is already running. Skipping.";

/** User-facing warning when an overdue holder was evicted. */
export function embedLockReclaimedMessage(record: EmbedLockRecord, nowMs: number = Date.now()): string {
  const ageMinutes = Math.round((nowMs - record.startedAt) / 60_000);
  const capMinutes = Math.round(embedLockMaxAgeMs(record.maxDurationMs) / 60_000);
  return (
    `Reclaimed the embed lock from process ${record.pid}: it has held it for ${ageMinutes} min, ` +
    `past its ${capMinutes} min limit. If that process is still running it is stuck in a native call; kill it.`
  );
}
