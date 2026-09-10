/**
 * Process-level exclusive lock for `qmd embed`.
 *
 * Concurrent embed runs against the same index can race on vectors_vec
 * (UNIQUE constraint on hash_seq). This lockfile keeps a second process from
 * starting while another embed holds the lock. Stale files left by crashed
 * processes are recovered via PID identity checks (same spirit as mcp-pid.ts).
 */

import { randomUUID } from "node:crypto";
import { existsSync, linkSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  createProcessRecord,
  processRecordStatus,
  readProcessRecord,
  sameProcessRecord,
  serializeProcessRecord,
  type QmdProcessRecord,
} from "./mcp-pid.js";

export type EmbedLockHandle = {
  lockPath: string;
  release: () => void;
};

/** Lockfile path sibling to the index database. */
export function embedLockPathForDb(dbPath: string): string {
  return join(dirname(dbPath), ".qmd-embed.lock");
}

/** True only when this exact process incarnation still owns an embed lock. */
export function isLiveEmbedLockHolder(record: QmdProcessRecord): boolean {
  // Live legacy PIDs and malformed/in-flight records fail closed. A numeric
  // legacy lock becomes reclaimable only after its process is dead.
  return processRecordStatus(record, "embed") !== "dead";
}

function createOwnedLock(lockPath: string): EmbedLockHandle {
  const record = createProcessRecord("embed");
  const serialized = serializeProcessRecord(record);
  const temporaryPath = `${lockPath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temporaryPath, serialized, { flag: "wx", flush: true });
    try {
      linkSync(temporaryPath, lockPath);
    } catch (error: unknown) {
      const code = typeof error === "object" && error !== null && "code" in error
        ? (error as NodeJS.ErrnoException).code
        : undefined;
      if (code === "EEXIST") throw error;
      writeFileSync(lockPath, serialized, { flag: "wx", flush: true });
    }
  } finally {
    try {
      unlinkSync(temporaryPath);
    } catch {
      // Best-effort cleanup after an interrupted or contended publish.
    }
  }
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    try {
      if (!existsSync(lockPath)) return;
      if (sameProcessRecord(record, readProcessRecord(lockPath))) unlinkSync(lockPath);
    } catch {
      // best-effort cleanup
    }
  };
  return { lockPath, release };
}

/**
 * Try to acquire an exclusive embed lock at `lockPath`.
 * Returns a handle with `release()` on success, or `null` if another live
 * qmd process already holds the lock.
 */
export function tryAcquireEmbedLock(lockPath: string): EmbedLockHandle | null {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return createOwnedLock(lockPath);
    } catch (err: unknown) {
      const code =
        typeof err === "object" && err !== null && "code" in err
          ? (err as NodeJS.ErrnoException).code
          : undefined;
      if (code !== "EEXIST") throw err;

      const holder = readProcessRecord(lockPath);
      if (holder.kind === "invalid") {
        try {
          const before = statSync(lockPath);
          const recheck = readProcessRecord(lockPath);
          const after = statSync(lockPath);
          if (
            Date.now() - before.mtimeMs >= 30_000
            && recheck.kind === "invalid"
            && before.mtimeMs === after.mtimeMs
            && before.size === after.size
          ) {
            unlinkSync(lockPath);
            continue;
          }
        } catch {
          // Another process changed the lock; retry the exclusive publish.
          continue;
        }
      }
      if (isLiveEmbedLockHolder(holder)) {
        return null;
      }

      // Dead or recycled PID — remove and retry once.
      try {
        if (sameProcessRecord(holder, readProcessRecord(lockPath))) unlinkSync(lockPath);
      } catch {
        // Another process may have claimed it; loop and try wx again.
      }
    }
  }

  // Final attempt lost the race to a live holder (or repeated EEXIST).
  const holder = readProcessRecord(lockPath);
  if (isLiveEmbedLockHolder(holder)) {
    return null;
  }
  return null;
}

/** User-facing message when a second embed is skipped. */
export const EMBED_LOCK_BUSY_MESSAGE =
  "Another embed process is already running. Skipping.";

export function embedLockBusyMessage(lockPath: string): string {
  return `${EMBED_LOCK_BUSY_MESSAGE}\nLock: ${lockPath}\nIf no embed is active and this file is corrupt, wait 30 seconds and retry; QMD will reclaim it.`;
}
