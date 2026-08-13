/**
 * Process-level exclusive lock for `qmd embed`.
 *
 * Concurrent embed runs against the same index can race on vectors_vec
 * (UNIQUE constraint on hash_seq). This lockfile keeps a second process from
 * starting while another embed holds the lock. Stale files left by crashed
 * processes are recovered via PID identity checks (same spirit as mcp-pid.ts).
 */

import { existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { isQmdMcpPid } from "./mcp-pid.js";

export type EmbedLockHandle = {
  lockPath: string;
  release: () => void;
};

/** Lockfile path sibling to the index database. */
export function embedLockPathForDb(dbPath: string): string {
  return join(dirname(dbPath), ".qmd-embed.lock");
}

function readLockPid(lockPath: string): number | null {
  try {
    const raw = readFileSync(lockPath, "utf-8").trim();
    const pid = parseInt(raw, 10);
    if (!Number.isInteger(pid) || pid <= 0) return null;
    return pid;
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

function createOwnedLock(lockPath: string): EmbedLockHandle {
  writeFileSync(lockPath, `${process.pid}\n`, { flag: "wx" });
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    try {
      if (!existsSync(lockPath)) return;
      const written = readLockPid(lockPath);
      if (written === process.pid) unlinkSync(lockPath);
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

      const holderPid = readLockPid(lockPath);
      if (holderPid !== null && isLiveEmbedLockHolder(holderPid)) {
        return null;
      }

      // Stale / unreadable / recycled PID — remove and retry once.
      try {
        unlinkSync(lockPath);
      } catch {
        // Another process may have claimed it; loop and try wx again.
      }
    }
  }

  // Final attempt lost the race to a live holder (or repeated EEXIST).
  const holderPid = readLockPid(lockPath);
  if (holderPid !== null && isLiveEmbedLockHolder(holderPid)) {
    return null;
  }
  return null;
}

/** User-facing message when a second embed is skipped. */
export const EMBED_LOCK_BUSY_MESSAGE =
  "Another embed process is already running. Skipping.";
