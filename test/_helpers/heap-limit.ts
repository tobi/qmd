import { readFileSync } from "node:fs";

/**
 * Peak resident memory this process has used, in bytes, or null where the
 * kernel keeps no such counter for it (anything but Linux).
 *
 * Budget tests that spawn a worker use this instead of `PRAGMA
 * hard_heap_limit`: the limit only binds where SQLite tracks its own
 * allocations, which Bun's bundled SQLite on Linux no longer does, whereas
 * VmHWM is the resident high-water mark of the whole process whatever SQLite
 * build is in use. Being monotonic, it also records a transient spike that is
 * gone by the time the next read happens - so reading it before and after one
 * synchronous statement is enough to see what that statement touched.
 */
export function peakRssBytes(): number | null {
  let status: string;
  try {
    status = readFileSync("/proc/self/status", "utf8");
  } catch {
    return null;
  }
  const match = /^VmHWM:\s+(\d+)\s+kB$/m.exec(status);
  return match ? Number(match[1]) * 1024 : null;
}
