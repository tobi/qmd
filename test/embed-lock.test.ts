/**
 * Unit tests for embed exclusive lock (#825).
 */
import { describe, test, expect } from "vitest";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { existsSync, writeFileSync, unlinkSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import {
  embedLockPathForDb,
  tryAcquireEmbedLock,
  isLiveEmbedLockHolder,
  isOverdueEmbedLockHolder,
  readEmbedLockRecord,
  embedLockMaxAgeMs,
  embedLockReclaimedMessage,
  EMBED_LOCK_BUSY_MESSAGE,
  DEFAULT_EMBED_LOCK_MAX_DURATION_MS,
  EMBED_LOCK_UNCAPPED_MAX_AGE_MS,
} from "../src/cli/embed-lock.ts";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

describe("embedLockPathForDb", () => {
  test("places .qmd-embed.lock next to the index database", () => {
    expect(embedLockPathForDb("/tmp/qmd-cache/index.sqlite")).toBe("/tmp/qmd-cache/.qmd-embed.lock");
    expect(embedLockPathForDb("/var/lib/qmd/custom.sqlite")).toBe("/var/lib/qmd/.qmd-embed.lock");
  });
});

describe("isLiveEmbedLockHolder", () => {
  test("treats the current process as a live holder", () => {
    expect(isLiveEmbedLockHolder(process.pid)).toBe(true);
  });

  test("rejects invalid and dead PIDs", () => {
    expect(isLiveEmbedLockHolder(0)).toBe(false);
    expect(isLiveEmbedLockHolder(-1)).toBe(false);
    expect(isLiveEmbedLockHolder(999999999)).toBe(false);
  });
});

describe("tryAcquireEmbedLock", () => {
  test("lock held → second acquire skips; release → acquire proceeds", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const first = tryAcquireEmbedLock(lockPath);
      expect(first).not.toBeNull();
      expect(existsSync(lockPath)).toBe(true);
      const record = JSON.parse(await readFile(lockPath, "utf-8"));
      expect(record.pid).toBe(process.pid);
      expect(record.maxDurationMs).toBe(DEFAULT_EMBED_LOCK_MAX_DURATION_MS);
      expect(typeof record.startedAt).toBe("number");

      // Same process still holds the lock — second caller must skip.
      expect(tryAcquireEmbedLock(lockPath)).toBeNull();

      first!.release();
      expect(existsSync(lockPath)).toBe(false);

      const again = tryAcquireEmbedLock(lockPath);
      expect(again).not.toBeNull();
      again!.release();
      expect(existsSync(lockPath)).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("cross-process: live qmd-like holder blocks; release allows proceed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-xp-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    const holderTs = join(thisDir, "_helpers", "embed-lock-holder.ts");
    const holdMs = 1000;

    // Include a bare `qmd` argv token so isQmdMcpPid(child) is true.
    const args = isBunRuntime
      ? [holderTs, lockPath, String(holdMs), "qmd", "embed"]
      : [tsxCli, holderTs, lockPath, String(holdMs), "qmd", "embed"];

    try {
      const child = spawn(process.execPath, args, { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
      child.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });

      await new Promise<void>((resolve, reject) => {
        const start = Date.now();
        const timer = setInterval(() => {
          if (stdout.includes("HOLD")) {
            clearInterval(timer);
            resolve();
          } else if (Date.now() - start > 8000) {
            clearInterval(timer);
            reject(new Error(`child never acquired lock: stdout=${stdout} stderr=${stderr}`));
          }
        }, 20);
      });

      expect(tryAcquireEmbedLock(lockPath)).toBeNull();

      const exitCode = await new Promise<number | null>((resolve) => {
        child.on("close", (code) => resolve(code));
      });
      expect(stderr).toBe("");
      expect(exitCode).toBe(0);
      expect(stdout).toContain("RELEASED");

      const after = tryAcquireEmbedLock(lockPath);
      expect(after).not.toBeNull();
      after!.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 20_000);

  test("replaces a stale lock from a dead PID and proceeds", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-stale-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      writeFileSync(lockPath, "999999999\n");
      const handle = tryAcquireEmbedLock(lockPath);
      expect(handle).not.toBeNull();
      expect(handle!.reclaimedFrom).toBeUndefined();
      expect(readEmbedLockRecord(lockPath)?.pid).toBe(process.pid);
      handle!.release();
      expect(existsSync(lockPath)).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("release is idempotent and only unlinks owned lock", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-own-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const handle = tryAcquireEmbedLock(lockPath);
      expect(handle).not.toBeNull();
      handle!.release();
      handle!.release();
      expect(existsSync(lockPath)).toBe(false);

      const again = tryAcquireEmbedLock(lockPath);
      expect(again).not.toBeNull();
      writeFileSync(lockPath, "1\n");
      again!.release();
      expect(existsSync(lockPath)).toBe(true);
      unlinkSync(lockPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("overdue holders (#735)", () => {
  const MIN = 60_000;
  const HOUR = 60 * MIN;

  test("embedLockMaxAgeMs is twice the cap, or the uncapped ceiling", () => {
    expect(embedLockMaxAgeMs(30 * MIN)).toBe(60 * MIN);
    expect(embedLockMaxAgeMs(0)).toBe(EMBED_LOCK_UNCAPPED_MAX_AGE_MS);
    expect(embedLockMaxAgeMs(Number.NaN)).toBe(EMBED_LOCK_UNCAPPED_MAX_AGE_MS);
  });

  test("isOverdueEmbedLockHolder compares age against the holder's own cap", () => {
    const record = { pid: 1, startedAt: 0, maxDurationMs: 30 * MIN, legacy: false };
    expect(isOverdueEmbedLockHolder(record, 59 * MIN)).toBe(false);
    expect(isOverdueEmbedLockHolder(record, 61 * MIN)).toBe(true);
    const uncapped = { ...record, maxDurationMs: 0 };
    expect(isOverdueEmbedLockHolder(uncapped, 23 * HOUR)).toBe(false);
    expect(isOverdueEmbedLockHolder(uncapped, 25 * HOUR)).toBe(true);
  });

  test("a live holder within its budget still blocks", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-fresh-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const t0 = 1_000_000_000_000;
      // process.pid is the one PID guaranteed to be "live" without spawning.
      writeFileSync(lockPath, `${JSON.stringify({ pid: process.pid, startedAt: t0, maxDurationMs: 30 * MIN })}\n`);
      expect(tryAcquireEmbedLock(lockPath, { now: () => t0 + 59 * MIN })).toBeNull();
      expect(readEmbedLockRecord(lockPath)?.startedAt).toBe(t0);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("a live holder past twice its cap is evicted and reported", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-zombie-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const t0 = 1_000_000_000_000;
      writeFileSync(lockPath, `${JSON.stringify({ pid: process.pid, startedAt: t0, maxDurationMs: 30 * MIN })}\n`);
      const now = t0 + 39 * HOUR;
      const handle = tryAcquireEmbedLock(lockPath, { now: () => now, maxDurationMs: 30 * MIN });
      expect(handle).not.toBeNull();
      expect(handle!.reclaimedFrom?.pid).toBe(process.pid);
      expect(handle!.reclaimedFrom?.startedAt).toBe(t0);
      const rewritten = readEmbedLockRecord(lockPath);
      expect(rewritten?.startedAt).toBe(now);
      const message = embedLockReclaimedMessage(handle!.reclaimedFrom!, now);
      expect(message).toContain(`process ${process.pid}`);
      expect(message).toContain("2340 min");
      expect(message).toContain("60 min limit");
      handle!.release();
      expect(existsSync(lockPath)).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("a holder that declared no cap is evicted only after the uncapped ceiling", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-uncapped-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const t0 = 1_000_000_000_000;
      writeFileSync(lockPath, `${JSON.stringify({ pid: process.pid, startedAt: t0, maxDurationMs: 0 })}\n`);
      expect(tryAcquireEmbedLock(lockPath, { now: () => t0 + 23 * HOUR })).toBeNull();
      const handle = tryAcquireEmbedLock(lockPath, { now: () => t0 + 25 * HOUR });
      expect(handle?.reclaimedFrom?.maxDurationMs).toBe(0);
      handle!.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("legacy bare-PID lock: age comes from mtime, cap from the caller", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-legacy-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      writeFileSync(lockPath, `${process.pid}\n`);
      const record = readEmbedLockRecord(lockPath, 30 * MIN);
      expect(record?.legacy).toBe(true);
      expect(record?.pid).toBe(process.pid);
      expect(record?.maxDurationMs).toBe(30 * MIN);

      // Fresh file: still blocks.
      expect(tryAcquireEmbedLock(lockPath, { maxDurationMs: 30 * MIN })).toBeNull();

      // Backdate the file two hours: evicted.
      const twoHoursAgo = new Date(Date.now() - 2 * HOUR);
      utimesSync(lockPath, twoHoursAgo, twoHoursAgo);
      const handle = tryAcquireEmbedLock(lockPath, { maxDurationMs: 30 * MIN });
      expect(handle).not.toBeNull();
      expect(handle!.reclaimedFrom?.legacy).toBe(true);
      handle!.release();
      expect(existsSync(lockPath)).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("malformed lock content is treated as stale", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-garbage-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      writeFileSync(lockPath, "{not json\n");
      expect(readEmbedLockRecord(lockPath)).toBeNull();
      const handle = tryAcquireEmbedLock(lockPath);
      expect(handle).not.toBeNull();
      expect(handle!.reclaimedFrom).toBeUndefined();
      handle!.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("EMBED_LOCK_BUSY_MESSAGE", () => {
  test("matches the issue-requested skip message", () => {
    expect(EMBED_LOCK_BUSY_MESSAGE).toBe("Another embed process is already running. Skipping.");
  });
});
