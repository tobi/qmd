/**
 * Unit tests for embed exclusive lock (#825).
 */
import { describe, test, expect } from "vitest";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { existsSync, writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import {
  embedLockPathForDb,
  tryAcquireEmbedLock,
  isLiveEmbedLockHolder,
  EMBED_LOCK_BUSY_MESSAGE,
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
      expect((await readFile(lockPath, "utf-8")).trim()).toBe(String(process.pid));

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
      expect((await readFile(lockPath, "utf-8")).trim()).toBe(String(process.pid));
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

describe("EMBED_LOCK_BUSY_MESSAGE", () => {
  test("matches the issue-requested skip message", () => {
    expect(EMBED_LOCK_BUSY_MESSAGE).toBe("Another embed process is already running. Skipping.");
  });
});
