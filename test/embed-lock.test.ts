/**
 * Consumer-contract tests for the exclusive `qmd embed` lock (#825).
 *
 * Ownership is an exclusive SQLite write transaction held open on a sidecar
 * database next to the index (see src/cli/embed-lock.ts), so these tests assert
 * what a caller observes — who may run, what release hands over, what a crash
 * leaves behind, how a legacy pidfile is treated — instead of lockfile
 * internals.
 */
import { describe, test, expect } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { closeSync, existsSync, openSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import {
  acquireEmbedLock,
  embedLockPathForDb,
  embedLockStatePath,
  isLiveEmbedLockHolder,
  tryAcquireEmbedLock,
  EMBED_LOCK_BUSY_MESSAGE,
} from "../src/cli/embed-lock.ts";

const thisDir = dirname(fileURLToPath(import.meta.url));
const holderScript = join(thisDir, "_helpers", "embed-lock-holder.ts");
// Loaded with `--import`, so the helper *is* the process we spawn: SIGKILL
// reaches the lock holder. The tsx CLI wrapper would exec a second node that
// owns the lock and forward its output, so a kill would orphan the real holder
// and readiness would arrive one event loop late.
const tsxImport = join(thisDir, "..", "node_modules", "tsx", "dist", "esm", "index.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

type Child = {
  proc: ChildProcess;
  output: () => string;
  /** Write one command line (GO / RELEASE) to the helper's stdin. */
  send: (line: string) => void;
  /** Resolves once the child's combined output matches `needle`. */
  waitFor: (needle: string | RegExp, timeoutMs?: number) => Promise<void>;
  done: Promise<number | null>;
};

/**
 * Spawn the lock-holder helper on `lockPath`; it reports HOLD / ACQUIRED / BUSY
 * on stdout and keeps the lock until this process writes RELEASE, so no
 * assertion races a hold timer. Trailing qmd tokens keep its command line
 * looking like qmd.
 */
function spawnHolder(lockPath: string, env: Record<string, string> = {}): Child {
  const args = isBunRuntime
    ? [holderScript, lockPath, "qmd", "embed"]
    : ["--import", tsxImport, holderScript, lockPath, "qmd", "embed"];
  const proc = spawn(process.execPath, args, {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });

  let output = "";
  const waiters = new Set<{ found: () => boolean; resolve: () => void }>();
  const publish = () => {
    for (const waiter of waiters) {
      if (!waiter.found()) continue;
      waiters.delete(waiter);
      waiter.resolve();
    }
  };
  proc.stdout?.on("data", (chunk: Buffer) => {
    output += chunk.toString();
    publish();
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    output += chunk.toString();
    publish();
  });

  const waitFor = (needle: string | RegExp, timeoutMs = 15_000): Promise<void> => {
    const found = () =>
      typeof needle === "string" ? output.includes(needle) : needle.test(output);
    if (found()) return Promise.resolve();
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const waiter = { found, resolve };
    waiters.add(waiter);
    // Failure guard only: it never decides the outcome, it bounds a hung child.
    const timer = setTimeout(() => {
      if (!waiters.delete(waiter)) return;
      reject(new Error(`child never printed ${needle}: ${output}`));
    }, timeoutMs);
    void promise.then(
      () => clearTimeout(timer),
      () => clearTimeout(timer),
    );
    return promise;
  };

  const { promise: done, resolve: onClose } = Promise.withResolvers<number | null>();
  proc.on("close", (code) => onClose(code));

  return {
    proc,
    output: () => output,
    send: (line: string) => proc.stdin?.write(`${line}\n`),
    waitFor,
    done,
  };
}

describe("embedLockPathForDb", () => {
  test("places .qmd-embed.lock next to the index database", () => {
    expect(embedLockPathForDb("/tmp/qmd-cache/index.sqlite")).toBe("/tmp/qmd-cache/.qmd-embed.lock");
    expect(embedLockPathForDb("/var/lib/qmd/custom.sqlite")).toBe("/var/lib/qmd/.qmd-embed.lock");
  });

  test("carries ownership in a sidecar database beside that path", () => {
    expect(embedLockStatePath(embedLockPathForDb("/tmp/qmd-cache/index.sqlite"))).toBe(
      "/tmp/qmd-cache/.qmd-embed.lock.sqlite",
    );
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
  test("held → second caller is busy fast; release → acquire proceeds", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    const statePath = embedLockStatePath(lockPath);
    try {
      const first = acquireEmbedLock(lockPath);
      expect(first.status).toBe("acquired");
      if (first.status !== "acquired") return;
      expect(first.handle.statePath).toBe(statePath);
      // Ownership is materialised next to the index, as a durable artifact.
      expect(existsSync(statePath)).toBe(true);

      // Same process still holds it — a second caller must skip, and must not
      // wait out a data-write busy timeout to find that out.
      const startedAt = Date.now();
      const second = acquireEmbedLock(lockPath);
      expect(second).toEqual({ status: "busy", reason: { kind: "sidecar" } });
      expect(Date.now() - startedAt).toBeLessThan(2_000);
      expect(tryAcquireEmbedLock(lockPath)).toBeNull();

      first.handle.release();
      // Ownership artifacts outlive the lock: deleting them would let a
      // concurrent acquirer create a fresh inode while a straggler still holds
      // the old one.
      expect(existsSync(statePath)).toBe(true);

      const again = tryAcquireEmbedLock(lockPath);
      expect(again).not.toBeNull();
      again!.release();
      again!.release(); // release is idempotent
      expect(existsSync(statePath)).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("cross-process: live holder blocks; its release allows proceed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-xp-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    const child = spawnHolder(lockPath);
    try {
      // The holder keeps the lock until this process says so, so the assertion
      // below cannot race a hold timer on a loaded runner.
      await child.waitFor("HOLD");

      expect(acquireEmbedLock(lockPath)).toEqual({ status: "busy", reason: { kind: "sidecar" } });
      expect(child.output()).not.toContain("RELEASED");
      expect(tryAcquireEmbedLock(lockPath)).toBeNull();

      child.send("RELEASE");
      expect(await child.done).toBe(0);
      expect(child.output()).toContain("RELEASED");

      const after = tryAcquireEmbedLock(lockPath);
      expect(after).not.toBeNull();
      after!.release();
    } finally {
      child.proc.kill("SIGKILL");
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("crash recovery: a killed holder needs no cleanup or stale detection", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-crash-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    const child = spawnHolder(lockPath);
    try {
      await child.waitFor("HOLD");
      expect(tryAcquireEmbedLock(lockPath)).toBeNull();

      // No release runs: the OS drops the holder's lock with the process.
      child.proc.kill("SIGKILL");
      await child.done;

      const recovered = tryAcquireEmbedLock(lockPath);
      expect(recovered).not.toBeNull();
      recovered!.release();
    } finally {
      child.proc.kill("SIGKILL");
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("simultaneous contenders: exactly one acquires", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-race-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    const contenders = 4;
    const children: Child[] = [];
    try {
      for (let i = 0; i < contenders; i++) {
        children.push(
          spawnHolder(lockPath, { EMBED_LOCK_TRY_ONCE: "1", EMBED_LOCK_BARRIER: "1" }),
        );
      }
      // Start them together so they collide on acquisition, not on spawn cost.
      await Promise.all(children.map((child) => child.waitFor("READY")));
      for (const child of children) child.send("GO");

      // Every contender reports its attempt before the winner lets go: the
      // verdict is decided while the lock is still held, so no straggler can
      // find it free and claim a second ACQUIRED.
      await Promise.all(
        children.map((child) => child.waitFor(/(^|\n)(?:ACQUIRED|BUSY)\n/)),
      );
      const winners = children.filter((child) => /(^|\n)ACQUIRED\n/.test(child.output()));
      const losers = children.filter((child) => /(^|\n)BUSY\n/.test(child.output()));
      expect(winners).toHaveLength(1);
      expect(losers).toHaveLength(contenders - 1);

      // Only the winner is still holding: a loser has already exited, and
      // writing to its closed stdin would surface as EPIPE.
      winners[0]!.send("RELEASE");
      const codes = await Promise.all(children.map((child) => child.done));
      expect(codes.every((code) => code === 0)).toBe(true);
      expect(winners[0]!.output()).toContain("RELEASED");

      const after = tryAcquireEmbedLock(lockPath);
      expect(after).not.toBeNull();
      after!.release();
    } finally {
      // Also on a failed assertion: a holder must not outlive the test.
      for (const child of children) child.proc.kill("SIGKILL");
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("legacy .qmd-embed.lock migration", () => {
  test("stale pidfile from a dead PID is ignored and reclaimed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-stale-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      writeFileSync(lockPath, "999999999\n");
      const result = acquireEmbedLock(lockPath);
      expect(result.status).toBe("acquired");
      if (result.status !== "acquired") return;
      expect(existsSync(lockPath)).toBe(false);
      result.handle.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("a live older-version holder is respected, not stolen", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-legacy-live-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      // The current process counts as a live qmd holder.
      writeFileSync(lockPath, `${process.pid}\n`);
      expect(acquireEmbedLock(lockPath)).toEqual({
        status: "busy",
        reason: { kind: "legacy", pid: process.pid },
      });
      // The advisory file survives, and backing off released the sidecar.
      expect(existsSync(lockPath)).toBe(true);

      writeFileSync(lockPath, "999999999\n");
      const after = tryAcquireEmbedLock(lockPath);
      expect(after).not.toBeNull();
      after!.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("half-written pidfile is treated as owned, then abandoned", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-empty-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      // Older qmd between its O_EXCL create and its PID write.
      writeFileSync(lockPath, "");
      const startedAt = Date.now();
      const result = acquireEmbedLock(lockPath);
      expect(result.status).toBe("acquired");
      if (result.status !== "acquired") return;
      // Bounded settle window, not an unbounded wait.
      expect(Date.now() - startedAt).toBeLessThan(5_000);
      expect(existsSync(lockPath)).toBe(false);
      result.handle.release();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("a pidfile cannot unseat the current owner", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-unseat-"));
    const lockPath = join(dir, ".qmd-embed.lock");
    try {
      const first = acquireEmbedLock(lockPath);
      expect(first.status).toBe("acquired");
      if (first.status !== "acquired") return;

      // An older version creating its pidfile — here preempted mid-create, so
      // the write can land through an unlinked descriptor (the PID is dead, so
      // nothing is waiting on it) — must not produce a second owner.
      const fd = openSync(lockPath, "wx");
      try {
        writeSync(fd, "999999999\n");
      } finally {
        closeSync(fd);
      }

      expect(acquireEmbedLock(lockPath)).toEqual({ status: "busy", reason: { kind: "sidecar" } });
      first.handle.release();

      const after = tryAcquireEmbedLock(lockPath);
      expect(after).not.toBeNull();
      after!.release();
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
