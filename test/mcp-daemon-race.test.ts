/**
 * Cross-process regressions for the MCP daemon pidfile (#806, pidfile races).
 *
 * The pidfile is mutated — published by a starter, dropped by start/stop/status,
 * unlinked by the daemon on exit — only inside a cross-process critical section
 * (an exclusive SQLite write transaction on `mcp.pid.lock.sqlite`). These tests
 * assert what a consumer observes under real contention:
 *
 *   - a held section makes every transition fail safe instead of half-applying;
 *   - a killed holder needs no cleanup (the kernel releases the transaction);
 *   - of simultaneous claimants exactly one wins, and the pidfile name never
 *     disappears while ownership moves (the old move-aside protocol left a
 *     window in which a live daemon could be displaced by a third starter);
 *   - `qmd mcp --http --daemon` under contention yields exactly one tracked,
 *     reachable daemon and a stoppable end state.
 *
 * Real child processes and real timers are intentional: the conditions under
 * test are observed through the OS process table, SQLite's file locks and a
 * listening HTTP port, none of which a fake clock can advance. Waits are
 * predicate- or event-based (`waitFor`, `child.once("close")`); the only fixed
 * delays are poll intervals and the sampling loop of the "name never
 * disappears" watcher.
 */

import { afterEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import {
  claimMcpDaemonPidfile,
  readMcpDaemonPidState,
  removeMcpDaemonPidfileIfRecords,
  removeStaleMcpDaemonPidfile,
} from "../src/cli/mcp-pid.ts";

const DEAD_PID = 999999999;
const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const helperScript = join(thisDir, "_helpers", "mcp-daemon-race-helper.ts");
const isBunRuntime = "Bun" in globalThis;

const cleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for condition");
    await delay(20);
  }
}

/** A throwing/passing liveness probe: true while `pid` still exists. */
function pidIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

type Helper = {
  proc: ChildProcess;
  output: () => string;
  waitFor: (needle: string, timeoutMs?: number) => Promise<void>;
  done: Promise<number | null>;
};

/** Spawn the race helper; trailing qmd tokens make it classify as a daemon. */
function spawnHelper(
  mode: "hold" | "claim",
  pidFile: string,
  durationMs: number,
  options: { barrier?: boolean } = {},
): Helper {
  const args = isBunRuntime
    ? [helperScript, mode, pidFile, String(durationMs), "qmd", "mcp", "--http"]
    : [tsxCli, helperScript, mode, pidFile, String(durationMs), "qmd", "mcp", "--http"];
  const proc = spawn(process.execPath, args, {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...(options.barrier ? { DAEMON_RACE_BARRIER: "1" } : {}) },
  });

  let output = "";
  const waiters = new Set<{ needle: string; resolve: () => void }>();
  const publish = () => {
    for (const waiter of waiters) {
      if (!output.includes(waiter.needle)) continue;
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

  const waitForOutput = (needle: string, timeoutMs = 20_000): Promise<void> => {
    if (output.includes(needle)) return Promise.resolve();
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const waiter = { needle, resolve };
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

  return { proc, output: () => output, waitFor: waitForOutput, done };
}

/** Throwaway pidfile directory (sync: used to seed files before any await). */
function tempPidDir(): { dir: string; pidFile: string } {
  const dir = mkdtempSync(join(tmpdir(), "qmd-mcp-race-"));
  cleanups.push(() => rm(dir, { recursive: true, force: true }));
  return { dir, pidFile: join(dir, "mcp.pid") };
}

describe("pidfile critical section", () => {
  test("a held section makes claim and cleanup fail safe, and neither mutates the file", async () => {
    const { pidFile } = tempPidDir();
    writeFileSync(pidFile, String(DEAD_PID)); // stale: cleanup would normally drop it
    const holder = spawnHelper("hold", pidFile, 15_000);

    try {
      await holder.waitFor("HOLD");

      // Claim: refused, nothing published, no clobbering behind the holder's back.
      expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({
        claimed: false,
        reason: "unclaimable",
        error: expect.stringContaining("another qmd process"),
      });
      // Cleanup: cannot re-verify, so it does not touch the file.
      expect(removeStaleMcpDaemonPidfile(pidFile)).toEqual({
        status: "locked-out",
        error: expect.stringContaining("another qmd process"),
      });
      // Exit-budget callers skip immediately rather than waiting on the holder.
      const startedAt = Date.now();
      expect(removeMcpDaemonPidfileIfRecords(pidFile, DEAD_PID, { budgetMs: 0 })).toBe(false);
      expect(Date.now() - startedAt).toBeLessThan(1_000);
      expect(readFileSync(pidFile, "utf-8")).toBe(String(DEAD_PID));
    } finally {
      holder.proc.kill("SIGKILL");
      await holder.done;
    }

    // Crash recovery: the OS dropped the holder's transaction with the process,
    // so the next transition proceeds with no cleanup or stale detection.
    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({ claimed: true, replaced: "stale" });
    expect(readFileSync(pidFile, "utf-8")).toBe(String(process.pid));
  }, 30_000);

  test("simultaneous claimants: exactly one wins and the PID file name never disappears", async () => {
    const { pidFile } = tempPidDir();
    // Seed a stale owner so the contested transition has something to replace
    // (the old protocol moved it aside, briefly leaving no pidfile at all).
    writeFileSync(pidFile, String(DEAD_PID));

    const contenders = 4;
    const children: Helper[] = [];
    let watching = true;
    let sawAbsent = false;

    try {
      for (let i = 0; i < contenders; i++) {
        children.push(spawnHelper("claim", pidFile, 1_500, { barrier: true }));
      }
      await Promise.all(children.map((child) => child.waitFor("READY")));

      const watcher = (async () => {
        while (watching) {
          if (!existsSync(pidFile)) sawAbsent = true;
          await delay(1);
        }
      })();

      for (const child of children) child.proc.stdin?.write("go\n");
      // Each claimant reports as soon as its transition is decided; the winner
      // then stays alive, so the file still describes a live daemon below.
      await Promise.all(
        children.map((child) => waitFor(() => /(?:CLAIMED|RUNNING|UNCLAIMABLE) \S+/.test(child.output()))),
      );

      const winners = children.filter((child) => /(^|\n)CLAIMED \d+\n/.test(child.output()));
      const losers = children.filter((child) => /(^|\n)RUNNING \d+\n/.test(child.output()));
      expect(winners).toHaveLength(1);
      expect(losers).toHaveLength(contenders - 1);

      const [winner] = winners;
      const claimedPid = winner?.output().match(/CLAIMED (\d+)/)?.[1];
      if (!claimedPid) throw new Error("no claimant won the pidfile");
      const winnerPid = Number(claimedPid);
      // The winner keeps its file, byte for byte, and is still its live owner.
      expect(readFileSync(pidFile, "utf-8")).toBe(String(winnerPid));
      expect(readMcpDaemonPidState(pidFile)).toEqual({ status: "running", pid: winnerPid });
      for (const loser of losers) {
        expect(loser.output()).toContain(`RUNNING ${winnerPid}`);
      }

      const codes = await Promise.all(children.map((child) => child.done));
      watching = false;
      await watcher;
      expect(codes.every((code) => code === 0)).toBe(true);
      expect(sawAbsent).toBe(false);
    } finally {
      // Also on a failed assertion: a still-running watcher would never settle.
      watching = false;
      for (const child of children) child.proc.kill("SIGKILL");
    }
  }, 30_000);
});

// ---------------------------------------------------------------------------
// CLI-level contention
// ---------------------------------------------------------------------------

type Sandbox = {
  root: string;
  cacheHome: string;
  dbPath: string;
  configDir: string;
  cwd: string;
};

async function createSandbox(): Promise<Sandbox> {
  const root = await mkdtemp(join(tmpdir(), "qmd-mcp-daemon-race-"));
  const sandbox: Sandbox = {
    root,
    cacheHome: join(root, "cache"),
    dbPath: join(root, "index.sqlite"),
    configDir: join(root, "config"),
    cwd: join(root, "work"),
  };
  await mkdir(sandbox.configDir, { recursive: true });
  await mkdir(sandbox.cwd, { recursive: true });
  await writeFile(join(sandbox.configDir, "index.yml"), "collections: {}\n");
  cleanups.push(async () => {
    // A detached daemon outlives the CLI that started it.
    stopSandboxDaemons(sandbox);
    await rm(root, { recursive: true, force: true });
  });
  return sandbox;
}

function sandboxPidFile(sandbox: Sandbox): string {
  return join(sandbox.cacheHome, "qmd", "mcp.pid");
}

/** Best-effort kill of every daemon a sandbox pidfile points at. */
function stopSandboxDaemons(sandbox: Sandbox): void {
  const pidFile = sandboxPidFile(sandbox);
  if (!existsSync(pidFile)) return;
  const pid = Number.parseInt(readFileSync(pidFile, "utf-8").trim(), 10);
  if (Number.isInteger(pid) && pid > 0) {
    try { process.kill(pid, "SIGTERM"); } catch { /* already gone */ }
  }
}

async function runSandboxQmd(
  sandbox: Sandbox,
  args: string[],
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const runner = isBunRuntime
    ? { command: process.execPath, args: [qmdScript, ...args] }
    : { command: process.execPath, args: [tsxCli, qmdScript, ...args] };
  const proc = spawn(runner.command, runner.args, {
    cwd: sandbox.cwd,
    env: {
      ...process.env,
      XDG_CACHE_HOME: sandbox.cacheHome,
      INDEX_PATH: sandbox.dbPath,
      QMD_CONFIG_DIR: sandbox.configDir,
      QMD_DOCTOR_DEVICE_PROBE: "0",
      PWD: sandbox.cwd,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  // Collect outside the listeners: both pipes may end before `close` fires.
  let stdout = "";
  let stderr = "";
  proc.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
  proc.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
  const exit = Promise.withResolvers<number>();
  proc.once("error", exit.reject);
  proc.on("close", (code) => exit.resolve(code ?? 1));
  const exitCode = await exit.promise;
  return { stdout, stderr, exitCode };
}

function randomPort(): number {
  return 10000 + Math.floor(Math.random() * 50000);
}

async function waitForServer(port: number, timeoutMs = 10_000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://localhost:${port}/health`);
      if (res.ok) return true;
    } catch {
      // not ready yet
    }
    await delay(100);
  }
  return false;
}

describe("qmd mcp daemon races", () => {
  test("concurrent --daemon starts: exactly one is reported started, tracked and reachable", async () => {
    const sandbox = await createSandbox();
    const ports = [randomPort(), randomPort(), randomPort()];

    const attempts = await Promise.all(
      ports.map(async (port) => ({
        port,
        result: await runSandboxQmd(sandbox, ["mcp", "--http", "--daemon", "--port", String(port)]),
      })),
    );

    const winners = attempts.filter(({ result }) => result.exitCode === 0);
    expect(winners).toHaveLength(1);
    for (const { result } of attempts) {
      if (result.exitCode === 0) continue;
      expect(result.exitCode).toBe(1);
      // A loser must say why: another daemon owns the pidfile.
      expect(result.stderr).toContain("Already running");
    }

    const [winner] = winners;
    if (!winner) throw new Error("no starter won the pidfile");
    const pidFile = sandboxPidFile(sandbox);
    const pid = Number.parseInt(readFileSync(pidFile, "utf-8").trim(), 10);
    expect(readMcpDaemonPidState(pidFile)).toEqual({ status: "running", pid });
    expect(pidIsAlive(pid)).toBe(true);
    expect(winner.result.stdout).toContain(`(PID ${pid})`);
    // The tracked daemon is the one serving the port it reported.
    expect(await waitForServer(winner.port)).toBe(true);

    const stopped = await runSandboxQmd(sandbox, ["mcp", "stop"]);
    expect(stopped.exitCode).toBe(0);
    expect(stopped.stdout).toContain("Stopped");
    expect(existsSync(pidFile)).toBe(false);
    await waitFor(() => !pidIsAlive(pid));
  }, 90_000);

  test("concurrent stop and status: no transition wedges, the daemon ends up stopped", async () => {
    const sandbox = await createSandbox();
    const port = randomPort();
    const started = await runSandboxQmd(sandbox, ["mcp", "--http", "--daemon", "--port", String(port)]);
    expect(started.exitCode).toBe(0);
    expect(await waitForServer(port)).toBe(true);

    const pidFile = sandboxPidFile(sandbox);
    const pid = Number.parseInt(readFileSync(pidFile, "utf-8").trim(), 10);

    const results = await Promise.all([
      runSandboxQmd(sandbox, ["mcp", "stop"]),
      runSandboxQmd(sandbox, ["status"]),
      runSandboxQmd(sandbox, ["mcp", "stop"]),
    ]);

    expect(results.map((result) => result.exitCode)).toEqual([0, 0, 0]);
    expect(existsSync(pidFile)).toBe(false);
    await waitFor(() => !pidIsAlive(pid));
  }, 60_000);

  test("status never drops a live daemon's pidfile", async () => {
    const sandbox = await createSandbox();
    const port = randomPort();
    const started = await runSandboxQmd(sandbox, ["mcp", "--http", "--daemon", "--port", String(port)]);
    expect(started.exitCode).toBe(0);
    expect(await waitForServer(port)).toBe(true);

    const pidFile = sandboxPidFile(sandbox);
    const pid = Number.parseInt(readFileSync(pidFile, "utf-8").trim(), 10);

    const status = await runSandboxQmd(sandbox, ["status"]);
    expect(status.exitCode).toBe(0);
    expect(readMcpDaemonPidState(pidFile)).toEqual({ status: "running", pid });

    const stopped = await runSandboxQmd(sandbox, ["mcp", "stop"]);
    expect(stopped.exitCode).toBe(0);
    expect(existsSync(pidFile)).toBe(false);
  }, 60_000);
});
