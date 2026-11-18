/**
 * Focused tests for the MCP daemon pidfile lifecycle state machine.
 *
 * Covers the transitions that the identity helpers exist to make safe:
 * absent / malformed / stale (dead or recycled PID) / running, the stop
 * transition (never signal a stale PID, only report "stopped" once it is
 * gone), and the exclusive start claim (two starters cannot both win).
 *
 * Real timers are intentional: every condition under test is observed through
 * the OS process table (PID liveness, cmdline, SIGTERM handling), so fake
 * timers could not advance it. Waits are event/predicate-based (`waitFor`,
 * `child.once("exit")`); the only fixed delay is the 20ms poll interval.
 */

import { describe, test, expect, afterEach } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import {
  claimMcpDaemonPidfile,
  classifyMcpDaemonPidfile,
  isQmdMcpPid,
  mcpDaemonLockStatePath,
  pidfileRecordsPid,
  readMcpDaemonPidState,
  removeMcpDaemonPidfileIfRecords,
  removeStaleMcpDaemonPidfile,
  stopMcpDaemon,
  waitForDaemonChildIdentity,
  waitForDaemonPidfileOwnership,
} from "../src/cli/mcp-pid.ts";

const DEAD_PID = 999999999;

const cleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/** Resolves once the child has exited and been reaped by this process. */
function onceExit(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  const { promise, resolve } = Promise.withResolvers<void>();
  child.once("exit", () => resolve());
  return promise;
}

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for condition");
    await delay(20);
  }
}

/** Create a throwaway pidfile; `content === undefined` means "no file". */
function tempPidFile(content?: string): string {
  const dir = mkdtempSync(join(tmpdir(), "qmd-mcp-pid-"));
  cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
  const pidFile = join(dir, "mcp.pid");
  if (content !== undefined) writeFileSync(pidFile, content);
  return pidFile;
}

function trackChild(child: ChildProcess): void {
  cleanups.push(async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const { promise, resolve } = Promise.withResolvers<void>();
    child.once("exit", () => resolve());
    child.kill("SIGKILL");
    await promise;
  });
}

/**
 * Stand-in for a qmd daemon: argv carries a `/qmd` token so the pidfile
 * identity check classifies it as running (same shape as the real child
 * spawned by `qmd mcp --http --daemon`).
 *
 * `ignoreSigterm` also waits for the child to *install* its handler before
 * returning. `isQmdMcpPid` is satisfied the instant the exec'd argv is
 * visible — milliseconds before the script body runs — so a SIGTERM sent right
 * after spawn would kill the child by default disposition and the stop test
 * would observe "stopped" instead of a signal the daemon ignores. The child
 * announces itself on stdout once the handler is in place, so readiness is
 * signalled, never inferred from a delay.
 */
async function spawnFakeQmd(options: { ignoreSigterm?: boolean } = {}): Promise<ChildProcess> {
  const script = options.ignoreSigterm
    ? "process.on('SIGTERM', () => {}); process.stdout.write('READY\\n'); setTimeout(() => {}, 60000);"
    : "setTimeout(() => {}, 60000);";
  const child = spawn(process.execPath, ["-e", script, "/tmp/qmd", "mcp", "--http"], {
    stdio: ["ignore", "pipe", "ignore"],
  });
  trackChild(child);
  let output = "";
  child.stdout?.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  await waitFor(
    () => child.pid !== undefined && isQmdMcpPid(child.pid) && (!options.ignoreSigterm || output.includes("READY")),
  );
  return child;
}

/** Live process that does not look like qmd — models a recycled PID. */
async function spawnDecoy(): Promise<ChildProcess> {
  const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 60000)"], { stdio: "ignore" });
  trackChild(child);
  await waitFor(() => child.pid !== undefined);
  return child;
}

describe("pidfile identity states", () => {
  test("absent when there is no pidfile", () => {
    expect(readMcpDaemonPidState(tempPidFile())).toEqual({ status: "absent" });
  });

  test("malformed contents are never interpreted as a PID", () => {
    expect(classifyMcpDaemonPidfile("")).toEqual({ status: "malformed", raw: "" });
    expect(classifyMcpDaemonPidfile("not-a-pid")).toEqual({ status: "malformed", raw: "not-a-pid" });
    expect(classifyMcpDaemonPidfile("-4")).toEqual({ status: "malformed", raw: "-4" });
    expect(classifyMcpDaemonPidfile("12 34")).toEqual({ status: "malformed", raw: "12 34" });
  });

  test("dead PID is stale/exited", () => {
    expect(classifyMcpDaemonPidfile(String(DEAD_PID))).toEqual({
      status: "stale",
      pid: DEAD_PID,
      reason: "exited",
    });
  });

  test("live non-qmd PID is stale/not-qmd, never running", async () => {
    const decoy = await spawnDecoy();
    expect(classifyMcpDaemonPidfile(`${decoy.pid}\n`)).toEqual({
      status: "stale",
      pid: decoy.pid,
      reason: "not-qmd",
    });
  });

  test("live qmd-shaped PID is running", async () => {
    const child = await spawnFakeQmd();
    expect(readMcpDaemonPidState(tempPidFile(`${child.pid}\n`))).toEqual({
      status: "running",
      pid: child.pid,
    });
  });

  test("a pidfile that exists but cannot be read is unreadable, not absent", () => {
    // A directory in the pidfile's place makes readFileSync fail with EISDIR —
    // the same fail-safe branch as EACCES. It must never look "absent".
    const pidFile = tempPidFile();
    mkdirSync(pidFile);

    const state = readMcpDaemonPidState(pidFile);
    expect(state.status).toBe("unreadable");
    expect(state.status === "unreadable" ? state.error : null).toBe("EISDIR");
    expect(existsSync(pidFile)).toBe(true);
  });

  test("pidfileRecordsPid compares only the recorded PID", () => {
    const pidFile = tempPidFile(`${process.pid}\n`);
    expect(pidfileRecordsPid(pidFile, process.pid)).toBe(true);
    expect(pidfileRecordsPid(pidFile, process.pid + 1)).toBe(false);
    expect(pidfileRecordsPid(tempPidFile("garbage"), process.pid)).toBe(false);
    expect(pidfileRecordsPid(tempPidFile(), process.pid)).toBe(false);
  });
});

describe("stop transition", () => {
  test("no pidfile → not-running", async () => {
    expect(await stopMcpDaemon(tempPidFile())).toEqual({ outcome: "not-running" });
  });

  test("malformed pidfile is dropped without signalling anything", async () => {
    const pidFile = tempPidFile("garbage");
    expect(await stopMcpDaemon(pidFile)).toEqual({ outcome: "cleaned-stale", reason: "malformed" });
    expect(existsSync(pidFile)).toBe(false);
  });

  test("unreadable pidfile is left untouched and never signalled", async () => {
    const pidFile = tempPidFile();
    mkdirSync(pidFile);

    expect(await stopMcpDaemon(pidFile)).toEqual({ outcome: "unreadable", error: "EISDIR" });
    expect(existsSync(pidFile)).toBe(true);
  });

  test("recycled PID is cleaned up and left alive", async () => {
    const decoy = await spawnDecoy();
    const pidFile = tempPidFile(String(decoy.pid));

    expect(await stopMcpDaemon(pidFile)).toEqual({ outcome: "cleaned-stale", reason: "not-qmd" });
    expect(existsSync(pidFile)).toBe(false);
    expect(decoy.exitCode).toBeNull();
    expect(() => process.kill(decoy.pid!, 0)).not.toThrow();
  });

  test("running daemon: stopped only after the PID is gone, pidfile removed", async () => {
    const child = await spawnFakeQmd();
    const pidFile = tempPidFile(String(child.pid));
    const exited = onceExit(child);

    const result = await stopMcpDaemon(pidFile, { timeoutMs: 5000 });

    expect(result).toEqual({ outcome: "stopped", pid: child.pid });
    expect(existsSync(pidFile)).toBe(false);
    // Wait for the reaping so the liveness probe below cannot see a zombie.
    await exited;
    expect(() => process.kill(child.pid!, 0)).toThrow();
  });

  test("daemon that ignores SIGTERM times out and keeps its pidfile", async () => {
    const child = await spawnFakeQmd({ ignoreSigterm: true });
    const pidFile = tempPidFile(String(child.pid));

    const result = await stopMcpDaemon(pidFile, { timeoutMs: 200, pollMs: 20 });

    expect(result).toEqual({ outcome: "timeout", pid: child.pid });
    expect(existsSync(pidFile)).toBe(true);
    // Still tracked, so a later stop can retry instead of losing the daemon.
    expect(readFileSync(pidFile, "utf-8")).toBe(String(child.pid));
  });
});

describe("start claim", () => {
  test("absent pidfile is claimed", () => {
    const pidFile = tempPidFile();
    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({ claimed: true, replaced: null });
    expect(readFileSync(pidFile, "utf-8")).toBe(String(process.pid));
  });

  test("stale pidfile is replaced", () => {
    const pidFile = tempPidFile(String(DEAD_PID));
    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({ claimed: true, replaced: "stale" });
    expect(readFileSync(pidFile, "utf-8")).toBe(String(process.pid));
  });

  test("malformed pidfile is replaced", () => {
    const pidFile = tempPidFile("garbage");
    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({ claimed: true, replaced: "malformed" });
    expect(readFileSync(pidFile, "utf-8")).toBe(String(process.pid));
  });

  test("a pidfile owned by a live daemon is not stolen", async () => {
    const child = await spawnFakeQmd();
    const pidFile = tempPidFile(String(child.pid));

    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({
      claimed: false,
      reason: "running",
      runningPid: child.pid,
    });
    // The live daemon keeps its file, byte for byte.
    expect(readFileSync(pidFile, "utf-8")).toBe(String(child.pid));
  });

  test("claim leaves no staging or quarantine litter behind", () => {
    const pidFile = tempPidFile("garbage");
    expect(claimMcpDaemonPidfile(pidFile, process.pid).claimed).toBe(true);
    const entries = readdirSync(join(pidFile, ".."));
    // The pidfile and the sidecar critical-section database (SQLite may keep
    // its WAL siblings); never a `.publish`, `.claim` or `.quarantine` leftover.
    expect(entries).toContain("mcp.pid");
    expect(entries).toContain(basename(mcpDaemonLockStatePath(pidFile)));
    expect(entries.filter((name) => /\.(publish|claim|quarantine)$/.test(name))).toEqual([]);
  });

  test("a PID that is not an identifiable qmd process is never published", async () => {
    // A live non-qmd PID is what a just-spawned child (its cmdline is still the
    // parent's) and a recycled PID both look like: publishing it would make the
    // next stop/status read the file as stale and lose the daemon.
    const decoy = await spawnDecoy();
    const pidFile = tempPidFile();

    expect(claimMcpDaemonPidfile(pidFile, decoy.pid!)).toEqual({
      claimed: false,
      reason: "unclaimable",
      error: expect.stringContaining("not an identifiable live qmd process"),
    });
    expect(existsSync(pidFile)).toBe(false);
  });

  test("a PID that is not alive is never published", () => {
    const pidFile = tempPidFile();
    expect(claimMcpDaemonPidfile(pidFile, DEAD_PID)).toEqual({
      claimed: false,
      reason: "unclaimable",
      error: expect.stringContaining("exited"),
    });
    expect(existsSync(pidFile)).toBe(false);
  });

  test("an unreadable pidfile is not claimed over", () => {
    const pidFile = tempPidFile();
    mkdirSync(pidFile);

    expect(claimMcpDaemonPidfile(pidFile, process.pid)).toEqual({
      claimed: false,
      reason: "unclaimable",
      error: expect.stringContaining("unreadable"),
    });
    expect(existsSync(pidFile)).toBe(true);
  });

  test("a pidfile whose critical-section sidecar cannot be opened is not claimed", () => {
    const pidFile = tempPidFile(String(DEAD_PID));
    // A directory where the sidecar database belongs: the section cannot be
    // taken, so nothing may be published (fail safe, no half-transition).
    mkdirSync(mcpDaemonLockStatePath(pidFile));

    expect(claimMcpDaemonPidfile(pidFile, process.pid).claimed).toBe(false);
    expect(readFileSync(pidFile, "utf-8")).toBe(String(DEAD_PID));
  });
});

describe("stale cleanup", () => {
  test("malformed and stale files are dropped", () => {
    const malformed = tempPidFile("garbage");
    expect(removeStaleMcpDaemonPidfile(malformed)).toEqual({
      status: "removed",
      state: { status: "malformed", raw: "garbage" },
    });
    expect(existsSync(malformed)).toBe(false);

    const stale = tempPidFile(String(DEAD_PID));
    expect(removeStaleMcpDaemonPidfile(stale)).toEqual({
      status: "removed",
      state: { status: "stale", pid: DEAD_PID, reason: "exited" },
    });
    expect(existsSync(stale)).toBe(false);
  });

  test("a live daemon's republished pidfile survives a stale cleanup (quarantine race)", async () => {
    const pidFile = tempPidFile(String(DEAD_PID));
    // The cleanup's decision input: a stale file.
    expect(readMcpDaemonPidState(pidFile).status).toBe("stale");

    // A concurrent `qmd mcp --http --daemon` publishes its live daemon before
    // the cleanup gets to act. The old protocol unlinked on the stale read and
    // lost track of that daemon.
    const daemon = await spawnFakeQmd();
    writeFileSync(pidFile, String(daemon.pid));

    expect(removeStaleMcpDaemonPidfile(pidFile)).toEqual({
      status: "kept",
      state: { status: "running", pid: daemon.pid },
    });
    expect(readMcpDaemonPidState(pidFile)).toEqual({ status: "running", pid: daemon.pid });
  });

  test("an unreadable pidfile is kept: it may describe a daemon we cannot inspect", () => {
    const pidFile = tempPidFile();
    mkdirSync(pidFile);

    expect(removeStaleMcpDaemonPidfile(pidFile)).toEqual({
      status: "kept",
      state: { status: "unreadable", error: "EISDIR" },
    });
    expect(existsSync(pidFile)).toBe(true);
  });
});

describe("owner cleanup", () => {
  test("removes only a pidfile that still records the owner", async () => {
    const daemon = await spawnFakeQmd();
    const pidFile = tempPidFile(String(daemon.pid));

    expect(removeMcpDaemonPidfileIfRecords(pidFile, daemon.pid! + 1)).toBe(false);
    expect(readFileSync(pidFile, "utf-8")).toBe(String(daemon.pid));

    expect(removeMcpDaemonPidfileIfRecords(pidFile, daemon.pid!)).toBe(true);
    expect(existsSync(pidFile)).toBe(false);
  });

  test("an absent or unreadable pidfile is not an error", () => {
    expect(removeMcpDaemonPidfileIfRecords(tempPidFile(), process.pid)).toBe(false);
    const pidFile = tempPidFile();
    mkdirSync(pidFile);
    expect(removeMcpDaemonPidfileIfRecords(pidFile, process.pid)).toBe(false);
    expect(existsSync(pidFile)).toBe(true);
  });
});

describe("startup gates", () => {
  test("child identity: running once the child's own cmdline is qmd", async () => {
    const child = await spawnFakeQmd();
    expect(await waitForDaemonChildIdentity(child.pid!)).toEqual({ status: "running" });
  });

  test("child identity: a child that dies before identifying is reported as exited", async () => {
    const child = spawn(process.execPath, ["-e", "process.exit(3)", "/tmp/qmd", "mcp"], { stdio: "ignore" });
    trackChild(child);
    const exited = onceExit(child);
    const hasExited = () => child.exitCode !== null || child.signalCode !== null;
    // Wait out the real death first: a starter whose child is already gone must
    // report a failure instead of waiting the whole startup window.
    await exited;

    expect(await waitForDaemonChildIdentity(child.pid!, { hasExited, timeoutMs: 5000 })).toEqual({
      status: "exited",
    });
  });

  test("child identity: a live PID that never identifies as qmd times out", async () => {
    const decoy = await spawnDecoy();
    expect(await waitForDaemonChildIdentity(decoy.pid!, { timeoutMs: 120, pollMs: 20 })).toEqual({
      status: "timeout",
    });
    expect(decoy.exitCode).toBeNull();
  });

  test("pidfile ownership: owned once the file names this daemon", async () => {
    const daemon = await spawnFakeQmd();
    const pidFile = tempPidFile();

    const waiting = waitForDaemonPidfileOwnership(pidFile, daemon.pid!, { timeoutMs: 5000, pollMs: 20 });
    // A stale file is neither ours nor a competitor's: the gate waits for the
    // publication instead of giving up on it.
    writeFileSync(pidFile, String(DEAD_PID));
    await delay(60);
    writeFileSync(pidFile, String(daemon.pid));

    expect(await waiting).toBe("owned");
  });

  test("pidfile ownership: taken when another live daemon owns the file", async () => {
    const daemon = await spawnFakeQmd();
    const pidFile = tempPidFile(String(daemon.pid));
    expect(await waitForDaemonPidfileOwnership(pidFile, daemon.pid! + 1)).toBe("taken");
  });

  test("pidfile ownership: a daemon whose starter never published times out", async () => {
    const pidFile = tempPidFile(String(DEAD_PID));
    expect(
      await waitForDaemonPidfileOwnership(pidFile, process.pid, { timeoutMs: 100, pollMs: 20 }),
    ).toBe("timeout");
  });
});

describe("stop while another transition wants the section", () => {
  test("waiting for the daemon to exit never holds the critical section", async () => {
    // The daemon's own exit handler takes the section to unlink its pidfile; if
    // `stop` held it while waiting, the two would deadlock until the timeout.
    const daemon = await spawnFakeQmd({ ignoreSigterm: true });
    const pidFile = tempPidFile(String(daemon.pid));

    // No wall-clock ordering involved: `stopMcpDaemon` signals synchronously and
    // then yields at its first poll, so by the time this expression returns, the
    // waiter is already parked *outside* the section.
    const stopping = stopMcpDaemon(pidFile, { timeoutMs: 600, pollMs: 20 });

    // The daemon's exit handler runs here: zero budget means "take the section
    // now or skip", so it can neither block nor be blocked by the waiter.
    expect(daemon.exitCode).toBeNull(); // the signal really was ignored
    expect(removeMcpDaemonPidfileIfRecords(pidFile, daemon.pid!, { budgetMs: 0 })).toBe(true);
    expect(existsSync(pidFile)).toBe(false);

    expect(await stopping).toEqual({ outcome: "timeout", pid: daemon.pid });
  });
});
