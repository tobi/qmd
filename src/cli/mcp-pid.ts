/**
 * MCP daemon pidfile identity + lifecycle helpers.
 *
 * A pidfile is only a *claim* about a process. It becomes unsafe after PID
 * reuse (e.g. post-reboot), when it is unparsable, when it cannot be read at
 * all, or when a starter loses a race against another starter. Every entry
 * point therefore classifies the file into an explicit state before acting:
 *
 *   absent      ENOENT                           -> nothing to stop, start freely
 *   unreadable  exists but read failed (EACCES…) -> never signalled, never deleted
 *   malformed   contents are not a positive PID  -> stale, drop the file
 *   stale       PID dead, or alive but not qmd    -> stale, drop the file, never signal
 *   running     PID alive and looks like qmd      -> the daemon this file describes
 *
 * Every *mutation* of a pidfile — publication by a starting daemon, stale
 * cleanup by start/stop/status, and the daemon unlinking its own file on exit
 * — runs inside one cross-process critical section: an exclusive SQLite write
 * transaction on a sidecar database next to the pidfile
 * (`mcp.pid.lock.sqlite`, see `mcpDaemonLockStatePath`). SQLite takes the
 * write lock before `BEGIN EXCLUSIVE` returns and the kernel drops it when the
 * process exits or dies, so:
 *
 *  - classify-then-act transitions are atomic with respect to every other qmd
 *    process. A concurrent starter/stopper/status can neither displace a live
 *    daemon's pidfile nor resurrect (or delete) one that was just
 *    republished. The earlier protocol moved a suspicious pidfile aside and
 *    linked it back, which left the name briefly absent — a window in which a
 *    concurrent `status` saw "no pidfile" and a third starter could publish
 *    into the restore path, losing track of a live daemon. There is no
 *    move-aside step left to race: recovery happens inside the section and
 *    the only files ever unlinked are the pidfile itself (after re-verifying
 *    it under the section) and a private staging file;
 *  - recovery needs no PID checks and no delete. A crashed holder's
 *    transaction is rolled back by the next opener, and the sidecar is never
 *    unlinked, so a reclaimer cannot delete a freshly acquired lock by
 *    pathname;
 *  - publication is a write to a private staging file followed by `rename`,
 *    so a lock-free reader (status, `stop`'s first look) never observes
 *    half-written contents and can never mistake a live daemon for a
 *    malformed file.
 *
 * The remaining hazard is on the *process* side, not the filesystem: a child
 * that has just been `spawn()`ed has not exec'd yet, so for a few milliseconds
 * its command line is the parent's (or empty). A starter that published that
 * PID would hand `stop`/`status` a file that classifies as `not-qmd` while a
 * real daemon owns it — the parser would "clean up" a live daemon. Three rules
 * close that window: `waitForDaemonChildIdentity` is what a starter waits on
 * before publishing, `claimMcpDaemonPidfile` refuses to publish a PID it cannot
 * itself identify as a live qmd process (a process may always publish itself),
 * and a managed daemon binds its port only after the pidfile names it
 * (`waitForDaemonPidfileOwnership`), so two concurrent starts can never both
 * serve and a child whose starter died never serves untracked.
 *
 * Sections never wait for anything else — no signal, no process exit — and are
 * bounded by a busy budget instead of queueing, so a `stop` that waits for a
 * daemon to die cannot deadlock against that daemon's own exit handler (which
 * takes the section with a zero budget and skips if it is taken).
 *
 * Start/stop transitions are driven by that state: a recycled PID is never
 * signalled, "stopped" is only reported once the PID is actually gone, of two
 * concurrent `--daemon` starts at most one ends up believing it owns the
 * pidfile, and no code path deletes a pidfile it did not first re-verify under
 * the critical section.
 */

import { existsSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { isBusyError, openDatabase, type Database } from "../db.js";

/**
 * How long `stop` waits for a daemon to exit before giving up on it.
 *
 * Comfortably above the server's own graceful-shutdown deadline
 * (`MCP_SHUTDOWN_TIMEOUT_MS`, 5 s) plus the listener and store teardown that
 * follows it: a daemon shutting down exactly as designed must not be reported
 * as a timeout, which would keep a pidfile it is about to remove.
 */
export const MCP_DAEMON_STOP_TIMEOUT_MS = 15_000;
/** Poll interval used while waiting for the daemon PID to disappear. */
export const MCP_DAEMON_STOP_POLL_MS = 50;
/** How long a start waits for its spawned child to identify itself as qmd. */
export const MCP_DAEMON_SPAWN_IDENTITY_TIMEOUT_MS = 10_000;
/** Poll interval used while waiting for the spawned child's identity. */
export const MCP_DAEMON_SPAWN_IDENTITY_POLL_MS = 20;
/**
 * Budget for taking the pidfile critical section. Sections only wrap a read
 * plus one filesystem operation, so contention is microseconds; the budget
 * exists to ride out a competitor's open/WAL install, not to queue.
 */
const PIDFILE_SECTION_BUDGET_MS = 2_000;
/**
 * Exit handlers must never wait on a competitor: a `stop` is *supposed* to be
 * waiting for this process to die, so blocking here would deadlock the two
 * against each other. Zero budget means "skip if someone else is in there".
 */
export const MCP_DAEMON_EXIT_SECTION_BUDGET_MS = 0;
/** Bounded re-decisions when a transition keeps losing to a competitor. */
const STOP_DECISION_ATTEMPTS = 3;

/** Await a single poll interval. */
function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/** errno code of a thrown filesystem/OS error, if it has one. */
function errorCode(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("code" in error)) return undefined;
  const code = error.code;
  return typeof code === "string" ? code : undefined;
}

/** Human-readable form of a thrown error. */
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Identity of a pidfile as observed on disk. `stale.pid` always carries the
 * recorded PID so callers can report it; it is never safe to signal.
 */
export type McpDaemonPidState =
  | { status: "absent" }
  | { status: "unreadable"; error: string }
  | { status: "malformed"; raw: string }
  | { status: "running"; pid: number }
  | { status: "stale"; pid: number; reason: "exited" | "not-qmd" };

/** Result of the stop transition. Only `stopped` removes the pidfile of a live daemon. */
export type McpDaemonStopOutcome =
  | { outcome: "not-running" }
  | { outcome: "unreadable"; error: string }
  | { outcome: "cleaned-stale"; reason: "malformed" | "exited" | "not-qmd" }
  | { outcome: "stopped"; pid: number }
  | { outcome: "unsignalable"; pid: number; error: string }
  | { outcome: "timeout"; pid: number }
  /** The pidfile kept changing, or its critical section stayed unavailable. */
  | { outcome: "busy"; error: string };

/** Result of the start transition's pidfile claim. */
export type McpDaemonStartClaim =
  | { claimed: true; replaced: "malformed" | "stale" | null }
  /** Another live daemon owns the file; the caller must not keep its child. */
  | { claimed: false; reason: "running"; runningPid: number }
  /** Ownership could not be established without clobbering a file: give up. */
  | { claimed: false; reason: "unclaimable"; error: string };

/** Result of dropping a pidfile that describes no live daemon. */
export type McpDaemonStaleCleanup =
  /** Re-verified as stale/malformed inside the section, then removed. */
  | { status: "removed"; state: McpDaemonPidState }
  /** Re-verification found something else (gone, live, unreadable): untouched. */
  | { status: "kept"; state: McpDaemonPidState }
  /** The critical section could not be taken: nothing was touched. */
  | { status: "locked-out"; error: string };

/** How far a freshly spawned daemon child has got. */
export type McpDaemonSpawnReadiness =
  | { status: "running" }
  | { status: "exited" }
  | { status: "timeout" };

/**
 * Pid/log filenames for the MCP HTTP daemon.
 * The default index keeps `mcp.pid` / `mcp.log` for compatibility; named
 * indexes are scoped so a named daemon can run alongside the default (#772).
 */
export function mcpDaemonStateFiles(indexName: string = "index"): { pidFile: string; logFile: string } {
  const suffix = !indexName || indexName === "index" ? "" : `-${indexName}`;
  return {
    pidFile: `mcp${suffix}.pid`,
    logFile: `mcp${suffix}.log`,
  };
}

/**
 * Sidecar database whose exclusive write transaction *is* the pidfile critical
 * section. It sits next to the pidfile, so each index's daemon state (default
 * vs `--index <name>`) is coordinated independently. The file is created on
 * demand and deliberately left in place: unlinking it would let a
 * concurrent opener create a different inode while a straggler still holds the
 * old one.
 */
export function mcpDaemonLockStatePath(pidFile: string): string {
  return `${pidFile}.lock.sqlite`;
}

/** True if a process command line looks like a qmd CLI invocation. */
export function looksLikeQmdMcpCommand(cmdline: string): boolean {
  const s = cmdline.trim();
  if (!s) return false;
  // Match bare `qmd`, `qmd.ts`/`qmd.js`, or a path ending in /qmd(.ts|.js)
  return /(?:^|[\s/\\])qmd(?:\.(?:ts|js))?(?:[\s]|$)/i.test(s);
}

/** Read process cmdline (Linux /proc preferred; ps fallback for macOS). */
export function readProcessCmdline(pid: number): string | null {
  if (!Number.isInteger(pid) || pid <= 0) return null;

  const procPath = `/proc/${pid}/cmdline`;
  if (existsSync(procPath)) {
    try {
      const raw = readFileSync(procPath, "utf-8");
      const cmdline = raw.replace(/\0/g, " ").trim();
      if (cmdline) return cmdline;
    } catch {
      // fall through to ps
    }
  }

  try {
    let cmdline = "";
    try {
      cmdline = execFileSync("ps", ["-p", String(pid), "-o", "args="], {
        encoding: "utf-8",
        timeout: 2000,
        stdio: ["ignore", "pipe", "ignore"],
      });
    } catch {
      cmdline = execFileSync("ps", ["-p", String(pid), "-o", "command="], {
        encoding: "utf-8",
        timeout: 2000,
        stdio: ["ignore", "pipe", "ignore"],
      });
    }
    const trimmed = cmdline.trim();
    return trimmed || null;
  } catch {
    return null;
  }
}

/** Parse pidfile contents strictly: a whole positive integer, else null. */
export function parsePidfileContent(raw: string | null): number | null {
  if (raw === null) return null;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const pid = Number(trimmed);
  return Number.isSafeInteger(pid) && pid > 0 ? pid : null;
}

/**
 * Classify a PID against the live process table.
 * `not-qmd` covers both "alive but a different program" (recycled PID) and
 * "alive but cmdline unreadable" — neither may be signalled or treated as
 * "already running".
 */
export function classifyProcessIdentity(pid: number): "running" | "exited" | "not-qmd" {
  if (!Number.isInteger(pid) || pid <= 0) return "exited";

  let alive = false;
  try {
    process.kill(pid, 0);
    alive = true;
  } catch (error) {
    // EPERM: the PID exists but belongs to someone else — alive, not ours.
    alive = (error as NodeJS.ErrnoException).code === "EPERM";
  }
  if (!alive) return "exited";

  const cmdline = readProcessCmdline(pid);
  if (!cmdline) return "not-qmd";
  return looksLikeQmdMcpCommand(cmdline) ? "running" : "not-qmd";
}

/**
 * Returns true only if `pid` is alive AND its command line looks like qmd.
 * If cmdline cannot be read or does not match, returns false (treat as stale).
 */
export function isQmdMcpPid(pid: number): boolean {
  return classifyProcessIdentity(pid) === "running";
}

/** Classify raw pidfile contents (`null` = file does not exist). */
export function classifyMcpDaemonPidfile(raw: string | null): McpDaemonPidState {
  if (raw === null) return { status: "absent" };

  const pid = parsePidfileContent(raw);
  if (pid === null) return { status: "malformed", raw };

  const identity = classifyProcessIdentity(pid);
  return identity === "running" ? { status: "running", pid } : { status: "stale", pid, reason: identity };
}

/**
 * Read and classify the pidfile at `pidFile`.
 *
 * Only ENOENT/ENOTDIR mean "no pidfile". Every other failure (EACCES, EIO,
 * EPERM, a directory in the way) is reported as `unreadable`: the file may
 * well belong to a live daemon, so callers must fail safe — never signal,
 * never delete, never claim over it.
 *
 * Lock-free by design: publication is a `rename` and removal re-verifies under
 * the critical section, so a reader either sees a complete old file, a
 * complete new file, or no file.
 */
export function readMcpDaemonPidState(pidFile: string): McpDaemonPidState {
  let raw: string;
  try {
    raw = readFileSync(pidFile, "utf-8");
  } catch (error) {
    const code = errorCode(error);
    if (code === "ENOENT" || code === "ENOTDIR") return { status: "absent" };
    return { status: "unreadable", error: code ?? errorMessage(error) };
  }
  return classifyMcpDaemonPidfile(raw);
}

/**
 * True if `pidFile` records exactly `pid`. Pure file check — no process-table
 * probe — so it is cheap enough for exit handlers, which only need to know
 * whether the file is theirs. A plain read; deleting on the strength of it
 * requires `removeMcpDaemonPidfileIfRecords`, which re-checks under the
 * critical section.
 */
export function pidfileRecordsPid(pidFile: string, pid: number): boolean {
  try {
    return parsePidfileContent(readFileSync(pidFile, "utf-8")) === pid;
  } catch {
    return false;
  }
}

/** True if a process exists for `pid` (zombies included; EPERM counts as alive). */
export function isProcessAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** Outcome of running a function inside the pidfile critical section. */
type SectionOutcome<T> =
  | { status: "ran"; value: T }
  | { status: "busy" }
  | { status: "failed"; error: string };

/**
 * Run `fn` while holding the cross-process pidfile critical section.
 *
 * Ownership is an exclusive SQLite write transaction on the sidecar database:
 * it is a single statement (so it cannot exist half-taken), the kernel drops
 * it when the process dies (so a crash needs no cleanup), and it can never be
 * stolen by unlinking a path. The transaction is released — rollback + close —
 * before this returns, including on error.
 *
 * Never throws: filesystem/database failures are reported as `failed` so
 * callers can fail safe (mutate nothing) instead of dying mid-transition.
 */
function runInPidfileSection<T>(pidFile: string, budgetMs: number, fn: () => T): SectionOutcome<T> {
  const statePath = mcpDaemonLockStatePath(pidFile);
  let db: Database | null = null;
  let ownsTransaction = false;
  try {
    db = openDatabase(statePath);
    // `openDatabase` installs a data-write budget (120s by default); the
    // critical section is decided in milliseconds or not at all.
    db.exec(`PRAGMA busy_timeout = ${Math.max(0, Math.floor(budgetMs))}`);
    db.exec("BEGIN EXCLUSIVE");
    ownsTransaction = true;
    return { status: "ran", value: fn() };
  } catch (error) {
    if (isBusyError(error)) return { status: "busy" };
    return { status: "failed", error: errorCode(error) ?? errorMessage(error) };
  } finally {
    if (db) {
      if (ownsTransaction) {
        try {
          db.exec("ROLLBACK");
        } catch {
          // Closing below releases whatever is left of the transaction.
        }
      }
      try {
        db.close();
      } catch {
        // Best effort: the lock is gone with the connection either way.
      }
    }
  }
}

/**
 * Publish `content` as `pidFile` atomically: a private staging file is written
 * and `rename`d into place, so no reader can observe a partial pidfile and a
 * replacement never goes through an absent name.
 *
 * Only called while holding the critical section, which is what makes the
 * caller's classify-then-publish decision atomic. The staging name carries the
 * writer's PID so two processes cannot share it even if the section were
 * somehow bypassed.
 */
function publishPidfileAtomic(pidFile: string, content: string): void {
  const stagingPath = `${pidFile}.${process.pid}.publish`;
  writeFileSync(stagingPath, content);
  try {
    renameSync(stagingPath, pidFile);
  } catch (error) {
    try {
      unlinkSync(stagingPath);
    } catch {
      // Ignore: the staging file is private litter at worst.
    }
    throw error;
  }
}

/**
 * Claim `pidFile` for the daemon `pid`.
 *
 * The decision runs entirely inside the critical section, so of two concurrent
 * starters exactly one wins and a live daemon's file is never replaced. A
 * successful claim republishes the name atomically, so a reader sees either the
 * previous owner's PID or the new one — never "no PID file".
 *
 * `pid` is the daemon process the caller is about to rely on. Unless it is this
 * process (a daemon publishing itself), it must already classify as a live qmd
 * process: otherwise the file would be published for a PID nobody can identify,
 * and the next `stop`/`status` would read it as stale and lose the daemon.
 * Starters wait for that with `waitForDaemonChildIdentity` before calling this.
 */
export function claimMcpDaemonPidfile(pidFile: string, pid: number): McpDaemonStartClaim {
  if (!Number.isInteger(pid) || pid <= 0) {
    throw new Error(`invalid pid for MCP daemon pidfile claim: ${pid}`);
  }

  if (pid !== process.pid) {
    const identity = classifyProcessIdentity(pid);
    if (identity !== "running") {
      return {
        claimed: false,
        reason: "unclaimable",
        error: `PID ${pid} is not an identifiable live qmd process (${identity})`,
      };
    }
  }

  const section = runInPidfileSection(pidFile, PIDFILE_SECTION_BUDGET_MS, (): McpDaemonStartClaim => {
    const state = readMcpDaemonPidState(pidFile);
    if (state.status === "running") {
      return { claimed: false, reason: "running", runningPid: state.pid };
    }
    if (state.status === "unreadable") {
      // Cannot inspect what we would be replacing: fail safe, never steal it.
      return { claimed: false, reason: "unclaimable", error: `PID file is unreadable (${state.error})` };
    }

    publishPidfileAtomic(pidFile, String(pid));
    return { claimed: true, replaced: state.status === "absent" ? null : state.status };
  });

  if (section.status === "ran") return section.value;
  return {
    claimed: false,
    reason: "unclaimable",
    error: section.status === "busy"
      ? "another qmd process is updating the PID file"
      : section.error,
  };
}

/**
 * Remove `pidFile` if, re-verified inside the critical section, it still
 * describes no live daemon.
 *
 * This is the only way stale state is dropped: the caller's earlier read is
 * only a hint. A daemon that a concurrent start published in the meantime is
 * `kept` untouched, which is what makes a racing `status`/`stop`/`start`
 * combination harmless. Callers pass an unreadable file through untouched, as
 * it may describe a daemon we cannot inspect.
 */
export function removeStaleMcpDaemonPidfile(pidFile: string): McpDaemonStaleCleanup {
  const section = runInPidfileSection(pidFile, PIDFILE_SECTION_BUDGET_MS, (): McpDaemonStaleCleanup => {
    const state = readMcpDaemonPidState(pidFile);
    if (state.status !== "stale" && state.status !== "malformed") {
      return { status: "kept", state };
    }
    try {
      unlinkSync(pidFile);
    } catch (error) {
      const code = errorCode(error);
      // ENOENT: a competitor already dropped it — the postcondition holds.
      if (code !== "ENOENT") return { status: "kept", state };
    }
    return { status: "removed", state };
  });

  if (section.status === "ran") return section.value;
  return {
    status: "locked-out",
    error: section.status === "busy"
      ? "another qmd process is updating the PID file"
      : section.error,
  };
}

/**
 * Remove `pidFile` if it still records `pid`, inside the critical section.
 *
 * This is the only safe unlink for an owner: read-then-unlink without the
 * section can delete a pidfile that a concurrent starter republished after the
 * owner's PID was replaced. Returns false when the file is not ours, when the
 * check could not run, or when the section was unavailable — leaving the file
 * behind is always safe, since the next transition re-classifies it.
 *
 * `budgetMs` is the section budget; exit handlers pass
 * `MCP_DAEMON_EXIT_SECTION_BUDGET_MS` so they skip instead of waiting on a
 * `stop` that is waiting for them.
 */
export function removeMcpDaemonPidfileIfRecords(
  pidFile: string,
  pid: number,
  options: { budgetMs?: number } = {},
): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;

  const section = runInPidfileSection(
    pidFile,
    options.budgetMs ?? PIDFILE_SECTION_BUDGET_MS,
    () => {
      if (!pidfileRecordsPid(pidFile, pid)) return false;
      try {
        unlinkSync(pidFile);
        return true;
      } catch (error) {
        return errorCode(error) === "ENOENT";
      }
    },
  );

  return section.status === "ran" ? section.value : false;
}

/**
 * Wait until the freshly spawned daemon `pid` identifies itself as qmd.
 *
 * `spawn()` returns before the child execs. Until it does, its command line is
 * the parent's or empty, so the PID classifies as `not-qmd` — the exact window
 * in which a published pidfile would look stale to a competitor. Starters wait
 * here first and publish only afterwards.
 *
 * `hasExited` lets the caller surface a child that died before exec (failed
 * exec, or a bind error) as `exited` rather than burning the whole timeout.
 */
export async function waitForDaemonChildIdentity(
  pid: number,
  options: { timeoutMs?: number; pollMs?: number; hasExited?: () => boolean } = {},
): Promise<McpDaemonSpawnReadiness> {
  const timeoutMs = options.timeoutMs ?? MCP_DAEMON_SPAWN_IDENTITY_TIMEOUT_MS;
  const pollMs = options.pollMs ?? MCP_DAEMON_SPAWN_IDENTITY_POLL_MS;
  const hasExited = options.hasExited ?? (() => false);
  const deadline = Date.now() + timeoutMs;

  for (;;) {
    if (classifyProcessIdentity(pid) === "running") {
      // A child that died right after exec leaves a zombie whose cmdline is
      // already gone; re-check so "running" is never reported for a corpse.
      return hasExited() ? { status: "exited" } : { status: "running" };
    }
    if (hasExited()) return { status: "exited" };
    if (Date.now() >= deadline) return { status: "timeout" };
    await delay(pollMs);
  }
}

/**
 * Wait until `pidFile` names `pid` — or names another live daemon.
 *
 * A daemon spawned by `qmd mcp --http --daemon` uses this as its start gate:
 * the pidfile is the ownership token that decides which of two concurrent
 * starts is allowed to serve, so the loser exits instead of racing the winner
 * for the port. The same gate covers the crashed-starter case: a child whose
 * parent died before publishing is never named by any pidfile and exits
 * instead of serving untracked.
 */
export async function waitForDaemonPidfileOwnership(
  pidFile: string,
  pid: number,
  options: { timeoutMs?: number; pollMs?: number } = {},
): Promise<"owned" | "taken" | "timeout"> {
  const timeoutMs = options.timeoutMs ?? MCP_DAEMON_SPAWN_IDENTITY_TIMEOUT_MS;
  const pollMs = options.pollMs ?? MCP_DAEMON_SPAWN_IDENTITY_POLL_MS;
  const deadline = Date.now() + timeoutMs;

  for (;;) {
    // Ownership is decided by the recorded PID, not by the daemon's own
    // command line: a process must not depend on being able to re-inspect
    // itself. Only *another* live daemon makes the file unavailable — a stale
    // or malformed file is mid-transition (the starter is about to replace it),
    // so it is neither ours nor somebody else's yet.
    if (pidfileRecordsPid(pidFile, pid)) return "owned";
    const state = readMcpDaemonPidState(pidFile);
    if (state.status === "running") return "taken";
    if (Date.now() >= deadline) return "timeout";
    await delay(pollMs);
  }
}

/** Signal the live daemon `pid` and wait until it is gone. */
async function stopRunningDaemon(
  pidFile: string,
  pid: number,
  timeoutMs: number,
  pollMs: number,
): Promise<McpDaemonStopOutcome> {
  try {
    process.kill(pid, "SIGTERM");
  } catch (error) {
    const code = errorCode(error);
    if (code === "ESRCH") {
      // It exited between classification and signal: the file is now stale.
      removeStaleMcpDaemonPidfile(pidFile);
      return { outcome: "cleaned-stale", reason: "exited" };
    }
    // EPERM and friends: the daemon may still be running, so keep tracking it.
    return { outcome: "unsignalable", pid, error: code ?? errorMessage(error) };
  }

  // Deliberately *outside* the critical section: the daemon's own exit handler
  // takes that section to unlink its pidfile, so holding it while waiting for
  // the PID to disappear would deadlock the daemon's exit against this wait
  // (and vice versa). The identity re-check matters in containers: an
  // exited-but-unreaped child is a zombie with an empty cmdline, and a plain
  // liveness probe would burn the whole timeout on it instead of reporting
  // "stopped".
  const deadline = Date.now() + timeoutMs;
  while (isProcessAlive(pid) && isQmdMcpPid(pid)) {
    if (Date.now() >= deadline) return { outcome: "timeout", pid };
    await delay(pollMs);
  }

  // Idempotent: the daemon usually unlinked its own file already, and the
  // identity check leaves a pidfile that another daemon republished alone.
  removeMcpDaemonPidfileIfRecords(pidFile, pid);
  return { outcome: "stopped", pid };
}

/**
 * Stop the daemon described by `pidFile`.
 *
 * Only `running` is signalled. Absent/malformed/stale files are dropped
 * without touching any process; `unreadable` files are left exactly as they
 * are (they may describe a live daemon we cannot inspect). A live daemon keeps
 * its pidfile until its PID is actually gone, so a daemon that ignores SIGTERM
 * is not orphaned as untracked: the caller sees `timeout` and can retry.
 *
 * Each decision is taken under the critical section and re-verified there, so
 * a stale file can only be dropped if it is *still* stale at that moment; if a
 * concurrent start republished the file, this loop decides again with the new
 * content (and stops that daemon) instead of deleting its file.
 */
export async function stopMcpDaemon(
  pidFile: string,
  options: { timeoutMs?: number; pollMs?: number } = {},
): Promise<McpDaemonStopOutcome> {
  const timeoutMs = options.timeoutMs ?? MCP_DAEMON_STOP_TIMEOUT_MS;
  const pollMs = options.pollMs ?? MCP_DAEMON_STOP_POLL_MS;

  for (let attempt = 0; attempt < STOP_DECISION_ATTEMPTS; attempt++) {
    const state = readMcpDaemonPidState(pidFile);
    switch (state.status) {
      case "absent":
        return { outcome: "not-running" };
      case "unreadable":
        // Fail safe: no signal, no unlink — the file may name a live daemon.
        return { outcome: "unreadable", error: state.error };
      case "running":
        return stopRunningDaemon(pidFile, state.pid, timeoutMs, pollMs);
      case "malformed":
      case "stale": {
        const cleanup = removeStaleMcpDaemonPidfile(pidFile);
        if (cleanup.status === "removed") {
          const reason = cleanup.state.status === "stale" ? cleanup.state.reason : "malformed";
          return { outcome: "cleaned-stale", reason };
        }
        if (cleanup.status === "locked-out") return { outcome: "busy", error: cleanup.error };
        // "kept": the file changed under us — decide again from what it says now.
        break;
      }
    }
  }

  return { outcome: "busy", error: "PID file kept changing while stopping" };
}
