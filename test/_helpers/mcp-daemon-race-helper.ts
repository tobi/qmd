/**
 * Child helper for MCP daemon pidfile cross-process tests.
 *
 * Usage: <mode> <pidFile> [durationMs]
 *   hold    take the pidfile critical section the way a competing qmd would
 *           (exclusive write transaction on the sidecar database), print HOLD,
 *           keep it for `durationMs`, then release
 *   claim   claim the pidfile for this process, print the claim outcome, and
 *           stay alive for `durationMs` so the winner keeps looking like a
 *           running daemon
 *
 * Env:
 *   DAEMON_RACE_BARRIER=1  print READY and wait for a line on stdin before
 *                          acting, so contenders collide on the transition
 *                          instead of on spawn cost.
 *
 * Trailing qmd tokens keep the child's command line looking like a qmd
 * invocation, so PID identity checks classify it as a daemon. Output is flushed
 * by natural exit: the helper never calls process.exit(), which could truncate
 * buffered pipe writes.
 */
import { openDatabase } from "../../src/db.ts";
import { claimMcpDaemonPidfile, mcpDaemonLockStatePath } from "../../src/cli/mcp-pid.ts";

function hold(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

async function waitForGo(): Promise<boolean> {
  process.stdout.write("READY\n");
  const { promise, resolve } = Promise.withResolvers<boolean>();
  process.stdin.resume();
  process.stdin.once("data", () => {
    process.stdin.pause();
    resolve(true);
  });
  // A closed stdin means the parent is gone; do not hang forever.
  process.stdin.once("end", () => resolve(false));
  return promise;
}

async function main(): Promise<void> {
  const mode = process.argv[2];
  const pidFile = process.argv[3];
  const durationMs = Number(process.argv[4] ?? "1000");
  if (!mode || !pidFile || !Number.isFinite(durationMs)) {
    console.error("usage: mcp-daemon-race-helper.ts <hold|claim> <pidFile> [durationMs]");
    process.exitCode = 2;
    return;
  }

  if (mode === "hold") {
    const db = openDatabase(mcpDaemonLockStatePath(pidFile));
    db.exec("PRAGMA busy_timeout = 0");
    if (process.env.DAEMON_RACE_BARRIER === "1" && !(await waitForGo())) return;
    db.exec("BEGIN EXCLUSIVE");
    process.stdout.write("HOLD\n");
    await hold(durationMs);
    db.exec("ROLLBACK");
    db.close();
    process.stdout.write("RELEASED\n");
    return;
  }

  if (mode === "claim") {
    if (process.env.DAEMON_RACE_BARRIER === "1" && !(await waitForGo())) return;
    const claim = claimMcpDaemonPidfile(pidFile, process.pid);
    if (claim.claimed) {
      process.stdout.write(`CLAIMED ${process.pid}\n`);
    } else if (claim.reason === "running") {
      process.stdout.write(`RUNNING ${claim.runningPid}\n`);
    } else {
      process.stdout.write(`UNCLAIMABLE ${claim.error}\n`);
    }
    await hold(durationMs);
    return;
  }

  console.error(`unknown mode: ${mode}`);
  process.exitCode = 2;
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
