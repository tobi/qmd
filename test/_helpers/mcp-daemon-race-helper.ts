/**
 * Child helper for MCP daemon pidfile cross-process tests.
 *
 * Usage: <mode> <pidFile> [ignored qmd tokens...]
 *   hold    take the pidfile critical section the way a competing qmd would
 *           (exclusive write transaction on the sidecar database), print HOLD,
 *           and keep it until the parent releases or kills this process
 *   claim   claim the pidfile for this process and print the claim outcome; a
 *           winner (CLAIMED) stays alive until the parent releases it, so it
 *           keeps looking like a running daemon for exactly as long as the
 *           assertions need it
 *
 * Lifetime is always parent-controlled, never a timer here: a stalled runner
 * must not be able to expire a holder the test is about to inspect. The parent
 * releases a helper by closing its stdin; when that happens — or when the parent
 * dies, which closes it too — the helper finishes and exits.
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

let stdinEnded = false;

/**
 * Drop stdin before returning. A *paused* stdin still holds its pipe handle,
 * which keeps the process alive on Node forever; destroying it means every
 * path out of main() exits once its work is done.
 */
function detachStdin(): void {
  process.stdin.pause();
  process.stdin.destroy();
}

/** Stay alive until the parent releases us by closing stdin. */
function waitForRelease(): Promise<void> {
  if (stdinEnded) return Promise.resolve();
  const { promise, resolve } = Promise.withResolvers<void>();
  process.stdin.once("end", resolve);
  process.stdin.resume();
  return promise;
}

/** Barrier gate: true to act, false when the parent went away. */
async function waitForGo(): Promise<boolean> {
  process.stdout.write("READY\n");
  const { promise, resolve } = Promise.withResolvers<boolean>();
  process.stdin.resume();
  process.stdin.once("data", () => resolve(true));
  // A closed stdin means the parent is gone; do not hang forever.
  process.stdin.once("end", () => resolve(false));
  return promise;
}

async function main(): Promise<void> {
  process.stdin.on("end", () => {
    stdinEnded = true;
  });

  const mode = process.argv[2];
  const pidFile = process.argv[3];
  if (!mode || !pidFile) {
    console.error("usage: mcp-daemon-race-helper.ts <hold|claim> <pidFile>");
    process.exitCode = 2;
    return;
  }

  try {
    if (mode === "hold") {
      const db = openDatabase(mcpDaemonLockStatePath(pidFile));
      db.exec("PRAGMA busy_timeout = 0");
      if (process.env.DAEMON_RACE_BARRIER === "1" && !(await waitForGo())) return;
      db.exec("BEGIN EXCLUSIVE");
      process.stdout.write("HOLD\n");
      await waitForRelease();
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
        // The pidfile now describes this process as a live daemon: it must not
        // exit until the parent has finished inspecting the file.
        await waitForRelease();
      } else if (claim.reason === "running") {
        process.stdout.write(`RUNNING ${claim.runningPid}\n`);
      } else {
        process.stdout.write(`UNCLAIMABLE ${claim.error}\n`);
      }
      return;
    }

    console.error(`unknown mode: ${mode}`);
    process.exitCode = 2;
  } finally {
    detachStdin();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
