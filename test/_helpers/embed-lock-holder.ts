/**
 * Child helper for embed-lock cross-process tests (#825).
 *
 * Args: <lockPath> [ignored qmd tokens...]
 * The trailing tokens keep the child's command line looking like a qmd
 * invocation, so PID-based liveness checks still classify it as qmd.
 *
 * The lock is held until the parent releases it, never for a duration of the
 * child's own choosing: the parent writes `RELEASE` and only then does the
 * holder leave the critical section. A caller asserting "a live holder blocks"
 * therefore cannot lose a race against the holder's clock when the runner is
 * loaded, and no caller has to guess a hold time.
 *
 * Env:
 *   EMBED_LOCK_BARRIER=1   print READY and wait for a line on stdin (GO) before
 *                          attempting, so contenders start together.
 *   EMBED_LOCK_TRY_ONCE=1  attempt once and report ACQUIRED/BUSY instead of
 *                          HOLD (contention tests); the acquirer holds until
 *                          RELEASE.
 *
 * The holder also releases when stdin closes (the parent died, taking its
 * cleanup with it) or after a long backstop, so a leaked child cannot wedge a
 * later run.
 *
 * Output is always flushed by natural exit; the helper never calls
 * process.exit(), which could truncate buffered pipe writes.
 */
import { acquireEmbedLock, tryAcquireEmbedLock } from "../../src/cli/embed-lock.ts";

/** Returned by nextCommand once the parent closed stdin. */
const ENDED = "ended";
/**
 * Leak backstop only: no assertion depends on this value. A parent that dies
 * without releasing still gets its lock back within this window.
 */
const RELEASE_BACKSTOP_MS = 60_000;

const commandQueue: string[] = [];
const commandWaiters: Array<(line: string) => void> = [];
let stdinBuffer = "";
let stdinEnded = false;

process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk: string) => {
  stdinBuffer += chunk;
  let newline = stdinBuffer.indexOf("\n");
  while (newline !== -1) {
    const line = stdinBuffer.slice(0, newline).trim();
    stdinBuffer = stdinBuffer.slice(newline + 1);
    const waiter = commandWaiters.shift();
    if (waiter) waiter(line);
    else commandQueue.push(line);
    newline = stdinBuffer.indexOf("\n");
  }
});
process.stdin.on("end", () => {
  stdinEnded = true;
  // Wake every reader: none of them can make progress any more.
  for (const waiter of commandWaiters.splice(0)) waiter(ENDED);
});

/** The next line the parent wrote; ENDED once stdin closes. */
function nextCommand(): Promise<string> {
  const queued = commandQueue.shift();
  if (queued !== undefined) return Promise.resolve(queued);
  if (stdinEnded) return Promise.resolve(ENDED);
  const { promise, resolve } = Promise.withResolvers<string>();
  commandWaiters.push(resolve);
  return promise;
}

/** Start gate for EMBED_LOCK_BARRIER; false when the parent went away. */
async function waitForGo(): Promise<boolean> {
  process.stdout.write("READY\n");
  for (;;) {
    const command = await nextCommand();
    if (command === ENDED) return false;
    if (command.length > 0) return true;
  }
}

async function waitForReleaseCommand(): Promise<void> {
  for (;;) {
    const command = await nextCommand();
    if (command === ENDED) return;
    if (command.toUpperCase() === "RELEASE") return;
  }
}

/** Keep the lock until the parent's RELEASE (or a failure path frees it). */
async function holdUntilRelease(): Promise<void> {
  const { promise: backstop, resolve: onBackstop } = Promise.withResolvers<void>();
  const timer = setTimeout(onBackstop, RELEASE_BACKSTOP_MS);
  try {
    await Promise.race([waitForReleaseCommand(), backstop]);
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<void> {
  const lockPath = process.argv[2];
  if (!lockPath) {
    console.error("usage: embed-lock-holder.ts <lockPath>");
    process.exitCode = 2;
    return;
  }

  try {
    process.stdin.resume();

    if (process.env.EMBED_LOCK_BARRIER === "1" && !(await waitForGo())) return;

    if (process.env.EMBED_LOCK_TRY_ONCE === "1") {
      const result = acquireEmbedLock(lockPath);
      if (result.status === "busy") {
        process.stdout.write("BUSY\n");
        return;
      }
      process.stdout.write("ACQUIRED\n");
      await holdUntilRelease();
      result.handle.release();
      process.stdout.write("RELEASED\n");
      return;
    }

    const handle = tryAcquireEmbedLock(lockPath);
    if (!handle) {
      console.error("child failed to acquire");
      process.exitCode = 2;
      return;
    }

    process.stdout.write("HOLD\n");
    await holdUntilRelease();
    handle.release();
    process.stdout.write("RELEASED\n");
  } finally {
    // A still-resumed stdin would keep the loop (and the process) alive.
    process.stdin.pause();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
