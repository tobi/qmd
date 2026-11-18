/**
 * Child helper for embed-lock cross-process tests (#825).
 *
 * Args: <lockPath> <holdMs> [ignored qmd tokens...]
 * The trailing tokens keep the child's command line looking like a qmd
 * invocation, so PID-based liveness checks still classify it as qmd.
 *
 * Env:
 *   EMBED_LOCK_BARRIER=1   print READY and wait for a line on stdin before
 *                          attempting, so contenders start together.
 *   EMBED_LOCK_TRY_ONCE=1  attempt once and report ACQUIRED/BUSY instead of
 *                          HOLD (contention tests).
 *
 * Output is always flushed by natural exit; the helper never calls
 * process.exit(), which could truncate buffered pipe writes.
 */
import { acquireEmbedLock, tryAcquireEmbedLock } from "../../src/cli/embed-lock.ts";

function waitForGo(): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  process.stdin.resume();
  process.stdin.once("data", () => {
    process.stdin.pause();
    resolve();
  });
  return promise;
}

function hold(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

async function main(): Promise<void> {
  const lockPath = process.argv[2];
  const holdMs = Number(process.argv[3] ?? "1000");
  if (!lockPath || !Number.isFinite(holdMs)) {
    console.error("usage: embed-lock-holder.ts <lockPath> <holdMs>");
    process.exitCode = 2;
    return;
  }

  if (process.env.EMBED_LOCK_BARRIER === "1") {
    process.stdout.write("READY\n");
    await waitForGo();
  }

  if (process.env.EMBED_LOCK_TRY_ONCE === "1") {
    const result = acquireEmbedLock(lockPath);
    if (result.status === "busy") {
      process.stdout.write("BUSY\n");
      return;
    }
    process.stdout.write("ACQUIRED\n");
    await hold(holdMs);
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
  await hold(holdMs);
  handle.release();
  process.stdout.write("RELEASED\n");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
