/**
 * Child helper for cross-process collection-config tests.
 *
 * Args: <configPath> <op> <name> <value> [extra]
 *
 * Ops:
 *   add-collection   addCollection(name, value)
 *   add-context      addContext(name, value, extra)
 *   hold-update      update() that prints HOLDING, waits for a line on stdin,
 *                    then adds collections `name` and `extra`. Used to hold the
 *                    config lock across another process's mutation.
 *   stress           value rounds of add+remove of a scratch collection, for
 *                    read-while-write assertions
 *
 * Output: ATTEMPT (before mutating), DONE / ERROR, plus op-specific lines.
 *
 * Env:
 *   CONFIG_CHILD_BARRIER=1  print READY and wait for a line on stdin before
 *                           starting, so contenders start together.
 *
 * Output is flushed by natural exit; the helper never calls process.exit(),
 * which could truncate buffered pipe writes.
 */
import { readSync } from "node:fs";
import { CollectionConfigScope } from "../../src/collections.ts";
import { sleepSync } from "../../src/db.ts";

/**
 * Transient stdin read failures: no byte buffered yet (`EAGAIN`, with
 * `EWOULDBLOCK` as its alias), or a signal that interrupted the read.
 */
function isTransientReadError(error: unknown): boolean {
  if (typeof error !== "object" || error === null || !("code" in error)) return false;
  const code = error.code;
  return code === "EAGAIN" || code === "EWOULDBLOCK" || code === "EINTR";
}

/** Block until one line arrives on stdin (or stdin closes). */
function waitForGoSync(): void {
  const byte = Buffer.alloc(1);
  let transient = 0;
  for (;;) {
    let read: number;
    try {
      read = readSync(0, byte, 0, 1, null);
    } catch (error) {
      if (!isTransientReadError(error)) throw error;
      // Node hands the child a non-blocking stdin pipe, so a byte that has not
      // landed yet reports EAGAIN instead of waiting. This barrier runs inside
      // a synchronous callback with no event loop to yield to, so park briefly
      // between attempts instead of spinning (same backoff as db.ts retries).
      sleepSync(Math.min(5 + transient, 25));
      transient++;
      continue;
    }
    transient = 0;
    if (read === 0) return; // stdin closed: the parent hung up.
    if (byte.toString("utf-8") === "\n") return;
  }
}

function run(scope: CollectionConfigScope, op: string, name: string, value: string, extra: string): void {
  switch (op) {
    case "add-collection":
      scope.addCollection(name, value);
      return;
    case "add-context":
      scope.addContext(name, value, extra);
      return;
    case "hold-update":
      scope.update((config) => {
        process.stdout.write("HOLDING\n");
        waitForGoSync();
        config.collections[name] = { path: value, pattern: "**/*.md" };
        if (extra) config.collections[extra] = { path: "/held/after", pattern: "**/*.md" };
      });
      return;
    case "stress": {
      const rounds = Number(value || "100");
      for (let round = 0; round < rounds; round++) {
        const scratch = `stress-${process.pid}-${round}`;
        scope.addCollection(scratch, `/tmp/${scratch}`, "**/*.md");
        scope.removeCollection(scratch);
      }
      return;
    }
    default:
      throw new Error(`unknown op: ${op}`);
  }
}

function main(): void {
  const [configPath, op, name = "", value = "", extra = ""] = process.argv.slice(2);
  if (!configPath || !op) {
    console.error("usage: config-mutation-child.ts <configPath> <op> <name> <value> [extra]");
    process.exitCode = 2;
    return;
  }

  if (process.env.CONFIG_CHILD_BARRIER === "1") {
    process.stdout.write("READY\n");
    waitForGoSync();
  }

  const scope = new CollectionConfigScope({ configPath });
  try {
    process.stdout.write("ATTEMPT\n");
    run(scope, op, name, value, extra);
    process.stdout.write("DONE\n");
  } catch (error) {
    process.stderr.write(`ERROR ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  } finally {
    scope.close();
  }
}

main();
