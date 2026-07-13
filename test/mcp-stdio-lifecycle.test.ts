/**
 * Lifecycle tests for the stdio MCP server's EOF shutdown (#751).
 *
 * Unit tests drive registerStdioEofShutdown with an injected fake stdin
 * (mirroring the DI style of the CLI's finishSuccessfulCliCommand tests).
 * The end-to-end test spawns the real server with a piped stdin and proves
 * the process exits once stdin closes instead of orphaning to PID 1.
 */

import { describe, test, expect } from "vitest";
import { EventEmitter } from "node:events";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { registerStdioEofShutdown, createInflightGate } from "../src/mcp/server";

class FakeStdin extends EventEmitter {
  readableEnded = false;
  destroyed = false;
}

type Recorded = {
  stdin: FakeStdin;
  calls: string[];
  exitCodes: number[];
  warnings: string[];
  shutdown: () => Promise<void>;
};

function register(overrides: {
  closeServer?: () => Promise<void>;
  disposeLlm?: () => Promise<void>;
  closeStore?: () => void | Promise<void>;
  waitForIdle?: (timeoutMs: number) => Promise<boolean>;
  idleTimeoutMs?: number;
  getExitCode?: () => number | undefined;
  stderrWrite?: (chunk: string) => unknown;
  stdin?: FakeStdin;
} = {}): Recorded {
  const stdin = overrides.stdin ?? new FakeStdin();
  const calls: string[] = [];
  const exitCodes: number[] = [];
  const warnings: string[] = [];

  const shutdown = registerStdioEofShutdown({
    stdin,
    closeServer: overrides.closeServer ?? (async () => { calls.push("server-close"); }),
    disposeLlm: overrides.disposeLlm ?? (async () => { calls.push("llm-dispose"); }),
    closeStore: overrides.closeStore ?? (() => { calls.push("store-close"); }),
    waitForIdle: overrides.waitForIdle ?? (async () => { calls.push("idle-wait"); return true; }),
    idleTimeoutMs: overrides.idleTimeoutMs,
    setExitCode: (code) => { exitCodes.push(code); },
    getExitCode: overrides.getExitCode ?? (() => undefined),
    stderr: { write: overrides.stderrWrite ?? ((chunk: string) => { warnings.push(chunk); return true; }) },
  });

  return { stdin, calls, exitCodes, warnings, shutdown };
}

describe("registerStdioEofShutdown", () => {
  test("stdin 'end' tears down in order: server, llm, store, then exitCode 0", async () => {
    const r = register();

    r.stdin.emit("end");
    await r.shutdown(); // same shared promise as the event-triggered run

    expect(r.calls).toEqual(["server-close", "idle-wait", "llm-dispose", "store-close"]);
    expect(r.exitCodes).toEqual([0]);
    expect(r.warnings.join("")).toContain("Shutting down (stdin closed)");
    expect(r.warnings.filter((w) => w.startsWith("QMD Warning"))).toEqual([]);
  });

  test("stdin 'close' triggers the same teardown", async () => {
    const r = register();

    r.stdin.emit("close");
    await r.shutdown();

    expect(r.calls).toEqual(["server-close", "idle-wait", "llm-dispose", "store-close"]);
    expect(r.exitCodes).toEqual([0]);
  });

  test("is idempotent: 'end' + 'close' + manual calls share one run", async () => {
    const r = register();

    r.stdin.emit("end");
    r.stdin.emit("close");
    const first = r.shutdown();
    const second = r.shutdown();
    expect(first).toBe(second);
    await first;

    expect(r.calls).toEqual(["server-close", "idle-wait", "llm-dispose", "store-close"]);
    expect(r.exitCodes).toEqual([0]);
    // Listeners are removed during shutdown, so late events cannot re-enter.
    r.stdin.emit("end");
    r.stdin.emit("close");
    await r.shutdown();
    expect(r.exitCodes).toEqual([0]);
  });

  test("a failing step is logged, later steps still run, exit code is 1", async () => {
    const r = register({
      closeServer: async () => { throw new Error("transport already gone"); },
    });

    r.stdin.emit("end");
    await r.shutdown();

    expect(r.calls).toEqual(["idle-wait", "llm-dispose", "store-close"]);
    expect(r.warnings.join("")).toContain("server.close() failed during stdio shutdown");
    expect(r.warnings.join("")).toContain("transport already gone");
    expect(r.exitCodes).toEqual([1]);
  });

  test("every step failing still finishes the chain instead of throwing", async () => {
    const r = register({
      closeServer: async () => { throw new Error("boom-server"); },
      disposeLlm: async () => { throw new Error("boom-llm"); },
      closeStore: () => { throw new Error("boom-store"); },
    });

    r.stdin.emit("end");
    await expect(r.shutdown()).resolves.toBeUndefined();

    expect(r.warnings.filter((w) => w.startsWith("QMD Warning"))).toHaveLength(3);
    expect(r.exitCodes).toEqual([1]);
  });

  test("stdin that already ended before registration still shuts down", async () => {
    const stdin = new FakeStdin();
    stdin.readableEnded = true;

    const r = register({ stdin });
    await r.shutdown();

    expect(r.calls).toEqual(["server-close", "idle-wait", "llm-dispose", "store-close"]);
    expect(r.exitCodes).toEqual([0]);
  });

  test("stdin destroyed before registration still shuts down", async () => {
    const stdin = new FakeStdin();
    stdin.destroyed = true;

    const r = register({ stdin });
    await r.shutdown();

    expect(r.exitCodes).toEqual([0]);
  });

  test("a drain deadline miss is logged but does not fail the shutdown", async () => {
    const r = register({
      waitForIdle: async () => false,
    });

    r.stdin.emit("end");
    await r.shutdown();

    expect(r.warnings.join("")).toContain("in-flight request did not settle");
    expect(r.exitCodes).toEqual([0]);
  });

  test("a successful shutdown preserves an earlier nonzero exit code", async () => {
    const r = register({ getExitCode: () => 1 });

    r.stdin.emit("end");
    await r.shutdown();

    // No setExitCode(0) call — the earlier failure status stays visible.
    expect(r.exitCodes).toEqual([]);
  });

  test("a failing shutdown still sets exit code 1 over an earlier 0", async () => {
    const r = register({
      getExitCode: () => 0,
      closeStore: () => { throw new Error("boom-store"); },
    });

    r.stdin.emit("end");
    await r.shutdown();

    expect(r.exitCodes).toEqual([1]);
  });

  test("a throwing stderr cannot break the teardown chain", async () => {
    const r = register({
      stderrWrite: () => { throw new Error("EPIPE"); },
    });

    r.stdin.emit("end");
    await expect(r.shutdown()).resolves.toBeUndefined();

    expect(r.calls).toEqual(["server-close", "idle-wait", "llm-dispose", "store-close"]);
    expect(r.exitCodes).toEqual([0]);
  });

  test("skips the llm-dispose step when disposeLlm is omitted (store owns it)", async () => {
    const stdin = new FakeStdin();
    const calls: string[] = [];
    const exitCodes: number[] = [];

    // Mirror how startMcpServer wires it: no disposeLlm — store.close() disposes
    // the store's own LlamaCpp, so there must be no extra global disposal step.
    const shutdown = registerStdioEofShutdown({
      stdin,
      closeServer: async () => { calls.push("server-close"); },
      waitForIdle: async () => { calls.push("idle-wait"); return true; },
      closeStore: () => { calls.push("store-close"); },
      setExitCode: (code) => { exitCodes.push(code); },
      getExitCode: () => undefined,
      stderr: { write: () => true },
    });

    stdin.emit("end");
    await shutdown();

    expect(calls).toEqual(["server-close", "idle-wait", "store-close"]);
    expect(exitCodes).toEqual([0]);
  });
});

describe("createInflightGate", () => {
  test("waitForIdle resolves immediately when nothing is tracked", async () => {
    const gate = createInflightGate();
    await expect(gate.waitForIdle(1000)).resolves.toBe(true);
  });

  test("waitForIdle waits for a tracked handler to settle", async () => {
    const gate = createInflightGate();
    let release!: () => void;
    const handler = gate.track(() => new Promise<void>((resolve) => { release = resolve; }));

    const running = handler();
    const idle = gate.waitForIdle(5000);

    release();
    await running;
    await expect(idle).resolves.toBe(true);
  });

  test("waitForIdle reports a missed deadline without throwing", async () => {
    const gate = createInflightGate();
    let release!: () => void;
    const handler = gate.track(() => new Promise<void>((resolve) => { release = resolve; }));

    const running = handler();
    await expect(gate.waitForIdle(20)).resolves.toBe(false);

    release();
    await running;
    await expect(gate.waitForIdle(20)).resolves.toBe(true);
  });

  test("a rejecting handler still releases the gate and keeps rejecting", async () => {
    const gate = createInflightGate();
    const handler = gate.track(async () => { throw new Error("handler failed"); });

    await expect(handler()).rejects.toThrow("handler failed");
    await expect(gate.waitForIdle(1000)).resolves.toBe(true);
  });
});

describe("qmd mcp stdio process lifecycle", () => {
  const repoRoot = fileURLToPath(new URL("..", import.meta.url));
  const cliPath = join(repoRoot, "src", "cli", "qmd.ts");

  test("exits cleanly after serving a request once stdin closes", async () => {
    const workDir = await mkdtemp(join(tmpdir(), "qmd-stdio-lifecycle-"));
    // Declared outside try so the finally can always reap the child — a failure
    // (timeout, assertion) before stdin.end() would otherwise leak exactly the
    // orphan process this test is about.
    let child: ReturnType<typeof spawn> | undefined;
    try {
      await writeFile(join(workDir, "index.yml"), "collections: {}\n");

      const runtimeArgs = process.versions.bun
        ? [cliPath, "mcp"]
        : ["--import", "tsx", cliPath, "mcp"];

      child = spawn(process.execPath, runtimeArgs, {
        cwd: repoRoot,
        env: {
          ...process.env,
          INDEX_PATH: join(workDir, "lifecycle.sqlite"),
          QMD_CONFIG_DIR: workDir,
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      const stderrChunks: string[] = [];
      child.stderr.on("data", (chunk) => stderrChunks.push(String(chunk)));

      // Complete one request/response round-trip so EOF arrives on a live,
      // already-connected server rather than during startup.
      const response = await new Promise<string>((resolve, reject) => {
        let buffer = "";
        const onData = (chunk: Buffer) => {
          buffer += String(chunk);
          if (buffer.includes("\n")) {
            child.stdout.off("data", onData);
            resolve(buffer);
          }
        };
        child.stdout.on("data", onData);
        child.once("error", reject);
        child.once("exit", (code) =>
          reject(new Error(`server exited before responding (code ${code}): ${stderrChunks.join("")}`))
        );
        child.stdin.write(
          JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: {
              protocolVersion: "2025-06-18",
              capabilities: {},
              clientInfo: { name: "lifecycle-test", version: "1.0.0" },
            },
          }) + "\n"
        );
      });
      expect(response).toContain('"jsonrpc":"2.0"');

      // Parent goes away: close stdin and require a clean, prompt exit.
      const exitCode = await new Promise<number | null>((resolve, reject) => {
        child.removeAllListeners("exit");
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(new Error(`server did not exit after stdin EOF: ${stderrChunks.join("")}`));
        }, 30_000);
        child.once("exit", (code) => {
          clearTimeout(timer);
          resolve(code);
        });
        child.stdin.end();
      });

      expect(exitCode).toBe(0);

      // The exit must have come from the EOF shutdown path, not from the event
      // loop happening to drain on its own (which also exits 0 whenever no
      // model is loaded — the pre-#751-fix false-negative). The breadcrumb is
      // written by registerStdioEofShutdown before teardown starts.
      expect(stderrChunks.join("")).toContain("Shutting down (stdin closed)");

      // Sanity: no WAL sidecar survives a clean database shutdown on the node
      // child (explicit close or final-connection teardown both checkpoint).
      // bun:sqlite can retain the sidecar after a clean close depending on
      // platform, so the bun child is not asserted on.
      if (!process.versions.bun) {
        expect(existsSync(join(workDir, "lifecycle.sqlite-wal"))).toBe(false);
      }
    } finally {
      if (child && child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
      await rm(workDir, { recursive: true, force: true });
    }
  }, 60_000);
});
