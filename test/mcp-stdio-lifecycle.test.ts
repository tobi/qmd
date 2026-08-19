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
import { registerStdioEofShutdown, createInflightGate, ServerShuttingDownError, startMcpServer } from "../src/mcp/server";
import { LlamaCpp } from "../src/llm.ts";
import type { QMDStore } from "../src/index.ts";
import { createShutdownCoordinator } from "../src/shutdown.ts";
import type { ShutdownTrigger } from "../src/shutdown.ts";

class FakeStdin extends EventEmitter {
  readableEnded = false;
  destroyed = false;
}

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

describe("registerStdioEofShutdown", () => {
  test("stdin 'end' and 'close' share one coordinator trigger", async () => {
    const triggers: ShutdownTrigger[] = [];
    const started = deferred();
    const gate = deferred();
    const stdin = new FakeStdin();
    const warnings: string[] = [];
    const shutdown = registerStdioEofShutdown({
      stdin,
      shutdown: async (trigger) => {
        triggers.push(trigger);
        started.resolve();
        await gate.promise;
      },
      stderr: { write: (chunk) => { warnings.push(chunk); return true; } },
    });

    stdin.emit("end");
    stdin.emit("close");
    const first = shutdown();
    const second = shutdown();
    expect(first).toBe(second);
    await started.promise;
    expect(triggers).toEqual([{ kind: "stdin-eof", exitCode: 0 }]);
    expect(warnings.join("")).toContain("Shutting down (stdin closed)");
    gate.resolve();
    await first;
  });

  test("stdin that already ended before registration still shuts down", async () => {
    const calls: string[] = [];
    const stdin = new FakeStdin();
    stdin.readableEnded = true;
    const shutdown = registerStdioEofShutdown({
      stdin,
      shutdown: async () => { calls.push("shutdown"); },
      stderr: { write: () => true },
    });
    await shutdown();
    expect(calls).toEqual(["shutdown"]);
  });

  test("a throwing stderr cannot break the trigger", async () => {
    const stdin = new FakeStdin();
    const shutdown = registerStdioEofShutdown({
      stdin,
      shutdown: async () => undefined,
      stderr: { write: () => { throw new Error("EPIPE"); } },
    });
    stdin.emit("end");
    await expect(shutdown()).resolves.toBeUndefined();
  });
});

describe("stdio EOF with active work", () => {
  test("does not dispose LLM or store until the tracked handler releases", async () => {
    const inflight = createInflightGate();
    const calls: string[] = [];
    const handlerGate = deferred();
    const handlerStarted = deferred();
    const handler = inflight.track(async () => {
      handlerStarted.resolve();
      await handlerGate.promise;
    });
    const running = handler();
    await handlerStarted.promise;

    const coordinator = createShutdownCoordinator({
      closeAdmission() {
        inflight.closeAdmission();
        calls.push("close-admission");
      },
      stopServing() { calls.push("stop-serving"); },
      requestAbort() { calls.push("request-abort"); },
      waitForInflight: () => inflight.waitForIdle(),
      waitForLlmIdle: async () => { calls.push("llm-idle"); },
      async disposeLlm() { calls.push("llm-dispose"); },
      closeStore() { calls.push("store-close"); },
      setExitCode(code) { calls.push(`exit:${code}`); },
      logError() { calls.push("log-error"); },
    });

    const stdin = new FakeStdin();
    const shutdown = registerStdioEofShutdown({
      stdin,
      shutdown: coordinator.shutdown,
      stderr: { write: () => true },
    });
    stdin.emit("end");
    const shutting = shutdown();
    await Promise.resolve();
    expect(calls).toContain("close-admission");
    expect(calls).not.toContain("llm-dispose");
    expect(calls).not.toContain("store-close");

    handlerGate.resolve();
    await running;
    await shutting;
    expect(calls).toEqual([
      "close-admission",
      "stop-serving",
      "request-abort",
      "llm-idle",
      "llm-dispose",
      "store-close",
      "exit:0",
    ]);
  });

  test("a new request after EOF is rejected", async () => {
    const inflight = createInflightGate();
    inflight.closeAdmission();
    await expect(inflight.run(async () => "nope")).rejects.toBeInstanceOf(ServerShuttingDownError);
  });
});

describe("createInflightGate", () => {
  test("waitForIdle resolves immediately when nothing is tracked", async () => {
    const gate = createInflightGate();
    await expect(gate.waitForIdle()).resolves.toBeUndefined();
  });

  test("waitForIdle waits for a tracked handler to settle", async () => {
    const gate = createInflightGate();
    let release!: () => void;
    const handler = gate.track(() => new Promise<void>((resolve) => { release = resolve; }));

    const running = handler();
    const idle = gate.waitForIdle();

    release();
    await running;
    await expect(idle).resolves.toBeUndefined();
  });

  test("a rejecting handler still releases the gate and keeps rejecting", async () => {
    const gate = createInflightGate();
    const handler = gate.track(async () => { throw new Error("handler failed"); });

    await expect(handler()).rejects.toThrow("handler failed");
    await expect(gate.waitForIdle()).resolves.toBeUndefined();
  });

  test("closed admission rejects new work without incrementing the count", async () => {
    const gate = createInflightGate();
    gate.closeAdmission();
    await expect(gate.run(async () => 1)).rejects.toBeInstanceOf(ServerShuttingDownError);
    expect(gate.getActiveCount()).toBe(0);
  });
});

describe("stdio MCP initialization vs EOF", () => {
  test("store initialization finishes before an already-ended stdin can dispose it", async () => {
    const order: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const origDispose = llm.dispose.bind(llm);
    llm.dispose = async () => {
      order.push("llm-dispose");
      return origDispose();
    };
    const store = {
      internal: { llm, close() {} },
      dbPath: ":memory:",
      async getStatus() {
        order.push("init-status");
        return { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
      },
      async getGlobalContext() { return undefined; },
      async getDefaultCollectionNames() { return []; },
      async close() { order.push("store-close"); },
    } as unknown as QMDStore;

    const stdin = new FakeStdin();
    stdin.readableEnded = true;
    const handle = await startMcpServer({
      store,
      stdin,
      createTransport: () => ({ close: async () => { order.push("transport-close"); } }),
      eofHardStop: () => { order.push("eof-hard-stop"); },
    });
    await handle.shutdown();

    expect(order[0]).toBe("init-status");
    expect(order.indexOf("init-status")).toBeLessThan(order.indexOf("store-close"));
  });

  test("EOF watchdog hard-stops instead of disposing under a hung drain", async () => {
    const stops: string[] = [];
    let fire!: () => void;
    const { createEofWatchdog } = await import("../src/shutdown.ts");
    const watchdog = createEofWatchdog({
      graceMs: 1,
      hardStop() { stops.push("hard-stop"); },
      schedule(fn) {
        fire = fn;
        return { clear() { stops.push("cleared"); } };
      },
    });
    watchdog.arm();
    fire();
    expect(stops).toEqual(["hard-stop"]);
    expect(stops).not.toContain("dispose");
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
