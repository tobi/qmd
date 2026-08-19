import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { LlamaCpp, SessionReleasedError, setDefaultLlamaCpp } from "../src/llm.ts";
import {
  CliExit,
  attachForegroundMcpShutdown,
  createCliShutdownCoordinator,
  handleProcessSignalForTest,
  installMainSignalHandlers,
  isSupervisedChild,
  runOwnedCliDispatch,
  setActiveShutdownForTest,
  resetCliStoreForTest,
  setGenerateEmbeddingsForTest,
  setStoreCloseObserverForTest,
  vectorIndex,
} from "../src/cli/qmd.ts";
import { createAdmissionLease } from "../src/shutdown.ts";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await sleep(25);
  }
  throw new Error("waitFor timed out");
}

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

describe("CLI signal shutdown seams", () => {
  let previousExitCode: string | number | undefined;
  beforeEach(() => {
    previousExitCode = process.exitCode;
  });
  afterEach(() => {
    setActiveShutdownForTest(undefined);
    process.exitCode = previousExitCode;
  });

  function installTestCoordinator(overrides: Parameters<typeof createCliShutdownCoordinator>[0] = {}) {
    const exitCodes: number[] = [];
    const calls: string[] = [];
    const coordinator = createCliShutdownCoordinator({
      holdOpen: () => () => { calls.push("hold-released"); },
      hardStop: (error): never => {
        calls.push("hard-stop");
        throw error instanceof Error ? error : new Error(String(error));
      },
      setExitCode: (code) => { exitCodes.push(code); },
      logError: () => { calls.push("log-error"); },
      disposeLlm: async () => { calls.push("dispose-llm"); },
      closeStore: () => { calls.push("close-store"); },
      // Tests here deliberately leave a shutdown hung. An inert scheduler keeps
      // them from arming real 30s watchdogs that would later fire during
      // unrelated tests; watchdog behaviour itself is covered in shutdown.test.ts.
      schedule: () => ({ clear() {} }),
      ...overrides,
    });
    setActiveShutdownForTest(coordinator.shutdown);
    return { coordinator, exitCodes, calls };
  }

  test("SIGINT and SIGTERM preserve 130 and 143", async () => {
    const sigint = installTestCoordinator();
    await handleProcessSignalForTest("SIGINT");
    expect(sigint.exitCodes).toEqual([130]);

    setActiveShutdownForTest(undefined);
    const sigterm = installTestCoordinator();
    await handleProcessSignalForTest("SIGTERM");
    expect(sigterm.exitCodes).toEqual([143]);
  });

  test("concurrent and repeated child shutdown calls share one promise", async () => {
    const { coordinator } = installTestCoordinator();
    const first = handleProcessSignalForTest("SIGINT");
    const second = handleProcessSignalForTest("SIGTERM");
    const third = coordinator.shutdown({ kind: "complete", exitCode: 0 });
    expect(first).toBe(second);
    expect(second).toBe(third);
    await Promise.all([first, second, third]);
  });

  test("an active non-LLM command prevents store/LLM disposal until it unwinds", async () => {
    const lease = createAdmissionLease();
    const started = deferred();
    const gate = deferred();
    const { calls, exitCodes } = installTestCoordinator({
      waitForInflight: () => lease.waitForIdle(),
    });

    const command = lease.run(async () => {
      started.resolve();
      await gate.promise;
    });
    await started.promise;

    const shutting = handleProcessSignalForTest("SIGINT");
    await Promise.resolve();
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");

    gate.resolve();
    await command;
    await shutting;
    expect(calls.filter((c) => c === "dispose-llm")).toEqual(["dispose-llm"]);
    expect(calls.filter((c) => c === "close-store")).toEqual(["close-store"]);
    expect(exitCodes).toEqual([130]);
  });

  test("late-created owned resources are still disposed after the command lease releases", async () => {
    const lease = createAdmissionLease();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    let created = false;
    const extra: string[] = [];
    const { calls } = installTestCoordinator({
      waitForInflight: () => lease.waitForIdle(),
      async disposeLlm() {
        extra.push(created ? "dispose-late-llm" : "dispose-missing-llm");
        await llm.dispose();
      },
    });

    const command = lease.run(async () => {
      started.resolve();
      await gate.promise;
      created = true;
    });
    await started.promise;

    const shutting = handleProcessSignalForTest("SIGTERM");
    await Promise.resolve();
    expect(extra).not.toContain("dispose-late-llm");
    expect(calls).not.toContain("close-store");

    gate.resolve();
    await command;
    await shutting;
    expect(extra).toContain("dispose-late-llm");
    expect(calls).toContain("close-store");
  });

  test("installMainSignalHandlers uninstalls without sending real process signals", () => {
    const { coordinator } = installTestCoordinator();
    const uninstall = installMainSignalHandlers(coordinator.shutdown);
    expect(typeof uninstall).toBe("function");
    uninstall();
  });

  test("a supervised child ignores duplicate signals instead of hard-stopping", async () => {
    const { coordinator } = installTestCoordinator({
      waitForInflight: () => new Promise<void>(() => {}),
    });
    const hardStops: string[] = [];
    const uninstall = installMainSignalHandlers(coordinator.shutdown, {
      supervised: true,
      hardStop: ((error: unknown) => {
        hardStops.push("hard-stop");
        throw error instanceof Error ? error : new Error(String(error));
      }) as never,
    });
    try {
      // One terminal Ctrl-C reaches the child directly and again via the
      // wrapper's forward; that pair must not kill a healthy drain.
      process.emit("SIGINT");
      process.emit("SIGINT");
      await Promise.resolve();
      expect(hardStops).toEqual([]);
    } finally {
      uninstall();
    }
  });

  test("an unsupervised child hard-stops on the second signal", async () => {
    const { coordinator } = installTestCoordinator({
      waitForInflight: () => new Promise<void>(() => {}),
    });
    const hardStops: string[] = [];
    const uninstall = installMainSignalHandlers(coordinator.shutdown, {
      supervised: false,
      hardStop: ((error: unknown) => {
        hardStops.push(error instanceof Error ? error.message : String(error));
        return undefined as never;
      }) as never,
    });
    try {
      process.emit("SIGINT");
      await Promise.resolve();
      expect(hardStops).toEqual([]);
      process.emit("SIGINT");
      expect(hardStops).toEqual(["Second SIGINT during shutdown"]);
    } finally {
      uninstall();
    }
  });

  test("isSupervisedChild reads exactly the QMD_SUPERVISED bit", () => {
    expect(isSupervisedChild({ QMD_SUPERVISED: "1" })).toBe(true);
    expect(isSupervisedChild({ QMD_SUPERVISED: "0" })).toBe(false);
    expect(isSupervisedChild({})).toBe(false);
  });
});

describe("runOwnedCliDispatch", () => {
  let previousExitCode: string | number | undefined;
  beforeEach(() => {
    previousExitCode = process.exitCode;
  });
  afterEach(() => {
    setActiveShutdownForTest(undefined);
    setGenerateEmbeddingsForTest();
    setStoreCloseObserverForTest();
    resetCliStoreForTest();
    setDefaultLlamaCpp(null);
    process.exitCode = previousExitCode;
  });

  test("SIGINT during a deferred embed error does not process.exit and waits for LLM disposal", async () => {
    const lease = createAdmissionLease();
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    const exitCodes: number[] = [];
    const coordinator = createCliShutdownCoordinator({
      waitForInflight: () => lease.waitForIdle(),
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      setExitCode: (code) => { exitCodes.push(code); },
      async disposeLlm() { calls.push("dispose-llm"); },
      closeStore() { calls.push("close-store"); },
    });
    setActiveShutdownForTest(coordinator.shutdown);

    const running = runOwnedCliDispatch({
      lease,
      shutdown: coordinator.shutdown,
      run: async () => {
        started.resolve();
        await gate.promise;
        throw new Error("embed failed after signal");
      },
    });

    await started.promise;
    const shutting = handleProcessSignalForTest("SIGINT");
    await Promise.resolve();
    expect(calls).not.toContain("dispose-llm");
    gate.resolve();
    await running;
    await shutting;
    expect(calls).toEqual(["dispose-llm", "close-store"]);
    expect(exitCodes).toEqual([130]);
    expect(process.exitCode === 130 || exitCodes.includes(130)).toBe(true);
  });

  test("an ordinary query/embed failure still performs cleanup and exits nonzero", async () => {
    const calls: string[] = [];
    const coordinator = createCliShutdownCoordinator({
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      async disposeLlm() { calls.push("dispose-llm"); },
      closeStore() { calls.push("close-store"); },
    });
    const prev = process.exitCode;
    process.exitCode = undefined;
    try {
      await runOwnedCliDispatch({
        shutdown: coordinator.shutdown,
        run: async () => { throw new Error("query failed"); },
      });
      expect(calls).toEqual(["dispose-llm", "close-store"]);
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = prev;
    }
  });

  test("the root command lease releases for expected error branches", async () => {
    const lease = createAdmissionLease();
    const coordinator = createCliShutdownCoordinator({
      waitForInflight: () => lease.waitForIdle(),
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      disposeLlm: async () => undefined,
      closeStore: () => undefined,
    });
    await runOwnedCliDispatch({
      lease,
      shutdown: coordinator.shutdown,
      run: async () => { throw new CliExit(1, "usage"); },
    });
    expect(lease.getActiveCount()).toBe(0);
  });

  test("a caught cancellation keeps signal 130 instead of overwriting with 1", async () => {
    const lease = createAdmissionLease();
    const started = deferred();
    const gate = deferred();
    const exitCodes: number[] = [];
    const coordinator = createCliShutdownCoordinator({
      waitForInflight: () => lease.waitForIdle(),
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      setExitCode: (code) => { exitCodes.push(code); },
      disposeLlm: async () => undefined,
      closeStore: () => undefined,
    });
    setActiveShutdownForTest(coordinator.shutdown);
    const prev = process.exitCode;
    process.exitCode = undefined;
    try {
      const running = runOwnedCliDispatch({
        lease,
        shutdown: coordinator.shutdown,
        run: async () => {
          started.resolve();
          await gate.promise;
          throw new SessionReleasedError("LlamaCpp is shutting down");
        },
      });
      await started.promise;
      const shutting = handleProcessSignalForTest("SIGINT");
      gate.resolve();
      await running;
      await shutting;
      expect(exitCodes).toEqual([130]);
      expect(process.exitCode === 130 || exitCodes.at(-1) === 130).toBe(true);
    } finally {
      process.exitCode = prev;
    }
  });

  test.each(["SIGINT", "SIGTERM"] as const)("%s during a blocked real embed() phase preserves 130/143 and waits", async (signal) => {
    const lease = createAdmissionLease();
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const started = deferred();
    const gate = deferred();
    const calls: string[] = [];
    Object.assign(llm as Record<string, unknown>, {
      ensureEmbedContext: async () => ({
        getEmbeddingFor: async () => {
          started.resolve();
          await gate.promise;
          return { vector: [1] };
        },
      }),
      llama: { dispose: async () => { calls.push("llama"); } },
    });
    const exitCodes: number[] = [];
    const coordinator = createCliShutdownCoordinator({
      waitForInflight: () => lease.waitForIdle(),
      waitForLlmIdle: () => llm.waitForSessionIdle(),
      disposeLlm: () => llm.dispose(),
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      setExitCode: (code) => { exitCodes.push(code); },
      closeStore: () => { calls.push("store"); },
    });
    setActiveShutdownForTest(coordinator.shutdown);

    const running = runOwnedCliDispatch({
      lease,
      shutdown: coordinator.shutdown,
      run: async () => { await llm.embed("q"); },
    });
    await started.promise;
    const shutting = handleProcessSignalForTest(signal);
    await Promise.resolve();
    expect(calls).toEqual([]);
    gate.resolve();
    await running;
    await shutting;
    expect(calls).toEqual(["llama", "store"]);
    expect(exitCodes).toEqual([signal === "SIGINT" ? 130 : 143]);
  });

  test("embed success path does not close the store before coordinator LLM disposal", async () => {
    const order: string[] = [];
    setStoreCloseObserverForTest(() => { order.push("closeDb"); });
    setGenerateEmbeddingsForTest(async () => ({
      docsProcessed: 1,
      chunksEmbedded: 1,
      errors: 0,
      durationMs: 1,
    }));
    const workDir = mkdtempSync(join(tmpdir(), "qmd-embed-order-"));
    const prevIndex = process.env.INDEX_PATH;
    process.env.INDEX_PATH = join(workDir, "index.sqlite");
    const coordinator = createCliShutdownCoordinator({
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      async disposeLlm() { order.push("llm"); },
      closeStore() { order.push("store-coord"); },
    });
    try {
      await runOwnedCliDispatch({
        shutdown: coordinator.shutdown,
        run: async () => { await vectorIndex("hf:test-model", true); },
      });
      expect(order).not.toContain("closeDb");
      expect(order.indexOf("llm")).toBeGreaterThanOrEqual(0);
      expect(order.indexOf("llm")).toBeLessThan(order.indexOf("store-coord"));
    } finally {
      if (prevIndex === undefined) delete process.env.INDEX_PATH;
      else process.env.INDEX_PATH = prevIndex;
      rmSync(workDir, { recursive: true, force: true });
    }
  });
});

describe("foreground MCP startup handoff", () => {
  let previousExitCode: string | number | undefined;
  beforeEach(() => {
    previousExitCode = process.exitCode;
  });
  afterEach(() => {
    setActiveShutdownForTest(undefined);
    process.exitCode = previousExitCode;
  });

  test("a signal during deferred module/startup cannot leave a server running", async () => {
    const started = deferred();
    const gate = deferred();
    const events: string[] = [];
    const coordinator = createCliShutdownCoordinator({
      holdOpen: () => () => undefined,
      hardStop: (error): never => { throw error instanceof Error ? error : new Error(String(error)); },
      disposeLlm: async () => { events.push("cli-dispose"); },
      closeStore: () => { events.push("cli-store"); },
    });
    setActiveShutdownForTest(coordinator.shutdown);

    const running = runOwnedCliDispatch({
      shutdown: coordinator.shutdown,
      keepAlive: () => events.includes("started"),
      run: async () => {
        await attachForegroundMcpShutdown({
          start: async () => {
            started.resolve();
            await gate.promise;
            events.push("started");
            return {
              shutdown: async () => { events.push("mcp-shutdown"); },
            };
          },
        });
      },
    });

    await started.promise;
    const shutting = handleProcessSignalForTest("SIGINT");
    await Promise.resolve();
    expect(events).toEqual([]);
    gate.resolve();
    await running;
    await shutting;
    expect(events).toContain("started");
    expect(events).toContain("mcp-shutdown");
    expect(events.indexOf("mcp-shutdown")).toBeLessThan(events.indexOf("cli-dispose") === -1 ? Number.POSITIVE_INFINITY : events.indexOf("cli-dispose"));
  });
});

describe("unsupervised direct child (no bin/qmd wrapper)", () => {
  const repoRoot = fileURLToPath(new URL("..", import.meta.url));

  const runtimeArgs = (fixture: string) => (process.versions.bun
    ? [fixture]
    : ["--import", "tsx", fixture]);

  // Nix wrappers, `node dist/cli/qmd.js` and `bun src/cli/qmd.ts` reach the
  // real coordinator with no supervising parent, so the child must own both
  // the deadline and the second-signal escalation. The fixture wires the real
  // coordinator and signal handlers to a drain that never finishes.
  const FIXTURE = `
import { appendFileSync } from "node:fs";
import { createCliShutdownCoordinator, installMainSignalHandlers, setActiveShutdownForTest } from "SRC/cli/qmd.ts";

const breadcrumb = process.env.FIXTURE_BREADCRUMB;
const coordinator = createCliShutdownCoordinator({
  closeAdmission: () => appendFileSync(breadcrumb, "drain-started\\n"),
  waitForInflight: () => new Promise(() => {}),
  waitForLlmIdle: async () => {},
  disposeLlm: async () => { appendFileSync(breadcrumb, "disposed\\n"); },
  closeStore: () => { appendFileSync(breadcrumb, "store-closed\\n"); },
});
setActiveShutdownForTest(coordinator.shutdown);
installMainSignalHandlers(coordinator.shutdown);
appendFileSync(breadcrumb, "ready\\n");
setInterval(() => {}, 1 << 30);
`;

  async function startFixture(workDir: string, env: Record<string, string>) {
    // `.mts` so tsx treats a fixture outside the repo as ESM.
    const fixturePath = join(workDir, "unsupervised-fixture.mts");
    writeFileSync(fixturePath, FIXTURE.replace("SRC", join(repoRoot, "src")));
    const breadcrumb = join(workDir, "breadcrumb.txt");
    const child = spawn(process.execPath, runtimeArgs(fixturePath), {
      cwd: repoRoot,
      env: { ...process.env, FIXTURE_BREADCRUMB: breadcrumb, QMD_SUPERVISED: undefined, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    await waitFor(() => existsSync(breadcrumb) && readFileSync(breadcrumb, "utf8").includes("ready"), 20000);
    return { child, breadcrumb };
  }

  test.skipIf(process.platform === "win32")("first signal drains, second signal terminates", async () => {
    const workDir = mkdtempSync(join(tmpdir(), "qmd-unsupervised-"));
    const { child, breadcrumb } = await startFixture(workDir, { QMD_SHUTDOWN_GRACE_MS: "60000" });
    try {
      child.kill("SIGINT");
      await waitFor(() => readFileSync(breadcrumb, "utf8").includes("drain-started"), 5000);
      await sleep(200);
      // The drain is stuck but nothing was disposed and the process is alive.
      expect(child.exitCode).toBeNull();
      expect(readFileSync(breadcrumb, "utf8")).not.toContain("disposed");

      child.kill("SIGINT");
      await waitFor(() => child.exitCode !== null || child.signalCode !== null, 5000);
      expect(child.signalCode).toBe("SIGKILL");
      expect(readFileSync(breadcrumb, "utf8")).not.toContain("store-closed");
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      rmSync(workDir, { recursive: true, force: true });
    }
  }, 40000);

  test.skipIf(process.platform === "win32")("grace expiry terminates a child that never gets a second signal", async () => {
    const workDir = mkdtempSync(join(tmpdir(), "qmd-unsupervised-grace-"));
    const { child, breadcrumb } = await startFixture(workDir, { QMD_SHUTDOWN_GRACE_MS: "600" });
    try {
      child.kill("SIGTERM");
      await waitFor(() => readFileSync(breadcrumb, "utf8").includes("drain-started"), 5000);
      await waitFor(() => child.exitCode !== null || child.signalCode !== null, 6000);
      expect(child.signalCode).toBe("SIGKILL");
      expect(readFileSync(breadcrumb, "utf8")).not.toContain("disposed");
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      rmSync(workDir, { recursive: true, force: true });
    }
  }, 40000);
});
