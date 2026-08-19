import { describe, expect, test } from "vitest";
import {
  createShutdownCoordinator,
  createShutdownHandoff,
  hardStopProcess,
  type ShutdownHooks,
  type ShutdownTrigger,
} from "../src/shutdown.ts";

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createHooks(overrides: Partial<ShutdownHooks> = {}) {
  const calls: string[] = [];
  const exitCodes: number[] = [];
  const errors: unknown[] = [];
  const hardStops: unknown[] = [];
  let inflight = deferred();
  inflight.resolve();
  let llmIdle = deferred();
  llmIdle.resolve();
  let serving = deferred();
  serving.resolve();

  const hooks: ShutdownHooks = {
    closeAdmission() { calls.push("close-admission"); },
    stopServing() {
      calls.push("stop-serving");
      return serving.promise;
    },
    requestAbort() { calls.push("request-abort"); },
    waitForInflight() {
      calls.push("wait-inflight");
      return inflight.promise;
    },
    waitForLlmIdle() {
      calls.push("wait-llm-idle");
      return llmIdle.promise;
    },
    async disposeLlm() { calls.push("dispose-llm"); },
    closeStore() { calls.push("close-store"); },
    setExitCode(code) {
      calls.push(`exit:${code}`);
      exitCodes.push(code);
    },
    logError(error) {
      calls.push("log-error");
      errors.push(error);
    },
    hardStop(error): never {
      calls.push("hard-stop");
      hardStops.push(error);
      throw error instanceof Error ? error : new Error(String(error));
    },
    // Several tests below deliberately leave a shutdown hung forever. Without
    // an inert scheduler they would arm real 30s watchdogs that later fire
    // during unrelated tests. The watchdog's own behaviour is covered by
    // "shutdown grace watchdog", which injects its own schedule.
    schedule() {
      return { clear() {} };
    },
    ...overrides,
  };

  return { hooks, calls, exitCodes, errors, hardStops, inflight, llmIdle, serving };
}

describe("createShutdownCoordinator", () => {
  test("runs teardown in the exact ownership order", async () => {
    const { hooks, calls, exitCodes } = createHooks();
    const coordinator = createShutdownCoordinator(hooks);

    await coordinator.shutdown({ kind: "complete", exitCode: 0 });

    expect(calls).toEqual([
      "close-admission",
      "stop-serving",
      "request-abort",
      "wait-inflight",
      "wait-llm-idle",
      "dispose-llm",
      "close-store",
      "exit:0",
    ]);
    expect(exitCodes).toEqual([0]);
  });

  test("concurrent calls share one promise and invoke each hook once", async () => {
    const { hooks, calls } = createHooks();
    const coordinator = createShutdownCoordinator(hooks);

    const first = coordinator.shutdown({ kind: "complete", exitCode: 0 });
    const second = coordinator.shutdown({ kind: "signal", signal: "SIGTERM", exitCode: 143 });
    expect(first).toBe(second);
    await Promise.all([first, second]);

    expect(calls.filter((c) => c === "close-admission")).toHaveLength(1);
    expect(calls.filter((c) => c === "dispose-llm")).toHaveLength(1);
    expect(calls.filter((c) => c === "close-store")).toHaveLength(1);
  });

  test("EOF then SIGTERM upgrades the exit code to 143", async () => {
    const serving = deferred();
    const { hooks, exitCodes } = createHooks({
      stopServing() { return serving.promise; },
    });
    const coordinator = createShutdownCoordinator(hooks);

    const eof = coordinator.shutdown({ kind: "stdin-eof", exitCode: 0 });
    const term = coordinator.shutdown({ kind: "signal", signal: "SIGTERM", exitCode: 143 });
    expect(eof).toBe(term);
    serving.resolve();
    await eof;

    expect(exitCodes).toEqual([143]);
  });

  test("SIGINT then SIGTERM preserves first-signal code 130", async () => {
    const serving = deferred();
    const { hooks, exitCodes } = createHooks({
      stopServing() { return serving.promise; },
    });
    const coordinator = createShutdownCoordinator(hooks);

    const first = coordinator.shutdown({ kind: "signal", signal: "SIGINT", exitCode: 130 });
    const second = coordinator.shutdown({ kind: "signal", signal: "SIGTERM", exitCode: 143 });
    expect(first).toBe(second);
    serving.resolve();
    await first;

    expect(exitCodes).toEqual([130]);
  });

  test("does not dispose LLM or store while a request is still active", async () => {
    const inflight = deferred();
    const { hooks, calls } = createHooks({
      waitForInflight() { return inflight.promise; },
    });
    const coordinator = createShutdownCoordinator(hooks);

    const shutting = coordinator.shutdown({ kind: "complete", exitCode: 0 });
    await Promise.resolve();
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");

    inflight.resolve();
    await shutting;
    expect(calls).toContain("dispose-llm");
    expect(calls).toContain("close-store");
  });

  test("does not close the store while the LLM is still active", async () => {
    const llmIdle = deferred();
    const { hooks, calls } = createHooks({
      waitForLlmIdle() { return llmIdle.promise; },
    });
    const coordinator = createShutdownCoordinator(hooks);

    const shutting = coordinator.shutdown({ kind: "complete", exitCode: 0 });
    await Promise.resolve();
    expect(calls).not.toContain("close-store");

    llmIdle.resolve();
    await shutting;
    expect(calls.at(-2)).toBe("close-store");
  });

  test("a request-wait failure does not dispose LLM or close the store", async () => {
    const { hooks, calls, hardStops } = createHooks({
      waitForInflight() { return Promise.reject(new Error("request still running")); },
    });
    const coordinator = createShutdownCoordinator(hooks);

    await expect(coordinator.shutdown({ kind: "complete", exitCode: 0 })).rejects.toThrow("request still running");
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");
    expect(hardStops).toHaveLength(1);
  });

  test("an LLM disposal failure does not close the store and invokes hard-stop", async () => {
    const { hooks, calls, hardStops } = createHooks({
      async disposeLlm() { throw new Error("context busy"); },
    });
    const coordinator = createShutdownCoordinator(hooks);

    await expect(coordinator.shutdown({ kind: "complete", exitCode: 0 })).rejects.toThrow("context busy");
    expect(calls).not.toContain("close-store");
    expect(hardStops).toHaveLength(1);
  });

  test("a hung lower-tier wait never starts parent disposal", async () => {
    const inflight = deferred();
    const { hooks, calls } = createHooks();
    hooks.waitForInflight = () => {
      calls.push("wait-inflight");
      return inflight.promise;
    };
    const coordinator = createShutdownCoordinator(hooks);

    const shutting = coordinator.shutdown({ kind: "signal", signal: "SIGINT", exitCode: 130 });
    await Promise.resolve();
    expect(calls).toEqual([
      "close-admission",
      "stop-serving",
      "request-abort",
      "wait-inflight",
    ]);
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");

    inflight.reject(new Error("deadline belongs to supervisor"));
    await expect(shutting).rejects.toThrow("deadline belongs to supervisor");
    expect(calls).not.toContain("dispose-llm");
  });

  test("aborts the coordinator signal during shutdown", async () => {
    const { hooks } = createHooks();
    const coordinator = createShutdownCoordinator(hooks);
    expect(coordinator.signal.aborted).toBe(false);
    await coordinator.shutdown({ kind: "signal", signal: "SIGINT", exitCode: 130 });
    expect(coordinator.signal.aborted).toBe(true);
  });

  test("a synchronous closeAdmission throw is logged and hard-stopped without lower tiers", async () => {
    const released: string[] = [];
    const { hooks, calls, hardStops } = createHooks({
      closeAdmission() { throw new Error("admission boom"); },
      holdOpen() {
        return () => { released.push("hold"); };
      },
    });
    const coordinator = createShutdownCoordinator(hooks);
    await expect(coordinator.shutdown({ kind: "complete", exitCode: 0 })).rejects.toThrow("admission boom");
    expect(calls).toContain("log-error");
    expect(hardStops).toHaveLength(1);
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");
    expect(released).toEqual([]);
  });

  test("a synchronous stopServing throw is settled and still fails the coordinator", async () => {
    const { hooks, calls, hardStops } = createHooks({
      stopServing() { throw new Error("listener boom"); },
    });
    const coordinator = createShutdownCoordinator(hooks);
    await expect(coordinator.shutdown({ kind: "complete", exitCode: 0 })).rejects.toThrow("listener boom");
    expect(calls).toContain("log-error");
    expect(hardStops).toHaveLength(1);
    expect(calls).not.toContain("dispose-llm");
  });

  test("a synchronous requestAbort throw is logged and hard-stopped without disposal", async () => {
    const { hooks, calls, hardStops } = createHooks({
      requestAbort() { throw new Error("abort boom"); },
    });
    const coordinator = createShutdownCoordinator(hooks);
    await expect(coordinator.shutdown({ kind: "signal", signal: "SIGTERM", exitCode: 143 })).rejects.toThrow("abort boom");
    expect(calls).toContain("log-error");
    expect(hardStops).toHaveLength(1);
    expect(calls).not.toContain("dispose-llm");
    expect(calls).not.toContain("close-store");
  });
});

describe("ShutdownTrigger typing", () => {
  test("accepts the three documented trigger kinds", () => {
    const triggers: ShutdownTrigger[] = [
      { kind: "complete", exitCode: 0 },
      { kind: "stdin-eof", exitCode: 0 },
      { kind: "signal", signal: "SIGINT", exitCode: 130 },
    ];
    expect(triggers).toHaveLength(3);
  });
});

describe("createShutdownHandoff", () => {
  test("replays a pending SIGINT into the attached owner", async () => {
    const handoff = createShutdownHandoff();
    const triggers: ShutdownTrigger[] = [];
    const started = deferred();
    const waiting = handoff.shutdown({ kind: "signal", signal: "SIGINT", exitCode: 130 });
    await Promise.resolve();
    await handoff.attach(async (trigger) => {
      triggers.push(trigger);
      started.resolve();
    });
    await waiting;
    await started.promise;
    expect(triggers).toEqual([{ kind: "signal", signal: "SIGINT", exitCode: 130 }]);
  });

  test("replays a pending SIGTERM and upgrades EOF", async () => {
    const handoff = createShutdownHandoff();
    const first = handoff.shutdown({ kind: "stdin-eof", exitCode: 0 });
    const second = handoff.shutdown({ kind: "signal", signal: "SIGTERM", exitCode: 143 });
    const seen: ShutdownTrigger[] = [];
    await handoff.attach(async (trigger) => { seen.push(trigger); });
    await Promise.all([first, second]);
    expect(seen).toEqual([{ kind: "signal", signal: "SIGTERM", exitCode: 143 }]);
  });

  test("fail rejects waiters so startup errors cannot hang a signal", async () => {
    const handoff = createShutdownHandoff();
    const waiting = handoff.shutdown({ kind: "signal", signal: "SIGINT", exitCode: 130 });
    handoff.fail(new Error("listen failed"));
    await expect(waiting).rejects.toThrow("listen failed");
  });
});

describe("shutdown grace watchdog", () => {
  const hooks = (overrides: Partial<Parameters<typeof createShutdownCoordinator>[0]> = {}) => ({
    closeAdmission() {},
    stopServing() {},
    requestAbort() {},
    waitForInflight: async () => {},
    waitForLlmIdle: async () => {},
    disposeLlm: async () => {},
    closeStore() {},
    setExitCode() {},
    logError() {},
    ...overrides,
  });

  test("arms one watchdog for every trigger kind", async () => {
    for (const trigger of [
      { kind: "complete", exitCode: 0 },
      { kind: "stdin-eof", exitCode: 0 },
      { kind: "signal", signal: "SIGINT", exitCode: 130 },
    ] as const) {
      const armed: number[] = [];
      const coordinator = createShutdownCoordinator(hooks({
        graceMs: 250,
        hardStop: (() => { throw new Error("unused"); }) as never,
        schedule(_fn, ms) {
          armed.push(ms);
          return { clear() {} };
        },
      }));
      await coordinator.shutdown(trigger);
      expect(armed).toEqual([250]);
    }
  });

  test("successful teardown disarms the watchdog and never hard-stops", async () => {
    const events: string[] = [];
    const coordinator = createShutdownCoordinator(hooks({
      hardStop: (() => { events.push("hard-stop"); throw new Error("x"); }) as never,
      schedule(_fn) {
        events.push("armed");
        return { clear() { events.push("cleared"); } };
      },
    }));
    await coordinator.shutdown({ kind: "complete", exitCode: 0 });
    expect(events).toEqual(["armed", "cleared"]);
  });

  // The hardStop hook is contractually never-returning, so the coordinator must
  // not catch it. The harness models that faithfully: the injected hard stop
  // throws out of the timer callback, and the drain it interrupted stays hung
  // (a resolvable drain would mean there was no deadline to miss).
  test("grace expiry throws out of the timer and no lower tier can follow", async () => {
    const events: string[] = [];
    let fire!: () => void;
    const coordinator = createShutdownCoordinator(hooks({
      waitForInflight: () => new Promise<void>(() => {}),
      waitForLlmIdle: async () => { events.push("wait-llm-idle"); },
      disposeLlm: async () => { events.push("dispose-llm"); },
      closeStore() { events.push("close-store"); },
      setExitCode() { events.push("set-exit-code"); },
      graceMs: 5,
      hardStop: ((error: unknown) => {
        events.push("hard-stop");
        throw error instanceof Error ? error : new Error(String(error));
      }) as never,
      schedule(fn) {
        fire = fn;
        return { clear() { events.push("cleared"); } };
      },
    }));

    void coordinator.shutdown({ kind: "stdin-eof", exitCode: 0 });
    await Promise.resolve();

    // Terminal: the throw reaches the scheduler, it is not swallowed into a
    // return that would hand control back to the event loop.
    expect(() => fire()).toThrow(/Shutdown exceeded 5ms grace/);

    // Nothing resumes teardown afterwards, and the watchdog is never disarmed.
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(events).toEqual(["hard-stop"]);
  });

  test("hardStopProcess reports the reason before terminating", () => {
    const written: string[] = [];
    const originalWrite = process.stderr.write;
    const originalKill = process.kill;
    const killed: Array<[number, string | number | undefined]> = [];
    // @ts-expect-error test double
    process.stderr.write = (chunk: string) => { written.push(String(chunk)); return true; };
    // @ts-expect-error test double
    process.kill = (pid: number, signal?: string | number) => { killed.push([pid, signal]); return true; };
    try {
      expect(() => hardStopProcess(new Error("drain stuck"))).toThrow("drain stuck");
    } finally {
      process.stderr.write = originalWrite;
      process.kill = originalKill;
    }
    expect(written.join("")).toContain("drain stuck");
    expect(killed).toEqual([[process.pid, "SIGKILL"]]);
  });
});
