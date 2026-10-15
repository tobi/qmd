import { describe, expect, test } from "vitest";
import * as scheduler from "../src/embedding-scheduler.js";
import { createMaintenanceJobs } from "../src/maintenance-jobs.js";

describe("opt-in embedding scheduler", () => {
  test("is disabled by default and starts no work or timers", () => {
    expect(scheduler).toHaveProperty("createEmbeddingScheduler");
    let timers = 0;
    let calls = 0;
    const owner = scheduler.createEmbeddingScheduler({
      maintenance: createMaintenanceJobs(),
      collections: () => ["docs"], pending: () => 1, activeQueries: () => 0,
      embed: async () => { calls++; return { chunksEmbedded: 1, errors: 0 }; },
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout: () => { timers++; return 1; }, clearTimeout: () => {},
        setInterval: () => { timers++; return 1; }, clearInterval: () => {} },
    });
    owner.start();
    expect(owner.tick()).toBeNull();
    expect(owner.health().enabled).toBe(false);
    expect(timers).toBe(0);
    expect(calls).toBe(0);
  });
  test("an enabled sweep holds the shared claim and forwards bounded resumable native options", async () => {
    const maintenance = createMaintenanceJobs();
    let release!: () => void;
    let enter!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const entered = new Promise<void>(resolve => { enter = resolve; });
    let debt = 1;
    const calls: Record<string, unknown>[] = [];
    const owner = scheduler.createEmbeddingScheduler({
      intervalMinutes: 1, maxDurationMs: 1000, maxDocsPerBatch: 2, maxBatchBytes: 4096,
      maintenance, collections: () => ["docs"], pending: () => debt, activeQueries: () => 0,
      async embed(options) { calls.push(options); enter(); await held; debt = 0; return { chunksEmbedded: 1, errors: 0 }; },
    });
    const job = owner.tick();
    expect(job).not.toBeNull();
    try {
      await entered;
      expect(maintenance.activeJobId).toBe(job?.id);
      expect(owner.tick()).toBeNull();
      expect(() => maintenance.startJob("update", {}, async () => ({}))).toThrow("Job already active");
      expect(calls).toHaveLength(1);
      expect(calls[0]).toMatchObject({ collection: "docs", force: false, chunkStrategy: "auto", resume: true, maxDocsPerBatch: 2, maxBatchBytes: 4096 });
      expect(calls[0]?.signal).toBeInstanceOf(AbortSignal);
      expect(calls[0]?.maxDurationMs).toBeGreaterThan(0);
      expect(calls[0]?.maxDurationMs).toBeLessThanOrEqual(1000);
    } finally { release(); await maintenance.waitForIdle(); await owner.stop(); }
    expect(maintenance.getJob(job!.id)).toMatchObject({ state: "succeeded", result: { chunksEmbedded: 1, needsEmbedding: 0 } });
  });

  test.each(["deadline", "querying", "policy", "debt"])("reloads %s before admitting the next collection", async reason => {
    const maintenance = createMaintenanceJobs();
    let time = 0;
    let querying = 0;
    let names = ["a", "b"];
    let secondDebt = 1;
    const calls: string[] = [];
    const owner = scheduler.createEmbeddingScheduler({
      intervalMinutes: 1, maxDurationMs: 100,
      maintenance, collections: () => names,
      pending: selected => selected?.[0] === "b" ? secondDebt : 1,
      activeQueries: () => querying,
      async embed(options) {
        calls.push(options.collection!);
        if (reason === "deadline") time = 101;
        if (reason === "querying") querying = 1;
        if (reason === "policy") names = ["a"];
        if (reason === "debt") secondDebt = 0;
        return { chunksEmbedded: 1, errors: 0 };
      },
      clock: { now: () => 0, monotonic: () => time,
        setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {} },
    });
    const job = owner.tick();
    await maintenance.waitForIdle();
    expect(calls).toEqual(["a"]);
    expect(maintenance.getJob(job!.id)).toMatchObject({ state: "partial", result: { chunksEmbedded: 1, needsEmbedding: 1 } });
    if (reason === "deadline" || reason === "querying") {
      expect(maintenance.getJob(job!.id)?.result).toMatchObject({ stopReason: reason });
    }
    await owner.stop();
  });

  test("does not admit scheduled embedding while foreground queries are active", () => {
    const maintenance = createMaintenanceJobs();
    const owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 1, maintenance,
      collections: () => ["docs"], pending: () => 1, activeQueries: () => 1,
      embed: async () => ({ chunksEmbedded: 1, errors: 0 }),
    });
    expect(owner.tick()).toBeNull();
    expect(maintenance.activeJobId).toBeNull();
  });

  test.each(["deadline", "shutdown"])("cooperative %s aborts but awaits the actual native batch", async reason => {
    const maintenance = createMaintenanceJobs();
    let release!: () => void;
    let enter!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const entered = new Promise<void>(resolve => { enter = resolve; });
    let deadline!: () => void;
    let signal!: AbortSignal;
    const owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 1, maxDurationMs: 100,
      maintenance, collections: () => ["docs"], pending: () => 1, activeQueries: () => 0,
      async embed(options) { signal = options.signal!; enter(); await held; return { chunksEmbedded: 1, errors: 0 }; },
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout: callback => { deadline = callback; return 1; }, clearTimeout: () => {},
        setInterval: () => 1, clearInterval: () => {} },
    });
    const job = owner.tick();
    await entered;
    let settled = false;
    let stopping: Promise<void> | undefined;
    try {
      if (reason === "deadline") deadline?.();
      else stopping = owner.stop().then(() => { settled = true; });
      expect(signal.aborted).toBe(true);
      expect(maintenance.activeJobId).toBe(job!.id);
      await Promise.resolve();
      expect(settled).toBe(false);
    } finally { release(); await stopping; await maintenance.waitForIdle(); await owner.stop(); }
    expect(maintenance.getJob(job!.id)).toMatchObject({ state: "partial", result: { chunksEmbedded: 1, stopReason: reason } });
    expect(maintenance.activeJobId).toBeNull();
    if (reason === "shutdown") expect(settled).toBe(true);
  });

  test("keeps committed progress when a later collection fails", async () => {
    const maintenance = createMaintenanceJobs();
    const owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 1, maintenance,
      collections: () => ["a", "b"], pending: () => 1, activeQueries: () => 0,
      sanitizeError: () => "sanitized",
      async embed(options) {
        if (options.collection === "a") return { chunksEmbedded: 3, errors: 0 };
        options.onProgress?.({ chunksEmbedded: 2 } as Parameters<NonNullable<typeof options.onProgress>>[0]);
        throw new Error("private detail");
      },
    });
    const job = owner.tick();
    await maintenance.waitForIdle();
    expect(maintenance.getJob(job!.id)).toMatchObject({ state: "partial", result: {
      chunksEmbedded: 5, embeddings: [{ collection: "a", chunksEmbedded: 3 }],
      collectionErrors: [{ collection: "b", error: "sanitized" }], needsEmbedding: 1,
    } });
    await owner.stop();
  });

  test("rotates the first eligible collection between bounded sweeps", async () => {
    const maintenance = createMaintenanceJobs();
    const calls: string[] = [];
    const owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 1, maintenance,
      collections: () => ["a", "b"], pending: () => 1, activeQueries: () => 0,
      async embed(options) { calls.push(options.collection!); return { chunksEmbedded: 1, errors: 0 }; },
    });
    owner.tick(); await maintenance.waitForIdle();
    owner.tick(); await maintenance.waitForIdle();
    expect(calls).toEqual(["a", "b", "b", "a"]);
    await owner.stop();
  });

  test("starts timers only explicitly and cannot create an interval after stop inside initial tick", async () => {
    const maintenance = createMaintenanceJobs();
    const timeouts: { callback: () => void; delay: number }[] = [];
    const intervals: { callback: () => void; delay: number }[] = [];
    const cleared: unknown[] = [];
    let owner!: ReturnType<typeof scheduler.createEmbeddingScheduler>;
    owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 2, initialDelaySeconds: 3, maintenance,
      collections: () => { void owner.stop(); return []; }, pending: () => 0, activeQueries: () => 0,
      embed: async () => ({ chunksEmbedded: 0, errors: 0 }),
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout: (callback, delay) => { const timer = { callback, delay }; timeouts.push(timer); return timer; },
        clearTimeout: timer => { cleared.push(timer); },
        setInterval: (callback, delay) => { const timer = { callback, delay }; intervals.push(timer); return timer; },
        clearInterval: timer => { cleared.push(timer); },
      },
    });
    expect(timeouts).toHaveLength(0);
    owner.start(); owner.start();
    expect(timeouts).toHaveLength(1);
    expect(timeouts[0]?.delay).toBe(3000);
    timeouts[0]!.callback();
    expect(intervals).toHaveLength(0);
    expect(cleared).toContain(timeouts[0]);
    await owner.stop(); owner.start();
    expect(timeouts).toHaveLength(1);
    expect(owner.health().next).toBeNull();
  });

  test.each([
    ["maxDurationMs", 0], ["maxDurationMs", -1], ["maxDurationMs", Infinity], ["maxDurationMs", NaN],
    ["maxDocsPerBatch", 0], ["maxDocsPerBatch", 1.5], ["maxBatchBytes", 0], ["maxBatchBytes", Infinity],
    ["intervalMinutes", -1], ["intervalMinutes", Infinity], ["initialDelaySeconds", -1], ["initialDelaySeconds", Infinity],
  ])("rejects invalid %s value %s before creating a timer or job", (key, value) => {
    const maintenance = createMaintenanceJobs();
    expect(() => scheduler.createEmbeddingScheduler({
      maintenance, collections: () => [], pending: () => 0, activeQueries: () => 0,
      embed: async () => ({ chunksEmbedded: 0, errors: 0 }), [key]: value,
    })).toThrow(RangeError);
    expect(maintenance.activeJobId).toBeNull();
  });

  test("a timer admission failure is recorded without escaping the callback or claiming work", async () => {
    const maintenance = createMaintenanceJobs({ admit() { throw new Error("private admission detail"); } });
    let callback!: () => void;
    const owner = scheduler.createEmbeddingScheduler({ intervalMinutes: 1, maintenance,
      collections: () => ["docs"], pending: () => 1, activeQueries: () => 0,
      sanitizeError: () => "sanitized",
      embed: async () => ({ chunksEmbedded: 0, errors: 0 }),
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout: action => { callback = action; return 1; }, clearTimeout: () => {},
        setInterval: () => 2, clearInterval: () => {} },
    });
    owner.start();
    expect(() => callback()).not.toThrow();
    expect(maintenance.activeJobId).toBeNull();
    expect(owner.health().last).toMatchObject({ state: "failed_to_start", error: "sanitized" });
    await owner.stop();
  });

  test.each(["start-clock", "start-timer", "initial-clock", "initial-interval", "interval-clock"])(
    "contains scheduling failure in %s and keeps timer ownership consistent", async phase => {
      const maintenance = createMaintenanceJobs();
      const timeouts: (() => void)[] = [];
      const intervals: (() => void)[] = [];
      let nowCalls = 0;
      let failing = true;
      const owner = scheduler.createEmbeddingScheduler({
        intervalMinutes: 1, maintenance,
        collections: () => ["docs"], pending: () => 0, activeQueries: () => 0,
        sanitizeError: () => "sanitized scheduling failure",
        embed: async () => ({ chunksEmbedded: 0, errors: 0 }),
        clock: {
          now() {
            nowCalls++;
            if (failing && ((phase === "start-clock" && nowCalls === 1) ||
              (phase === "initial-clock" && nowCalls === 2) ||
              (phase === "interval-clock" && nowCalls === 3))) throw new Error("private clock detail");
            return 0;
          },
          monotonic: () => 0,
          setTimeout(callback) {
            if (failing && phase === "start-timer") throw new Error("private timeout detail");
            timeouts.push(callback); return callback;
          },
          clearTimeout: () => {},
          setInterval(callback) {
            if (failing && phase === "initial-interval") throw new Error("private interval detail");
            intervals.push(callback); return callback;
          },
          clearInterval: () => {},
        },
      });
      expect(() => owner.start()).not.toThrow();
      if (!phase.startsWith("start-")) expect(() => timeouts[0]!()).not.toThrow();
      if (phase === "interval-clock") expect(() => intervals[0]!()).not.toThrow();
      expect(owner.health()).toMatchObject({
        next: null, last: { state: "failed_to_start", error: "sanitized scheduling failure" },
      });
      expect(maintenance.activeJobId).toBeNull();
      failing = false;
      if (phase === "interval-clock") {
        expect(() => intervals[0]!()).not.toThrow();
        expect(intervals).toHaveLength(1);
      } else {
        const count = timeouts.length;
        owner.start();
        expect(timeouts).toHaveLength(count + 1);
      }
      expect(owner.health().next).not.toBeNull();
      await owner.stop();
      expect(owner.health().next).toBeNull();
    },
  );

  test("a throwing deadline cleanup retains the committed sweep and releases its controller", async () => {
    const maintenance = createMaintenanceJobs();
    let debt = 1;
    let signal!: AbortSignal;
    const owner = scheduler.createEmbeddingScheduler({
      intervalMinutes: 1, maintenance,
      collections: () => ["docs"], pending: () => debt, activeQueries: () => 0,
      sanitizeError: () => "sanitized cleanup failure",
      async embed(options) { signal = options.signal!; debt = 0; return { chunksEmbedded: 3, errors: 0 }; },
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout: () => 1, clearTimeout: () => { throw new Error("private clear detail"); },
        setInterval: () => 2, clearInterval: () => {} },
    });
    const job = owner.tick();
    await maintenance.waitForIdle();
    await owner.stop();
    expect(signal.aborted).toBe(false);
    expect(maintenance.activeJobId).toBeNull();
    expect(maintenance.getJob(job!.id)).toMatchObject({
      state: "partial",
      result: { chunksEmbedded: 3, needsEmbedding: 0,
        embeddings: [{ collection: "docs", chunksEmbedded: 3, errors: 0 }],
        collectionErrors: [{ collection: null, error: "sanitized cleanup failure" }] },
    });
  });

  test.each(["timeout", "interval"])("a throwing stop %s cleanup still aborts and drains native work", async phase => {
    const maintenance = createMaintenanceJobs();
    let release!: () => void;
    let enter!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const entered = new Promise<void>(resolve => { enter = resolve; });
    let initial!: () => void;
    let interval!: () => void;
    let timerId = 0;
    let signal!: AbortSignal;
    let failCleanup = false;
    const owner = scheduler.createEmbeddingScheduler({
      intervalMinutes: 1, maintenance,
      collections: () => ["docs"], pending: () => 1, activeQueries: () => 0,
      sanitizeError: () => "sanitized stop cleanup failure",
      async embed(options) { signal = options.signal!; enter(); await held; return { chunksEmbedded: 3, errors: 0 }; },
      clock: { now: () => 0, monotonic: () => 0,
        setTimeout(callback) { if (++timerId === 1) initial = callback; return timerId; },
        clearTimeout(timer) { if (failCleanup && phase === "timeout" && timer === 1) throw new Error("private timeout clear detail"); },
        setInterval(callback) { interval = callback; return "interval"; },
        clearInterval() { if (failCleanup && phase === "interval") throw new Error("private interval clear detail"); } },
    });
    owner.start();
    if (phase === "interval") initial();
    else owner.tick();
    await entered;
    const jobId = maintenance.activeJobId!;
    failCleanup = true;
    let settled = false;
    const stopping = owner.stop().then(() => { settled = true; return null; },
      error => { settled = true; return error; });
    await Promise.resolve();
    (phase === "timeout" ? initial : interval)();
    const nextAfterFailedClear = owner.health().next;
    const abortedBeforeRelease = signal.aborted;
    const settledBeforeRelease = settled;
    release();
    const error = await stopping;
    await maintenance.waitForIdle();
    expect(nextAfterFailedClear).toBeNull();
    expect(abortedBeforeRelease).toBe(true);
    expect(settledBeforeRelease).toBe(false);
    expect(error).toBeNull();
    expect(owner.health()).toMatchObject({
      next: null, last: { state: "cleanup_failed", error: "sanitized stop cleanup failure" },
    });
    expect(maintenance.activeJobId).toBeNull();
    expect(maintenance.getJob(jobId)).toMatchObject({
      state: "partial", result: { chunksEmbedded: 3, stopReason: "shutdown" },
    });
  });

  test.each(["timer", "clock"])("a throwing deadline setup %s returns a sanitized failed sweep receipt", async phase => {
    const maintenance = createMaintenanceJobs();
    let embedded = false;
    let clears = 0;
    const owner = scheduler.createEmbeddingScheduler({
      intervalMinutes: 1, maintenance,
      collections: () => ["docs"], pending: () => 1, activeQueries: () => 0,
      sanitizeError: () => "sanitized deadline setup failure",
      async embed() { embedded = true; return { chunksEmbedded: 1, errors: 0 }; },
      clock: {
        now: () => 0,
        monotonic() { if (phase === "clock") throw new Error("private deadline clock detail"); return 0; },
        setTimeout() { if (phase === "timer") throw new Error("private deadline timer detail"); return 1; },
        clearTimeout() { clears++; },
        setInterval: () => 2, clearInterval: () => {},
      },
    });
    const job = owner.tick();
    await maintenance.waitForIdle();
    await owner.stop();
    expect(embedded).toBe(false);
    expect(clears).toBe(phase === "clock" ? 1 : 0);
    expect(maintenance.activeJobId).toBeNull();
    expect(maintenance.getJob(job!.id)).toMatchObject({
      state: "failed", error: null,
      result: { chunksEmbedded: 0, needsEmbedding: 1, embeddings: [],
        collectionErrors: [{ collection: null, error: "sanitized deadline setup failure" }] },
    });
  });

  test.each(["start-clock", "initial-timer", "initial-clear", "initial-clock", "interval-timer", "interval-clock"])(
    "shutdown inside %s cannot publish a later timer or next timestamp", async phase => {
      const maintenance = createMaintenanceJobs();
      const liveTimers = new Set<string>();
      let initial: (() => void) | undefined;
      let interval: (() => void) | undefined;
      let reads = 0;
      let requested = false;
      let stopping: Promise<void> | undefined;
      let embedded = 0;
      let owner!: ReturnType<typeof scheduler.createEmbeddingScheduler>;
      const shutdown = () => {
        if (requested) return;
        requested = true;
        stopping = owner.stop();
      };
      owner = scheduler.createEmbeddingScheduler({
        intervalMinutes: 1, maintenance,
        collections: () => ["docs"], pending: () => 0, activeQueries: () => 0,
        embed: async () => { embedded++; return { chunksEmbedded: 0, errors: 0 }; },
        clock: {
          now() {
            reads++;
            if ((phase === "start-clock" && reads === 1) ||
              (phase === "initial-clock" && reads === 2) ||
              (phase === "interval-clock" && reads === 3)) shutdown();
            return 0;
          },
          monotonic: () => 0,
          setTimeout(callback) {
            initial = callback;
            liveTimers.add("initial");
            if (phase === "initial-timer") shutdown();
            return "initial";
          },
          clearTimeout(timer) {
            liveTimers.delete(timer as string);
            if (phase === "initial-clear") shutdown();
          },
          setInterval(callback) {
            interval = callback;
            liveTimers.add("interval");
            if (phase === "interval-timer") shutdown();
            return "interval";
          },
          clearInterval(timer) { liveTimers.delete(timer as string); },
        },
      });
      owner.start();
      if (!requested) initial!();
      if (!requested) interval!();
      await stopping;
      expect(requested).toBe(true);
      expect([...liveTimers]).toEqual([]);
      expect(owner.health().next).toBeNull();
      initial?.();
      interval?.();
      owner.start();
      expect([...liveTimers]).toEqual([]);
      expect(owner.health().next).toBeNull();
      expect(maintenance.activeJobId).toBeNull();
      expect(embedded).toBe(0);
    },
  );

});
