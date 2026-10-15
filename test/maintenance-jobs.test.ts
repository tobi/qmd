import { describe, expect, test } from "vitest";
import * as maintenance from "../src/maintenance-jobs.js";

describe("serialized maintenance jobs", () => {
  test("holds one shared claim until actual work finishes", async () => {
    expect(maintenance).toHaveProperty("createMaintenanceJobs");
    const jobs = maintenance.createMaintenanceJobs();
    let finish!: () => void;
    const held = new Promise<void>(resolve => { finish = resolve; });
    const started = jobs.startJob("embed", {}, async () => {
      await held;
      return { chunksEmbedded: 1 };
    });
    expect(jobs.activeJobId).toBe(started.id);
    expect(() => jobs.startJob("update", {}, async () => ({}))).toThrow("Job already active");
    finish();
    await jobs.waitForIdle();
    expect(jobs.activeJobId).toBeNull();
    expect(jobs.getJob(started.id)?.state).toBe("succeeded");
    const next = jobs.startJob("update", {}, async () => ({}));
    await jobs.waitForIdle();
    expect(jobs.getJob(next.id)?.state).toBe("succeeded");
  });

  test("shutdown stops admission and awaits in-flight work after cooperative abort", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    expect(jobs).toHaveProperty("stop");
    let finish!: () => void;
    let enter!: () => void;
    let observedSignal: AbortSignal | undefined;
    const held = new Promise<void>(resolve => { finish = resolve; });
    const entered = new Promise<void>(resolve => { enter = resolve; });
    const job = jobs.startJob("embed", {}, async (_progress, signal) => {
      observedSignal = signal;
      enter();
      await held;
      return { chunksEmbedded: 1 };
    });
    await entered;
    let stopped = false;
    const stopping = jobs.stop().then(() => { stopped = true; });
    try {
      expect(observedSignal?.aborted).toBe(true);
      expect(jobs.activeJobId).toBe(job.id);
      expect(() => jobs.startJob("update", {}, async () => ({}))).toThrow("stopping");
      await Promise.resolve();
      expect(stopped).toBe(false);
    } finally {
      finish();
      await stopping;
    }
    expect(stopped).toBe(true);
    expect(jobs.activeJobId).toBeNull();
  });

  test("retains only the configured number of jobs and preserves an active claim", async () => {
    const jobs = maintenance.createMaintenanceJobs({ maxRetainedJobs: 2 });
    const first = jobs.startJob("update", {}, async () => ({}));
    await jobs.waitForIdle();
    const second = jobs.startJob("update", {}, async () => ({}));
    await jobs.waitForIdle();
    let finish!: () => void;
    const held = new Promise<void>(resolve => { finish = resolve; });
    const third = jobs.startJob("embed", {}, async () => { await held; });
    try {
      expect(jobs.getJob(first.id)).toBeUndefined();
      expect(jobs.getJob(second.id)?.state).toBe("succeeded");
      expect(jobs.getJob(third.id)).toBeDefined();
      expect(jobs.activeJobId).toBe(third.id);
    } finally {
      finish();
      await jobs.waitForIdle();
    }
    expect(jobs.getJob(third.id)?.state).toBe("succeeded");
  });

  test.each([0, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid retention budget %s before admitting any job",
    maxRetainedJobs => {
      expect(() => maintenance.createMaintenanceJobs({ maxRetainedJobs })).toThrow(RangeError);
    },
  );

  test("retains partial committed results and releases the claim after reported errors", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    const result = { chunksEmbedded: 2, errors: 1 };
    const job = jobs.startJob("embed", {}, async () => result);
    await jobs.waitForIdle();
    expect(jobs.getJob(job.id)?.state).toBe("partial");
    expect(jobs.getJob(job.id)?.result).toEqual(result);
    expect(jobs.activeJobId).toBeNull();
  });

  test("reports partial nested embedding results without discarding committed collections", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    const result = { embeddings: [{ collection: "a", chunksEmbedded: 2, errors: 0 }, { collection: "b", errors: 1 }] };
    const job = jobs.startJob("update", {}, async () => result);
    await jobs.waitForIdle();
    expect(jobs.getJob(job.id)?.state).toBe("partial");
    expect(jobs.getJob(job.id)?.result).toEqual(result);
    expect(jobs.activeJobId).toBeNull();
  });

  test("supports a caller classifier for unembedded debt", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    const result = { remainingChunks: 3 };
    const job = jobs.startJob("embed", {}, async () => result,
      value => (value as typeof result).remainingChunks > 0 ? "partial" : "succeeded");
    await jobs.waitForIdle();
    expect(jobs.getJob(job.id)?.state).toBe("partial");
    expect(jobs.getJob(job.id)?.result).toEqual(result);
  });

  test("rejects an invalid classifier state while preserving results and releasing the claim", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    const result = { chunksEmbedded: 2 };
    const job = jobs.startJob("embed", {}, async () => result,
      () => "queued" as "succeeded");
    await jobs.waitForIdle();
    expect(jobs.getJob(job.id)?.state).toBe("failed");
    expect(jobs.getJob(job.id)?.result).toEqual(result);
    expect(jobs.getJob(job.id)?.error).toContain("Invalid maintenance terminal state");
    expect(jobs.activeJobId).toBeNull();
    const next = jobs.startJob("update", {}, async () => ({}));
    await jobs.waitForIdle();
    expect(jobs.getJob(next.id)?.state).toBe("succeeded");
  });

  test("enforces synchronous host admission before creating a job claim", async () => {
    let allowed = false;
    const admitted: string[] = [];
    const jobs = maintenance.createMaintenanceJobs({
      admit(type: string) {
        admitted.push(type);
        if (!allowed) throw new Error("storage deferred");
      },
    });
    let executed = false;
    expect(() => jobs.startJob("embed", {}, async () => { executed = true; })).toThrow("storage deferred");
    expect(jobs.activeJobId).toBeNull();
    await jobs.waitForIdle();
    expect(executed).toBe(false);
    allowed = true;
    const job = jobs.startJob("update", {}, async () => ({}));
    await jobs.waitForIdle();
    expect(admitted).toEqual(["embed", "update"]);
    expect(jobs.getJob(job.id)?.state).toBe("succeeded");
  });

  test("holds the admission barrier against a reentrant job request", async () => {
    let once = false;
    let innerError: unknown;
    let innerExecutions = 0;
    const jobs = maintenance.createMaintenanceJobs({
      admit() {
        if (once) return;
        once = true;
        try {
          jobs.startJob("inner", {}, async () => { innerExecutions++; });
        } catch (error) { innerError = error; }
      },
    });
    const job = jobs.startJob("outer", {}, async () => ({}));
    await jobs.waitForIdle();
    expect(innerError).toBeInstanceOf(Error);
    expect((innerError as Error).message).toContain("Job already active");
    expect(innerExecutions).toBe(0);
    expect(jobs.getJob(job.id)?.state).toBe("succeeded");
  });

  test("rejects a job when shutdown occurs inside its admission hook", async () => {
    let stopping: Promise<void> | undefined;
    let executed = false;
    const jobs = maintenance.createMaintenanceJobs({
      admit() { stopping = jobs.stop(); },
    });
    expect(() => jobs.startJob("embed", {}, async () => { executed = true; })).toThrow("stopping");
    await stopping;
    await jobs.waitForIdle();
    expect(executed).toBe(false);
    expect(jobs.activeJobId).toBeNull();
  });

  test("uses host error sanitization for failed work receipts", async () => {
    const original = new Error("private endpoint detail");
    const seen: unknown[] = [];
    const jobs = maintenance.createMaintenanceJobs({
      sanitizeError(error: unknown) { seen.push(error); return "redacted failure"; },
    });
    const job = jobs.startJob("embed", {}, async () => { throw original; });
    await jobs.waitForIdle();
    expect(seen).toEqual([original]);
    expect(jobs.getJob(job.id)?.state).toBe("failed");
    expect(jobs.getJob(job.id)?.error).toBe("redacted failure");
    expect(jobs.activeJobId).toBeNull();
  });

  test("bounds default error receipts to one thousand characters", async () => {
    const jobs = maintenance.createMaintenanceJobs();
    const message = "x".repeat(2000);
    const job = jobs.startJob("embed", {}, async () => { throw new Error(message); });
    await jobs.waitForIdle();
    expect(jobs.getJob(job.id)?.state).toBe("failed");
    expect(jobs.getJob(job.id)?.error).toBe(message.slice(0, 1000));
  });

  test("records queued running and terminal timestamps from the host clock", async () => {
    let timestamp = "2026-10-14T06:00:00.000Z";
    let release!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const jobs = maintenance.createMaintenanceJobs({ now: () => timestamp });
    const job = jobs.startJob("embed", {}, async () => { await held; return { chunksEmbedded: 1 }; });
    try {
      expect(job.createdAt).toBe(timestamp);
      expect(job.startedAt).toBeNull();
      expect(job.finishedAt).toBeNull();
      timestamp = "2026-10-14T06:00:01.000Z";
      await Promise.resolve();
      expect(jobs.getJob(job.id)?.state).toBe("running");
      expect(jobs.getJob(job.id)?.startedAt).toBe(timestamp);
      timestamp = "2026-10-14T06:00:02.000Z";
    } finally {
      release();
      await jobs.waitForIdle();
    }
    expect(jobs.getJob(job.id)?.finishedAt).toBe(timestamp);
    expect(jobs.getJob(job.id)?.createdAt).toBe("2026-10-14T06:00:00.000Z");
  });
  test("a throwing terminal clock retains committed results and releases completion", async () => {
    let reads = 0;
    const jobs = maintenance.createMaintenanceJobs({
      now() {
        if (++reads === 3) throw new Error("synthetic private clock detail");
        return "2026-10-14T06:00:00.000Z";
      },
    });
    const result = { chunksEmbedded: 2 };
    const job = jobs.startJob("embed", {}, async () => result);
    const idle = jobs.waitForIdle();
    for (let turn = 0; turn < 8; turn++) await Promise.resolve();
    const claimAfterDrain = jobs.activeJobId;
    let timer!: ReturnType<typeof setTimeout>;
    const settled = await Promise.race([
      Promise.all([idle, jobs.stop()]).then(() => true),
      new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), 50); }),
    ]);
    clearTimeout(timer);
    expect(claimAfterDrain).toBeNull();
    expect(settled).toBe(true);
    expect(jobs.getJob(job.id)).toMatchObject({
      state: "failed", result, finishedAt: null,
      error: "Maintenance completion timestamp unavailable",
    });
  });

  test.each(["host", "default"])("a throwing %s error sanitizer keeps private failures out of stable receipts", async mode => {
    const original = mode === "host" ? new Error("synthetic private execution detail") : {
      toString() { throw new Error("synthetic private coercion detail"); },
    };
    const jobs = maintenance.createMaintenanceJobs(mode === "host" ? {
      sanitizeError() { throw new Error("synthetic private sanitizer detail"); },
    } : {});
    const result = { chunksEmbedded: 2 };
    const job = jobs.startJob("embed", {}, async () => result, () => { throw original; });
    const idle = jobs.waitForIdle();
    for (let turn = 0; turn < 8; turn++) await Promise.resolve();
    const claimAfterDrain = jobs.activeJobId;
    let timer!: ReturnType<typeof setTimeout>;
    const settled = await Promise.race([
      Promise.all([idle, jobs.stop()]).then(() => true),
      new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), 50); }),
    ]);
    clearTimeout(timer);
    expect(claimAfterDrain).toBeNull();
    expect(settled).toBe(true);
    expect(jobs.getJob(job.id)).toMatchObject({
      state: "failed", result, error: "Maintenance error unavailable",
    });
  });


  test("notifies the host once with a complete terminal job receipt", async () => {
    const receipts: maintenance.MaintenanceJob[] = [];
    const jobs = maintenance.createMaintenanceJobs({
      finished(job: maintenance.MaintenanceJob) { receipts.push(job); },
    });
    const result = { chunksEmbedded: 1 };
    const job = jobs.startJob("embed", { collection: "docs" }, async () => result);
    await jobs.waitForIdle();
    expect(receipts).toHaveLength(1);
    expect(receipts[0]).toEqual(jobs.getJob(job.id));
    expect(receipts[0]?.finishedAt).not.toBeNull();
    expect(receipts[0]?.result).toEqual(result);
  });
  test("the creation clock cannot admit reentrant work before claim registration", async () => {
    let once = false;
    let innerError: unknown;
    let innerExecutions = 0;
    const jobs = maintenance.createMaintenanceJobs({
      now() {
        if (!once) {
          once = true;
          try { jobs.startJob("inner", {}, async () => { innerExecutions++; }); }
          catch (error) { innerError = error; }
        }
        return "2026-10-14T07:00:00.000Z";
      },
    });
    const outer = jobs.startJob("outer", {}, async () => ({ committed: true }));
    await jobs.waitForIdle();
    expect(innerError).toBeInstanceOf(Error);
    expect((innerError as Error).message).toBe("Job already active");
    expect(innerExecutions).toBe(0);
    expect(jobs.getJob(outer.id)).toMatchObject({ state: "succeeded", result: { committed: true } });
    expect(jobs.activeJobId).toBeNull();
  });

  test("shutdown inside the creation clock prevents claim registration and execution", async () => {
    let stopping: Promise<void> | undefined;
    let executed = false;
    const jobs = maintenance.createMaintenanceJobs({
      now() {
        stopping = jobs.stop();
        return "2026-10-14T07:00:00.000Z";
      },
    });
    expect(() => jobs.startJob("embed", {}, async () => { executed = true; })).toThrow("stopping");
    await stopping;
    for (let turn = 0; turn < 8; turn++) await Promise.resolve();
    expect(executed).toBe(false);
    expect(jobs.activeJobId).toBeNull();
  });

});
