import { randomUUID } from "node:crypto";

export type MaintenanceJobState = "queued" | "running" | "succeeded" | "partial" | "failed";
export type MaintenanceJob = {
  id: string;
  type: string;
  state: MaintenanceJobState;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  parameters: Record<string, unknown>;
  progress: unknown;
  result: unknown;
  error: string | null;
};

/** One synchronous maintenance claim shared by an SDK host's callers. */
export function createMaintenanceJobs({
  maxRetainedJobs = 20, admit = () => {}, finished = () => {}, now = () => new Date().toISOString(),
  sanitizeError = (error: unknown) => (error instanceof Error ? error.message : String(error)).slice(0, 1000),
}: {
  maxRetainedJobs?: number;
  admit?: (type: string) => void;
  finished?: (job: MaintenanceJob) => void;
  now?: () => string;
  sanitizeError?: (error: unknown) => string;
} = {}) {
  if (!Number.isSafeInteger(maxRetainedJobs) || maxRetainedJobs < 1) {
    throw new RangeError("maxRetainedJobs must be a positive safe integer");
  }
  const jobs = new Map<string, MaintenanceJob>();
  let activeJobId: string | null = null;
  let completion: Promise<void> | null = null;
  let controller: AbortController | null = null;
  let stopping = false;
  let admitting = false;

  function pruneJobs() {
    for (const job of jobs.values()) {
      if (jobs.size <= maxRetainedJobs) break;
      if (job.state !== "queued" && job.state !== "running") jobs.delete(job.id);
    }
  }

  function startJob(
    type: string,
    parameters: Record<string, unknown>,
    execute: (setProgress: (progress: unknown) => void, signal: AbortSignal) => Promise<unknown>,
    classify?: (result: unknown) => "succeeded" | "partial" | "failed",
  ): MaintenanceJob {
    if (stopping) throw new Error("Maintenance is stopping");
    if (activeJobId !== null || admitting) throw new Error("Job already active");
    admitting = true;
    try {
      admit(type);
      if (stopping) throw new Error("Maintenance is stopping");
      const createdAt = now();
      if (stopping) throw new Error("Maintenance is stopping");
      const job: MaintenanceJob = {
        id: randomUUID(), type, state: "queued", createdAt, startedAt: null, finishedAt: null,
        parameters, progress: null, result: null, error: null,
      };
      const jobController = new AbortController();
      controller = jobController;
      let resolveCompletion!: () => void;
      completion = new Promise<void>(resolve => { resolveCompletion = resolve; });
      jobs.set(job.id, job);
      activeJobId = job.id;
      pruneJobs();
      queueMicrotask(() => {
        void (async () => {
          try {
            if (jobController.signal.aborted) throw new Error("Maintenance stopped before execution");
            job.state = "running";
            job.startedAt = now();
            job.result = await execute(progress => { job.progress = progress; }, jobController.signal);
            const reportedErrors = (value: unknown): boolean =>
              value !== null && typeof value === "object" && "errors" in value &&
              typeof value.errors === "number" && Number.isFinite(value.errors) && value.errors > 0;
            const nested = job.result !== null && typeof job.result === "object" &&
              "embeddings" in job.result && Array.isArray(job.result.embeddings)
              ? job.result.embeddings : [];
            const state = classify ? classify(job.result) :
              reportedErrors(job.result) || nested.some(reportedErrors) ? "partial" : "succeeded";
            if (!["succeeded", "partial", "failed"].includes(state)) {
              throw new Error("Invalid maintenance terminal state");
            }
            job.state = state;
          } catch (error) {
            job.state = "failed";
            try {
              job.error = sanitizeError(error);
            } catch {
              job.error = "Maintenance error unavailable";
            }
          } finally {
            try {
              job.finishedAt = now();
            } catch {
              job.state = "failed";
              job.error = "Maintenance completion timestamp unavailable";
            } finally {
              // Host notification is observational and must not strand the claim.
              try { finished({ ...job }); } catch { /* Preserve the work receipt. */ }
              activeJobId = null;
              completion = null;
              controller = null;
              pruneJobs();
              resolveCompletion();
            }
          }
        })();
      });
      return { ...job };
    } finally { admitting = false; }
  }

  return {
    get activeJobId() { return activeJobId; },
    startJob,
    getJob(id: string) {
      const job = jobs.get(id);
      return job ? { ...job } : undefined;
    },
    async waitForIdle() { await completion; },
    async stop() {
      stopping = true;
      controller?.abort();
      await completion;
    },
  };
}
