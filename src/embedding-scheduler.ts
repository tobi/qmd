import { performance } from "node:perf_hooks";
import type { QMDStore } from "./index.js";
import type { createMaintenanceJobs } from "./maintenance-jobs.js";

type EmbedOptions = NonNullable<Parameters<QMDStore["embed"]>[0]>;
type SchedulerClock = {
  now(): number;
  monotonic(): number;
  setTimeout(callback: () => void, milliseconds: number): unknown;
  clearTimeout(timer: unknown): void;
  setInterval(callback: () => void, milliseconds: number): unknown;
  clearInterval(timer: unknown): void;
};
const systemClock: SchedulerClock = {
  now: Date.now, monotonic: () => performance.now(), setTimeout, setInterval,
  clearTimeout: timer => clearTimeout(timer as ReturnType<typeof setTimeout>),
  clearInterval: timer => clearInterval(timer as ReturnType<typeof setInterval>),
};
export type EmbeddingSchedulerOptions = {
  intervalMinutes?: number;
  initialDelaySeconds?: number;
  maxDurationMs?: number;
  maxDocsPerBatch?: number;
  maxBatchBytes?: number;
  maintenance: ReturnType<typeof createMaintenanceJobs>;
  /** Reload the host's embedding-enabled collection names on each call. */
  collections(): string[];
  pending(collections?: string[]): number;
  activeQueries(): number;
  embed(options: EmbedOptions): Promise<{ chunksEmbedded: number; errors: number }>;
  clock?: SchedulerClock;
  sanitizeError?: (error: unknown) => string;
};

/** Optional host-owned timer; construction alone never starts background work. */
export function createEmbeddingScheduler({
  intervalMinutes = 0, initialDelaySeconds = 0, maxDurationMs = 30 * 60 * 1000, maxDocsPerBatch = 10,
  maxBatchBytes = 4 * 1024 * 1024, maintenance, collections, pending, activeQueries, embed,
  clock = systemClock,
  sanitizeError = error => (error instanceof Error ? error.message : String(error)).slice(0, 1000),
}: EmbeddingSchedulerOptions) {
  for (const [name, value] of [["maxDurationMs", maxDurationMs], ["maxDocsPerBatch", maxDocsPerBatch], ["maxBatchBytes", maxBatchBytes]] as const) {
    if (!Number.isSafeInteger(value) || value < 1 || (name === "maxDurationMs" && value > 2147483647)) {
      throw new RangeError(`${name} must be a positive bounded safe integer`);
    }
  }
  for (const [name, value, multiplier] of [["intervalMinutes", intervalMinutes, 60000], ["initialDelaySeconds", initialDelaySeconds, 1000]] as const) {
    if (!Number.isFinite(value) || value < 0 || value * multiplier > 2147483647 || (value > 0 && value * multiplier < 1)) {
      throw new RangeError(`${name} must be a nonnegative bounded timer duration`);
    }
  }
  let last: { state: string; error?: string; jobId?: string } | null = null;
  let stopped = false;
  let started = false;
  let initialTimer: unknown = null;
  let intervalTimer: unknown = null;
  let next: string | null = null;
  let lastFirstCollection: string | null = null;
  const safeError = (error: unknown) => {
    try { return sanitizeError(error); } catch { return "Embedding error unavailable"; }
  };
  let runController: AbortController | null = null;
  let sweepCompletion: Promise<void> | null = null;
  return {
    start() {
      if (intervalMinutes <= 0 || started || stopped) return;
      started = true;
      const intervalMs = intervalMinutes * 60000;
      try {
        const initialNow = clock.now();
        if (stopped) return;
        next = new Date(initialNow + initialDelaySeconds * 1000).toISOString();
        const initial = clock.setTimeout(() => {
          if (stopped) return;
          const timer = initialTimer;
          initialTimer = null;
          try {
            clock.clearTimeout(timer);
            if (stopped) return;
            const initialTickNow = clock.now();
            if (stopped) return;
            next = new Date(initialTickNow + intervalMs).toISOString();
            this.tick();
            if (stopped) return;
            const interval = clock.setInterval(() => {
              if (stopped) return;
              try {
                const intervalNow = clock.now();
                if (stopped) return;
                next = new Date(intervalNow + intervalMs).toISOString();
                this.tick();
              } catch (error) {
                next = null;
                last = { state: "failed_to_start", error: safeError(error) };
              }
            }, intervalMs);
            if (stopped) {
              clock.clearInterval(interval);
              return;
            }
            intervalTimer = interval;
          } catch (error) {
            started = false;
            next = null;
            last = { state: "failed_to_start", error: safeError(error) };
          }
        }, initialDelaySeconds * 1000);
        if (stopped) {
          clock.clearTimeout(initial);
          return;
        }
        initialTimer = initial;
      } catch (error) {
        started = false;
        next = null;
        last = { state: "failed_to_start", error: safeError(error) };
      }
    },
    async stop() {
      stopped = true;
      const initial = initialTimer, interval = intervalTimer;
      initialTimer = intervalTimer = null; next = null;
      if (initial !== null) {
        try { clock.clearTimeout(initial); }
        catch (error) { last = { ...last, state: "cleanup_failed", error: safeError(error) }; }
      }
      if (interval !== null) {
        try { clock.clearInterval(interval); }
        catch (error) { last = { ...last, state: "cleanup_failed", error: safeError(error) }; }
      }
      runController?.abort(); await sweepCompletion;
    },
    tick() {
      try {
      if (intervalMinutes <= 0 || stopped || maintenance.activeJobId || activeQueries() > 0) return null;
      let selected = collections().filter(name => pending([name]) > 0);
      if (!selected.length || stopped) return null;
      const previous = lastFirstCollection === null ? -1 : selected.indexOf(lastFirstCollection);
      if (previous >= 0 && selected.length > 1) {
        const start = (previous + 1) % selected.length;
        selected = [...selected.slice(start), ...selected.slice(0, start)];
      }
      const job = maintenance.startJob("scheduled_embed", { collections: selected, trigger: "timer", force: false },
        async (setProgress, maintenanceSignal) => {
          const controller = new AbortController();
          runController = controller;
          if (stopped) controller.abort();
          const signal = AbortSignal.any([controller.signal, maintenanceSignal]);
          let deadlineTimer: unknown = null;
          let deadline = 0;
          const result = { embeddings: [] as { collection: string; chunksEmbedded: number; errors: number }[], collectionErrors: [] as { collection: string | null; error: string }[], chunksEmbedded: 0, needsEmbedding: null as number | null, stopReason: null as string | null };
          const stopReason = () => stopped || maintenanceSignal.aborted ? "shutdown"
            : signal.aborted || clock.monotonic() >= deadline ? "deadline"
            : activeQueries() > 0 ? "querying" : null;
          try {
          deadlineTimer = clock.setTimeout(() => controller.abort(), maxDurationMs);
          deadline = clock.monotonic() + maxDurationMs;
          for (const collection of selected) {
            const reason = stopReason();
            if (reason) { result.stopReason = reason; break; }
            if (!collections().includes(collection) || pending([collection]) === 0) continue;
            let observedChunks = 0;
            try {
            const value = await embed({ collection, force: false, chunkStrategy: "auto", resume: true,
              maxDocsPerBatch, maxBatchBytes, maxDurationMs: Math.max(1, deadline - clock.monotonic()),
              signal, onProgress: progress => {
                observedChunks = Math.max(observedChunks, progress.chunksEmbedded || 0);
                setProgress({ ...progress, collection });
              },
            });
            result.embeddings.push({ ...value, collection });
            result.chunksEmbedded += value.chunksEmbedded;
            } catch (error) {
              result.chunksEmbedded += observedChunks;
              result.collectionErrors.push({ collection, error: safeError(error) });
            }
          }
          result.stopReason = stopReason() || result.stopReason;
          } catch (error) {
            result.collectionErrors.push({ collection: null, error: safeError(error) });
          } finally {
            try { if (deadlineTimer !== null) clock.clearTimeout(deadlineTimer); }
            catch (error) { result.collectionErrors.push({ collection: null, error: safeError(error) }); }
            finally { runController = null; }
          }
          try { result.needsEmbedding = pending(); }
          catch (error) { result.collectionErrors.push({ collection: null, error: safeError(error) }); }
          return result;
        }, value => {
          const result = value as { chunksEmbedded: number; collectionErrors: unknown[]; needsEmbedding: number | null; stopReason: string | null; embeddings: { errors: number }[] };
          const errors = result.collectionErrors.length + result.embeddings.reduce((sum, item) => sum + item.errors, 0);
          if (errors > 0 && result.chunksEmbedded === 0 && !result.embeddings.some(item => item.errors === 0)) return "failed";
          return result.needsEmbedding !== 0 || result.stopReason || errors > 0 ? "partial" : "succeeded";
        });
      lastFirstCollection = selected[0] ?? null;
      sweepCompletion = maintenance.waitForIdle();
      last = { state: "started", jobId: job.id };
      return job;
      } catch (error) {
        last = { state: "failed_to_start", error: safeError(error) };
        return null;
      }
    },
    health() { return { enabled: intervalMinutes > 0, next, last }; },
  };
}
