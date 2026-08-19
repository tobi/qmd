/**
 * Transport-independent shutdown coordinator.
 *
 * One memoized shutdown promise owns admission close, cooperative abort,
 * request/LLM drain, LLM disposal, and store closure. A missed deadline is
 * not another teardown branch: the coordinator hard-stops instead of disposing
 * parents under active native work.
 *
 * The grace watchdog lives here, not in a launcher, because `bin/qmd` is only
 * one of the ways QMD starts. Nix wrappers, `node dist/cli/qmd.js`,
 * `bun src/cli/qmd.ts` and the detached HTTP daemon have no supervising
 * parent, so every trigger kind arms the same deadline.
 */

export type ShutdownTrigger =
  | { kind: "complete"; exitCode: 0 }
  | { kind: "stdin-eof"; exitCode: 0 }
  | { kind: "signal"; signal: "SIGINT" | "SIGTERM"; exitCode: 130 | 143 };

export type ShutdownHooks = {
  /** Synchronous admission barriers. Must not wait. */
  closeAdmission(): void;

  /**
   * Initiates listener/transport closure. The returned promise may remain
   * pending until currently active requests settle.
   */
  stopServing(): void | Promise<void>;

  /** Broadcasts cooperative cancellation to currently active LLM sessions. */
  requestAbort(reason: Error): void;

  /** Both waits have no "continue anyway" timeout. */
  waitForInflight(): Promise<void>;
  waitForLlmIdle(): Promise<void>;

  /** Strict stage-one teardown. */
  disposeLlm(): Promise<void>;

  /** Database/store closure, only after LLM teardown succeeds. */
  closeStore(): void | Promise<void>;

  setExitCode(code: number): void;
  logError(error: unknown): void;

  /**
   * Must not return in production. Defaults to self-SIGKILL: once teardown has
   * failed or blown its deadline, the process cannot dispose native parents
   * safely, and a JS-level exit path may never run if the event loop is stuck
   * inside llama.cpp.
   */
  hardStop?(error: unknown): never;

  /** Keeps Node alive while shutdown is pending, even if all other handles disappear. */
  holdOpen?(): () => void;

  /** Deadline for the whole teardown. Defaults to `QMD_SHUTDOWN_GRACE_MS`. */
  graceMs?: number;

  /** Timer injection point for tests. */
  schedule?(fn: () => void, ms: number): { clear(): void };
};

export type ShutdownCoordinator = {
  shutdown(trigger: ShutdownTrigger): Promise<void>;
  readonly signal: AbortSignal;
};

const settle = <T>(run: () => T | Promise<T>): Promise<{ ok: true; value: T } | { ok: false; error: unknown }> => {
  try {
    return Promise.resolve(run()).then(
      (value) => ({ ok: true as const, value }),
      (error) => ({ ok: false as const, error }),
    );
  } catch (error) {
    return Promise.resolve({ ok: false as const, error });
  }
};

export function createShutdownHold(): () => void {
  const timer = setInterval(() => {}, 1 << 30);
  return () => clearInterval(timer);
}

export type AdmissionLease = {
  closeAdmission(): void;
  enter(): () => void;
  run<T>(fn: () => T | Promise<T>): Promise<T>;
  waitForIdle(): Promise<void>;
  getActiveCount(): number;
};

export function createAdmissionLease(): AdmissionLease {
  let closed = false;
  let active = 0;
  const waiters: Array<() => void> = [];

  const leave = () => {
    active = Math.max(0, active - 1);
    if (active === 0) {
      while (waiters.length > 0) waiters.shift()!();
    }
  };

  const enter = (): (() => void) => {
    if (closed) throw new Error("Admission closed");
    active += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      leave();
    };
  };

  return {
    closeAdmission() {
      closed = true;
    },
    enter,
    async run(fn) {
      const release = enter();
      try {
        return await fn();
      } finally {
        release();
      }
    },
    waitForIdle() {
      if (active === 0) return Promise.resolve();
      return new Promise<void>((resolve) => waiters.push(resolve));
    },
    getActiveCount() {
      return active;
    },
  };
}

export type ShutdownHandoff = {
  shutdown(trigger: ShutdownTrigger): Promise<void>;
  attach(fn: (trigger: ShutdownTrigger) => Promise<void>): Promise<void> | void;
  fail(error: unknown): void;
};

/**
 * Holds shutdown requests until the real owner is attached. Used so a signal
 * during async MCP startup cannot complete against the empty CLI coordinator
 * and then leave the newly started server running.
 */
export function createShutdownHandoff(): ShutdownHandoff {
  let attached: ((trigger: ShutdownTrigger) => Promise<void>) | undefined;
  let pending: ShutdownTrigger | undefined;
  let failed: unknown;
  const waiters: Array<{
    resolve: () => void;
    reject: (error: unknown) => void;
  }> = [];

  const remember = (trigger: ShutdownTrigger): void => {
    if (!pending || (pending.exitCode === 0 && trigger.exitCode !== 0)) {
      pending = trigger;
    }
  };

  return {
    shutdown(trigger) {
      if (failed !== undefined) return Promise.reject(failed);
      if (attached) return attached(trigger);
      remember(trigger);
      return new Promise<void>((resolve, reject) => {
        waiters.push({ resolve, reject });
      });
    },
    attach(fn) {
      attached = fn;
      const trigger = pending;
      pending = undefined;
      if (!trigger) return;
      const result = fn(trigger);
      void Promise.resolve(result).then(
        () => {
          for (const waiter of waiters.splice(0)) waiter.resolve();
        },
        (error) => {
          for (const waiter of waiters.splice(0)) waiter.reject(error);
        },
      );
      return result;
    },
    fail(error) {
      failed = error;
      for (const waiter of waiters.splice(0)) waiter.reject(error);
    },
  };
}

export function resolveShutdownGraceMs(env = process.env.QMD_SHUTDOWN_GRACE_MS, fallback = 30_000): number {
  if (env === undefined || env === "") return fallback;
  const n = Number(env);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

/**
 * Terminate without further native teardown.
 *
 * Reaching this point means either a teardown stage failed or the grace
 * deadline expired, so disposing parents beneath possibly-live native work is
 * exactly what must not happen. `process.exit()` would still run libc `exit()`
 * and the ggml static destructors, and it cannot preempt a blocked event loop,
 * so self-SIGKILL is the only reliable stop.
 */
export function hardStopProcess(error: unknown): never {
  try {
    process.stderr.write(`QMD fatal: ${formatShutdownError(error)}; terminating without native teardown.\n`);
  } catch {}
  process.kill(process.pid, "SIGKILL");
  throw error instanceof Error ? error : new Error(String(error));
}

export function createShutdownCoordinator(hooks: ShutdownHooks): ShutdownCoordinator {
  let shutdownPromise: Promise<void> | undefined;
  let requestedExitCode = 0;
  const abortController = new AbortController();
  const hardStop = hooks.hardStop ?? hardStopProcess;
  const graceMs = hooks.graceMs ?? resolveShutdownGraceMs();
  const schedule = hooks.schedule ?? ((fn: () => void, ms: number) => {
    const timer = setTimeout(fn, ms);
    return { clear: () => clearTimeout(timer) };
  });
  let watchdog: { clear(): void } | undefined;

  const shutdown = (trigger: ShutdownTrigger): Promise<void> => {
    // A signal upgrades an EOF/success shutdown. Repeated signals do not
    // start another teardown.
    if (trigger.exitCode !== 0 && requestedExitCode === 0) {
      requestedExitCode = trigger.exitCode;
    }

    if (shutdownPromise) return shutdownPromise;

    shutdownPromise = (async () => {
      let releaseHold: (() => void) | undefined;
      // Armed for every trigger kind, disarmed only after a clean teardown.
      // hardStop never returns, so the throw is deliberately not caught: a
      // caught hard stop would hand control back to the event loop and let
      // teardown carry on past the deadline, which is exactly the failure the
      // deadline exists to prevent.
      watchdog = schedule(() => {
        hardStop(new Error(`Shutdown exceeded ${graceMs}ms grace`));
      }, graceMs);
      try {
        releaseHold = hooks.holdOpen?.();

        const reason =
          trigger.kind === "signal"
            ? new Error(`Shutdown requested by ${trigger.signal}`)
            : new Error(`Shutdown requested by ${trigger.kind}`);

        // 1. No new request or LLM session may enter after this point.
        hooks.closeAdmission();

        // 2. Initiate listener/transport closure, but do not await it yet:
        //    the close promise may depend on active handlers finishing.
        const servingResult = settle(hooks.stopServing);

        // 3. Ask active work to stop at the next cooperative boundary.
        if (!abortController.signal.aborted) {
          abortController.abort(reason);
        }
        hooks.requestAbort(reason);

        // 4. Active request code must return before dependencies are removed.
        await hooks.waitForInflight();

        // 5. Every LLM session/native operation lease must be released.
        await hooks.waitForLlmIdle();

        // The listener/transport must now be completely closed.
        const serving = await servingResult;
        if (!serving.ok) throw serving.error;

        // 6. Stage-one dependency teardown.
        await hooks.disposeLlm();

        // 7. Only now close SQLite/store ownership.
        await hooks.closeStore();

        // 8. Natural exit.
        hooks.setExitCode(requestedExitCode);
        watchdog?.clear();
        watchdog = undefined;
        releaseHold?.();
      } catch (error) {
        try {
          hooks.logError(error);
        } catch {
          // logging must not hide the original failure
        }

        // Do not release the hold, do not disarm the watchdog, and do not
        // continue to lower tiers. A known failure hard-stops immediately;
        // a hung wait is killed when the watchdog fires.
        hardStop(error);
      }
    })();

    return shutdownPromise;
  };

  return {
    shutdown,
    signal: abortController.signal,
  };
}

export function applyProcessExitCode(code: number): void {
  const prior = typeof process.exitCode === "number" ? process.exitCode : undefined;
  if (prior === undefined || prior === 0 || code !== 0) {
    process.exitCode = code;
  }
}

export function formatShutdownError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
