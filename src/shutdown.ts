/**
 * Transport-independent shutdown coordinator.
 *
 * One memoized shutdown promise owns admission close, cooperative abort,
 * request/LLM drain, LLM disposal, and store closure. A missed deadline is
 * not another teardown branch: the external supervisor must SIGKILL instead
 * of disposing parents under active native work.
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
   * Must not return in production. For supervised children this can be
   * omitted on the normal path; the supervisor owns the deadline.
   */
  hardStop?(error: unknown): never;

  /** Keeps Node alive while shutdown is pending, even if all other handles disappear. */
  holdOpen?(): () => void;
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

export type EofWatchdog = {
  arm(): void;
  disarm(): void;
};

/**
 * Stdio EOF is not observed by bin/qmd, so the child must own a hard-stop
 * deadline. The watchdog only kills the process; it never continues parent
 * disposal. A JS timer cannot fire if the event loop is blocked in native code.
 */
export function createEofWatchdog(options: {
  graceMs?: number;
  hardStop?: (error: unknown) => void;
  schedule?: (fn: () => void, ms: number) => { clear(): void };
} = {}): EofWatchdog {
  const graceMs = options.graceMs ?? resolveShutdownGraceMs();
  const hardStop = options.hardStop ?? ((error: unknown) => {
    try {
      process.stderr.write("qmd: stdio shutdown exceeded grace; forcing exit\n");
    } catch {}
    process.kill(process.pid, "SIGKILL");
    throw error instanceof Error ? error : new Error(String(error));
  });
  const schedule = options.schedule ?? ((fn, ms) => {
    const timer = setTimeout(fn, ms);
    return { clear: () => clearTimeout(timer) };
  });
  let handle: { clear(): void } | undefined;

  return {
    arm() {
      if (handle) return;
      handle = schedule(() => {
        hardStop(new Error("stdio EOF shutdown exceeded grace"));
      }, graceMs);
    },
    disarm() {
      handle?.clear();
      handle = undefined;
    },
  };
}

export function createShutdownCoordinator(hooks: ShutdownHooks): ShutdownCoordinator {
  let shutdownPromise: Promise<void> | undefined;
  let requestedExitCode = 0;
  const abortController = new AbortController();

  const shutdown = (trigger: ShutdownTrigger): Promise<void> => {
    // A signal upgrades an EOF/success shutdown. Repeated signals do not
    // start another teardown.
    if (trigger.exitCode !== 0 && requestedExitCode === 0) {
      requestedExitCode = trigger.exitCode;
    }

    if (shutdownPromise) return shutdownPromise;

    shutdownPromise = (async () => {
      let releaseHold: (() => void) | undefined;
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
        releaseHold?.();
      } catch (error) {
        try {
          hooks.logError(error);
        } catch {
          // logging must not hide the original failure
        }

        // Do not release the hold and do not continue to lower tiers.
        // A known failure can hard-stop immediately; a hung wait is killed
        // by the external supervisor's deadline.
        if (hooks.hardStop) hooks.hardStop(error);
        throw error;
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
