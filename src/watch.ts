/**
 * Watch mode for `qmd update` — periodic re-indexing.
 *
 * Polls collections on a fixed interval. Cheap because reindexCollection()
 * hashes files and skips unchanged ones. Optionally re-embeds after each tick.
 *
 * Cross-platform: uses setInterval rather than fs.watch so behavior is
 * consistent on Linux, macOS, and Windows without an extra dependency.
 */

const DURATION_RE = /^(\d+(?:\.\d+)?)(ms|s|m|h)?$/i;

/**
 * Parse a duration string like "30s", "5m", "1.5h", "500ms" into milliseconds.
 * Bare numbers are treated as seconds for ergonomics.
 *
 * @throws Error on invalid input
 */
export function parseDurationMs(input: string): number {
  if (typeof input !== "string" || input.trim() === "") {
    throw new Error(`Invalid duration: ${JSON.stringify(input)}`);
  }
  const match = input.trim().match(DURATION_RE);
  if (!match) {
    throw new Error(
      `Invalid duration: ${JSON.stringify(input)}. Use 30s, 5m, 1h, or 500ms.`,
    );
  }
  const value = Number(match[1]);
  const unit = (match[2] ?? "s").toLowerCase();
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid duration: ${JSON.stringify(input)} (must be > 0)`);
  }
  switch (unit) {
    case "ms":
      return Math.round(value);
    case "s":
      return Math.round(value * 1000);
    case "m":
      return Math.round(value * 60_000);
    case "h":
      return Math.round(value * 3_600_000);
    default:
      throw new Error(`Unknown duration unit: ${unit}`);
  }
}

export interface WatchOptions {
  /** Interval between ticks in milliseconds. */
  intervalMs: number;
  /** Run one tick body. May throw; the loop handles it. */
  tick: () => Promise<void>;
  /** Optional logger; defaults to console.error so stdio MCP stays clean. */
  log?: (msg: string) => void;
  /** Stop after this many consecutive failures (default 3). */
  maxConsecutiveFailures?: number;
  /**
   * For tests: stop the loop after N ticks. Undefined = run forever (until
   * SIGINT/SIGTERM or maxConsecutiveFailures).
   */
  maxTicks?: number;
  /**
   * For tests: skip installing real signal handlers. Production code leaves
   * this undefined.
   */
  installSignalHandlers?: boolean;
}

export interface WatchResult {
  ticks: number;
  failures: number;
  stoppedBy: "signal" | "max-ticks" | "max-failures";
}

/**
 * Run a tick function on a fixed interval until interrupted.
 *
 * Behavior:
 * - Fires one tick immediately on start, then every intervalMs.
 * - Ticks never overlap; if a tick takes longer than the interval, the next
 *   tick fires immediately on completion.
 * - SIGINT / SIGTERM stop the loop cleanly between ticks.
 * - maxConsecutiveFailures (default 3) triggers a non-zero exit.
 */
export async function runWatchLoop(opts: WatchOptions): Promise<WatchResult> {
  const intervalMs = opts.intervalMs;
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new Error(`Invalid intervalMs: ${intervalMs}`);
  }
  const log = opts.log ?? ((msg: string) => process.stderr.write(msg + "\n"));
  const maxFailures = opts.maxConsecutiveFailures ?? 3;
  const installSignals = opts.installSignalHandlers !== false;

  let ticks = 0;
  let consecutiveFailures = 0;
  let totalFailures = 0;
  let stop = false;
  let stoppedBy: WatchResult["stoppedBy"] = "signal";
  let resolveStop: (() => void) | null = null;

  const onSignal = (sig: NodeJS.Signals) => {
    log(`Received ${sig}, stopping watch loop after current tick...`);
    stop = true;
    stoppedBy = "signal";
    if (resolveStop) resolveStop();
  };

  if (installSignals) {
    process.on("SIGINT", onSignal);
    process.on("SIGTERM", onSignal);
  }

  try {
    while (!stop) {
      const tickStart = Date.now();
      try {
        await opts.tick();
        consecutiveFailures = 0;
      } catch (err) {
        consecutiveFailures += 1;
        totalFailures += 1;
        const msg = err instanceof Error ? err.message : String(err);
        log(`Watch tick failed (${consecutiveFailures}/${maxFailures}): ${msg}`);
        if (consecutiveFailures >= maxFailures) {
          stoppedBy = "max-failures";
          stop = true;
          break;
        }
      }

      ticks += 1;
      if (opts.maxTicks !== undefined && ticks >= opts.maxTicks) {
        stoppedBy = "max-ticks";
        break;
      }
      if (stop) break;

      const elapsed = Date.now() - tickStart;
      const delay = Math.max(0, intervalMs - elapsed);
      if (delay > 0) {
        await new Promise<void>((res) => {
          const timer = setTimeout(() => {
            resolveStop = null;
            res();
          }, delay);
          // If a signal fires while we wait, cancel the timer and resolve so
          // the loop exits promptly instead of waiting out the interval.
          resolveStop = () => {
            clearTimeout(timer);
            resolveStop = null;
            res();
          };
        });
      }
    }
  } finally {
    if (installSignals) {
      process.off("SIGINT", onSignal);
      process.off("SIGTERM", onSignal);
    }
  }

  return { ticks, failures: totalFailures, stoppedBy };
}
