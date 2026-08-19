/**
 * VRAM precondition for GPU-dependent integration suites.
 *
 * Integration suites under `test/` load real GGUF models and exercise rerank,
 * embed, and expand against a real `node-llama-cpp` runtime. When another
 * workload is hogging the GPU (Ollama parking a large model, gbrain mid-job,
 * a stale `qmd serve`), those allocations fail and the suite turns into a
 * pile of red `Failed to create any rerank context` lines that do not
 * represent code regressions.
 *
 * This helper probes free VRAM once per process and reports whether there
 * is enough headroom for the integration suite to run meaningfully. The
 * existing skip-gates layer onto it: `CI`, `QMD_REMOTE_URL`, and now this
 * probe each contribute a reason to skip. The probe value is cached so
 * multiple suites do not re-probe.
 *
 * Escape hatches:
 *  - `QMD_SKIP_GPU_INTEGRATION=1` forces a skip without probing (useful on
 *    healthy boxes when you want fast unit-loop iteration).
 *  - `QMD_TEST_EVACUATE_VRAM=1` opts into an Ollama-only eviction attempt
 *    when the initial probe shows insufficient VRAM. Off by default: test
 *    infra silently terminating another process's loaded models would be
 *    astonishing without explicit consent.
 *  - A probe failure (cannot load `node-llama-cpp`, `getVramState` throws,
 *    etc.) fails open and returns `true` so a real failure surfaces in the
 *    test rather than getting silently masked.
 *
 * The dependency on `node-llama-cpp` is injected through `options.getLlama`
 * and the eviction step through `options.evacuate` so unit tests of this
 * helper can drive the gate without the real runtime.
 */

export type LlamaLike = {
  gpu: false | string;
  getVramState: () => Promise<{ total: number; used: number; free: number }>;
};

export type PreconditionOptions = {
  getLlama?: () => Promise<LlamaLike>;
  minFreeGB?: number;
  log?: (message: string) => void;
  evacuate?: () => Promise<void>;
};

const DEFAULT_MIN_FREE_GB = 4;
const DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";

let cachedResult: boolean | null = null;
let loggedOnce = false;

async function defaultGetLlama(): Promise<LlamaLike> {
  const mod = await import("node-llama-cpp");
  return await mod.getLlama() as unknown as LlamaLike;
}

/**
 * Best-effort: list Ollama's currently loaded models and ask the server to
 * unload each by re-issuing the request with `keep_alive: 0`. A missing or
 * unreachable Ollama is fine — nothing to evict. Other VRAM consumers
 * (qmd-serve, gbrain) are deliberately out of scope: killing a user-facing
 * daemon is too astonishing even behind an opt-in flag.
 */
async function defaultEvacuate(log: (m: string) => void): Promise<void> {
  const host = process.env.OLLAMA_HOST ?? DEFAULT_OLLAMA_HOST;
  try {
    const res = await fetch(`${host}/api/ps`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) {
      log(`⚠ Eviction: Ollama /api/ps returned ${res.status}; skipping.`);
      return;
    }
    const body = await res.json() as { models?: { name?: string }[] };
    const names = (body.models ?? []).map(m => m.name).filter((n): n is string => typeof n === "string");
    if (names.length === 0) {
      log("⚠ Eviction: Ollama reports no loaded models; nothing to evict.");
      return;
    }
    log(`⚠ Eviction: asking Ollama to unload ${names.length} model(s): ${names.join(", ")}`);
    for (const name of names) {
      try {
        await fetch(`${host}/api/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: name, keep_alive: 0 }),
          signal: AbortSignal.timeout(5000),
        });
      } catch (e) {
        log(`⚠ Eviction: unload of ${name} failed: ${String(e)}`);
      }
    }
  } catch (e) {
    log(`⚠ Eviction: Ollama unreachable at ${host}: ${String(e)}`);
  }
}

async function probeFreeGB(
  options: PreconditionOptions,
): Promise<{ ok: boolean; freeGB: number | null }> {
  const getter = options.getLlama ?? defaultGetLlama;
  try {
    const llama = await getter();
    if (!llama.gpu) return { ok: true, freeGB: null };
    const vram = await llama.getVramState();
    const freeGB = vram.free / (1024 * 1024 * 1024);
    const minFreeGB = options.minFreeGB ?? DEFAULT_MIN_FREE_GB;
    return { ok: freeGB >= minFreeGB, freeGB };
  } catch {
    // Fail open at the probe layer: the caller decides what to do with `ok`
    // when the probe itself was reliable vs. when it threw.  Returning ok=true
    // here means a probe failure does not gate the suite — a real code
    // failure is more useful than a silent skip masking it.
    return { ok: true, freeGB: null };
  }
}

export async function hasSufficientFreeVramForIntegration(
  options: PreconditionOptions = {},
): Promise<boolean> {
  if (cachedResult !== null) return cachedResult;

  const log = options.log ?? ((m: string) => console.warn(m));
  const minFreeGB = options.minFreeGB ?? DEFAULT_MIN_FREE_GB;

  if (process.env.QMD_SKIP_GPU_INTEGRATION === "1") {
    cachedResult = false;
    if (!loggedOnce) {
      log("⚠ Integration tests skipped: QMD_SKIP_GPU_INTEGRATION=1");
      loggedOnce = true;
    }
    return false;
  }

  const first = await probeFreeGB(options);
  if (first.ok) {
    cachedResult = true;
    return true;
  }

  // Insufficient on the first probe.  If the operator opted into eviction,
  // try to free VRAM from known consumers and re-probe once.  Otherwise skip.
  if (process.env.QMD_TEST_EVACUATE_VRAM === "1") {
    const evacuate = options.evacuate ?? (() => defaultEvacuate(log));
    log(
      `⚠ Insufficient VRAM (${first.freeGB?.toFixed(2) ?? "?"} GB free, need ≥ ${minFreeGB} GB); `
        + `QMD_TEST_EVACUATE_VRAM=1, attempting to evict known consumers...`,
    );
    try {
      await evacuate();
    } catch (e) {
      log(`⚠ Eviction step threw: ${String(e)}; re-probing anyway.`);
    }
    const second = await probeFreeGB(options);
    cachedResult = second.ok;
    if (second.ok) {
      log(`✓ Eviction succeeded; ${second.freeGB?.toFixed(2) ?? "?"} GB free, integration tests will run.`);
    } else if (!loggedOnce) {
      log(
        `⚠ Integration tests skipped: post-eviction only ${second.freeGB?.toFixed(2) ?? "?"} GB free `
          + `(need ≥ ${minFreeGB} GB).`,
      );
      loggedOnce = true;
    }
    return second.ok;
  }

  cachedResult = false;
  if (!loggedOnce) {
    log(
      `⚠ Integration tests skipped: only ${first.freeGB?.toFixed(2) ?? "?"} GB free GPU VRAM `
        + `(need ≥ ${minFreeGB} GB). Set QMD_TEST_EVACUATE_VRAM=1 to attempt eviction first, `
        + `or QMD_SKIP_GPU_INTEGRATION=1 to silence this probe.`,
    );
    loggedOnce = true;
  }
  return false;
}

/**
 * Test-only: reset the cached probe result so a unit test can re-exercise
 * the helper. Not for production code paths.
 */
export function _resetVramPreconditionCacheForTests(): void {
  cachedResult = null;
  loggedOnce = false;
}
