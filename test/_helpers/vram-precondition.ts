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
 *  - A probe failure (cannot load `node-llama-cpp`, `getVramState` throws,
 *    etc.) fails open and returns `true` so a real failure surfaces in the
 *    test rather than getting silently masked.
 *
 * The dependency on `node-llama-cpp` is injected through `options.getLlama`
 * so unit tests of this helper can drive the gate without the real runtime.
 */

export type LlamaLike = {
  gpu: false | string;
  getVramState: () => Promise<{ total: number; used: number; free: number }>;
};

export type PreconditionOptions = {
  getLlama?: () => Promise<LlamaLike>;
  minFreeGB?: number;
  log?: (message: string) => void;
};

const DEFAULT_MIN_FREE_GB = 4;

let cachedResult: boolean | null = null;
let loggedOnce = false;

async function defaultGetLlama(): Promise<LlamaLike> {
  const mod = await import("node-llama-cpp");
  return await mod.getLlama() as unknown as LlamaLike;
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

  const getter = options.getLlama ?? defaultGetLlama;

  try {
    const llama = await getter();
    if (!llama.gpu) {
      cachedResult = true;
      return true;
    }
    const vram = await llama.getVramState();
    const freeGB = vram.free / (1024 * 1024 * 1024);
    cachedResult = freeGB >= minFreeGB;
    if (!cachedResult && !loggedOnce) {
      log(
        `⚠ Integration tests skipped: only ${freeGB.toFixed(2)} GB free GPU VRAM `
          + `(need ≥ ${minFreeGB} GB). Set QMD_SKIP_GPU_INTEGRATION=1 to silence this probe.`,
      );
      loggedOnce = true;
    }
    return cachedResult;
  } catch {
    // Fail open: if we cannot probe, let the test attempt run. A real failure
    // is more useful than a silent skip masking the symptom.
    cachedResult = true;
    return true;
  }
}

/**
 * Test-only: reset the cached probe result so a unit test can re-exercise
 * the helper. Not for production code paths.
 */
export function _resetVramPreconditionCacheForTests(): void {
  cachedResult = null;
  loggedOnce = false;
}
