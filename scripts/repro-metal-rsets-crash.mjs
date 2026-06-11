#!/usr/bin/env node
/**
 * Minimal reproduction of llama.cpp issue ggml-org/llama.cpp#22593:
 *
 *   ggml-metal-device.m:612: GGML_ASSERT([rsets->data count] == 0) failed
 *
 * Root cause (per the upstream issue and proposed fix PR #22595):
 *   `ggml_metal_buffer_rset_free` releases the per-buffer residency set object
 *   but does NOT call the symmetric `ggml_metal_device_rsets_rm`. So the
 *   device's `rsets->data` array accumulates dangling references. When the
 *   process exits and libc fires the process-static `ggml_metal_device`
 *   destructor in `__cxa_finalize_ranges`, the destructor asserts the
 *   array is empty — and it isn't.
 *
 * Observed downstream behavior:
 *   - With EXPLICIT `dispose()` of every JS handle in order, the assertion
 *     does NOT fire. node-llama-cpp's dispose path tears the Metal buffers
 *     down before the static dtor runs, so the device's rsets array is
 *     empty by exit time. (Tested locally — clean exit.)
 *   - With NO dispose (the typical real-world case: synchronous `exit()`,
 *     `--watch` mode, `process.exit()` after results are written, or any
 *     code path where GC + finalizers race with libc exit), the rset
 *     references linger until the static dtor fires, and the assertion
 *     trips.
 *
 * What this script does:
 *   1. Load node-llama-cpp + a small GGUF model on the Metal backend.
 *      This allocates at least one Metal buffer → calls rsets_add internally.
 *   2. Run an inference (creating an embedding context populates buffers
 *      that the dispose path would normally clean up).
 *   3. Skip explicit dispose. Just let the process exit.
 *
 * Expected behavior on macOS 15+ with Apple Silicon, current llama.cpp
 * (bundled in node-llama-cpp 3.18.1, llama.cpp tag b8390):
 *   - Without GGML_METAL_NO_RESIDENCY:
 *       Script writes "ok" and main() returns, then ggml_abort fires the
 *       assertion, prints a multi-kB backtrace, and the process exits with
 *       SIGABRT (exit code 134).
 *   - With GGML_METAL_NO_RESIDENCY=1:
 *       Clean exit code 0. Residency-set code path is skipped entirely.
 *   - With --dispose flag (manual cleanup):
 *       Clean exit code 0 even without the env var, as long as JS dispose()
 *       runs successfully before libc exit.
 *
 * Usage:
 *   # Reproduce the crash (no dispose, no env var)
 *   node scripts/repro-metal-rsets-crash.mjs
 *
 *   # Verify the documented workaround
 *   GGML_METAL_NO_RESIDENCY=1 node scripts/repro-metal-rsets-crash.mjs
 *
 *   # Verify that explicit dispose also avoids the crash
 *   node scripts/repro-metal-rsets-crash.mjs --dispose
 *
 * Refs:
 *   https://github.com/ggml-org/llama.cpp/issues/22593  (root-cause analysis)
 *   https://github.com/ggml-org/llama.cpp/pull/22595    (one-line fix, open)
 *   https://github.com/tobi/qmd/issues/368              (downstream report)
 *   https://github.com/tobi/qmd/issues/674              (downstream, current)
 *   https://github.com/tobi/qmd/pull/600                (downstream workaround PR)
 */

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

const DEFAULT_MODEL = resolve(
  homedir(),
  ".cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
);

const args = process.argv.slice(2);
const wantsDispose = args.includes("--dispose");
const modelPath = args.find((a) => !a.startsWith("--")) ?? DEFAULT_MODEL;

if (!existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`);
  console.error("Pass a path to any local GGUF as argv[1], or run `qmd embed` once to populate the default cache path.");
  process.exit(2);
}

console.error(
  `[repro] GGML_METAL_NO_RESIDENCY=${process.env.GGML_METAL_NO_RESIDENCY ?? "(unset)"}`,
);
console.error(`[repro] dispose=${wantsDispose}`);
console.error(`[repro] loading: ${modelPath}`);

const { getLlama } = await import("node-llama-cpp");

const llama = await getLlama();
const model = await llama.loadModel({ modelPath });
const context = await model.createEmbeddingContext();

console.error(`[repro] backend: ${llama.gpu}`);

// Run actual inference so the buffer-allocation path is hit.
await context.getEmbeddingFor("repro text");

if (wantsDispose) {
  console.error("[repro] explicit dispose…");
  await context.dispose();
  await model.dispose();
  await llama.dispose();
}

console.error("[repro] main() returning via process.exit(0)");
console.log("ok");

// CRITICAL: use process.exit(), not `return`. node-llama-cpp registers a
// `process.once('beforeExit', …)` hook that auto-disposes WeakRef'd Llama
// instances when the event loop empties naturally. `process.exit()` skips
// `beforeExit`, so the rsets stay populated until libc's `exit()` fires the
// static dtor — which is when the upstream assertion bug trips.
//
// CLI tools (qmd query, qmd vsearch, qmd embed, etc.) all call process.exit()
// after writing results, which is why every real downstream report crashes
// even though the minimal "let main return" version does not.
process.exit(0);
