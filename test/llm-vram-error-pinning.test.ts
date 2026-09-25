/**
 * llm-vram-error-pinning.test.ts — Tripwire tests for `isInsufficientVramError`.
 *
 * `isInsufficientVramError` is the trigger for the two-pass reclaim path in
 * `withReclaim`.  Its regex catalog has to match the real strings that
 * node-llama-cpp emits, or the reclaim never fires under genuine VRAM
 * pressure and the user just gets a raw failure.
 *
 * These tests pin against a curated list of strings sourced from:
 *  - node-llama-cpp source (`LlamaContext.js` `InsufficientMemoryError`
 *    template at line 30: `A context size of ${resolvedContextSize}${...} is
 *    too large for the available VRAM`).
 *  - The synthetic strings this codebase throws itself when an inner
 *    `createRankingContext` / `createEmbeddingContext` loop exhausts retries.
 *  - CUDA / GGML out-of-memory variants observed under load.
 *
 * Treat any failure here as a signal that either:
 *   (a) the upstream library reworded an error and the regex needs a new
 *       alternative, OR
 *   (b) a new error path the reclaim should recognize landed in our own code
 *       and is missing from the catalog.
 *
 * Adding a positive case here that fails before the regex update is the
 * preferred way to fix a "reclaim didn't fire when it should have" report.
 */

import { describe, test, expect } from "vitest";
import { isInsufficientVramError } from "../src/llm.js";

/**
 * Real strings the regex MUST recognize as VRAM pressure.
 *
 * Each entry pairs a string with the source it came from so a future
 * reviewer can trace why the catalog includes it.  Adding a new entry?
 * Cite the source (library file + line, or a docs/solutions/ ref).
 */
const POSITIVE_CASES: readonly { source: string; message: string }[] = [
  {
    source: "node-llama-cpp LlamaContext.js:30 — InsufficientMemoryError template (single-sequence)",
    message: "A context size of 2048 is too large for the available VRAM",
  },
  {
    source: "node-llama-cpp LlamaContext.js:30 — InsufficientMemoryError template (multi-sequence)",
    message: "A context size of 4096 with 4 sequences is too large for the available VRAM",
  },
  {
    source: "src/llm.ts — synthetic error thrown when all rerank context allocations fail",
    message: "Failed to create any rerank context",
  },
  {
    source: "src/llm.ts — synthetic error thrown when all embedding context allocations fail",
    message: "Failed to create any embedding context",
  },
  {
    source: "Generic CUDA / GGML out-of-memory variant",
    message: "CUDA error: out of memory",
  },
  {
    source: "GGML out-of-memory variant under load",
    message: "ggml_cuda_compute_forward: out of memory",
  },
  {
    source: "Lower-case variant of the OOM message",
    message: "ran out of memory while allocating buffer",
  },
  {
    source: "node-llama-cpp insufficientVRAM safety check message variant",
    message: "Insufficient VRAM to load the model",
  },
];

/**
 * Real strings the regex MUST NOT recognize.  False positives here would
 * cause `withReclaim` to chew through retries on errors that dropping
 * models cannot fix — wasting time and leaving the user no closer to a
 * diagnostic for the real failure.
 */
const NEGATIVE_CASES: readonly { reason: string; message: string }[] = [
  {
    reason: "Model file missing / unreadable — not a VRAM problem",
    message: "model file not found at /tmp/missing.gguf",
  },
  {
    reason: "Wrong model architecture for the requested operation",
    message: "Computing rankings is not supported for this model.",
  },
  {
    reason: "Network failure during model download",
    message: "fetch failed: connect ECONNREFUSED 127.0.0.1:443",
  },
  {
    reason: "GGUF file corruption check",
    message: "Invalid GGUF magic bytes",
  },
  {
    reason: "Generic runtime error with no memory-related keywords",
    message: "Something else went wrong",
  },
  {
    reason: "Mentions 'memory' but not OOM (red-team: substring greediness)",
    message: "Loaded model into memory successfully",
  },
  {
    reason: "Mentions 'VRAM' but as part of a status string (red-team)",
    message: "VRAM usage: 12.3 GB / 24 GB",
  },
];

describe("isInsufficientVramError — pinned positives", () => {
  test.each(POSITIVE_CASES)(
    "recognizes: $message ($source)",
    ({ message }) => {
      expect(isInsufficientVramError(new Error(message))).toBe(true);
    },
  );
});

describe("isInsufficientVramError — pinned negatives", () => {
  test.each(NEGATIVE_CASES)(
    "does NOT recognize: $message ($reason)",
    ({ message }) => {
      expect(isInsufficientVramError(new Error(message))).toBe(false);
    },
  );
});

describe("isInsufficientVramError — input shape handling", () => {
  test("returns false for null", () => {
    expect(isInsufficientVramError(null)).toBe(false);
  });

  test("returns false for undefined", () => {
    expect(isInsufficientVramError(undefined)).toBe(false);
  });

  test("returns false for empty string", () => {
    expect(isInsufficientVramError("")).toBe(false);
  });

  test("accepts a bare string with a positive message", () => {
    expect(isInsufficientVramError("A context size of 2048 is too large for the available VRAM")).toBe(true);
  });

  test("accepts a non-Error object via String() coercion", () => {
    const weird = { toString: () => "out of memory" };
    expect(isInsufficientVramError(weird)).toBe(true);
  });

  test("regex is case-insensitive across the catalog", () => {
    expect(isInsufficientVramError(new Error("A CONTEXT SIZE OF 2048 IS TOO LARGE FOR THE AVAILABLE VRAM"))).toBe(true);
    expect(isInsufficientVramError(new Error("out OF memory"))).toBe(true);
  });
});
