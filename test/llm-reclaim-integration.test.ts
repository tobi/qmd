/**
 * llm-reclaim-integration.test.ts — End-to-end reclaim test against a real GPU.
 *
 * The synthetic tests in `llm-low-vram.test.ts` prove the orchestration of
 * `withReclaim`. They do NOT prove that:
 *  - the dispose helpers (`disposeEmbedModel`, `disposeRerankModel`,
 *    `disposeGenerateModel`) actually free real VRAM,
 *  - the real node-llama-cpp error message under genuine pressure matches
 *    `isInsufficientVramError` (also pinned in
 *    `llm-vram-error-pinning.test.ts`, but that pins string-by-string;
 *    this exercises a live throw),
 *  - the three-attempt cap actually holds against a real failing allocation
 *    rather than a synthetic one.
 *
 * This file fills the gap by forcing a real allocation failure inside a
 * lowVram-mode `LlamaCpp`. We bump the static `RERANK_CONTEXT_SIZE` to a
 * value no GPU can satisfy, count how many times `rerankImpl` runs, and
 * verify the failure surfaces with a VRAM-shaped error after exactly three
 * attempts (initial + first-pass retry + second-pass retry).
 *
 * The recovery path (reclaim disposes other models and the retry succeeds)
 * requires precise VRAM ballast and tear-down that is hard to make
 * deterministic on a shared GPU; that scenario stays covered by the
 * synthetic tests for now.
 *
 * Gated behind `QMD_RECLAIM_INTEGRATION=1` because:
 *  - It downloads the rerank model on first run (can take minutes).
 *  - It exercises real GPU allocation, so it cannot run in CI or on a
 *    remote-LLM-routed setup.
 *  - It pairs naturally with B3's VRAM precondition: this test does not
 *    care about free VRAM, it only needs enough headroom to LOAD the
 *    rerank model (~400 MB).
 *
 * Run with: `QMD_RECLAIM_INTEGRATION=1 bun test --preload ./src/test-preload.ts test/llm-reclaim-integration.test.ts`
 */

import { describe, test, expect } from "vitest";
import { LlamaCpp, isInsufficientVramError, type RerankDocument } from "../src/llm.js";

const GATE_ON = process.env.QMD_RECLAIM_INTEGRATION === "1";

describe.skipIf(!GATE_ON)("LlamaCpp reclaim integration (real GPU)", () => {
  // Each test creates and disposes its own LlamaCpp so a failure does not
  // leak GPU resources into the next test.

  test("oversized rerank context triggers reclaim and gives up cleanly after 3 attempts", async () => {
    const llm = new LlamaCpp({ lowVram: true });

    // Save and replace the static context-size cap.  The static is captured
    // at class load, so process.env mutation alone would have no effect.
    // Reflective swap is the cleanest way to force a real allocation that
    // no GPU can satisfy.
    const fieldOwner = LlamaCpp as unknown as { RERANK_CONTEXT_SIZE: number };
    const originalSize = fieldOwner.RERANK_CONTEXT_SIZE;
    fieldOwner.RERANK_CONTEXT_SIZE = 1_048_576;  // 1 M tokens, ~290 GB context

    // Spy on rerankImpl to count attempts.  Wrap, do not replace — the real
    // implementation must run so the real lib error surfaces.
    const target = llm as unknown as {
      rerankImpl: (...args: unknown[]) => Promise<unknown>;
    };
    const originalImpl = target.rerankImpl.bind(llm);
    let attempts = 0;
    target.rerankImpl = async (...args: unknown[]) => {
      attempts++;
      return originalImpl(...args);
    };

    const docs: RerankDocument[] = [{ file: "a.md", text: "doc" }];
    try {
      let caught: unknown = null;
      try {
        await llm.rerank("q", docs);
      } catch (e) {
        caught = e;
      }

      // The reclaim path must have run all three attempts before giving up.
      expect(attempts).toBe(3);
      expect(caught).not.toBeNull();
      // The surfaced error must be one the reclaim recognized.  If this
      // assertion ever fails, it means the lib's real error string drifted
      // and the regex in `isInsufficientVramError` needs a new alternative
      // (add a positive case to llm-vram-error-pinning.test.ts first, then
      // update the regex).
      expect(isInsufficientVramError(caught)).toBe(true);
    } finally {
      fieldOwner.RERANK_CONTEXT_SIZE = originalSize;
      await llm.dispose();
    }
  }, /* timeout ms */ 120_000);

  test("non-VRAM rerank failure does not retry (model file missing)", async () => {
    // Build a LlamaCpp pointed at a model URI that cannot resolve, so the
    // first attempt throws a non-VRAM error.  Reclaim must NOT swallow it
    // into retries — it should surface immediately.
    const llm = new LlamaCpp({
      lowVram: true,
      rerankModel: "/nonexistent/path/to/no-such-model.gguf",
    });

    const target = llm as unknown as {
      rerankImpl: (...args: unknown[]) => Promise<unknown>;
    };
    const originalImpl = target.rerankImpl.bind(llm);
    let attempts = 0;
    target.rerankImpl = async (...args: unknown[]) => {
      attempts++;
      return originalImpl(...args);
    };

    const docs: RerankDocument[] = [{ file: "a.md", text: "doc" }];
    try {
      let caught: unknown = null;
      try {
        await llm.rerank("q", docs);
      } catch (e) {
        caught = e;
      }
      // A missing model file is not a VRAM problem.  Exactly one attempt,
      // and the error must NOT be classified as VRAM (the regex would
      // chew through retries if it did).
      expect(attempts).toBe(1);
      expect(caught).not.toBeNull();
      expect(isInsufficientVramError(caught)).toBe(false);
    } finally {
      await llm.dispose();
    }
  }, /* timeout ms */ 60_000);
});
