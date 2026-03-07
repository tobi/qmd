/**
 * Test for Issue #303: qmd embed crash on oversized chunks
 *
 * Verifies that the embedding process handles chunks exceeding the model's
 * 2048-token context window without crashing (SIGABRT).
 *
 * Run: npx vitest run --reporter=verbose test/oversized-chunk.test.ts
 */

import { describe, test, expect } from "vitest";
import {
  withLLMSession,
  type ILLMSession,
} from "../src/llm.js";
import { formatDocForEmbedding } from "../src/store.js";

// Skip if models are not downloaded
const modelsAvailable = process.env.QMD_SKIP_MODEL_TESTS !== "1";

describe.skipIf(!modelsAvailable)("Oversized chunk embedding (Issue #303)", () => {

  // First test absorbs model compilation + download overhead
  test("embed() truncates oversized text instead of crashing", async () => {
    await withLLMSession(async (session: ILLMSession) => {
      // ~2500 tokens (well over 2048 context window)
      const oversized = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(200);

      // Before fix: SIGABRT / "Failed to get embedding dimensions"
      // After fix: truncates and returns valid embedding
      const result = await session.embed(oversized);

      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBeGreaterThan(0);
    });
  }, 180_000);

  test("virtual probe returns valid embedding dimensions", async () => {
    await withLLMSession(async (session: ILLMSession) => {
      const result = await session.embed("dimension probe");

      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBeGreaterThan(0);
    });
  }, 30_000);

  test("embedBatch() handles mix of normal and oversized texts", async () => {
    await withLLMSession(async (session: ILLMSession) => {
      const normal = "This is a normal sized text for embedding.";
      const oversized = "function example() { return 'data'; }\n".repeat(500);

      const results = await session.embedBatch([normal, oversized, normal]);

      expect(results).toHaveLength(3);
      for (const r of results) {
        expect(r).not.toBeNull();
        expect(r!.embedding.length).toBeGreaterThan(0);
      }

      // All embeddings should have the same dimensions
      const dim = results[0]!.embedding.length;
      expect(results[1]!.embedding.length).toBe(dim);
      expect(results[2]!.embedding.length).toBe(dim);
    });
  }, 30_000);

  test("formatted oversized chunk with title does not crash", async () => {
    await withLLMSession(async (session: ILLMSession) => {
      const title = "Large Code File";
      const content = "const x = " + "'a'.repeat(100);\n".repeat(500);
      const formatted = formatDocForEmbedding(content, title);

      const result = await session.embed(formatted);

      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBeGreaterThan(0);
    });
  }, 30_000);

});
