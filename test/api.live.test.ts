import { describe, expect, test } from "vitest";
import { ApiLLM } from "../src/api.js";

describe.skipIf(!process.env.OPENAI_API_KEY)("ApiLLM Embeddings (live)", () => {
  test("OpenAI /v1/embeddings returns a non-empty vector", async () => {
    const llm = new ApiLLM({
      baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      apiKey: process.env.OPENAI_API_KEY,
      embedModel: process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small",
    });

    const result = await llm.embed("QMD embedding live test");
    expect(result).not.toBeNull();
    expect(Array.isArray(result?.embedding)).toBe(true);
    expect(result!.embedding.length).toBeGreaterThan(10);
    expect(Number.isFinite(result!.embedding[0])).toBe(true);
  }, 30000);
});

