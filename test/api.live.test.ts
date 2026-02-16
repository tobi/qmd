import { describe, expect, test } from "vitest";
import { ApiLLM } from "../src/api.js";

describe.skipIf(!process.env.OPENAI_API_KEY)("ApiLLM Embeddings (live)", () => {
  test("OpenAI /v1/embeddings returns a non-empty vector", async () => {
    const llm = new ApiLLM({
      embedBaseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      embedApiKey: process.env.OPENAI_API_KEY,
      embedModel: process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small",
    });

    const result = await llm.embed("QMD embedding live test");
    expect(result).not.toBeNull();
    expect(Array.isArray(result?.embedding)).toBe(true);
    expect(result!.embedding.length).toBeGreaterThan(10);
    expect(Number.isFinite(result!.embedding[0])).toBe(true);
  }, 30000);
});

describe.skipIf(!process.env.COHERE_API_KEY)("ApiLLM Rerank (live)", () => {
  test("Cohere /v1/rerank returns ranked documents with finite scores", async () => {
    const llm = new ApiLLM({
      rerankBaseUrl: process.env.COHERE_BASE_URL || "https://api.cohere.com/v1",
      rerankApiKey: process.env.COHERE_API_KEY,
      rerankModel: process.env.COHERE_RERANK_MODEL || "rerank-v3.5",
    });

    const docs = [
      { file: "france.md", text: "Paris is the capital city of France." },
      { file: "pets.md", text: "Cats and dogs are common household pets." },
      { file: "germany.md", text: "Berlin is the capital city of Germany." },
    ];

    const result = await llm.rerank("What is the capital of France?", docs);
    expect(result.results.length).toBe(3);
    expect(result.results[0]!.file).toBe("france.md");
    expect(Number.isFinite(result.results[0]!.score)).toBe(true);
    expect(result.results[0]!.score).toBeGreaterThanOrEqual(result.results[1]!.score);
  }, 30000);
});
