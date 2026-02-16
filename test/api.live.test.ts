import { describe, expect, test } from "vitest";
import { ApiLLM } from "../src/api.js";

const compatibilityProviders = [
  {
    name: "OpenAI",
    key: process.env.OPENAI_API_KEY || "",
    baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    embedModel: process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small",
    chatModel: process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini",
  },
  {
    name: "Cohere",
    key: process.env.COHERE_API_KEY || "",
    baseUrl: process.env.COHERE_COMPAT_BASE_URL || "https://api.cohere.ai/compatibility/v1",
    embedModel: process.env.COHERE_EMBED_MODEL || "embed-v4.0",
    chatModel: process.env.COHERE_CHAT_MODEL || "command-a-03-2025",
  },
];

describe("ApiLLM Embeddings (live)", () => {
  for (const provider of compatibilityProviders) {
    test.skipIf(!provider.key)(`${provider.name} /v1/embeddings returns a non-empty vector`, async () => {
      const llm = new ApiLLM({
        embedBaseUrl: provider.baseUrl,
        embedApiKey: provider.key,
        embedModel: provider.embedModel,
      });

      const result = await llm.embed(`QMD embedding live test (${provider.name})`);
      expect(result).not.toBeNull();
      expect(Array.isArray(result?.embedding)).toBe(true);
      expect(result!.embedding.length).toBeGreaterThan(10);
      expect(Number.isFinite(result!.embedding[0])).toBe(true);
    }, 30000);
  }
});

describe("ApiLLM Query Expansion (live)", () => {
  for (const provider of compatibilityProviders) {
    test.skipIf(!provider.key)(`${provider.name} chat completions expands query with non-strict output mode`, async () => {
      const llm = new ApiLLM({
        chatBaseUrl: provider.baseUrl,
        chatApiKey: provider.key,
        chatModel: provider.chatModel,
        strictJsonOutput: false,
      });

      const result = await llm.expandQuery("how to authenticate API requests");
      expect(result.length).toBeGreaterThanOrEqual(1);
      for (const item of result) {
        expect(["lex", "vec", "hyde"]).toContain(item.type);
        expect(item.text.length).toBeGreaterThan(0);
      }
    }, 30000);

    test.skipIf(!provider.key)(`${provider.name} chat completions expands query with strict JSON output mode`, async () => {
      const llm = new ApiLLM({
        chatBaseUrl: provider.baseUrl,
        chatApiKey: provider.key,
        chatModel: provider.chatModel,
        strictJsonOutput: true,
      });

      const result = await llm.expandQuery("how to authenticate API requests", { includeLexical: false });
      expect(result.length).toBeGreaterThanOrEqual(1);
      for (const item of result) {
        expect(["vec", "hyde"]).toContain(item.type);
        expect(item.text.length).toBeGreaterThan(0);
      }
    }, 30000);
  }
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
