import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiLLM } from "../src/api.js";

describe("ApiLLM (contract)", () => {
  const fetchMock = vi.fn();
  const originalFetch = globalThis.fetch;
  const originalQmdApiKey = process.env.QMD_API_KEY;
  const originalOpenAiApiKey = process.env.OPENAI_API_KEY;
  const originalQmdApiRerankKey = process.env.QMD_API_RERANK_KEY;
  const originalCohereApiKey = process.env.COHERE_API_KEY;

  beforeEach(() => {
    fetchMock.mockReset();
    (globalThis as { fetch: typeof fetch }).fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    (globalThis as { fetch: typeof fetch }).fetch = originalFetch;
    process.env.QMD_API_KEY = originalQmdApiKey;
    process.env.OPENAI_API_KEY = originalOpenAiApiKey;
    process.env.QMD_API_RERANK_KEY = originalQmdApiRerankKey;
    process.env.COHERE_API_KEY = originalCohereApiKey;
  });

  test("embed sends OpenAI-compatible /embeddings request, normalizes model, and parses response", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2, 0.3] }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      embedBaseUrl: "https://example.test/v1",
      embedApiKey: "test-key",
      embedModel: "test-embed-model",
    });

    const result = await llm.embed("hello", { model: "embeddinggemma" });

    expect(result).not.toBeNull();
    expect(result?.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(result?.model).toBe("test-embed-model");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://example.test/v1/embeddings");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({
      "Content-Type": "application/json",
      "Authorization": "Bearer test-key",
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "test-embed-model",
      input: ["hello"],
    });
  });

  test("embedBatch returns one result per input and null for missing vectors", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [
            { embedding: [1, 2] },
            {},
            { embedding: [3, 4] },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      embedBaseUrl: "https://example.test/v1",
      embedApiKey: "test-key",
      embedModel: "test-embed-model",
    });

    const results = await llm.embedBatch(["a", "b", "c"]);
    expect(results).toHaveLength(3);
    expect(results[0]?.embedding).toEqual([1, 2]);
    expect(results[1]).toBeNull();
    expect(results[2]?.embedding).toEqual([3, 4]);
  });

  test("embed returns null and avoids fetch when API key is missing", async () => {
    process.env.QMD_API_KEY = "";
    process.env.OPENAI_API_KEY = "";
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const llm = new ApiLLM({
      embedBaseUrl: "https://example.test/v1",
      embedApiKey: "",
      embedModel: "test-embed-model",
    });

    const result = await llm.embed("hello");
    expect(result).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    consoleErrorSpy.mockRestore();
  });

  test("rerank sends Cohere-compatible /rerank request and maps response by index", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          results: [
            { index: 1, relevance_score: 0.91 },
            { index: 0, relevance_score: 0.24 },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      embedBaseUrl: "https://example.test/v1",
      embedApiKey: "embed-key",
      rerankBaseUrl: "https://rerank.test/v1",
      rerankApiKey: "rerank-key",
      rerankModel: "rerank-v3.5",
    });

    const result = await llm.rerank(
      "capital of france",
      [
        { file: "a.md", text: "Berlin is the capital of Germany." },
        { file: "b.md", text: "Paris is the capital of France." },
      ],
      { model: "ExpedientFalcon/qwen3-reranker:0.6b-q8_0" }
    );

    expect(result.model).toBe("rerank-v3.5");
    expect(result.results).toEqual([
      { file: "b.md", score: 0.91, index: 1 },
      { file: "a.md", score: 0.24, index: 0 },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://rerank.test/v1/rerank");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({
      "Content-Type": "application/json",
      "Authorization": "Bearer rerank-key",
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "rerank-v3.5",
      query: "capital of france",
      documents: [
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
      ],
      top_n: 2,
    });
  });

  test("rerank throws and avoids fetch when rerank API key is missing", async () => {
    process.env.QMD_API_KEY = "";
    process.env.OPENAI_API_KEY = "";
    process.env.QMD_API_RERANK_KEY = "";
    process.env.COHERE_API_KEY = "";

    const llm = new ApiLLM({
      embedBaseUrl: "https://example.test/v1",
      embedApiKey: "",
      rerankApiKey: "",
      rerankModel: "rerank-v3.5",
    });

    await expect(
      llm.rerank("q", [{ file: "doc.md", text: "t" }])
    ).rejects.toThrow("missing API key");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
