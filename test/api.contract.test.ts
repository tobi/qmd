import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiLLM } from "../src/api.js";

describe("ApiLLM (contract)", () => {
  const fetchMock = vi.fn();
  const originalFetch = globalThis.fetch;
  const originalQmdEmbedApiKey = process.env.QMD_EMBED_API_KEY;
  const originalQmdChatApiKey = process.env.QMD_CHAT_API_KEY;
  const originalQmdChatStrictJsonOutput = process.env.QMD_CHAT_STRICT_JSON_OUTPUT;
  const originalQmdChatModel = process.env.QMD_CHAT_MODEL;
  const originalQmdRerankApiKey = process.env.QMD_RERANK_API_KEY;

  beforeEach(() => {
    fetchMock.mockReset();
    (globalThis as { fetch: typeof fetch }).fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    (globalThis as { fetch: typeof fetch }).fetch = originalFetch;
    process.env.QMD_EMBED_API_KEY = originalQmdEmbedApiKey;
    process.env.QMD_CHAT_API_KEY = originalQmdChatApiKey;
    process.env.QMD_CHAT_STRICT_JSON_OUTPUT = originalQmdChatStrictJsonOutput;
    process.env.QMD_CHAT_MODEL = originalQmdChatModel;
    process.env.QMD_RERANK_API_KEY = originalQmdRerankApiKey;
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
    process.env.QMD_EMBED_API_KEY = "";
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

  test("generate fails explicitly for API backend", async () => {
    const llm = new ApiLLM({});

    await expect(
      llm.generate("hello")
    ).rejects.toThrow("not implemented for API backend");
    expect(fetchMock).not.toHaveBeenCalled();
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

  test("rerank sends Voyage-compatible top_k and accepts data response shape", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [
            { index: 0, relevance_score: 0.12 },
            { index: 1, relevance_score: 0.95 },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      rerankBaseUrl: "https://api.voyageai.com/v1",
      rerankApiKey: "voyage-key",
      rerankModel: "rerank-2.5-lite",
    });

    const result = await llm.rerank(
      "capital of france",
      [
        { file: "a.md", text: "Berlin is the capital of Germany." },
        { file: "b.md", text: "Paris is the capital of France." },
      ]
    );

    expect(result.model).toBe("rerank-2.5-lite");
    expect(result.results).toEqual([
      { file: "b.md", score: 0.95, index: 1 },
      { file: "a.md", score: 0.12, index: 0 },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://api.voyageai.com/v1/rerank");
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "rerank-2.5-lite",
      query: "capital of france",
      documents: [
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
      ],
      top_k: 2,
    });
  });

  test("rerank throws and avoids fetch when rerank API key is missing", async () => {
    process.env.QMD_EMBED_API_KEY = "";
    process.env.QMD_RERANK_API_KEY = "";

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

  test("expandQuery accepts line format when strict JSON is disabled (default)", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{
            message: {
              content: "lex: api auth docs\nvec: api authentication guide\nhyde: A guide to API authentication setup",
            },
          }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      chatBaseUrl: "https://chat.example.test/v1",
      chatApiKey: "chat-key",
      chatModel: "gpt-4o-mini",
    });

    const result = await llm.expandQuery("api auth docs");
    expect(result).toEqual([
      { type: "lex", text: "api auth docs" },
      { type: "vec", text: "api authentication guide" },
      { type: "hyde", text: "A guide to API authentication setup" },
    ]);

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://chat.example.test/v1/chat/completions");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({
      "Content-Type": "application/json",
      "Authorization": "Bearer chat-key",
    });
  });

  test("expandQuery uses strict JSON mode from env and parses JSON output", async () => {
    process.env.QMD_CHAT_STRICT_JSON_OUTPUT = "true";
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{
            message: {
              content: JSON.stringify([
                { type: "lex", text: "api auth docs" },
                { type: "vec", text: "api authentication guide" },
              ]),
            },
          }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      chatBaseUrl: "https://chat.example.test/v1",
      chatApiKey: "chat-key",
      chatModel: "gpt-4o-mini",
    });

    const result = await llm.expandQuery("api auth docs", { includeLexical: false });
    expect(result).toEqual([
      { type: "vec", text: "api authentication guide" },
    ]);
  });

  test("expandQuery rejects line output when strict JSON mode is enabled", async () => {
    process.env.QMD_CHAT_STRICT_JSON_OUTPUT = "true";
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{
            message: {
              content: "vec: api authentication guide",
            },
          }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      chatBaseUrl: "https://chat.example.test/v1",
      chatApiKey: "chat-key",
      chatModel: "gpt-4o-mini",
    });

    await expect(
      llm.expandQuery("api auth docs")
    ).rejects.toThrow("strict JSON output is enabled");
  });
});
