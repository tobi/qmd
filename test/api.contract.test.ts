import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiLLM } from "../src/api.js";

describe("ApiLLM Embeddings (contract)", () => {
  const fetchMock = vi.fn();
  const originalFetch = globalThis.fetch;
  const originalQmdApiKey = process.env.QMD_API_KEY;
  const originalOpenAiApiKey = process.env.OPENAI_API_KEY;

  beforeEach(() => {
    fetchMock.mockReset();
    (globalThis as { fetch: typeof fetch }).fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    (globalThis as { fetch: typeof fetch }).fetch = originalFetch;
    process.env.QMD_API_KEY = originalQmdApiKey;
    process.env.OPENAI_API_KEY = originalOpenAiApiKey;
  });

  test("embed sends OpenAI-compatible /embeddings request and parses response", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2, 0.3] }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const llm = new ApiLLM({
      baseUrl: "https://example.test/v1",
      apiKey: "test-key",
      embedModel: "test-embed-model",
    });

    const result = await llm.embed("hello");

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
      baseUrl: "https://example.test/v1",
      apiKey: "test-key",
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
      baseUrl: "https://example.test/v1",
      apiKey: "",
      embedModel: "test-embed-model",
    });

    const result = await llm.embed("hello");
    expect(result).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    consoleErrorSpy.mockRestore();
  });
});
