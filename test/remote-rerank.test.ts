import { afterEach, describe, expect, test, vi } from "vitest";
import {
  OpenAIRerankClient,
  RemoteRerankProtocolError,
  RemoteRerankTransportError,
  getRerankCacheIdentity,
  resolveRerankConfig,
} from "../src/remote-rerank.js";

const config = {
  provider: "openai" as const,
  endpoint: "http://127.0.0.1:8088/v1",
  model: "Qwen3-Reranker-0.6B-Q4_K_M.gguf",
};

afterEach(() => vi.restoreAllMocks());

describe("OpenAIRerankClient", () => {
  test("maps ranked out-of-order indexes and preserves normalized relevance_score", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      model: "Qwen3-Reranker-0.6B-Q4_K_M.gguf",
      results: [
        { index: 2, relevance_score: 0.97 },
        { index: 0, relevance_score: 0.61 },
        { index: 1, relevance_score: 0.02 },
      ],
    }), { status: 200, headers: { "Content-Type": "application/json" } }));

    const documents = [
      { file: "a.md", text: "alpha" },
      { file: "b.md", text: "beta" },
      { file: "c.md", text: "gamma" },
    ];
    const result = await new OpenAIRerankClient(config).rerank("query", documents);

    expect(result).toEqual({
      model: config.model,
      results: [
        { file: "c.md", index: 2, score: 0.97 },
        { file: "a.md", index: 0, score: 0.61 },
        { file: "b.md", index: 1, score: 0.02 },
      ],
    });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://127.0.0.1:8088/v1/rerank");
    expect(JSON.parse(String(init?.body))).toEqual({
      model: config.model,
      query: "query",
      documents: ["alpha", "beta", "gamma"],
    });
  });

  test.each([
    ["missing results", { model: config.model }],
    ["wrong cardinality", { model: config.model, results: [{ index: 0, relevance_score: 0.5 }] }],
    ["duplicate index", { model: config.model, results: [{ index: 0, relevance_score: 0.5 }, { index: 0, relevance_score: 0.4 }] }],
    ["negative index", { model: config.model, results: [{ index: -1, relevance_score: 0.5 }, { index: 1, relevance_score: 0.4 }] }],
    ["fractional index", { model: config.model, results: [{ index: 0.5, relevance_score: 0.5 }, { index: 1, relevance_score: 0.4 }] }],
    ["out-of-range index", { model: config.model, results: [{ index: 0, relevance_score: 0.5 }, { index: 2, relevance_score: 0.4 }] }],
    ["NaN score", { model: config.model, results: [{ index: 0, relevance_score: Number.NaN }, { index: 1, relevance_score: 0.4 }] }],
    ["infinite score", { model: config.model, results: [{ index: 0, relevance_score: Infinity }, { index: 1, relevance_score: 0.4 }] }],
    ["score below zero", { model: config.model, results: [{ index: 0, relevance_score: -0.1 }, { index: 1, relevance_score: 0.4 }] }],
    ["score above one", { model: config.model, results: [{ index: 0, relevance_score: 1.1 }, { index: 1, relevance_score: 0.4 }] }],
    ["missing response model", { results: [{ index: 0, relevance_score: 0.5 }, { index: 1, relevance_score: 0.4 }] }],
    ["wrong response model", { model: "other", results: [{ index: 0, relevance_score: 0.5 }, { index: 1, relevance_score: 0.4 }] }],
  ])("rejects %s", async (_name, body) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }));
    await expect(new OpenAIRerankClient(config).rerank("q", [
      { file: "a.md", text: "a" },
      { file: "b.md", text: "b" },
    ])).rejects.toBeInstanceOf(RemoteRerankProtocolError);
  });

  test.each([
    ["malformed URL", { endpoint: "not a URL" }],
    ["unsupported scheme", { endpoint: "file:///tmp/v1" }],
    ["wrong path", { endpoint: "http://127.0.0.1:8088/api" }],
    ["URL credentials", { endpoint: "http://user:pass@127.0.0.1:8088/v1" }],
    ["URL query", { endpoint: "http://127.0.0.1:8088/v1?token=secret" }],
    ["URL fragment", { endpoint: "http://127.0.0.1:8088/v1#secret" }],
    ["non-string model", { model: 42 }],
    ["empty model", { model: "" }],
    ["whitespace model", { model: "   " }],
    ["padded model", { model: " model " }],
    ["non-string key", { apiKey: 7 }],
    ["empty key", { apiKey: "" }],
    ["whitespace key", { apiKey: "   " }],
    ["padded key", { apiKey: " key " }],
    ["zero timeout", { timeoutMs: 0 }],
    ["fractional timeout", { timeoutMs: 1.5 }],
    ["non-array aliases", { modelAliases: "alias" }],
    ["invalid alias", { modelAliases: [""] }],
    ["whitespace alias", { modelAliases: ["   "] }],
    ["padded alias", { modelAliases: [" alias "] }],
    ["duplicate aliases", { modelAliases: ["alias", "alias"] }],
    ["trim-equivalent aliases", { modelAliases: ["alias", " alias"] }],
    ["primary model as alias", { modelAliases: [config.model] }],
  ])("rejects invalid config: %s", (_name, override) => {
    expect(() => resolveRerankConfig({ ...config, ...override } as never)).toThrow(RemoteRerankProtocolError);
  });

  test("plain string reranker configurations remain local", () => {
    expect(resolveRerankConfig("http://models.example/reranker.gguf")).toEqual({
      kind: "local", model: "http://models.example/reranker.gguf",
    });
  });

  test("cache identity includes endpoint semantics but excludes API keys", () => {
    const a = resolveRerankConfig({ ...config, endpoint: "http://host-a/v1", apiKey: "key-a" });
    const b = resolveRerankConfig({ ...config, endpoint: "http://host-a/v1", apiKey: "key-b" });
    const c = resolveRerankConfig({ ...config, endpoint: "http://host-b/v1", apiKey: "key-a" });
    expect(getRerankCacheIdentity(a)).toBe(getRerankCacheIdentity(b));
    expect(getRerankCacheIdentity(c)).not.toBe(getRerankCacheIdentity(a));
    expect(getRerankCacheIdentity(a)).not.toContain("key-a");
  });

  test("exposes the only supported endpoint failure policy as fail-closed", () => {
    expect(new OpenAIRerankClient(config).failurePolicy).toBe("fail-closed");
    expect(() => resolveRerankConfig({ ...config, failurePolicy: "fallback-local" } as never))
      .toThrow(RemoteRerankProtocolError);
  });

  test.each([
    ["HTTP error", async () => new Response("secret response", { status: 503 })],
    ["network error", async () => { throw new Error("connect secret-token"); }],
  ])("fails closed with safe typed transport error for %s", async (_name, implementation) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(implementation as typeof fetch);
    const error = await new OpenAIRerankClient({ ...config, apiKey: "secret-token" })
      .rerank("q", [{ file: "a.md", text: "a" }]).catch((value: unknown) => value);
    expect(error).toBeInstanceOf(RemoteRerankTransportError);
    expect(String(error)).not.toContain("secret-token");
    expect((error as Error & { cause?: unknown }).cause).toBeUndefined();
  });
});
