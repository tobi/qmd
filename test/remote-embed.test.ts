import { afterEach, describe, expect, test, vi } from "vitest";
import {
  RemoteEmbeddingProtocolError,
  RemoteEmbeddingTransportError,
  OpenAIEmbeddingClient,
  resolveEmbeddingConfig,
} from "../src/remote-embed.js";

const NATIVE_DIM = 2560;
const OUTPUT_DIM = 1024;
const vector = (seed = 1) => Array.from({ length: NATIVE_DIM }, (_, i) => seed + i / 10000);
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const directJsonResponse = (body: unknown) => ({
  ok: true,
  status: 200,
  json: async () => body,
}) as Response;
const indexed = (items: Array<{ index: number; embedding: unknown }>, model = "Qwen3-Embedding-4B-Q8_0.gguf") => ({
  object: "list", model, data: items.map(item => ({ object: "embedding", ...item })),
});
const client = (overrides: Partial<ConstructorParameters<typeof OpenAIEmbeddingClient>[0]> = {}) => new OpenAIEmbeddingClient({
  provider: "openai",
  endpoint: "http://127.0.0.1:8086/v1/embeddings",
  model: "Qwen3-Embedding-4B-Q8_0.gguf",
  nativeDimensions: NATIVE_DIM,
  dimensions: OUTPUT_DIM,
  reduction: "mrl-prefix",
  normalization: "l2",
  formatVersion: "qwen3-v1",
  ...overrides,
});

afterEach(() => vi.restoreAllMocks());

describe("OpenAIEmbeddingClient", () => {
  test("posts untruncated input and restores reordered data by index", async () => {
    const a = vector(1), b = vector(2);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([
      { index: 1, embedding: b }, { index: 0, embedding: a },
    ])));
    const embeddings = await client().embedBatch(["a".repeat(13000), "b"]);
    expect(embeddings).toHaveLength(2);
    expect(embeddings[0]).toHaveLength(OUTPUT_DIM);
    expect(embeddings[1]).toHaveLength(OUTPUT_DIM);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://127.0.0.1:8086/v1/embeddings");
    expect(JSON.parse(String(init?.body))).toEqual({ model: "Qwen3-Embedding-4B-Q8_0.gguf", input: ["a".repeat(13000), "b"] });
  });

  test("selects the exact leading 1024 components and L2-normalizes only that prefix", async () => {
    const native = Array(NATIVE_DIM).fill(0);
    native[0] = 3;
    native[1] = 4;
    native[OUTPUT_DIM] = 12;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([{ index: 0, embedding: native }])));

    const embedding = await client().embed("query or document");

    expect(embedding).toHaveLength(OUTPUT_DIM);
    expect(embedding[0]).toBeCloseTo(0.6, 7);
    expect(embedding[1]).toBeCloseTo(0.8, 7);
    expect(embedding.slice(2)).toEqual(Array(OUTPUT_DIM - 2).fill(0));
  });

  test("validates the full native vector before truncating the tail", async () => {
    const native = vector();
    native[OUTPUT_DIM + 100] = Number.POSITIVE_INFINITY;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(directJsonResponse(indexed([{ index: 0, embedding: native }])));

    await expect(client().embed("a")).rejects.toThrow("finite Float32");
  });

  test("rejects a zero prefix even when the discarded tail is nonzero", async () => {
    const native = Array(NATIVE_DIM).fill(0);
    native[OUTPUT_DIM] = 1;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([{ index: 0, embedding: native }])));

    await expect(client().embed("a")).rejects.toThrow("zero or invalid norm");
  });

  test("rejects an underflowed prefix norm and never returns invalid normalized Float32 output", async () => {
    const native = Array(NATIVE_DIM).fill(0);
    native[0] = Number.MIN_VALUE;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([{ index: 0, embedding: native }])));

    await expect(client().embed("a")).rejects.toThrow("zero or invalid norm");
  });

  test.each([
    ["wrong cardinality", indexed([{ index: 0, embedding: vector() }])],
    ["missing index", indexed([{ index: 0, embedding: vector() }, { index: 2, embedding: vector(2) }])],
    ["duplicate index", indexed([{ index: 0, embedding: vector() }, { index: 0, embedding: vector(2) }])],
    ["negative index", indexed([{ index: -1, embedding: vector() }, { index: 1, embedding: vector(2) }])],
    ["non-integer index", indexed([{ index: 0.5, embedding: vector() }, { index: 1, embedding: vector(2) }])],
    ["out of range index", indexed([{ index: 0, embedding: vector() }, { index: 2, embedding: vector(2) }])],
    ["missing data", { model: "Qwen3-Embedding-4B-Q8_0.gguf" }],
    ["empty vector", indexed([{ index: 0, embedding: [] }, { index: 1, embedding: vector() }])],
    ["malformed vector", indexed([{ index: 0, embedding: "bad" }, { index: 1, embedding: vector() }])],
    ["non numeric vector", indexed([{ index: 0, embedding: [...vector().slice(0, -1), "x"] }, { index: 1, embedding: vector() }])],
    ["non-finite vector", indexed([{ index: 0, embedding: [...vector().slice(0, -1), Infinity] }, { index: 1, embedding: vector() }])],
    ["Float32-overflow vector", indexed([{ index: 0, embedding: [...vector().slice(0, -1), 1e300] }, { index: 1, embedding: vector() }])],
    ["wrong dimension", indexed([{ index: 0, embedding: [1, 2] }, { index: 1, embedding: [3, 4] }])],
    ["inconsistent dimension", indexed([{ index: 0, embedding: vector() }, { index: 1, embedding: vector().slice(1) }])],
    ["missing response model", { data: [{ index: 0, embedding: vector() }, { index: 1, embedding: vector(2) }] }],
    ["wrong response model", indexed([{ index: 0, embedding: vector() }, { index: 1, embedding: vector(2) }], "other-model")],
  ])("rejects %s as a typed protocol error", async (_name, body) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(body));
    await expect(client().embedBatch(["a", "b"])).rejects.toBeInstanceOf(RemoteEmbeddingProtocolError);
  });

  test("accepts an explicitly configured response model alias", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([{ index: 0, embedding: vector() }], "server-alias")));
    expect(await client({ modelAliases: ["server-alias"] }).embedBatch(["a"])).toHaveLength(1);
  });

  test.each([
    ["HTTP failure", async () => response({ error: { message: "no" } }, 503)],
    ["invalid JSON", async () => new Response("not-json", { status: 200 })],
    ["network failure", async () => { throw new Error("connect secret-token"); }],
    ["timeout", async () => { throw new DOMException("timed out", "TimeoutError"); }],
    ["abort", async () => { throw new DOMException("aborted", "AbortError"); }],
  ])("turns %s into a typed safe error", async (_name, implementation) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(implementation as typeof fetch);
    const error = await client({ apiKey: "secret-token" }).embedBatch(["a"]).catch(e => e);
    expect(error).toBeInstanceOf(_name === "invalid JSON" ? RemoteEmbeddingProtocolError : RemoteEmbeddingTransportError);
    expect(String(error)).not.toContain("secret-token");
    expect((error as Error & { cause?: unknown }).cause).toBeUndefined();
  });

  test("sends an optional bearer key without exposing it in identity", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(response(indexed([{ index: 0, embedding: vector() }])));
    await client({ apiKey: "secret-token" }).embedBatch(["a"]);
    expect((fetchMock.mock.calls[0]![1]!.headers as Record<string, string>).Authorization).toBe("Bearer secret-token");
  });
});

describe("embedding configuration compatibility", () => {
  test("resolves explicit native and stored dimensions with required MRL semantics", () => {
    expect(resolveEmbeddingConfig({
      provider: "openai",
      endpoint: "http://host/v1/embeddings",
      model: "qwen",
      nativeDimensions: 2560,
      dimensions: 1024,
      reduction: "mrl-prefix",
      normalization: "l2",
      apiKey: "key",
    })).toMatchObject({
      kind: "remote",
      endpoint: "http://host/v1/embeddings",
      model: "qwen",
      nativeDimensions: 2560,
      dimensions: 1024,
      reduction: "mrl-prefix",
      normalization: "l2",
      apiKey: "key",
    });
  });
  test.each([
    ["non-string model", { model: 42 }],
    ["missing native dimensions", { nativeDimensions: undefined }],
    ["wrong native dimensions", { nativeDimensions: 1024 }],
    ["fractional native dimensions", { nativeDimensions: 2559.5 }],
    ["negative native dimensions", { nativeDimensions: -2560 }],
    ["missing output dimensions", { dimensions: undefined }],
    ["wrong output dimensions", { dimensions: 768 }],
    ["output dimensions equal native dimensions", { dimensions: 2560 }],
    ["output dimensions greater than native dimensions", { dimensions: 4096 }],
    ["fractional output dimensions", { dimensions: 1023.5 }],
    ["negative output dimensions", { dimensions: -1024 }],
    ["missing reduction", { reduction: undefined }],
    ["wrong reduction", { reduction: "mean-pool" }],
    ["missing normalization", { normalization: undefined }],
    ["wrong normalization", { normalization: "none" }],
    ["invalid normalization", { normalization: "cosine" }],
    ["empty formatVersion", { formatVersion: "" }],
    ["non-string formatVersion", { formatVersion: 7 }],
    ["zero timeoutMs", { timeoutMs: 0 }],
    ["fractional timeoutMs", { timeoutMs: 1.5 }],
    ["non-array modelAliases", { modelAliases: "alias" }],
    ["non-string model alias", { modelAliases: [7] }],
    ["empty model alias", { modelAliases: [""] }],
    ["non-string apiKey", { apiKey: 7 }],
    ["empty apiKey", { apiKey: "" }],
  ])("rejects %s at runtime", (_name, override) => {
    expect(() => resolveEmbeddingConfig({
      provider: "openai",
      endpoint: "http://127.0.0.1:8086/v1",
      model: "qwen",
      nativeDimensions: 2560,
      dimensions: 1024,
      reduction: "mrl-prefix",
      normalization: "l2",
      ...override,
    } as any)).toThrow(RemoteEmbeddingProtocolError);
  });

  test.each([
    ["not a URL", "not-a-url"],
    ["wrong scheme", "ftp://127.0.0.1/v1"],
    ["wrong path", "http://127.0.0.1:8086/api"],
    ["embedded credentials", "http://user:secret-token@127.0.0.1:8086/v1"],
    ["query credentials", "http://127.0.0.1:8086/v1?api_key=secret-token"],
  ])("rejects %s with a typed safe endpoint error", (_name, endpoint) => {
    let error: unknown;
    try {
      resolveEmbeddingConfig({
        provider: "openai",
        endpoint,
        model: "qwen",
        nativeDimensions: 2560,
        dimensions: 1024,
        reduction: "mrl-prefix",
        normalization: "l2",
      });
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(RemoteEmbeddingProtocolError);
    expect(String(error)).not.toContain("secret-token");
  });

  test("keeps every string model URI local, including HTTP GGUF URLs", () => {
    expect(resolveEmbeddingConfig("hf:org/repo/model.gguf")).toEqual({ kind: "local", model: "hf:org/repo/model.gguf" });
    expect(resolveEmbeddingConfig("https://models.example/qwen.gguf")).toEqual({
      kind: "local",
      model: "https://models.example/qwen.gguf",
    });
  });
});
