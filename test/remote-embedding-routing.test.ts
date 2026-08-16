import { afterEach, describe, expect, test, vi } from "vitest";
import { LlamaCpp, setDefaultLlamaCpp, setNodeLlamaCppModuleForTest } from "../src/llm.js";
import { chunkDocumentByTokens, getEmbeddingFingerprint } from "../src/store.js";

const remote = (endpoint = "http://127.0.0.1:8086/v1/embeddings", apiKey = "secret") => ({
  provider: "openai" as const,
  endpoint,
  model: "Qwen3-Embedding-4B-Q8_0.gguf",
  nativeDimensions: 2560,
  dimensions: 1024,
  apiKey,
  reduction: "mrl-prefix" as const,
  normalization: "l2" as const,
  formatVersion: "qwen3-query-document-v1",
});
const vector = () => Array.from({ length: 2560 }, (_, i) => i / 2560);
const ok = (count: number) => new Response(JSON.stringify({
  model: "Qwen3-Embedding-4B-Q8_0.gguf",
  data: Array.from({ length: count }, (_, index) => ({ index, embedding: vector() })),
}));

afterEach(() => {
  vi.restoreAllMocks();
  setDefaultLlamaCpp(null);
  setNodeLlamaCppModuleForTest(null);
});

describe("remote embedding routing", () => {
  test("embeds remotely without resolving, initializing, or downloading local embedding GGUF", async () => {
    const getLlama = vi.fn();
    const resolveModelFile = vi.fn();
    setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: vi.fn() as any, LlamaLogLevel: { error: 0 } });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok(1));
    const llm = new LlamaCpp({ embedModel: remote() });

    const result = await llm.embed("untruncated remote input");

    expect(result?.embedding).toHaveLength(1024);
    expect(result?.model).toBe("Qwen3-Embedding-4B-Q8_0.gguf");
    expect(getLlama).not.toHaveBeenCalled();
    expect(resolveModelFile).not.toHaveBeenCalled();
  });

  test("applies the same 1024-dimensional transformation to single, query, and batch paths", async () => {
    const native = Array(2560).fill(0);
    native[0] = 3;
    native[1] = 4;
    native[1024] = 12;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      const count = (JSON.parse(String(init?.body)).input as string[]).length;
      return new Response(JSON.stringify({
        model: "Qwen3-Embedding-4B-Q8_0.gguf",
        data: Array.from({ length: count }, (_, index) => ({ index, embedding: native })),
      }));
    });
    const llm = new LlamaCpp({ embedModel: remote() });

    const single = await llm.embed("single");
    const query = await llm.embed("query", { isQuery: true });
    const batch = await llm.embedBatch(["document one", "document two"]);

    for (const result of [single, query, ...batch]) {
      expect(result?.embedding).toHaveLength(1024);
      expect(result?.embedding[0]).toBeCloseTo(0.6, 7);
      expect(result?.embedding[1]).toBeCloseTo(0.8, 7);
      expect(result?.embedding.slice(2)).toEqual(Array(1022).fill(0));
    }
  });

  test("fails closed with the typed remote error and never falls back locally", async () => {
    const getLlama = vi.fn();
    const resolveModelFile = vi.fn();
    setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: vi.fn() as any, LlamaLogLevel: { error: 0 } });
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    const llm = new LlamaCpp({ embedModel: remote() });

    await expect(llm.embedBatch(["a", "b"])).rejects.toMatchObject({ name: "RemoteEmbeddingTransportError" });
    expect(getLlama).not.toHaveBeenCalled();
    expect(resolveModelFile).not.toHaveBeenCalled();
  });

  test("rejects a conflicting explicit model before calling the remote endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(ok(1));
    const llm = new LlamaCpp({ embedModel: remote() });

    await expect(llm.embed("text", { model: "other-model" })).rejects.toThrow("does not match");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("default remote token chunking never initializes a local embedding context", async () => {
    const getLlama = vi.fn();
    const resolveModelFile = vi.fn();
    setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: vi.fn() as any, LlamaLogLevel: { error: 0 } });
    setDefaultLlamaCpp(new LlamaCpp({ embedModel: remote() }));

    const chunks = await chunkDocumentByTokens("remote chunking must not load a local model");

    expect(chunks).toHaveLength(1);
    expect(getLlama).not.toHaveBeenCalled();
    expect(resolveModelFile).not.toHaveBeenCalled();
  });
});

describe("remote embedding fingerprint", () => {
  test("preserves the upstream local fingerprint for backward compatibility", () => {
    expect(getEmbeddingFingerprint()).toBe("c37385");
  });

  test("includes semantic fields but excludes endpoint and secrets", () => {
    const a = getEmbeddingFingerprint(remote("http://a/v1/embeddings", "key-a"));
    const b = getEmbeddingFingerprint(remote("http://b/v1/embeddings", "key-b"));
    expect(a).toBe(b);
    expect(a).toBe("6618ed");
    expect(getEmbeddingFingerprint({ ...remote(), nativeDimensions: 2048 })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), dimensions: 768 })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), model: "other" })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), reduction: "other" })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), normalization: "none" })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), formatVersion: "v2" })).not.toBe(a);
    expect(getEmbeddingFingerprint({ ...remote(), provider: "openai-compatible" as any })).not.toBe(a);
  });
});
