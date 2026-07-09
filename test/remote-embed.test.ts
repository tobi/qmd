/**
 * remote-embed.test.ts - Unit tests for the remote OpenAI-compatible embedding backend
 *
 * No GPU, no real network: `fetch` is mocked with vi.spyOn.
 */

import { describe, test, expect, afterEach, vi } from "vitest";
import {
  isRemoteEmbedModel,
  parseRemoteEmbedUri,
  RemoteEmbedder,
  RemoteEmbedAllFailedError,
  type RemoteEmbedEndpoint,
} from "../src/remote-embed.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function embeddingsBody(vectors: number[][]) {
  return { data: vectors.map((embedding) => ({ embedding })) };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("isRemoteEmbedModel", () => {
  test("true for http/https URIs", () => {
    expect(isRemoteEmbedModel("http://localhost:1234/v1#kure-v1")).toBe(true);
    expect(isRemoteEmbedModel("https://api.openai.com/v1#text-embedding-3-small")).toBe(true);
  });

  test("false for hf: URIs", () => {
    expect(isRemoteEmbedModel("hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf")).toBe(false);
  });

  test("false for a bare local path", () => {
    expect(isRemoteEmbedModel("/path/to/model.gguf")).toBe(false);
  });

  test("false for a plain model name", () => {
    expect(isRemoteEmbedModel("text-embedding-kure-v1")).toBe(false);
  });
});

describe("parseRemoteEmbedUri", () => {
  test("parses http URI with fragment", () => {
    const result = parseRemoteEmbedUri("http://localhost:1234/v1#kure-v1");
    expect(result).toEqual({
      raw: "http://localhost:1234/v1#kure-v1",
      apiBase: "http://localhost:1234/v1",
      modelId: "kure-v1",
    });
  });

  test("parses https URI with fragment", () => {
    const result = parseRemoteEmbedUri("https://api.openai.com/v1#text-embedding-3-small");
    expect(result.apiBase).toBe("https://api.openai.com/v1");
    expect(result.modelId).toBe("text-embedding-3-small");
  });

  test("preserves trailing slash before fragment", () => {
    const result = parseRemoteEmbedUri("http://localhost:1234/v1/#kure-v1");
    expect(result.apiBase).toBe("http://localhost:1234/v1/");
    expect(result.modelId).toBe("kure-v1");
  });

  test("percent-decodes the model id fragment", () => {
    const result = parseRemoteEmbedUri("http://localhost:1234/v1#text-embedding%2Fkure");
    expect(result.modelId).toBe("text-embedding/kure");
  });

  test("throws when there is no fragment", () => {
    expect(() => parseRemoteEmbedUri("http://localhost:1234/v1")).toThrow(/#model-id/);
  });

  test("throws when the fragment is empty", () => {
    expect(() => parseRemoteEmbedUri("http://localhost:1234/v1#")).toThrow(/#model-id/);
  });
});

describe("RemoteEmbedder header construction", () => {
  test("includes Bearer header when apiKey is set", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(embeddingsBody([[1, 0, 0]])));
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint], { apiKey: "secret-key" });

    await embedder.embedBatch(["hello"]);

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("http://localhost:1234/v1/embeddings");
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer secret-key");
    expect(JSON.parse(init?.body as string)).toEqual({ model: "m", input: ["hello"] });
  });

  test("omits Authorization header when no apiKey is set", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(embeddingsBody([[1, 0, 0]])));
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint]);

    await embedder.embedBatch(["hello"]);

    const [, init] = fetchSpy.mock.calls[0]!;
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });
});

describe("RemoteEmbedder batch order", () => {
  test("preserves input order in output", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(embeddingsBody([[1, 0], [0, 1], [0.5, 0.5]]))
    );
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint], { normalize: false });

    const result = await embedder.embedBatch(["a", "b", "c"]);

    expect(result).toEqual([[1, 0], [0, 1], [0.5, 0.5]]);
  });
});

describe("RemoteEmbedder per-endpoint TTL", () => {
  test("skips an unhealthy endpoint within TTL window, retries after TTL", async () => {
    vi.useFakeTimers();
    const endpointA = parseRemoteEmbedUri("http://a:1234/v1#m");
    const endpointB = parseRemoteEmbedUri("http://b:1234/v1#m");
    const embedder = new RemoteEmbedder([endpointA, endpointB], { healthTtlMs: 15000 });

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      if (String(url).startsWith("http://a:1234")) {
        throw new Error("connection refused");
      }
      return jsonResponse(embeddingsBody([[1, 0]]));
    });

    // First call: A fails, B succeeds. A is now marked unhealthy.
    await embedder.embedBatch(["x"]);
    expect(fetchSpy).toHaveBeenCalledTimes(2);

    fetchSpy.mockClear();

    // Second call within TTL: A should be skipped entirely.
    await embedder.embedBatch(["y"]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(String(fetchSpy.mock.calls[0]![0])).toContain("http://b:1234");

    fetchSpy.mockClear();

    // Advance past the TTL: A should be tried again.
    vi.advanceTimersByTime(15001);
    await embedder.embedBatch(["z"]);
    expect(String(fetchSpy.mock.calls[0]![0])).toContain("http://a:1234");
  });
});

describe("RemoteEmbedder sequential fallback", () => {
  test("falls back to the second endpoint when the first fails, without retrying the first", async () => {
    const endpointA = parseRemoteEmbedUri("http://a:1234/v1#m");
    const endpointB = parseRemoteEmbedUri("http://b:1234/v1#m");
    const embedder = new RemoteEmbedder([endpointA, endpointB]);

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      if (String(url).startsWith("http://a:1234")) {
        return jsonResponse({ error: "server error" }, 500);
      }
      return jsonResponse(embeddingsBody([[0.6, 0.8]]));
    });

    const result = await embedder.embedBatch(["x"]);

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result).toEqual([[0.6, 0.8]]);
  });
});

describe("RemoteEmbedder all-fail", () => {
  test("rejects with RemoteEmbedAllFailedError when every endpoint fails", async () => {
    const endpointA = parseRemoteEmbedUri("http://a:1234/v1#m");
    const endpointB = parseRemoteEmbedUri("http://b:1234/v1#m");
    const embedder = new RemoteEmbedder([endpointA, endpointB]);

    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network down"));

    await expect(embedder.embedBatch(["x"])).rejects.toThrow(RemoteEmbedAllFailedError);
  });
});

describe("RemoteEmbedder L2 normalization", () => {
  test("normalizes vectors to unit norm by default", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(embeddingsBody([[3, 4]])));
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint]);

    const [vector] = await embedder.embedBatch(["x"]);
    const norm = Math.sqrt(vector!.reduce((sum, v) => sum + v * v, 0));

    expect(norm).toBeCloseTo(1, 6);
    expect(vector).toEqual([0.6, 0.8]);
  });

  test("does not normalize when normalize:false", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(embeddingsBody([[3, 4]])));
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint], { normalize: false });

    const [vector] = await embedder.embedBatch(["x"]);

    expect(vector).toEqual([3, 4]);
  });
});

describe("RemoteEmbedder.ping", () => {
  test("returns true on HTTP 200", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 200 }));
    const endpoint: RemoteEmbedEndpoint = { raw: "http://x/v1#m", apiBase: "http://x/v1", modelId: "m" };
    const embedder = new RemoteEmbedder([endpoint]);

    expect(await embedder.ping(endpoint)).toBe(true);
  });

  test("returns false on non-2xx status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    const endpoint: RemoteEmbedEndpoint = { raw: "http://x/v1#m", apiBase: "http://x/v1", modelId: "m" };
    const embedder = new RemoteEmbedder([endpoint]);

    expect(await embedder.ping(endpoint)).toBe(false);
  });

  test("returns false when fetch throws", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("connection refused"));
    const endpoint: RemoteEmbedEndpoint = { raw: "http://x/v1#m", apiBase: "http://x/v1", modelId: "m" };
    const embedder = new RemoteEmbedder([endpoint]);

    expect(await embedder.ping(endpoint)).toBe(false);
  });
});

describe("RemoteEmbedder.embed", () => {
  test("returns the single embedded vector", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(embeddingsBody([[1, 0]])));
    const endpoint = parseRemoteEmbedUri("http://localhost:1234/v1#m");
    const embedder = new RemoteEmbedder([endpoint], { normalize: false });

    expect(await embedder.embed("hello")).toEqual([1, 0]);
  });
});
