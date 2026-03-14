/**
 * Tests for remote embedding provider support.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  RemoteEmbeddingProvider,
  resolveApiKey,
  parseEmbeddingConfig,
  createRemoteEmbeddingProvider,
  type RemoteEmbeddingConfig,
} from "../src/remote-embedding.js";

// =============================================================================
// resolveApiKey
// =============================================================================

describe("resolveApiKey", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("uses literal api_key from config", () => {
    const config: RemoteEmbeddingConfig = {
      provider: "gemini",
      model: "gemini-embedding-2-preview",
      api_key: "my-literal-key",
    };
    expect(resolveApiKey(config)).toBe("my-literal-key");
  });

  it("resolves ${ENV_VAR} syntax", () => {
    process.env.MY_API_KEY = "resolved-key-from-env";
    const config: RemoteEmbeddingConfig = {
      provider: "gemini",
      model: "gemini-embedding-2-preview",
      api_key: "${MY_API_KEY}",
    };
    expect(resolveApiKey(config)).toBe("resolved-key-from-env");
  });

  it("falls back to GOOGLE_API_KEY for gemini provider", () => {
    process.env.GOOGLE_API_KEY = "google-fallback-key";
    const config: RemoteEmbeddingConfig = {
      provider: "gemini",
      model: "gemini-embedding-2-preview",
    };
    expect(resolveApiKey(config)).toBe("google-fallback-key");
  });

  it("falls back to OPENAI_API_KEY for openai provider", () => {
    process.env.OPENAI_API_KEY = "openai-fallback-key";
    const config: RemoteEmbeddingConfig = {
      provider: "openai",
      model: "text-embedding-3-small",
    };
    expect(resolveApiKey(config)).toBe("openai-fallback-key");
  });

  it("throws when no key is available", () => {
    delete process.env.GOOGLE_API_KEY;
    delete process.env.OPENAI_API_KEY;
    const config: RemoteEmbeddingConfig = {
      provider: "gemini",
      model: "gemini-embedding-2-preview",
    };
    expect(() => resolveApiKey(config)).toThrow(/No API key found/);
  });

  it("does not resolve partial ${} patterns", () => {
    process.env.GOOGLE_API_KEY = "fallback";
    const config: RemoteEmbeddingConfig = {
      provider: "gemini",
      model: "gemini-embedding-2-preview",
      api_key: "not-${a-var}",
    };
    // Should use the literal value (it's not a pure ${VAR} pattern)
    expect(resolveApiKey(config)).toBe("not-${a-var}");
  });
});

// =============================================================================
// parseEmbeddingConfig
// =============================================================================

describe("parseEmbeddingConfig", () => {
  it("returns null when no embedding config", () => {
    expect(parseEmbeddingConfig({})).toBeNull();
    expect(parseEmbeddingConfig({ embedding: undefined })).toBeNull();
  });

  it("returns null when missing provider or model", () => {
    expect(parseEmbeddingConfig({
      embedding: { provider: "gemini" } as any,
    })).toBeNull();
    expect(parseEmbeddingConfig({
      embedding: { model: "foo" } as any,
    })).toBeNull();
  });

  it("throws on unsupported provider", () => {
    expect(() => parseEmbeddingConfig({
      embedding: { provider: "unknown" as any, model: "foo" },
    })).toThrow(/Unsupported embedding provider/);
  });

  it("parses valid gemini config", () => {
    const config = parseEmbeddingConfig({
      embedding: {
        provider: "gemini",
        model: "gemini-embedding-2-preview",
        dimensions: 3072,
      },
    });
    expect(config).toEqual({
      provider: "gemini",
      model: "gemini-embedding-2-preview",
      dimensions: 3072,
    });
  });

  it("parses valid openai config", () => {
    const config = parseEmbeddingConfig({
      embedding: {
        provider: "openai",
        model: "text-embedding-3-small",
        base_url: "http://localhost:8080/v1",
      },
    });
    expect(config).toEqual({
      provider: "openai",
      model: "text-embedding-3-small",
      base_url: "http://localhost:8080/v1",
    });
  });
});

// =============================================================================
// createRemoteEmbeddingProvider
// =============================================================================

describe("createRemoteEmbeddingProvider", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns null when no embedding config", () => {
    expect(createRemoteEmbeddingProvider({})).toBeNull();
  });

  it("creates a provider with valid config", () => {
    process.env.GOOGLE_API_KEY = "test-key";
    const provider = createRemoteEmbeddingProvider({
      embedding: {
        provider: "gemini",
        model: "gemini-embedding-2-preview",
        dimensions: 3072,
      },
    });
    expect(provider).not.toBeNull();
    expect(provider!.provider).toBe("gemini");
    expect(provider!.model).toBe("gemini-embedding-2-preview");
    expect(provider!.modelUri).toBe("gemini:gemini-embedding-2-preview");
    expect(provider!.dimensions).toBe(3072);
  });
});

// =============================================================================
// RemoteEmbeddingProvider
// =============================================================================

describe("RemoteEmbeddingProvider", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.restoreAllMocks();
  });

  describe("Gemini provider", () => {
    it("calls the correct endpoint for single embed", async () => {
      process.env.GOOGLE_API_KEY = "test-gemini-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "gemini",
        model: "gemini-embedding-2-preview",
      });

      const mockResponse = {
        ok: true,
        json: async () => ({
          embedding: { values: [0.1, 0.2, 0.3] },
        }),
        text: async () => "",
      };

      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse as any);

      const result = await provider.embed("hello world");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, opts] = fetchSpy.mock.calls[0]!;
      expect(url).toContain("generativelanguage.googleapis.com");
      expect(url).toContain("gemini-embedding-2-preview:embedContent");
      expect((opts as any).headers["x-goog-api-key"]).toBe("test-gemini-key");
      expect(result.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(result.model).toBe("gemini:gemini-embedding-2-preview");
    });

    it("calls batch endpoint for multiple texts", async () => {
      process.env.GOOGLE_API_KEY = "test-gemini-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "gemini",
        model: "gemini-embedding-2-preview",
      });

      const mockResponse = {
        ok: true,
        json: async () => ({
          embeddings: [
            { values: [0.1, 0.2] },
            { values: [0.3, 0.4] },
          ],
        }),
        text: async () => "",
      };

      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse as any);

      const results = await provider.embedBatch(["hello", "world"]);

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url] = fetchSpy.mock.calls[0]!;
      expect(url).toContain("batchEmbedContents");
      expect(results).toHaveLength(2);
      expect(results[0]!.embedding).toEqual([0.1, 0.2]);
      expect(results[1]!.embedding).toEqual([0.3, 0.4]);
    });

    it("truncates to configured dimensions", async () => {
      process.env.GOOGLE_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "gemini",
        model: "test-model",
        dimensions: 2,
      });

      vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        json: async () => ({
          embedding: { values: [0.1, 0.2, 0.3, 0.4] },
        }),
        text: async () => "",
      } as any);

      const result = await provider.embed("test");
      expect(result.embedding).toEqual([0.1, 0.2]);
    });

    it("throws on API error", async () => {
      process.env.GOOGLE_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "gemini",
        model: "test-model",
      });

      vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: false,
        status: 403,
        text: async () => "Forbidden",
      } as any);

      await expect(provider.embed("test")).rejects.toThrow(/Gemini embedding API error \(403\)/);
    });
  });

  describe("OpenAI provider", () => {
    it("calls the correct endpoint for single embed", async () => {
      process.env.OPENAI_API_KEY = "test-openai-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "openai",
        model: "text-embedding-3-small",
      });

      const mockResponse = {
        ok: true,
        json: async () => ({
          data: [{ embedding: [0.5, 0.6, 0.7], index: 0 }],
        }),
        text: async () => "",
      };

      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockResponse as any);

      const result = await provider.embed("hello");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, opts] = fetchSpy.mock.calls[0]!;
      expect(url).toBe("https://api.openai.com/v1/embeddings");
      expect((opts as any).headers["Authorization"]).toBe("Bearer test-openai-key");
      expect(result.embedding).toEqual([0.5, 0.6, 0.7]);
      expect(result.model).toBe("openai:text-embedding-3-small");
    });

    it("uses custom base_url", async () => {
      process.env.OPENAI_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "openai",
        model: "local-model",
        base_url: "http://localhost:8080/v1",
      });

      vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: [1, 2, 3], index: 0 }],
        }),
        text: async () => "",
      } as any);

      await provider.embed("test");

      const [url] = vi.mocked(globalThis.fetch).mock.calls[0]!;
      expect(url).toBe("http://localhost:8080/v1/embeddings");
    });

    it("sends dimensions when configured", async () => {
      process.env.OPENAI_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "openai",
        model: "text-embedding-3-small",
        dimensions: 256,
      });

      vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ embedding: Array(256).fill(0), index: 0 }],
        }),
        text: async () => "",
      } as any);

      await provider.embed("test");

      const [, opts] = vi.mocked(globalThis.fetch).mock.calls[0]!;
      const body = JSON.parse((opts as any).body);
      expect(body.dimensions).toBe(256);
    });

    it("handles batch embedding with correct index ordering", async () => {
      process.env.OPENAI_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "openai",
        model: "test-model",
      });

      vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [
            { embedding: [0.3, 0.4], index: 1 }, // Out of order
            { embedding: [0.1, 0.2], index: 0 },
          ],
        }),
        text: async () => "",
      } as any);

      const results = await provider.embedBatch(["first", "second"]);
      expect(results[0]!.embedding).toEqual([0.1, 0.2]); // Sorted by index
      expect(results[1]!.embedding).toEqual([0.3, 0.4]);
    });
  });

  describe("empty batch", () => {
    it("returns empty array for empty input", async () => {
      process.env.GOOGLE_API_KEY = "test-key";
      const provider = new RemoteEmbeddingProvider({
        provider: "gemini",
        model: "test-model",
      });

      const results = await provider.embedBatch([]);
      expect(results).toEqual([]);
    });
  });
});
