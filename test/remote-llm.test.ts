/**
 * Tests for RemoteLLM and HybridLLM
 *
 * Uses a local HTTP server to mock OpenAI-compatible endpoints.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { createServer, type Server, type IncomingMessage, type ServerResponse } from "http";
import { RemoteLLM, remoteConfigFromEnv, type RemoteLLMConfig } from "../src/remote-llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";
import { isRemoteModel, formatQueryForEmbedding, formatDocForEmbedding, getDefaultLLM, setDefaultLLM, LlamaCpp } from "../src/llm.js";
import type { LLM, EmbeddingResult, RerankResult, Queryable, GenerateResult, ModelInfo } from "../src/llm.js";

// =============================================================================
// Mock HTTP server
// =============================================================================

type MockHandler = (req: IncomingMessage, body: string) => { status: number; body: any };

let server: Server;
let serverPort: number;
let mockHandler: MockHandler;

function setMockHandler(handler: MockHandler) {
  mockHandler = handler;
}

beforeAll(async () => {
  server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(chunk as Buffer);
    }
    const body = Buffer.concat(chunks).toString();

    try {
      const result = mockHandler(req, body);
      res.writeHead(result.status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result.body));
    } catch (err: any) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (typeof addr === "object" && addr) {
        serverPort = addr.port;
      }
      resolve();
    });
  });
});

afterAll(() => {
  server.close();
});

function baseUrl(): string {
  return `http://127.0.0.1:${serverPort}/v1`;
}

function createRemoteLLM(overrides?: Partial<RemoteLLMConfig>): RemoteLLM {
  return new RemoteLLM({
    embedApiUrl: baseUrl(),
    embedApiModel: "test-model",
    ...overrides,
  });
}

// =============================================================================
// RemoteLLM Tests
// =============================================================================

describe("RemoteLLM", () => {
  describe("embed", () => {
    it("should embed a single text", async () => {
      setMockHandler((req, body) => {
        const parsed = JSON.parse(body);
        expect(parsed.model).toBe("test-model");
        expect(parsed.input).toEqual(["hello world"]);
        return {
          status: 200,
          body: {
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          },
        };
      });

      const llm = createRemoteLLM();
      const result = await llm.embed("hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(result!.model).toBe("test-model");
    });

    it("should embed a batch of texts", async () => {
      setMockHandler((_req, body) => {
        const parsed = JSON.parse(body);
        return {
          status: 200,
          body: {
            data: parsed.input.map((text: string, i: number) => ({
              embedding: [i * 0.1, i * 0.2],
              index: i,
            })),
          },
        };
      });

      const llm = createRemoteLLM();
      const results = await llm.embedBatch(["text1", "text2", "text3"]);
      expect(results).toHaveLength(3);
      expect(results[0]!.embedding).toEqual([0, 0]);
      expect(results[2]!.embedding).toEqual([0.2, 0.4]);
    });

    it("should return empty array for empty input", async () => {
      const llm = createRemoteLLM();
      const results = await llm.embedBatch([]);
      expect(results).toEqual([]);
    });

    it("should split large batches", async () => {
      const requestBodies: string[][] = [];
      setMockHandler((_req, body) => {
        const parsed = JSON.parse(body);
        requestBodies.push(parsed.input);
        return {
          status: 200,
          body: {
            data: parsed.input.map((_: string, i: number) => ({
              embedding: [1.0],
              index: i,
            })),
          },
        };
      });

      const llm = createRemoteLLM({ maxBatchSize: 2 });
      const texts = ["a", "b", "c", "d", "e"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(5);
      // Should have made 3 requests: [a,b], [c,d], [e]
      expect(requestBodies).toHaveLength(3);
      expect(requestBodies[0]).toEqual(["a", "b"]);
      expect(requestBodies[1]).toEqual(["c", "d"]);
      expect(requestBodies[2]).toEqual(["e"]);
    });

    it("should sort response by index", async () => {
      setMockHandler(() => ({
        status: 200,
        body: {
          // Return in reverse order
          data: [
            { embedding: [0.3], index: 2 },
            { embedding: [0.1], index: 0 },
            { embedding: [0.2], index: 1 },
          ],
        },
      }));

      const llm = createRemoteLLM();
      const results = await llm.embedBatch(["a", "b", "c"]);
      expect(results[0]!.embedding).toEqual([0.1]);
      expect(results[1]!.embedding).toEqual([0.2]);
      expect(results[2]!.embedding).toEqual([0.3]);
    });
  });

  describe("auth", () => {
    it("should send Authorization header when key is set", async () => {
      let authHeader: string | undefined;
      setMockHandler((req) => {
        authHeader = req.headers["authorization"] as string;
        return {
          status: 200,
          body: { data: [{ embedding: [1.0], index: 0 }] },
        };
      });

      const llm = createRemoteLLM({ embedApiKey: "test-key-123" });
      await llm.embed("test");
      expect(authHeader).toBe("Bearer test-key-123");
    });

    it("should not send Authorization header when no key", async () => {
      let authHeader: string | undefined;
      setMockHandler((req) => {
        authHeader = req.headers["authorization"] as string;
        return {
          status: 200,
          body: { data: [{ embedding: [1.0], index: 0 }] },
        };
      });

      const llm = createRemoteLLM();
      await llm.embed("test");
      expect(authHeader).toBeUndefined();
    });
  });

  describe("dimension validation", () => {
    it("should reject dimension mismatch after first response", async () => {
      let callCount = 0;
      setMockHandler(() => {
        callCount++;
        const dim = callCount === 1 ? [1.0, 2.0, 3.0] : [1.0, 2.0];
        return {
          status: 200,
          body: { data: [{ embedding: dim, index: 0 }] },
        };
      });

      const llm = createRemoteLLM();
      // First call succeeds and locks dimensions to 3
      await llm.embed("first");
      // Second call should fail because dimensions changed
      await expect(llm.embed("second")).rejects.toThrow("dimension mismatch");
    });
  });

  describe("error handling", () => {
    it("should throw on HTTP error", async () => {
      setMockHandler(() => ({
        status: 500,
        body: { error: "Internal server error" },
      }));

      const llm = createRemoteLLM();
      await expect(llm.embed("test")).rejects.toThrow("500");
    });

    it("should open circuit breaker after failures", async () => {
      setMockHandler(() => ({
        status: 500,
        body: { error: "down" },
      }));

      const llm = createRemoteLLM();
      // Fail 3 times to trip the breaker
      for (let i = 0; i < 3; i++) {
        await expect(llm.embed("test")).rejects.toThrow();
      }
      // Next call should fail immediately with circuit breaker message
      await expect(llm.embed("test")).rejects.toThrow("circuit breaker");
    });
  });

  describe("rerank", () => {
    it("should rerank documents", async () => {
      setMockHandler((_req, body) => {
        const parsed = JSON.parse(body);
        expect(parsed.model).toBe("rerank-model");
        expect(parsed.query).toBe("test query");
        expect(parsed.documents).toEqual(["doc A text", "doc B text"]);
        return {
          status: 200,
          body: {
            results: [
              { index: 1, relevance_score: 0.9 },
              { index: 0, relevance_score: 0.3 },
            ],
          },
        };
      });

      const llm = createRemoteLLM({
        rerankApiModel: "rerank-model",
      });
      const result = await llm.rerank(
        "test query",
        [
          { file: "a.md", text: "doc A text" },
          { file: "b.md", text: "doc B text" },
        ]
      );

      expect(result.model).toBe("rerank-model");
      expect(result.results).toHaveLength(2);
      expect(result.results.find(r => r.file === "b.md")!.score).toBe(0.9);
      expect(result.results.find(r => r.file === "a.md")!.score).toBe(0.3);
    });

    it("should throw when rerankApiModel not configured", async () => {
      const llm = createRemoteLLM();
      await expect(
        llm.rerank("query", [{ file: "a.md", text: "text" }])
      ).rejects.toThrow("rerankApiModel");
    });
  });

  describe("unsupported operations", () => {
    it("should throw on generate", async () => {
      const llm = createRemoteLLM();
      await expect(llm.generate("prompt")).rejects.toThrow("does not support text generation");
    });

    it("should throw on expandQuery", async () => {
      const llm = createRemoteLLM();
      await expect(llm.expandQuery("query")).rejects.toThrow("does not support query expansion");
    });
  });
});

// =============================================================================
// HybridLLM Tests
// =============================================================================

describe("HybridLLM", () => {
  // Simple mock local LLM
  function createMockLocalLLM(): LLM {
    return {
      embedModelName: "local-model",
      embed: async () => ({ embedding: [0.5], model: "local-model" }),
      embedBatch: async (texts) => texts.map(() => ({ embedding: [0.5], model: "local-model" })),
      generate: async () => ({ text: "expanded", model: "local-model", done: true }),
      modelExists: async (model) => ({ name: model, exists: true }),
      expandQuery: async () => [{ type: "lex" as const, text: "expanded query" }],
      rerank: async () => ({ results: [], model: "local-model" }),
      dispose: async () => {},
    };
  }

  it("should route embed to remote", async () => {
    setMockHandler(() => ({
      status: 200,
      body: { data: [{ embedding: [0.9], index: 0 }] },
    }));

    const remote = createRemoteLLM();
    const local = createMockLocalLLM();
    const hybrid = new HybridLLM(remote, local);

    const result = await hybrid.embed("test");
    // Should come from remote (0.9), not local (0.5)
    expect(result!.embedding).toEqual([0.9]);
  });

  it("should route embedBatch to remote", async () => {
    setMockHandler((_req, body) => {
      const parsed = JSON.parse(body);
      return {
        status: 200,
        body: {
          data: parsed.input.map((_: string, i: number) => ({
            embedding: [0.9 + i * 0.01],
            index: i,
          })),
        },
      };
    });

    const remote = createRemoteLLM();
    const local = createMockLocalLLM();
    const hybrid = new HybridLLM(remote, local);

    const results = await hybrid.embedBatch(["a", "b"]);
    expect(results[0]!.embedding).toEqual([0.9]);
    expect(results[1]!.embedding).toEqual([0.91]);
  });

  it("should route generate to local", async () => {
    const local = createMockLocalLLM();
    const remote = createRemoteLLM();
    const hybrid = new HybridLLM(remote, local);

    const result = await hybrid.generate("prompt");
    expect(result!.text).toBe("expanded");
    expect(result!.model).toBe("local-model");
  });

  it("should route expandQuery to local", async () => {
    const local = createMockLocalLLM();
    const remote = createRemoteLLM();
    const hybrid = new HybridLLM(remote, local);

    const result = await hybrid.expandQuery("test query");
    expect(result[0]!.text).toBe("expanded query");
  });

  it("should use remote embedModelName", async () => {
    const remote = createRemoteLLM({ embedApiModel: "BAAI/bge-m3" });
    const local = createMockLocalLLM();
    const hybrid = new HybridLLM(remote, local);

    expect(hybrid.embedModelName).toBe("BAAI/bge-m3");
  });
});

// =============================================================================
// Config Tests
// =============================================================================

describe("remoteConfigFromEnv", () => {
  const origEnv = { ...process.env };

  beforeEach(() => {
    // Clear any QMD_ env vars
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("QMD_") && key.includes("API")) {
        delete process.env[key];
      }
    }
  });

  afterAll(() => {
    // Restore original env
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("QMD_") && key.includes("API")) {
        delete process.env[key];
      }
    }
    Object.assign(process.env, origEnv);
  });

  it("should return null when no config", () => {
    expect(remoteConfigFromEnv()).toBeNull();
  });

  it("should parse env vars", () => {
    process.env.QMD_EMBED_API_URL = "http://gpu:8000/v1";
    process.env.QMD_EMBED_API_MODEL = "bge-m3";
    process.env.QMD_EMBED_API_KEY = "secret";

    const config = remoteConfigFromEnv();
    expect(config).not.toBeNull();
    expect(config!.embedApiUrl).toBe("http://gpu:8000/v1");
    expect(config!.embedApiModel).toBe("bge-m3");
    expect(config!.embedApiKey).toBe("secret");
  });

  it("should use YAML config as fallback", () => {
    const config = remoteConfigFromEnv({
      embed_api_url: "http://yaml:8000/v1",
      embed_api_model: "yaml-model",
    });
    expect(config).not.toBeNull();
    expect(config!.embedApiUrl).toBe("http://yaml:8000/v1");
  });

  it("should prefer env vars over YAML", () => {
    process.env.QMD_EMBED_API_URL = "http://env:8000/v1";
    process.env.QMD_EMBED_API_MODEL = "env-model";

    const config = remoteConfigFromEnv({
      embed_api_url: "http://yaml:8000/v1",
      embed_api_model: "yaml-model",
    });
    expect(config!.embedApiUrl).toBe("http://env:8000/v1");
    expect(config!.embedApiModel).toBe("env-model");
  });
});

// =============================================================================
// Embedding format tests
// =============================================================================

describe("isRemoteModel", () => {
  it("should detect remote models", () => {
    expect(isRemoteModel("BAAI/bge-m3")).toBe(true);
    expect(isRemoteModel("intfloat/multilingual-e5-large")).toBe(true);
    expect(isRemoteModel("text-embedding-ada-002")).toBe(true);
  });

  it("should detect local models", () => {
    expect(isRemoteModel("hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf")).toBe(false);
    expect(isRemoteModel("/path/to/model.gguf")).toBe(false);
  });
});

describe("formatQueryForEmbedding with remote models", () => {
  it("should return raw query for remote models", () => {
    expect(formatQueryForEmbedding("test query", "BAAI/bge-m3")).toBe("test query");
  });

  it("should add prefix for local nomic models", () => {
    expect(formatQueryForEmbedding("test query", "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf")).toContain("task:");
  });
});

describe("formatDocForEmbedding with remote models", () => {
  it("should return raw text for remote models", () => {
    expect(formatDocForEmbedding("doc text", undefined, "BAAI/bge-m3")).toBe("doc text");
  });

  it("should include title when provided for remote models", () => {
    expect(formatDocForEmbedding("doc text", "My Title", "BAAI/bge-m3")).toBe("My Title\ndoc text");
  });
});

// =============================================================================
// Local-only path (no remote config)
// =============================================================================

describe("Local-only LlamaCpp path", () => {
  afterEach(() => {
    // Reset to default so other tests aren't affected
    setDefaultLLM(null);
  });

  it("getDefaultLLM() returns a LlamaCpp instance when nothing is configured", () => {
    setDefaultLLM(null);
    const llm = getDefaultLLM();
    expect(llm).toBeInstanceOf(LlamaCpp);
  });

  it("LlamaCpp instance satisfies the LLM interface", () => {
    const llm = new LlamaCpp();
    // All LLM interface methods exist
    expect(typeof llm.embed).toBe("function");
    expect(typeof llm.embedBatch).toBe("function");
    expect(typeof llm.generate).toBe("function");
    expect(typeof llm.modelExists).toBe("function");
    expect(typeof llm.expandQuery).toBe("function");
    expect(typeof llm.rerank).toBe("function");
    expect(typeof llm.dispose).toBe("function");
    expect(typeof llm.embedModelName).toBe("string");
  });

  it("LlamaCpp has tokenize method (used by chunkDocumentByTokens duck-typing)", () => {
    const llm = new LlamaCpp();
    expect(typeof llm.tokenize).toBe("function");
  });

  it("setDefaultLLM with LlamaCpp is retrievable via getDefaultLLM", () => {
    const llm = new LlamaCpp();
    setDefaultLLM(llm);
    expect(getDefaultLLM()).toBe(llm);
  });

  it("remoteConfigFromEnv returns null when no env vars or YAML set", () => {
    // Clear any remote env vars
    const saved: Record<string, string | undefined> = {};
    for (const key of ["QMD_EMBED_API_URL", "QMD_EMBED_API_MODEL"]) {
      saved[key] = process.env[key];
      delete process.env[key];
    }
    try {
      expect(remoteConfigFromEnv()).toBeNull();
      expect(remoteConfigFromEnv({})).toBeNull();
      expect(remoteConfigFromEnv({ embed_api_url: undefined })).toBeNull();
    } finally {
      for (const [key, val] of Object.entries(saved)) {
        if (val !== undefined) process.env[key] = val;
      }
    }
  });

  it("formatQueryForEmbedding adds nomic prefix for default local model", () => {
    // Default model is embeddinggemma (hf: URI), should get task prefix
    const formatted = formatQueryForEmbedding("hello");
    expect(formatted).toContain("task:");
    expect(formatted).toContain("hello");
  });

  it("formatDocForEmbedding adds nomic prefix for default local model", () => {
    const formatted = formatDocForEmbedding("doc content", "My Doc");
    expect(formatted).toContain("title: My Doc");
    expect(formatted).toContain("text: doc content");
  });
});
