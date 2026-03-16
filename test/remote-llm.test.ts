import { describe, test, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { RemoteLLM, type RemoteLLMConfig } from "../src/remote-llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";
import http from "http";

// =============================================================================
// Mock HTTP Server
// =============================================================================

let server: http.Server;
let baseUrl: string;

// Track last requests for assertions
let lastRequest: { path: string; body: any } | null = null;

function createMockServer(): Promise<{ server: http.Server; baseUrl: string }> {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        const parsed = body ? JSON.parse(body) : {};
        lastRequest = { path: req.url || "", body: parsed };

        res.setHeader("Content-Type", "application/json");

        if (req.url === "/v1/embeddings") {
          const input = parsed.input as string[];
          const data = input.map((text: string, index: number) => ({
            embedding: Array.from({ length: 768 }, (_, i) => Math.sin(i + text.length)),
            index,
          }));
          res.end(JSON.stringify({ data, model: parsed.model }));
        } else if (req.url === "/v1/rerank") {
          const docs = parsed.documents as string[];
          const results = docs.map((_: string, index: number) => ({
            index,
            relevance_score: 1 - index * 0.1,
          }));
          res.end(JSON.stringify({ results }));
        } else if (req.url === "/v1/models") {
          res.end(JSON.stringify({
            data: [
              { id: "bge-m3" },
              { id: "bge-reranker-v2-m3" },
            ],
          }));
        } else {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "not found" }));
        }
      });
    });

    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      if (typeof addr === "object" && addr) {
        resolve({ server: srv, baseUrl: `http://127.0.0.1:${addr.port}/v1` });
      }
    });
  });
}

// =============================================================================
// Tests
// =============================================================================

describe("RemoteLLM", () => {
  beforeAll(async () => {
    const mock = await createMockServer();
    server = mock.server;
    baseUrl = mock.baseUrl;
  });

  afterAll(() => {
    server?.close();
  });

  beforeEach(() => {
    lastRequest = null;
  });

  function createRemote(overrides?: Partial<RemoteLLMConfig>): RemoteLLM {
    return new RemoteLLM({
      baseUrl,
      embedModel: "bge-m3",
      rerankModel: "bge-reranker-v2-m3",
      ...overrides,
    });
  }

  test("isRemote is true", () => {
    const remote = createRemote();
    expect(remote.isRemote).toBe(true);
  });

  // ── Embedding ──────────────────────────────────────────────────────────

  test("embed() returns embedding for single text", async () => {
    const remote = createRemote();
    const result = await remote.embed("hello world");
    expect(result).not.toBeNull();
    expect(result!.embedding).toHaveLength(768);
    expect(result!.model).toBe("bge-m3");
  });

  test("embedBatch() sends correct request format", async () => {
    const remote = createRemote();
    const texts = ["hello", "world", "test"];
    const results = await remote.embedBatch(texts);

    expect(results).toHaveLength(3);
    expect(results.every((r) => r !== null)).toBe(true);
    expect(lastRequest?.path).toBe("/v1/embeddings");
    expect(lastRequest?.body.model).toBe("bge-m3");
    expect(lastRequest?.body.input).toEqual(texts);
  });

  test("embedBatch() returns empty array for empty input", async () => {
    const remote = createRemote();
    const results = await remote.embedBatch([]);
    expect(results).toEqual([]);
  });

  test("embedBatch() preserves order via index field", async () => {
    const remote = createRemote();
    const results = await remote.embedBatch(["short", "a longer piece of text"]);
    expect(results).toHaveLength(2);
    // Different input lengths produce different embeddings
    expect(results[0]!.embedding).not.toEqual(results[1]!.embedding);
  });

  // ── Reranking ──────────────────────────────────────────────────────────

  test("rerank() sends correct request format", async () => {
    const remote = createRemote();
    const docs = [
      { file: "a.md", text: "first document" },
      { file: "b.md", text: "second document" },
    ];
    const result = await remote.rerank("test query", docs);

    expect(lastRequest?.path).toBe("/v1/rerank");
    expect(lastRequest?.body.model).toBe("bge-reranker-v2-m3");
    expect(lastRequest?.body.query).toBe("test query");
    expect(lastRequest?.body.documents).toEqual(["first document", "second document"]);
    expect(lastRequest?.body.return_documents).toBe(false);

    expect(result.results).toHaveLength(2);
    expect(result.results[0]!.file).toBe("a.md");
    expect(result.results[0]!.score).toBeGreaterThan(0);
    expect(result.model).toBe("bge-reranker-v2-m3");
  });

  test("rerank() returns empty results for empty documents", async () => {
    const remote = createRemote();
    const result = await remote.rerank("query", []);
    expect(result.results).toEqual([]);
  });

  // ── Model Exists ───────────────────────────────────────────────────────

  test("modelExists() returns true for available model", async () => {
    const remote = createRemote();
    const result = await remote.modelExists("bge-m3");
    expect(result.exists).toBe(true);
    expect(result.name).toBe("bge-m3");
  });

  test("modelExists() returns false for unknown model", async () => {
    const remote = createRemote();
    const result = await remote.modelExists("nonexistent-model");
    expect(result.exists).toBe(false);
  });

  // ── Unsupported Operations ─────────────────────────────────────────────

  test("generate() throws not supported", async () => {
    const remote = createRemote();
    await expect(remote.generate("hello")).rejects.toThrow("not support");
  });

  test("expandQuery() throws not supported", async () => {
    const remote = createRemote();
    await expect(remote.expandQuery("test")).rejects.toThrow("not support");
  });

  // ── Auth Header ────────────────────────────────────────────────────────

  test("sends Authorization header when apiKey is set", async () => {
    // Create a server that captures headers
    let capturedAuth: string | undefined;
    const authServer = http.createServer((req, res) => {
      capturedAuth = req.headers.authorization;
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        res.setHeader("Content-Type", "application/json");
        const parsed = JSON.parse(body);
        const input = parsed.input as string[];
        res.end(JSON.stringify({
          data: input.map((_, i: number) => ({ embedding: [0, 1, 2], index: i })),
          model: "test",
        }));
      });
    });

    await new Promise<void>((resolve) => authServer.listen(0, "127.0.0.1", resolve));
    const addr = authServer.address();
    const port = typeof addr === "object" && addr ? addr.port : 0;

    try {
      const remote = new RemoteLLM({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        apiKey: "test-key-123",
      });
      await remote.embed("test");
      expect(capturedAuth).toBe("Bearer test-key-123");
    } finally {
      authServer.close();
    }
  });

  // ── Dispose ────────────────────────────────────────────────────────────

  test("dispose() is a no-op", async () => {
    const remote = createRemote();
    await expect(remote.dispose()).resolves.toBeUndefined();
  });
});

// =============================================================================
// HybridLLM Tests
// =============================================================================

describe("HybridLLM", () => {
  let mockServer: http.Server;
  let mockBaseUrl: string;

  beforeAll(async () => {
    const mock = await createMockServer();
    mockServer = mock.server;
    mockBaseUrl = mock.baseUrl;
  });

  afterAll(() => {
    mockServer?.close();
  });

  test("isRemote is true", () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => ({ text: "gen", model: "local", done: true }),
      expandQuery: async () => [{ type: "vec" as const, text: "expanded" }],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    expect(hybrid.isRemote).toBe(true);
  });

  test("routes embed to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl, embedModel: "bge-m3" });
    const localCalled = { embed: false };
    const local = {
      embed: async () => { localCalled.embed = true; return null; },
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);

    const result = await hybrid.embed("test");
    expect(result).not.toBeNull();
    expect(result!.model).toBe("bge-m3");
    expect(localCalled.embed).toBe(false);
  });

  test("routes embedBatch to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const results = await hybrid.embedBatch(["a", "b"]);
    expect(results).toHaveLength(2);
  });

  test("routes rerank to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl, rerankModel: "bge-reranker-v2-m3" });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const result = await hybrid.rerank("query", [{ file: "a.md", text: "doc" }]);
    expect(result.model).toBe("bge-reranker-v2-m3");
  });

  test("routes generate to local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => ({ text: "local-generated", model: "qwen", done: true }),
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const result = await hybrid.generate("test prompt");
    expect(result).not.toBeNull();
    expect(result!.text).toBe("local-generated");
  });

  test("routes expandQuery to local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [{ type: "vec" as const, text: "expanded-locally" }],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const results = await hybrid.expandQuery("test");
    expect(results).toHaveLength(1);
    expect(results[0]!.text).toBe("expanded-locally");
  });

  test("modelExists tries remote first then local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async (model: string) => ({ name: model, exists: model === "local-only-model" }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);

    // Remote has bge-m3
    const remoteModel = await hybrid.modelExists("bge-m3");
    expect(remoteModel.exists).toBe(true);

    // Falls back to local
    const localModel = await hybrid.modelExists("local-only-model");
    expect(localModel.exists).toBe(true);
  });

  test("dispose() calls both", async () => {
    let localDisposed = false;
    let remoteDisposed = false;
    const remote = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "remote" }),
      modelExists: async () => ({ name: "remote", exists: false }),
      dispose: async () => { remoteDisposed = true; },
      isRemote: true as const,
    };
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => { localDisposed = true; },
    };
    const hybrid = new HybridLLM(local, remote);
    await hybrid.dispose();
    expect(localDisposed).toBe(true);
    expect(remoteDisposed).toBe(true);
  });
});
