import http from "node:http";
import { afterAll, afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { LLM } from "../src/llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";
import { RemoteLLM } from "../src/remote-llm.js";

type RecordedRequest = {
  path: string;
  method: string;
  headers: http.IncomingHttpHeaders;
  body: any;
};

async function createServer(
  handler: (req: http.IncomingMessage, res: http.ServerResponse, body: any, requests: RecordedRequest[]) => void,
): Promise<{ server: http.Server; baseUrl: string; requests: RecordedRequest[] }> {
  const requests: RecordedRequest[] = [];
  const server = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      const body = raw ? JSON.parse(raw) : {};
      requests.push({
        path: req.url ?? "",
        method: req.method ?? "GET",
        headers: req.headers,
        body,
      });
      handler(req, res, body, requests);
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address !== "object") {
    throw new Error("Failed to get server address");
  }

  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
  };
}

function createLocalLLM(overrides: Partial<LLM> = {}): LLM {
  return {
    embed: async () => ({ embedding: [0.1], model: "local" }),
    embedBatch: async (texts) => texts.map(() => ({ embedding: [0.1], model: "local" })),
    generate: async () => ({ text: "local-generated", model: "local", done: true }),
    modelExists: async (model) => ({ name: model, exists: true }),
    expandQuery: async () => [{ type: "vec", text: "expanded-local" }],
    rerank: async (_query, documents) => ({
      results: documents.map((document, index) => ({ file: document.file, score: 1 - index * 0.1, index })),
      model: "local",
    }),
    dispose: async () => {},
    ...overrides,
  };
}

const serversToClose: http.Server[] = [];

afterAll(async () => {
  await Promise.all(serversToClose.map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("RemoteLLM", () => {
  beforeEach(() => {
    delete process.env.QMD_REMOTE_EMBED_URL;
    delete process.env.QMD_REMOTE_RERANK_URL;
    delete process.env.QMD_REMOTE_API_KEY;
    delete process.env.QMD_REMOTE_CONNECT_TIMEOUT;
    delete process.env.QMD_REMOTE_READ_TIMEOUT;
  });

  test("uses separate embed and rerank URLs", async () => {
    const embedServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        data: (body.input as string[]).map((_: string, index: number) => ({ index, embedding: [0.1, 0.2, 0.3] })),
        model: body.model,
      }));
    });
    const rerankServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        results: (body.documents as string[]).map((_: string, index: number) => ({ index, relevance_score: 1 - index * 0.1 })),
        model: body.model,
      }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      embedModel: "embed-model",
      rerankModel: "rerank-model",
    });

    await remote.embed("hello");
    await remote.rerank("query", [{ file: "a.md", text: "doc-a" }]);

    expect(embedServer.requests).toHaveLength(1);
    expect(embedServer.requests[0]?.path).toBe("/v1/embeddings");
    expect(rerankServer.requests).toHaveLength(1);
    expect(rerankServer.requests[0]?.path).toBe("/v1/rerank");
  });

  test("adds Qwen instruct prefix for query embeddings and strips legacy prefixes", async () => {
    let lastInput: string[] = [];
    const embedServer = await createServer((_req, res, body) => {
      lastInput = body.input;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
        model: body.model,
      }));
    });
    const rerankServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ results: [] }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      embedModel: "Qwen3-Embedding-8B",
    });

    await remote.embed("task: search result | query: cats", { isQuery: true });
    expect(lastInput).toEqual(["Instruct: Retrieve relevant documents for the given query\nQuery: cats"]);
  });

  test("locks embedding dimension and throws on mismatch", async () => {
    let callCount = 0;
    const embedServer = await createServer((_req, res) => {
      callCount += 1;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        data: [{ index: 0, embedding: callCount === 1 ? [1, 2, 3] : [1, 2, 3, 4] }],
      }));
    });
    const rerankServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ results: [] }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
    });

    await remote.embed("first");
    await expect(remote.embed("second")).rejects.toThrow("Embedding dimension mismatch");
  });

  test("opens the embed circuit after two failures and keeps rerank circuit independent", async () => {
    let embedHits = 0;
    const embedServer = await createServer((_req, res) => {
      embedHits += 1;
      res.statusCode = 500;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ error: "embed failed" }));
    });
    const rerankServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        results: (body.documents as string[]).map((_: string, index: number) => ({ index, relevance_score: 0.9 - index * 0.1 })),
      }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      breakerCooldownMs: 50,
    });

    await expect(remote.embed("first")).rejects.toThrow("HTTP 500");
    await expect(remote.embed("second")).rejects.toThrow("HTTP 500");
    await expect(remote.embed("third")).rejects.toThrow("circuit-breaker cooldown");
    expect(embedHits).toBe(2);

    const rerank = await remote.rerank("query", [{ file: "a.md", text: "doc-a" }]);
    expect(rerank.results[0]?.file).toBe("a.md");
  });

  test("retries after cooldown in half-open state", async () => {
    let callCount = 0;
    const embedServer = await createServer((_req, res) => {
      callCount += 1;
      res.setHeader("content-type", "application/json");
      if (callCount <= 2) {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: "boom" }));
        return;
      }
      res.end(JSON.stringify({
        data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
      }));
    });
    const rerankServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ results: [] }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      breakerCooldownMs: 20,
    });

    await expect(remote.embed("first")).rejects.toThrow();
    await expect(remote.embed("second")).rejects.toThrow();
    await expect(remote.embed("third")).rejects.toThrow("circuit-breaker cooldown");

    await new Promise((resolve) => setTimeout(resolve, 30));
    const result = await remote.embed("fourth");
    expect(result?.embedding).toHaveLength(3);
    expect(callCount).toBe(3);
  });

  test("uses Authorization header when api key is configured", async () => {
    const embedServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        data: (body.input as string[]).map((_: string, index: number) => ({ index, embedding: [0.1, 0.2, 0.3] })),
      }));
    });
    const rerankServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ results: [] }));
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      apiKey: "secret-token",
    });

    await remote.embed("hello");
    expect(embedServer.requests[0]?.headers.authorization).toBe("Bearer secret-token");
  });

  test("enforces connect timeout", async () => {
    const fetchStub = vi.fn(async (_url: string, init?: RequestInit) => {
      const signal = init?.signal as AbortSignal | undefined;
      return await new Promise<Response>((resolve, reject) => {
        signal?.addEventListener("abort", () => reject(signal.reason));
        void resolve;
      });
    });
    vi.stubGlobal("fetch", fetchStub);

    const remote = new RemoteLLM({
      embedUrl: "http://example.com",
      rerankUrl: "http://example.com",
      connectTimeoutMs: 20,
      readTimeoutMs: 100,
    });

    await expect(remote.embed("hello")).rejects.toThrow("connect timeout after 20ms");
  });

  test("enforces read timeout", async () => {
    const fetchStub = vi.fn(async () => new Response(null, {
      status: 200,
    }));

    const responseTextSpy = vi.spyOn(Response.prototype, "text").mockImplementation(async () => {
      return await new Promise<string>(() => {});
    });

    vi.stubGlobal("fetch", fetchStub);

    const remote = new RemoteLLM({
      embedUrl: "http://example.com",
      rerankUrl: "http://example.com",
      connectTimeoutMs: 50,
      readTimeoutMs: 20,
    });

    await expect(remote.embed("hello")).rejects.toThrow("read timeout after 20ms");
    responseTextSpy.mockRestore();
  });
});

describe("HybridLLM", () => {
  test("routes embed and rerank to remote, generate and expansion to local", async () => {
    const calls = {
      localGenerate: 0,
      localExpand: 0,
      remoteEmbed: 0,
      remoteRerank: 0,
    };

    const local = createLocalLLM({
      generate: async () => {
        calls.localGenerate += 1;
        return { text: "local-generated", model: "local", done: true };
      },
      expandQuery: async () => {
        calls.localExpand += 1;
        return [{ type: "vec", text: "expanded-local" }];
      },
    });

    const remote = createLocalLLM({
      isRemote: true,
      embed: async () => {
        calls.remoteEmbed += 1;
        return { embedding: [1, 2, 3], model: "remote-embed" };
      },
      embedBatch: async (texts) => {
        calls.remoteEmbed += texts.length;
        return texts.map(() => ({ embedding: [1, 2, 3], model: "remote-embed" }));
      },
      rerank: async (_query, documents) => {
        calls.remoteRerank += 1;
        return {
          results: documents.map((document, index) => ({ file: document.file, score: 1 - index * 0.1, index })),
          model: "remote-rerank",
        };
      },
    });

    const hybrid = new HybridLLM(local, remote);

    expect((await hybrid.embed("hello"))?.model).toBe("remote-embed");
    expect((await hybrid.generate("prompt"))?.text).toBe("local-generated");
    expect((await hybrid.expandQuery("query"))[0]?.text).toBe("expanded-local");
    expect((await hybrid.rerank("query", [{ file: "a.md", text: "doc-a" }])).model).toBe("remote-rerank");

    expect(calls.remoteEmbed).toBe(1);
    expect(calls.remoteRerank).toBe(1);
    expect(calls.localGenerate).toBe(1);
    expect(calls.localExpand).toBe(1);
  });
});
