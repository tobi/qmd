import http from "node:http";
import {
  afterAll,
  afterEach,
  beforeEach,
  describe,
  expect,
  test,
  vi,
} from "vitest";
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
  handler: (
    req: http.IncomingMessage,
    res: http.ServerResponse,
    body: any,
    requests: RecordedRequest[],
  ) => void,
): Promise<{
  server: http.Server;
  baseUrl: string;
  requests: RecordedRequest[];
}> {
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
    embedBatch: async (texts) =>
      texts.map(() => ({ embedding: [0.1], model: "local" })),
    generate: async () => ({
      text: "local-generated",
      model: "local",
      done: true,
    }),
    modelExists: async (model) => ({ name: model, exists: true }),
    expandQuery: async () => [{ type: "vec", text: "expanded-local" }],
    rerank: async (_query, documents) => ({
      results: documents.map((document, index) => ({
        file: document.file,
        score: 1 - index * 0.1,
        index,
      })),
      model: "local",
    }),
    tokenize: async (text) =>
      Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => i),
    countTokens: async (text) => Math.ceil(text.length / 4),
    detokenize: async (tokens) => "x".repeat((tokens as number[]).length * 4),
    dispose: async () => {},
    ...overrides,
  };
}

const serversToClose: http.Server[] = [];

afterAll(async () => {
  await Promise.all(
    serversToClose.map(
      (server) => new Promise<void>((resolve) => server.close(() => resolve())),
    ),
  );
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
      res.end(
        JSON.stringify({
          data: (body.input as string[]).map((_: string, index: number) => ({
            index,
            embedding: [0.1, 0.2, 0.3],
          })),
          model: body.model,
        }),
      );
    });
    const rerankServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          results: (body.documents as string[]).map(
            (_: string, index: number) => ({
              index,
              relevance_score: 1 - index * 0.1,
            }),
          ),
          model: body.model,
        }),
      );
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
      res.end(
        JSON.stringify({
          data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
          model: body.model,
        }),
      );
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
    expect(lastInput).toEqual([
      "Instruct: Retrieve relevant documents for the given query\nQuery: cats",
    ]);
  });

  test("locks embedding dimension and throws on mismatch", async () => {
    let callCount = 0;
    const embedServer = await createServer((_req, res) => {
      callCount += 1;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          data: [
            { index: 0, embedding: callCount === 1 ? [1, 2, 3] : [1, 2, 3, 4] },
          ],
        }),
      );
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
    await expect(remote.embed("second")).rejects.toThrow(
      "Embedding dimension mismatch",
    );
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
      res.end(
        JSON.stringify({
          results: (body.documents as string[]).map(
            (_: string, index: number) => ({
              index,
              relevance_score: 0.9 - index * 0.1,
            }),
          ),
        }),
      );
    });
    serversToClose.push(embedServer.server, rerankServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: rerankServer.baseUrl,
      breakerCooldownMs: 50,
    });

    await expect(remote.embed("first")).rejects.toThrow("HTTP 500");
    await expect(remote.embed("second")).rejects.toThrow("HTTP 500");
    await expect(remote.embed("third")).rejects.toThrow(
      "circuit-breaker cooldown",
    );
    expect(embedHits).toBe(2);

    const rerank = await remote.rerank("query", [
      { file: "a.md", text: "doc-a" },
    ]);
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
      res.end(
        JSON.stringify({
          data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
        }),
      );
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
    await expect(remote.embed("third")).rejects.toThrow(
      "circuit-breaker cooldown",
    );

    await new Promise((resolve) => setTimeout(resolve, 30));
    const result = await remote.embed("fourth");
    expect(result?.embedding).toHaveLength(3);
    expect(callCount).toBe(3);
  });

  test("uses Authorization header when api key is configured", async () => {
    const embedServer = await createServer((_req, res, body) => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          data: (body.input as string[]).map((_: string, index: number) => ({
            index,
            embedding: [0.1, 0.2, 0.3],
          })),
        }),
      );
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
    expect(embedServer.requests[0]?.headers.authorization).toBe(
      "Bearer secret-token",
    );
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

    await expect(remote.embed("hello")).rejects.toThrow(
      "connect timeout after 20ms",
    );
  });

  test("enforces read timeout", async () => {
    const fetchStub = vi.fn(
      async () =>
        new Response(null, {
          status: 200,
        }),
    );

    const responseTextSpy = vi
      .spyOn(Response.prototype, "text")
      .mockImplementation(async () => {
        return await new Promise<string>(() => {});
      });

    vi.stubGlobal("fetch", fetchStub);

    const remote = new RemoteLLM({
      embedUrl: "http://example.com",
      rerankUrl: "http://example.com",
      connectTimeoutMs: 50,
      readTimeoutMs: 20,
    });

    await expect(remote.embed("hello")).rejects.toThrow(
      "read timeout after 20ms",
    );
    responseTextSpy.mockRestore();
  });

  test("generate calls chat completions and returns text", async () => {
    const chatServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          model: "gpt-5.4-mini",
          choices: [{ message: { content: "Generated response text" } }],
        }),
      );
    });
    serversToClose.push(chatServer.server);

    const remote = new RemoteLLM({
      embedUrl: chatServer.baseUrl,
      rerankUrl: chatServer.baseUrl,
    });

    const result = await remote.generate("What is 2+2?");
    expect(result?.text).toBe("Generated response text");
    expect(result?.model).toBe("gpt-5.4-mini");
    expect(result?.done).toBe(true);
  });

  test("generate and expandQuery use genUrl when set, not embedUrl", async () => {
    const embedRequests: string[] = [];
    const genRequests: string[] = [];

    const embedServer = await createServer((req, res) => {
      embedRequests.push(req.url ?? "");
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ data: [{ embedding: [0.1] }], model: "m" }));
    });
    const genServer = await createServer((req, res) => {
      genRequests.push(req.url ?? "");
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          choices: [{ message: { content: "vec: test query" } }],
        }),
      );
    });
    serversToClose.push(embedServer.server, genServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: embedServer.baseUrl,
      genUrl: genServer.baseUrl,
    });

    await remote.embed("hello");
    await remote.expandQuery("my query");

    expect(embedRequests.some((u) => u.includes("/v1/embeddings"))).toBe(true);
    expect(genRequests.some((u) => u.includes("/v1/chat/completions"))).toBe(
      true,
    );
    // embedServer must NOT have received a chat completions request
    expect(embedRequests.some((u) => u.includes("/v1/chat/completions"))).toBe(
      false,
    );
  });

  test("generate defaults to embedUrl when genUrl not set (backward compat)", async () => {
    const requests: string[] = [];
    const server = await createServer((req, res) => {
      requests.push(req.url ?? "");
      res.setHeader("content-type", "application/json");
      if (req.url?.includes("/v1/chat/completions")) {
        res.end(
          JSON.stringify({
            choices: [{ message: { content: "ok" } }],
          }),
        );
      } else {
        res.end(JSON.stringify({ data: [{ embedding: [0.1] }], model: "m" }));
      }
    });
    serversToClose.push(server.server);

    const remote = new RemoteLLM({
      embedUrl: server.baseUrl,
      rerankUrl: server.baseUrl,
      // no genUrl — should default to embedUrl
    });

    await remote.embed("hello");
    await remote.generate("hi");

    // Both embed and chat go to the same server
    expect(requests.some((u) => u.includes("/v1/embeddings"))).toBe(true);
    expect(requests.some((u) => u.includes("/v1/chat/completions"))).toBe(true);
  });

  test("modelExists logs a warning when /v1/models is unreachable", async () => {
    const stderrSpy = vi
      .spyOn(process.stderr, "write")
      .mockImplementation(() => true);

    const remote = new RemoteLLM({
      embedUrl: "http://127.0.0.1:1", // unreachable
      rerankUrl: "http://127.0.0.1:1",
      connectTimeoutMs: 50,
      readTimeoutMs: 50,
    });

    const result = await remote.modelExists("some-model");
    expect(result.exists).toBe(true); // fail-open
    const logged = stderrSpy.mock.calls.map((c) => String(c[0])).join("");
    expect(logged).toMatch(/modelExists.*fail-open|assuming available/);
    stderrSpy.mockRestore();
  });

  test("legacy prefix stripping logs at debug level", async () => {
    const stderrSpy = vi
      .spyOn(process.stderr, "write")
      .mockImplementation(() => true);

    const embedServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ data: [{ embedding: [0.1] }], model: "m" }));
    });
    serversToClose.push(embedServer.server);

    const remote = new RemoteLLM({
      embedUrl: embedServer.baseUrl,
      rerankUrl: embedServer.baseUrl,
      embedModel: "Qwen3-Embedding-8B",
    });

    // Send text that has a Qwen query prefix (simulating migration scenario)
    await remote.embed(
      "Instruct: Retrieve relevant documents for the given query\nQuery: test",
      { isQuery: true },
    );

    const logged = stderrSpy.mock.calls.map((c) => String(c[0])).join("");
    expect(logged).toMatch(/\[debug\].*stripped.*prefix/i);
    stderrSpy.mockRestore();
  });

  test("expandQuery parses typed lines from chat completions", async () => {
    const chatServer = await createServer((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "lex: machine learning search\nvec: neural network retrieval\nhyde: A document about deep learning algorithms",
              },
            },
          ],
        }),
      );
    });
    serversToClose.push(chatServer.server);

    const remote = new RemoteLLM({
      embedUrl: chatServer.baseUrl,
      rerankUrl: chatServer.baseUrl,
    });

    const results = await remote.expandQuery("deep learning");
    expect(results).toHaveLength(3);
    expect(results[0]).toEqual({
      type: "lex",
      text: "machine learning search",
    });
    expect(results[1]).toEqual({
      type: "vec",
      text: "neural network retrieval",
    });
    expect(results[2]).toEqual({
      type: "hyde",
      text: "A document about deep learning algorithms",
    });
  });

  test("expandQuery falls back to lex+vec+hyde when chat fails", async () => {
    const chatServer = await createServer((_req, res) => {
      res.statusCode = 500;
      res.end("{}");
    });
    serversToClose.push(chatServer.server);

    const remote = new RemoteLLM({
      embedUrl: chatServer.baseUrl,
      rerankUrl: chatServer.baseUrl,
    });

    const results = await remote.expandQuery("search term");
    expect(results.some((q) => q.type === "hyde")).toBe(true);
    expect(results.some((q) => q.type === "vec")).toBe(true);
    expect(results.some((q) => q.type === "lex")).toBe(true);
  });

  test("tokenize and countTokens use ~4 chars/token approximation", async () => {
    const remote = new RemoteLLM({
      embedUrl: "http://localhost:1",
      rerankUrl: "http://localhost:1",
    });

    const text = "hello world test"; // 16 chars → 4 tokens
    const tokens = await remote.tokenize(text);
    expect(tokens).toHaveLength(4);
    expect(await remote.countTokens(text)).toBe(4);
    // detokenize returns a character-length approximation, not empty string
    const detoken = await remote.detokenize(tokens);
    expect(detoken.length).toBe(tokens.length * 4); // CHARS_PER_TOKEN = 4
    expect(detoken.length).toBeGreaterThan(0);
  });

  test("chat circuit breaker is independent of embed circuit", async () => {
    let chatHits = 0;
    // One server that succeeds for /v1/embeddings but fails for /v1/chat/completions
    const mixedServer = await createServer((req, res) => {
      if (req.url?.includes("/v1/chat/completions")) {
        chatHits += 1;
        res.statusCode = 500;
        res.end("{}");
      } else {
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3] }],
            model: "m",
          }),
        );
      }
    });
    serversToClose.push(mixedServer.server);

    const remote = new RemoteLLM({
      embedUrl: mixedServer.baseUrl,
      rerankUrl: mixedServer.baseUrl,
      breakerCooldownMs: 60_000,
    });

    // Trip the chat circuit via two failures
    await expect(remote.generate("p1")).rejects.toThrow();
    await expect(remote.generate("p2")).rejects.toThrow();
    await expect(remote.generate("p3")).rejects.toThrow(
      "circuit-breaker cooldown",
    );
    expect(chatHits).toBe(2);

    // Embed circuit is still closed — embeddings work fine
    const result = await remote.embed("hello");
    expect(result?.embedding).toHaveLength(3);
  });
});

describe("HybridLLM", () => {
  test("routes all operations to remote — embed, rerank, generate, and expansion", async () => {
    const calls = {
      remoteEmbed: 0,
      remoteRerank: 0,
      remoteGenerate: 0,
      remoteExpand: 0,
    };

    const local = createLocalLLM({});

    const remote = createLocalLLM({
      isRemote: true,
      embed: async () => {
        calls.remoteEmbed += 1;
        return { embedding: [1, 2, 3], model: "remote-embed" };
      },
      embedBatch: async (texts) => {
        calls.remoteEmbed += texts.length;
        return texts.map(() => ({
          embedding: [1, 2, 3],
          model: "remote-embed",
        }));
      },
      rerank: async (_query, documents) => {
        calls.remoteRerank += 1;
        return {
          results: documents.map((document, index) => ({
            file: document.file,
            score: 1 - index * 0.1,
            index,
          })),
          model: "remote-rerank",
        };
      },
      generate: async () => {
        calls.remoteGenerate += 1;
        return { text: "remote-generated", model: "remote", done: true };
      },
      expandQuery: async () => {
        calls.remoteExpand += 1;
        return [{ type: "vec", text: "expanded-remote" }];
      },
    });

    const hybrid = new HybridLLM(local, remote);

    expect((await hybrid.embed("hello"))?.model).toBe("remote-embed");
    expect((await hybrid.generate("prompt"))?.text).toBe("remote-generated");
    expect((await hybrid.expandQuery("query"))[0]?.text).toBe(
      "expanded-remote",
    );
    expect(
      (await hybrid.rerank("query", [{ file: "a.md", text: "doc-a" }])).model,
    ).toBe("remote-rerank");

    expect(calls.remoteEmbed).toBe(1);
    expect(calls.remoteRerank).toBe(1);
    expect(calls.remoteGenerate).toBe(1);
    expect(calls.remoteExpand).toBe(1);
  });

  test("tokenize/detokenize delegate to remote without as-any casts", async () => {
    const remote = createLocalLLM({
      isRemote: true,
      tokenize: async (text: string) =>
        Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => i),
      countTokens: async (text: string) => Math.ceil(text.length / 4),
      detokenize: async (tokens: readonly unknown[]) =>
        "x".repeat((tokens as number[]).length * 4),
    });

    const hybrid = new HybridLLM(createLocalLLM({}), remote);

    const tokens = await hybrid.tokenize("hello world"); // 11 chars → 3 tokens
    expect(tokens).toHaveLength(3);
    expect(await hybrid.countTokens("hello world")).toBe(3);
    const detoken = await hybrid.detokenize(tokens);
    expect(detoken.length).toBe(12); // 3 * 4
    expect(detoken.length).toBeGreaterThan(0);
  });
});
