import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  OpenRouterLLM,
  getDefaultLLM,
  getDefaultLLMProvider,
  disposeDefaultLLM,
  resetDefaultLLMForTests,
} from "./llm.js";

function jsonResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const originalFetch = globalThis.fetch;

describe("OpenRouter provider", () => {
  beforeEach(() => {
    delete process.env.QMD_LLM_PROVIDER;
    delete process.env.QMD_OPENROUTER_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.QMD_OPENROUTER_API_KEY_FILE;
    delete process.env.QMD_OPENROUTER_BASE_URL;
    delete process.env.QMD_OPENROUTER_EMBED_MODEL;
    delete process.env.QMD_OPENROUTER_GENERATE_MODEL;
    delete process.env.QMD_OPENROUTER_RERANK_MODEL;
    resetDefaultLLMForTests();
  });

  afterEach(async () => {
    globalThis.fetch = originalFetch;
    await disposeDefaultLLM();
    resetDefaultLLMForTests();
  });

  test("uses OpenRouter provider when QMD_LLM_PROVIDER=openrouter", () => {
    process.env.QMD_LLM_PROVIDER = "openrouter";
    process.env.QMD_OPENROUTER_API_KEY = "test-key";

    const llm = getDefaultLLM();

    expect(getDefaultLLMProvider()).toBe("openrouter");
    expect(llm).toBeInstanceOf(OpenRouterLLM);
  });

  test("prints remote notice only once per process", () => {
    process.env.QMD_LLM_PROVIDER = "openrouter";
    process.env.QMD_OPENROUTER_API_KEY = "test-key";

    const stderrAny = process.stderr as any;
    const originalWrite = stderrAny.write;
    const writes: string[] = [];
    stderrAny.write = (chunk: any) => {
      writes.push(String(chunk));
      return true;
    };

    try {
      getDefaultLLM();
      getDefaultLLM();
    } finally {
      stderrAny.write = originalWrite;
    }

    const notices = writes.filter(line => line.includes("OpenRouter"));
    expect(notices.length).toBe(1);
  });

  test("embed sends OpenRouter embeddings request", async () => {
    const calls: Array<{ url: string; body: any; headers: Record<string, string> }> = [];
    globalThis.fetch = async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const parsedBody = JSON.parse(String(init?.body || "{}"));
      calls.push({
        url: String(url),
        body: parsedBody,
        headers: init?.headers as Record<string, string>,
      });
      return jsonResponse({
        data: [{ index: 0, embedding: [0.5, 0.25, -0.1] }],
      });
    };

    const llm = new OpenRouterLLM({
      apiKey: "abc123",
      baseUrl: "https://openrouter.ai/api/v1",
      embedModel: "openai/text-embedding-3-small",
    });

    const result = await llm.embed("hello world");

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([0.5, 0.25, -0.1]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("https://openrouter.ai/api/v1/embeddings");
    expect(calls[0]!.body.model).toBe("openai/text-embedding-3-small");
    expect(calls[0]!.body.input).toBe("hello world");
    expect(calls[0]!.headers.Authorization).toBe("Bearer abc123");
  });

  test("embedBatch maps embeddings by index order", async () => {
    globalThis.fetch = async () => jsonResponse({
      data: [
        { index: 1, embedding: [0, 1] },
        { index: 0, embedding: [1, 0] },
      ],
    });

    const llm = new OpenRouterLLM({ apiKey: "test-key" });
    const results = await llm.embedBatch(["first", "second"]);

    expect(results).toHaveLength(2);
    expect(results[0]!.embedding).toEqual([1, 0]);
    expect(results[1]!.embedding).toEqual([0, 1]);
  });

  test("expandQuery parses typed query lines and filters lexical when disabled", async () => {
    globalThis.fetch = async () => jsonResponse({
      choices: [
        {
          message: {
            content: "lex: deploy auth service\nvec: deploy authentication stack\nhyde: documentation for deploying auth service",
          },
        },
      ],
    });

    const llm = new OpenRouterLLM({ apiKey: "test-key" });
    const queryables = await llm.expandQuery("deploy auth service", { includeLexical: false });

    expect(queryables.some(q => q.type === "lex")).toBe(false);
    expect(queryables.some(q => q.type === "vec")).toBe(true);
    expect(queryables.some(q => q.type === "hyde")).toBe(true);
  });

  test("rerank uses embedding similarity and sorts descending", async () => {
    globalThis.fetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const body = JSON.parse(String(init?.body || "{}"));

      if (typeof body.input === "string") {
        return jsonResponse({
          data: [{ index: 0, embedding: [1, 0] }],
        });
      }

      return jsonResponse({
        data: [
          { index: 0, embedding: [0.9, 0] },
          { index: 1, embedding: [0, 1] },
        ],
      });
    };

    const llm = new OpenRouterLLM({ apiKey: "test-key", rerankModel: "openai/text-embedding-3-small" });
    const reranked = await llm.rerank("auth query", [
      { file: "a.md", text: "authentication docs" },
      { file: "b.md", text: "gardening notes" },
    ]);

    expect(reranked.results).toHaveLength(2);
    expect(reranked.results[0]!.file).toBe("a.md");
    expect(reranked.results[0]!.score).toBeGreaterThan(reranked.results[1]!.score);
  });

  test("loads API key from file when env var is not set", async () => {
    const tempDir = await mkdtemp(join(tmpdir(), "qmd-openrouter-test-"));
    const keyFile = join(tempDir, "openrouter.key");
    await writeFile(keyFile, "file-key-123\n", "utf-8");

    try {
      const authHeaders: string[] = [];
      globalThis.fetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
        const headers = init?.headers as Record<string, string>;
        authHeaders.push(headers.Authorization || "");
        return jsonResponse({
          data: [{ index: 0, embedding: [0.1, 0.2] }],
        });
      };

      const llm = new OpenRouterLLM({ apiKeyFile: keyFile });
      await llm.embed("test");

      expect(authHeaders[0]).toBe("Bearer file-key-123");
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  });
});
