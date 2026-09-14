/**
 * llm-openai.test.ts - Unit tests for the OpenAI-compatible backend.
 *
 * Every HTTP call goes through a stubbed global fetch: the tests check the
 * request shapes an OpenAI-compatible server must receive and how responses
 * are mapped back, without a server.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import {
  OpenAiLLM,
  LlamaCpp,
  createLlm,
  isOpenAiModelUri,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  parseExpansionLines,
} from "../src/llm.js";

type Call = { url: string; init: RequestInit; body: any };

function stubFetch(handler: (call: Call) => unknown | Promise<unknown>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call: Call = { url, init, body: init.body ? JSON.parse(String(init.body)) : undefined };
    calls.push(call);
    const result = await handler(call);
    if (result instanceof Response) return result;
    return new Response(JSON.stringify(result), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

const config = {
  baseUrl: "http://127.0.0.1:4100/v1/",
  apiKey: "test-key",
  embedModel: "openai:text-embedding-v4",
  generateModel: "openai:general",
  rerankModel: "openai:auto",
};

describe("OpenAiLLM", () => {
  const savedEnv = { ...process.env };

  beforeEach(() => {
    delete process.env.QMD_OPENAI_BASE_URL;
    delete process.env.QMD_OPENAI_API_KEY;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    process.env = { ...savedEnv };
  });

  test("requires an endpoint and openai: model URIs", () => {
    expect(() => new OpenAiLLM({ ...config, baseUrl: undefined })).toThrow(/QMD_OPENAI_BASE_URL/);
    expect(() => new OpenAiLLM({ ...config, rerankModel: "hf:org/repo/file.gguf" })).toThrow(/openai:<model id>/);
    process.env.QMD_OPENAI_BASE_URL = "http://env-host/v1";
    expect(new OpenAiLLM({ ...config, baseUrl: undefined }).endpoint).toBe("http://env-host/v1");
  });

  test("embedBatch posts /embeddings in batches of 10 with the bare model id and bearer auth", async () => {
    const calls = stubFetch(({ body }) => ({
      data: body.input.map((text: string, index: number) => ({ index, embedding: [text.length, 1, 2] })),
    }));
    const llm = new OpenAiLLM(config);
    const texts = Array.from({ length: 23 }, (_, i) => "x".repeat(i + 1));

    const results = await llm.embedBatch(texts);

    expect(calls).toHaveLength(3);
    expect(calls.map(c => c.body.input.length).sort()).toEqual([10, 10, 3]);
    expect(calls[0]!.url).toBe("http://127.0.0.1:4100/v1/embeddings");
    expect(calls[0]!.init.method).toBe("POST");
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe("Bearer test-key");
    expect(calls[0]!.body.model).toBe("text-embedding-v4");
    // Every vector lands at the position of its text, whichever batch answered first.
    results.forEach((result, i) => {
      expect(result?.embedding[0]).toBe(i + 1);
      expect(result?.model).toBe("openai:text-embedding-v4");
    });
  });

  test("embed returns the single vector and surfaces HTTP errors with the body", async () => {
    stubFetch(() => ({ data: [{ index: 0, embedding: [0.1, 0.2] }] }));
    const llm = new OpenAiLLM(config);
    expect((await llm.embed("hello"))?.embedding).toEqual([0.1, 0.2]);

    stubFetch(() => new Response(JSON.stringify({ error: { message: "embedding model 'auto' not found" } }), { status: 404 }));
    await expect(llm.embed("hello")).rejects.toThrow(/\/v1\/embeddings answered 404: .*not found/);
  });

  test("QMD_OPENAI_API_KEY wins over the configured key; no header without either", async () => {
    let calls = stubFetch(() => ({ data: [] }));
    process.env.QMD_OPENAI_API_KEY = "env-key";
    await new OpenAiLLM(config).embedBatch(["a"]);
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe("Bearer env-key");

    delete process.env.QMD_OPENAI_API_KEY;
    calls = stubFetch(() => ({ data: [] }));
    await new OpenAiLLM({ ...config, apiKey: undefined }).embedBatch(["a"]);
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBeUndefined();
  });

  test("generate posts a non-streaming chat completion merged with generate params", async () => {
    const calls = stubFetch(() => ({ choices: [{ message: { role: "assistant", content: "hi there" } }] }));
    const llm = new OpenAiLLM({ ...config, generateParams: { reasoning_effort: "none" } });

    const result = await llm.generate("say hi", { maxTokens: 20, temperature: 0.1 });

    expect(result).toEqual({ text: "hi there", model: "openai:general", done: true });
    expect(calls[0]!.url).toBe("http://127.0.0.1:4100/v1/chat/completions");
    expect(calls[0]!.body).toEqual({
      model: "general",
      messages: [{ role: "user", content: "say hi" }],
      max_tokens: 20,
      temperature: 0.1,
      stream: false,
      reasoning_effort: "none",
    });
  });

  test("expandQuery parses lex/vec/hyde lines, drops thinking, and falls back on error", async () => {
    stubFetch(() => ({ choices: [{ message: { content: "<think>lex: nothing relevant</think>lex: splitroute engine\nvec: which engine did splitroute choose\nhyde: The splitroute engine is option B.\nignored line" } }] }));
    const llm = new OpenAiLLM(config);

    expect(await llm.expandQuery("splitroute engine option")).toEqual([
      { type: "lex", text: "splitroute engine" },
      { type: "vec", text: "which engine did splitroute choose" },
      { type: "hyde", text: "The splitroute engine is option B." },
    ]);

    stubFetch(() => new Response("boom", { status: 500 }));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      expect(await llm.expandQuery("splitroute", { includeLexical: false })).toEqual([{ type: "vec", text: "splitroute" }]);
    } finally {
      errorSpy.mockRestore();
    }
  });

  test("rerank posts /rerank once per distinct text and maps scores back to every file", async () => {
    const calls = stubFetch(({ body }) => ({
      results: body.documents.map((_: string, index: number) => ({ index, relevance_score: index === 1 ? 0.9 : 0.1 })),
    }));
    const llm = new OpenAiLLM(config);

    const result = await llm.rerank("q", [
      { file: "a.md", text: "same" },
      { file: "b.md", text: "best" },
      { file: "c.md", text: "same" },
    ]);

    expect(calls[0]!.url).toBe("http://127.0.0.1:4100/v1/rerank");
    expect(calls[0]!.body).toEqual({ model: "auto", query: "q", documents: ["same", "best"] });
    expect(result.model).toBe("openai:auto");
    expect(result.results).toEqual([
      { file: "b.md", score: 0.9, index: 1 },
      { file: "a.md", score: 0.1, index: 0 },
      { file: "c.md", score: 0.1, index: 2 },
    ]);
    expect(await llm.rerank("q", [])).toEqual({ results: [], model: "openai:auto" });
  });

  test("listModels GETs /models; modelExists and dispose are trivial", async () => {
    const calls = stubFetch(() => ({ data: [{ id: "general" }, { id: "plans" }] }));
    const llm = new OpenAiLLM(config);
    expect(await llm.listModels()).toEqual(["general", "plans"]);
    expect(calls[0]!.init.method).toBe("GET");
    expect(await llm.modelExists("openai:general")).toEqual({ name: "openai:general", exists: true });
    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(llm.tokenize).toBeUndefined();
  });
});

describe("openai: model URIs", () => {
  test("embedding text is sent raw, unlike the gemma default", () => {
    expect(isOpenAiModelUri("openai:text-embedding-v4")).toBe(true);
    expect(isOpenAiModelUri("hf:org/repo/file.gguf")).toBe(false);
    expect(formatQueryForEmbedding("q", "openai:text-embedding-v4")).toBe("q");
    expect(formatDocForEmbedding("body", "Title", "openai:text-embedding-v4")).toBe("Title\nbody");
    expect(formatQueryForEmbedding("q", "hf:ggml-org/embeddinggemma-300M-GGUF/x.gguf")).toBe("task: search result | query: q");
  });

  test("createLlm picks the backend from the URIs and refuses a mix", () => {
    expect(createLlm({ embed: "openai:e", generate: "openai:g", rerank: "openai:r", openaiBaseUrl: "http://h/v1" })).toBeInstanceOf(OpenAiLLM);
    expect(createLlm({})).toBeInstanceOf(LlamaCpp);
    expect(() => createLlm({ embed: "openai:e", openaiBaseUrl: "http://h/v1" })).toThrow(/all be openai: URIs or all local/);
  });

  test("parseExpansionLines keeps only typed lines that mention the query", () => {
    expect(parseExpansionLines("lex: foo bar\nvec: unrelated\nhyde: about foo", "foo", true)).toEqual([
      { type: "lex", text: "foo bar" },
      { type: "hyde", text: "about foo" },
    ]);
    expect(parseExpansionLines("", "foo", false)).toEqual([
      { type: "hyde", text: "Information about foo" },
      { type: "vec", text: "foo" },
    ]);
  });
});
