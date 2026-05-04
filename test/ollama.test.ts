/**
 * ollama.test.ts - Unit tests for the Ollama LLM provider
 *
 * Tests the OllamaLLM class from src/ollama.ts using mocked fetch.
 * No actual Ollama server required.
 *
 * Run with: npx vitest run test/ollama.test.ts
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { OllamaLLM, createOllamaLLM } from "../src/ollama.js";
import { getDefaultLLM, LlamaCpp } from "../src/llm.js";

// =============================================================================
// Helpers
// =============================================================================

/** Create a mock Response object for fetch */
function mockResponse(
  body: unknown,
  status = 200,
  ok = true,
): Response {
  return {
    ok,
    status,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(typeof body === "string" ? body : JSON.stringify(body)),
  } as unknown as Response;
}

// =============================================================================
// Configuration Tests
// =============================================================================

describe("OllamaLLM configuration", () => {
  test("uses default values when no config or env vars provided", () => {
    const prevBaseUrl = process.env.QMD_OLLAMA_BASE_URL;
    const prevEmbed = process.env.QMD_OLLAMA_EMBED_MODEL;
    const prevGenerate = process.env.QMD_OLLAMA_GENERATE_MODEL;
    const prevRerank = process.env.QMD_OLLAMA_RERANK_MODEL;
    delete process.env.QMD_OLLAMA_BASE_URL;
    delete process.env.QMD_OLLAMA_EMBED_MODEL;
    delete process.env.QMD_OLLAMA_GENERATE_MODEL;
    delete process.env.QMD_OLLAMA_RERANK_MODEL;

    try {
      const llm = new OllamaLLM();
      expect((llm as any)._baseUrl).toBe("http://localhost:11434");
      expect(llm.embedModelId).toBe("nomic-embed-text");
      expect(llm.generateModelId).toBe("qwen3:1.7b");
      expect(llm.rerankModelId).toBe("qwen3:0.6b");
    } finally {
      if (prevBaseUrl !== undefined) process.env.QMD_OLLAMA_BASE_URL = prevBaseUrl;
      else delete process.env.QMD_OLLAMA_BASE_URL;
      if (prevEmbed !== undefined) process.env.QMD_OLLAMA_EMBED_MODEL = prevEmbed;
      else delete process.env.QMD_OLLAMA_EMBED_MODEL;
      if (prevGenerate !== undefined) process.env.QMD_OLLAMA_GENERATE_MODEL = prevGenerate;
      else delete process.env.QMD_OLLAMA_GENERATE_MODEL;
      if (prevRerank !== undefined) process.env.QMD_OLLAMA_RERANK_MODEL = prevRerank;
      else delete process.env.QMD_OLLAMA_RERANK_MODEL;
    }
  });

  test("reads configuration from env vars (QMD_OLLAMA_BASE_URL, etc.)", () => {
    const prevBaseUrl = process.env.QMD_OLLAMA_BASE_URL;
    const prevEmbed = process.env.QMD_OLLAMA_EMBED_MODEL;
    const prevGenerate = process.env.QMD_OLLAMA_GENERATE_MODEL;
    const prevRerank = process.env.QMD_OLLAMA_RERANK_MODEL;

    process.env.QMD_OLLAMA_BASE_URL = "http://my-ollama:12345";
    process.env.QMD_OLLAMA_EMBED_MODEL = "mxbai-embed-large";
    process.env.QMD_OLLAMA_GENERATE_MODEL = "llama3:8b";
    process.env.QMD_OLLAMA_RERANK_MODEL = "reranker-v2";

    try {
      const llm = new OllamaLLM();
      expect((llm as any)._baseUrl).toBe("http://my-ollama:12345");
      expect(llm.embedModelId).toBe("mxbai-embed-large");
      expect(llm.generateModelId).toBe("llama3:8b");
      expect(llm.rerankModelId).toBe("reranker-v2");
    } finally {
      if (prevBaseUrl !== undefined) process.env.QMD_OLLAMA_BASE_URL = prevBaseUrl;
      else delete process.env.QMD_OLLAMA_BASE_URL;
      if (prevEmbed !== undefined) process.env.QMD_OLLAMA_EMBED_MODEL = prevEmbed;
      else delete process.env.QMD_OLLAMA_EMBED_MODEL;
      if (prevGenerate !== undefined) process.env.QMD_OLLAMA_GENERATE_MODEL = prevGenerate;
      else delete process.env.QMD_OLLAMA_GENERATE_MODEL;
      if (prevRerank !== undefined) process.env.QMD_OLLAMA_RERANK_MODEL = prevRerank;
      else delete process.env.QMD_OLLAMA_RERANK_MODEL;
    }
  });

  test("constructor config overrides env vars", () => {
    const prevBaseUrl = process.env.QMD_OLLAMA_BASE_URL;
    process.env.QMD_OLLAMA_BASE_URL = "http://env-url:11434";

    try {
      const llm = new OllamaLLM({
        baseUrl: "http://config-url:9999",
        embedModel: "config-embed",
        generateModel: "config-gen",
        rerankModel: "config-rerank",
      });
      expect((llm as any)._baseUrl).toBe("http://config-url:9999");
      expect(llm.embedModelId).toBe("config-embed");
      expect(llm.generateModelId).toBe("config-gen");
      expect(llm.rerankModelId).toBe("config-rerank");
    } finally {
      if (prevBaseUrl !== undefined) process.env.QMD_OLLAMA_BASE_URL = prevBaseUrl;
      else delete process.env.QMD_OLLAMA_BASE_URL;
    }
  });

  test("rerank model uses its own default (qwen3:0.6b), not generate model", () => {
    const prevRerank = process.env.QMD_OLLAMA_RERANK_MODEL;
    delete process.env.QMD_OLLAMA_RERANK_MODEL;

    try {
      const llm = new OllamaLLM({ generateModel: "my-gen-model" });
      expect(llm.rerankModelId).toBe("qwen3:0.6b");
    } finally {
      if (prevRerank !== undefined) process.env.QMD_OLLAMA_RERANK_MODEL = prevRerank;
      else delete process.env.QMD_OLLAMA_RERANK_MODEL;
    }
  });

  test("createOllamaLLM factory creates OllamaLLM instance", () => {
    const llm = createOllamaLLM({ baseUrl: "http://test:1234" });
    expect(llm).toBeInstanceOf(OllamaLLM);
    expect((llm as any)._baseUrl).toBe("http://test:1234");
  });
});

// =============================================================================
// embed() Tests
// =============================================================================

describe("OllamaLLM.embed()", () => {
  let llm: OllamaLLM;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    llm = new OllamaLLM({ baseUrl: "http://localhost:11434" });
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        embeddings: [[0.1, 0.2, 0.3, -0.4, 0.5]],
        model: "nomic-embed-text",
      }),
    );
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  test("calls Ollama /api/embed endpoint with correct payload", async () => {
    await llm.embed("hello world");

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("http://localhost:11434/api/embed");
    expect((init as RequestInit).method).toBe("POST");

    const body = JSON.parse((init as RequestInit).body as string) as {
      model: string;
      input: string;
    };
    expect(body.model).toBe("nomic-embed-text");
    expect(body.input).toBe("hello world");
  });

  test("returns embedding vector and model name", async () => {
    const result = await llm.embed("test text");

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([0.1, 0.2, 0.3, -0.4, 0.5]);
    expect(result!.model).toBe("nomic-embed-text");
  });

  test("uses model from options when provided", async () => {
    await llm.embed("test", { model: "custom-model" });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ) as { model: string };
    expect(body.model).toBe("custom-model");
  });

  test("returns null on non-OK response instead of throwing", async () => {
    fetchSpy.mockResolvedValue(
      mockResponse("model not found", 404, false),
    );

    const result = await llm.embed("test");
    expect(result).toBeNull();
  });

  test("returns null when response has no embeddings", async () => {
    fetchSpy.mockResolvedValue(
      mockResponse({ embeddings: [], model: "test" }),
    );

    const result = await llm.embed("test");
    expect(result).toBeNull();
  });

  test("returns null on network error instead of throwing", async () => {
    fetchSpy.mockRejectedValue(new Error("fetch failed"));

    const result = await llm.embed("test");
    expect(result).toBeNull();
  });
});

// =============================================================================
// generate() Tests
// =============================================================================

describe("OllamaLLM.generate()", () => {
  let llm: OllamaLLM;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    llm = new OllamaLLM({ baseUrl: "http://localhost:11434" });
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: "Paris is the capital of France.",
        model: "qwen3:1.7b",
        done: true,
      }),
    );
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  test("calls Ollama /api/generate endpoint with correct payload", async () => {
    await llm.generate("What is the capital of France?");

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("http://localhost:11434/api/generate");
    expect((init as RequestInit).method).toBe("POST");

    const body = JSON.parse((init as RequestInit).body as string) as {
      model: string;
      prompt: string;
      stream: boolean;
    };
    expect(body.model).toBe("qwen3:1.7b");
    expect(body.prompt).toBe("What is the capital of France?");
    expect(body.stream).toBe(false);
  });

  test("returns generated text and model name", async () => {
    const result = await llm.generate("test prompt");

    expect(result).not.toBeNull();
    expect(result!.text).toBe("Paris is the capital of France.");
    expect(result!.model).toBe("qwen3:1.7b");
    expect(result!.done).toBe(true);
  });

  test("passes maxTokens and temperature in options", async () => {
    await llm.generate("test", { maxTokens: 100, temperature: 0.3 });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ) as { options: { num_predict: number; temperature: number } };
    expect(body.options.num_predict).toBe(100);
    expect(body.options.temperature).toBe(0.3);
  });

  test("uses default maxTokens=150 and temperature=0.7", async () => {
    await llm.generate("test");

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ) as { options: { num_predict: number; temperature: number } };
    expect(body.options.num_predict).toBe(150);
    expect(body.options.temperature).toBe(0.7);
  });

  test("returns null on non-OK response instead of throwing", async () => {
    fetchSpy.mockResolvedValue(
      mockResponse("internal error", 500, false),
    );

    const result = await llm.generate("test");
    expect(result).toBeNull();
  });

  test("returns null on network error instead of throwing", async () => {
    fetchSpy.mockRejectedValue(new TypeError("Failed to fetch"));

    const result = await llm.generate("test");
    expect(result).toBeNull();
  });
});

// =============================================================================
// expandQuery() Tests
// =============================================================================

describe("OllamaLLM.expandQuery()", () => {
  let llm: OllamaLLM;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    llm = new OllamaLLM({ baseUrl: "http://localhost:11434" });
  });

  afterEach(() => {
    if (fetchSpy) fetchSpy.mockRestore();
  });

  test("parses lex/vec/hyde lines from generated response", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: `lex: authentication setup guide
vec: how to configure authentication and user verification
hyde: This document describes the authentication setup process for projects.`,
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    const results = await llm.expandQuery("authentication setup");

    expect(results.length).toBeGreaterThanOrEqual(1);

    const lexEntries = results.filter((r) => r.type === "lex");
    const vecEntries = results.filter((r) => r.type === "vec");
    const hydeEntries = results.filter((r) => r.type === "hyde");

    // All three types should be present
    expect(lexEntries.length).toBeGreaterThanOrEqual(1);
    expect(vecEntries.length).toBeGreaterThanOrEqual(1);
    expect(hydeEntries.length).toBeGreaterThanOrEqual(1);
  });

  test("sends /no_think prefix in prompt to model", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: "vec: authentication",
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    await llm.expandQuery("auth");

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ) as { prompt: string };
    expect(body.prompt).toContain("/no_think");
    expect(body.prompt).toContain("auth");
  });

  test("excludes lexical queries when includeLexical is false", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: `lex: keyword variant
vec: semantic variant
hyde: hypothetical document`,
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    const results = await llm.expandQuery("test", { includeLexical: false });

    expect(results.every((r) => r.type !== "lex")).toBe(true);
  });

  test("falls back to fallbackQuery when generate returns nothing parseable", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: "I don't understand the query format at all.",
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    const results = await llm.expandQuery("xyzzy");

    // Fallback: hyde, lex, vec entries
    expect(results.length).toBeGreaterThanOrEqual(1);
    // Fallback always includes hyde, lex (default), and vec
    expect(results.some((r) => r.type === "vec")).toBe(true);
  });

  test("falls back when generate returns empty response", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: "",
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    const results = await llm.expandQuery("broken query");
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results.some((r) => r.type === "vec")).toBe(true);
  });

  test("falls back on network error gracefully", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("ECONNREFUSED"),
    );

    const results = await llm.expandQuery("test");
    // Returns fallback query entries, not a rejection
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results.some((r) => r.type === "vec")).toBe(true);
  });

  test("supports intent parameter in prompt", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        response: "vec: expanded authentication",
        model: "qwen3:1.7b",
        done: true,
      }),
    );

    await llm.expandQuery("auth", { intent: "setup" });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0]![1] as RequestInit).body as string,
    ) as { prompt: string };
    expect(body.prompt).toContain("Query intent: setup");
  });
});

// =============================================================================
// rerank() Tests
// =============================================================================

describe("OllamaLLM.rerank()", () => {
  let llm: OllamaLLM;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    llm = new OllamaLLM({ baseUrl: "http://localhost:11434" });
  });

  afterEach(() => {
    if (fetchSpy) fetchSpy.mockRestore();
  });

  test("returns results ordered by score (descending)", async () => {
    // Mock chat responses for each document (scoreDocument uses /api/chat)
    let callIndex = 0;
    const scores = ["relevance: 0.2", "relevance: 0.95", "relevance: 0.6"];
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      const score = scores[callIndex++] ?? "relevance: 0.0";
      return mockResponse({
        message: { role: "assistant", content: score },
        model: "qwen3:0.6b",
        done: true,
      });
    });

    const result = await llm.rerank("What is Python?", [
      { file: "java.md", text: "Java is a programming language." },
      { file: "python.md", text: "Python is a versatile programming language." },
      { file: "cooking.md", text: "How to bake a cake." },
    ]);

    expect(result.results).toHaveLength(3);
    // Highest score first
    expect(result.results[0]!.file).toBe("python.md");
    expect(result.results[0]!.score).toBeCloseTo(0.95, 2);
    expect(result.results[1]!.file).toBe("cooking.md");
    expect(result.results[1]!.score).toBeCloseTo(0.6, 1);
    expect(result.results[2]!.file).toBe("java.md");
    expect(result.results[2]!.score).toBeCloseTo(0.2, 1);

    // Verify scores are descending
    for (let i = 1; i < result.results.length; i++) {
      expect(result.results[i]!.score).toBeLessThanOrEqual(
        result.results[i - 1]!.score,
      );
    }
  });

  test("returns empty results for empty document list", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ response: "", model: "test", done: true }),
    );

    const result = await llm.rerank("query", []);
    expect(result.results).toHaveLength(0);
    expect(result.model).toBeTruthy();
    // Should not call fetch for empty documents
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("deduplicates identical document texts", async () => {
    let callCount = 0;
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      callCount++;
      return mockResponse({
        message: { role: "assistant", content: "relevance: 0.8" },
        model: "qwen3:0.6b",
        done: true,
      });
    });

    const result = await llm.rerank("test", [
      { file: "a.md", text: "shared chunk" },
      { file: "b.md", text: "shared chunk" },
    ]);

    expect(result.results).toHaveLength(2);
    // Both docs should have same score
    expect(result.results[0]!.score).toBeCloseTo(0.8, 1);
    expect(result.results[1]!.score).toBeCloseTo(0.8, 1);
    // Should only call fetch once for deduplicated text
    expect(callCount).toBe(1);
  });

  test("assigns 0.1 score for 'no' keyword when model response is unparseable", async () => {
    // The implementation has a fallback that detects "no" / "not relevant" / "irrelevant"
    // and returns 0.1. When no keywords match at all, it returns 0.0.
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        message: { role: "assistant", content: "The document is completely unrelated." },
        model: "qwen3:0.6b",
        done: true,
      }),
    );

    const result = await llm.rerank("test", [
      { file: "a.md", text: "document a" },
    ]);

    expect(result.results).toHaveLength(1);
    // "unrelated" does not match any keyword, so score is 0.0
    expect(result.results[0]!.score).toBe(0.0);
  });

  test("clamps scores to [0, 1] range", async () => {
    let callIndex = 0;
    const scores = ["relevance: 1.5", "relevance: -0.3"];
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      const score = scores[callIndex++] ?? "relevance: 0.0";
      return mockResponse({
        message: { role: "assistant", content: score },
        model: "qwen3:0.6b",
        done: true,
      });
    });

    const result = await llm.rerank("test", [
      { file: "a.md", text: "doc a" },
      { file: "b.md", text: "doc b" },
    ]);

    for (const r of result.results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  test("uses /api/chat endpoint for scoring", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        message: { role: "assistant", content: "relevance: 0.5" },
        model: "qwen3:0.6b",
        done: true,
      }),
    );

    await llm.rerank("test", [{ file: "a.md", text: "doc a" }]);

    const [url] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("http://localhost:11434/api/chat");
  });

  test("handles network errors gracefully, returning 0.0 scores", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("Network error"),
    );

    const result = await llm.rerank("query", [
      { file: "a.md", text: "text" },
    ]);

    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.score).toBe(0.0);
  });
});

// =============================================================================
// modelExists() Tests
// =============================================================================

describe("OllamaLLM.modelExists()", () => {
  let llm: OllamaLLM;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    llm = new OllamaLLM({ baseUrl: "http://localhost:11434" });
  });

  afterEach(() => {
    if (fetchSpy) fetchSpy.mockRestore();
  });

  test("returns exists:true when model is available", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        name: "nomic-embed-text",
        details: { parent_model: "", format: "gguf", family: "nomic", parameter_size: "137M" },
      }, 200, true),
    );

    const result = await llm.modelExists("nomic-embed-text");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("nomic-embed-text");

    // Verify correct endpoint
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("http://localhost:11434/api/show");
    expect((init as RequestInit).method).toBe("POST");
    const body = JSON.parse((init as RequestInit).body as string) as {
      name: string;
    };
    expect(body.name).toBe("nomic-embed-text");
  });

  test("returns exists:false when model is not found", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ error: "model not found" }, 404, false),
    );

    const result = await llm.modelExists("nonexistent-model");
    expect(result.exists).toBe(false);
    expect(result.name).toBe("nonexistent-model");
  });

  test("returns exists:false when Ollama is unreachable (network error)", async () => {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("ECONNREFUSED"),
    );

    const result = await llm.modelExists("any-model");
    expect(result.exists).toBe(false);
    expect(result.name).toBe("any-model");
  });
});

// =============================================================================
// dispose() Tests
// =============================================================================

describe("OllamaLLM.dispose()", () => {
  test("marks the instance as disposed", async () => {
    const llm = new OllamaLLM();
    await llm.dispose();
    expect((llm as any).disposed).toBe(true);
  });

  test("prevents further operations after disposal", async () => {
    const llm = new OllamaLLM();
    await llm.dispose();

    await expect(llm.embed("test")).rejects.toThrow("OllamaLLM has been disposed");
    await expect(llm.generate("test")).rejects.toThrow("OllamaLLM has been disposed");
    await expect(llm.expandQuery("test")).rejects.toThrow("OllamaLLM has been disposed");
    await expect(llm.rerank("q", [{ file: "a.md", text: "t" }])).rejects.toThrow(
      "OllamaLLM has been disposed",
    );
    await expect(llm.modelExists("test")).rejects.toThrow("OllamaLLM has been disposed");
  });
});

// =============================================================================
// Error Handling / Fallback Tests
// =============================================================================

describe("OllamaLLM error handling when Ollama is unavailable", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  test("embed() returns null on fetch rejection (doesn't throw)", async () => {
    fetchSpy.mockRejectedValue(new Error("fetch failed"));

    const llm = new OllamaLLM();
    const result = await llm.embed("test");
    expect(result).toBeNull();
  });

  test("generate() returns null on fetch rejection (doesn't throw)", async () => {
    fetchSpy.mockRejectedValue(new TypeError("Failed to fetch"));

    const llm = new OllamaLLM();
    const result = await llm.generate("test");
    expect(result).toBeNull();
  });

  test("expandQuery() returns fallback queries on network error", async () => {
    fetchSpy.mockRejectedValue(new Error("ECONNREFUSED"));

    const llm = new OllamaLLM();
    const results = await llm.expandQuery("test");

    // Should get fallback query entries, not a rejection
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results.some((r) => r.type === "vec")).toBe(true);
    expect(results.some((r) => r.type === "lex")).toBe(true);
    expect(results.some((r) => r.type === "hyde")).toBe(true);
  });

  test("rerank() returns 0.0 scores on network error", async () => {
    fetchSpy.mockRejectedValue(new Error("Network error"));

    const llm = new OllamaLLM();
    const result = await llm.rerank("query", [{ file: "a.md", text: "text" }]);

    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.score).toBe(0.0);
  });

  test("modelExists() returns exists:false on network error without throwing", async () => {
    fetchSpy.mockRejectedValue(new Error("Connection refused"));

    const llm = new OllamaLLM();
    const result = await llm.modelExists("any-model");

    // modelExists swallows errors and returns exists:false
    expect(result.exists).toBe(false);
    expect(result.name).toBe("any-model");
  });
});

// =============================================================================
// getDefaultLLM() Tests
// =============================================================================

describe("getDefaultLLM()", () => {
  test("returns OllamaLLM when QMD_LLM_BACKEND=ollama", () => {
    const prev = process.env.QMD_LLM_BACKEND;
    process.env.QMD_LLM_BACKEND = "ollama";

    try {
      const llm = getDefaultLLM();
      expect(llm).toBeInstanceOf(OllamaLLM);
    } finally {
      if (prev !== undefined) process.env.QMD_LLM_BACKEND = prev;
      else delete process.env.QMD_LLM_BACKEND;
    }
  });

  test("returns LlamaCpp when QMD_LLM_BACKEND is not ollama", () => {
    const prev = process.env.QMD_LLM_BACKEND;
    delete process.env.QMD_LLM_BACKEND;

    try {
      const llm = getDefaultLLM();
      expect(llm).toBeInstanceOf(LlamaCpp);
    } finally {
      if (prev !== undefined) process.env.QMD_LLM_BACKEND = prev;
      else delete process.env.QMD_LLM_BACKEND;
    }
  });

  test("returns LlamaCpp when QMD_LLM_BACKEND is empty string", () => {
    const prev = process.env.QMD_LLM_BACKEND;
    process.env.QMD_LLM_BACKEND = "";

    try {
      const llm = getDefaultLLM();
      expect(llm).toBeInstanceOf(LlamaCpp);
    } finally {
      if (prev !== undefined) process.env.QMD_LLM_BACKEND = prev;
      else delete process.env.QMD_LLM_BACKEND;
    }
  });

  test("returns LlamaCpp when QMD_LLM_BACKEND is some other value", () => {
    const prev = process.env.QMD_LLM_BACKEND;
    process.env.QMD_LLM_BACKEND = "llamacpp";

    try {
      const llm = getDefaultLLM();
      expect(llm).toBeInstanceOf(LlamaCpp);
    } finally {
      if (prev !== undefined) process.env.QMD_LLM_BACKEND = prev;
      else delete process.env.QMD_LLM_BACKEND;
    }
  });
});
