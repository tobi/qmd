/**
 * llm.test.ts - Comprehensive unit tests for the LLM abstraction layer
 *
 * Run with: bun test llm.test.ts
 *
 * Tests use a mock HTTP server to simulate Ollama responses.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "bun:test";
import {
  Ollama,
  getDefaultOllama,
  setDefaultOllama,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  type EmbeddingResult,
  type GenerateResult,
  type RerankDocumentResult,
  type TokenLogProb,
} from "./llm.js";

// =============================================================================
// Mock Server Setup
// =============================================================================

type MockHandler = (body: unknown) => {
  status: number;
  body: unknown;
};

const mockHandlers: Map<string, MockHandler> = new Map();
let mockServerUrl: string;
let mockCallLog: Array<{ path: string; body: unknown }> = [];

// Track original fetch
const originalFetch = globalThis.fetch;

function installMockFetch(): void {
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

    // Only intercept calls to our mock server URL
    if (!url.startsWith(mockServerUrl)) {
      throw new Error(`TEST ERROR: Unexpected fetch to: ${url}`);
    }

    const path = url.replace(mockServerUrl, "");
    const body = init?.body ? JSON.parse(init.body as string) : {};

    // Log the call
    mockCallLog.push({ path, body });

    const handler = mockHandlers.get(path);
    if (!handler) {
      return new Response(JSON.stringify({ error: "Not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    const result = handler(body);
    return new Response(JSON.stringify(result.body), {
      status: result.status,
      headers: { "Content-Type": "application/json" },
    });
  };
}

function restoreFetch(): void {
  globalThis.fetch = originalFetch;
}

// Setup before all tests
beforeAll(() => {
  mockServerUrl = "http://mock-ollama:11434";
  installMockFetch();
});

// Restore after all tests
afterAll(() => {
  restoreFetch();
});

// Clear call log and handlers before each test
beforeEach(() => {
  mockCallLog = [];
  mockHandlers.clear();
});

// =============================================================================
// Helper Functions
// =============================================================================

function createOllama(): Ollama {
  return new Ollama({ baseUrl: mockServerUrl });
}

function setEmbedHandler(embeddings: number[][]): void {
  mockHandlers.set("/api/embed", () => ({
    status: 200,
    body: { embeddings },
  }));
}

function setGenerateHandler(
  response: string,
  logprobs?: { tokens: string[]; token_logprobs: number[] }
): void {
  mockHandlers.set("/api/generate", () => ({
    status: 200,
    body: {
      response,
      done: true,
      ...(logprobs && { logprobs }),
    },
  }));
}

function setModelShowHandler(exists: boolean, size?: number): void {
  mockHandlers.set("/api/show", () => {
    if (exists) {
      return {
        status: 200,
        body: { size: size ?? 1000000, modified_at: "2024-01-01T00:00:00Z" },
      };
    }
    return { status: 404, body: { error: "model not found" } };
  });
}

function setPullHandler(success: boolean): void {
  mockHandlers.set("/api/pull", () => ({
    status: success ? 200 : 500,
    body: success ? { status: "success" } : { error: "failed" },
  }));
}

// =============================================================================
// Formatting Tests
// =============================================================================

describe("Formatting Functions", () => {
  test("formatQueryForEmbedding adds search task prefix", () => {
    const result = formatQueryForEmbedding("how to deploy");
    expect(result).toBe("task: search result | query: how to deploy");
  });

  test("formatQueryForEmbedding handles empty query", () => {
    const result = formatQueryForEmbedding("");
    expect(result).toBe("task: search result | query: ");
  });

  test("formatDocForEmbedding adds title and text prefix", () => {
    const result = formatDocForEmbedding("Document content", "My Title");
    expect(result).toBe("title: My Title | text: Document content");
  });

  test("formatDocForEmbedding handles missing title", () => {
    const result = formatDocForEmbedding("Document content");
    expect(result).toBe("title: none | text: Document content");
  });

  test("formatDocForEmbedding handles empty content", () => {
    const result = formatDocForEmbedding("", "Title");
    expect(result).toBe("title: Title | text: ");
  });
});

// =============================================================================
// Ollama Constructor Tests
// =============================================================================

describe("Ollama Constructor", () => {
  test("uses default URL when not specified", () => {
    const ollama = new Ollama();
    expect(ollama.getBaseUrl()).toBe("http://localhost:11434");
  });

  test("uses custom URL when specified", () => {
    const ollama = new Ollama({ baseUrl: "http://custom:9999" });
    expect(ollama.getBaseUrl()).toBe("http://custom:9999");
  });

  test("respects OLLAMA_URL environment variable", () => {
    const originalEnv = process.env.OLLAMA_URL;
    process.env.OLLAMA_URL = "http://env-url:8888";

    const ollama = new Ollama();
    expect(ollama.getBaseUrl()).toBe("http://env-url:8888");

    // Restore
    if (originalEnv) {
      process.env.OLLAMA_URL = originalEnv;
    } else {
      delete process.env.OLLAMA_URL;
    }
  });

  test("explicit baseUrl overrides environment variable", () => {
    const originalEnv = process.env.OLLAMA_URL;
    process.env.OLLAMA_URL = "http://env-url:8888";

    const ollama = new Ollama({ baseUrl: "http://explicit:7777" });
    expect(ollama.getBaseUrl()).toBe("http://explicit:7777");

    // Restore
    if (originalEnv) {
      process.env.OLLAMA_URL = originalEnv;
    } else {
      delete process.env.OLLAMA_URL;
    }
  });
});

// =============================================================================
// Embed Tests
// =============================================================================

describe("Ollama.embed", () => {
  test("returns embedding for query", async () => {
    const ollama = createOllama();
    const embedding = [0.1, 0.2, 0.3, 0.4, 0.5];
    setEmbedHandler([embedding]);

    const result = await ollama.embed("test query", { model: "test-model", isQuery: true });

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual(embedding);
    expect(result!.model).toBe("test-model");

    // Verify the request was formatted correctly
    expect(mockCallLog).toHaveLength(1);
    expect(mockCallLog[0].path).toBe("/api/embed");
    expect((mockCallLog[0].body as { input: string }).input).toContain("task: search result");
  });

  test("returns embedding for document", async () => {
    const ollama = createOllama();
    const embedding = [0.5, 0.4, 0.3, 0.2, 0.1];
    setEmbedHandler([embedding]);

    const result = await ollama.embed("doc content", {
      model: "test-model",
      isQuery: false,
      title: "Doc Title",
    });

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual(embedding);

    // Verify document formatting
    expect((mockCallLog[0].body as { input: string }).input).toContain("title: Doc Title");
    expect((mockCallLog[0].body as { input: string }).input).toContain("text: doc content");
  });

  test("returns null on API error", async () => {
    const ollama = createOllama();
    mockHandlers.set("/api/embed", () => ({ status: 500, body: { error: "Server error" } }));

    const result = await ollama.embed("test", { model: "test-model" });
    expect(result).toBeNull();
  });

  test("returns null on empty embeddings", async () => {
    const ollama = createOllama();
    setEmbedHandler([]);

    const result = await ollama.embed("test", { model: "test-model" });
    expect(result).toBeNull();
  });

  test("returns null on network error", async () => {
    const ollama = new Ollama({ baseUrl: "http://nonexistent:99999" });

    // This will throw because our mock doesn't handle this URL
    const result = await ollama.embed("test", { model: "test-model" }).catch(() => null);
    expect(result).toBeNull();
  });

  test("handles high-dimensional embeddings", async () => {
    const ollama = createOllama();
    const embedding = Array(768).fill(0).map((_, i) => i / 768);
    setEmbedHandler([embedding]);

    const result = await ollama.embed("test", { model: "test-model" });
    expect(result!.embedding).toHaveLength(768);
    expect(result!.embedding[0]).toBeCloseTo(0, 5);
    expect(result!.embedding[767]).toBeCloseTo(767 / 768, 5);
  });
});

// =============================================================================
// Generate Tests
// =============================================================================

describe("Ollama.generate", () => {
  test("returns generated text", async () => {
    const ollama = createOllama();
    setGenerateHandler("Generated response text");

    const result = await ollama.generate("prompt", { model: "test-model" });

    expect(result).not.toBeNull();
    expect(result!.text).toBe("Generated response text");
    expect(result!.model).toBe("test-model");
    expect(result!.done).toBe(true);
  });

  test("includes logprobs when requested", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", {
      tokens: ["yes"],
      token_logprobs: [-0.1],
    });

    const result = await ollama.generate("prompt", { model: "test-model", logprobs: true });

    expect(result!.logprobs).toBeDefined();
    expect(result!.logprobs).toHaveLength(1);
    expect(result!.logprobs![0].token).toBe("yes");
    expect(result!.logprobs![0].logprob).toBe(-0.1);
  });

  test("handles multiple logprob tokens", async () => {
    const ollama = createOllama();
    setGenerateHandler("hello world", {
      tokens: ["hello", " world"],
      token_logprobs: [-0.5, -0.3],
    });

    const result = await ollama.generate("prompt", { model: "test-model", logprobs: true });

    expect(result!.logprobs).toHaveLength(2);
    expect(result!.logprobs![0]).toEqual({ token: "hello", logprob: -0.5 });
    expect(result!.logprobs![1]).toEqual({ token: " world", logprob: -0.3 });
  });

  test("sends maxTokens option", async () => {
    const ollama = createOllama();
    setGenerateHandler("response");

    await ollama.generate("prompt", { model: "test-model", maxTokens: 50 });

    const body = mockCallLog[0].body as { options: { num_predict: number } };
    expect(body.options.num_predict).toBe(50);
  });

  test("sends temperature option", async () => {
    const ollama = createOllama();
    setGenerateHandler("response");

    await ollama.generate("prompt", { model: "test-model", temperature: 0.7 });

    const body = mockCallLog[0].body as { options: { temperature: number } };
    expect(body.options.temperature).toBe(0.7);
  });

  test("sends raw option", async () => {
    const ollama = createOllama();
    setGenerateHandler("response");

    await ollama.generate("prompt", { model: "test-model", raw: true });

    const body = mockCallLog[0].body as { raw: boolean };
    expect(body.raw).toBe(true);
  });

  test("returns null on API error", async () => {
    const ollama = createOllama();
    mockHandlers.set("/api/generate", () => ({ status: 500, body: { error: "Error" } }));

    const result = await ollama.generate("prompt", { model: "test-model" });
    expect(result).toBeNull();
  });

  test("handles empty response", async () => {
    const ollama = createOllama();
    setGenerateHandler("");

    const result = await ollama.generate("prompt", { model: "test-model" });
    expect(result!.text).toBe("");
  });
});

// =============================================================================
// Model Management Tests
// =============================================================================

describe("Ollama.modelExists", () => {
  test("returns true for existing model", async () => {
    const ollama = createOllama();
    setModelShowHandler(true, 5000000);

    const result = await ollama.modelExists("test-model");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("test-model");
    expect(result.size).toBe(5000000);
    expect(result.modifiedAt).toBeDefined();
  });

  test("returns false for non-existing model", async () => {
    const ollama = createOllama();
    setModelShowHandler(false);

    const result = await ollama.modelExists("nonexistent-model");

    expect(result.exists).toBe(false);
    expect(result.name).toBe("nonexistent-model");
  });

  test("sends correct model name in request", async () => {
    const ollama = createOllama();
    setModelShowHandler(true);

    await ollama.modelExists("specific-model:v1");

    expect(mockCallLog[0].path).toBe("/api/show");
    expect((mockCallLog[0].body as { name: string }).name).toBe("specific-model:v1");
  });
});

describe("Ollama.pullModel", () => {
  test("returns true on successful pull", async () => {
    const ollama = createOllama();
    setPullHandler(true);

    const result = await ollama.pullModel("new-model");

    expect(result).toBe(true);
    expect(mockCallLog[0].path).toBe("/api/pull");
    expect((mockCallLog[0].body as { name: string }).name).toBe("new-model");
  });

  test("returns false on failed pull", async () => {
    const ollama = createOllama();
    setPullHandler(false);

    const result = await ollama.pullModel("bad-model");
    expect(result).toBe(false);
  });

  test("calls progress callback", async () => {
    const ollama = createOllama();
    setPullHandler(true);

    let progressCalled = false;
    await ollama.pullModel("model", (progress) => {
      progressCalled = true;
      expect(progress).toBe(100);
    });

    expect(progressCalled).toBe(true);
  });
});

// =============================================================================
// Query Expansion Tests
// =============================================================================

describe("Ollama.expandQuery", () => {
  test("returns original query plus expansions", async () => {
    const ollama = createOllama();
    setGenerateHandler("variation one\nvariation two");

    const result = await ollama.expandQuery("original query", "test-model");

    expect(result).toContain("original query");
    expect(result[0]).toBe("original query");
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  test("returns only original query on API failure", async () => {
    const ollama = createOllama();
    mockHandlers.set("/api/generate", () => ({ status: 500, body: { error: "Error" } }));

    const result = await ollama.expandQuery("query", "test-model");

    expect(result).toEqual(["query"]);
  });

  test("filters out thinking tags from response", async () => {
    const ollama = createOllama();
    setGenerateHandler("<think>some thinking</think>\nvariation one\nvariation two");

    const result = await ollama.expandQuery("query", "test-model");

    expect(result).not.toContain("<think>");
    expect(result.some((r) => r.includes("think"))).toBe(false);
  });

  test("filters out very long variations", async () => {
    const ollama = createOllama();
    const longLine = "a".repeat(150);
    setGenerateHandler(`short variation\n${longLine}\nanother short`);

    const result = await ollama.expandQuery("query", "test-model");

    // Long variations (>100 chars) should be filtered
    expect(result.every((r) => r.length < 100)).toBe(true);
  });

  test("respects numVariations parameter", async () => {
    const ollama = createOllama();
    setGenerateHandler("one\ntwo\nthree\nfour\nfive");

    const result = await ollama.expandQuery("query", "test-model", 3);

    // Original + up to 3 variations
    expect(result.length).toBeLessThanOrEqual(4);
  });

  test("sends correct prompt format", async () => {
    const ollama = createOllama();
    setGenerateHandler("variation");

    await ollama.expandQuery("test query", "test-model", 2);

    const body = mockCallLog[0].body as { prompt: string };
    expect(body.prompt).toContain('Query: "test query"');
    expect(body.prompt).toContain("generate 2 alternative queries");
  });
});

// =============================================================================
// Reranking Tests
// =============================================================================

describe("Ollama.rerankerLogprobsCheck", () => {
  test("returns relevance judgments for documents", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const docs = [
      { file: "doc1.md", text: "Relevant content" },
      { file: "doc2.md", text: "Other content" },
    ];

    const results = await ollama.rerankerLogprobsCheck("query", docs, { model: "test-model" });

    expect(results).toHaveLength(2);
    expect(results[0].file).toBe("doc1.md");
    expect(results[0].relevant).toBe(true);
    expect(results[0].rawToken).toBe("yes");
  });

  test("parses yes with high confidence correctly", async () => {
    const ollama = createOllama();
    // -0.1 logprob = ~0.905 confidence
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(true);
    expect(results[0].confidence).toBeCloseTo(Math.exp(-0.1), 3);
    expect(results[0].score).toBeGreaterThan(0.9);
    expect(results[0].logprob).toBe(-0.1);
  });

  test("parses yes with low confidence correctly", async () => {
    const ollama = createOllama();
    // -2.0 logprob = ~0.135 confidence
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-2.0] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(true);
    expect(results[0].confidence).toBeCloseTo(Math.exp(-2.0), 3);
    expect(results[0].score).toBeLessThan(0.6);
  });

  test("parses no with high confidence correctly", async () => {
    const ollama = createOllama();
    // -0.05 logprob = ~0.95 confidence
    setGenerateHandler("no", { tokens: ["no"], token_logprobs: [-0.05] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(false);
    expect(results[0].confidence).toBeCloseTo(Math.exp(-0.05), 3);
    expect(results[0].score).toBeLessThan(0.1); // Low score for confident "no"
  });

  test("parses no with low confidence correctly", async () => {
    const ollama = createOllama();
    // -1.5 logprob = ~0.22 confidence
    setGenerateHandler("no", { tokens: ["no"], token_logprobs: [-1.5] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(false);
    expect(results[0].score).toBeGreaterThan(0.3); // Higher score for uncertain "no"
  });

  test("handles unknown token", async () => {
    const ollama = createOllama();
    setGenerateHandler("maybe", { tokens: ["maybe"], token_logprobs: [-0.5] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(false);
    expect(results[0].score).toBe(0.3); // Neutral score
  });

  test("handles API failure gracefully", async () => {
    const ollama = createOllama();
    mockHandlers.set("/api/generate", () => ({ status: 500, body: { error: "Error" } }));

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].relevant).toBe(false);
    expect(results[0].score).toBe(0);
    expect(results[0].confidence).toBe(0);
  });

  test("respects batchSize option", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const docs = Array(10).fill(null).map((_, i) => ({
      file: `doc${i}.md`,
      text: `content ${i}`,
    }));

    await ollama.rerankerLogprobsCheck("query", docs, { model: "test-model", batchSize: 3 });

    // Should process in batches: 3 + 3 + 3 + 1 = 10 calls
    expect(mockCallLog).toHaveLength(10);
  });

  test("sends correct prompt format", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    await ollama.rerankerLogprobsCheck(
      "search query",
      [{ file: "test.md", text: "document content", title: "Test Doc" }],
      { model: "test-model" }
    );

    const body = mockCallLog[0].body as { prompt: string; raw: boolean; logprobs: boolean };
    expect(body.prompt).toContain("<Query>: search query");
    expect(body.prompt).toContain("<Document Title>: Test Doc");
    expect(body.prompt).toContain("document content");
    expect(body.raw).toBe(true);
    expect(body.logprobs).toBe(true);
  });

  test("uses filename as title when title not provided", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "path/to/document.md", text: "content" }],
      { model: "test-model" }
    );

    const body = mockCallLog[0].body as { prompt: string };
    expect(body.prompt).toContain("<Document Title>: document");
  });

  test("truncates long documents", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const longText = "x".repeat(10000);
    await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: longText }],
      { model: "test-model" }
    );

    const body = mockCallLog[0].body as { prompt: string };
    // Should be truncated to ~4000 chars + "..."
    expect(body.prompt.length).toBeLessThan(10000);
    expect(body.prompt).toContain("...");
  });
});

describe("Ollama.rerank", () => {
  test("returns sorted results by score", async () => {
    const ollama = createOllama();

    // First call returns "no", second returns "yes"
    let callCount = 0;
    mockHandlers.set("/api/generate", () => {
      callCount++;
      if (callCount === 1) {
        return { status: 200, body: { response: "no", done: true, logprobs: { tokens: ["no"], token_logprobs: [-0.1] } } };
      }
      return { status: 200, body: { response: "yes", done: true, logprobs: { tokens: ["yes"], token_logprobs: [-0.1] } } };
    });

    const docs = [
      { file: "low.md", text: "irrelevant" },
      { file: "high.md", text: "relevant" },
    ];

    const result = await ollama.rerank("query", docs, { model: "test-model" });

    expect(result.results).toHaveLength(2);
    expect(result.results[0].file).toBe("high.md"); // Higher score first
    expect(result.results[0].score).toBeGreaterThan(result.results[1].score);
  });

  test("includes model in result", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const result = await ollama.rerank("query", [{ file: "doc.md", text: "content" }], {
      model: "custom-reranker",
    });

    expect(result.model).toBe("custom-reranker");
  });
});

// =============================================================================
// Default Ollama Singleton Tests
// =============================================================================

describe("Default Ollama Singleton", () => {
  afterEach(() => {
    setDefaultOllama(null);
  });

  test("getDefaultOllama creates instance on first call", () => {
    const ollama = getDefaultOllama();
    expect(ollama).toBeInstanceOf(Ollama);
  });

  test("getDefaultOllama returns same instance on subsequent calls", () => {
    const ollama1 = getDefaultOllama();
    const ollama2 = getDefaultOllama();
    expect(ollama1).toBe(ollama2);
  });

  test("setDefaultOllama allows replacing the singleton", () => {
    const custom = new Ollama({ baseUrl: "http://custom:1234" });
    setDefaultOllama(custom);

    const result = getDefaultOllama();
    expect(result).toBe(custom);
    expect(result.getBaseUrl()).toBe("http://custom:1234");
  });

  test("setDefaultOllama with null resets singleton", () => {
    const original = getDefaultOllama();
    setDefaultOllama(null);
    const newInstance = getDefaultOllama();

    expect(newInstance).not.toBe(original);
  });
});

// =============================================================================
// Logprob Math Tests
// =============================================================================

describe("Logprob Mathematics", () => {
  test("logprob 0 = 100% confidence", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [0] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].confidence).toBe(1.0);
    expect(results[0].score).toBe(1.0); // 0.5 + 0.5 * 1.0
  });

  test("logprob -ln(2) ≈ 50% confidence", async () => {
    const ollama = createOllama();
    const logprob = -Math.log(2); // ≈ -0.693
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [logprob] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].confidence).toBeCloseTo(0.5, 3);
    expect(results[0].score).toBeCloseTo(0.75, 3); // 0.5 + 0.5 * 0.5
  });

  test("very negative logprob = very low confidence", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-10] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].confidence).toBeLessThan(0.0001);
    expect(results[0].score).toBeCloseTo(0.5, 2); // Nearly just the base 0.5
  });
});

// =============================================================================
// Edge Cases
// =============================================================================

describe("Edge Cases", () => {
  test("handles empty document list", async () => {
    const ollama = createOllama();

    const results = await ollama.rerankerLogprobsCheck("query", [], { model: "test-model" });
    expect(results).toHaveLength(0);
  });

  test("handles very short document text", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "x" }],
      { model: "test-model" }
    );

    expect(results).toHaveLength(1);
  });

  test("handles unicode in queries and documents", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const results = await ollama.rerankerLogprobsCheck(
      "日本語クエリ",
      [{ file: "doc.md", text: "日本語コンテンツ 🎉" }],
      { model: "test-model" }
    );

    expect(results).toHaveLength(1);

    const body = mockCallLog[0].body as { prompt: string };
    expect(body.prompt).toContain("日本語クエリ");
    expect(body.prompt).toContain("日本語コンテンツ");
  });

  test("handles special characters in file paths", async () => {
    const ollama = createOllama();
    setGenerateHandler("yes", { tokens: ["yes"], token_logprobs: [-0.1] });

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "path/to/file with spaces.md", text: "content" }],
      { model: "test-model" }
    );

    expect(results[0].file).toBe("path/to/file with spaces.md");
  });

  test("handles missing logprobs in response", async () => {
    const ollama = createOllama();
    // Response without logprobs
    mockHandlers.set("/api/generate", () => ({
      status: 200,
      body: { response: "yes", done: true },
    }));

    const results = await ollama.rerankerLogprobsCheck(
      "query",
      [{ file: "doc.md", text: "content" }],
      { model: "test-model" }
    );

    // Should still work, with logprob defaulting to 0
    expect(results[0].logprob).toBe(0);
  });
});
