/**
 * Integration tests for RemoteLLM against live vLLM servers.
 *
 * Requires environment variables:
 *   VLLM_EMBED_URL   - e.g. http://gpu-host:8002/v1
 *   VLLM_EMBED_MODEL - e.g. Qwen/Qwen3-Embedding-0.6B
 *   VLLM_RERANK_URL  - e.g. http://gpu-host:8001/v1
 *   VLLM_RERANK_MODEL - e.g. qwen3-reranker-4b
 *
 * Skip these tests when no server is available (all tests guard on EMBED_URL).
 */

import { describe, it, expect, beforeAll, beforeEach } from "vitest";
import { RemoteLLM } from "../src/remote-llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";
import { formatQueryForEmbedding, formatDocForEmbedding } from "../src/llm.js";
import type { LLM } from "../src/llm.js";

const EMBED_URL = process.env.VLLM_EMBED_URL ?? "";
const EMBED_MODEL = process.env.VLLM_EMBED_MODEL ?? "";
const RERANK_URL = process.env.VLLM_RERANK_URL ?? "";
const RERANK_MODEL = process.env.VLLM_RERANK_MODEL ?? "";

const SKIP = !EMBED_URL || !EMBED_MODEL;

let remoteLlm: RemoteLLM;

beforeAll(() => {
  if (SKIP) return;
  remoteLlm = new RemoteLLM({
    embedApiUrl: EMBED_URL,
    embedApiModel: EMBED_MODEL,
    rerankApiUrl: RERANK_URL,
    rerankApiModel: RERANK_MODEL,
  });
});

// =============================================================================
// Connectivity
// =============================================================================

describe.skipIf(SKIP)("Server connectivity", () => {
  it("can reach the embedding server", async () => {
    const res = await fetch(`${EMBED_URL}/models`);
    expect(res.ok).toBe(true);
    const json = await res.json() as any;
    expect(json.data.length).toBeGreaterThan(0);
  });

  it("can reach the reranking server", async () => {
    const res = await fetch(`${RERANK_URL}/models`);
    expect(res.ok).toBe(true);
  });
});

// =============================================================================
// Single embedding
// =============================================================================

describe.skipIf(SKIP)("Single embedding", () => {
  it("returns a non-empty embedding vector", async () => {
    const result = await remoteLlm.embed("The quick brown fox jumps over the lazy dog");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
    expect(result!.model).toBe(EMBED_MODEL);
  });

  it("embedding values are finite numbers", async () => {
    const result = await remoteLlm.embed("test embedding quality");
    expect(result).not.toBeNull();
    for (const val of result!.embedding) {
      expect(Number.isFinite(val)).toBe(true);
    }
  });

  it("embedding is normalized (L2 norm ≈ 1.0)", async () => {
    const result = await remoteLlm.embed("normalization check");
    expect(result).not.toBeNull();
    const norm = Math.sqrt(result!.embedding.reduce((sum, v) => sum + v * v, 0));
    expect(norm).toBeCloseTo(1.0, 1); // within 0.1
  });

  it("different texts produce different embeddings", async () => {
    const [a, b] = await Promise.all([
      remoteLlm.embed("cats are wonderful pets"),
      remoteLlm.embed("quantum computing research paper"),
    ]);
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    // Cosine similarity should be < 1 (they are different)
    const dot = a!.embedding.reduce((sum, v, i) => sum + v * b!.embedding[i]!, 0);
    expect(dot).toBeLessThan(0.95);
  });

  it("similar texts produce similar embeddings", async () => {
    const [a, b] = await Promise.all([
      remoteLlm.embed("how to train a puppy"),
      remoteLlm.embed("puppy training tips"),
    ]);
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    const dot = a!.embedding.reduce((sum, v, i) => sum + v * b!.embedding[i]!, 0);
    expect(dot).toBeGreaterThan(0.7);
  });
});

// =============================================================================
// Dimension consistency
// =============================================================================

describe.skipIf(SKIP)("Dimension consistency", () => {
  it("all embeddings have the same dimension", async () => {
    const texts = [
      "short",
      "a medium length sentence about embedding dimensions",
      "a much longer piece of text that goes on and on to test whether the embedding dimension stays consistent regardless of input length, which it absolutely should because the model always projects to a fixed-size output vector",
    ];
    const results = await Promise.all(texts.map(t => remoteLlm.embed(t)));
    const dims = results.map(r => r!.embedding.length);
    expect(new Set(dims).size).toBe(1);
    console.log(`  Embedding dimension: ${dims[0]}`);
  });
});

// =============================================================================
// Batch embedding
// =============================================================================

describe.skipIf(SKIP)("Batch embedding", () => {
  it("embeds a batch of texts", async () => {
    const texts = [
      "document one about machine learning",
      "document two about cooking recipes",
      "document three about space exploration",
    ];
    const results = await remoteLlm.embedBatch(texts);
    expect(results).toHaveLength(3);
    for (const r of results) {
      expect(r).not.toBeNull();
      expect(r!.embedding.length).toBeGreaterThan(0);
    }
  });

  it("batch results match individual results", async () => {
    const texts = ["alpha text", "beta text"];
    const [batchResults, individual1, individual2] = await Promise.all([
      remoteLlm.embedBatch(texts),
      remoteLlm.embed("alpha text"),
      remoteLlm.embed("beta text"),
    ]);

    // Compare batch[0] with individual1
    expect(batchResults[0]!.embedding.length).toBe(individual1!.embedding.length);
    // Embeddings should be very close (may not be exactly identical due to batching)
    const dot = batchResults[0]!.embedding.reduce(
      (sum, v, i) => sum + v * individual1!.embedding[i]!, 0
    );
    expect(dot).toBeGreaterThan(0.99);
  });

  it("handles empty batch", async () => {
    const results = await remoteLlm.embedBatch([]);
    expect(results).toEqual([]);
  });

  it("handles large batch (>32 texts, triggers splitting)", async () => {
    const texts = Array.from({ length: 50 }, (_, i) => `document number ${i} about topic ${i % 5}`);
    const results = await remoteLlm.embedBatch(texts);
    expect(results).toHaveLength(50);
    for (const r of results) {
      expect(r).not.toBeNull();
    }
  });
});

// =============================================================================
// Edge cases
// =============================================================================

describe.skipIf(SKIP)("Edge cases", () => {
  it("handles very short text", async () => {
    const result = await remoteLlm.embed("a");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
  });

  it("handles text with special characters", async () => {
    const result = await remoteLlm.embed("café résumé naïve 日本語 中文 🎉 <script>alert('xss')</script>");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
  });

  it("handles multi-paragraph text", async () => {
    const text = `# Introduction

This is a long document with multiple paragraphs and markdown formatting.

## Section 1

Some content here with **bold** and *italic* text.

## Section 2

More content with a list:
- item one
- item two
- item three

\`\`\`python
def hello():
    print("hello world")
\`\`\`
`;
    const result = await remoteLlm.embed(text);
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Reranking
// =============================================================================

describe.skipIf(SKIP)("Reranking", () => {
  it("reranks documents by relevance", async () => {
    const query = "how to bake chocolate chip cookies";
    const documents = [
      { file: "space.md", text: "The Mars rover collected soil samples from the crater rim." },
      { file: "cookies.md", text: "Preheat oven to 375°F. Mix flour, butter, sugar and chocolate chips. Bake for 12 minutes." },
      { file: "quantum.md", text: "Quantum entanglement allows particles to be correlated over large distances." },
      { file: "baking.md", text: "Cookie recipes require precise measurements of ingredients like flour and sugar." },
    ];

    const result = await remoteLlm.rerank(query, documents);
    expect(result.model).toBe(RERANK_MODEL);
    expect(result.results).toHaveLength(4);

    // The cookie/baking docs should rank higher than space/quantum
    const scores = new Map(result.results.map(r => [r.file, r.score]));
    console.log("  Rerank scores:", Object.fromEntries(scores));

    expect(scores.get("cookies.md")!).toBeGreaterThan(scores.get("space.md")!);
    expect(scores.get("cookies.md")!).toBeGreaterThan(scores.get("quantum.md")!);
  });

  it("scores are between 0 and 1", async () => {
    const result = await remoteLlm.rerank("test query", [
      { file: "a.md", text: "relevant document about testing" },
      { file: "b.md", text: "unrelated document about gardening" },
    ]);
    for (const r of result.results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  it("preserves file mapping through index", async () => {
    const documents = [
      { file: "first.md", text: "First document" },
      { file: "second.md", text: "Second document" },
      { file: "third.md", text: "Third document" },
    ];
    const result = await remoteLlm.rerank("query", documents);
    const files = new Set(result.results.map(r => r.file));
    expect(files).toEqual(new Set(["first.md", "second.md", "third.md"]));
  });

  it("handles single document", async () => {
    const result = await remoteLlm.rerank("test", [
      { file: "only.md", text: "The only document" },
    ]);
    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.file).toBe("only.md");
  });

  it("handles many documents", async () => {
    const documents = Array.from({ length: 20 }, (_, i) => ({
      file: `doc${i}.md`,
      text: `Document ${i} contains some text about topic ${i % 4}`,
    }));
    const result = await remoteLlm.rerank("topic about topic 2", documents);
    expect(result.results).toHaveLength(20);
  });
});

// =============================================================================
// Embedding format (remote models skip prefixes)
// =============================================================================

describe.skipIf(SKIP)("Embedding format for remote models", () => {
  it("formatQueryForEmbedding returns raw text for remote model name", () => {
    const formatted = formatQueryForEmbedding("search query", EMBED_MODEL);
    expect(formatted).toBe("search query");
  });

  it("formatDocForEmbedding returns raw text for remote model name", () => {
    const formatted = formatDocForEmbedding("doc content", undefined, EMBED_MODEL);
    expect(formatted).toBe("doc content");
  });

  it("formatDocForEmbedding with title prepends title for remote model", () => {
    const formatted = formatDocForEmbedding("doc content", "Title", EMBED_MODEL);
    expect(formatted).toBe("Title\ndoc content");
  });
});

// =============================================================================
// HybridLLM integration
// =============================================================================

describe.skipIf(SKIP)("HybridLLM with real remote backend", () => {
  // Mock local LLM for generate/expandQuery
  function createMockLocal(): LLM {
    return {
      embedModelName: "local-embed-model",
      embed: async () => ({ embedding: [0.5], model: "local" }),
      embedBatch: async (texts) => texts.map(() => ({ embedding: [0.5], model: "local" })),
      generate: async () => ({ text: "generated text", model: "local", done: true }),
      modelExists: async (model) => ({ name: model, exists: true }),
      expandQuery: async () => [{ type: "lex" as const, text: "expanded" }],
      rerank: async () => ({ results: [], model: "local" }),
      dispose: async () => {},
    };
  }

  it("routes embed through remote, returning real embeddings", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    const result = await hybrid.embed("testing hybrid embedding");
    expect(result).not.toBeNull();
    // Real embedding has many dimensions, not just [0.5]
    expect(result!.embedding.length).toBeGreaterThan(10);
    expect(result!.model).toBe(EMBED_MODEL);
  });

  it("routes embedBatch through remote", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    const results = await hybrid.embedBatch(["text one", "text two"]);
    expect(results).toHaveLength(2);
    expect(results[0]!.embedding.length).toBeGreaterThan(10);
  });

  it("routes rerank through remote", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    const result = await hybrid.rerank("cookies", [
      { file: "a.md", text: "baking cookies at 350 degrees" },
      { file: "b.md", text: "orbiting space station" },
    ]);
    expect(result.model).toBe(RERANK_MODEL);
    expect(result.results).toHaveLength(2);
  });

  it("routes generate through local mock", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    const result = await hybrid.generate("prompt");
    expect(result!.text).toBe("generated text");
    expect(result!.model).toBe("local");
  });

  it("routes expandQuery through local mock", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    const result = await hybrid.expandQuery("query");
    expect(result[0]!.text).toBe("expanded");
  });

  it("embedModelName comes from remote", async () => {
    const hybrid = new HybridLLM(remoteLlm, createMockLocal());
    expect(hybrid.embedModelName).toBe(EMBED_MODEL);
  });
});

// =============================================================================
// End-to-end: embed → cosine similarity search
// =============================================================================

describe.skipIf(SKIP)("End-to-end embed + search simulation", () => {
  it("finds the most relevant document via cosine similarity", async () => {
    // Index some "documents"
    const docs = [
      { file: "git.md", text: "Git is a distributed version control system for tracking changes in source code" },
      { file: "cooking.md", text: "To make pasta, boil water, add salt, cook noodles for 8 minutes" },
      { file: "docker.md", text: "Docker containers package applications with their dependencies for consistent deployment" },
      { file: "gardening.md", text: "Tomatoes need full sun and regular watering to produce fruit" },
      { file: "typescript.md", text: "TypeScript adds static type checking to JavaScript for safer code" },
    ];

    // Embed all documents
    const docEmbeddings = await remoteLlm.embedBatch(docs.map(d => d.text));

    // Embed a query
    const queryResult = await remoteLlm.embed("how to use version control for my code");
    expect(queryResult).not.toBeNull();

    // Compute cosine similarities
    const similarities = docEmbeddings.map((docEmb, i) => {
      const dot = queryResult!.embedding.reduce((sum, v, j) => sum + v * docEmb!.embedding[j]!, 0);
      return { file: docs[i]!.file, similarity: dot };
    });

    similarities.sort((a, b) => b.similarity - a.similarity);
    console.log("  Similarity ranking:");
    for (const s of similarities) {
      console.log(`    ${s.file}: ${s.similarity.toFixed(4)}`);
    }

    // git.md should be the top result for a version control query
    expect(similarities[0]!.file).toBe("git.md");
    // cooking/gardening should be near the bottom
    const cookingRank = similarities.findIndex(s => s.file === "cooking.md");
    const gitRank = similarities.findIndex(s => s.file === "git.md");
    expect(gitRank).toBeLessThan(cookingRank);
  });
});
