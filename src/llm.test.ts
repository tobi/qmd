/**
 * llm.test.ts - Unit tests for the LLM abstraction layer (node-llama-cpp)
 *
 * Run with: bun test src/llm.test.ts
 *
 * These tests require the actual models to be downloaded. Run the embed or
 * rerank functions first to trigger model downloads.
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import {
  LlamaCpp,
  getDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  type RerankDocument,
} from "./llm.js";

// =============================================================================
// Singleton Tests (no model loading required)
// =============================================================================

describe("Default LlamaCpp Singleton", () => {
  // Test singleton behavior without resetting to avoid orphan instances
  test("getDefaultLlamaCpp returns same instance on subsequent calls", () => {
    const llm1 = getDefaultLlamaCpp();
    const llm2 = getDefaultLlamaCpp();
    expect(llm1).toBe(llm2);
    expect(llm1).toBeInstanceOf(LlamaCpp);
  });
});

// =============================================================================
// Model Existence Tests
// =============================================================================

describe("LlamaCpp.modelExists", () => {
  test("returns exists:true for HuggingFace model URIs", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("hf:org/repo/model.gguf");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("hf:org/repo/model.gguf");
  });

  test("returns exists:false for non-existent local paths", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("/nonexistent/path/model.gguf");

    expect(result.exists).toBe(false);
    expect(result.name).toBe("/nonexistent/path/model.gguf");
  });
});

// =============================================================================
// Integration Tests (require actual models)
// =============================================================================

describe("LlamaCpp Integration", () => {
  // Use the singleton to avoid multiple Metal contexts
  const llm = getDefaultLlamaCpp();

  afterAll(async () => {
    // Ensure native resources are released to avoid ggml-metal asserts on process exit.
    await disposeDefaultLlamaCpp();
  });

  describe("embed", () => {
    test("returns embedding with correct dimensions", async () => {
      const result = await llm.embed("Hello world");

      expect(result).not.toBeNull();
      expect(result!.embedding).toBeInstanceOf(Array);
      expect(result!.embedding.length).toBeGreaterThan(0);
      // embeddinggemma outputs 768 dimensions
      expect(result!.embedding.length).toBe(768);
    });

    test("returns consistent embeddings for same input", async () => {
      const result1 = await llm.embed("test text");
      const result2 = await llm.embed("test text");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Embeddings should be identical for the same input
      for (let i = 0; i < result1!.embedding.length; i++) {
        expect(result1!.embedding[i]).toBeCloseTo(result2!.embedding[i], 5);
      }
    });

    test("returns different embeddings for different inputs", async () => {
      const result1 = await llm.embed("cats are great");
      const result2 = await llm.embed("database optimization");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Calculate cosine similarity - should be less than 1.0 (not identical)
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;
      for (let i = 0; i < result1!.embedding.length; i++) {
        dotProduct += result1!.embedding[i] * result2!.embedding[i];
        norm1 += result1!.embedding[i] ** 2;
        norm2 += result2!.embedding[i] ** 2;
      }
      const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

      expect(similarity).toBeLessThan(0.95); // Should be meaningfully different
    });
  });

  describe("embedBatch", () => {
    test("returns embeddings for multiple texts", async () => {
      const texts = ["Hello world", "Test text", "Another document"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding.length).toBe(768);
      }
    });

    test("returns same results as individual embed calls", async () => {
      const texts = ["cats are great", "dogs are awesome"];

      // Get batch embeddings
      const batchResults = await llm.embedBatch(texts);

      // Get individual embeddings
      const individualResults = await Promise.all(texts.map(t => llm.embed(t)));

      // Compare - should be identical
      for (let i = 0; i < texts.length; i++) {
        expect(batchResults[i]).not.toBeNull();
        expect(individualResults[i]).not.toBeNull();
        for (let j = 0; j < batchResults[i]!.embedding.length; j++) {
          expect(batchResults[i]!.embedding[j]).toBeCloseTo(individualResults[i]!.embedding[j], 5);
        }
      }
    });

    test("handles empty array", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    test("batch is faster than sequential", async () => {
      const texts = Array(10).fill(null).map((_, i) => `Document number ${i} with content`);

      // Time batch
      const batchStart = Date.now();
      await llm.embedBatch(texts);
      const batchTime = Date.now() - batchStart;

      // Time sequential
      const seqStart = Date.now();
      for (const text of texts) {
        await llm.embed(text);
      }
      const seqTime = Date.now() - seqStart;

      console.log(`Batch: ${batchTime}ms, Sequential: ${seqTime}ms`);
      // Performance is machine/load dependent. We only assert batch isn't drastically worse.
      expect(batchTime).toBeLessThanOrEqual(seqTime * 3);
    });
  });

  describe("rerank", () => {
    test("scores capital of France question correctly", async () => {
      const query = "What is the capital of France?";
      const documents: RerankDocument[] = [
        { file: "butterflies.txt", text: "Butterflies indeed fly through the garden." },
        { file: "france.txt", text: "The capital of France is Paris." },
        { file: "canada.txt", text: "The capital of Canada is Ottawa." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(3);

      // The France document should score highest
      expect(result.results[0].file).toBe("france.txt");
      expect(result.results[0].score).toBeGreaterThan(0.7);

      // Canada should be somewhat relevant (also about capitals)
      expect(result.results[1].file).toBe("canada.txt");

      // Butterflies should score lowest
      expect(result.results[2].file).toBe("butterflies.txt");
      expect(result.results[2].score).toBeLessThan(0.6);
    });

    test("scores authentication query correctly", async () => {
      const query = "How do I configure authentication?";
      const documents: RerankDocument[] = [
        { file: "weather.md", text: "The weather today is sunny with mild temperatures." },
        { file: "auth.md", text: "Authentication can be configured by setting the AUTH_SECRET environment variable." },
        { file: "pizza.md", text: "Our restaurant serves the best pizza in town." },
        { file: "jwt.md", text: "JWT authentication requires a secret key and expiration time." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(4);

      // Auth documents should score highest
      const topTwo = result.results.slice(0, 2).map((r) => r.file);
      expect(topTwo).toContain("auth.md");
      expect(topTwo).toContain("jwt.md");

      // Irrelevant documents should score lowest
      const bottomTwo = result.results.slice(2).map((r) => r.file);
      expect(bottomTwo).toContain("weather.md");
      expect(bottomTwo).toContain("pizza.md");
    });

    test("handles programming queries correctly", async () => {
      const query = "How do I handle errors in JavaScript?";
      const documents: RerankDocument[] = [
        { file: "cooking.md", text: "To make a good pasta, boil water and add salt." },
        { file: "errors.md", text: "Use try-catch blocks to handle JavaScript errors gracefully." },
        { file: "python.md", text: "Python uses try-except for exception handling." },
      ];

      const result = await llm.rerank(query, documents);

      // JavaScript errors doc should score highest
      expect(result.results[0].file).toBe("errors.md");
      expect(result.results[0].score).toBeGreaterThan(0.7);

      // Python doc might be somewhat relevant (same concept, different language)
      // Cooking should be least relevant
      expect(result.results[2].file).toBe("cooking.md");
    });

    test("handles empty document list", async () => {
      const result = await llm.rerank("test query", []);
      expect(result.results).toHaveLength(0);
    });

    test("handles single document", async () => {
      const result = await llm.rerank("test", [{ file: "doc.md", text: "content" }]);
      expect(result.results).toHaveLength(1);
      expect(result.results[0].file).toBe("doc.md");
    });

    test("preserves original file paths", async () => {
      const documents: RerankDocument[] = [
        { file: "path/to/doc1.md", text: "content one" },
        { file: "another/path/doc2.md", text: "content two" },
      ];

      const result = await llm.rerank("query", documents);

      const files = result.results.map((r) => r.file).sort();
      expect(files).toEqual(["another/path/doc2.md", "path/to/doc1.md"]);
    });

    test("returns scores between 0 and 1", async () => {
      const documents: RerankDocument[] = [
        { file: "a.md", text: "The quick brown fox jumps over the lazy dog." },
        { file: "b.md", text: "Machine learning algorithms process data efficiently." },
        { file: "c.md", text: "React components use JSX syntax for rendering." },
      ];

      const result = await llm.rerank("Tell me about animals", documents);

      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }
    });

    test("batch reranks multiple documents efficiently", async () => {
      // Create 10 documents to verify batch processing works
      const documents: RerankDocument[] = Array(10)
        .fill(null)
        .map((_, i) => ({
          file: `doc${i}.md`,
          text: `Document number ${i} with some content about topic ${i % 3}`,
        }));

      const start = Date.now();
      const result = await llm.rerank("topic 1", documents);
      const elapsed = Date.now() - start;

      expect(result.results).toHaveLength(10);

      // Verify all documents are returned with valid scores
      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }

      // Log timing for monitoring batch performance
      console.log(`Batch rerank of 10 docs took ${elapsed}ms`);
    });
  });

  describe("expandQuery", () => {
    test("returns at least the original query", async () => {
      const result = await llm.expandQuery("test query");

      expect(result).toContain("test query");
      expect(result.length).toBeGreaterThanOrEqual(1);
    }, 30000); // 30s timeout for model loading

    test("returns original query first", async () => {
      const result = await llm.expandQuery("authentication setup");

      expect(result[0]).toBe("authentication setup");
    });
  });
});

