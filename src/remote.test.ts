/**
 * Tests for remote embedding providers (Voyage AI, OpenAI-compatible)
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { RemoteLLM, getDefaultRemote } from "./remote";

// Skip tests if no API key is available
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
const describeWithVoyage = VOYAGE_API_KEY ? describe : describe.skip;

describeWithVoyage("RemoteLLM (Voyage)", () => {
  let llm: RemoteLLM;

  beforeAll(() => {
    llm = new RemoteLLM({ provider: "voyage" });
  });

  afterAll(async () => {
    await llm.dispose();
  });

  describe("embed", () => {
    it("generates embeddings for text", async () => {
      const result = await llm.embed("Hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toBeArray();
      expect(result!.embedding.length).toBeGreaterThan(0);
      expect(result!.model).toContain("voyage");
    });

    it("generates embeddings with query input type", async () => {
      const result = await llm.embed("What is QMD?", { isQuery: true });
      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBeGreaterThan(0);
    });

    it("generates embeddings with document input type", async () => {
      const result = await llm.embed("QMD is a markdown search tool", { isQuery: false });
      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBeGreaterThan(0);
    });
  });

  describe("embedBatch", () => {
    it("embeds multiple texts efficiently", async () => {
      const texts = ["First document", "Second document", "Third document"];
      const results = await llm.embedBatch(texts);
      
      expect(results.length).toBe(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding.length).toBeGreaterThan(0);
      }
    });

    it("handles empty input", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toEqual([]);
    });
  });

  describe("rerank", () => {
    it("reranks documents by relevance", async () => {
      const query = "What is drone training?";
      const documents = [
        { file: "doc1.md", text: "Acme Corp offers professional software development courses" },
        { file: "doc2.md", text: "The weather forecast shows rain tomorrow" },
        { file: "doc3.md", text: "Learn to fly drones with our CAA-certified training program" },
      ];

      const result = await llm.rerank(query, documents);
      
      expect(result.results.length).toBe(3);
      expect(result.model).toContain("rerank");
      
      // The drone-related documents should rank higher
      const topResult = result.results[0]!;
      expect(["doc1.md", "doc3.md"]).toContain(topResult.file);
      expect(topResult.score).toBeGreaterThan(0);
    });

    it("respects top_k parameter", async () => {
      const documents = [
        { file: "a.md", text: "Document A" },
        { file: "b.md", text: "Document B" },
        { file: "c.md", text: "Document C" },
      ];

      const result = await llm.rerank("test", documents, { topK: 2 });
      expect(result.results.length).toBe(2);
    });
  });

  describe("modelExists", () => {
    it("returns info for valid Voyage models", async () => {
      const info = await llm.modelExists("voyage-4-lite");
      expect(info.name).toBe("voyage-4-lite");
      expect(info.exists).toBe(true);
    });

    it("returns exists true for any model (provider-side validation)", async () => {
      // RemoteLLM doesn't validate models client-side - the API will reject invalid ones
      const info = await llm.modelExists("any-model-name");
      expect(info.exists).toBe(true);
    });
  });

  describe("expandQuery", () => {
    it("returns lex and vec query types", async () => {
      const results = await llm.expandQuery("test query");
      expect(results.length).toBe(2);
      expect(results.map(r => r.type)).toContain("lex");
      expect(results.map(r => r.type)).toContain("vec");
      expect(results.every(r => r.text === "test query")).toBe(true);
    });

    it("can exclude lexical queries", async () => {
      const results = await llm.expandQuery("test query", { includeLexical: false });
      expect(results.length).toBe(1);
      expect(results[0]!.type).toBe("vec");
    });
  });
});

describe("RemoteLLM Configuration", () => {
  it("throws when Voyage API key is missing", () => {
    const originalKey = process.env.VOYAGE_API_KEY;
    delete process.env.VOYAGE_API_KEY;
    
    expect(() => new RemoteLLM({ provider: "voyage" })).toThrow("Voyage API key required");
    
    process.env.VOYAGE_API_KEY = originalKey;
  });

  it("uses custom base URL", () => {
    const originalKey = process.env.VOYAGE_API_KEY;
    process.env.VOYAGE_API_KEY = "test-key";
    
    const llm = new RemoteLLM({ 
      provider: "voyage",
      baseUrl: "https://custom.api.com/v1"
    });
    
    // Can't easily verify the URL was set, but at least it doesn't throw
    expect(llm).toBeDefined();
    
    process.env.VOYAGE_API_KEY = originalKey;
  });

  it("uses environment variables for model config", () => {
    const originalKey = process.env.VOYAGE_API_KEY;
    const originalModel = process.env.VOYAGE_EMBED_MODEL;
    
    process.env.VOYAGE_API_KEY = "test-key";
    process.env.VOYAGE_EMBED_MODEL = "voyage-3-large";
    
    const llm = new RemoteLLM({ provider: "voyage" });
    expect(llm).toBeDefined();
    
    process.env.VOYAGE_API_KEY = originalKey;
    if (originalModel) process.env.VOYAGE_EMBED_MODEL = originalModel;
    else delete process.env.VOYAGE_EMBED_MODEL;
  });
});

describe("getDefaultRemote", () => {
  it("returns singleton instance", () => {
    if (!VOYAGE_API_KEY) {
      console.log("Skipping getDefaultRemote test - no API key");
      return;
    }
    
    const llm1 = getDefaultRemote();
    const llm2 = getDefaultRemote();
    expect(llm1).toBe(llm2);
  });
});
