import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RemoteLLM } from "../src/remote-llm";
import * as piAi from "@mariozechner/pi-ai";

// Mock the entire module
vi.mock("@mariozechner/pi-ai", async () => {
  return {
    getModel: vi.fn(),
    complete: vi.fn(),
  };
});

describe("RemoteLLM", () => {
  const config = {
    apiKey: "test-key",
    baseURL: "https://api.example.com/v1"
  };

  let llm: RemoteLLM;

  beforeEach(() => {
    llm = new RemoteLLM(config);
    global.fetch = vi.fn();
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should initialize with correct config", () => {
    expect(llm).toBeInstanceOf(RemoteLLM);
  });

  // ---------------------------------------------------------------------------
  // embed
  // ---------------------------------------------------------------------------

  it("should call embeddings endpoint via fetch", async () => {
    const mockResponse = {
      data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }]
    };
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const result = await llm.embed("test");
    expect(result).toEqual({
      embedding: [0.1, 0.2, 0.3],
      model: "text-embedding-3-small"
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "https://api.example.com/v1/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Authorization": "Bearer test-key"
        }),
        body: JSON.stringify({
          model: "text-embedding-3-small",
          input: "test"
        })
      })
    );
  });

  it("should return null when embed API returns no data", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ data: [] })
    });
    const result = await llm.embed("test");
    expect(result).toBeNull();
  });

  it("should return null when embed API call throws", async () => {
    (global.fetch as any).mockRejectedValue(new Error("network error"));
    const result = await llm.embed("test");
    expect(result).toBeNull();
  });

  // ---------------------------------------------------------------------------
  // embedBatch
  // ---------------------------------------------------------------------------

  it("should send batch embeddings in a single request", async () => {
    const mockResponse = {
      data: [
        { embedding: [0.1, 0.2], index: 0 },
        { embedding: [0.3, 0.4], index: 1 },
      ]
    };
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const results = await llm.embedBatch(["hello", "world"]);
    expect(results).toHaveLength(2);
    expect(results[0]!.embedding).toEqual([0.1, 0.2]);
    expect(results[1]!.embedding).toEqual([0.3, 0.4]);

    // Should be a single fetch call with array input
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
    expect(Array.isArray(body.input)).toBe(true);
    expect(body.input).toEqual(["hello", "world"]);
  });

  it("should return empty array for empty embedBatch input", async () => {
    const results = await llm.embedBatch([]);
    expect(results).toHaveLength(0);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("should sort embedBatch results by index", async () => {
    // API may return results out of order
    const mockResponse = {
      data: [
        { embedding: [0.3, 0.4], index: 1 },
        { embedding: [0.1, 0.2], index: 0 },
      ]
    };
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const results = await llm.embedBatch(["first", "second"]);
    expect(results[0]!.embedding).toEqual([0.1, 0.2]);
    expect(results[1]!.embedding).toEqual([0.3, 0.4]);
  });

  // ---------------------------------------------------------------------------
  // generate
  // ---------------------------------------------------------------------------

  it("should call pi-ai complete for generate", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);

    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "Hello world" }]
    });

    // Use an OpenAI baseURL so resolvePiModel() routes through pi-ai's
    // getModel() registry (non-OpenAI URLs synthesize the model object
    // inline — see resolvePiModel comment). Other tests in this file
    // intentionally use api.example.com/v1 to exercise the non-OpenAI path.
    const openaiLlm = new RemoteLLM({ ...config, baseURL: "https://api.openai.com/v1" });
    const result = await openaiLlm.generate("Hi");

    expect(piAi.getModel).toHaveBeenCalled();
    expect(piAi.complete).toHaveBeenCalledWith(
      expect.objectContaining({ id: "gpt-3.5-turbo" }),
      expect.objectContaining({
        messages: [expect.objectContaining({ role: "user", content: "Hi" })]
      }),
      expect.objectContaining({
        apiKey: config.apiKey,
      })
    );

    expect(result).toEqual({
      text: "Hello world",
      model: "openai/gpt-3.5-turbo",
      done: true
    });
  });

  it("should concatenate multiple text content blocks", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [
        { type: "text", text: "Hello " },
        { type: "thinking", thinking: "internal" },
        { type: "text", text: "world" },
      ]
    });

    const result = await llm.generate("Hi");
    expect(result!.text).toBe("Hello world");
  });

  it("should return null when generate throws", async () => {
    (piAi.getModel as any).mockReturnValue({ id: "gpt-3.5-turbo", api: "openai" });
    (piAi.complete as any).mockRejectedValue(new Error("API error"));
    const result = await llm.generate("Hi");
    expect(result).toBeNull();
  });

  // ---------------------------------------------------------------------------
  // rerank
  // ---------------------------------------------------------------------------

  it("should return empty results for empty document list", async () => {
    const result = await llm.rerank("query", []);
    expect(result.results).toHaveLength(0);
  });

  it("should score and sort documents via generate", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    // Scores: doc[0]=0.9 (most relevant), doc[1]=0.1 (least relevant)
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "[0.9, 0.1]" }]
    });

    const llmWithRerank = new RemoteLLM({ ...config, rerankModel: "openai/gpt-4o-mini" });
    const docs = [
      { file: "a.md", text: "The capital of France is Paris." },
      { file: "b.md", text: "Dogs are great pets." },
    ];
    const result = await llmWithRerank.rerank("What is the capital of France?", docs);

    expect(result.results).toHaveLength(2);
    expect(result.results[0]!.file).toBe("a.md");
    expect(result.results[1]!.file).toBe("b.md");
    expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
  });

  it("should use rerankModel over generateModel", async () => {
    const llmWithRerank = new RemoteLLM({
      ...config,
      generateModel: "openai/gpt-3.5-turbo",
      rerankModel: "openai/gpt-4o-mini"
    });

    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "[0.5]" }]
    });

    const result = await llmWithRerank.rerank("query", [{ file: "a.md", text: "content" }]);
    expect(result.model).toBe("openai/gpt-4o-mini");
  });

  it("should fall back to generateModel when no rerankModel configured", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "[0.5]" }]
    });

    const result = await llm.rerank("query", [{ file: "a.md", text: "content" }]);
    expect(result.model).toBe("openai/gpt-3.5-turbo");
  });

  it("should normalize scores within each batch to 0-1 range", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    // Model returns scores on a 1-10 scale instead of 0-1
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "[2, 8, 5]" }]
    });

    const docs = [
      { file: "a.md", text: "low relevance" },
      { file: "b.md", text: "high relevance" },
      { file: "c.md", text: "medium relevance" },
    ];
    const result = await llm.rerank("query", docs);

    for (const r of result.results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
    // b.md had score 8 (highest), should rank first after normalization
    expect(result.results[0]!.file).toBe("b.md");
  });

  it("should fall back to 0.5 scores when generate returns null", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockRejectedValue(new Error("API error"));

    const docs = [
      { file: "a.md", text: "content a" },
      { file: "b.md", text: "content b" },
    ];
    const result = await llm.rerank("query", docs);

    expect(result.results).toHaveLength(2);
    for (const r of result.results) {
      expect(r.score).toBe(0.5);
    }
  });

  it("should fall back to 0.5 scores when response JSON is malformed", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "not valid json at all" }]
    });

    const docs = [{ file: "a.md", text: "content" }];
    const result = await llm.rerank("query", docs);
    expect(result.results[0]!.score).toBe(0.5);
  });

  it("should fall back to 0.5 scores when array length mismatches document count", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    // Returns 1 score for 2 documents
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "[0.9]" }]
    });

    const docs = [
      { file: "a.md", text: "content a" },
      { file: "b.md", text: "content b" },
    ];
    const result = await llm.rerank("query", docs);
    for (const r of result.results) {
      expect(r.score).toBe(0.5);
    }
  });

  it("should process documents in batches of 15", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);

    // 20 documents → 2 batches (15 + 5)
    let callCount = 0;
    (piAi.complete as any).mockImplementation((_model: any, ctx: any) => {
      callCount++;
      const msgCount = ctx.messages[0].content.match(/\[\d+\]/g)?.length ?? 0;
      const scores = Array(msgCount).fill(0.5);
      return Promise.resolve({ content: [{ type: "text", text: JSON.stringify(scores) }] });
    });

    const docs = Array.from({ length: 20 }, (_, i) => ({
      file: `doc${i}.md`,
      text: `content ${i}`
    }));

    const result = await llm.rerank("query", docs);
    expect(result.results).toHaveLength(20);
    expect(callCount).toBe(2); // 2 batches
  });

  it("should strip markdown code fences from rerank response", async () => {
    const mockModel = { id: "gpt-4o-mini", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "```json\n[0.8, 0.2]\n```" }]
    });

    const docs = [
      { file: "a.md", text: "relevant" },
      { file: "b.md", text: "irrelevant" },
    ];
    const result = await llm.rerank("query", docs);
    expect(result.results[0]!.file).toBe("a.md");
    expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
  });

  // ---------------------------------------------------------------------------
  // expandQuery
  // ---------------------------------------------------------------------------

  it("should parse expandQuery JSON response into Queryable array", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: JSON.stringify([
        { type: "lex", text: "pizza" },
        { type: "vec", text: "Italian food" },
        { type: "hyde", text: "A document about pizza toppings" },
      ])}]
    });

    const result = await llm.expandQuery("pizza");
    expect(result).toHaveLength(3);
    expect(result.map(q => q.type)).toEqual(["lex", "vec", "hyde"]);
  });

  it("should filter out lex entries when includeLexical is false", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: JSON.stringify([
        { type: "lex", text: "pizza" },
        { type: "vec", text: "Italian food" },
      ])}]
    });

    const result = await llm.expandQuery("pizza", { includeLexical: false });
    expect(result.every(q => q.type !== "lex")).toBe(true);
  });

  it("should fall back to original query on expandQuery parse failure", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "not json" }]
    });

    const result = await llm.expandQuery("pizza");
    expect(result.length).toBeGreaterThan(0);
    expect(result.some(q => q.text === "pizza")).toBe(true);
  });

  it("should strip markdown fences from expandQuery response", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "```json\n[{\"type\":\"vec\",\"text\":\"pizza\"}]\n```" }]
    });

    const result = await llm.expandQuery("pizza");
    expect(result).toHaveLength(1);
    expect(result[0]!.type).toBe("vec");
  });
});
