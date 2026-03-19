/**
 * Integration tests for ModalLLM — the LLM-layer adapter that wraps ModalBackend.
 *
 * Tests cover:
 * - Backend swap: modal.inference=false -> LlamaCpp, modal.inference=true -> ModalLLM
 * - ModalLLM.embed() formats prompt locally then sends raw text to modal backend
 * - ModalLLM.expandQuery() constructs chat-templated prompt with special tokens
 * - ModalLLM.rerank() deduplicates texts, calls modalBackend.rerank(), maps scores back
 * - ModalLLM.modelExists() returns stub info
 * - Startup validation: ping() called, failure -> hard error
 * - withLLMSession uses ModalSession when modal is active
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

// Mock getModalConfig before importing anything from llm.ts
const mockGetModalConfig = vi.fn(() => ({
  inference: false,
  gpu: "T4",
  scaledown_window: 15,
}));

vi.mock("../src/collections.js", () => ({
  getModalConfig: () => mockGetModalConfig(),
}));

// Mock ModalBackend
const mockEmbed = vi.fn();
const mockGenerate = vi.fn();
const mockRerank = vi.fn();
const mockPing = vi.fn();
const mockDispose = vi.fn();

vi.mock("../src/modal.js", () => ({
  ModalBackend: vi.fn(() => ({
    embed: mockEmbed,
    generate: mockGenerate,
    rerank: mockRerank,
    ping: mockPing,
    dispose: mockDispose,
  })),
}));

import {
  ModalLLM,
  getDefaultLLM,
  validateModalConnection,
  getOrCreateModalLLM,
  resetModalLLM,
  type Queryable,
  type RerankDocument,
} from "../src/llm.js";

// ============================================================================
// Helpers
// ============================================================================

function enableModal(): void {
  mockGetModalConfig.mockReturnValue({
    inference: true,
    gpu: "T4",
    scaledown_window: 15,
  });
}

function disableModal(): void {
  mockGetModalConfig.mockReturnValue({
    inference: false,
    gpu: "T4",
    scaledown_window: 15,
  });
}

// ============================================================================
// Tests
// ============================================================================

describe("ModalLLM", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetModalLLM();
    disableModal();
  });

  describe("embed", () => {
    test("formats query text locally then sends to modal backend", async () => {
      const modalLLM = new ModalLLM();
      const fakeVector = [0.1, 0.2, 0.3];
      mockEmbed.mockResolvedValue([fakeVector]);

      const result = await modalLLM.embed("test query", { isQuery: true });

      expect(mockEmbed).toHaveBeenCalledTimes(1);
      // Should have formatted with task prefix before sending
      const sentText = mockEmbed.mock.calls[0]![0][0];
      expect(sentText).toContain("task: search result");
      expect(sentText).toContain("test query");
      expect(result).toEqual({
        embedding: fakeVector,
        model: "modal",
      });
    });

    test("formats document text locally then sends to modal backend", async () => {
      const modalLLM = new ModalLLM();
      const fakeVector = [0.4, 0.5, 0.6];
      mockEmbed.mockResolvedValue([fakeVector]);

      const result = await modalLLM.embed("some document content", {
        isQuery: false,
        title: "My Doc",
      });

      expect(mockEmbed).toHaveBeenCalledTimes(1);
      const sentText = mockEmbed.mock.calls[0]![0][0];
      expect(sentText).toContain("title: My Doc");
      expect(sentText).toContain("some document content");
      expect(result).toEqual({
        embedding: fakeVector,
        model: "modal",
      });
    });

    test("returns null when backend returns empty array", async () => {
      const modalLLM = new ModalLLM();
      mockEmbed.mockResolvedValue([]);

      const result = await modalLLM.embed("test");
      expect(result).toBeNull();
    });
  });

  describe("expandQuery", () => {
    test("constructs Qwen3 chat template with special tokens and grammar", async () => {
      const modalLLM = new ModalLLM();
      mockGenerate.mockResolvedValue(
        'lex: test search terms\nvec: semantic meaning of test\nhyde: A document about testing\n'
      );

      const result = await modalLLM.expandQuery("test");

      expect(mockGenerate).toHaveBeenCalledTimes(1);
      const [prompt, grammar, maxTokens, model] = mockGenerate.mock.calls[0]!;

      // Verify chat template structure with exact special tokens
      expect(prompt).toContain("<|im_start|>system");
      expect(prompt).toContain("You are a helpful assistant.");
      expect(prompt).toContain("<|im_end|>");
      expect(prompt).toContain("<|im_start|>user");
      expect(prompt).toContain("/no_think Expand this search query: test");
      expect(prompt).toContain("<|im_start|>assistant");

      // Verify grammar string
      expect(grammar).toContain('root ::= line+');
      expect(grammar).toContain('"lex" | "vec" | "hyde"');

      // Verify max tokens and model
      expect(maxTokens).toBe(600);
      expect(model).toBe("expand");

      // Verify parsed result
      expect(result).toEqual([
        { type: "lex", text: "test search terms" },
        { type: "vec", text: "semantic meaning of test" },
        { type: "hyde", text: "A document about testing" },
      ]);
    });

    test("returns fallback when generation returns empty/invalid", async () => {
      const modalLLM = new ModalLLM();
      mockGenerate.mockResolvedValue("");

      const result = await modalLLM.expandQuery("my query");

      // Should return fallback queries
      expect(result.length).toBeGreaterThan(0);
      expect(result.some((q: Queryable) => q.type === "vec" || q.type === "hyde")).toBe(true);
    });

    test("filters out lines that do not contain any query terms", async () => {
      const modalLLM = new ModalLLM();
      mockGenerate.mockResolvedValue(
        'lex: cats are fluffy\nvec: search for dogs\nhyde: information about cats\n'
      );

      const result = await modalLLM.expandQuery("cats");

      // "search for dogs" should be filtered out since it doesn't contain "cats"
      expect(result).toEqual([
        { type: "lex", text: "cats are fluffy" },
        { type: "hyde", text: "information about cats" },
      ]);
    });

    test("includes intent in prompt when provided", async () => {
      const modalLLM = new ModalLLM();
      mockGenerate.mockResolvedValue('vec: test vector\n');

      await modalLLM.expandQuery("test", { intent: "find documentation" });

      const prompt = mockGenerate.mock.calls[0]![0];
      expect(prompt).toContain("Query intent: find documentation");
    });

    test("respects includeLexical=false option", async () => {
      const modalLLM = new ModalLLM();
      mockGenerate.mockResolvedValue(
        'lex: test terms\nvec: test semantic\nhyde: about test\n'
      );

      const result = await modalLLM.expandQuery("test", { includeLexical: false });

      expect(result.every((q: Queryable) => q.type !== "lex")).toBe(true);
    });
  });

  describe("rerank", () => {
    test("deduplicates texts, calls modalBackend.rerank(), maps scores back", async () => {
      const modalLLM = new ModalLLM();
      const docs: RerankDocument[] = [
        { file: "a.md", text: "unique text A" },
        { file: "b.md", text: "shared text" },
        { file: "c.md", text: "shared text" },   // duplicate of b
        { file: "d.md", text: "unique text D" },
      ];

      // Backend receives deduplicated texts: ["unique text A", "shared text", "unique text D"]
      // Returns scores in that order
      mockRerank.mockResolvedValue([0.3, 0.9, 0.5]);

      const result = await modalLLM.rerank("my query", docs);

      expect(mockRerank).toHaveBeenCalledTimes(1);
      const [query, texts] = mockRerank.mock.calls[0]!;
      expect(query).toBe("my query");
      // Should have deduplicated
      expect(texts).toEqual(["unique text A", "shared text", "unique text D"]);

      // Results should be sorted by score descending
      // "shared text" (0.9) -> b.md (idx 1), c.md (idx 2)
      // "unique text D" (0.5) -> d.md (idx 3)
      // "unique text A" (0.3) -> a.md (idx 0)
      expect(result.results).toEqual([
        { file: "b.md", score: 0.9, index: 1 },
        { file: "c.md", score: 0.9, index: 2 },
        { file: "d.md", score: 0.5, index: 3 },
        { file: "a.md", score: 0.3, index: 0 },
      ]);
      expect(result.model).toBe("modal");
    });

    test("truncates long documents using character-based approximation", async () => {
      const modalLLM = new ModalLLM();
      // Create a very long document (> typical context window in chars)
      const longText = "x".repeat(20000);
      const docs: RerankDocument[] = [
        { file: "long.md", text: longText },
        { file: "short.md", text: "short" },
      ];

      mockRerank.mockResolvedValue([0.5, 0.8]);

      await modalLLM.rerank("query", docs);

      const [, texts] = mockRerank.mock.calls[0]!;
      // The long text should have been truncated
      expect(texts[0].length).toBeLessThan(longText.length);
      // Short text should be unchanged
      expect(texts[1]).toBe("short");
    });

    test("handles empty documents array", async () => {
      const modalLLM = new ModalLLM();
      mockRerank.mockResolvedValue([]);

      const result = await modalLLM.rerank("query", []);

      expect(result.results).toEqual([]);
      expect(result.model).toBe("modal");
    });
  });

  describe("modelExists", () => {
    test("returns stub info indicating models always exist on Modal", async () => {
      const modalLLM = new ModalLLM();
      const info = await modalLLM.modelExists("any-model");
      expect(info).toEqual({ name: "modal", exists: true });
    });
  });

  describe("dispose", () => {
    test("calls modalBackend.dispose()", async () => {
      const modalLLM = new ModalLLM();
      await modalLLM.dispose();
      expect(mockDispose).toHaveBeenCalledTimes(1);
    });
  });
});

describe("getDefaultLLM", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetModalLLM();
  });

  afterEach(() => {
    disableModal();
  });

  test("returns LlamaCpp when modal.inference=false", () => {
    disableModal();
    const llm = getDefaultLLM();
    // Should NOT be a ModalLLM instance
    expect(llm).not.toBeInstanceOf(ModalLLM);
  });

  test("returns ModalLLM when modal.inference=true", () => {
    enableModal();
    const llm = getDefaultLLM();
    expect(llm).toBeInstanceOf(ModalLLM);
  });

  test("returns same ModalLLM singleton on repeated calls", () => {
    enableModal();
    const a = getDefaultLLM();
    const b = getDefaultLLM();
    expect(a).toBe(b);
  });
});

describe("validateModalConnection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetModalLLM();
  });

  test("calls ping() and resolves on success", async () => {
    mockPing.mockResolvedValue(true);
    await expect(validateModalConnection()).resolves.toBeUndefined();
    expect(mockPing).toHaveBeenCalledTimes(1);
  });

  test("throws when ping() fails", async () => {
    mockPing.mockRejectedValue(new Error("connection refused"));
    await expect(validateModalConnection()).rejects.toThrow(
      /Modal inference.*not reachable/
    );
  });

  test("throws when ping() returns false", async () => {
    mockPing.mockResolvedValue(false);
    await expect(validateModalConnection()).rejects.toThrow(
      /Modal inference.*not reachable/
    );
  });
});

describe("withLLMSession with Modal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetModalLLM();
  });

  afterEach(() => {
    disableModal();
  });

  test("modal session delegates embed to ModalLLM", async () => {
    enableModal();
    const fakeVector = [0.1, 0.2];
    mockEmbed.mockResolvedValue([fakeVector]);

    const { withLLMSession } = await import("../src/llm.js");

    const result = await withLLMSession(async (session) => {
      return session.embed("test", { isQuery: true });
    });

    expect(mockEmbed).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ embedding: fakeVector, model: "modal" });
  });

  test("modal session isValid starts true", async () => {
    enableModal();

    const { withLLMSession } = await import("../src/llm.js");

    await withLLMSession(async (session) => {
      expect(session.isValid).toBe(true);
    });
  });
});
