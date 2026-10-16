import { describe, expect, test, vi } from "vitest";
import { LlamaCpp, type RerankDocument, type RerankResult, type RerankTokenBudget } from "../src/llm.js";
import { boundedPassageWindow, fitPassageWindow } from "../src/search-locations.js";
import { createStore, formatRerankQuery } from "../src/store.js";

interface TestReranker {
  _ciMode: boolean;
  touchActivity(): void;
  ensureRerankModel(): Promise<{
    tokenize(text: string): string[];
    detokenize(tokens: string[]): string;
  }>;
  ensureRerankContexts(): Promise<{
    rankAll(query: string, texts: string[]): Promise<number[]>;
  }[]>;
  getRerankTokenBudget(query: string): Promise<RerankTokenBudget>;
  rerank(query: string, documents: RerankDocument[]): Promise<RerankResult>;
}

describe("source passages through the rerank model", () => {
  test("keeps a centered anchor in the exact text scored by rankAll", async () => {
    const llm = new LlamaCpp({});
    const reranker = llm as TestReranker;
    const detokenize = vi.fn((tokens: string[]) => tokens.join(""));
    const rankAll = vi.fn(async (_query: string, texts: string[]) => texts.map(() => 0.75));
    reranker._ciMode = false;
    reranker.touchActivity = () => {};
    reranker.ensureRerankModel = async () => ({ tokenize: text => Array.from(text), detokenize });
    reranker.ensureRerankContexts = async () => [{ rankAll }];

    try {
      const query = "Orbit command";
      const marker = "Orbit command";
      const body = `${"filler ".repeat(8_000)}${marker}${" filler".repeat(8_000)}`;
      const anchor = {
        startUtf16: body.indexOf(marker),
        endUtf16: body.indexOf(marker) + marker.length,
      };
      const budget = { maxUtf8Bytes: 48_000 };
      const tokenBudget = await llm.getRerankTokenBudget(query);
      const unfitted = boundedPassageWindow(body, anchor, budget);
      expect(tokenBudget.countTokens(unfitted.text)).toBeGreaterThan(tokenBudget.maxDocumentTokens);
      expect(unfitted.text.slice(0, tokenBudget.maxDocumentTokens)).not.toContain(marker);

      const passage = fitPassageWindow(body, anchor, budget, tokenBudget);
      expect(passage.text).toContain(marker);
      expect(passage.text).toBe(body.slice(passage.startUtf16, passage.endUtf16));
      expect(tokenBudget.countTokens(passage.text)).toBeLessThanOrEqual(tokenBudget.maxDocumentTokens);

      await llm.rerank(query, [{ file: "qmd://docs/orbit.md", text: passage.text }]);
      expect(rankAll).toHaveBeenCalledWith(query, [passage.text]);
      expect(detokenize).not.toHaveBeenCalled();
    } finally {
      await llm.dispose();
    }
  });

  test("budgets the same store-selected model and intent-prefixed query as scoring", async () => {
    const store = createStore(":memory:");
    const llm = new LlamaCpp({});
    const reranker = llm as TestReranker;
    const tokenize = vi.fn((text: string) => Array.from(text));
    reranker._ciMode = false;
    reranker.touchActivity = () => {};
    reranker.ensureRerankModel = async () => ({ tokenize, detokenize: tokens => tokens.join("") });
    store.llm = llm;

    try {
      const query = "Orbit";
      const intent = "configuration commands";
      const plain = await store.getRerankTokenBudget(query);
      const scoped = await store.getRerankTokenBudget(query, intent);
      expect(tokenize).toHaveBeenCalledWith(formatRerankQuery(query, intent));
      expect(plain.maxDocumentTokens - scoped.maxDocumentTokens).toBe(intent.length + 2);
      expect(scoped.countTokens("😀Orbit")).toBe(6);
      await expect(llm.getRerankTokenBudget("x".repeat(plain.maxDocumentTokens + query.length + 1)))
        .rejects.toThrow("Rerank query exceeds the context window");
    } finally {
      store.close();
      await llm.dispose();
    }
  });
});
