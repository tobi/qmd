import { afterEach, describe, expect, test, vi } from "vitest";
import { LlamaCpp, setNodeLlamaCppModuleForTest } from "../src/llm.js";

const remote = {
  provider: "openai" as const,
  endpoint: "http://127.0.0.1:8088/v1",
  model: "Qwen3-Reranker-0.6B-Q4_K_M.gguf",
};

const ok = () => new Response(JSON.stringify({
  model: remote.model,
  results: [
    { index: 1, relevance_score: 0.9 },
    { index: 0, relevance_score: 0.1 },
  ],
}));

afterEach(() => {
  vi.restoreAllMocks();
  setNodeLlamaCppModuleForTest(null);
});

describe("remote rerank routing", () => {
  test("reranks remotely without initializing or resolving a local GGUF", async () => {
    const getLlama = vi.fn();
    const resolveModelFile = vi.fn();
    setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: vi.fn() as never, LlamaLogLevel: { error: 0 } });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok());
    const llm = new LlamaCpp({ rerankModel: remote });

    const result = await llm.rerank("query", [
      { file: "a.md", text: "a" },
      { file: "b.md", text: "b" },
    ]);

    expect(result.results.map(item => item.index)).toEqual([1, 0]);
    expect(result.results.map(item => item.score)).toEqual([0.9, 0.1]);
    expect(getLlama).not.toHaveBeenCalled();
    expect(resolveModelFile).not.toHaveBeenCalled();
  });

  test("fails closed and never falls back to a local reranker", async () => {
    const getLlama = vi.fn();
    const resolveModelFile = vi.fn();
    setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: vi.fn() as never, LlamaLogLevel: { error: 0 } });
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    const llm = new LlamaCpp({ rerankModel: remote });

    await expect(llm.rerank("query", [{ file: "a.md", text: "a" }]))
      .rejects.toMatchObject({ name: "RemoteRerankTransportError" });
    expect(getLlama).not.toHaveBeenCalled();
    expect(resolveModelFile).not.toHaveBeenCalled();
  });
});
