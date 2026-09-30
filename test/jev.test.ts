/**
 * jev.test.ts - Tests for the TypeSafe Jev remote reranker
 *
 * Run with: npx vitest run test/jev.test.ts
 */

import { describe, test, expect } from "vitest";
import { buildJevRequest, isJevModel, jevRerank, JEV_ENDPOINT, type JevFetch } from "../src/jev.js";
import { DEFAULT_EMBED_MODEL_URI, DEFAULT_GENERATE_MODEL_URI, DEFAULT_RERANK_MODEL_URI } from "../src/llm.js";
import { gatedModels } from "../src/trust.js";

type RecordedCall = { url: string; init: RequestInit };

// Scores each request by looking up its document text, so parallel ordering can't affect results.
function fakeJev(scoreByText: Map<string, number>, calls: RecordedCall[]): JevFetch {
  return async (url, init) => {
    calls.push({ url, init });
    const { state } = JSON.parse(String(init.body));
    const noul = scoreByText.get(state.document) ?? 0;
    return Response.json({ answers: { relevant: { type: "noul", noul } }, usage: { input_tokens: 10, output_tokens: 1 } });
  };
}

describe("isJevModel", () => {
  test("matches the typesafe: prefix only", () => {
    expect(isJevModel("typesafe:jev-latest")).toBe(true);
    expect(isJevModel("hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf")).toBe(false);
  });
});

describe("buildJevRequest", () => {
  test("asks one noul question about the query/document pair", () => {
    const request = JSON.parse(buildJevRequest("jev-latest", "cap theorem", { file: "a.md", text: "Consistency vs availability" }));
    expect(request.model).toBe("jev-latest");
    expect(request.state).toEqual({ query: "cap theorem", document: "Consistency vs availability" });
    expect(request.questions.relevant.type).toBe("noul");
  });

  test("includes the title when present", () => {
    const request = JSON.parse(buildJevRequest("jev-latest", "q", { file: "a.md", text: "body", title: "Distributed Systems" }));
    expect(request.state.document_title).toBe("Distributed Systems");
  });
});

describe("jevRerank", () => {
  test("scores every document and sorts by relevance, keeping original indexes", async () => {
    const calls: RecordedCall[] = [];
    const scores = new Map([["low", 0.1], ["high", 0.9], ["mid", 0.5]]);
    const documents = [
      { file: "low.md", text: "low" },
      { file: "high.md", text: "high" },
      { file: "mid.md", text: "mid" },
    ];

    const result = await jevRerank("typesafe:jev-latest", "query", documents, "test-key", fakeJev(scores, calls));

    expect(result.model).toBe("typesafe:jev-latest");
    expect(result.results).toEqual([
      { file: "high.md", score: 0.9, index: 1 },
      { file: "mid.md", score: 0.5, index: 2 },
      { file: "low.md", score: 0.1, index: 0 },
    ]);
    expect(calls).toHaveLength(3);
    expect(calls[0]!.url).toBe(JEV_ENDPOINT);
    expect(new Headers(calls[0]!.init.headers).get("Authorization")).toBe("Bearer test-key");
    expect(JSON.parse(String(calls[0]!.init.body)).model).toBe("jev-latest");
  });

  test("throws a clear error when the API key is missing", async () => {
    await expect(jevRerank("typesafe:jev-latest", "q", [{ file: "a.md", text: "a" }], undefined))
      .rejects.toThrow("TYPESAFE_API_KEY is not set");
  });

  test("surfaces API errors instead of returning fallback scores", async () => {
    const failing: JevFetch = async () => new Response("rate limited", { status: 429 });
    await expect(jevRerank("typesafe:jev-latest", "q", [{ file: "a.md", text: "a" }], "test-key", failing))
      .rejects.toThrow("Jev rerank failed (429): rate limited");
  });

  test("rejects a response without a noul score", async () => {
    const malformed: JevFetch = async () => Response.json({ answers: {} });
    await expect(jevRerank("typesafe:jev-latest", "q", [{ file: "a.md", text: "a" }], "test-key", malformed))
      .rejects.toThrow("Jev rerank returned no relevance score");
  });
});

// A cloned repo's .qmd/index.yml must not be able to send chunks to a remote API unapproved.
describe("project-local trust gate", () => {
  test("a typesafe: rerank model requires approval", () => {
    const builtins = { embed: DEFAULT_EMBED_MODEL_URI, rerank: DEFAULT_RERANK_MODEL_URI, generate: DEFAULT_GENERATE_MODEL_URI };
    expect(gatedModels({ rerank: "typesafe:jev-latest" }, builtins)).toEqual([{ slot: "rerank", uri: "typesafe:jev-latest" }]);
  });
});
