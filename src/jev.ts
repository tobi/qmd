/**
 * jev.ts - Remote reranking via TypeSafe Jev, selected with a `typesafe:<model>` rerank model.
 * Each chunk gets an independent P(relevant), matching the local reranker's per-chunk cache semantics.
 */

import type { RerankDocument, RerankResult } from "./llm.js";

export const JEV_MODEL_PREFIX = "typesafe:";
export const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
const JEV_TIMEOUT_MS = 30_000;

// Narrower than `typeof fetch` so tests can pass a plain function.
export type JevFetch = (url: string, init: RequestInit) => Promise<Response>;

export function isJevModel(modelUri: string): boolean {
  return modelUri.startsWith(JEV_MODEL_PREFIX);
}

// Fields optional: this is untrusted API output, validated in scoreDocument.
type JevNoulResponse = {
  answers?: { relevant?: { noul?: number } };
};

export function buildJevRequest(model: string, query: string, document: RerankDocument): string {
  return JSON.stringify({
    model,
    state: document.title
      ? { query, document_title: document.title, document: document.text }
      : { query, document: document.text },
    questions: {
      relevant: {
        type: "noul",
        instructions: "Does the document contain information that answers or directly addresses the search query?",
      },
    },
  });
}

async function scoreDocument(
  model: string,
  query: string,
  document: RerankDocument,
  apiKey: string,
  fetchImpl: JevFetch,
): Promise<number> {
  const response = await fetchImpl(JEV_ENDPOINT, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: buildJevRequest(model, query, document),
    signal: AbortSignal.timeout(JEV_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`Jev rerank failed (${response.status}): ${await response.text()}`);
  }
  const body = (await response.json()) as JevNoulResponse;
  const noul = body.answers?.relevant?.noul;
  if (typeof noul !== "number") {
    throw new Error(`Jev rerank returned no relevance score: ${JSON.stringify(body)}`);
  }
  return noul;
}

export async function jevRerank(
  modelUri: string,
  query: string,
  documents: RerankDocument[],
  apiKey: string | undefined,
  fetchImpl: JevFetch = fetch,
): Promise<RerankResult> {
  if (!apiKey) {
    throw new Error(
      `TYPESAFE_API_KEY is not set (required for rerank model ${modelUri}).\n` +
      `Set it in your environment, or switch models.rerank back to a local model.`,
    );
  }
  const model = modelUri.slice(JEV_MODEL_PREFIX.length);
  const scores = await Promise.all(
    documents.map((document) => scoreDocument(model, query, document, apiKey, fetchImpl)),
  );
  const results = documents
    .map((document, index) => ({ file: document.file, score: scores[index]!, index }))
    .sort((a, b) => b.score - a.score);
  return { results, model: modelUri };
}
