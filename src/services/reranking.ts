/**
 * Reranking service using LLM-based relevance scoring
 */

import { Database } from 'bun:sqlite';
import type { RerankResponse } from '../models/types.ts';
import { DEFAULT_RERANK_MODEL, OLLAMA_URL } from '../config/constants.ts';
import { progress } from '../config/terminal.ts';
import { getCacheKey } from '../utils/hash.ts';
import { ensureModelAvailable } from './ollama.ts';

// Qwen3-Reranker system prompt
const RERANK_SYSTEM = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`;

/**
 * Format reranking prompt for query and document
 * @param query - Search query
 * @param title - Document title
 * @param doc - Document text
 * @returns Formatted prompt
 */
function formatRerankPrompt(query: string, title: string, doc: string): string {
  return `<Instruct>: Determine if this document from a Shopify knowledge base is relevant to the search query. The query may reference specific Shopify programs, competitions, features, or named concepts (e.g., "Build a Business" competition, "Shop Pay", "Polaris"). Match documents that discuss the queried topic, even if phrasing differs.
<Query>: ${query}
<Document Title>: ${title}
<Document>: ${doc}`;
}

/**
 * Parse reranker response (yes/no with logprobs)
 * @param data - Reranker response
 * @returns Relevance score (0-1)
 */
function parseRerankResponse(data: RerankResponse): number {
  if (!data.logprobs || data.logprobs.length === 0) {
    throw new Error("Reranker response missing logprobs");
  }

  const firstToken = data.logprobs[0];
  const token = firstToken.token.toLowerCase().trim();
  const confidence = Math.exp(firstToken.logprob);

  if (token === "yes") {
    return confidence;
  }
  if (token === "no") {
    return (1 - confidence) * 0.3;
  }

  throw new Error(`Unexpected reranker token: "${token}"`);
}

/**
 * Get cached rerank result
 * @param db - Database instance
 * @param cacheKey - Cache key
 * @returns Cached result or null
 */
function getCachedResult(db: Database, cacheKey: string): string | null {
  const stmt = db.prepare(`SELECT result FROM ollama_cache WHERE hash = ?`);
  const row = stmt.get(cacheKey) as { result: string } | undefined;
  return row?.result || null;
}

/**
 * Store rerank result in cache
 * @param db - Database instance
 * @param cacheKey - Cache key
 * @param result - Result to cache
 */
function setCachedResult(db: Database, cacheKey: string, result: string): void {
  const stmt = db.prepare(`
    INSERT OR REPLACE INTO ollama_cache (hash, result, created_at)
    VALUES (?, ?, ?)
  `);
  stmt.run(cacheKey, result, new Date().toISOString());
}

/**
 * Rerank a single document
 * @param prompt - Formatted rerank prompt
 * @param model - Reranking model
 * @param db - Database for caching (optional)
 * @param retried - Internal retry flag
 * @returns Relevance score
 */
async function rerankSingle(
  prompt: string,
  model: string,
  db?: Database,
  retried: boolean = false
): Promise<number> {
  // Use generate with raw template for qwen3-reranker format
  const fullPrompt = `<|im_start|>system
${RERANK_SYSTEM}<|im_end|>
<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;

  const requestBody = {
    model,
    prompt: fullPrompt,
    raw: true,
    stream: false,
    logprobs: true,
    options: { num_predict: 1 },
  };

  // Check cache
  const cacheKey = db ? getCacheKey(`${OLLAMA_URL}/api/generate`, requestBody) : "";
  if (db && cacheKey) {
    const cached = getCachedResult(db, cacheKey);
    if (cached) {
      const data = JSON.parse(cached) as RerankResponse;
      return parseRerankResponse(data);
    }
  }

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return rerankSingle(prompt, model, db, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json() as RerankResponse;

  // Cache the result
  if (db && cacheKey) {
    setCachedResult(db, cacheKey, JSON.stringify(data));
  }

  return parseRerankResponse(data);
}

/**
 * Rerank multiple documents in parallel
 * @param query - Search query
 * @param documents - Documents to rerank
 * @param model - Reranking model
 * @param db - Database for caching (optional)
 * @returns Reranked documents with scores
 */
export async function rerank(
  query: string,
  documents: { file: string; text: string }[],
  model: string = DEFAULT_RERANK_MODEL,
  db?: Database
): Promise<{ file: string; score: number }[]> {
  const results: { file: string; score: number }[] = [];
  const total = documents.length;
  const PARALLEL = 5;

  process.stderr.write(`Reranking ${total} documents with ${model} (parallel: ${PARALLEL})...\n`);
  progress.indeterminate();

  // Process in parallel batches
  for (let i = 0; i < documents.length; i += PARALLEL) {
    const batch = documents.slice(i, i + PARALLEL);
    const batchResults = await Promise.all(
      batch.map(async (doc) => {
        const prompt = formatRerankPrompt(query, doc.file, doc.text);
        const score = await rerankSingle(prompt, model, db);
        return { file: doc.file, score };
      })
    );

    results.push(...batchResults);

    const pct = ((i + batch.length) / total) * 100;
    progress.set(pct);
  }

  progress.clear();

  // Sort by score descending
  return results.sort((a, b) => b.score - a.score);
}
