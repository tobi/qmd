/**
 * rerank.ts — Document reranking via Cohere-compatible /v1/rerank endpoint.
 *
 * Sends a query and candidate documents to a remote reranker (typically
 * OpenRouter or Cohere) and returns relevance scores.
 *
 * ## Graceful degradation
 *
 * - No API key configured → returns uniform scores (1.0 for all docs)
 * - Circuit breaker open → returns uniform scores (1.0 for all docs)
 * - Network error → logs error, returns uniform scores (1.0 for all docs)
 *
 * This ensures search still returns results even when reranking is
 * unavailable — they just won't be relevance-sorted.
 *
 * @module remote/rerank
 */

import type { RerankResult, RerankDocument } from '../llm.js';
import type { EndpointConfig } from './types.js';
import type { CircuitBreaker } from './circuit-breaker.js';
import type { Logger } from './log.js';
import { consoleLogger } from './log.js';
import { nodePost } from './transport.js';

// =============================================================================
// rerank
// =============================================================================

/**
 * Rerank documents by relevance to a query using a remote reranking API.
 *
 * Uses the Cohere-compatible /v1/rerank endpoint format. Works with
 * OpenRouter, Cohere, and any compatible provider.
 *
 * @param cfg           - Rerank endpoint config
 * @param breaker       - Circuit breaker for this endpoint
 * @param query         - Search query
 * @param documents     - Documents to rerank (each has file and text fields)
 * @param readTimeoutMs - HTTP read timeout in ms
 * @param log           - Logger instance (defaults to console)
 * @returns RerankResult with scored documents
 */
export async function rerank(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  query: string,
  documents: RerankDocument[],
  readTimeoutMs: number,
  log: Logger = consoleLogger,
): Promise<RerankResult> {
  // Build uniform-scores result once, reused across all fallback paths
  const uniformResult: RerankResult = {
    results: documents.map((doc, i) => ({
      file: doc.file,
      score: 1.0,
      index: i,
    })),
    model: cfg.model,
  };

  if (!cfg.apiKey) {
    log.warn(
      'RemoteLLM: rerank endpoint has no API key, returning uniform scores',
    );
    return uniformResult;
  }

  if (!breaker.canAttempt()) {
    log.warn(
      'RemoteLLM: rerank circuit breaker is open, returning uniform scores',
    );
    return uniformResult;
  }

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (cfg.apiKey) {
      headers['Authorization'] = `Bearer ${cfg.apiKey.trim()}`;
    }

    const data = await nodePost(
      `${cfg.baseUrl}/rerank`,
      headers,
      {
        model: cfg.model,
        query,
        documents: documents.map((d) => d.text),
        top_n: documents.length,
      },
      readTimeoutMs,
    ) as {
      results: Array<{ index: number; relevance_score: number }>;
      model?: string;
    };

    breaker.onSuccess();
    return {
      results: data.results.map((r) => ({
        index: r.index,
        score: r.relevance_score,
        file: documents[r.index]?.file ?? `doc-${r.index}`,
      })),
      model: data.model || cfg.model,
    };
  } catch (err) {
    breaker.onFailure();
    log.error(
      'RemoteLLM: rerank failed:',
      err instanceof Error ? err.message : String(err),
    );
    return uniformResult;
  }
}
