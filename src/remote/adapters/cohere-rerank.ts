/**
 * cohere-rerank.ts — Cohere-compatible rerank protocol adapter.
 *
 * Supports `/rerank`, `/v1/rerank`, and `/v2/rerank` endpoint variants
 * and caches the last successful path per base URL.
 */

import type { RerankAdapter } from './types.js';
import type { RerankDocument, RerankResult } from '../../llm.js';
import { nodePost } from '../transport.js';

const rerankPathCache = new Map<string, string>();

function cacheKey(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function buildRerankUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/v2/rerank')) return [trimmed];
  if (trimmed.endsWith('/v1/rerank')) return [trimmed];
  if (trimmed.endsWith('/rerank')) {
    const v1Sibling = trimmed.replace(/\/rerank$/, '/v1/rerank');
    const v2Sibling = trimmed.replace(/\/rerank$/, '/v2/rerank');
    return Array.from(new Set([trimmed, v1Sibling, v2Sibling]));
  }
  if (trimmed.endsWith('/v2')) return [`${trimmed}/rerank`, `${trimmed}/v2/rerank`];
  if (trimmed.endsWith('/v1')) return [`${trimmed}/rerank`, `${trimmed}/v1/rerank`];

  return [
    `${trimmed}/rerank`,
    `${trimmed}/v1/rerank`,
    `${trimmed}/v2/rerank`,
  ];
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // Common path/contract mismatch statuses across provider variants.
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
}

function normalizeRerankResponse(
  raw: unknown,
  documents: RerankDocument[],
  fallbackModel: string,
): RerankResult {
  if (!raw || typeof raw !== 'object') {
    return { results: [], model: fallbackModel };
  }

  const data = raw as Record<string, unknown>;
  const model = typeof data['model'] === 'string' ? data['model'] : fallbackModel;
  const rawResults = Array.isArray(data['results']) ? data['results'] : [];

  if (rawResults.length === 0) {
    return { results: [], model };
  }

  const results = rawResults
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const r = item as Record<string, unknown>;
      const index = typeof r['index'] === 'number' ? r['index'] : null;
      const score =
        typeof r['relevance_score'] === 'number'
          ? r['relevance_score']
          : (typeof r['score'] === 'number' ? r['score'] : null);
      if (index === null || score === null) return null;
      return {
        index,
        score,
        file: documents[index]?.file ?? `doc-${index}`,
      };
    })
    .filter((r): r is { index: number; score: number; file: string } => r !== null);

  return { results, model };
}

function uniformResult(documents: RerankDocument[], model: string): RerankResult {
  return {
    results: documents.map((doc, i) => ({
      file: doc.file,
      score: 1.0,
      index: i,
    })),
    model,
  };
}

async function postRerankWithEndpointFallback(
  baseUrl: string,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  const key = cacheKey(baseUrl);
  const cached = rerankPathCache.get(key);
  const endpoints = orderByCachedFirst(buildRerankUrlCandidates(baseUrl), cached);

  let lastRecoverableError: Error | null = null;
  for (const endpoint of endpoints) {
    try {
      const response = await nodePost(endpoint, headers, body, timeoutMs);
      rerankPathCache.set(key, endpoint);
      return response;
    } catch (err) {
      if (shouldTryNextEndpoint(err)) {
        lastRecoverableError = err instanceof Error ? err : new Error(String(err));
        continue;
      }
      throw err;
    }
  }

  if (lastRecoverableError) throw lastRecoverableError;
  throw new Error('Rerank request failed for all endpoint candidates');
}

export const cohereRerankAdapter: RerankAdapter = {
  id: 'cohere/rerank',

  async rerank(ctx, query, documents, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    const activeModel = options?.model || cfg.model;
    const fallback = uniformResult(documents, activeModel);
    if (documents.length === 0) return fallback;

    if (!cfg.apiKey) {
      log.warn(
        'RemoteLLM: rerank endpoint has no API key, returning uniform scores',
      );
      return fallback;
    }

    if (!breaker.canAttempt()) {
      log.warn(
        'RemoteLLM: rerank circuit breaker is open, returning uniform scores',
      );
      return fallback;
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${cfg.apiKey.trim()}`,
    };

    const body: Record<string, unknown> = {
      model: activeModel,
      query,
      documents: documents.map((d) => d.text),
      top_n: documents.length,
    };

    try {
      const response = await postRerankWithEndpointFallback(
        cfg.baseUrl,
        headers,
        body,
        readTimeoutMs,
      );
      const parsed = normalizeRerankResponse(response, documents, activeModel);
      if (parsed.results.length === 0) {
        throw new Error('Cohere rerank response contained no valid results');
      }
      breaker.onSuccess();
      return parsed;
    } catch (err) {
      breaker.onFailure();
      log.error(
        'RemoteLLM: rerank failed:',
        err instanceof Error ? err.message : String(err),
      );
      return fallback;
    }
  },
};
