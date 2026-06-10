/**
 * vllm-score.ts — vLLM-compatible Score API adapter.
 *
 * Uses `/score` (with `/v1/score` fallback) and maps response `data[].score`
 * into QMD's rerank result format.
 */

import type { RerankAdapter } from './types.js';
import type { RerankDocument, RerankResult } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';

function cacheKey(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

const scorePathCache = new Map<string, string>();

function buildScoreUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/v1/score')) return [trimmed];
  if (trimmed.endsWith('/score')) {
    const v1Sibling = trimmed.replace(/\/score$/, '/v1/score');
    return Array.from(new Set([trimmed, v1Sibling]));
  }
  if (trimmed.endsWith('/v1')) {
    const sibling = trimmed.replace(/\/v1$/, '/score');
    return Array.from(new Set([`${trimmed}/score`, sibling]));
  }

  return [`${trimmed}/score`, `${trimmed}/v1/score`];
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
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

function normalizeScoreResponse(
  raw: unknown,
  documents: RerankDocument[],
  fallbackModel: string,
): RerankResult {
  if (!raw || typeof raw !== 'object') {
    return { results: [], model: fallbackModel };
  }

  const data = raw as Record<string, unknown>;
  const model = typeof data['model'] === 'string' ? data['model'] : fallbackModel;
  const rawScores = Array.isArray(data['data']) ? data['data'] : [];

  const results = rawScores
    .map((item, i) => {
      if (!item || typeof item !== 'object') return null;
      const row = item as Record<string, unknown>;
      const score = typeof row['score'] === 'number' ? row['score'] : null;
      const index = typeof row['index'] === 'number' ? row['index'] : i;
      if (score === null) return null;
      return {
        index,
        score,
        file: documents[index]?.file ?? `doc-${index}`,
      };
    })
    .filter((r): r is { index: number; score: number; file: string } => r !== null)
    .sort((a, b) => b.score - a.score);

  return { results, model };
}

async function postScoreWithFallback(
  baseUrl: string,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  const key = cacheKey(baseUrl);
  const cached = scorePathCache.get(key);
  const endpoints = orderByCachedFirst(buildScoreUrlCandidates(baseUrl), cached);

  let lastRecoverableError: Error | null = null;
  for (const endpoint of endpoints) {
    try {
      const response = await nodePost(endpoint, headers, body, timeoutMs);
      scorePathCache.set(key, endpoint);
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
  throw new Error('Score request failed for all endpoint candidates');
}

export const vllmScoreAdapter: RerankAdapter = {
  id: 'vllm/score',

  async rerank(ctx, query, documents, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    const activeModel = options?.model || cfg.model;
    const fallback = uniformResult(documents, activeModel);
    if (documents.length === 0) return fallback;

    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: score circuit breaker is open, returning uniform scores');
      return fallback;
    }

    const body: Record<string, unknown> = {
      model: activeModel,
      queries: query,
      documents: documents.map((d) => d.text),
      encoding_format: 'float',
    };

    try {
      const response = await postScoreWithFallback(
        cfg.baseUrl,
        buildBearerHeaders(cfg.apiKey),
        body,
        readTimeoutMs,
      );
      const parsed = normalizeScoreResponse(response, documents, activeModel);
      if (parsed.results.length === 0) {
        throw new Error('vLLM score response contained no valid results');
      }
      breaker.onSuccess();
      return parsed;
    } catch (err) {
      breaker.onFailure();
      log.error(
        'RemoteLLM: score failed:',
        err instanceof Error ? err.message : String(err),
      );
      return fallback;
    }
  },
};
