/**
 * ollama-embed.ts — Ollama-native `/api/embed` protocol adapter.
 *
 * Supports Ollama's native embed contract:
 *   POST /api/embed
 *   Body: { model, input: string | string[] }
 *   Response: { embeddings: number[][], model?, ... }
 *
 * Also accepts OpenAI-compatible fallback shapes from proxy layers:
 *   { data: [{ index?, embedding: number[] }] }
 */

import type { EmbedAdapter } from './types.js';
import type { EmbedOptions, EmbeddingResult } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';

const endpointPathCache = new Map<string, string>();

function cacheKey(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function buildEmbedUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/api/embed')) return [trimmed];
  if (trimmed.endsWith('/embed')) {
    const apiSibling = trimmed.replace(/\/embed$/, '/api/embed');
    return Array.from(new Set([trimmed, apiSibling]));
  }
  if (trimmed.endsWith('/api')) {
    return [`${trimmed}/embed`];
  }
  if (trimmed.endsWith('/v1')) {
    const root = trimmed.replace(/\/v1$/, '');
    return Array.from(new Set([`${root}/api/embed`, `${trimmed}/embeddings`]));
  }

  return Array.from(new Set([
    `${trimmed}/api/embed`,
    `${trimmed}/embed`,
    `${trimmed}/v1/embeddings`,
  ]));
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
}

function isNumericVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((n) => typeof n === 'number' && Number.isFinite(n));
}

function normalizeEmbedVectors(raw: unknown): number[][] {
  if (!raw || typeof raw !== 'object') return [];
  const data = raw as Record<string, unknown>;

  const embeddings = data['embeddings'];
  if (Array.isArray(embeddings) && embeddings.every(isNumericVector)) {
    return embeddings as number[][];
  }
  if (isNumericVector(embeddings)) {
    return [embeddings];
  }

  const embedding = data['embedding'];
  if (isNumericVector(embedding)) {
    return [embedding];
  }

  const legacyData = data['data'];
  if (Array.isArray(legacyData)) {
    const collected: Array<{ index: number; embedding: number[] }> = [];
    let needsSort = false;
    for (const item of legacyData) {
      if (!item || typeof item !== 'object') return [];
      const row = item as Record<string, unknown>;
      const itemEmbedding = row['embedding'];
      if (!isNumericVector(itemEmbedding)) return [];
      const index = typeof row['index'] === 'number' ? row['index'] : collected.length;
      if (index !== collected.length) needsSort = true;
      collected.push({ index, embedding: itemEmbedding });
    }
    if (needsSort) {
      collected.sort((a, b) => a.index - b.index);
    }
    return collected.map((row) => row.embedding);
  }

  return [];
}

function validateAndNormalizeVectors(
  vectors: number[][],
  expectedCount: number,
  dimState: { dimensions: number | null },
): number[][] {
  if (vectors.length !== expectedCount) {
    throw new Error(
      `Embedding count mismatch: expected ${expectedCount}, got ${vectors.length}`,
    );
  }

  for (const vec of vectors) {
    if (!isNumericVector(vec) || vec.length === 0) {
      throw new Error('Invalid embedding vector in response');
    }
    if (dimState.dimensions === null) {
      dimState.dimensions = vec.length;
    } else if (vec.length !== dimState.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${dimState.dimensions}, got ${vec.length}. ` +
        `This usually means the remote model changed. Rebuild the index with 'qmd embed -f'.`,
      );
    }
  }

  return vectors;
}

async function postEmbedWithEndpointFallback(
  baseUrl: string,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  const key = cacheKey(baseUrl);
  const cached = endpointPathCache.get(key);
  const endpoints = orderByCachedFirst(buildEmbedUrlCandidates(baseUrl), cached);

  let lastRecoverableError: Error | null = null;
  for (const endpoint of endpoints) {
    try {
      const response = await nodePost(endpoint, headers, body, timeoutMs);
      endpointPathCache.set(key, endpoint);
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
  throw new Error('Ollama embed request failed for all endpoint candidates');
}

async function sendBatch(
  ctx: Parameters<EmbedAdapter['embedBatch']>[0],
  texts: string[],
  options?: EmbedOptions,
): Promise<(EmbeddingResult | null)[]> {
  const { cfg, breaker, dimState, readTimeoutMs } = ctx;
  const activeModel = options?.model || cfg.model;

  try {
    const response = await postEmbedWithEndpointFallback(
      cfg.baseUrl,
      buildBearerHeaders(cfg.apiKey),
      { model: activeModel, input: texts },
      readTimeoutMs,
    );

    const rawVectors = normalizeEmbedVectors(response);
    const vectors = validateAndNormalizeVectors(rawVectors, texts.length, dimState);
    breaker.onSuccess();
    return vectors.map((embedding) => ({ embedding, model: activeModel }));
  } catch (err) {
    breaker.onFailure();
    throw err;
  }
}

async function sendBatchWithRetry(
  ctx: Parameters<EmbedAdapter['embedBatch']>[0],
  texts: string[],
  options?: EmbedOptions,
): Promise<(EmbeddingResult | null)[]> {
  const { maxRetries, log } = ctx;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await sendBatch(ctx, texts, options);
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (lastError.message.includes('dimension mismatch')) {
        throw lastError;
      }

      if (attempt < maxRetries - 1) {
        const delay = 1000 * Math.pow(2, attempt);
        log.warn(
          `Remote embed batch failed (attempt ${attempt + 1}/${maxRetries}), ` +
          `retrying in ${delay}ms: ${lastError.message}`,
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  log.error(
    `Remote embed batch failed after ${maxRetries} retries: ${lastError?.message}. ` +
    `Returning nulls for ${texts.length} texts.`,
  );
  return texts.map(() => null);
}

export const ollamaEmbedAdapter: EmbedAdapter = {
  id: 'ollama/embed',

  async embedBatch(ctx, texts, options) {
    if (texts.length === 0) return [];

    if (!ctx.breaker.canAttempt()) {
      throw new Error(
        `Remote embedding circuit breaker is open — endpoint ${ctx.cfg.baseUrl} is unavailable. ` +
        `Will retry after cooldown.`,
      );
    }

    const results: (EmbeddingResult | null)[] = [];
    for (let i = 0; i < texts.length; i += ctx.maxBatchSize) {
      const batch = texts.slice(i, i + ctx.maxBatchSize);
      const batchResults = await sendBatchWithRetry(ctx, batch, options);
      results.push(...batchResults);
    }
    return results;
  },
};

