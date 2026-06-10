/**
 * cohere-embed.ts — Cohere-compatible `/v2/embed` protocol adapter.
 *
 * Supports Cohere and vLLM-compatible Embed v2 servers with adaptive
 * request-shape fallback:
 * - `inputs: [{ content: [{ type: "text", text }] }]`
 * - `texts: string[]`
 *
 * It also caches the last successful endpoint path and request shape
 * per base URL to avoid repeated probe attempts.
 */

import type { EmbedAdapter, EmbedAdapterContext } from './types.js';
import type { EmbedOptions, EmbeddingResult } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';

type CohereEmbedMode = 'inputs' | 'texts';
type CohereInputType =
  | 'search_document'
  | 'search_query'
  | 'document'
  | 'query'
  | 'classification'
  | 'clustering'
  | 'image';

const endpointPathCache = new Map<string, string>();
const requestModeCache = new Map<string, CohereEmbedMode>();
const inputTypeCache = new Map<string, CohereInputType>();
const inputTypeFamilyCache = new Map<string, 'cohere_search' | 'generic_plain'>();

function cacheKey(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function buildEmbedUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/v2/embed')) return [trimmed];
  if (trimmed.endsWith('/v1/embed')) {
    const root = trimmed.replace(/\/v1\/embed$/, '');
    return [`${root}/v2/embed`];
  }
  if (trimmed.endsWith('/v1')) {
    const root = trimmed.replace(/\/v1$/, '');
    return [`${root}/v2/embed`];
  }
  if (trimmed.endsWith('/v2')) {
    return [`${trimmed}/embed`];
  }
  if (trimmed.endsWith('/embed')) {
    const sibling = trimmed.replace(/\/embed$/, '/v2/embed');
    return [sibling];
  }

  return [`${trimmed}/v2/embed`];
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function orderModesByCachedFirst(cached?: CohereEmbedMode): CohereEmbedMode[] {
  const modes: CohereEmbedMode[] = ['inputs', 'texts'];
  if (!cached || !modes.includes(cached)) return modes;
  return [cached, ...modes.filter((m) => m !== cached)];
}

function orderInputTypesByCachedFirst(
  values: CohereInputType[],
  cached?: CohereInputType,
): CohereInputType[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextShapeOrEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // Schema/path mismatches should try another contract shape or endpoint path.
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
}

function shouldSkipToNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // Not-found / method-not-allowed strongly indicate wrong endpoint path.
  return /HTTP (404|405|501)\b/.test(err.message);
}

function shouldTryNextInputType(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // vLLM-compatible servers may reject Cohere's search_* values and accept
  // model-native keys (e.g. "document"/"query").
  return /unsupported input_type/i.test(err.message);
}

function isCohereHost(baseUrl: string): boolean {
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host.includes('cohere.ai') || host.includes('cohere.com');
  } catch {
    return false;
  }
}

function buildInputTypeCandidates(baseUrl: string, options?: EmbedOptions): CohereInputType[] {
  const key = cacheKey(baseUrl);
  const cachedFamily = inputTypeFamilyCache.get(key);
  const preferCohereSearchTypes = cachedFamily
    ? cachedFamily === 'cohere_search'
    : isCohereHost(baseUrl);
  if (options?.isQuery) {
    return preferCohereSearchTypes ? ['search_query', 'query'] : ['query', 'search_query'];
  }
  return preferCohereSearchTypes ? ['search_document', 'document'] : ['document', 'search_document'];
}

function buildEmbedBody(
  model: string,
  texts: string[],
  inputType: CohereInputType,
  mode: CohereEmbedMode,
): Record<string, unknown> {
  const base: Record<string, unknown> = {
    model,
    input_type: inputType,
    embedding_types: ['float'],
  };

  if (mode === 'inputs') {
    return {
      ...base,
      inputs: texts.map((text) => ({
        content: [{ type: 'text', text }],
      })),
    };
  }

  return {
    ...base,
    texts,
  };
}

function isNumericVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((n) => typeof n === 'number' && Number.isFinite(n));
}

function normalizeCohereVectors(raw: unknown): number[][] {
  if (!raw || typeof raw !== 'object') return [];
  const data = raw as Record<string, unknown>;

  // Cohere v2 primary shape:
  // { embeddings: { float: number[][], int8?: number[][], ... } }
  const embeddings = data['embeddings'];
  if (embeddings && typeof embeddings === 'object') {
    const e = embeddings as Record<string, unknown>;
    const preferredTypes = ['float', 'int8', 'uint8', 'binary', 'ubinary'];
    for (const key of preferredTypes) {
      const candidate = e[key];
      if (Array.isArray(candidate) && candidate.every(isNumericVector)) {
        return candidate as number[][];
      }
    }
  }

  // Some compat providers return { embeddings: number[][] }.
  if (Array.isArray(embeddings) && embeddings.every(isNumericVector)) {
    return embeddings as number[][];
  }

  // OpenAI-style fallback from some adapters/proxies: { data: [{ embedding }] }.
  const legacyData = data['data'];
  if (Array.isArray(legacyData)) {
    const collected: Array<{ index: number; embedding: number[] }> = [];
    let needsSort = false;
    for (const item of legacyData) {
      if (!item || typeof item !== 'object') return [];
      const row = item as Record<string, unknown>;
      const embedding = row['embedding'];
      if (!isNumericVector(embedding)) return [];
      const index = typeof row['index'] === 'number' ? row['index'] : collected.length;
      if (index !== collected.length) needsSort = true;
      collected.push({ index, embedding });
    }
    if (needsSort) {
      collected.sort((a, b) => a.index - b.index);
    }
    return collected.map((row) => row.embedding);
  }

  return [];
}

async function postEmbedWithFallback(
  ctx: EmbedAdapterContext,
  model: string,
  texts: string[],
  options?: EmbedOptions,
): Promise<number[][]> {
  const { cfg, readTimeoutMs } = ctx;
  const key = cacheKey(cfg.baseUrl);
  const cachedEndpoint = endpointPathCache.get(key);
  const cachedMode = requestModeCache.get(key);
  const inputTypeKey = `${key}|${options?.isQuery === true ? 'query' : 'document'}`;
  const cachedInputType = inputTypeCache.get(inputTypeKey);

  const endpoints = orderByCachedFirst(
    buildEmbedUrlCandidates(cfg.baseUrl),
    cachedEndpoint,
  );
  const modes = orderModesByCachedFirst(cachedMode);
  const inputTypes = orderInputTypesByCachedFirst(
    buildInputTypeCandidates(cfg.baseUrl, options),
    cachedInputType,
  );
  const headers = buildBearerHeaders(cfg.apiKey);

  let lastRecoverableError: Error | null = null;

  for (const endpoint of endpoints) {
    for (const mode of modes) {
      for (const inputType of inputTypes) {
        try {
          const response = await nodePost(
            endpoint,
            headers,
            buildEmbedBody(model, texts, inputType, mode),
            readTimeoutMs,
          );
          const vectors = normalizeCohereVectors(response);
          if (vectors.length === 0) {
            throw new Error('Cohere embed response contained no numeric embeddings');
          }
          if (inputType.startsWith('search_')) {
            inputTypeFamilyCache.set(key, 'cohere_search');
          } else {
            inputTypeFamilyCache.set(key, 'generic_plain');
          }
          endpointPathCache.set(key, endpoint);
          requestModeCache.set(key, mode);
          inputTypeCache.set(inputTypeKey, inputType);
          return vectors;
        } catch (err) {
          if (shouldTryNextShapeOrEndpoint(err)) {
            lastRecoverableError = err instanceof Error ? err : new Error(String(err));
            if (shouldTryNextInputType(err)) {
              if (inputType.startsWith('search_')) {
                inputTypeFamilyCache.set(key, 'generic_plain');
              }
              continue;
            }
            if (shouldSkipToNextEndpoint(err)) {
              break;
            }
            continue;
          }
          throw err;
        }
      }
    }
  }

  if (lastRecoverableError) throw lastRecoverableError;
  throw new Error('Cohere embed request failed for all endpoint/payload candidates');
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

async function sendBatch(
  ctx: EmbedAdapterContext,
  texts: string[],
  options?: EmbedOptions,
): Promise<(EmbeddingResult | null)[]> {
  const { cfg, breaker, dimState } = ctx;
  const activeModel = options?.model || cfg.model;

  try {
    const rawVectors = await postEmbedWithFallback(ctx, activeModel, texts, options);
    const vectors = validateAndNormalizeVectors(rawVectors, texts.length, dimState);
    breaker.onSuccess();
    return vectors.map((embedding) => ({ embedding, model: activeModel }));
  } catch (err) {
    breaker.onFailure();
    throw err;
  }
}

async function sendBatchWithRetry(
  ctx: EmbedAdapterContext,
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

      // Dimension mismatch is a hard incompatibility; retries won't help.
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

export const cohereV2EmbedAdapter: EmbedAdapter = {
  id: 'cohere/v2-embed',

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
