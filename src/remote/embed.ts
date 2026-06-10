/**
 * embed.ts — Remote embedding pipeline via OpenAI-compatible /v1/embeddings.
 *
 * Standalone functions that accept config and circuit breaker as parameters.
 * Called by RemoteLLM.embed() / embedBatch() which pass through this.embedCfg
 * and this.embedBreaker.
 *
 * ## Features
 *
 * - Batch splitting (maxBatchSize) for large input arrays
 * - Retry with exponential backoff (up to maxRetries per batch)
 * - Dimension validation: first response locks expectedDimensions; mismatches throw
 * - Model override via EmbedOptions.model for A/B testing
 * - Result ordering preserved via index field from API response
 * - Graceful degradation: returns nulls for a batch after retries exhausted
 *   rather than failing the entire operation
 *
 * @module remote/embed
 */

import type {
  EmbeddingResult,
  EmbedOptions,
} from '../llm.js';
import type { EndpointConfig } from './types.js';
import type { CircuitBreaker } from './circuit-breaker.js';
import type { Logger } from './log.js';
import { consoleLogger } from './log.js';
import { nodePost } from './transport.js';

// =============================================================================
// embedBatch
// =============================================================================

/**
 * Get embeddings for multiple texts via remote API.
 *
 * Splits input into batches of maxBatchSize. Each batch is sent via
 * sendBatch() which retries up to maxRetries times with exponential
 * backoff. Circuit breaker gates all requests. Dimension validation
 * runs on the first successful response.
 *
 * If a batch exhausts all retries, it returns nulls for those texts
 * rather than failing the entire operation. This allows partial progress
 * on large embedding jobs.
 *
 * @param cfg           - Embed endpoint config (baseUrl, model, apiKey)
 * @param breaker       - Circuit breaker for this endpoint
 * @param texts         - Array of texts to embed (order preserved in output)
 * @param maxBatchSize  - Max texts per HTTP request
 * @param readTimeoutMs - HTTP read timeout in ms
 * @param dimState      - Mutable { dimensions: number | null } — set on first response,
 *                        validated on subsequent responses
 * @param maxRetries    - Max retries per batch (default 3)
 * @param log           - Logger instance (defaults to console)
 * @param options       - Optional overrides (options.model replaces cfg.model for this call)
 * @returns Array of EmbeddingResult (or null per-text on failure), same order as input
 * @throws If circuit breaker is open
 */
export async function embedBatch(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  texts: string[],
  maxBatchSize: number,
  readTimeoutMs: number,
  dimState: { dimensions: number | null },
  maxRetries: number = 3,
  log: Logger = consoleLogger,
  options?: EmbedOptions,
): Promise<(EmbeddingResult | null)[]> {
  if (texts.length === 0) return [];

  if (!breaker.canAttempt()) {
    throw new Error(
      `Remote embedding circuit breaker is open — endpoint ${cfg.baseUrl} is unavailable. ` +
      `Will retry after cooldown.`,
    );
  }

  const results: (EmbeddingResult | null)[] = [];
  const modelOverride = options?.model;

  for (let i = 0; i < texts.length; i += maxBatchSize) {
    const batch = texts.slice(i, i + maxBatchSize);
    const batchResults = await sendBatchWithRetry(
      cfg, breaker, batch, readTimeoutMs, dimState, maxRetries, log, modelOverride,
    );
    results.push(...batchResults);
  }

  return results;
}

// =============================================================================
// sendBatchWithRetry — single batch with retry + backoff
// =============================================================================

/**
 * Send a single batch to the embed API with retry and exponential backoff.
 *
 * Retries up to maxRetries times (1s, 2s, 4s, ...). On final failure,
 * returns nulls for all texts in the batch rather than throwing, so the
 * caller can make partial progress.
 *
 * @param cfg           - Embed endpoint config
 * @param breaker       - Circuit breaker (onSuccess/onFailure called)
 * @param texts         - Batch of texts to embed
 * @param readTimeoutMs - HTTP read timeout
 * @param dimState      - Mutable dimension tracker
 * @param maxRetries    - Max retry attempts
 * @param log           - Logger instance
 * @param modelOverride - If provided, overrides cfg.model for this request
 * @returns Array of EmbeddingResult on success, array of nulls on final failure
 */
async function sendBatchWithRetry(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  texts: string[],
  readTimeoutMs: number,
  dimState: { dimensions: number | null },
  maxRetries: number,
  log: Logger,
  modelOverride?: string,
): Promise<(EmbeddingResult | null)[]> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await sendBatch(
        cfg, breaker, texts, readTimeoutMs, dimState, modelOverride,
      );
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      // Dimension mismatch is a fatal config error — do not retry
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

  // All retries exhausted — return nulls for this batch
  log.error(
    `Remote embed batch failed after ${maxRetries} retries: ${lastError?.message}. ` +
    `Returning nulls for ${texts.length} texts.`,
  );
  return texts.map(() => null);
}

// =============================================================================
// sendBatch — single HTTP call to /v1/embeddings
// =============================================================================

/**
 * Send a single batch to the embed API with dimension validation.
 *
 * @param cfg           - Embed endpoint config
 * @param breaker       - Circuit breaker (onSuccess/onFailure called)
 * @param texts         - Batch of texts to embed
 * @param readTimeoutMs - HTTP read timeout
 * @param dimState      - Mutable dimension tracker
 * @param modelOverride - If provided, overrides cfg.model for this request
 * @returns Array of EmbeddingResult (never null for individual items on success)
 * @throws On network errors, HTTP errors, or dimension mismatch
 */
async function sendBatch(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  texts: string[],
  readTimeoutMs: number,
  dimState: { dimensions: number | null },
  modelOverride?: string,
): Promise<(EmbeddingResult | null)[]> {
  const headers: Record<string, string> = {};
  if (cfg.apiKey) {
    headers['Authorization'] = `Bearer ${cfg.apiKey.trim()}`;
  }

  const activeModel = modelOverride || cfg.model;

  try {
    const data = await nodePost(
      `${cfg.baseUrl}/embeddings`,
      headers,
      { model: activeModel, input: texts },
      readTimeoutMs,
    ) as { data: Array<{ embedding: number[]; index?: number }> };

    // Dimension validation: lock expected dimensions on first response
    if (data.data.length > 0) {
      const dim = data.data[0]!.embedding.length;
      if (dimState.dimensions === null) {
        dimState.dimensions = dim;
      } else if (dim !== dimState.dimensions) {
        throw new Error(
          `Embedding dimension mismatch: expected ${dimState.dimensions}, got ${dim}. ` +
          `This usually means the remote model changed. Rebuild the index with 'qmd embed -f'.`,
        );
      }
    }

    // Sort by index to preserve input order (API may reorder)
    const sorted = [...data.data].sort(
      (a, b) => (a.index ?? 0) - (b.index ?? 0),
    );
    const results: (EmbeddingResult | null)[] = sorted.map((item) => ({
      embedding: item.embedding,
      model: activeModel,
    }));

    breaker.onSuccess();
    return results;
  } catch (err) {
    breaker.onFailure();
    throw err;
  }
}
