/**
 * probe.ts — Health checks for remote API endpoints.
 *
 * Two operations:
 * - probe(): Send a test embed to verify connectivity and detect dimensions
 * - modelExists(): Check if a model is available on the remote server
 *
 * @module remote/probe
 */

import type { EmbeddingResult, ModelInfo } from '../llm.js';
import type { EndpointConfig } from './types.js';
import { buildBearerHeaders, nodeGet } from './transport.js';

// =============================================================================
// modelExists
// =============================================================================

/**
 * Check if a model exists on the remote server.
 *
 * Calls GET /models on the embed endpoint's server (with /v1 suffix stripped).
 * Works with vLLM, Ollama, and OpenAI-compatible servers.
 *
 * Falls back to returning `exists: true` if the server doesn't support
 * the /models endpoint — better to try the embed and fail explicitly
 * than to block on a health check.
 *
 * @param cfg   - Embed endpoint config (for baseUrl and apiKey)
 * @param model - Model name to check
 * @returns ModelInfo with exists flag
 */
export async function modelExists(
  cfg: EndpointConfig,
  model: string,
): Promise<ModelInfo> {
  try {
    // Strip /v1 suffix for the /models endpoint (vLLM, OpenAI convention)
    const baseForModels = cfg.baseUrl.replace(/\/v1$/, '');
    const data = await nodeGet(
      `${baseForModels}/models`, buildBearerHeaders(cfg.apiKey), 5000,
    ) as { data?: Array<{ id: string }> };

    if (data.data) {
      const exists = data.data.some(
        (m: { id: string }) => m.id === model,
      );
      return { name: model, exists };
    }
  } catch {
    // Server doesn't support /models — fall through to optimistic default
  }

  return { name: model, exists: true };
}

// =============================================================================
// probe
// =============================================================================

/**
 * Verify the remote server is reachable and return embedding dimensions.
 *
 * Calls the provided embed function with a test string ("dimension-probe").
 * Useful at startup to catch misconfiguration early — a failed probe means
 * subsequent embed calls will also fail.
 *
 * Called asynchronously by createStore() during initialization. Failures
 * are logged as warnings, not thrown, so startup continues gracefully.
 *
 * @param embedFn - Function that takes text and returns EmbeddingResult | null
 *                  (typically RemoteLLM.embed or equivalent)
 * @returns Object with ok (boolean), dimensions (number), and optional error message
 */
export async function probe(
  embedFn: (text: string) => Promise<EmbeddingResult | null>,
): Promise<{ ok: boolean; dimensions: number; error?: string }> {
  try {
    const result = await embedFn('dimension-probe');
    if (result) {
      return { ok: true, dimensions: result.embedding.length };
    }
    return { ok: false, dimensions: 0, error: 'Null result from probe' };
  } catch (err) {
    return { ok: false, dimensions: 0, error: String(err) };
  }
}
