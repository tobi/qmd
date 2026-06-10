/**
 * generate.ts — Text generation via remote chat completions API.
 *
 * **Experimental**: This module is fully implemented but not yet wired into
 * any QMD search pipeline. It exists for future use cases like answer
 * generation from retrieved documents.
 *
 * Uses the generate endpoint config (defaults to OpenRouter with
 * google/gemini-2.0-flash-lite-001 if not explicitly configured).
 *
 * @module remote/generate
 */

import type { GenerateResult, GenerateOptions } from '../llm.js';
import type { EndpointConfig } from './types.js';
import type { CircuitBreaker } from './circuit-breaker.js';
import type { Logger } from './log.js';
import { consoleLogger } from './log.js';
import { buildBearerHeaders, nodePost } from './transport.js';

// =============================================================================
// generate
// =============================================================================

/**
 * Generate text via chat completions API.
 *
 * @param cfg           - Generate endpoint config
 * @param breaker       - Circuit breaker for fault isolation
 * @param prompt        - User prompt text
 * @param readTimeoutMs - HTTP read timeout in ms
 * @param log           - Logger instance (defaults to console)
 * @param options       - Generation options (maxTokens, temperature)
 * @returns GenerateResult with text and model name, or null on failure
 */
export async function generate(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  prompt: string,
  readTimeoutMs: number,
  log: Logger = consoleLogger,
  options?: GenerateOptions,
): Promise<GenerateResult | null> {
  if (!breaker.canAttempt()) {
    log.warn(
      'RemoteLLM: generate circuit breaker is open, returning null',
    );
    return null;
  }

  try {
    const data = await nodePost(
      `${cfg.baseUrl}/chat/completions`,
      buildBearerHeaders(cfg.apiKey),
      {
        model: cfg.model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: options?.maxTokens ?? 1024,
        temperature: options?.temperature ?? 0.7,
      },
      readTimeoutMs,
    ) as {
      choices: Array<{ message: { content: string } }>;
      model: string;
    };

    breaker.onSuccess();
    return {
      text: data.choices[0]?.message?.content ?? '',
      model: data.model || cfg.model,
      done: true,
    };
  } catch (err) {
    breaker.onFailure();
    log.error(
      'RemoteLLM: generate failed:',
      err instanceof Error ? err.message : String(err),
    );
    return null;
  }
}
