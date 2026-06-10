/**
 * openai-completions.ts — Legacy OpenAI `/v1/completions` protocol adapter.
 *
 * The `/v1/completions` endpoint is the older text-completion API that
 * accepts a plain-text prompt (no message roles). Still used by some
 * self-hosted and compatible providers.
 *
 * ## Protocol contract
 *
 *   POST /v1/completions
 *   Body: { model, prompt, max_tokens, temperature }
 *   Response: { choices[{text,index}], model }
 *
 * ## Important behavioral differences vs chat completions
 *
 * - **No system/user separation**: The expand system prompt is prepended
 *   to the user prompt as a single combined `prompt` string.
 * - **Response extraction**: Uses `choices[0].text` instead of
 *   `choices[0].message.content`.
 * - **Fallback semantics are identical**: expand → `expandFallback()`,
 *   generate → `null`.
 *
 * @module remote/adapters/openai-completions
 */

import type { ExpandAdapter, ExpandAdapterContext, GenerateAdapter, GenerateAdapterContext } from './types.js';
import type { GenerateOptions } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';
import { parseExpandResponse, expandFallback } from '../expand.js';
import {
  normalizeCompletionsText,
  normalizeModelName,
  checkGate,
  handleGenerateError,
  EXPAND_SYSTEM_PROMPT,
  buildExpandUserPrompt,
} from './normalization.js';

// =============================================================================
// Expand adapter — `/v1/completions` for query expansion
// =============================================================================

export const openaiCompletionsExpandAdapter: ExpandAdapter = {
  id: 'openai/completions-expand',

  async expandQuery(ctx, query, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    if (!cfg.apiKey) {
      log.warn('RemoteLLM: expand endpoint has no API key, returning passthrough query');
      return [{ type: 'lex', text: query }];
    }

    const gate = checkGate(breaker);
    if (!gate.allowed) {
      log.warn(`RemoteLLM: expand ${gate.reason}`);
      return expandFallback(query, options?.includeLexical ?? true);
    }

    const includeLexical = options?.includeLexical ?? true;

    // Completions endpoint: combine system prompt + user prompt into one string
    const combinedPrompt = `${EXPAND_SYSTEM_PROMPT}\n\n${buildExpandUserPrompt(query, options?.intent)}`;

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/completions`,
        buildBearerHeaders(cfg.apiKey),
        {
          model: cfg.model,
          prompt: combinedPrompt,
          max_tokens: 600,
          temperature: 0.7,
        },
        readTimeoutMs,
      );

      const content = normalizeCompletionsText(data);

      if (content.trim().length > 0) {
        const queryables = parseExpandResponse(content, query, includeLexical);
        if (queryables.length === 0) {
          log.warn(
            'RemoteLLM: expandQuery received response but could not parse any ' +
            `valid query variants. Raw response (first 200 chars): "${content.slice(0, 200)}"`,
          );
        }
        breaker.onSuccess();
        return queryables.length > 0 ? queryables : expandFallback(query, includeLexical);
      }

      log.warn('RemoteLLM: expandQuery returned empty response, using fallback');
      breaker.onSuccess();
      return expandFallback(query, includeLexical);
    } catch (err) {
      breaker.onFailure();
      log.error(
        'RemoteLLM: expandQuery failed:',
        err instanceof Error ? err.message : String(err),
      );
      return expandFallback(query, includeLexical);
    }
  },
};

// =============================================================================
// Generate adapter — `/v1/completions` for text generation
// =============================================================================

export const openaiCompletionsGenerateAdapter: GenerateAdapter = {
  id: 'openai/completions-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
      return null;
    }

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/completions`,
        buildBearerHeaders(cfg.apiKey),
        {
          model: cfg.model,
          prompt,
          max_tokens: options?.maxTokens ?? 1024,
          temperature: options?.temperature ?? 0.7,
        },
        readTimeoutMs,
      );

      breaker.onSuccess();
      return {
        text: normalizeCompletionsText(data),
        model: normalizeModelName(data, cfg.model),
        done: true,
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (completions)');
    }
  },
};
