/**
 * openai-chat.ts — OpenAI `/v1/chat/completions` protocol adapter.
 *
 * Provides expandQuery and generate implementations that speak the
 * standard chat completions protocol used by OpenAI, OpenRouter, vLLM,
 * and most OpenAI-compatible providers.
 *
 * ## Protocol contract
 *
 *   POST /v1/chat/completions
 *   Body: { model, messages[{role,content}], max_tokens, temperature }
 *   Response: { choices[{message:{content}}], model }
 *
 * ## Fallback behavior
 *
 * - **expand**: Returns `expandFallback()` when API key is missing,
 *   circuit breaker is open, or response is unparseable.
 * - **generate**: Returns `null` when circuit breaker is open or on
 *   any error (network, HTTP, JSON parse).
 *
 * @module remote/adapters/openai-chat
 */

import type { ExpandAdapter, ExpandAdapterContext, GenerateAdapter, GenerateAdapterContext } from './types.js';
import type { GenerateOptions } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';
import { parseExpandResponse, expandFallback } from '../expand.js';
import {
  normalizeChatCompletionText,
  normalizeModelName,
  checkGate,
  handleGenerateError,
  EXPAND_SYSTEM_PROMPT,
  buildExpandUserPrompt,
} from './normalization.js';

// =============================================================================
// Expand adapter — `/v1/chat/completions` for query expansion
// =============================================================================

export const openaiChatCompletionsExpandAdapter: ExpandAdapter = {
  id: 'openai/chat-completions-expand',

  async expandQuery(ctx, query, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    // No API key → passthrough
    if (!cfg.apiKey) {
      log.warn('RemoteLLM: expand endpoint has no API key, returning passthrough query');
      return [{ type: 'lex', text: query }];
    }

    // Circuit breaker check
    const gate = checkGate(breaker);
    if (!gate.allowed) {
      log.warn(`RemoteLLM: expand ${gate.reason}`);
      return expandFallback(query, options?.includeLexical ?? true);
    }

    const includeLexical = options?.includeLexical ?? true;

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/chat/completions`,
        buildBearerHeaders(cfg.apiKey),
        {
          model: cfg.model,
          messages: [
            { role: 'system', content: EXPAND_SYSTEM_PROMPT },
            { role: 'user', content: buildExpandUserPrompt(query, options?.intent) },
          ],
          max_tokens: 600,
          temperature: 0.7,
        },
        readTimeoutMs,
      );

      const content = normalizeChatCompletionText(data);

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
// Generate adapter — `/v1/chat/completions` for text generation
// =============================================================================

export const openaiChatCompletionsGenerateAdapter: GenerateAdapter = {
  id: 'openai/chat-completions-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    // Circuit breaker check
    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
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
      );

      breaker.onSuccess();
      return {
        text: normalizeChatCompletionText(data),
        model: normalizeModelName(data, cfg.model),
        done: true,
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (chat)');
    }
  },
};
