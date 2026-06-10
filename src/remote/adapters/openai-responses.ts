/**
 * openai-responses.ts — OpenAI `/v1/responses` protocol adapter.
 *
 * The `/v1/responses` endpoint is OpenAI's newer API that replaces the
 * older Assistants API. It uses a different request/response shape than
 * chat completions while providing similar functionality.
 *
 * ## Protocol contract
 *
 *   POST /v1/responses
 *   Body: { model, instructions?, input, max_output_tokens, temperature }
 *   Response: { output[{type,content[{type,text}]}], model }
 *
 * ## Key differences from chat completions
 *
 * - **Request**: Uses `instructions` (system) + `input` (user) instead
 *   of `messages` array. For simple generation, `input` is a plain
 *   string containing the user prompt.
 * - **Response**: Nested typed blocks: `output[].content[].text` with
 *   each block having a `type` discriminator (`output_text`, etc.).
 * - **Token parameter**: `max_output_tokens` instead of `max_tokens`.
 *
 * @module remote/adapters/openai-responses
 */

import type { ExpandAdapter, ExpandAdapterContext, GenerateAdapter, GenerateAdapterContext } from './types.js';
import type { GenerateOptions } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';
import { parseExpandResponse, expandFallback } from '../expand.js';
import {
  normalizeResponseAPIText,
  normalizeModelName,
  checkGate,
  handleGenerateError,
  EXPAND_SYSTEM_PROMPT,
  buildExpandUserPrompt,
} from './normalization.js';

// =============================================================================
// Expand adapter — `/v1/responses` for query expansion
// =============================================================================

export const openaiResponsesExpandAdapter: ExpandAdapter = {
  id: 'openai/responses-expand',

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

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/responses`,
        buildBearerHeaders(cfg.apiKey),
        {
          model: cfg.model,
          instructions: EXPAND_SYSTEM_PROMPT,
          input: buildExpandUserPrompt(query, options?.intent),
          max_output_tokens: 600,
          temperature: 0.7,
        },
        readTimeoutMs,
      );

      const content = normalizeResponseAPIText(data);

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
// Generate adapter — `/v1/responses` for text generation
// =============================================================================

export const openaiResponsesGenerateAdapter: GenerateAdapter = {
  id: 'openai/responses-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
      return null;
    }

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/responses`,
        buildBearerHeaders(cfg.apiKey),
        {
          model: cfg.model,
          input: prompt,
          max_output_tokens: options?.maxTokens ?? 1024,
          temperature: options?.temperature ?? 0.7,
        },
        readTimeoutMs,
      );

      breaker.onSuccess();
      return {
        text: normalizeResponseAPIText(data),
        model: normalizeModelName(data, cfg.model),
        done: true,
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (responses)');
    }
  },
};
