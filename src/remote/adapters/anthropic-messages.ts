/**
 * anthropic-messages.ts — Anthropic `/v1/messages` protocol adapter.
 *
 * Provides expandQuery and generate implementations that speak the
 * Anthropic Messages API protocol used by Claude models and
 * Anthropic-compatible providers.
 *
 * ## Protocol contract
 *
 *   POST /v1/messages
 *   Headers: x-api-key, anthropic-version
 *   Body: { model, max_tokens (required!), messages[{role,content}], system?, temperature? }
 *   Response: { id, type, role, content[{type,text}], model, stop_reason, usage }
 *
 * ## Key differences from OpenAI chat
 *
 * 1. Endpoint: `/v1/messages` (not `/v1/chat/completions`)
 * 2. Auth: `x-api-key` header (NOT `Authorization: Bearer`)
 * 3. System prompt: top-level `system` string field (NOT a message with role="system")
 * 4. Messages only have `user`/`assistant` roles (no `system` role)
 * 5. Content blocks: response content is always an array of content blocks
 * 6. `max_tokens` is REQUIRED (not optional)
 * 7. Response has `stop_reason` (not `finish_reason`)
 *
 * ## Fallback behavior
 *
 * - **expand**: Returns `expandFallback()` when API key is missing,
 *   circuit breaker is open, or response is unparseable.
 * - **generate**: Returns `null` when circuit breaker is open or on
 *   any error (network, HTTP, JSON parse).
 *
 * @module remote/adapters/anthropic-messages
 */

import type { ExpandAdapter, ExpandAdapterContext, GenerateAdapter, GenerateAdapterContext } from './types.js';
import type { GenerateOptions } from '../../llm.js';
import { nodePost } from '../transport.js';
import { parseExpandResponse, expandFallback } from '../expand.js';
import {
  normalizeAnthropicMessagesText,
  normalizeModelName,
  checkGate,
  handleGenerateError,
  EXPAND_SYSTEM_PROMPT,
  buildExpandUserPrompt,
} from './normalization.js';

// =============================================================================
// Shared helpers
// =============================================================================

/** Anthropic API version header value — required by the API. */
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * Build request headers for Anthropic API.
 * Uses x-api-key (not Authorization: Bearer) and anthropic-version.
 */
function buildHeaders(apiKey: string): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    'x-api-key': apiKey.trim(),
    'anthropic-version': ANTHROPIC_VERSION,
  };
}

/**
 * Build the messages payload for the Anthropic Messages API.
 *
 * Unlike OpenAI, Anthropic:
 * - Has NO "system" role in messages — system prompt is a top-level field
 * - Requires `max_tokens` on every request
 */
function buildMessagesPayload(
  model: string,
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  systemPrompt: string | undefined,
  maxTokens: number,
  temperature: number,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    max_tokens: maxTokens,
    messages,
    temperature,
  };
  if (systemPrompt) {
    body['system'] = systemPrompt;
  }
  return body;
}

// =============================================================================
// Expand adapter — `/v1/messages` for query expansion
// =============================================================================

export const anthropicMessagesExpandAdapter: ExpandAdapter = {
  id: 'anthropic/messages-expand',

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
    const headers = buildHeaders(cfg.apiKey);

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/messages`,
        headers,
        buildMessagesPayload(
          cfg.model,
          [{ role: 'user', content: buildExpandUserPrompt(query, options?.intent) }],
          EXPAND_SYSTEM_PROMPT,
          600,
          0.7,
        ),
        readTimeoutMs,
      );

      const content = normalizeAnthropicMessagesText(data);

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
        'RemoteLLM: expandQuery (anthropic) failed:',
        err instanceof Error ? err.message : String(err),
      );
      return expandFallback(query, includeLexical);
    }
  },
};

// =============================================================================
// Generate adapter — `/v1/messages` for text generation
// =============================================================================

export const anthropicMessagesGenerateAdapter: GenerateAdapter = {
  id: 'anthropic/messages-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;

    // No API key → return null immediately (no HTTP attempt)
    if (!cfg.apiKey) {
      log.warn('RemoteLLM: generate endpoint has no API key, returning null');
      return null;
    }

    // Circuit breaker check
    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
      return null;
    }

    const headers = buildHeaders(cfg.apiKey);

    try {
      const data = await nodePost(
        `${cfg.baseUrl}/messages`,
        headers,
        buildMessagesPayload(
          cfg.model,
          [{ role: 'user', content: prompt }],
          undefined, // no system prompt for generate
          options?.maxTokens ?? 1024,
          options?.temperature ?? 0.7,
        ),
        readTimeoutMs,
      );

      breaker.onSuccess();
      return {
        text: normalizeAnthropicMessagesText(data),
        model: normalizeModelName(data, cfg.model),
        done: true,
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (anthropic)');
    }
  },
};
