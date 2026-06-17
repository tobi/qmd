/**
 * normalization.ts — Normalization helpers for remote API protocol adapters.
 *
 * Supports both OpenAI-style protocols (chat completions, text completions,
 * responses) and Anthropic's Messages API. Each protocol differs in
 * request/response shapes but shares error-handling semantics,
 * circuit-breaker gating, and fallback behavior.
 *
 * Phase 2 extracted shared patterns. Phase 3 added Anthropic support.
 *
 * @module remote/adapters/normalization
 */

import type { CircuitBreaker } from '../circuit-breaker.js';
import type { Logger } from '../log.js';

// =============================================================================
// Text extraction — each protocol has a different response shape
// =============================================================================

/**
 * Extract the assistant message text from a `/v1/chat/completions` response.
 *
 * Handles two content shapes commonly seen in the wild:
 * - Plain string: `message.content === "hello"` (standard OpenAI shape)
 * - Content blocks: `message.content === [{type:"text", text:"hello"}]`
 *   (used by some providers and when multimodal is enabled)
 *
 * Returns empty string for malformed/missing data.
 */
export function normalizeChatCompletionText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;
  const choices = Array.isArray(d['choices']) ? d['choices'] : [];
  if (choices.length === 0) return '';
  const first = choices[0] as Record<string, unknown> | undefined;
  if (!first) return '';
  const message = first['message'] as Record<string, unknown> | undefined;
  const content = message?.['content'];

  // Plain string — standard OpenAI / vLLM shape
  if (typeof content === 'string') return content;

  // Content blocks — e.g. [{type:"text", text:"hello"}]
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!block || typeof block !== 'object') continue;
      const b = block as Record<string, unknown>;
      if (b['type'] === 'text' && typeof b['text'] === 'string') return b['text'];
    }
  }

  return '';
}

/**
 * Extract generated text from a legacy `/v1/completions` response.
 * The older completions endpoint returns `choices[0].text`.
 */
export function normalizeCompletionsText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;
  const choices = Array.isArray(d['choices']) ? d['choices'] : [];
  if (choices.length === 0) return '';
  const first = choices[0] as Record<string, unknown> | undefined;
  return typeof first?.['text'] === 'string' ? first['text'] : '';
}

/**
 * Extract response text from OpenAI `/v1/responses` API.
 *
 * The responses API has several valid output shapes depending on provider
 * and API version:
 *
 * 1. Standard: `output: [{type:"message", content:[{type:"output_text", text:"..."}]}]`
 * 2. Top-level shortcut: `{output_text: "..."}` (simpler implementations)
 * 3. Content variant: `{type:"text", text:"..."}` inside message content blocks
 *    (seen in some compatible providers)
 *
 * Handles all three. Non-message output types (reasoning, tool_call) are
 * skipped.
 */
export function normalizeResponseAPIText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;

  // Top-level output_text shortcut
  if (typeof d['output_text'] === 'string') return d['output_text'];

  const output = Array.isArray(d['output']) ? d['output'] : [];
  for (const item of output) {
    if (!item || typeof item !== 'object') continue;
    const o = item as Record<string, unknown>;
    if (o['type'] !== 'message') continue;
    const content = Array.isArray(o['content']) ? o['content'] : [];
    for (const block of content) {
      if (!block || typeof block !== 'object') continue;
      const b = block as Record<string, unknown>;
      // Standard shape: type: "output_text"
      if (b['type'] === 'output_text' && typeof b['text'] === 'string') return b['text'];
      // Compatible-provider variant: type: "text"
      if (b['type'] === 'text' && typeof b['text'] === 'string') return b['text'];
    }
  }
  return '';
}

/**
 * Extract model name from any OpenAI-style response.
 * All three protocols include `model` at the top level.
 */
export function normalizeModelName(data: unknown, fallback: string): string {
  if (!data || typeof data !== 'object') return fallback;
  const d = data as Record<string, unknown>;
  return typeof d['model'] === 'string' ? d['model'] : fallback;
}

// =============================================================================
// Anthropic Messages API — text extraction from content blocks
// =============================================================================

/**
 * Extract the assistant message text from an Anthropic `/v1/messages` response.
 *
 * Anthropic's response always wraps the assistant reply in a `content` array
 * of content blocks. The standard text block is `{type: "text", text: "..."}`.
 *
 * Handles:
 * - Standard: `content: [{type:"text", text:"hello"}]`
 * - Multi-block: concatenates all text blocks in order
 * - Gracefully handles empty/missing data
 *
 * Returns empty string for malformed/missing data.
 */
export function normalizeAnthropicMessagesText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;
  const content = Array.isArray(d['content']) ? d['content'] : [];
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== 'object') continue;
    const b = block as Record<string, unknown>;
    if (b['type'] === 'text' && typeof b['text'] === 'string') {
      parts.push(b['text']);
    }
  }
  // Join text blocks deterministically with newline separation to avoid
  // word-merging that raw concatenation (join('')) would cause.
  // Empty/whitespace-only blocks are omitted.
  const trimmed = parts.map(p => p.trim()).filter(p => p.length > 0);
  return trimmed.join('\n');
}

// =============================================================================
// Circuit breaker gating
// =============================================================================

/**
 * Gating result for circuit-breaker checks.
 *
 * Adapters use this to decide whether to proceed with a request,
 * return a null (for generate), or fall back (for expand).
 */
export type GateResult =
  | { allowed: true }
  | { allowed: false; reason: string };

/**
 * Check the circuit breaker and return a gating decision.
 *
 * Used by both expand and generate adapters before making HTTP calls.
 * Adapters that return null on failure (generate) can check `allowed`
 * and return null immediately. Adapters that fall back (expand) can
 * use the reason for logging.
 */
export function checkGate(breaker: CircuitBreaker): GateResult {
  if (!breaker.canAttempt()) {
    return {
      allowed: false,
      reason: `circuit breaker is open, returning fallback`,
    };
  }
  return { allowed: true };
}

// =============================================================================
// Error normalization
// =============================================================================

/**
 * Log an error and report failure to the circuit breaker.
 *
 * Shared error handler for generate adapters, which return null on failure.
 * Returns `null` for convenience.
 */
export function handleGenerateError(
  err: unknown,
  breaker: CircuitBreaker,
  log: Logger,
  context: string,
): null {
  breaker.onFailure();
  log.error(
    `RemoteLLM: ${context} failed:`,
    err instanceof Error ? err.message : String(err),
  );
  return null;
}

// =============================================================================
// Expand system prompt (shared across all expand adapters)
// =============================================================================

/**
 * Standard system prompt for query expansion.
 * Used by all three OpenAI-style expand adapters.
 */
export const EXPAND_SYSTEM_PROMPT =
  'You are a search query expansion assistant. ' +
  'Given a search query, produce expanded variants in EXACTLY this format:\n' +
  'lex: <keyword/BM25 variant>\n' +
  'vec: <semantic paraphrase>\n' +
  'hyde: <one-sentence hypothetical document excerpt>\n\n' +
  'Output only those three lines. No explanation, no extra text.';

/**
 * Build the user prompt for query expansion, with optional intent.
 */
export function buildExpandUserPrompt(query: string, intent?: string): string {
  return intent
    ? `Expand this search query: ${query}\nQuery intent: ${intent}`
    : `Expand this search query: ${query}`;
}
