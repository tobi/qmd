/**
 * expand.ts — Query expansion via remote chat completions API.
 *
 * Sends a search query to an LLM (typically via OpenRouter) and parses the
 * response into typed Queryable variants: lex (keyword), vec (semantic),
 * hyde (hypothetical document excerpt).
 *
 * ## Pipeline
 *
 * 1. Check API key → fallback to passthrough if missing
 * 2. Check circuit breaker → fallback to expandFallback() if open
 * 3. POST /v1/chat/completions with a system prompt for lex/vec/hyde output
 * 4. Parse response via parseExpandResponse() with term-overlap validation
 * 5. On parse failure → warn + fallback to expandFallback()
 *
 * @module remote/expand
 */

import type { Queryable } from '../llm.js';
import type { EndpointConfig } from './types.js';
import type { CircuitBreaker } from './circuit-breaker.js';
import type { Logger } from './log.js';
import { consoleLogger } from './log.js';
import { nodePost } from './transport.js';

// =============================================================================
// expandQuery
// =============================================================================

/**
 * Expand a search query into typed variants (lex, vec, hyde) via remote LLM.
 *
 * @param cfg           - Expand endpoint config
 * @param breaker       - Circuit breaker for this endpoint
 * @param query         - Original search query
 * @param readTimeoutMs - HTTP read timeout in ms
 * @param options       - Expansion options (includeLexical, intent)
 * @returns Array of Queryable variants for hybrid search
 */
export async function expandQuery(
  cfg: EndpointConfig,
  breaker: CircuitBreaker,
  query: string,
  readTimeoutMs: number,
  options?: { includeLexical?: boolean; intent?: string },
  log: Logger = consoleLogger,
): Promise<Queryable[]> {
  if (!cfg.apiKey) {
    log.warn(
      'RemoteLLM: expand endpoint has no API key, returning passthrough query',
    );
    return [{ type: 'lex', text: query }];
  }

  if (!breaker.canAttempt()) {
    log.warn(
      'RemoteLLM: expand circuit breaker is open, returning passthrough query',
    );
    return expandFallback(query, options?.includeLexical ?? true);
  }

  const includeLexical = options?.includeLexical ?? true;
  const systemPrompt =
    'You are a search query expansion assistant. ' +
    'Given a search query, produce expanded variants in EXACTLY this format:\n' +
    'lex: <keyword/BM25 variant>\n' +
    'vec: <semantic paraphrase>\n' +
    'hyde: <one-sentence hypothetical document excerpt>\n\n' +
    'Output only those three lines. No explanation, no extra text.';
  const userPrompt = options?.intent
    ? `Expand this search query: ${query}\nQuery intent: ${options.intent}`
    : `Expand this search query: ${query}`;

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (cfg.apiKey) {
      headers['Authorization'] = `Bearer ${cfg.apiKey.trim()}`;
    }

    const data = await nodePost(
      `${cfg.baseUrl}/chat/completions`,
      headers,
      {
        model: cfg.model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        max_tokens: 600,
        temperature: 0.7,
      },
      readTimeoutMs,
    ) as { choices: Array<{ message: { content: string } }> };

    const content = data.choices[0]?.message?.content ?? '';

    // Warn if the LLM returned non-empty content that we couldn't parse
    if (content.trim().length > 0) {
      const queryables = parseExpandResponse(content, query, includeLexical);
      if (queryables.length === 0) {
        log.warn(
          'RemoteLLM: expandQuery received response but could not parse any ' +
          `valid query variants. Raw response (first 200 chars): "${content.slice(0, 200)}"`,
        );
      }
      breaker.onSuccess();
      return queryables.length > 0
        ? queryables
        : expandFallback(query, includeLexical);
    }

    // Empty response — fall back
    log.warn(
      'RemoteLLM: expandQuery returned empty response, using fallback',
    );
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
}

// =============================================================================
// parseExpandResponse
// =============================================================================

/**
 * Parse the chat completion response into Queryable[].
 *
 * Expects lines in `type: content` format where type ∈ {lex, vec, hyde}.
 * Performs validation:
 * - Strips markdown formatting (bold, quotes) that LLMs sometimes wrap output in
 * - Requires minimum length (4+ chars)
 * - Checks that expanded text contains at least one term from the original query
 *
 * @param content        - Raw LLM response text
 * @param originalQuery  - Original search query (for term overlap validation)
 * @param includeLexical - Whether to include 'lex' type results
 * @returns Parsed Queryable array (may be empty if parsing fails)
 */
export function parseExpandResponse(
  content: string,
  originalQuery: string,
  includeLexical: boolean,
): Queryable[] {
  const lines = content.trim().split('\n');
  const queryTerms = originalQuery
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

  const hasQueryTerm = (text: string): boolean => {
    if (queryTerms.length === 0) return true;
    const lower = text.toLowerCase();
    return queryTerms.some((term) => lower.includes(term));
  };

  const queryables: Queryable[] = [];
  for (const line of lines) {
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) continue;
    const type = line.slice(0, colonIdx).trim().toLowerCase();
    if (type !== 'lex' && type !== 'vec' && type !== 'hyde') continue;
    const raw = line.slice(colonIdx + 1).trim();
    const text = raw.replace(/\*\*/g, '').replace(/^"|"$/g, '');
    if (!text || text.length <= 3) continue;
    if (!hasQueryTerm(text)) continue;
    queryables.push({ type: type as 'lex' | 'vec' | 'hyde', text });
  }

  return includeLexical
    ? queryables
    : queryables.filter((q) => q.type !== 'lex');
}

// =============================================================================
// expandFallback
// =============================================================================

/**
 * Generate a sensible fallback when query expansion cannot produce valid results.
 *
 * Creates generic lex/vec/hyde entries from the raw query string.
 * This ensures degraded search still works even when the expand model
 * is unavailable or returns unparseable output.
 *
 * @param query          - Original search query
 * @param includeLexical - Whether to include the 'lex' type entry
 * @returns Fallback Queryable array (never empty)
 */
export function expandFallback(
  query: string,
  includeLexical: boolean,
): Queryable[] {
  const fallback: Queryable[] = [
    { type: 'hyde', text: `Information about ${query}` },
    { type: 'lex', text: query },
    { type: 'vec', text: query },
  ];
  return includeLexical
    ? fallback
    : fallback.filter((q) => q.type !== 'lex');
}
