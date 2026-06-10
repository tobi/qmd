/**
 * embedding-provider.ts - Remote embedding provider factory
 *
 * Creates an LLM implementation that talks to OpenAI-compatible /v1/embeddings
 * endpoint. Works with vLLM, Ollama, OpenAI, text-embeddings-inference, etc.
 *
 * This is a drop-in replacement for LlamaCpp that implements the same LLM interface.
 * Import and use in place of `new LlamaCpp()`:
 *
 *   const llm = await createRemoteLLM({ config });
 *
 * Phase 5: Per-endpoint config + real expand/rerank/generate via OpenRouter.
 */

import * as http from 'http';
import * as https from 'https';
import { URL } from 'url';
import type {
  LLM,
  EmbeddingResult,
  EmbedOptions,
  GenerateResult,
  GenerateOptions,
  ModelInfo,
  Queryable,
  RerankResult,
  RerankDocument,
  RerankOptions,
} from './llm.js';
import type { ModelsConfig } from './collections.js';

// =============================================================================
// Types
// =============================================================================

export type EndpointConfig = {
  baseUrl: string;
  model: string;
  apiKey?: string;
};

export type RemoteLLMConfig = {
  embed?: EndpointConfig;
  expand?: EndpointConfig;
  rerank?: EndpointConfig;
  generate?: EndpointConfig;

  /** Backward compat: old flat config fields treated as embed endpoint */
  baseUrl?: string;
  embedModel?: string;
  apiKey?: string;

  /** Max texts per batch request (default: 32) */
  maxBatchSize?: number;
  /** Connect timeout in ms (default: 5000) */
  connectTimeoutMs?: number;
  /** Read timeout for embedding requests in ms (default: 30000) */
  embedReadTimeoutMs?: number;
  /** Read timeout for rerank requests in ms (default: 60000) */
  rerankReadTimeoutMs?: number;
  /** Read timeout for expand/generate requests in ms (default: 30000) */
  expandReadTimeoutMs?: number;
};

// =============================================================================
// Helpers
// =============================================================================

const OPENROUTER_DEFAULT_URL = 'https://openrouter.ai/api/v1';

/**
 * Low-level HTTP/HTTPS POST using Node's built-in http/https modules.
 *
 * Bypasses the global fetch() / undici entirely, which avoids the Node v24
 * undici ByteString bug (https://github.com/nodejs/node/issues/xxxx) that
 * crashes when response bodies contain Unicode code points > U+00FF (e.g.
 * U+2026 HORIZONTAL ELLIPSIS returned by OpenRouter).
 *
 * Returns the parsed JSON response body, or throws on non-2xx status.
 */
async function nodePost(urlStr: string, headers: Record<string, string>, body: unknown, timeoutMs = 30000): Promise<any> {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const bodyBuf = Buffer.from(JSON.stringify(body), 'utf-8');
    const requestHeaders: Record<string, string> = {
      ...headers,
      'Content-Type': 'application/json',
      'Content-Length': String(bodyBuf.length),
    };

    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request(
      {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        headers: requestHeaders,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          const status = res.statusCode ?? 0;
          if (status < 200 || status >= 300) {
            reject(new Error(`HTTP ${status}: ${raw.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (e) {
            reject(new Error(`JSON parse failed: ${(e as Error).message} — body: ${raw.slice(0, 200)}`));
          }
        });
        res.on('error', reject);
      },
    );

    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.write(bodyBuf);
    req.end();
  });
}

/**
 * Low-level HTTP/HTTPS GET using Node's built-in http/https modules.
 * Used for /models endpoint health checks.
 */
async function nodeGet(urlStr: string, headers: Record<string, string>, timeoutMs = 5000): Promise<any> {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request(
      {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'GET',
        headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          const status = res.statusCode ?? 0;
          if (status < 200 || status >= 300) {
            reject(new Error(`HTTP ${status}: ${raw.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (e) {
            reject(new Error(`JSON parse failed: ${(e as Error).message}`));
          }
        });
        res.on('error', reject);
      },
    );

    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.end();
  });
}


/**
 * Resolve an EndpointConfig from env vars with fallback chain.
 */
function resolveEndpoint(
  _name: string,
  envSuffix: string,
  modelDefault: string,
  urlDefault: string,
): EndpointConfig {
  const baseUrl =
    (process.env[`QMD_${envSuffix}_BASE_URL`] || urlDefault).replace(/\/+$/, '');
  const model = process.env[`QMD_${envSuffix}_MODEL`] || modelDefault;
  const apiKey = (process.env[`QMD_${envSuffix}_API_KEY`] || process.env.OPENAI_API_KEY || '').trim();
  return { baseUrl, model, apiKey };
}

/**
 * Resolve a RemoteLLMConfig from environment variables and optional YAML models config.
 *
 * Priority: environment variables > YAML models config > local-first defaults.
 *
 * This allows users to configure remote expand/generate/rerank endpoints
 * in their index.yml under the `models:` key, falling back to env vars.
 * Local-first defaults: all endpoints default to empty (local LlamaCpp)
 * unless explicitly configured via env vars or YAML.
 * 
 * 
 */
export function remoteConfigFromEnv(models?: ModelsConfig): RemoteLLMConfig {
  // --- Embed ---
  // Priority: QMD_EMBED_* env vars > models.embed_api_* YAML > OPENAI_* fallback > empty (local-first)
  // Empty baseUrl means: fall back to local LlamaCpp embeddings.
  const embed: EndpointConfig = {
    baseUrl: (process.env.QMD_EMBED_BASE_URL || models?.embed_api_url || process.env.OPENAI_BASE_URL || '').replace(/\/+$/, ''),
    model: process.env.QMD_EMBED_MODEL || models?.embed_api_model || '',
    apiKey: (process.env.QMD_EMBED_API_KEY || models?.embed_api_key || process.env.OPENAI_API_KEY || '').trim(),
  };

  // --- Expand ---
  // Priority: QMD_EXPAND_* env vars > models.expand_api_* YAML > empty (local-first)
  const expand: EndpointConfig = {
    baseUrl: (process.env.QMD_EXPAND_BASE_URL || models?.expand_api_url || '').replace(/\/+$/, ''),
    model: process.env.QMD_EXPAND_MODEL || models?.expand_api_model || '',
    apiKey: (process.env.QMD_EXPAND_API_KEY || models?.expand_api_key || process.env.OPENAI_API_KEY || '').trim(),
  };

  // --- Rerank ---
  // Priority: QMD_RERANK_* env vars > models.rerank_api_* YAML > empty (local-first)
  const rerank: EndpointConfig = {
    baseUrl: (process.env.QMD_RERANK_BASE_URL || models?.rerank_api_url || '').replace(/\/+$/, ''),
    model: process.env.QMD_RERANK_MODEL || models?.rerank_api_model || '',
    apiKey: (process.env.QMD_RERANK_API_KEY || models?.rerank_api_key || process.env.OPENAI_API_KEY || '').trim(),
  };

  // --- Generate ---
  // Priority: QMD_GENERATE_* env vars > models.generate_api_* YAML > empty (local-first)
  const generate: EndpointConfig = {
    baseUrl: (process.env.QMD_GENERATE_BASE_URL || models?.generate_api_url || '').replace(/\/+$/, ''),
    model: process.env.QMD_GENERATE_MODEL || models?.generate_api_model || '',
    apiKey: (process.env.QMD_GENERATE_API_KEY || models?.generate_api_key || process.env.OPENAI_API_KEY || '').trim(),
  };

  return { embed, expand, rerank, generate };
}

// =============================================================================
// Circuit Breaker
// =============================================================================

type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Simple circuit breaker that tracks consecutive failures.
 *
 * - Closed: normal operation, requests proceed.
 * - Open: too many consecutive failures; requests are rejected immediately.
 * - Half-open: cooldown has elapsed; one request is allowed to probe.
 *
 * Auto-recovers after `cooldownMs` in the open state.
 */
class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures = 0;
  private lastFailureTime = 0;
  private readonly maxFailures: number;
  private readonly cooldownMs: number;

  constructor(maxFailures = 3, cooldownMs = 10 * 60 * 1000) {
    this.maxFailures = maxFailures;
    this.cooldownMs = cooldownMs;
  }

  canAttempt(): boolean {
    if (this.state === 'closed') return true;
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime >= this.cooldownMs) {
        this.state = 'half-open';
        return true;
      }
      return false;
    }
    // half-open: allow one attempt
    return true;
  }

  onSuccess(): void {
    this.state = 'closed';
    this.failures = 0;
  }

  onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.state === 'half-open' || this.failures >= this.maxFailures) {
      this.state = 'open';
    }
  }

  getState(): CircuitState {
    return this.state;
  }
}

// =============================================================================
// RemoteLLM — implements the LLM interface via HTTP
// =============================================================================

export class RemoteLLM implements LLM {
  /** Embed endpoint config */
  readonly embedCfg: EndpointConfig;
  /** Query expansion endpoint config */
  readonly expandCfg: EndpointConfig;
  /** Rerank endpoint config */
  readonly rerankCfg: EndpointConfig;
  /** Text generation endpoint config */
  readonly generateCfg: EndpointConfig;

  readonly maxBatchSize: number;
  readonly connectTimeoutMs: number;
  readonly embedReadTimeoutMs: number;
  readonly rerankReadTimeoutMs: number;
  readonly expandReadTimeoutMs: number;

  /** Circuit breakers for each endpoint */
  private readonly embedBreaker = new CircuitBreaker();
  private readonly rerankBreaker = new CircuitBreaker();
  private readonly expandBreaker = new CircuitBreaker();

  /**
   * Expected embedding dimensions, set on first successful embed response.
   * If a subsequent response returns different dimensions, an error is thrown
   * to prevent silent vector corruption.
   */
  private expectedDimensions: number | null = null;

  constructor(config: RemoteLLMConfig = {}) {
    // --- Backward compat: treat old flat config as embed endpoint ---
    const oldBaseUrl = config.baseUrl || process.env.OPENAI_BASE_URL;
    const oldModel = config.embedModel;
    const oldKey = config.apiKey;

    // --- Embed endpoint ---
    if (config.embed) {
      this.embedCfg = {
        baseUrl: config.embed.baseUrl.replace(/\/+$/, ''),
        model: config.embed.model,
        apiKey: (config.embed.apiKey || process.env.OPENAI_API_KEY || "").trim(),
      };
    } else if (oldBaseUrl) {
      this.embedCfg = {
        baseUrl: oldBaseUrl.replace(/\/+$/, ''),
        model: oldModel || process.env.QMD_EMBED_MODEL || 'Qwen/Qwen3-Embedding-0.6B',
        apiKey: (oldKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else {
      this.embedCfg = resolveEndpoint('embed', 'EMBED', 'Qwen/Qwen3-Embedding-0.6B', 'http://localhost:11434/v1');
    }

    // --- Expand endpoint ---
    if (config.expand) {
      this.expandCfg = {
        baseUrl: config.expand.baseUrl.replace(/\/+$/, ''),
        model: config.expand.model,
        apiKey: (config.expand.apiKey || process.env.OPENAI_API_KEY || "").trim(),
      };
    } else {
      this.expandCfg = resolveEndpoint('expand', 'EXPAND', 'google/gemini-2.0-flash-lite-001', OPENROUTER_DEFAULT_URL);
    }

    // --- Rerank endpoint ---
    if (config.rerank) {
      this.rerankCfg = {
        baseUrl: config.rerank.baseUrl.replace(/\/+$/, ''),
        model: config.rerank.model,
        apiKey: (config.rerank.apiKey || process.env.OPENAI_API_KEY || "").trim(),
      };
    } else {
      this.rerankCfg = resolveEndpoint('rerank', 'RERANK', 'cohere/rerank-v3.5', OPENROUTER_DEFAULT_URL);
    }

    // --- Generate endpoint ---
    if (config.generate) {
      this.generateCfg = {
        baseUrl: config.generate.baseUrl.replace(/\/+$/, ''),
        model: config.generate.model,
        apiKey: (config.generate.apiKey || process.env.OPENAI_API_KEY || "").trim(),
      };
    } else {
      this.generateCfg = resolveEndpoint('generate', 'GENERATE', 'google/gemini-2.0-flash-lite-001', OPENROUTER_DEFAULT_URL);
    }

    this.maxBatchSize = config.maxBatchSize ?? parseInt(process.env.QMD_EMBED_BATCH_SIZE || '32', 10);
    this.connectTimeoutMs = config.connectTimeoutMs ?? parseInt(process.env.QMD_REMOTE_CONNECT_TIMEOUT || '5000', 10);
    this.embedReadTimeoutMs = config.embedReadTimeoutMs ?? parseInt(process.env.QMD_REMOTE_READ_TIMEOUT || '30000', 10);
    this.rerankReadTimeoutMs = config.rerankReadTimeoutMs ?? parseInt(process.env.QMD_REMOTE_RERANK_TIMEOUT || '60000', 10);
    this.expandReadTimeoutMs = config.expandReadTimeoutMs ?? parseInt(process.env.QMD_REMOTE_EXPAND_TIMEOUT || '30000', 10);
  }

  // =========================================================================
  // Model name getters
  // =========================================================================

  get embedModelName(): string {
    return this.embedCfg.model;
  }

  get generateModelName(): string | undefined {
    return this.expandCfg.model;
  }

  get rerankModelName(): string | undefined {
    return this.rerankCfg.model;
  }

  // =========================================================================
  // LLM interface: embed
  // =========================================================================

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], options);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[], _options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    if (!this.embedBreaker.canAttempt()) {
      throw new Error(
        `Remote embedding circuit breaker is open — endpoint ${this.embedCfg.baseUrl} is unavailable. ` +
        `Will retry after cooldown.`
      );
    }

    const results: (EmbeddingResult | null)[] = [];

    for (let i = 0; i < texts.length; i += this.maxBatchSize) {
      const batch = texts.slice(i, i + this.maxBatchSize);
      const batchResults = await this.embedBatchRequest(batch);
      results.push(...batchResults);
    }

    return results;
  }

  private async embedBatchRequest(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    const headers: Record<string, string> = {};
    if (this.embedCfg.apiKey) {
      headers['Authorization'] = `Bearer ${this.embedCfg.apiKey.trim()}`;
    }

    try {
      const data = await nodePost(
        `${this.embedCfg.baseUrl}/embeddings`,
        headers,
        { model: this.embedCfg.model, input: texts },
        this.embedReadTimeoutMs,
      ) as { data: Array<{ embedding: number[]; index?: number }> };

      // Dimension validation: track expected dimensions on first response
      if (data.data.length > 0) {
        const dim = data.data[0]!.embedding.length;
        if (this.expectedDimensions === null) {
          this.expectedDimensions = dim;
        } else if (dim !== this.expectedDimensions) {
          throw new Error(
            `Embedding dimension mismatch: expected ${this.expectedDimensions}, got ${dim}. ` +
            `This usually means the remote model changed.`
          );
        }
      }

      // Sort by index to preserve input order
      const sorted = [...data.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      const results: (EmbeddingResult | null)[] = sorted.map(item => ({
        embedding: item.embedding,
        model: this.embedCfg.model,
      }));

      this.embedBreaker.onSuccess();
      return results;
    } catch (err) {
      this.embedBreaker.onFailure();
      throw err;
    }
  }

  // =========================================================================
  // LLM interface: generate
  // =========================================================================

  async generate(prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    const ep = this.generateCfg;
    const headers: Record<string, string> = {};
    if (ep.apiKey) {
      headers['Authorization'] = `Bearer ${ep.apiKey.trim()}`;
    }

    try {
      const data = await nodePost(
        `${ep.baseUrl}/chat/completions`,
        headers,
        {
          model: ep.model,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: _options?.maxTokens ?? 1024,
          temperature: _options?.temperature ?? 0.7,
        },
        this.expandReadTimeoutMs,
      ) as { choices: Array<{ message: { content: string } }>; model: string };

      return {
        text: data.choices[0]?.message?.content ?? '',
        model: data.model || ep.model,
        done: true,
      };
    } catch (err) {
      console.error('RemoteLLM: generate failed:', err instanceof Error ? err.message : String(err));
      return null;
    }
  }

  // =========================================================================
  // LLM interface: modelExists
  // =========================================================================

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const headers: Record<string, string> = {};
      if (this.embedCfg.apiKey) {
        headers['Authorization'] = `Bearer ${this.embedCfg.apiKey}`;
      }

      // Try the embed base URL's models endpoint (works for vLLM, OpenAI, etc.)
      const baseForModels = this.embedCfg.baseUrl.replace(/\/v1$/, '');
      const data = await nodeGet(`${baseForModels}/models`, headers, 5000) as { data?: Array<{ id: string }> };
      if (data.data) {
        const exists = data.data.some((m: { id: string }) => m.id === model);
        return { name: model, exists };
      }
    } catch {
      // Fall through — return exists: true as before
    }

    return { name: model, exists: true };
  }

  // =========================================================================
  // LLM interface: expandQuery
  // =========================================================================

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    const ep = this.expandCfg;
    if (!ep.apiKey) {
      console.warn('RemoteLLM: expand endpoint has no API key, returning passthrough query');
      return [{ type: 'lex', text: query }];
    }

    if (!this.expandBreaker.canAttempt()) {
      console.warn('RemoteLLM: expand circuit breaker is open, returning passthrough query');
      return this.expandFallback(query, options?.includeLexical ?? true);
    }

    const includeLexical = options?.includeLexical ?? true;
    const intent = options?.intent ? ` with intent: ${options.intent}` : '';
    const systemPrompt =
      'You are a search query expansion assistant. ' +
      'Given a search query, produce expanded variants in EXACTLY this format:\n' +
      'lex: <keyword/BM25 variant>\n' +
      'vec: <semantic paraphrase>\n' +
      'hyde: <one-sentence hypothetical document excerpt>\n\n' +
      'Output only those three lines. No explanation, no extra text.';
    const userPrompt = intent
      ? `Expand this search query: ${query}\nQuery intent: ${options?.intent}`
      : `Expand this search query: ${query}`;

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (ep.apiKey) {
        headers['Authorization'] = `Bearer ${ep.apiKey.trim()}`;
      }

      const data = await nodePost(
        `${ep.baseUrl}/chat/completions`,
        headers,
        {
          model: ep.model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          max_tokens: 600,
          temperature: 0.7,
        },
        this.expandReadTimeoutMs,
      ) as { choices: Array<{ message: { content: string } }> };

      const content = data.choices[0]?.message?.content ?? '';

      const queryables = this.parseExpandResponse(content, query, includeLexical);

      this.expandBreaker.onSuccess();
      return queryables;
    } catch (err) {
      this.expandBreaker.onFailure();
      console.error('RemoteLLM: expandQuery failed:', err instanceof Error ? err.message : String(err));
      return this.expandFallback(query, includeLexical);
    }
  }

  /**
   * Parse the chat completion response from the expand endpoint into Queryable[].
   * Expects lines in "type: content" format where type ∈ {lex, vec, hyde}.
   * Validates that expanded text contains terms from the original query.
   */
  private parseExpandResponse(content: string, originalQuery: string, includeLexical: boolean): Queryable[] {
    const lines = content.trim().split('\n');
    const queryTerms = originalQuery.toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);

    const hasQueryTerm = (text: string): boolean => {
      if (queryTerms.length === 0) return true;
      const lower = text.toLowerCase();
      return queryTerms.some(term => lower.includes(term));
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

    const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
    if (filtered.length > 0) return filtered;

    // Parsing failed completely — return a sensible fallback
    return this.expandFallback(originalQuery, includeLexical);
  }

  /**
   * Generate a sensible fallback when query expansion fails or cannot be parsed.
   */
  private expandFallback(query: string, includeLexical: boolean): Queryable[] {
    const fallback: Queryable[] = [
      { type: 'hyde', text: `Information about ${query}` },
      { type: 'lex', text: query },
      { type: 'vec', text: query },
    ];
    return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
  }

  // =========================================================================
  // LLM interface: rerank
  // =========================================================================

  async rerank(query: string, documents: RerankDocument[], _options?: RerankOptions): Promise<RerankResult> {
    const ep = this.rerankCfg;
    if (!ep.apiKey) {
      console.warn('RemoteLLM: rerank endpoint has no API key, returning uniform scores');
      return {
        results: documents.map((doc, i) => ({ file: doc.file, score: 1.0, index: i })),
        model: ep.model,
      };
    }

    if (!this.rerankBreaker.canAttempt()) {
      console.warn('RemoteLLM: rerank circuit breaker is open, returning uniform scores');
      return {
        results: documents.map((doc, i) => ({ file: doc.file, score: 1.0, index: i })),
        model: ep.model,
      };
    }

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (ep.apiKey) {
        headers['Authorization'] = `Bearer ${ep.apiKey.trim()}`;
      }

      const data = await nodePost(
        `${ep.baseUrl}/rerank`,
        headers,
        {
          model: ep.model,
          query,
          documents: documents.map(d => d.text),
          top_n: documents.length,
        },
        this.rerankReadTimeoutMs,
      ) as { results: Array<{ index: number; relevance_score: number }>; model?: string };

      this.rerankBreaker.onSuccess();
      return {
        results: data.results.map(r => ({
          index: r.index,
          score: r.relevance_score,
          file: documents[r.index]?.file ?? `doc-${r.index}`,
        })),
        model: data.model || ep.model,
      };
    } catch (err) {
      this.rerankBreaker.onFailure();
      console.error('RemoteLLM: rerank failed:', err instanceof Error ? err.message : String(err));
      // Fall back to uniform scores
      return {
        results: documents.map((doc, i) => ({ file: doc.file, score: 1.0, index: i })),
        model: ep.model,
      };
    }
  }

  // =========================================================================
  // LLM interface: dispose
  // =========================================================================

  async dispose(): Promise<void> {
    // Nothing to clean up for HTTP client
  }

  // =========================================================================
  // Utility: health check / probe
  // =========================================================================

  /**
   * Verify the remote server is reachable and return embedding dimensions.
   * Useful at startup to catch misconfiguration early.
   */
  async probe(): Promise<{ ok: boolean; dimensions: number; error?: string }> {
    try {
      const result = await this.embed('dimension-probe');
      if (result) {
        return { ok: true, dimensions: result.embedding.length };
      }
      return { ok: false, dimensions: 0, error: 'Null result from probe' };
    } catch (err) {
      return { ok: false, dimensions: 0, error: String(err) };
    }
  }
}
