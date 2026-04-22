/**
 * remote-llm.ts - OpenAI-compatible remote embedding, reranking & query expansion backend
 *
 * Implements the LLM interface by calling HTTP endpoints (vLLM, Ollama, OpenAI, etc.).
 * Supports embed, rerank, and (when expandApiModel is set) query expansion via chat completions.
 * generate() is not supported — use HybridLLM to pair with a local backend.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export type RemoteLLMConfig = {
  /** Base URL for embedding endpoint (e.g. http://gpu-host:8000/v1) */
  embedApiUrl: string;
  /** Model name for embedding (e.g. BAAI/bge-m3) */
  embedApiModel: string;
  /** Optional bearer token for embedding endpoint */
  embedApiKey?: string;
  /** Base URL for rerank endpoint (defaults to embedApiUrl) */
  rerankApiUrl?: string;
  /** Model name for reranking */
  rerankApiModel?: string;
  /** Optional bearer token for rerank endpoint */
  rerankApiKey?: string;
  /** Base URL for query expansion endpoint (defaults to embedApiUrl) */
  expandApiUrl?: string;
  /** Model name for query expansion via chat completions (enables remote expansion when set) */
  expandApiModel?: string;
  /** Optional bearer token for expansion endpoint (defaults to embedApiKey) */
  expandApiKey?: string;
  /** Read timeout for query expansion in ms (default: 30000) */
  expandReadTimeoutMs?: number;
  /** Connect timeout in ms (default: 5000) */
  connectTimeoutMs?: number;
  /** Read timeout for embedding in ms (default: 30000) */
  embedReadTimeoutMs?: number;
  /** Read timeout for reranking in ms (default: 60000) */
  rerankReadTimeoutMs?: number;
  /** Max texts per embed HTTP request (default: 32) */
  maxBatchSize?: number;
};

// =============================================================================
// Circuit Breaker
// =============================================================================

type CircuitState = "closed" | "open" | "half-open";

class CircuitBreaker {
  private state: CircuitState = "closed";
  private failures = 0;
  private lastFailureTime = 0;
  private readonly maxFailures: number;
  private readonly cooldownMs: number;

  constructor(maxFailures = 3, cooldownMs = 10 * 60 * 1000) {
    this.maxFailures = maxFailures;
    this.cooldownMs = cooldownMs;
  }

  canAttempt(): boolean {
    if (this.state === "closed") return true;
    if (this.state === "open") {
      if (Date.now() - this.lastFailureTime >= this.cooldownMs) {
        this.state = "half-open";
        return true;
      }
      return false;
    }
    // half-open: allow one attempt
    return true;
  }

  onSuccess(): void {
    this.state = "closed";
    this.failures = 0;
  }

  onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.state === "half-open" || this.failures >= this.maxFailures) {
      this.state = "open";
    }
  }

  getState(): CircuitState {
    return this.state;
  }
}

// =============================================================================
// RemoteLLM
// =============================================================================

export class RemoteLLM implements LLM {
  private readonly config: Required<
    Pick<RemoteLLMConfig, "embedApiUrl" | "embedApiModel" | "connectTimeoutMs" | "embedReadTimeoutMs" | "rerankReadTimeoutMs" | "maxBatchSize">
  > & RemoteLLMConfig;

  private readonly embedBreaker = new CircuitBreaker();
  private readonly rerankBreaker = new CircuitBreaker();
  private readonly expandBreaker = new CircuitBreaker();
  private expectedDimensions: number | null = null;

  constructor(config: RemoteLLMConfig) {
    this.config = {
      connectTimeoutMs: 5000,
      embedReadTimeoutMs: 30000,
      rerankReadTimeoutMs: 60000,
      maxBatchSize: 32,
      ...config,
    };
  }

  get embedModelName(): string {
    return this.config.embedApiModel;
  }

  /** True when expandApiModel is configured and remote query expansion is available. */
  get supportsExpand(): boolean {
    return !!this.config.expandApiModel;
  }

  // ---------------------------------------------------------------------------
  // Embedding
  // ---------------------------------------------------------------------------

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], options);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[], _options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    if (!this.embedBreaker.canAttempt()) {
      throw new Error(
        `Remote embedding circuit breaker is open — endpoint ${this.config.embedApiUrl} is unavailable. ` +
        `Will retry after cooldown.`
      );
    }

    const batchSize = this.config.maxBatchSize;
    const results: (EmbeddingResult | null)[] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchResults = await this.embedBatchRequest(batch);
      results.push(...batchResults);
    }

    return results;
  }

  private async embedBatchRequest(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    const url = normalizeUrl(this.config.embedApiUrl, "/embeddings");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.config.embedApiKey) {
      headers["Authorization"] = `Bearer ${this.config.embedApiKey}`;
    }

    const body = JSON.stringify({
      model: this.config.embedApiModel,
      input: texts,
    });

    try {
      const response = await fetchWithTimeout(url, {
        method: "POST",
        headers,
        body,
      }, this.config.embedReadTimeoutMs);

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Embedding API returned ${response.status}: ${errText}`);
      }

      const json = await response.json() as {
        data: { embedding: number[]; index: number }[];
      };

      // Validate dimensions consistency
      if (json.data.length > 0) {
        const dim = json.data[0]!.embedding.length;
        if (this.expectedDimensions === null) {
          this.expectedDimensions = dim;
        } else if (dim !== this.expectedDimensions) {
          throw new Error(
            `Embedding dimension mismatch: expected ${this.expectedDimensions}, got ${dim}. ` +
            `This usually means the remote model changed.`
          );
        }
      }

      // Sort by index to match input order
      const sorted = [...json.data].sort((a, b) => a.index - b.index);
      const results: (EmbeddingResult | null)[] = sorted.map(item => ({
        embedding: item.embedding,
        model: this.config.embedApiModel,
      }));

      this.embedBreaker.onSuccess();
      return results;
    } catch (err) {
      this.embedBreaker.onFailure();
      throw err;
    }
  }

  // ---------------------------------------------------------------------------
  // Reranking
  // ---------------------------------------------------------------------------

  async rerank(query: string, documents: RerankDocument[], _options?: RerankOptions): Promise<RerankResult> {
    const rerankUrl = this.config.rerankApiUrl || this.config.embedApiUrl;
    const rerankModel = this.config.rerankApiModel;
    const rerankKey = this.config.rerankApiKey || this.config.embedApiKey;

    if (!rerankModel) {
      throw new Error("Remote reranking requires rerankApiModel to be configured");
    }

    if (!this.rerankBreaker.canAttempt()) {
      throw new Error(
        `Remote rerank circuit breaker is open — endpoint ${rerankUrl} is unavailable. ` +
        `Will retry after cooldown.`
      );
    }

    const url = normalizeUrl(rerankUrl, "/rerank");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (rerankKey) {
      headers["Authorization"] = `Bearer ${rerankKey}`;
    }

    const body = JSON.stringify({
      model: rerankModel,
      query,
      documents: documents.map(d => d.text),
    });

    try {
      const response = await fetchWithTimeout(url, {
        method: "POST",
        headers,
        body,
      }, this.config.rerankReadTimeoutMs);

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Rerank API returned ${response.status}: ${errText}`);
      }

      const json = await response.json() as {
        results: { index: number; relevance_score: number }[];
      };

      const results = json.results.map(r => ({
        file: documents[r.index]!.file,
        score: r.relevance_score,
        index: r.index,
      }));

      this.rerankBreaker.onSuccess();
      return { results, model: rerankModel };
    } catch (err) {
      this.rerankBreaker.onFailure();
      throw err;
    }
  }

  // ---------------------------------------------------------------------------
  // Unsupported operations (these require local models)
  // ---------------------------------------------------------------------------

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    throw new Error("RemoteLLM does not support text generation — use HybridLLM to route generation to a local backend");
  }

  async modelExists(_model: string): Promise<ModelInfo> {
    return { name: this.config.embedApiModel, exists: true };
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    if (!this.config.expandApiModel) {
      throw new Error("RemoteLLM: expandApiModel not configured — set expandApiUrl and expandApiModel to enable remote query expansion");
    }

    if (!this.expandBreaker.canAttempt()) {
      throw new Error(
        `Remote query expansion circuit breaker is open — endpoint unavailable. Will retry after cooldown.`
      );
    }

    const expandUrl = this.config.expandApiUrl || this.config.embedApiUrl;
    const expandKey = this.config.expandApiKey || this.config.embedApiKey;
    const includeLexical = options?.includeLexical ?? true;
    const intent = options?.intent;

    const systemPrompt =
      "You are a search query expansion assistant. " +
      "Given a search query, produce expanded variants in EXACTLY this format:\n" +
      "lex: <keyword/BM25 variant>\n" +
      "vec: <semantic paraphrase>\n" +
      "hyde: <one-sentence hypothetical document excerpt>\n\n" +
      "Output only those three lines. No explanation, no extra text.";

    const userPrompt = intent
      ? `Expand this search query: ${query}\nQuery intent: ${intent}`
      : `Expand this search query: ${query}`;

    const url = normalizeUrl(expandUrl, "/chat/completions");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (expandKey) headers["Authorization"] = `Bearer ${expandKey}`;

    const body = JSON.stringify({
      model: this.config.expandApiModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.7,
      max_tokens: 200,
    });

    try {
      const response = await fetchWithTimeout(url, {
        method: "POST",
        headers,
        body,
      }, this.config.expandReadTimeoutMs ?? 30000);

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Query expansion API returned ${response.status}: ${errText}`);
      }

      const json = await response.json() as {
        choices: { message: { content: string } }[];
      };
      const content = json.choices[0]?.message?.content ?? "";

      this.expandBreaker.onSuccess();
      return parseExpandResponse(content, query, includeLexical);
    } catch (err) {
      this.expandBreaker.onFailure();
      throw err;
    }
  }

  async dispose(): Promise<void> {
    // Nothing to dispose for HTTP client
  }
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Parse the chat completion response from the expand endpoint into Queryable[].
 * Expects lines in "type: content" format where type ∈ {lex, vec, hyde}.
 */
function parseExpandResponse(content: string, originalQuery: string, includeLexical: boolean): Queryable[] {
  const lines = content.trim().split("\n");
  const queryTerms = originalQuery.toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const hasQueryTerm = (text: string): boolean => {
    if (queryTerms.length === 0) return true;
    const lower = text.toLowerCase();
    return queryTerms.some(term => lower.includes(term));
  };

  const queryables: Queryable[] = lines.flatMap(line => {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) return [];
    const type = line.slice(0, colonIdx).trim() as QueryType;
    if (type !== "lex" && type !== "vec" && type !== "hyde") return [];
    const text = line.slice(colonIdx + 1).trim();
    if (!text || !hasQueryTerm(text)) return [];
    return [{ type, text }];
  });

  const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== "lex");
  if (filtered.length > 0) return filtered;

  // Fallback when model output couldn't be parsed
  const fallback: Queryable[] = [
    { type: "hyde", text: `Information about ${originalQuery}` },
    { type: "lex", text: originalQuery },
    { type: "vec", text: originalQuery },
  ];
  return includeLexical ? fallback : fallback.filter(q => q.type !== "lex");
}

/**
 * Normalize a base URL and append a path, handling trailing slashes.
 */
function normalizeUrl(baseUrl: string, path: string): string {
  const base = baseUrl.replace(/\/+$/, "");
  return `${base}${path}`;
}

/**
 * Fetch with a timeout using AbortSignal.timeout().
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  return fetch(url, {
    ...init,
    signal: AbortSignal.timeout(timeoutMs),
  });
}

// =============================================================================
// Configuration from environment
// =============================================================================

/**
 * Create a RemoteLLMConfig from environment variables and optional YAML config.
 * Returns null if remote embedding is not configured.
 */
export function remoteConfigFromEnv(yamlModels?: {
  embed_api_url?: string;
  embed_api_model?: string;
  embed_api_key?: string;
  rerank_api_url?: string;
  rerank_api_model?: string;
  rerank_api_key?: string;
  expand_api_url?: string;
  expand_api_model?: string;
  expand_api_key?: string;
}): RemoteLLMConfig | null {
  const embedApiUrl = process.env.QMD_EMBED_API_URL || yamlModels?.embed_api_url;
  const embedApiModel = process.env.QMD_EMBED_API_MODEL || yamlModels?.embed_api_model;

  if (!embedApiUrl || !embedApiModel) return null;

  return {
    embedApiUrl,
    embedApiModel,
    embedApiKey: process.env.QMD_EMBED_API_KEY || yamlModels?.embed_api_key,
    rerankApiUrl: process.env.QMD_RERANK_API_URL || yamlModels?.rerank_api_url,
    rerankApiModel: process.env.QMD_RERANK_API_MODEL || yamlModels?.rerank_api_model,
    rerankApiKey: process.env.QMD_RERANK_API_KEY || yamlModels?.rerank_api_key,
    expandApiUrl: process.env.QMD_EXPAND_API_URL || yamlModels?.expand_api_url,
    expandApiModel: process.env.QMD_EXPAND_API_MODEL || yamlModels?.expand_api_model,
    expandApiKey: process.env.QMD_EXPAND_API_KEY || yamlModels?.expand_api_key,
    connectTimeoutMs: parseEnvInt("QMD_REMOTE_CONNECT_TIMEOUT", 5000),
    embedReadTimeoutMs: parseEnvInt("QMD_REMOTE_READ_TIMEOUT", 30000),
    rerankReadTimeoutMs: parseEnvInt("QMD_REMOTE_RERANK_TIMEOUT", 60000),
    expandReadTimeoutMs: parseEnvInt("QMD_REMOTE_EXPAND_TIMEOUT", 30000),
    maxBatchSize: parseEnvInt("QMD_REMOTE_BATCH_SIZE", 32),
  };
}

function parseEnvInt(name: string, defaultValue: number): number {
  const val = process.env[name];
  if (!val) return defaultValue;
  const parsed = parseInt(val, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : defaultValue;
}
