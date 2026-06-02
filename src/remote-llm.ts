/**
 * remote-llm.ts - OpenAI-compatible remote embedding & reranking backend
 *
 * Implements the LLM interface by calling HTTP endpoints (vLLM, Ollama, OpenAI, etc.).
 * Only supports embed/rerank operations — generate/expandQuery throw.
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
  RerankDocumentResult,
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
  /** Base URL for query-expansion endpoint (defaults to embedApiUrl). Hits POST <url>/chat/completions. */
  expandApiUrl?: string;
  /** Model name for query expansion (any chat-completion model). */
  expandApiModel?: string;
  /** Optional bearer token for expand endpoint. */
  expandApiKey?: string;
  /** Connect timeout in ms (default: 5000) */
  connectTimeoutMs?: number;
  /** Read timeout for embedding in ms (default: 30000) */
  embedReadTimeoutMs?: number;
  /** Read timeout for reranking in ms (default: 60000) */
  rerankReadTimeoutMs?: number;
  /** Read timeout for query expansion in ms (default: 30000) */
  expandReadTimeoutMs?: number;
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

/** Floor for halve-truncating a single oversized document during rerank recovery. */
const RERANK_MIN_DOC_CHARS = 32;

/**
 * True when a rerank request failed because the payload exceeded what the
 * server/model can process (the whole batch, or a single very long document).
 * Drives batch-splitting recovery; non-oversized errors propagate instead so
 * the circuit breaker / HybridLLM local fallback can react.
 */
function isOversizedRerankError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const m = err.message.toLowerCase();
  return m.includes("too large to process")
    || m.includes("payload too large")
    || m.includes(" 413")
    || m.includes("context length")
    || m.includes("maximum context")
    || m.includes("too long");
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

  /** Rerank model — defaults to the embedding model when no separate rerank model is configured. */
  get rerankModelName(): string {
    return this.config.rerankApiModel || this.config.embedApiModel;
  }

  /** Remote backend exposes no local generation model; use the embed model as a placeholder identifier. */
  get generateModelName(): string {
    return this.config.embedApiModel;
  }

  get usesRemoteEmbedding(): boolean {
    return true;
  }

  /** True when expandApiModel is configured and remote query expansion is available. */
  get supportsExpand(): boolean {
    return !!this.config.expandApiModel;
  }

  /** True when rerankApiModel is configured and remote reranking is available. */
  get supportsRerank(): boolean {
    return !!this.config.rerankApiModel;
  }

  /**
   * Remote backends have no local tokenizer. HybridLLM proxies tokenize/detokenize
   * to the local LlamaCpp; bare RemoteLLM use will throw.
   */
  async tokenize(_text: string): Promise<readonly never[]> {
    throw new Error("RemoteLLM.tokenize is unavailable; use HybridLLM to access the local tokenizer.");
  }

  async detokenize(_tokens: readonly never[]): Promise<string> {
    throw new Error("RemoteLLM.detokenize is unavailable; use HybridLLM to access the local tokenizer.");
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

    // One rerank request for a slice of `documents` starting at `start`.
    // Response indices are local to the submitted slice, so they are remapped
    // to original document positions via `start`. relevance_score is
    // sigmoid-normalized (see note below).
    const rerankBatch = async (
      batch: RerankDocument[],
      start: number,
    ): Promise<RerankDocumentResult[]> => {
      const response = await fetchWithTimeout(url, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: rerankModel,
          query,
          documents: batch.map(d => d.text),
        }),
      }, this.config.rerankReadTimeoutMs);

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Rerank API returned ${response.status}: ${errText}`);
      }

      const json = await response.json() as {
        results: { index: number; relevance_score: number }[];
      };

      // Normalize relevance_score via sigmoid σ(x) = 1/(1+e^-x).
      //
      // Many cross-encoder rerankers exposed through llama.cpp's /v1/rerank
      // endpoint (notably bge-reranker-v2-m3, BAAI/bge-reranker-large, jina-
      // reranker-v2) emit log-odds (range roughly -10..+10, negative = poor
      // match, positive = good match). The qmd consumer (store.ts blend
      // formula and the --min-score default of 0.3) assumes a 0..1 probability
      // range. Without normalization, every blended score ends up negative and
      // the default min-score filter drops all results.
      //
      // Sigmoid is monotonic so ordering is preserved. For rerankers that
      // already emit 0..1 scores (some Voyage / Cohere endpoints), sigmoid
      // is a no-op-ish squash that keeps them in [0,1]. Safe in both directions.
      return json.results.map(r => {
        const documentIndex = start + r.index;
        return {
          file: documents[documentIndex]!.file,
          score: 1 / (1 + Math.exp(-r.relevance_score)),
          index: documentIndex,
        };
      });
    };

    // Recovery for servers that reject an over-large rerank payload (the batch
    // or a single very long document exceeds the model's context). On an
    // oversized error we bisect the batch; a single oversized document is
    // halve-truncated down to a floor and re-scored on the truncated text.
    // Adapted from the recovery in tobi/qmd#619 (loopyd). Non-oversized errors
    // propagate so the circuit breaker / HybridLLM local fallback can react.
    const scoreBatch = async (
      batch: RerankDocument[],
      start: number,
    ): Promise<RerankDocumentResult[]> => {
      try {
        return await rerankBatch(batch, start);
      } catch (err) {
        if (!isOversizedRerankError(err) || batch.length === 0) {
          throw err;
        }
        if (batch.length === 1) {
          const doc = batch[0]!;
          const nextLength = Math.max(RERANK_MIN_DOC_CHARS, Math.floor(doc.text.length / 2));
          if (doc.text.length <= RERANK_MIN_DOC_CHARS || nextLength >= doc.text.length) {
            throw err;
          }
          return scoreBatch([{ ...doc, text: doc.text.slice(0, nextLength) }], start);
        }
        const mid = Math.ceil(batch.length / 2);
        const left = await scoreBatch(batch.slice(0, mid), start);
        const right = await scoreBatch(batch.slice(mid), start + mid);
        return [...left, ...right];
      }
    };

    try {
      const results = await scoreBatch(documents, 0);
      // Sub-batch recovery merges results out of global rank order; sort by the
      // (monotonic) normalized score for a coherent final ranking. No-op on the
      // common single-request path, where the server already returns ranked.
      results.sort((a, b) => b.score - a.score);
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

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    const expandUrl = this.config.expandApiUrl || this.config.embedApiUrl;
    const expandModel = this.config.expandApiModel;
    const expandKey = this.config.expandApiKey || this.config.embedApiKey;
    const includeLexical = options?.includeLexical ?? true;
    const intent = options?.intent;

    // Shared fallback shape — used whenever the remote call fails OR returns
    // nothing parseable. Mirrors LocalLLM.expandQuery's fallback exactly so
    // downstream code doesn't see a behavior difference.
    const defaultFallback = (): Queryable[] => {
      const triple: Queryable[] = [
        { type: "hyde", text: `Information about ${query}` },
        { type: "lex", text: query },
        { type: "vec", text: query },
      ];
      return includeLexical ? triple : triple.filter(q => q.type !== "lex");
    };

    if (!expandModel) {
      // Configured to use remote but no expand model set → safe default.
      return defaultFallback();
    }

    // Prompt the chat model to emit the lex/vec/hyde format that
    // LocalLLM.expandQuery produces via grammar-constrained sampling.
    // Without llama.cpp grammar we have to ask politely; parsing below is
    // tolerant of variations and falls back if the model goes off-script.
    const systemPrompt =
      "You expand search queries for a hybrid retrieval system. " +
      "Output 3 to 6 query variants, one per line, each prefixed with its type. " +
      "Types:\n" +
      "  lex  - keyword/BM25-friendly phrasing (extract distinctive terms)\n" +
      "  vec  - semantic embedding-friendly phrasing (paraphrase intent)\n" +
      "  hyde - a hypothetical answer or document passage that would match the query\n" +
      "\n" +
      "Format strictly as:\n" +
      "<type>: <variant>\n" +
      "\n" +
      "No preamble, no explanation, no blank lines. Every line MUST start with " +
      "exactly 'lex: ', 'vec: ', or 'hyde: '. Each variant should contain at " +
      "least one term from the original query.";

    const userPrompt = intent
      ? `Expand this search query: ${query}\nQuery intent: ${intent}`
      : `Expand this search query: ${query}`;

    const url = normalizeUrl(expandUrl, "/chat/completions");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (expandKey) headers["Authorization"] = `Bearer ${expandKey}`;

    const body = JSON.stringify({
      model: expandModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.7,
      top_p: 0.8,
      max_tokens: 600,
    });

    let content = "";
    try {
      const response = await fetchWithTimeout(
        url,
        { method: "POST", headers, body },
        this.config.expandReadTimeoutMs ?? 30000,
      );
      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Expand API returned ${response.status}: ${errText}`);
      }
      const json = (await response.json()) as {
        choices?: { message?: { content?: string } }[];
      };
      content = json.choices?.[0]?.message?.content ?? "";
    } catch (err) {
      // Network error, timeout, or non-2xx — fall back to the default triple.
      // Don't escalate to caller; LocalLLM also masks failures behind the
      // fallback to keep search resilient.
      console.error("Remote query expansion failed:", err);
      return defaultFallback();
    }

    // Parse — mirror LocalLLM's parsing exactly so downstream sees consistent
    // shapes regardless of which backend produced the expansion.
    const lines = content.trim().split("\n");
    const queryLower = query.toLowerCase();
    const queryTerms = queryLower
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

    const hasQueryTerm = (text: string): boolean => {
      const lower = text.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some(term => lower.includes(term));
    };

    const queryables: Queryable[] = lines
      .map((line): Queryable | null => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim().toLowerCase();
        if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!text) return null;
        if (!hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      })
      .filter((q): q is Queryable => q !== null);

    const filtered = includeLexical
      ? queryables
      : queryables.filter(q => q.type !== "lex");

    if (filtered.length > 0) return filtered;
    return defaultFallback();
  }

  async dispose(): Promise<void> {
    // Nothing to dispose for HTTP client
  }
}

// =============================================================================
// Helpers
// =============================================================================

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
