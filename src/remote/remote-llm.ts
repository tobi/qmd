/**
 * remote-llm.ts — RemoteLLM class implementing QMD's LLM interface over HTTP.
 *
 * Thin composition layer that wires together the standalone modules in
 * src/remote/. Each LLM method delegates to the corresponding module,
 * passing through the endpoint config and circuit breaker.
 *
 * ## Endpoints
 *
 * | Method        | Module    | HTTP Call                    | Used By                       |
 * |---------------|-----------|------------------------------|-------------------------------|
 * | embed()       | embed.ts  | POST /v1/embeddings          | qmd embed, searchVec          |
 * | embedBatch()  | embed.ts  | POST /v1/embeddings (batch)  | generateEmbeddings (store.ts) |
 * | generate()    | generate  | POST /v1/chat/completions    | Text generation (experimental)|
 * | expandQuery() | expand.ts | POST /v1/chat/completions    | Query expansion (hybridQuery) |
 * | rerank()      | rerank.ts | POST /v1/rerank              | Result reranking              |
 * | modelExists() | probe.ts  | GET /models                  | Health check                  |
 * | probe()       | probe.ts  | POST /v1/embeddings (test)   | Startup dimension detection   |
 *
 * ## Constructor Config Resolution
 *
 * 1. Explicit per-endpoint config objects (embed, expand, rerank, generate)
 * 2. Old flat config fields (baseUrl, embedModel, apiKey) → mapped to embed
 * 3. Environment variables (QMD_EMBED_BASE_URL, QMD_EMBED_MODEL, etc.)
 * 4. Hardcoded defaults (localhost:11434 for embed, OpenRouter for expand/rerank)
 *
 * @module remote/remote-llm
 */

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
} from '../llm.js';
import type { EndpointConfig, RemoteLLMConfig } from './types.js';
import type { Logger } from './log.js';
import { consoleLogger } from './log.js';
import { CircuitBreaker } from './circuit-breaker.js';
import {
  resolveEndpoint,
  resolveEndpointFormat,
  remoteConfigFromEnv,
  OPENROUTER_DEFAULT_URL,
} from './config.js';
import type { RemoteAdapterBundle } from './adapters/types.js';
import { resolveAdapterBundle } from './adapters/registry.js';
import { modelExists, probe as runProbe } from './probe.js';

// Re-export types and config for backward compat
export { type EndpointConfig, type RemoteLLMConfig } from './types.js';
export { remoteConfigFromEnv } from './config.js';

export class RemoteLLM implements LLM {
  // ── Endpoint configs ────────────────────────────────────────────────

  readonly embedCfg: EndpointConfig;
  readonly expandCfg: EndpointConfig;
  readonly rerankCfg: EndpointConfig;
  readonly generateCfg: EndpointConfig;

  // ── Operational settings ────────────────────────────────────────────

  readonly maxBatchSize: number;
  readonly embedMaxRetries: number;
  readonly embedReadTimeoutMs: number;
  readonly rerankReadTimeoutMs: number;
  readonly expandReadTimeoutMs: number;

  // ── Logger ──────────────────────────────────────────────────────────

  /** Pluggable logger — swap for silent or custom in tests. */
  readonly log: Logger;

  // ── Internal state ──────────────────────────────────────────────────

  private readonly embedBreaker = new CircuitBreaker();
  private readonly rerankBreaker = new CircuitBreaker();
  private readonly expandBreaker = new CircuitBreaker();
  private readonly generateBreaker = new CircuitBreaker();
  private readonly adapters: RemoteAdapterBundle;

  /**
   * Mutable dimension tracker shared with embed.ts.
   * Set to null before first embed. After first successful response,
   * locked to that dimension. Subsequent responses with different
   * dimensions throw to prevent silent vector corruption.
   */
  private readonly dimState = { dimensions: null as number | null };

  // ── Constructor ─────────────────────────────────────────────────────

  constructor(config: RemoteLLMConfig = {}, logger?: Logger) {
    this.log = logger ?? consoleLogger;
    const oldBaseUrl = config.baseUrl || process.env.OPENAI_BASE_URL;
    const oldModel = config.embedModel;
    const oldKey = config.apiKey;

    // Embed endpoint
    if (config.embed) {
      this.embedCfg = {
        baseUrl: config.embed.baseUrl.replace(/\/+$/, ''),
        format: config.embed.format ?? 'auto',
        model: config.embed.model,
        apiKey: (config.embed.apiKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else if (oldBaseUrl) {
      this.embedCfg = {
        baseUrl: oldBaseUrl.replace(/\/+$/, ''),
        format: 'auto',
        model: oldModel || process.env.QMD_EMBED_MODEL || 'Qwen/Qwen3-Embedding-0.6B',
        apiKey: (oldKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else {
      this.embedCfg = {
        ...resolveEndpoint('embed', 'EMBED', 'Qwen/Qwen3-Embedding-0.6B', 'http://localhost:11434/v1'),
        format: resolveEndpointFormat('embed', 'EMBED'),
      };
    }

    // Expand endpoint
    if (config.expand) {
      this.expandCfg = {
        baseUrl: config.expand.baseUrl.replace(/\/+$/, ''),
        format: config.expand.format ?? 'auto',
        model: config.expand.model,
        apiKey: (config.expand.apiKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else {
      this.expandCfg = {
        ...resolveEndpoint('expand', 'EXPAND', 'google/gemini-2.0-flash-lite-001', OPENROUTER_DEFAULT_URL),
        format: resolveEndpointFormat('expand', 'EXPAND'),
      };
    }

    // Rerank endpoint
    if (config.rerank) {
      this.rerankCfg = {
        baseUrl: config.rerank.baseUrl.replace(/\/+$/, ''),
        format: config.rerank.format ?? 'auto',
        model: config.rerank.model,
        apiKey: (config.rerank.apiKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else {
      this.rerankCfg = {
        ...resolveEndpoint('rerank', 'RERANK', 'cohere/rerank-v3.5', OPENROUTER_DEFAULT_URL),
        format: resolveEndpointFormat('rerank', 'RERANK'),
      };
    }

    // Generate endpoint
    if (config.generate) {
      this.generateCfg = {
        baseUrl: config.generate.baseUrl.replace(/\/+$/, ''),
        format: config.generate.format ?? 'auto',
        model: config.generate.model,
        apiKey: (config.generate.apiKey || process.env.OPENAI_API_KEY || '').trim(),
      };
    } else {
      this.generateCfg = {
        ...resolveEndpoint('generate', 'GENERATE', 'google/gemini-2.0-flash-lite-001', OPENROUTER_DEFAULT_URL),
        format: resolveEndpointFormat('generate', 'GENERATE'),
      };
    }

    // Operational settings
    this.maxBatchSize = config.maxBatchSize ??
      parseInt(process.env.QMD_EMBED_BATCH_SIZE || '32', 10);
    this.embedMaxRetries = config.embedMaxRetries ??
      parseInt(process.env.QMD_EMBED_MAX_RETRIES || '3', 10);
    this.embedReadTimeoutMs = config.embedReadTimeoutMs ??
      parseInt(process.env.QMD_REMOTE_READ_TIMEOUT || '30000', 10);
    this.rerankReadTimeoutMs = config.rerankReadTimeoutMs ??
      parseInt(process.env.QMD_REMOTE_RERANK_TIMEOUT || '60000', 10);
    this.expandReadTimeoutMs = config.expandReadTimeoutMs ??
      parseInt(process.env.QMD_REMOTE_EXPAND_TIMEOUT || '30000', 10);

    this.adapters = resolveAdapterBundle({
      embed: this.embedCfg,
      expand: this.expandCfg,
      rerank: this.rerankCfg,
      generate: this.generateCfg,
    });
  }

  // ── Model name getters ──────────────────────────────────────────────

  get embedModelName(): string { return this.embedCfg.model; }
  get generateModelName(): string | undefined { return this.expandCfg.model; }
  get rerankModelName(): string | undefined { return this.rerankCfg.model; }

  // ── LLM interface: embed ────────────────────────────────────────────

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], options);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.adapters.embed.embedBatch({
      cfg: this.embedCfg,
      breaker: this.embedBreaker,
      log: this.log,
      maxBatchSize: this.maxBatchSize,
      readTimeoutMs: this.embedReadTimeoutMs,
      maxRetries: this.embedMaxRetries,
      dimState: this.dimState,
    }, texts, options);
  }

  // ── LLM interface: generate ─────────────────────────────────────────

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.adapters.generate.generate({
      cfg: this.generateCfg,
      breaker: this.generateBreaker,
      log: this.log,
      readTimeoutMs: this.expandReadTimeoutMs,
    }, prompt, options);
  }

  // ── LLM interface: modelExists ──────────────────────────────────────

  async modelExists(model: string): Promise<ModelInfo> {
    return modelExists(this.embedCfg, model);
  }

  // ── LLM interface: expandQuery ──────────────────────────────────────

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    return this.adapters.expand.expandQuery({
      cfg: this.expandCfg,
      breaker: this.expandBreaker,
      log: this.log,
      readTimeoutMs: this.expandReadTimeoutMs,
    }, query, options);
  }

  // ── LLM interface: rerank ───────────────────────────────────────────

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult> {
    return this.adapters.rerank.rerank({
      cfg: this.rerankCfg,
      breaker: this.rerankBreaker,
      log: this.log,
      readTimeoutMs: this.rerankReadTimeoutMs,
    }, query, documents, options);
  }

  // ── LLM interface: dispose ──────────────────────────────────────────

  async dispose(): Promise<void> {
    // No-op: http/https manages connection pooling automatically
  }

  // ── Utility: probe ──────────────────────────────────────────────────

  async probe(): Promise<{ ok: boolean; dimensions: number; error?: string }> {
    return runProbe((text: string) => this.embed(text));
  }
}
