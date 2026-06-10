/**
 * types.ts — Configuration types for remote API endpoints.
 *
 * Each endpoint (embed, expand, rerank, generate) has independent config:
 * a base URL, a model name, and an optional API key. This allows different
 * providers per endpoint — e.g. vLLM for embedding, OpenRouter for reranking.
 *
 * @module remote/types
 */
import type { RemoteApiFormat } from '../collections.js';

/** Logical endpoint roles used by RemoteLLM. */
export type EndpointRole = 'embed' | 'expand' | 'rerank' | 'generate';

/**
 * Configuration for a single remote API endpoint.
 *
 * Each endpoint can target a different server and model. apiKey is optional —
 * endpoints without one gracefully degrade (expand/rerank return passthrough
 * or uniform results).
 */
export type EndpointConfig = {
  /** Base URL for the API (e.g. https://openrouter.ai/api/v1). Trailing slash trimmed. */
  baseUrl: string;
  /** Protocol contract for this endpoint (used by adapter selection). */
  format?: RemoteApiFormat;
  /** Model identifier string (e.g. google/gemini-2.0-flash-lite-001) */
  model: string;
  /** Optional Bearer token for authenticated endpoints */
  apiKey?: string;
};

/**
 * Full configuration for RemoteLLM.
 *
 * Canonical form uses per-endpoint config objects. Backward-compat
 * flat fields (baseUrl, embedModel, apiKey) are treated as the embed
 * endpoint and overridden by explicit `embed` config.
 */
export type RemoteLLMConfig = {
  /** Embedding endpoint config (POST /v1/embeddings) */
  embed?: EndpointConfig;
  /** Query expansion endpoint config (POST /v1/chat/completions) */
  expand?: EndpointConfig;
  /** Reranking endpoint config (POST /v1/rerank) */
  rerank?: EndpointConfig;
  /** Text generation endpoint config (POST /v1/chat/completions) */
  generate?: EndpointConfig;

  /** @deprecated Use `embed.baseUrl` instead. Old flat config for embed endpoint. */
  baseUrl?: string;
  /** @deprecated Use `embed.model` instead. Old flat config for embed endpoint. */
  embedModel?: string;
  /** @deprecated Use `embed.apiKey` instead. Old flat config for embed endpoint. */
  apiKey?: string;

  /** Max texts per batch request (default: 32). Lower for slow servers. */
  maxBatchSize?: number;
  /** Read timeout for embedding requests in ms (default: 30000) */
  embedReadTimeoutMs?: number;
  /** Read timeout for rerank requests in ms (default: 60000). Reranking is slower. */
  rerankReadTimeoutMs?: number;
  /** Read timeout for expand/generate requests in ms (default: 30000) */
  expandReadTimeoutMs?: number;
  /** Max retries for failed embed batch requests (default: 3). Uses exponential backoff. */
  embedMaxRetries?: number;
};
