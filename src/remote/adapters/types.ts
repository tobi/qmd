/**
 * Adapter contracts for remote endpoint protocols.
 *
 * Phase 1 introduces these interfaces so RemoteLLM can delegate endpoint
 * behavior to pluggable adapters instead of embedding protocol-specific logic
 * in the orchestration class.
 */

import type {
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from '../../llm.js';
import type { CircuitBreaker } from '../circuit-breaker.js';
import type { Logger } from '../log.js';
import type { EndpointConfig } from '../types.js';

export type EmbedAdapterContext = {
  cfg: EndpointConfig;
  breaker: CircuitBreaker;
  log: Logger;
  maxBatchSize: number;
  readTimeoutMs: number;
  maxRetries: number;
  dimState: { dimensions: number | null };
};

export type ExpandAdapterContext = {
  cfg: EndpointConfig;
  breaker: CircuitBreaker;
  log: Logger;
  readTimeoutMs: number;
};

export type RerankAdapterContext = {
  cfg: EndpointConfig;
  breaker: CircuitBreaker;
  log: Logger;
  readTimeoutMs: number;
};

export type GenerateAdapterContext = {
  cfg: EndpointConfig;
  breaker: CircuitBreaker;
  log: Logger;
  readTimeoutMs: number;
};

export type EmbedAdapter = {
  /** Stable adapter identifier for diagnostics. */
  id: string;
  embedBatch(
    ctx: EmbedAdapterContext,
    texts: string[],
    options?: EmbedOptions,
  ): Promise<(EmbeddingResult | null)[]>;
};

export type ExpandAdapter = {
  id: string;
  expandQuery(
    ctx: ExpandAdapterContext,
    query: string,
    options?: { includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]>;
};

export type RerankAdapter = {
  id: string;
  rerank(
    ctx: RerankAdapterContext,
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult>;
};

export type GenerateAdapter = {
  id: string;
  generate(
    ctx: GenerateAdapterContext,
    prompt: string,
    options?: GenerateOptions,
  ): Promise<GenerateResult | null>;
};

export type RemoteAdapterBundle = {
  embed: EmbedAdapter;
  expand: ExpandAdapter;
  rerank: RerankAdapter;
  generate: GenerateAdapter;
};

