/**
 * Adapter registry and resolver.
 *
 * Phase 1 keeps behavior stable by routing all currently supported formats to
 * legacy adapters, while centralizing selection logic in one place so future
 * protocol-specific adapters can be added without editing RemoteLLM.
 */

import type { RemoteApiFormat } from '../../collections.js';
import type { EndpointConfig } from '../types.js';
import type {
  EmbedAdapter,
  ExpandAdapter,
  GenerateAdapter,
  RemoteAdapterBundle,
  RerankAdapter,
} from './types.js';
import {
  legacyEmbedAdapter,
  legacyExpandAdapter,
  legacyGenerateAdapter,
  legacyRerankAdapter,
} from './legacy.js';

const EMBED_ADAPTERS: Partial<Record<RemoteApiFormat, EmbedAdapter>> = {
  auto: legacyEmbedAdapter,
  openai_v1_embeddings: legacyEmbedAdapter,
  cohere_v2_embed: legacyEmbedAdapter,
  ollama_embed: legacyEmbedAdapter,
  vllm_pooling: legacyEmbedAdapter,
};

const EXPAND_ADAPTERS: Partial<Record<RemoteApiFormat, ExpandAdapter>> = {
  auto: legacyExpandAdapter,
  openai_chat_completions: legacyExpandAdapter,
  openai_completions: legacyExpandAdapter,
  openai_responses: legacyExpandAdapter,
  anthropic_messages: legacyExpandAdapter,
};

const RERANK_ADAPTERS: Partial<Record<RemoteApiFormat, RerankAdapter>> = {
  auto: legacyRerankAdapter,
  cohere_v1_rerank: legacyRerankAdapter,
  cohere_v2_rerank: legacyRerankAdapter,
  vllm_score: legacyRerankAdapter,
};

const GENERATE_ADAPTERS: Partial<Record<RemoteApiFormat, GenerateAdapter>> = {
  auto: legacyGenerateAdapter,
  openai_chat_completions: legacyGenerateAdapter,
  openai_completions: legacyGenerateAdapter,
  openai_responses: legacyGenerateAdapter,
  anthropic_messages: legacyGenerateAdapter,
};

function pickAdapter<T>(
  registry: Partial<Record<RemoteApiFormat, T>>,
  cfg: EndpointConfig,
  fallback: T,
): T {
  const format = cfg.format ?? 'auto';
  return registry[format] ?? fallback;
}

export function resolveAdapterBundle(cfgs: {
  embed: EndpointConfig;
  expand: EndpointConfig;
  rerank: EndpointConfig;
  generate: EndpointConfig;
}): RemoteAdapterBundle {
  return {
    embed: pickAdapter(EMBED_ADAPTERS, cfgs.embed, legacyEmbedAdapter),
    expand: pickAdapter(EXPAND_ADAPTERS, cfgs.expand, legacyExpandAdapter),
    rerank: pickAdapter(RERANK_ADAPTERS, cfgs.rerank, legacyRerankAdapter),
    generate: pickAdapter(GENERATE_ADAPTERS, cfgs.generate, legacyGenerateAdapter),
  };
}

