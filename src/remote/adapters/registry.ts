/**
 * Adapter registry and resolver.
 *
 * Phase 1 kept behavior stable by routing all formats to legacy adapters.
 * Phase 2 wired OpenAI-specific protocol adapters.
 * Phase 3 wires Anthropic Messages protocol adapters.
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
import {
  openaiChatCompletionsExpandAdapter,
  openaiChatCompletionsGenerateAdapter,
} from './openai-chat.js';
import {
  openaiCompletionsExpandAdapter,
  openaiCompletionsGenerateAdapter,
} from './openai-completions.js';
import {
  openaiResponsesExpandAdapter,
  openaiResponsesGenerateAdapter,
} from './openai-responses.js';
import {
  anthropicMessagesExpandAdapter,
  anthropicMessagesGenerateAdapter,
} from './anthropic-messages.js';
import { cohereV2EmbedAdapter } from './cohere-embed.js';
import { cohereRerankAdapter } from './cohere-rerank.js';
import { vllmScoreAdapter } from './vllm-score.js';

const EMBED_ADAPTERS: Partial<Record<RemoteApiFormat, EmbedAdapter>> = {
  auto: legacyEmbedAdapter,
  openai_v1_embeddings: legacyEmbedAdapter,
  cohere_v2_embed: cohereV2EmbedAdapter,
  ollama_embed: legacyEmbedAdapter,
  vllm_pooling: legacyEmbedAdapter,
};

const EXPAND_ADAPTERS: Partial<Record<RemoteApiFormat, ExpandAdapter>> = {
  auto: legacyExpandAdapter,
  openai_chat_completions: openaiChatCompletionsExpandAdapter,
  openai_completions: openaiCompletionsExpandAdapter,
  openai_responses: openaiResponsesExpandAdapter,
  anthropic_messages: anthropicMessagesExpandAdapter,
};

const RERANK_ADAPTERS: Partial<Record<RemoteApiFormat, RerankAdapter>> = {
  auto: legacyRerankAdapter,
  cohere_v1_rerank: cohereRerankAdapter,
  cohere_v2_rerank: cohereRerankAdapter,
  vllm_score: vllmScoreAdapter,
};

const GENERATE_ADAPTERS: Partial<Record<RemoteApiFormat, GenerateAdapter>> = {
  auto: legacyGenerateAdapter,
  openai_chat_completions: openaiChatCompletionsGenerateAdapter,
  openai_completions: openaiCompletionsGenerateAdapter,
  openai_responses: openaiResponsesGenerateAdapter,
  anthropic_messages: anthropicMessagesGenerateAdapter,
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
