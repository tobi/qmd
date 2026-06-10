/**
 * Legacy adapters that preserve current RemoteLLM behavior.
 *
 * These adapters wrap existing endpoint modules (embed.ts, expand.ts, rerank.ts,
 * generate.ts) and expose them via the new adapter contracts.
 */

import { embedBatch as legacyEmbedBatch } from '../embed.js';
import { expandQuery as legacyExpandQuery } from '../expand.js';
import { rerank as legacyRerank } from '../rerank.js';
import { generate as legacyGenerate } from '../generate.js';
import type {
  EmbedAdapter,
  ExpandAdapter,
  GenerateAdapter,
  RerankAdapter,
} from './types.js';

export const legacyEmbedAdapter: EmbedAdapter = {
  id: 'legacy/openai-embeddings',
  async embedBatch(ctx, texts, options) {
    return legacyEmbedBatch(
      ctx.cfg,
      ctx.breaker,
      texts,
      ctx.maxBatchSize,
      ctx.readTimeoutMs,
      ctx.dimState,
      ctx.maxRetries,
      ctx.log,
      options,
    );
  },
};

export const legacyExpandAdapter: ExpandAdapter = {
  id: 'legacy/openai-chat-expand',
  async expandQuery(ctx, query, options) {
    return legacyExpandQuery(
      ctx.cfg,
      ctx.breaker,
      query,
      ctx.readTimeoutMs,
      options,
      ctx.log,
    );
  },
};

export const legacyRerankAdapter: RerankAdapter = {
  id: 'legacy/cohere-rerank',
  async rerank(ctx, query, documents, _options) {
    return legacyRerank(
      ctx.cfg,
      ctx.breaker,
      query,
      documents,
      ctx.readTimeoutMs,
      ctx.log,
    );
  },
};

export const legacyGenerateAdapter: GenerateAdapter = {
  id: 'legacy/openai-chat-generate',
  async generate(ctx, prompt, options) {
    return legacyGenerate(
      ctx.cfg,
      ctx.breaker,
      prompt,
      ctx.readTimeoutMs,
      ctx.log,
      options,
    );
  },
};

