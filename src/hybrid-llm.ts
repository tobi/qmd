/**
 * hybrid-llm.ts - Compositor that routes LLM operations between remote and local backends
 *
 * Embed/rerank → remote (GPU-heavy, benefits from offloading)
 * Generate → local LlamaCpp
 * expandQuery → remote when configured, otherwise local
 * tokenize/countTokens → local LlamaCpp (CPU-cheap, needed for chunking)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  LLMExpandQueryOptions,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";
import type { Token as LlamaToken } from "node-llama-cpp";
import { RemoteLLM } from "./remote-llm.js";

export class HybridLLM implements LLM {
  constructor(
    private readonly remote: LLM,
    private readonly local: LLM,
  ) {}

  get embedModelName(): string {
    return this.remote.embedModelName;
  }

  get generateModelName(): string {
    return this.local.generateModelName;
  }

  get expandModelName(): string {
    if (this.remote instanceof RemoteLLM && this.remote.supportsExpand) {
      return this.remote.expandModelName ?? this.remote.generateModelName;
    }
    return this.local.expandModelName ?? this.local.generateModelName;
  }

  get rerankModelName(): string {
    if (this.remote instanceof RemoteLLM && !this.remote.supportsRerank) {
      return this.local.rerankModelName;
    }
    return this.remote.rerankModelName;
  }

  get usesRemoteEmbedding(): boolean {
    return this.remote.usesRemoteEmbedding === true;
  }

  // Route to remote
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // When remote is a RemoteLLM without a rerank model configured, fall back to local rerank
    // (same fallback shape as expandQuery → local).
    if (this.remote instanceof RemoteLLM && !this.remote.supportsRerank) {
      return this.local.rerank(query, documents, options);
    }
    try {
      return await this.remote.rerank(query, documents, options);
    } catch (error) {
      console.error("Remote rerank failed; falling back to local rerank:", error);
      return this.local.rerank(query, documents, options);
    }
  }

  // Route to local
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  tokenize(text: string): Promise<readonly LlamaToken[]> {
    return this.local.tokenize(text);
  }

  detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    return this.local.detokenize(tokens);
  }

  async expandQuery(query: string, options?: LLMExpandQueryOptions): Promise<Queryable[]> {
    // Route to remote when configured for it; otherwise local (same fallback
    // shape as rerank → local when remote doesn't support rerank).
    if (this.remote instanceof RemoteLLM && this.remote.supportsExpand) {
      try {
        return await this.remote.expandQuery(query, options);
      } catch (error) {
        console.error("Remote query expansion failed; falling back to local expansion:", error);
      }
    }
    return this.local.expandQuery(query, options);
  }

  modelExists(model: string): Promise<ModelInfo> {
    return this.local.modelExists(model);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.remote.dispose(), this.local.dispose()]);
  }
}
