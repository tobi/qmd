/**
 * hybrid-llm.ts - Compositor that routes LLM operations between remote and local backends
 *
 * Embed/rerank → remote (GPU-heavy, benefits from offloading)
 * Generate/expandQuery → local LlamaCpp (QMD's fine-tuned query expansion model)
 * tokenize/countTokens → local LlamaCpp (CPU-cheap, needed for chunking)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
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

  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // When remote is a RemoteLLM without a rerank model configured, fall back to local rerank
    // (same fallback shape as expandQuery → local).
    if (this.remote instanceof RemoteLLM && !this.remote.supportsRerank) {
      return this.local.rerank(query, documents, options);
    }
    return this.remote.rerank(query, documents, options);
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

  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    // Route to remote when configured for it; otherwise local (same fallback
    // shape as rerank → local when remote doesn't support rerank).
    if (this.remote instanceof RemoteLLM && this.remote.supportsExpand) {
      return this.remote.expandQuery(query, options);
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
