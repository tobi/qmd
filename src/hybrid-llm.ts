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

export class HybridLLM implements LLM {
  constructor(
    private readonly remote: LLM,
    private readonly local: LLM,
  ) {}

  get embedModelName(): string {
    return this.remote.embedModelName;
  }

  // Route to remote
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  // Route to local
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    return this.local.expandQuery(query, options);
  }

  modelExists(model: string): Promise<ModelInfo> {
    return this.local.modelExists(model);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.remote.dispose(), this.local.dispose()]);
  }
}
