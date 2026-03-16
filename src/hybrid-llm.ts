/**
 * hybrid-llm.ts - Combines local LlamaCpp (generate/expandQuery) with remote LLM (embed/rerank).
 *
 * When QMD_REMOTE_URL is set, this class routes:
 * - embed/embedBatch/rerank → remote server (e.g. omlx with bge-m3, bge-reranker)
 * - generate/expandQuery → local LlamaCpp (QMD fine-tuned model)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankOptions,
  RerankResult,
  RerankDocument,
  Queryable,
} from "./llm.js";

export class HybridLLM implements LLM {
  readonly isRemote = true;

  constructor(
    private local: LLM,
    private remote: LLM,
  ) {}

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    return this.local.expandQuery(query, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // Try remote first, fallback to local
    const remoteResult = await this.remote.modelExists(model);
    if (remoteResult.exists) return remoteResult;
    return this.local.modelExists(model);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.local.dispose(), this.remote.dispose()]);
  }
}
