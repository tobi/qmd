import type {
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  LLM,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

export class HybridLLM implements LLM {
  readonly isRemote = true;

  constructor(
    private readonly local: LLM,
    private readonly remote: LLM,
  ) {}

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    return this.local.expandQuery(query, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    const [remoteResult, localResult] = await Promise.all([
      this.remote.modelExists(model).catch(() => ({ name: model, exists: false })),
      this.local.modelExists(model).catch(() => ({ name: model, exists: false })),
    ]);

    return remoteResult.exists ? remoteResult : localResult;
  }

  async dispose(): Promise<void> {
    await Promise.allSettled([this.local.dispose(), this.remote.dispose()]);
  }
}
