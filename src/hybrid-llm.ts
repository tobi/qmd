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

/**
 * Routes all operations to a remote LLM provider (OpenAI API).
 *
 * When a local LlamaCpp is provided, it serves as a fallback for operations
 * the remote doesn't support. When local is null, all operations go remote
 * (expandQuery, generate, tokenize use the remote implementation).
 */
export class HybridLLM implements LLM {
  readonly isRemote = true;

  constructor(
    private readonly local: LLM | null,
    private readonly remote: LLM,
  ) {}

  async embed(
    text: string,
    options: EmbedOptions = {},
  ): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  async embedBatch(
    texts: string[],
    options: EmbedOptions = {},
  ): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {},
  ): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  async generate(
    prompt: string,
    options: GenerateOptions = {},
  ): Promise<GenerateResult | null> {
    return this.remote.generate(prompt, options);
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    return this.remote.expandQuery(query, options);
  }

  // Tokenization: delegate to remote (character-based approximation).
  // Remote has no real tokenizer but the approximation is good enough
  // for document chunking (~4 chars/token for English text).
  async tokenize(text: string): Promise<readonly unknown[]> {
    return this.remote.tokenize!(text);
  }

  async countTokens(text: string): Promise<number> {
    return this.remote.countTokens!(text);
  }

  async detokenize(tokens: readonly unknown[]): Promise<string> {
    return this.remote.detokenize!(tokens);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    const results = await Promise.allSettled([
      this.remote.modelExists(model),
      ...(this.local ? [this.local.modelExists(model)] : []),
    ]);
    for (const r of results) {
      if (r.status === "fulfilled" && r.value.exists) return r.value;
    }
    return { name: model, exists: false };
  }

  async dispose(): Promise<void> {
    const disposables = [this.remote.dispose()];
    if (this.local) disposables.push(this.local.dispose());
    await Promise.allSettled(disposables);
  }
}
