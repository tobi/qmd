/**
 * voyage.ts - VoyageAI embedding provider for QMD
 *
 * Implements the embedding interface using Voyage-4-large for SOTA retrieval quality.
 * Uses local models for reranking and query expansion (keeps those fast and private).
 */

import { VoyageAIClient } from "voyageai";
import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankDocument,
  RerankOptions,
  RerankResult,
  Queryable,
  LlamaCppConfig,
} from "./llm";

// Voyage models ranked by quality (MTEB/RTEB benchmarks)
export const VOYAGE_MODELS = {
  // Flagship - SOTA on benchmarks, MoE architecture
  "voyage-4-large": { dimensions: 2048, maxTokens: 32000 },
  // High quality, mid-size efficiency
  "voyage-4": { dimensions: 1024, maxTokens: 32000 },
  // Efficient, good quality
  "voyage-4-lite": { dimensions: 512, maxTokens: 32000 },
  // Open weights, local dev
  "voyage-4-nano": { dimensions: 256, maxTokens: 32000 },
  // Legacy models
  "voyage-3-large": { dimensions: 1024, maxTokens: 32000 },
  "voyage-3": { dimensions: 1024, maxTokens: 32000 },
} as const;

export type VoyageModel = keyof typeof VOYAGE_MODELS;

function formatDocForEmbedding(text: string, title?: string): string {
  return title ? `${title}\n\n${text}` : text;
}

export type VoyageConfig = {
  apiKey?: string;
  documentModel?: VoyageModel;
  queryModel?: VoyageModel;
  asymmetric?: boolean;
  outputDimension?: number;
  llamaCppConfig?: LlamaCppConfig;
};

export class VoyageLLM implements LLM {
  private client: VoyageAIClient;
  private documentModel: VoyageModel;
  private queryModel: VoyageModel;
  private outputDimension?: number;
  private llamaCppConfig?: LlamaCppConfig;
  private _llamaCpp: LLM | null = null;

  private async getLlamaCpp(): Promise<LLM> {
    if (!this._llamaCpp) {
      const { LlamaCpp } = await import("./llm");
      this._llamaCpp = new LlamaCpp(this.llamaCppConfig);
    }
    return this._llamaCpp;
  }

  constructor(config: VoyageConfig = {}) {
    const apiKey = config.apiKey || process.env.VOYAGE_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Voyage API key required. Set VOYAGE_API_KEY environment variable or pass apiKey in config."
      );
    }

    this.client = new VoyageAIClient({ apiKey });
    this.documentModel = config.documentModel || "voyage-4-large";
    this.queryModel = config.asymmetric
      ? config.queryModel || "voyage-4-lite"
      : this.documentModel;
    this.outputDimension = config.outputDimension;
    this.llamaCppConfig = config.llamaCppConfig;
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const isQuery = options.isQuery ?? false;
      const model = isQuery ? this.queryModel : this.documentModel;

      const formattedText = isQuery
        ? text
        : formatDocForEmbedding(text, options.title);

      const response = await this.client.embed({
        input: formattedText,
        model,
        inputType: isQuery ? "query" : "document",
        ...(this.outputDimension && { outputDimension: this.outputDimension }),
      });

      if (!response.data || response.data.length === 0) {
        console.error("Voyage API returned empty embeddings");
        return null;
      }

      return {
        embedding: response.data[0].embedding,
        model,
      };
    } catch (error) {
      console.error("Voyage embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently.
   * Voyage API supports up to 128 inputs per request.
   */
  async embedBatch(
    texts: string[],
    options: { isQuery?: boolean; titles?: string[] } = {}
  ): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      const isQuery = options.isQuery ?? false;
      const model = isQuery ? this.queryModel : this.documentModel;

      const formattedTexts = texts.map((text, i) => {
        if (isQuery) return text;
        const title = options.titles?.[i];
        return formatDocForEmbedding(text, title);
      });

      const BATCH_SIZE = 128;
      const results: (EmbeddingResult | null)[] = [];

      for (let i = 0; i < formattedTexts.length; i += BATCH_SIZE) {
        const batch = formattedTexts.slice(i, i + BATCH_SIZE);

        const response = await this.client.embed({
          input: batch,
          model,
          inputType: isQuery ? "query" : "document",
          ...(this.outputDimension && { outputDimension: this.outputDimension }),
        });

        if (!response.data) {
          results.push(...batch.map(() => null));
          continue;
        }

        for (const item of response.data) {
          results.push({
            embedding: item.embedding,
            model,
          });
        }
      }

      return results;
    } catch (error) {
      console.error("Voyage batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.generate(prompt, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    if (model in VOYAGE_MODELS) {
      return { name: model, exists: true };
    }
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.modelExists(model);
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.expandQuery(query, options);
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.rerank(query, documents, options);
  }

  async dispose(): Promise<void> {
    if (this._llamaCpp) {
      await this._llamaCpp.dispose();
      this._llamaCpp = null;
    }
  }

  async tokenize(text: string): Promise<readonly number[]> {
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.tokenize(text);
  }

  async detokenize(tokens: readonly number[]): Promise<string> {
    const llamaCpp = await this.getLlamaCpp();
    return llamaCpp.detokenize(tokens);
  }

  getConfig(): { documentModel: VoyageModel; queryModel: VoyageModel; asymmetric: boolean } {
    return {
      documentModel: this.documentModel,
      queryModel: this.queryModel,
      asymmetric: this.documentModel !== this.queryModel,
    };
  }
}

export type EmbeddingProvider = "local" | "voyage";

export type EmbeddingProviderConfig = {
  provider: EmbeddingProvider;
  voyage?: VoyageConfig;
  llamaCpp?: LlamaCppConfig;
};
