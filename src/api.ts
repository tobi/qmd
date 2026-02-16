/**
 * api.ts - API-backed LLM implementation (incremental rollout)
 *
 * Current phase: embeddings (/v1/embeddings) and rerank (/v1/rerank).
 * Query expansion/generation can delegate to a fallback backend.
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

const DEFAULT_EMBED_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_EMBED_MODEL = "text-embedding-3-small";
const DEFAULT_RERANK_BASE_URL = "https://api.cohere.com/v1";
const DEFAULT_RERANK_MODEL = "rerank-v3.5";

type OpenAIEmbeddingResponse = {
  data?: Array<{ embedding?: number[] }>;
};

type CohereRerankResponse = {
  results?: Array<{ index?: number; relevance_score?: number }>;
};

export type ApiLLMConfig = {
  embedBaseUrl?: string;
  embedApiKey?: string;
  embedModel?: string;
  rerankBaseUrl?: string;
  rerankApiKey?: string;
  rerankModel?: string;
  fallbackLLM?: LLM;
};

/**
 * API-backed LLM implementation.
 * Embeddings/reranking are remote; query expansion/generation can fallback.
 */
export class ApiLLM implements LLM {
  private readonly embedBaseUrl: string;
  private readonly embedApiKey: string;
  private readonly embedModel: string;
  private readonly rerankBaseUrl: string;
  private readonly rerankApiKey: string;
  private readonly rerankModel: string;
  private readonly fallbackLLM?: LLM;

  constructor(config: ApiLLMConfig = {}) {
    const normalizedEmbedBaseUrl = (
      config.embedBaseUrl
      || process.env.QMD_API_BASE_URL
      || process.env.OPENAI_BASE_URL
      || DEFAULT_EMBED_BASE_URL
    ).replace(/\/+$/, "");
    this.embedBaseUrl = normalizedEmbedBaseUrl;

    this.embedApiKey = config.embedApiKey || process.env.QMD_API_KEY || process.env.OPENAI_API_KEY || "";
    this.embedModel = config.embedModel || process.env.QMD_API_EMBED_MODEL || process.env.OPENAI_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.rerankBaseUrl = (
      config.rerankBaseUrl
      || process.env.QMD_API_RERANK_BASE_URL
      || process.env.COHERE_BASE_URL
      || (process.env.COHERE_API_KEY ? DEFAULT_RERANK_BASE_URL : normalizedEmbedBaseUrl)
    ).replace(/\/+$/, "");
    this.rerankApiKey = config.rerankApiKey || process.env.QMD_API_RERANK_KEY || process.env.COHERE_API_KEY || this.embedApiKey;
    this.rerankModel = config.rerankModel || process.env.QMD_API_RERANK_MODEL || process.env.COHERE_RERANK_MODEL || DEFAULT_RERANK_MODEL;
    this.fallbackLLM = config.fallbackLLM;
  }

  private getHeaders(apiKey: string): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    };
  }

  private getFallback(method: string): LLM {
    if (!this.fallbackLLM) {
      throw new Error(`ApiLLM.${method} is not implemented without fallback backend`);
    }
    return this.fallbackLLM;
  }

  private isLikelyLocalModel(model: string): boolean {
    const lower = model.toLowerCase();
    return (
      model.startsWith("hf:")
      || lower.includes(".gguf")
      || lower === "embeddinggemma"
      || lower.includes("qwen3-reranker")
      || lower.startsWith("expedientfalcon/")
    );
  }

  private resolveModel(modelOverride: string | undefined, configuredModel: string): string {
    if (!modelOverride) return configuredModel;
    return this.isLikelyLocalModel(modelOverride) ? configuredModel : modelOverride;
  }

  private async requestEmbeddings(texts: string[], modelOverride?: string): Promise<OpenAIEmbeddingResponse | null> {
    if (!this.embedApiKey) {
      console.error("ApiLLM embedding error: missing API key (set QMD_API_KEY or OPENAI_API_KEY)");
      return null;
    }

    const model = this.resolveModel(modelOverride, this.embedModel);
    try {
      const resp = await fetch(`${this.embedBaseUrl}/embeddings`, {
        method: "POST",
        headers: this.getHeaders(this.embedApiKey),
        body: JSON.stringify({
          model,
          input: texts,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        console.error(`ApiLLM embedding error: ${resp.status} ${resp.statusText} ${body}`.trim());
        return null;
      }
      return await resp.json() as OpenAIEmbeddingResponse;
    } catch (error) {
      console.error("ApiLLM embedding error:", error);
      return null;
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const model = this.resolveModel(options.model, this.embedModel);
    const response = await this.requestEmbeddings([text], model);
    const vector = response?.data?.[0]?.embedding;
    if (!vector || !Array.isArray(vector)) return null;

    return {
      embedding: vector,
      model,
    };
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const response = await this.requestEmbeddings(texts);
    if (!response?.data || !Array.isArray(response.data)) {
      return texts.map(() => null);
    }

    const results: (EmbeddingResult | null)[] = [];
    for (let i = 0; i < texts.length; i++) {
      const vector = response.data[i]?.embedding;
      if (!vector || !Array.isArray(vector)) {
        results.push(null);
      } else {
        results.push({
          embedding: vector,
          model: this.embedModel,
        });
      }
    }
    return results;
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.getFallback("generate").generate(prompt, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]> {
    return this.getFallback("expandQuery").expandQuery(query, options);
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    if (!this.rerankApiKey) {
      throw new Error("ApiLLM rerank error: missing API key (set QMD_API_RERANK_KEY or COHERE_API_KEY)");
    }
    if (documents.length === 0) {
      return { results: [], model: this.resolveModel(options.model, this.rerankModel) };
    }

    const model = this.resolveModel(options.model, this.rerankModel);

    let response: CohereRerankResponse;
    try {
      const resp = await fetch(`${this.rerankBaseUrl}/rerank`, {
        method: "POST",
        headers: this.getHeaders(this.rerankApiKey),
        body: JSON.stringify({
          model,
          query,
          documents: documents.map((doc) => doc.text),
          top_n: documents.length,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`ApiLLM rerank error: ${resp.status} ${resp.statusText} ${body}`.trim());
      }
      response = await resp.json() as CohereRerankResponse;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`ApiLLM rerank request failed: ${detail}`);
    }

    if (!Array.isArray(response.results)) {
      throw new Error("ApiLLM rerank error: invalid response (missing results array)");
    }

    const scoreByIndex = new Map<number, number>();
    for (const item of response.results) {
      if (typeof item.index !== "number" || typeof item.relevance_score !== "number") continue;
      scoreByIndex.set(item.index, item.relevance_score);
    }

    const results = documents
      .map((doc, index) => ({
        file: doc.file,
        score: scoreByIndex.get(index) ?? 0,
        index,
      }))
      .sort((a, b) => b.score - a.score);

    return {
      results,
      model,
    };
  }

  async dispose(): Promise<void> {
    // No API client resources to dispose in this implementation.
  }
}
