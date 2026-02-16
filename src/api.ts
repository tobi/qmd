/**
 * api.ts - API-backed LLM implementation (incremental rollout)
 *
 * Current phase: embeddings via OpenAI-compatible /v1/embeddings.
 * Other capabilities can delegate to a fallback backend.
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

const DEFAULT_API_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_EMBED_MODEL = "text-embedding-3-small";

type OpenAIEmbeddingResponse = {
  data?: Array<{ embedding?: number[] }>;
};

export type ApiLLMConfig = {
  baseUrl?: string;
  apiKey?: string;
  embedModel?: string;
  fallbackLLM?: LLM;
};

/**
 * API-backed LLM implementation.
 * Embeddings are remote; other methods delegate to fallback when provided.
 */
export class ApiLLM implements LLM {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly embedModel: string;
  private readonly fallbackLLM?: LLM;

  constructor(config: ApiLLMConfig = {}) {
    this.baseUrl = (
      config.baseUrl
      || process.env.QMD_API_BASE_URL
      || process.env.OPENAI_BASE_URL
      || DEFAULT_API_BASE_URL
    ).replace(/\/+$/, "");
    this.apiKey = config.apiKey || process.env.QMD_API_KEY || process.env.OPENAI_API_KEY || "";
    this.embedModel = config.embedModel || process.env.QMD_API_EMBED_MODEL || process.env.OPENAI_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.fallbackLLM = config.fallbackLLM;
  }

  private getHeaders(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${this.apiKey}`,
    };
  }

  private getFallback(method: string): LLM {
    if (!this.fallbackLLM) {
      throw new Error(`ApiLLM.${method} is not implemented without fallback backend`);
    }
    return this.fallbackLLM;
  }

  private async requestEmbeddings(texts: string[], modelOverride?: string): Promise<OpenAIEmbeddingResponse | null> {
    if (!this.apiKey) {
      console.error("ApiLLM embedding error: missing API key (set QMD_API_KEY or OPENAI_API_KEY)");
      return null;
    }

    const model = modelOverride || this.embedModel;
    try {
      const resp = await fetch(`${this.baseUrl}/embeddings`, {
        method: "POST",
        headers: this.getHeaders(),
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
    const response = await this.requestEmbeddings([text], options.model);
    const vector = response?.data?.[0]?.embedding;
    if (!vector || !Array.isArray(vector)) return null;

    return {
      embedding: vector,
      model: options.model || this.embedModel,
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
    return this.getFallback("rerank").rerank(query, documents, options);
  }

  async dispose(): Promise<void> {
    // No API client resources to dispose in this implementation.
  }
}

