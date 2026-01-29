/**
 * remote.ts - Remote embedding providers for QMD
 *
 * Supports:
 * - Voyage AI (voyage-3-lite, rerank-2)
 * - OpenAI-compatible APIs (OpenAI, Azure, Ollama, etc.)
 *
 * Set QMD_PROVIDER to "voyage" or "openai" to use remote embeddings.
 * Requires appropriate API key in environment.
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
  RerankDocumentResult,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

// Voyage AI defaults (voyage-4 series released Jan 2026)
const VOYAGE_API_BASE = "https://api.voyageai.com/v1";
const VOYAGE_EMBED_MODEL = "voyage-4-lite";
const VOYAGE_RERANK_MODEL = "rerank-2";

// OpenAI defaults
const OPENAI_API_BASE = "https://api.openai.com/v1";
const OPENAI_EMBED_MODEL = "text-embedding-3-small";

// =============================================================================
// Types
// =============================================================================

interface EmbedRequest {
  input: string | string[];
  model: string;
  input_type?: "query" | "document"; // Voyage-specific
  dimensions?: number; // OpenAI-specific
}

interface EmbedResponse {
  object: "list";
  data: Array<{
    object: "embedding";
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens?: number;
    total_tokens: number;
  };
}

interface RerankRequest {
  query: string;
  documents: string[];
  model: string;
  top_k?: number;
}

interface RerankResponse {
  object: "list";
  data: Array<{
    index: number;
    relevance_score: number;
  }>;
  model: string;
  usage: {
    total_tokens: number;
  };
}

// =============================================================================
// Remote LLM Implementation
// =============================================================================

export type RemoteProvider = "voyage" | "openai";

export type RemoteConfig = {
  provider?: RemoteProvider;
  apiKey?: string;
  baseUrl?: string;
  embedModel?: string;
  rerankModel?: string;
};

/**
 * LLM implementation using remote embedding APIs (Voyage, OpenAI-compatible)
 */
export class RemoteLLM implements LLM {
  private provider: RemoteProvider;
  private apiKey: string;
  private baseUrl: string;
  private embedModel: string;
  private rerankModel: string;

  constructor(config: RemoteConfig = {}) {
    // Detect provider from env or config
    this.provider = config.provider || 
      (process.env.QMD_PROVIDER as RemoteProvider) || 
      "voyage";

    // Get API key based on provider
    if (this.provider === "voyage") {
      this.apiKey = config.apiKey || process.env.VOYAGE_API_KEY || "";
      this.baseUrl = config.baseUrl || process.env.VOYAGE_API_BASE || VOYAGE_API_BASE;
      this.embedModel = config.embedModel || process.env.VOYAGE_EMBED_MODEL || VOYAGE_EMBED_MODEL;
      this.rerankModel = config.rerankModel || process.env.VOYAGE_RERANK_MODEL || VOYAGE_RERANK_MODEL;
      
      if (!this.apiKey) {
        throw new Error("Voyage API key required. Set VOYAGE_API_KEY environment variable.");
      }
    } else {
      // OpenAI or OpenAI-compatible
      this.apiKey = config.apiKey || process.env.OPENAI_API_KEY || "";
      this.baseUrl = config.baseUrl || process.env.OPENAI_API_BASE || OPENAI_API_BASE;
      this.embedModel = config.embedModel || process.env.OPENAI_EMBED_MODEL || OPENAI_EMBED_MODEL;
      this.rerankModel = ""; // OpenAI doesn't have native reranking
      
      // Only require API key if using default OpenAI endpoint
      if (!this.apiKey && this.baseUrl === OPENAI_API_BASE) {
        throw new Error("OpenAI API key required. Set OPENAI_API_KEY environment variable.");
      }
    }
  }

  /**
   * Make a request to the API
   */
  private async request<T>(endpoint: string, body: object): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error (${response.status}): ${error}`);
    }

    return response.json() as Promise<T>;
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  /**
   * Check if a model name is valid for this provider
   */
  private isValidModel(model: string | undefined): boolean {
    if (!model) return false;
    if (this.provider === "voyage") {
      return model.startsWith("voyage-") || model.startsWith("rerank-");
    }
    // OpenAI models typically contain "embedding" or start with "text-"
    return model.includes("embedding") || model.startsWith("text-");
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      // Only use passed model if it's valid for this provider, otherwise use default
      const model = this.isValidModel(options.model) ? options.model! : this.embedModel;
      
      const body: EmbedRequest = {
        input: [text],
        model,
      };

      // Add input_type for Voyage
      if (this.provider === "voyage") {
        body.input_type = options.isQuery ? "query" : "document";
      }

      const response = await this.request<EmbedResponse>("/embeddings", body);

      if (!response.data || response.data.length === 0 || !response.data[0]) {
        return null;
      }

      return {
        embedding: response.data[0].embedding,
        model: response.model,
      };
    } catch (error) {
      console.error(`${this.provider} embedding error:`, error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   */
  async embedBatch(
    texts: string[],
    options: EmbedOptions = {}
  ): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    // Voyage supports 128 per batch, OpenAI supports 2048
    const BATCH_SIZE = this.provider === "voyage" ? 128 : 2048;
    const results: (EmbeddingResult | null)[] = [];
    
    // Only use passed model if it's valid for this provider
    const model = this.isValidModel(options.model) ? options.model! : this.embedModel;

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);

      try {
        const body: EmbedRequest = {
          input: batch,
          model,
        };

        if (this.provider === "voyage") {
          body.input_type = options.isQuery ? "query" : "document";
        }

        const response = await this.request<EmbedResponse>("/embeddings", body);

        // Map results back by index
        const batchResults: (EmbeddingResult | null)[] = new Array(batch.length).fill(null);
        for (const item of response.data) {
          batchResults[item.index] = {
            embedding: item.embedding,
            model: response.model,
          };
        }
        results.push(...batchResults);
      } catch (error) {
        console.error(`${this.provider} batch embedding error:`, error);
        results.push(...new Array(batch.length).fill(null));
      }
    }

    return results;
  }

  /**
   * Text generation - not supported by embedding APIs
   */
  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult | null> {
    console.warn(`${this.provider} provider does not support text generation.`);
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return {
      name: model,
      exists: true,
    };
  }

  /**
   * Expand query - simplified version without LLM generation
   */
  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const results: Queryable[] = [];

    if (includeLexical) {
      results.push({ type: "lex", text: query });
    }
    results.push({ type: "vec", text: query });

    return results;
  }

  /**
   * Rerank documents - only Voyage supports this natively
   */
  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel };
    }

    // Only Voyage has native reranking
    if (this.provider !== "voyage") {
      console.warn("Reranking not supported by OpenAI provider, returning original order.");
      return {
        results: documents.map((d, i) => ({
          file: d.file,
          score: 1 - (i * 0.01), // Fake descending scores
          index: i,
        })),
        model: "none",
      };
    }

    try {
      const docTexts = documents.map((d) => d.text);

      const body: RerankRequest = {
        query,
        documents: docTexts,
        model: options.model || this.rerankModel,
      };
      if (options.topK) {
        body.top_k = options.topK;
      }

      const response = await this.request<RerankResponse>("/rerank", body);

      const results: RerankDocumentResult[] = response.data.map((item) => ({
        file: documents[item.index]?.file ?? "",
        score: item.relevance_score,
        index: item.index,
      }));

      results.sort((a, b) => b.score - a.score);

      return {
        results,
        model: response.model,
      };
    } catch (error) {
      console.error("Voyage rerank error:", error);
      return {
        results: documents.map((d, i) => ({
          file: d.file,
          score: 0,
          index: i,
        })),
        model: this.rerankModel,
      };
    }
  }

  /**
   * Get the name of the embedding model being used
   */
  getEmbedModel(): string {
    return this.embedModel;
  }

  /**
   * Get the name of the reranker model being used
   */
  getRerankModel(): string {
    return this.rerankModel;
  }

  /**
   * Get the provider name
   */
  getProvider(): string {
    return this.provider;
  }

  /**
   * Dispose - no-op for API-based provider
   */
  async dispose(): Promise<void> {
    // Nothing to dispose
  }
}

// =============================================================================
// Singleton
// =============================================================================

let defaultRemote: RemoteLLM | null = null;

/**
 * Get the default Remote instance
 */
export function getDefaultRemote(): RemoteLLM {
  if (!defaultRemote) {
    defaultRemote = new RemoteLLM();
  }
  return defaultRemote;
}

/**
 * Set a custom default Remote instance
 */
export function setDefaultRemote(llm: RemoteLLM | null): void {
  defaultRemote = llm;
}

/**
 * Dispose the default Remote instance
 */
export async function disposeDefaultRemote(): Promise<void> {
  if (defaultRemote) {
    await defaultRemote.dispose();
    defaultRemote = null;
  }
}
