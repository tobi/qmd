/**
 * openai-llm.ts - OpenAI API embeddings for QMD
 * 
 * Provides embedding generation using OpenAI's API instead of local models.
 * Much faster and more reliable than local llama-cpp, costs ~$0.02/1M tokens.
 */

import OpenAI from 'openai';
import type { 
  LLM, 
  EmbedOptions, 
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  RerankOptions,
  RerankResult,
  RerankDocument,
  ModelInfo,
  Queryable
} from './llm.js';

export type OpenAIConfig = {
  apiKey?: string;
  embedModel?: string;
  baseURL?: string;
};

const DEFAULT_EMBED_MODEL = 'text-embedding-3-small';

/**
 * OpenAI LLM implementation - primarily for embeddings
 */
export class OpenAIEmbedding implements LLM {
  private client: OpenAI;
  private embedModel: string;

  constructor(config: OpenAIConfig = {}) {
    this.client = new OpenAI({ 
      apiKey: config.apiKey || process.env.OPENAI_API_KEY,
      baseURL: config.baseURL,
    });
    this.embedModel = config.embedModel || DEFAULT_EMBED_MODEL;
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const response = await this.client.embeddings.create({
        model: this.embedModel,
        input: text,
      });
      return {
        embedding: response.data[0].embedding,
        model: this.embedModel,
      };
    } catch (error) {
      console.error('OpenAI embedding error:', error);
      throw error; // Re-throw to see the full error
    }
  }

  getModelName(): string {
    return this.embedModel;
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    try {
      // OpenAI supports batch embedding natively
      const response = await this.client.embeddings.create({
        model: this.embedModel,
        input: texts,
      });
      return response.data.map(item => ({
        embedding: item.embedding,
        model: this.embedModel,
      }));
    } catch (error) {
      console.error('OpenAI batch embedding error:', error);
      return texts.map(() => null);
    }
  }

  // Stub implementations for other LLM interface methods
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    // Not implemented - use local model for generation
    console.warn('OpenAIEmbedding.generate() not implemented, use local model');
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return {
      name: model,
      exists: model === this.embedModel,
    };
  }

  async expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]> {
    // Simple implementation - just return lexical query
    return [{ type: 'lex', text: query }];
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // Not implemented - use local model for reranking
    console.warn('OpenAIEmbedding.rerank() not implemented, returning original order');
    return {
      results: documents.map((doc, index) => ({
        file: doc.file,
        score: 1 - (index * 0.01), // Preserve original order with decreasing scores
        index,
      })),
      model: 'passthrough',
    };
  }

  async dispose(): Promise<void> {
    // No resources to dispose for API client
  }
}

// Singleton instance
let defaultOpenAI: OpenAIEmbedding | null = null;

export function getDefaultOpenAI(config?: OpenAIConfig): OpenAIEmbedding {
  if (!defaultOpenAI) {
    defaultOpenAI = new OpenAIEmbedding(config);
  }
  return defaultOpenAI;
}

export function setDefaultOpenAI(llm: OpenAIEmbedding | null): void {
  defaultOpenAI = llm;
}
