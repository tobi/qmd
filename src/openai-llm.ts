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
  expansionModel?: string;
  baseURL?: string;
};

const DEFAULT_EMBED_MODEL = 'text-embedding-3-small';
const DEFAULT_EXPANSION_MODEL = 'gpt-4o-mini';

// Retry configuration
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 60000;

/**
 * Sleep for a given number of milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff + jitter
 * Handles rate limits (429) and transient errors (5xx)
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  options: { maxRetries?: number; context?: string } = {}
): Promise<T> {
  const maxRetries = options.maxRetries ?? MAX_RETRIES;
  const context = options.context ?? 'operation';
  
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: unknown) {
      lastError = error;
      
      // Check if we should retry
      const isRateLimit = error instanceof Error && 
        ('status' in error && (error as { status: number }).status === 429);
      const isServerError = error instanceof Error && 
        ('status' in error && (error as { status: number }).status >= 500);
      const isRetryable = isRateLimit || isServerError;
      
      if (!isRetryable || attempt === maxRetries) {
        throw error;
      }
      
      // Calculate delay with exponential backoff + jitter
      const exponentialDelay = BASE_DELAY_MS * Math.pow(2, attempt);
      const jitter = Math.random() * 0.3 * exponentialDelay; // 0-30% jitter
      const delay = Math.min(exponentialDelay + jitter, MAX_DELAY_MS);
      
      // Check for Retry-After header hint
      let retryAfter = 0;
      if (error instanceof Error && 'headers' in error) {
        const headers = (error as { headers?: { get?: (k: string) => string | null } }).headers;
        const retryAfterHeader = headers?.get?.('retry-after');
        if (retryAfterHeader) {
          retryAfter = parseInt(retryAfterHeader, 10) * 1000;
        }
      }
      
      const finalDelay = Math.max(delay, retryAfter);
      console.warn(`[OpenAI] ${context} failed (attempt ${attempt + 1}/${maxRetries + 1}), ` +
        `retrying in ${Math.round(finalDelay / 1000)}s...`);
      
      await sleep(finalDelay);
    }
  }
  
  throw lastError;
}

/**
 * OpenAI LLM implementation - primarily for embeddings
 */
export class OpenAIEmbedding implements LLM {
  private client: OpenAI;
  private embedModel: string;
  private expansionModel: string;

  constructor(config: OpenAIConfig = {}) {
    this.client = new OpenAI({ 
      apiKey: config.apiKey || process.env.OPENAI_API_KEY,
      baseURL: config.baseURL,
    });
    this.embedModel = config.embedModel || DEFAULT_EMBED_MODEL;
    this.expansionModel = config.expansionModel || DEFAULT_EXPANSION_MODEL;
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return withRetry(async () => {
      const response = await this.client.embeddings.create({
        model: this.embedModel,
        input: text,
      });
      const data = response.data[0];
      if (!data) {
        throw new Error('No embedding data returned from OpenAI');
      }
      return {
        embedding: data.embedding,
        model: this.embedModel,
      };
    }, { context: 'embed' });
  }

  getModelName(): string {
    return this.embedModel;
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return withRetry(async () => {
      // OpenAI supports batch embedding natively
      const response = await this.client.embeddings.create({
        model: this.embedModel,
        input: texts,
      });
      return response.data.map(item => ({
        embedding: item.embedding,
        model: this.embedModel,
      }));
    }, { context: `embedBatch(${texts.length} texts)` });
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
    const includeLexical = options?.includeLexical ?? true;
    
    try {
      const response = await withRetry(() => this.client.chat.completions.create({
        model: this.expansionModel,
        messages: [
          {
            role: 'system',
            content: `You are a search query expander. Given a search query, generate expanded versions for different search backends.

Output format (one per line):
lex: <keyword query for BM25 text search>
vec: <semantic query for vector search>
hyde: <hypothetical document snippet that would answer the query>

Generate 1-2 of each type. Be concise. Include the original query terms.`
          },
          {
            role: 'user',
            content: query
          }
        ],
        temperature: 0.7,
        max_tokens: 300,
      }), { context: 'expandQuery' });

      const content = response.choices[0]?.message?.content || '';
      const lines = content.trim().split('\n');
      
      const queryables: Queryable[] = [];
      for (const line of lines) {
        const colonIdx = line.indexOf(':');
        if (colonIdx === -1) continue;
        
        const type = line.slice(0, colonIdx).trim().toLowerCase();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') continue;
        
        const text = line.slice(colonIdx + 1).trim();
        if (!text) continue;
        
        queryables.push({ type: type as 'lex' | 'vec' | 'hyde', text });
      }

      // Filter lex if not requested
      const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
      
      if (filtered.length > 0) return filtered;
      
      // Fallback if parsing failed
      const fallback: Queryable[] = [
        { type: 'vec', text: query },
        { type: 'hyde', text: `Information about ${query}` },
      ];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
      
    } catch (error) {
      console.error('OpenAI query expansion error:', error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }
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
