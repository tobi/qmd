/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import { getLlama, resolveModelFile, type Llama, type LlamaModel, type LlamaEmbeddingContext, type LlamaContext, type LlamaChatSession } from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Structured query expansion result
 */
export type ExpandedQuery = {
  lexicalQuery: string | null;  // Alternative query for BM25/keyword search
  vectorQuery: string;          // Alternative query for semantic search
  hyde: string;                 // Hypothetical document that would answer the query
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations
   */
  expandQuery(query: string, numVariations?: number): Promise<string[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
};

/**
 * LLM implementation using node-llama-cpp
 */
export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContext: LlamaEmbeddingContext | null = null;
  private generateModel: LlamaModel | null = null;
  private generateContext: LlamaContext | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  private initPromise: Promise<void> | null = null;

  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    if (!this.llama) {
      this.llama = await getLlama({ logLevel: "error" });
    }
    return this.llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    // resolveModelFile handles HF URIs and downloads to the cache dir
    return await resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Load embedding model and context (lazy)
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    if (!this.embedContext) {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.embedModelUri);
      this.embedModel = await llama.loadModel({ modelPath });
      this.embedContext = await this.embedModel.createEmbeddingContext();
    }
    return this.embedContext;
  }

  /**
   * Load generation model and context (lazy)
   */
  private async ensureGenerateContext(): Promise<LlamaContext> {
    if (!this.generateContext) {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.generateModelUri);
      this.generateModel = await llama.loadModel({ modelPath });
      // Create context with 4 sequences for parallel generation support
      this.generateContext = await this.generateModel.createContext({ sequences: 4 });
    }
    return this.generateContext;
  }

  /**
   * Load rerank model and context (lazy)
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      this.rerankModel = await llama.loadModel({ modelPath });
      this.rerankContext = await this.rerankModel.createRankingContext();
    }
    return this.rerankContext;
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns array of token IDs
   */
  async tokenize(text: string): Promise<number[]> {
    await this.ensureEmbedContext();  // Ensure model is loaded
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.tokenize(text);
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: number[]): Promise<string> {
    await this.ensureEmbedContext();
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const context = await this.ensureEmbedContext();
      const embedding = await context.getEmbeddingFor(text);

      return {
        embedding: Array.from(embedding.vector),
        model: this.embedModelUri,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      const context = await this.ensureEmbedContext();

      // node-llama-cpp handles batching internally when we make parallel requests
      const embeddings = await Promise.all(
        texts.map(async (text) => {
          try {
            const embedding = await context.getEmbeddingFor(text);
            return {
              embedding: Array.from(embedding.vector),
              model: this.embedModelUri,
            };
          } catch (err) {
            console.error("Embedding error for text:", err);
            return null;
          }
        })
      );

      return embeddings;
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    try {
      const context = await this.ensureGenerateContext();
      const { LlamaChatSession } = await import("node-llama-cpp");
      const session = new LlamaChatSession({
        contextSequence: context.getSequence(),
      });

      const maxTokens = options.maxTokens ?? 150;
      const temperature = options.temperature ?? 0;

      let result = "";
      try {
        await session.prompt(prompt, {
          maxTokens,
          temperature,
          onTextChunk: (text) => {
            result += text;
          },
        });
      } finally {
        // Dispose session to release the sequence
        await session.dispose();
      }

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } catch (error) {
      console.error("Generation error:", error);
      return null;
    }
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, numVariations: number = 2): Promise<string[]> {
    const prompt = `You are a search query expander. Given a search query, generate ${numVariations} alternative queries that would help find relevant documents.

Rules:
- Use synonyms and related terminology
- Rephrase to capture different angles
- Keep proper nouns exactly as written
- Each variation should be 3-8 words, natural search terms
- Do NOT append words like "search" or "find"

Query: "${query}"

Output exactly ${numVariations} variations, one per line, no numbering or bullets:`;

    const result = await this.generate(prompt, {
      maxTokens: 150,
      temperature: 0,
    });

    if (!result) {
      return [query];
    }

    // Parse response - filter out thinking tags and clean up
    const cleanText = result.text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    const lines = cleanText
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 2 && l.length < 100 && !l.startsWith("<"));

    return [query, ...lines.slice(0, numVariations)];
  }

  /**
   * Expand query using structured output with JSON schema grammar.
   * Returns different query types optimized for different retrieval methods.
   *
   * @param query - Original search query
   * @param includeLexical - Whether to include lexical query (false for vector-only search)
   */
  async expandQueryStructured(query: string, includeLexical: boolean = true): Promise<ExpandedQuery> {
    const llama = await this.ensureLlama();
    const context = await this.ensureGenerateContext();

    // Define JSON schema for structured output
    const schema = {
      type: "object" as const,
      properties: {
        lexicalQuery: {
          type: "string" as const,
          description: "Alternative keyword-based query using synonyms (3-6 words)"
        },
        vectorQuery: {
          type: "string" as const,
          description: "Semantically rephrased query capturing the intent (5-10 words)"
        },
        hyde: {
          type: "string" as const,
          description: "Write a short passage (50-100 words) that directly answers the query as if from a relevant document"
        }
      },
      required: ["vectorQuery", "hyde"] as const
    };

    const grammar = await llama.createGrammarForJsonSchema(schema);

    const systemPrompt = includeLexical
      ? `You expand search queries into structured alternatives for a hybrid search system.
Given a query, generate:
1. lexicalQuery: Alternative keywords using synonyms (for BM25 keyword search)
2. vectorQuery: Semantically rephrased query (for vector/embedding search)
3. hyde: Write a brief example passage (50-100 words) that answers the query, as if excerpted from a relevant document

Keep proper nouns exactly as written. Be concise.`
      : `You expand search queries for semantic search.
Given a query, generate:
1. vectorQuery: Semantically rephrased query capturing the full intent
2. hyde: Write a brief example passage (50-100 words) that answers the query, as if excerpted from a relevant document

Keep proper nouns exactly as written. Be concise. Set lexicalQuery to empty string.`;

    const prompt = `Query: "${query}"

Generate the structured expansion:`;

    const { LlamaChatSession } = await import("node-llama-cpp");
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
      systemPrompt,
    });

    try {
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 300,
        temperature: 0,
      });

      const parsed = grammar.parse(result) as {
        lexicalQuery?: string;
        vectorQuery: string;
        hyde: string;
      };

      return {
        lexicalQuery: includeLexical && parsed.lexicalQuery ? parsed.lexicalQuery : null,
        vectorQuery: parsed.vectorQuery || query,
        hyde: parsed.hyde || "",
      };
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      return {
        lexicalQuery: includeLexical ? query : null,
        vectorQuery: query,
        hyde: "",
      };
    } finally {
      await session.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    try {
      const context = await this.ensureRerankContext();

      // Build a map from document text to original indices (for lookup after sorting)
      const textToDoc = new Map<string, { file: string; index: number }>();
      documents.forEach((doc, index) => {
        textToDoc.set(doc.text, { file: doc.file, index });
      });

      // Extract just the text for ranking
      const texts = documents.map((doc) => doc.text);

      // Use the proper ranking API - returns [{document: string, score: number}] sorted by score
      const ranked = await context.rankAndSort(query, texts);

      // Map back to our result format using the text-to-doc map
      const results: RerankDocumentResult[] = ranked.map((item) => {
        const docInfo = textToDoc.get(item.document)!;
        return {
          file: docInfo.file,
          score: item.score,
          index: docInfo.index,
        };
      });

      return {
        results,
        model: this.rerankModelUri,
      };
    } catch (error) {
      console.error("Rerank error:", error);
      // Return documents in original order with zero scores on error
      return {
        results: documents.map((doc, index) => ({
          file: doc.file,
          score: 0,
          index,
        })),
        model: this.rerankModelUri,
      };
    }
  }

  async dispose(): Promise<void> {
    // Dispose contexts
    if (this.embedContext) {
      await this.embedContext.dispose();
      this.embedContext = null;
    }
    if (this.generateContext) {
      await this.generateContext.dispose();
      this.generateContext = null;
    }
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Dispose models
    if (this.embedModel) {
      await this.embedModel.dispose();
      this.embedModel = null;
    }
    if (this.generateModel) {
      await this.generateModel.dispose();
      this.generateModel = null;
    }
    if (this.rerankModel) {
      await this.rerankModel.dispose();
      this.rerankModel = null;
    }

    // Dispose llama
    if (this.llama) {
      await this.llama.dispose();
      this.llama = null;
    }
  }
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp();
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}

// =============================================================================
// Legacy exports for backwards compatibility
// =============================================================================

// Keep Ollama as an alias for now during transition
export { LlamaCpp as Ollama };
export type { LlamaCppConfig as OllamaConfig };

export function getDefaultOllama(): LlamaCpp {
  return getDefaultLlamaCpp();
}

export function setDefaultOllama(llm: LlamaCpp | null): void {
  setDefaultLlamaCpp(llm);
}
