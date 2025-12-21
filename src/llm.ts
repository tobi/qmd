/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import {
  getLlama,
  resolveModelFile,
  LlamaChatSession,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
  type Token as LlamaToken,
} from "node-llama-cpp";
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
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 2 minutes
const DEFAULT_INACTIVITY_TIMEOUT_MS = 2 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContext: LlamaEmbeddingContext | null = null;
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  // Ensure we don't load the same model concurrently (which can allocate duplicate VRAM).
  private embedModelLoadPromise: Promise<LlamaModel> | null = null;
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;


  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
  }

  /**
   * Reset the inactivity timer. Called after each model operation.
   * When timer fires, models are unloaded to free memory.
   */
  private touchActivity(): void {
    // Clear existing timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Only set timer if we have disposable contexts and timeout is enabled
    if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        this.unloadIdleResources().catch(err => {
          console.error("Error unloading idle resources:", err);
        });
      }, this.inactivityTimeoutMs);
      // Don't keep process alive just for this timer
      this.inactivityTimer.unref();
    }
  }

  /**
   * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
   */
  private hasLoadedContexts(): boolean {
    return !!(this.embedContext || this.rerankContext);
  }

  /**
   * Unload idle resources but keep the instance alive for future use.
   *
   * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
   * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
   */
  async unloadIdleResources(): Promise<void> {
    // Don't unload if already disposed
    if (this.disposed) {
      return;
    }

    // Clear timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Dispose contexts first
    if (this.embedContext) {
      await this.embedContext.dispose();
      this.embedContext = null;
    }
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Optionally dispose models too (opt-in)
    if (this.disposeModelsOnInactivity) {
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
      // Reset load promises so models can be reloaded later
      this.embedModelLoadPromise = null;
      this.generateModelLoadPromise = null;
      this.rerankModelLoadPromise = null;
    }

    // Note: We keep llama instance alive - it's lightweight
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
      this.llama = await getLlama({ logLevel: LlamaLogLevel.error });
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
   * Load embedding model (lazy)
   */
  private async ensureEmbedModel(): Promise<LlamaModel> {
    if (this.embedModel) {
      return this.embedModel;
    }
    if (this.embedModelLoadPromise) {
      return await this.embedModelLoadPromise;
    }

    this.embedModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.embedModelUri);
      const model = await llama.loadModel({ modelPath });
      this.embedModel = model;
      return model;
    })();

    try {
      return await this.embedModelLoadPromise;
    } finally {
      // Keep the resolved model cached; clear only the in-flight promise.
      this.embedModelLoadPromise = null;
    }
  }

  /**
   * Load embedding context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    if (!this.embedContext) {
      const model = await this.ensureEmbedModel();
      this.embedContext = await model.createEmbeddingContext();
    }
    this.touchActivity();
    return this.embedContext;
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    if (!this.generateModel) {
      if (this.generateModelLoadPromise) {
        return await this.generateModelLoadPromise;
      }

      this.generateModelLoadPromise = (async () => {
        const llama = await this.ensureLlama();
        const modelPath = await this.resolveModel(this.generateModelUri);
        const model = await llama.loadModel({ modelPath });
        this.generateModel = model;
        return model;
      })();

      try {
        await this.generateModelLoadPromise;
      } finally {
        this.generateModelLoadPromise = null;
      }
    }
    this.touchActivity();
    if (!this.generateModel) {
      throw new Error("Generate model not loaded");
    }
    return this.generateModel;
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    if (this.rerankModel) {
      return this.rerankModel;
    }
    if (this.rerankModelLoadPromise) {
      return await this.rerankModelLoadPromise;
    }

    this.rerankModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      const model = await llama.loadModel({ modelPath });
      this.rerankModel = model;
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
      this.rerankModelLoadPromise = null;
    }
  }

  /**
   * Load rerank context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const model = await this.ensureRerankModel();
      this.rerankContext = await model.createRankingContext();
    }
    this.touchActivity();
    return this.rerankContext;
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
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
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
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
    // Ensure model is loaded
    await this.ensureGenerateModel();

    // Create fresh context -> sequence -> session for each call
    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

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

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
      await context.dispose();
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
    await this.ensureGenerateModel();

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
      required: [] as const
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
1. vectorQuery: Semantically rephrased query capturing the full intent (must be different from the original query)
2. HyDE: Write a brief example passage (50-100 words) that answers the query, as if excerpted from a relevant document

Keep proper nouns exactly as written. Be concise.`;

    const prompt = `Query: "${query}"

Generate the structured expansion:`;

    // Create fresh context for each call
    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence, systemPrompt });

    try {
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 500,
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
      // Dispose context (disposes session too per lifecycle rules)
      await context.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
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
  }

  async dispose(): Promise<void> {
    // Prevent double-dispose
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    // Clear inactivity timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Disposing llama cascades to models and contexts automatically
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
    if (this.llama) {
      await this.llama.dispose();
    }

    // Clear references
    this.embedContext = null;
    this.rerankContext = null;
    this.embedModel = null;
    this.generateModel = null;
    this.rerankModel = null;
    this.llama = null;

    // Clear any in-flight load promises
    this.embedModelLoadPromise = null;
    this.generateModelLoadPromise = null;
    this.rerankModelLoadPromise = null;
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

