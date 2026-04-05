/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */
import { type Token as LlamaToken } from "node-llama-cpp";
/**
 * Detect if a model URI uses the Qwen3-Embedding format.
 * Qwen3-Embedding uses a different prompting style than nomic/embeddinggemma.
 */
export declare function isQwen3EmbeddingModel(modelUri: string): boolean;
/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma (default).
 * Uses Qwen3-Embedding instruct format when a Qwen embedding model is active.
 */
export declare function formatQueryForEmbedding(query: string, modelUri?: string): string;
/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields (default).
 * Qwen3-Embedding encodes documents as raw text without special prefixes.
 */
export declare function formatDocForEmbedding(text: string, title?: string, modelUri?: string): string;
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
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
    /** Max session duration in ms (default: 10 minutes) */
    maxDuration?: number;
    /** External abort signal */
    signal?: AbortSignal;
    /** Debug name for logging */
    name?: string;
};
/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
    embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
    embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
    expandQuery(query: string, options?: {
        context?: string;
        includeLexical?: boolean;
    }): Promise<Queryable[]>;
    rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
    /** Whether this session is still valid (not released or aborted) */
    readonly isValid: boolean;
    /** Abort signal for this session (aborts on release or maxDuration) */
    readonly signal: AbortSignal;
}
/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';
/**
 * A single query and its target backend type
 */
export type Queryable = {
    type: QueryType;
    text: string;
};
/**
 * Document to rerank
 */
export type RerankDocument = {
    file: string;
    text: string;
    title?: string;
};
export declare const LFM2_GENERATE_MODEL = "hf:LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf";
export declare const LFM2_INSTRUCT_MODEL = "hf:LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";
export declare const DEFAULT_EMBED_MODEL_URI: string;
export declare const DEFAULT_RERANK_MODEL_URI = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
export declare const DEFAULT_GENERATE_MODEL_URI = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
export declare const DEFAULT_MODEL_CACHE_DIR: string;
export type PullResult = {
    model: string;
    path: string;
    sizeBytes: number;
    refreshed: boolean;
};
export declare function pullModels(models: string[], options?: {
    refresh?: boolean;
    cacheDir?: string;
}): Promise<PullResult[]>;
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
     * Expand a search query into multiple variations for different backends.
     * Returns a list of Queryable objects.
     */
    expandQuery(query: string, options?: {
        context?: string;
        includeLexical?: boolean;
    }): Promise<Queryable[]>;
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
export type LlamaCppConfig = {
    embedModel?: string;
    generateModel?: string;
    rerankModel?: string;
    modelCacheDir?: string;
    /**
     * Context size used for query expansion generation contexts.
     * Default: 2048. Can also be set via QMD_EXPAND_CONTEXT_SIZE.
     */
    expandContextSize?: number;
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
export declare class LlamaCpp implements LLM {
    private readonly _ciMode;
    private llama;
    private embedModel;
    private embedContexts;
    private generateModel;
    private rerankModel;
    private rerankContexts;
    private embedModelUri;
    private generateModelUri;
    private rerankModelUri;
    private modelCacheDir;
    private expandContextSize;
    private embedModelLoadPromise;
    private generateModelLoadPromise;
    private rerankModelLoadPromise;
    private inactivityTimer;
    private inactivityTimeoutMs;
    private disposeModelsOnInactivity;
    private disposed;
    constructor(config?: LlamaCppConfig);
    /**
     * Reset the inactivity timer. Called after each model operation.
     * When timer fires, models are unloaded to free memory (if no active sessions).
     */
    private touchActivity;
    /**
     * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
     */
    private hasLoadedContexts;
    /**
     * Unload idle resources but keep the instance alive for future use.
     *
     * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
     * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
     */
    unloadIdleResources(): Promise<void>;
    /**
     * Ensure model cache directory exists
     */
    private ensureModelCacheDir;
    /**
     * Initialize the llama instance (lazy)
     */
    private ensureLlama;
    /**
     * Resolve a model URI to a local path, downloading if needed
     */
    private resolveModel;
    /**
     * Load embedding model (lazy)
     */
    private ensureEmbedModel;
    /**
     * Compute how many parallel contexts to create.
     *
     * GPU: constrained by VRAM (25% of free, capped at 8).
     * CPU: constrained by cores. Splitting threads across contexts enables
     *      true parallelism (each context runs on its own cores). Use at most
     *      half the math cores, with at least 4 threads per context.
     */
    private computeParallelism;
    /**
     * Get the number of threads each context should use, given N parallel contexts.
     * Splits available math cores evenly across contexts.
     */
    private threadsPerContext;
    /**
     * Load embedding contexts (lazy). Creates multiple for parallel embedding.
     * Uses promise guard to prevent concurrent context creation race condition.
     */
    private embedContextsCreatePromise;
    private ensureEmbedContexts;
    /**
     * Get a single embed context (for single-embed calls). Uses first from pool.
     */
    private ensureEmbedContext;
    /**
     * Load generation model (lazy) - context is created fresh per call
     */
    private ensureGenerateModel;
    /**
     * Load rerank model (lazy)
     */
    private ensureRerankModel;
    /**
     * Load rerank contexts (lazy). Creates multiple contexts for parallel ranking.
     * Each context has its own sequence, so they can evaluate independently.
     *
     * Tuning choices:
     * - contextSize 1024: reranking chunks are ~800 tokens max, 1024 is plenty
     * - flashAttention: ~20% less VRAM per context (568 vs 711 MB)
     * - Combined: drops from 11.6 GB (auto, no flash) to 568 MB per context (20×)
     */
    private static readonly RERANK_CONTEXT_SIZE;
    private static readonly EMBED_CONTEXT_SIZE;
    private ensureRerankContexts;
    /**
     * Tokenize text using the embedding model's tokenizer
     * Returns tokenizer tokens (opaque type from node-llama-cpp)
     */
    tokenize(text: string): Promise<readonly LlamaToken[]>;
    /**
     * Count tokens in text using the embedding model's tokenizer
     */
    countTokens(text: string): Promise<number>;
    /**
     * Detokenize token IDs back to text
     */
    detokenize(tokens: readonly LlamaToken[]): Promise<string>;
    /**
     * Truncate text to fit within the embedding model's context window.
     * Uses the model's own tokenizer for accurate token counting, then
     * detokenizes back to text if truncation is needed.
     * Returns the (possibly truncated) text and whether truncation occurred.
     */
    private truncateToContextSize;
    embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
    /**
     * Batch embed multiple texts efficiently
     * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
     */
    embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
    generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;
    modelExists(modelUri: string): Promise<ModelInfo>;
    expandQuery(query: string, options?: {
        context?: string;
        includeLexical?: boolean;
        intent?: string;
    }): Promise<Queryable[]>;
    private static readonly RERANK_TEMPLATE_OVERHEAD;
    private static readonly RERANK_TARGET_DOCS_PER_CONTEXT;
    rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
    /**
     * Get device/GPU info for status display.
     * Initializes llama if not already done.
     */
    getDeviceInfo(): Promise<{
        gpu: string | false;
        gpuOffloading: boolean;
        gpuDevices: string[];
        vram?: {
            total: number;
            used: number;
            free: number;
        };
        cpuCores: number;
    }>;
    dispose(): Promise<void>;
}
/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export declare class SessionReleasedError extends Error {
    constructor(message?: string);
}
/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const embeddings = await session.embedBatch(texts);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
 * ```
 */
export declare function withLLMSession<T>(fn: (session: ILLMSession) => Promise<T>, options?: LLMSessionOptions): Promise<T>;
/**
 * Execute a function with a scoped LLM session using a specific LlamaCpp instance.
 * Unlike withLLMSession, this does not use the global singleton.
 */
export declare function withLLMSessionForLlm<T>(llm: LlamaCpp, fn: (session: ILLMSession) => Promise<T>, options?: LLMSessionOptions): Promise<T>;
/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export declare function canUnloadLLM(): boolean;
/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export declare function getDefaultLlamaCpp(): LlamaCpp;
/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export declare function setDefaultLlamaCpp(llm: LlamaCpp | null): void;
/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export declare function disposeDefaultLlamaCpp(): Promise<void>;
