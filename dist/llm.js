/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */
import { getLlama, resolveModelFile, LlamaChatSession, LlamaLogLevel, } from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";
// =============================================================================
// Embedding Formatting Functions
// =============================================================================
/**
 * Detect if a model URI uses the Qwen3-Embedding format.
 * Qwen3-Embedding uses a different prompting style than nomic/embeddinggemma.
 */
export function isQwen3EmbeddingModel(modelUri) {
    return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}
/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma (default).
 * Uses Qwen3-Embedding instruct format when a Qwen embedding model is active.
 */
export function formatQueryForEmbedding(query, modelUri) {
    const uri = modelUri ?? process.env.QMD_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;
    if (isQwen3EmbeddingModel(uri)) {
        return `Instruct: Retrieve relevant documents for the given query\nQuery: ${query}`;
    }
    return `task: search result | query: ${query}`;
}
/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields (default).
 * Qwen3-Embedding encodes documents as raw text without special prefixes.
 */
export function formatDocForEmbedding(text, title, modelUri) {
    const uri = modelUri ?? process.env.QMD_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;
    if (isQwen3EmbeddingModel(uri)) {
        // Qwen3-Embedding: documents are raw text, no task prefix
        return title ? `${title}\n${text}` : text;
    }
    return `title: ${title || "none"} | text: ${text}`;
}
// =============================================================================
// Model Configuration
// =============================================================================
// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
// Override via QMD_EMBED_MODEL env var (e.g. hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf)
const DEFAULT_EMBED_MODEL = process.env.QMD_EMBED_MODEL ?? "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
// Alternative generation models for query expansion:
// LiquidAI LFM2 - hybrid architecture optimized for edge/on-device inference
// Use these as base for fine-tuning with configs/sft_lfm2.yaml
export const LFM2_GENERATE_MODEL = "hf:LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf";
export const LFM2_INSTRUCT_MODEL = "hf:LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";
export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;
// Local model cache directory
const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
    ? join(process.env.XDG_CACHE_HOME, "qmd", "models")
    : join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;
function parseHfUri(model) {
    if (!model.startsWith("hf:"))
        return null;
    const without = model.slice(3);
    const parts = without.split("/");
    if (parts.length < 3)
        return null;
    const repo = parts.slice(0, 2).join("/");
    const file = parts.slice(2).join("/");
    return { repo, file };
}
async function getRemoteEtag(ref) {
    const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
    try {
        const resp = await fetch(url, { method: "HEAD" });
        if (!resp.ok)
            return null;
        const etag = resp.headers.get("etag");
        return etag || null;
    }
    catch {
        return null;
    }
}
export async function pullModels(models, options = {}) {
    const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
    if (!existsSync(cacheDir)) {
        mkdirSync(cacheDir, { recursive: true });
    }
    const results = [];
    for (const model of models) {
        let refreshed = false;
        const hfRef = parseHfUri(model);
        const filename = model.split("/").pop();
        const entries = readdirSync(cacheDir, { withFileTypes: true });
        const cached = filename
            ? entries
                .filter((entry) => entry.isFile() && entry.name.includes(filename))
                .map((entry) => join(cacheDir, entry.name))
            : [];
        if (hfRef && filename) {
            const etagPath = join(cacheDir, `${filename}.etag`);
            const remoteEtag = await getRemoteEtag(hfRef);
            const localEtag = existsSync(etagPath)
                ? readFileSync(etagPath, "utf-8").trim()
                : null;
            const shouldRefresh = options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;
            if (shouldRefresh) {
                for (const candidate of cached) {
                    if (existsSync(candidate))
                        unlinkSync(candidate);
                }
                if (existsSync(etagPath))
                    unlinkSync(etagPath);
                refreshed = cached.length > 0;
            }
        }
        else if (options.refresh && filename) {
            for (const candidate of cached) {
                if (existsSync(candidate))
                    unlinkSync(candidate);
                refreshed = true;
            }
        }
        const path = await resolveModelFile(model, cacheDir);
        const sizeBytes = existsSync(path) ? statSync(path).size : 0;
        if (hfRef && filename) {
            const remoteEtag = await getRemoteEtag(hfRef);
            if (remoteEtag) {
                const etagPath = join(cacheDir, `${filename}.etag`);
                writeFileSync(etagPath, remoteEtag + "\n", "utf-8");
            }
        }
        results.push({ model, path, sizeBytes, refreshed });
    }
    return results;
}
/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;
const DEFAULT_EXPAND_CONTEXT_SIZE = 2048;
function resolveExpandContextSize(configValue) {
    if (configValue !== undefined) {
        if (!Number.isInteger(configValue) || configValue <= 0) {
            throw new Error(`Invalid expandContextSize: ${configValue}. Must be a positive integer.`);
        }
        return configValue;
    }
    const envValue = process.env.QMD_EXPAND_CONTEXT_SIZE?.trim();
    if (!envValue)
        return DEFAULT_EXPAND_CONTEXT_SIZE;
    const parsed = Number.parseInt(envValue, 10);
    if (!Number.isInteger(parsed) || parsed <= 0) {
        process.stderr.write(`QMD Warning: invalid QMD_EXPAND_CONTEXT_SIZE="${envValue}", using default ${DEFAULT_EXPAND_CONTEXT_SIZE}.\n`);
        return DEFAULT_EXPAND_CONTEXT_SIZE;
    }
    return parsed;
}
export class LlamaCpp {
    _ciMode = !!process.env.CI;
    llama = null;
    embedModel = null;
    embedContexts = [];
    generateModel = null;
    rerankModel = null;
    rerankContexts = [];
    embedModelUri;
    generateModelUri;
    rerankModelUri;
    modelCacheDir;
    expandContextSize;
    // Ensure we don't load the same model/context concurrently (which can allocate duplicate VRAM).
    embedModelLoadPromise = null;
    generateModelLoadPromise = null;
    rerankModelLoadPromise = null;
    // Inactivity timer for auto-unloading models
    inactivityTimer = null;
    inactivityTimeoutMs;
    disposeModelsOnInactivity;
    // Track disposal state to prevent double-dispose
    disposed = false;
    constructor(config = {}) {
        this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
        this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
        this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
        this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
        this.expandContextSize = resolveExpandContextSize(config.expandContextSize);
        this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
        this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
    }
    /**
     * Reset the inactivity timer. Called after each model operation.
     * When timer fires, models are unloaded to free memory (if no active sessions).
     */
    touchActivity() {
        // Clear existing timer
        if (this.inactivityTimer) {
            clearTimeout(this.inactivityTimer);
            this.inactivityTimer = null;
        }
        // Only set timer if we have disposable contexts and timeout is enabled
        if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
            this.inactivityTimer = setTimeout(() => {
                // Check if session manager allows unloading
                // canUnloadLLM is defined later in this file - it checks the session manager
                // We use dynamic import pattern to avoid circular dependency issues
                if (typeof canUnloadLLM === 'function' && !canUnloadLLM()) {
                    // Active sessions/operations - reschedule timer
                    this.touchActivity();
                    return;
                }
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
    hasLoadedContexts() {
        return !!(this.embedContexts.length > 0 || this.rerankContexts.length > 0);
    }
    /**
     * Unload idle resources but keep the instance alive for future use.
     *
     * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
     * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
     */
    async unloadIdleResources() {
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
        for (const ctx of this.embedContexts) {
            await ctx.dispose();
        }
        this.embedContexts = [];
        for (const ctx of this.rerankContexts) {
            await ctx.dispose();
        }
        this.rerankContexts = [];
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
    ensureModelCacheDir() {
        if (!existsSync(this.modelCacheDir)) {
            mkdirSync(this.modelCacheDir, { recursive: true });
        }
    }
    /**
     * Initialize the llama instance (lazy)
     */
    async ensureLlama() {
        if (!this.llama) {
            const llama = await getLlama({
                // attempt to build
                build: "autoAttempt",
                logLevel: LlamaLogLevel.error
            });
            if (llama.gpu === false) {
                process.stderr.write("QMD Warning: no GPU acceleration, running on CPU (slow). Run 'qmd status' for details.\n");
            }
            this.llama = llama;
        }
        return this.llama;
    }
    /**
     * Resolve a model URI to a local path, downloading if needed
     */
    async resolveModel(modelUri) {
        this.ensureModelCacheDir();
        // resolveModelFile handles HF URIs and downloads to the cache dir
        return await resolveModelFile(modelUri, this.modelCacheDir);
    }
    /**
     * Load embedding model (lazy)
     */
    async ensureEmbedModel() {
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
            // Model loading counts as activity - ping to keep alive
            this.touchActivity();
            return model;
        })();
        try {
            return await this.embedModelLoadPromise;
        }
        finally {
            // Keep the resolved model cached; clear only the in-flight promise.
            this.embedModelLoadPromise = null;
        }
    }
    /**
     * Compute how many parallel contexts to create.
     *
     * GPU: constrained by VRAM (25% of free, capped at 8).
     * CPU: constrained by cores. Splitting threads across contexts enables
     *      true parallelism (each context runs on its own cores). Use at most
     *      half the math cores, with at least 4 threads per context.
     */
    async computeParallelism(perContextMB) {
        const llama = await this.ensureLlama();
        if (llama.gpu) {
            try {
                const vram = await llama.getVramState();
                const freeMB = vram.free / (1024 * 1024);
                const maxByVram = Math.floor((freeMB * 0.25) / perContextMB);
                return Math.max(1, Math.min(8, maxByVram));
            }
            catch {
                return 2;
            }
        }
        // CPU: split cores across contexts. At least 4 threads per context.
        const cores = llama.cpuMathCores || 4;
        const maxContexts = Math.floor(cores / 4);
        return Math.max(1, Math.min(4, maxContexts));
    }
    /**
     * Get the number of threads each context should use, given N parallel contexts.
     * Splits available math cores evenly across contexts.
     */
    async threadsPerContext(parallelism) {
        const llama = await this.ensureLlama();
        if (llama.gpu)
            return 0; // GPU: let the library decide
        const cores = llama.cpuMathCores || 4;
        return Math.max(1, Math.floor(cores / parallelism));
    }
    /**
     * Load embedding contexts (lazy). Creates multiple for parallel embedding.
     * Uses promise guard to prevent concurrent context creation race condition.
     */
    embedContextsCreatePromise = null;
    async ensureEmbedContexts() {
        if (this.embedContexts.length > 0) {
            this.touchActivity();
            return this.embedContexts;
        }
        if (this.embedContextsCreatePromise) {
            return await this.embedContextsCreatePromise;
        }
        this.embedContextsCreatePromise = (async () => {
            const model = await this.ensureEmbedModel();
            // Embed contexts are ~143 MB each (nomic-embed 2048 ctx)
            const n = await this.computeParallelism(150);
            const threads = await this.threadsPerContext(n);
            for (let i = 0; i < n; i++) {
                try {
                    this.embedContexts.push(await model.createEmbeddingContext({
                        contextSize: LlamaCpp.EMBED_CONTEXT_SIZE,
                        ...(threads > 0 ? { threads } : {}),
                    }));
                }
                catch {
                    if (this.embedContexts.length === 0)
                        throw new Error("Failed to create any embedding context");
                    break;
                }
            }
            this.touchActivity();
            return this.embedContexts;
        })();
        try {
            return await this.embedContextsCreatePromise;
        }
        finally {
            this.embedContextsCreatePromise = null;
        }
    }
    /**
     * Get a single embed context (for single-embed calls). Uses first from pool.
     */
    async ensureEmbedContext() {
        const contexts = await this.ensureEmbedContexts();
        return contexts[0];
    }
    /**
     * Load generation model (lazy) - context is created fresh per call
     */
    async ensureGenerateModel() {
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
            }
            finally {
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
    async ensureRerankModel() {
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
            // Model loading counts as activity - ping to keep alive
            this.touchActivity();
            return model;
        })();
        try {
            return await this.rerankModelLoadPromise;
        }
        finally {
            this.rerankModelLoadPromise = null;
        }
    }
    /**
     * Load rerank contexts (lazy). Creates multiple contexts for parallel ranking.
     * Each context has its own sequence, so they can evaluate independently.
     *
     * Tuning choices:
     * - contextSize 1024: reranking chunks are ~800 tokens max, 1024 is plenty
     * - flashAttention: ~20% less VRAM per context (568 vs 711 MB)
     * - Combined: drops from 11.6 GB (auto, no flash) to 568 MB per context (20×)
     */
    // Qwen3 reranker template adds ~200 tokens overhead (system prompt, tags, etc.)
    // Default 2048 was too small for longer documents (e.g. session transcripts,
    // CJK text, or large markdown files) — callers hit "input lengths exceed
    // context size" errors even after truncation because the overhead estimate
    // was insufficient.  4096 comfortably fits the largest real-world chunks
    // while staying well below the 40 960-token auto size.
    // Override with QMD_RERANK_CONTEXT_SIZE env var if you need more headroom.
    static RERANK_CONTEXT_SIZE = (() => {
        const v = parseInt(process.env.QMD_RERANK_CONTEXT_SIZE ?? "", 10);
        return Number.isFinite(v) && v > 0 ? v : 4096;
    })();
    static EMBED_CONTEXT_SIZE = (() => {
        const v = parseInt(process.env.QMD_EMBED_CONTEXT_SIZE ?? "", 10);
        return Number.isFinite(v) && v > 0 ? v : 2048;
    })();
    async ensureRerankContexts() {
        if (this.rerankContexts.length === 0) {
            const model = await this.ensureRerankModel();
            // ~960 MB per context with flash attention at contextSize 2048
            const n = Math.min(await this.computeParallelism(1000), 4);
            const threads = await this.threadsPerContext(n);
            for (let i = 0; i < n; i++) {
                try {
                    this.rerankContexts.push(await model.createRankingContext({
                        contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
                        flashAttention: true,
                        ...(threads > 0 ? { threads } : {}),
                    }));
                }
                catch {
                    if (this.rerankContexts.length === 0) {
                        // Flash attention might not be supported — retry without it
                        try {
                            this.rerankContexts.push(await model.createRankingContext({
                                contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
                                ...(threads > 0 ? { threads } : {}),
                            }));
                        }
                        catch {
                            throw new Error("Failed to create any rerank context");
                        }
                    }
                    break;
                }
            }
        }
        this.touchActivity();
        return this.rerankContexts;
    }
    // ==========================================================================
    // Tokenization
    // ==========================================================================
    /**
     * Tokenize text using the embedding model's tokenizer
     * Returns tokenizer tokens (opaque type from node-llama-cpp)
     */
    async tokenize(text) {
        await this.ensureEmbedContext(); // Ensure model is loaded
        if (!this.embedModel) {
            throw new Error("Embed model not loaded");
        }
        return this.embedModel.tokenize(text);
    }
    /**
     * Count tokens in text using the embedding model's tokenizer
     */
    async countTokens(text) {
        const tokens = await this.tokenize(text);
        return tokens.length;
    }
    /**
     * Detokenize token IDs back to text
     */
    async detokenize(tokens) {
        await this.ensureEmbedContext();
        if (!this.embedModel) {
            throw new Error("Embed model not loaded");
        }
        return this.embedModel.detokenize(tokens);
    }
    // ==========================================================================
    // Core API methods
    // ==========================================================================
    /**
     * Truncate text to fit within the embedding model's context window.
     * Uses the model's own tokenizer for accurate token counting, then
     * detokenizes back to text if truncation is needed.
     * Returns the (possibly truncated) text and whether truncation occurred.
     */
    async truncateToContextSize(text) {
        if (!this.embedModel)
            return { text, truncated: false };
        const maxTokens = this.embedModel.trainContextSize;
        if (maxTokens <= 0)
            return { text, truncated: false };
        const tokens = this.embedModel.tokenize(text);
        if (tokens.length <= maxTokens)
            return { text, truncated: false };
        // Leave a small margin (4 tokens) for BOS/EOS overhead
        const safeLimit = Math.max(1, maxTokens - 4);
        const truncatedTokens = tokens.slice(0, safeLimit);
        const truncatedText = this.embedModel.detokenize(truncatedTokens);
        return { text: truncatedText, truncated: true };
    }
    async embed(text, options = {}) {
        // Ping activity at start to keep models alive during this operation
        this.touchActivity();
        try {
            const context = await this.ensureEmbedContext();
            // Guard: truncate text that exceeds model context window to prevent GGML crash
            const { text: safeText, truncated } = await this.truncateToContextSize(text);
            if (truncated) {
                console.warn(`⚠ Text truncated to fit embedding context (${this.embedModel?.trainContextSize} tokens)`);
            }
            const embedding = await context.getEmbeddingFor(safeText);
            return {
                embedding: Array.from(embedding.vector),
                model: this.embedModelUri,
            };
        }
        catch (error) {
            console.error("Embedding error:", error);
            return null;
        }
    }
    /**
     * Batch embed multiple texts efficiently
     * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
     */
    async embedBatch(texts) {
        if (this._ciMode)
            throw new Error("LLM operations are disabled in CI (set CI=true)");
        // Ping activity at start to keep models alive during this operation
        this.touchActivity();
        if (texts.length === 0)
            return [];
        try {
            const contexts = await this.ensureEmbedContexts();
            const n = contexts.length;
            if (n === 1) {
                // Single context: sequential (no point splitting)
                const context = contexts[0];
                const embeddings = [];
                for (const text of texts) {
                    try {
                        const { text: safeText, truncated } = await this.truncateToContextSize(text);
                        if (truncated) {
                            console.warn(`⚠ Batch text truncated to fit embedding context (${this.embedModel?.trainContextSize} tokens)`);
                        }
                        const embedding = await context.getEmbeddingFor(safeText);
                        this.touchActivity();
                        embeddings.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
                    }
                    catch (err) {
                        console.error("Embedding error for text:", err);
                        embeddings.push(null);
                    }
                }
                return embeddings;
            }
            // Multiple contexts: split texts across contexts for parallel evaluation
            const chunkSize = Math.ceil(texts.length / n);
            const chunks = Array.from({ length: n }, (_, i) => texts.slice(i * chunkSize, (i + 1) * chunkSize));
            const chunkResults = await Promise.all(chunks.map(async (chunk, i) => {
                const ctx = contexts[i];
                const results = [];
                for (const text of chunk) {
                    try {
                        const { text: safeText, truncated } = await this.truncateToContextSize(text);
                        if (truncated) {
                            console.warn(`⚠ Batch text truncated to fit embedding context (${this.embedModel?.trainContextSize} tokens)`);
                        }
                        const embedding = await ctx.getEmbeddingFor(safeText);
                        this.touchActivity();
                        results.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
                    }
                    catch (err) {
                        console.error("Embedding error for text:", err);
                        results.push(null);
                    }
                }
                return results;
            }));
            return chunkResults.flat();
        }
        catch (error) {
            console.error("Batch embedding error:", error);
            return texts.map(() => null);
        }
    }
    async generate(prompt, options = {}) {
        if (this._ciMode)
            throw new Error("LLM operations are disabled in CI (set CI=true)");
        // Ping activity at start to keep models alive during this operation
        this.touchActivity();
        // Ensure model is loaded
        await this.ensureGenerateModel();
        // Create fresh context -> sequence -> session for each call
        const context = await this.generateModel.createContext();
        const sequence = context.getSequence();
        const session = new LlamaChatSession({ contextSequence: sequence });
        const maxTokens = options.maxTokens ?? 150;
        // Qwen3 recommends temp=0.7, topP=0.8, topK=20 for non-thinking mode
        // DO NOT use greedy decoding (temp=0) - causes repetition loops
        const temperature = options.temperature ?? 0.7;
        let result = "";
        try {
            await session.prompt(prompt, {
                maxTokens,
                temperature,
                topK: 20,
                topP: 0.8,
                onTextChunk: (text) => {
                    result += text;
                },
            });
            return {
                text: result,
                model: this.generateModelUri,
                done: true,
            };
        }
        finally {
            // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
            await context.dispose();
        }
    }
    async modelExists(modelUri) {
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
    async expandQuery(query, options = {}) {
        if (this._ciMode)
            throw new Error("LLM operations are disabled in CI (set CI=true)");
        // Ping activity at start to keep models alive during this operation
        this.touchActivity();
        const llama = await this.ensureLlama();
        await this.ensureGenerateModel();
        const includeLexical = options.includeLexical ?? true;
        const context = options.context;
        const grammar = await llama.createGrammar({
            grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
        });
        const intent = options.intent;
        const prompt = intent
            ? `/no_think Expand this search query: ${query}\nQuery intent: ${intent}`
            : `/no_think Expand this search query: ${query}`;
        // Create a bounded context for expansion to prevent large default VRAM allocations.
        const genContext = await this.generateModel.createContext({
            contextSize: this.expandContextSize,
        });
        const sequence = genContext.getSequence();
        const session = new LlamaChatSession({ contextSequence: sequence });
        try {
            // Qwen3 recommended settings for non-thinking mode:
            // temp=0.7, topP=0.8, topK=20, presence_penalty for repetition
            // DO NOT use greedy decoding (temp=0) - causes infinite loops
            const result = await session.prompt(prompt, {
                grammar,
                maxTokens: 600,
                temperature: 0.7,
                topK: 20,
                topP: 0.8,
                repeatPenalty: {
                    lastTokens: 64,
                    presencePenalty: 0.5,
                },
            });
            const lines = result.trim().split("\n");
            const queryLower = query.toLowerCase();
            const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
            const hasQueryTerm = (text) => {
                const lower = text.toLowerCase();
                if (queryTerms.length === 0)
                    return true;
                return queryTerms.some(term => lower.includes(term));
            };
            const queryables = lines.map(line => {
                const colonIdx = line.indexOf(":");
                if (colonIdx === -1)
                    return null;
                const type = line.slice(0, colonIdx).trim();
                if (type !== 'lex' && type !== 'vec' && type !== 'hyde')
                    return null;
                const text = line.slice(colonIdx + 1).trim();
                if (!hasQueryTerm(text))
                    return null;
                return { type: type, text };
            }).filter((q) => q !== null);
            // Filter out lex entries if not requested
            const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
            if (filtered.length > 0)
                return filtered;
            const fallback = [
                { type: 'hyde', text: `Information about ${query}` },
                { type: 'lex', text: query },
                { type: 'vec', text: query },
            ];
            return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
        }
        catch (error) {
            console.error("Structured query expansion failed:", error);
            // Fallback to original query
            const fallback = [{ type: 'vec', text: query }];
            if (includeLexical)
                fallback.unshift({ type: 'lex', text: query });
            return fallback;
        }
        finally {
            await genContext.dispose();
        }
    }
    // Qwen3 reranker chat template overhead (system prompt, tags, separators).
    // Measured at ~350 tokens on real queries; use 512 as a safe upper bound so
    // the truncation budget never lets a document slip past the context limit.
    static RERANK_TEMPLATE_OVERHEAD = 512;
    static RERANK_TARGET_DOCS_PER_CONTEXT = 10;
    async rerank(query, documents, options = {}) {
        if (this._ciMode)
            throw new Error("LLM operations are disabled in CI (set CI=true)");
        // Ping activity at start to keep models alive during this operation
        this.touchActivity();
        const contexts = await this.ensureRerankContexts();
        const model = await this.ensureRerankModel();
        // Truncate documents that would exceed the rerank context size.
        // Budget = contextSize - template overhead - query tokens
        const queryTokens = model.tokenize(query).length;
        const maxDocTokens = LlamaCpp.RERANK_CONTEXT_SIZE - LlamaCpp.RERANK_TEMPLATE_OVERHEAD - queryTokens;
        const truncationCache = new Map();
        const truncatedDocs = documents.map((doc) => {
            const cached = truncationCache.get(doc.text);
            if (cached !== undefined) {
                return cached === doc.text ? doc : { ...doc, text: cached };
            }
            const tokens = model.tokenize(doc.text);
            const truncatedText = tokens.length <= maxDocTokens
                ? doc.text
                : model.detokenize(tokens.slice(0, maxDocTokens));
            truncationCache.set(doc.text, truncatedText);
            if (truncatedText === doc.text)
                return doc;
            return { ...doc, text: truncatedText };
        });
        // Deduplicate identical effective texts before scoring.
        // This avoids redundant work for repeated chunks and fixes collisions where
        // multiple docs map to the same chunk text.
        const textToDocs = new Map();
        truncatedDocs.forEach((doc, index) => {
            const existing = textToDocs.get(doc.text);
            if (existing) {
                existing.push({ file: doc.file, index });
            }
            else {
                textToDocs.set(doc.text, [{ file: doc.file, index }]);
            }
        });
        // Extract just the text for ranking
        const texts = Array.from(textToDocs.keys());
        // Split documents across contexts for parallel evaluation.
        // Each context has its own sequence with a lock, so parallelism comes
        // from multiple contexts evaluating different chunks simultaneously.
        const activeContextCount = Math.max(1, Math.min(contexts.length, Math.ceil(texts.length / LlamaCpp.RERANK_TARGET_DOCS_PER_CONTEXT)));
        const activeContexts = contexts.slice(0, activeContextCount);
        const chunkSize = Math.ceil(texts.length / activeContexts.length);
        const chunks = Array.from({ length: activeContexts.length }, (_, i) => texts.slice(i * chunkSize, (i + 1) * chunkSize)).filter(chunk => chunk.length > 0);
        const allScores = await Promise.all(chunks.map((chunk, i) => activeContexts[i].rankAll(query, chunk)));
        // Reassemble scores in original order and sort
        const flatScores = allScores.flat();
        const ranked = texts
            .map((text, i) => ({ document: text, score: flatScores[i] }))
            .sort((a, b) => b.score - a.score);
        // Map back to our result format.
        const results = [];
        for (const item of ranked) {
            const docInfos = textToDocs.get(item.document) ?? [];
            for (const docInfo of docInfos) {
                results.push({
                    file: docInfo.file,
                    score: item.score,
                    index: docInfo.index,
                });
            }
        }
        return {
            results,
            model: this.rerankModelUri,
        };
    }
    /**
     * Get device/GPU info for status display.
     * Initializes llama if not already done.
     */
    async getDeviceInfo() {
        const llama = await this.ensureLlama();
        const gpuDevices = await llama.getGpuDeviceNames();
        let vram;
        if (llama.gpu) {
            try {
                const state = await llama.getVramState();
                vram = { total: state.total, used: state.used, free: state.free };
            }
            catch { /* no vram info */ }
        }
        return {
            gpu: llama.gpu,
            gpuOffloading: llama.supportsGpuOffloading,
            gpuDevices,
            vram,
            cpuCores: llama.cpuMathCores,
        };
    }
    async dispose() {
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
        // Note: llama.dispose() can hang indefinitely, so we use a timeout
        if (this.llama) {
            const disposePromise = this.llama.dispose();
            const timeoutPromise = new Promise((resolve) => setTimeout(resolve, 1000));
            await Promise.race([disposePromise, timeoutPromise]);
        }
        // Clear references
        this.embedContexts = [];
        this.rerankContexts = [];
        this.embedModel = null;
        this.generateModel = null;
        this.rerankModel = null;
        this.llama = null;
        // Clear any in-flight load/create promises
        this.embedModelLoadPromise = null;
        this.embedContextsCreatePromise = null;
        this.generateModelLoadPromise = null;
        this.rerankModelLoadPromise = null;
    }
}
// =============================================================================
// Session Management Layer
// =============================================================================
/**
 * Manages LLM session lifecycle with reference counting.
 * Coordinates with LlamaCpp idle timeout to prevent disposal during active sessions.
 */
class LLMSessionManager {
    llm;
    _activeSessionCount = 0;
    _inFlightOperations = 0;
    constructor(llm) {
        this.llm = llm;
    }
    get activeSessionCount() {
        return this._activeSessionCount;
    }
    get inFlightOperations() {
        return this._inFlightOperations;
    }
    /**
     * Returns true only when both session count and in-flight operations are 0.
     * Used by LlamaCpp to determine if idle unload is safe.
     */
    canUnload() {
        return this._activeSessionCount === 0 && this._inFlightOperations === 0;
    }
    acquire() {
        this._activeSessionCount++;
    }
    release() {
        this._activeSessionCount = Math.max(0, this._activeSessionCount - 1);
    }
    operationStart() {
        this._inFlightOperations++;
    }
    operationEnd() {
        this._inFlightOperations = Math.max(0, this._inFlightOperations - 1);
    }
    getLlamaCpp() {
        return this.llm;
    }
}
/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
    constructor(message = "LLM session has been released or aborted") {
        super(message);
        this.name = "SessionReleasedError";
    }
}
/**
 * Scoped LLM session with automatic lifecycle management.
 * Wraps LlamaCpp methods with operation tracking and abort handling.
 */
class LLMSession {
    manager;
    released = false;
    abortController;
    maxDurationTimer = null;
    name;
    constructor(manager, options = {}) {
        this.manager = manager;
        this.name = options.name || "unnamed";
        this.abortController = new AbortController();
        // Link external abort signal if provided
        if (options.signal) {
            if (options.signal.aborted) {
                this.abortController.abort(options.signal.reason);
            }
            else {
                options.signal.addEventListener("abort", () => {
                    this.abortController.abort(options.signal.reason);
                }, { once: true });
            }
        }
        // Set up max duration timer
        const maxDuration = options.maxDuration ?? 10 * 60 * 1000; // Default 10 minutes
        if (maxDuration > 0) {
            this.maxDurationTimer = setTimeout(() => {
                this.abortController.abort(new Error(`Session "${this.name}" exceeded max duration of ${maxDuration}ms`));
            }, maxDuration);
            this.maxDurationTimer.unref(); // Don't keep process alive
        }
        // Acquire session lease
        this.manager.acquire();
    }
    get isValid() {
        return !this.released && !this.abortController.signal.aborted;
    }
    get signal() {
        return this.abortController.signal;
    }
    /**
     * Release the session and decrement ref count.
     * Called automatically by withLLMSession when the callback completes.
     */
    release() {
        if (this.released)
            return;
        this.released = true;
        if (this.maxDurationTimer) {
            clearTimeout(this.maxDurationTimer);
            this.maxDurationTimer = null;
        }
        this.abortController.abort(new Error("Session released"));
        this.manager.release();
    }
    /**
     * Wrap an operation with tracking and abort checking.
     */
    async withOperation(fn) {
        if (!this.isValid) {
            throw new SessionReleasedError();
        }
        this.manager.operationStart();
        try {
            // Check abort before starting
            if (this.abortController.signal.aborted) {
                throw new SessionReleasedError(this.abortController.signal.reason?.message || "Session aborted");
            }
            return await fn();
        }
        finally {
            this.manager.operationEnd();
        }
    }
    async embed(text, options) {
        return this.withOperation(() => this.manager.getLlamaCpp().embed(text, options));
    }
    async embedBatch(texts) {
        return this.withOperation(() => this.manager.getLlamaCpp().embedBatch(texts));
    }
    async expandQuery(query, options) {
        return this.withOperation(() => this.manager.getLlamaCpp().expandQuery(query, options));
    }
    async rerank(query, documents, options) {
        return this.withOperation(() => this.manager.getLlamaCpp().rerank(query, documents, options));
    }
}
// Session manager for the default LlamaCpp instance
let defaultSessionManager = null;
/**
 * Get the session manager for the default LlamaCpp instance.
 */
function getSessionManager() {
    const llm = getDefaultLlamaCpp();
    if (!defaultSessionManager || defaultSessionManager.getLlamaCpp() !== llm) {
        defaultSessionManager = new LLMSessionManager(llm);
    }
    return defaultSessionManager;
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
export async function withLLMSession(fn, options) {
    const manager = getSessionManager();
    const session = new LLMSession(manager, options);
    try {
        return await fn(session);
    }
    finally {
        session.release();
    }
}
/**
 * Execute a function with a scoped LLM session using a specific LlamaCpp instance.
 * Unlike withLLMSession, this does not use the global singleton.
 */
export async function withLLMSessionForLlm(llm, fn, options) {
    const manager = new LLMSessionManager(llm);
    const session = new LLMSession(manager, options);
    try {
        return await fn(session);
    }
    finally {
        session.release();
    }
}
/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM() {
    if (!defaultSessionManager)
        return true;
    return defaultSessionManager.canUnload();
}
// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================
let defaultLlamaCpp = null;
/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp() {
    if (!defaultLlamaCpp) {
        const embedModel = process.env.QMD_EMBED_MODEL;
        defaultLlamaCpp = new LlamaCpp(embedModel ? { embedModel } : {});
    }
    return defaultLlamaCpp;
}
/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm) {
    defaultLlamaCpp = llm;
}
/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp() {
    if (defaultLlamaCpp) {
        await defaultLlamaCpp.dispose();
        defaultLlamaCpp = null;
    }
}
