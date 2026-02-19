/**
 * llm.ts - LLM abstraction layer for QMD
 *
 * Provides embeddings, text generation, and reranking using local GGUF models
 * or OpenRouter-hosted models.
 */

import {
  getLlama,
  getLlamaGpuTypes,
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
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";

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
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]>;
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

/**
 * Backing inference provider
 */
export type LLMProvider = "local" | "openrouter";

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
const DEFAULT_PROVIDER: LLMProvider = "local";

const DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small";
const DEFAULT_OPENROUTER_GENERATE_MODEL = "openai/gpt-4o-mini";
const DEFAULT_OPENROUTER_RERANK_MODEL = "openai/text-embedding-3-small";
const DEFAULT_OPENROUTER_API_KEY_FILE = join(homedir(), ".config", "qmd", "openrouter.key");

// Alternative generation models for query expansion:
// LiquidAI LFM2 - hybrid architecture optimized for edge/on-device inference
// Use these as base for fine-tuning with configs/sft_lfm2.yaml
export const LFM2_GENERATE_MODEL = "hf:LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf";
export const LFM2_INSTRUCT_MODEL = "hf:LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;
export const DEFAULT_OPENROUTER_BASE_URL_URI = DEFAULT_OPENROUTER_BASE_URL;
export const DEFAULT_OPENROUTER_API_KEY_PATH = DEFAULT_OPENROUTER_API_KEY_FILE;

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;

export type PullResult = {
  model: string;
  path: string;
  sizeBytes: number;
  refreshed: boolean;
};

type HfRef = {
  repo: string;
  file: string;
};

function parseHfUri(model: string): HfRef | null {
  if (!model.startsWith("hf:")) return null;
  const without = model.slice(3);
  const parts = without.split("/");
  if (parts.length < 3) return null;
  const repo = parts.slice(0, 2).join("/");
  const file = parts.slice(2).join("/");
  return { repo, file };
}

async function getRemoteEtag(ref: HfRef): Promise<string | null> {
  const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return null;
    const etag = resp.headers.get("etag");
    return etag || null;
  } catch {
    return null;
  }
}

export async function pullModels(
  models: string[],
  options: { refresh?: boolean; cacheDir?: string } = {}
): Promise<PullResult[]> {
  const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const results: PullResult[] = [];
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
      const shouldRefresh =
        options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;

      if (shouldRefresh) {
        for (const candidate of cached) {
          if (existsSync(candidate)) unlinkSync(candidate);
        }
        if (existsSync(etagPath)) unlinkSync(etagPath);
        refreshed = cached.length > 0;
      }
    } else if (options.refresh && filename) {
      for (const candidate of cached) {
        if (existsSync(candidate)) unlinkSync(candidate);
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

function normalizeProvider(provider: string | undefined): LLMProvider {
  const value = (provider || DEFAULT_PROVIDER).trim().toLowerCase();
  if (value === "local" || value === "openrouter") {
    return value;
  }
  console.error(`Unknown QMD_LLM_PROVIDER="${provider}". Falling back to "local".`);
  return "local";
}

function stripTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

function trimSingleLine(value: string): string {
  const firstLine = value.split(/\r?\n/, 1)[0] ?? "";
  return firstLine.trim();
}

function loadOpenRouterApiKey(config: { apiKey?: string; apiKeyFile?: string } = {}): string {
  const directKey = config.apiKey?.trim();
  if (directKey) return directKey;

  const envKey = process.env.QMD_OPENROUTER_API_KEY?.trim() || process.env.OPENROUTER_API_KEY?.trim();
  if (envKey) return envKey;

  const keyFile = config.apiKeyFile || process.env.QMD_OPENROUTER_API_KEY_FILE || DEFAULT_OPENROUTER_API_KEY_FILE;
  if (existsSync(keyFile)) {
    const fileKey = trimSingleLine(readFileSync(keyFile, "utf-8"));
    if (fileKey) return fileKey;
    throw new Error(`OpenRouter API key file exists but is empty: ${keyFile}`);
  }

  throw new Error(
    `OpenRouter API key missing. Set QMD_OPENROUTER_API_KEY (or OPENROUTER_API_KEY), ` +
    `or write the key to ${keyFile}`
  );
}

function parseExpandedQueryLines(raw: string, query: string, includeLexical: boolean): Queryable[] {
  const lines = raw.trim().split("\n").map(line => line.trim()).filter(Boolean);
  const queryLower = query.toLowerCase();
  const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

  const hasQueryTerm = (text: string): boolean => {
    const lower = text.toLowerCase();
    if (queryTerms.length === 0) return true;
    return queryTerms.some(term => lower.includes(term));
  };

  const queryables: Queryable[] = lines.map(line => {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) return null;
    const type = line.slice(0, colonIdx).trim();
    if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
    const text = line.slice(colonIdx + 1).trim();
    if (!hasQueryTerm(text)) return null;
    return { type: type as QueryType, text };
  }).filter((q): q is Queryable => q !== null);

  const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== "lex");
  if (filtered.length > 0) return filtered;

  const fallback: Queryable[] = [
    { type: "hyde", text: `Information about ${query}` },
    { type: "lex", text: query },
    { type: "vec", text: query },
  ];
  return includeLexical ? fallback : fallback.filter(q => q.type !== "lex");
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

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
   * Batch embed multiple texts
   */
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;

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
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;

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
// OpenRouter Implementation
// =============================================================================

type OpenRouterEmbeddingResponse = {
  data?: Array<{
    embedding?: number[];
    index?: number;
  }>;
};

type OpenRouterChatResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ text?: string }>;
    };
  }>;
};

export type OpenRouterConfig = {
  apiKey?: string;
  apiKeyFile?: string;
  baseUrl?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  appName?: string;
  appUrl?: string;
  requestTimeoutMs?: number;
};

export class OpenRouterLLM implements LLM {
  private apiKey: string;
  private baseUrl: string;
  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private appName: string;
  private appUrl: string;
  private requestTimeoutMs: number;

  constructor(config: OpenRouterConfig = {}) {
    this.apiKey = loadOpenRouterApiKey(config);
    this.baseUrl = stripTrailingSlash(config.baseUrl || process.env.QMD_OPENROUTER_BASE_URL || DEFAULT_OPENROUTER_BASE_URL);
    this.embedModelUri = config.embedModel || process.env.QMD_OPENROUTER_EMBED_MODEL || DEFAULT_OPENROUTER_EMBED_MODEL;
    this.generateModelUri = config.generateModel || process.env.QMD_OPENROUTER_GENERATE_MODEL || DEFAULT_OPENROUTER_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || process.env.QMD_OPENROUTER_RERANK_MODEL || DEFAULT_OPENROUTER_RERANK_MODEL;
    this.appName = config.appName || process.env.QMD_OPENROUTER_APP_NAME || "qmd";
    this.appUrl = config.appUrl || process.env.QMD_OPENROUTER_APP_URL || "https://github.com/tobi/qmd";
    this.requestTimeoutMs = config.requestTimeoutMs ?? 60_000;
  }

  private async postJson<T>(path: string, payload: unknown): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.requestTimeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          "HTTP-Referer": this.appUrl,
          "X-Title": this.appName,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const raw = await response.text();
      if (!response.ok) {
        const body = raw.slice(0, 500);
        throw new Error(`OpenRouter ${path} failed (${response.status}): ${body}`);
      }

      return JSON.parse(raw) as T;
    } finally {
      clearTimeout(timeout);
    }
  }

  private static contentToString(content: string | Array<{ text?: string }> | undefined): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map(part => (typeof part?.text === "string" ? part.text : ""))
        .join("");
    }
    return "";
  }

  private async requestEmbeddings(input: string | string[], model: string): Promise<number[][]> {
    const response = await this.postJson<OpenRouterEmbeddingResponse>("/embeddings", {
      model,
      input,
      encoding_format: "float",
    });

    const rows = Array.isArray(response.data) ? [...response.data] : [];
    rows.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));

    return rows.map((row) => {
      if (!Array.isArray(row.embedding)) return [];
      return row.embedding;
    });
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const model = options.model || this.embedModelUri;
      const vectors = await this.requestEmbeddings(text, model);
      const vector = vectors[0];
      if (!vector || vector.length === 0) {
        throw new Error("OpenRouter embedding response missing embedding vector");
      }

      return { embedding: vector, model };
    } catch (error) {
      console.error("OpenRouter embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    try {
      const data = await this.requestEmbeddings(texts, this.embedModelUri);

      return texts.map((_, i) => {
        const vector = data[i];
        if (!vector || vector.length === 0) return null;
        return { embedding: vector, model: this.embedModelUri };
      });
    } catch (error) {
      console.error("OpenRouter batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    try {
      const model = options.model || this.generateModelUri;
      const response = await this.postJson<OpenRouterChatResponse>("/chat/completions", {
        model,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.maxTokens ?? 150,
        messages: [
          { role: "user", content: prompt },
        ],
      });

      const content = OpenRouterLLM.contentToString(response.choices?.[0]?.message?.content);
      if (!content) {
        throw new Error("OpenRouter completion response missing content");
      }

      return {
        text: content,
        model,
        done: true,
      };
    } catch (error) {
      console.error("OpenRouter generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async expandQuery(query: string, options: { context?: string; includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const context = options.context;
    const contextBlock = context ? `Context: ${context}\n` : "";
    const lexicalRule = includeLexical
      ? "You may use lex, vec, and hyde query types."
      : "Use only vec and hyde query types (no lex entries).";

    const prompt = [
      "Expand the search query into short retrieval variants.",
      "Output only lines in this exact format: type: text",
      "Allowed type values: lex, vec, hyde.",
      lexicalRule,
      "Keep at least one important term from the original query in each line.",
      contextBlock,
      `Original query: ${query}`,
    ].filter(Boolean).join("\n");

    try {
      const response = await this.postJson<OpenRouterChatResponse>("/chat/completions", {
        model: this.generateModelUri,
        temperature: 0.2,
        max_tokens: 300,
        messages: [{ role: "user", content: prompt }],
      });

      const content = OpenRouterLLM.contentToString(response.choices?.[0]?.message?.content);
      return parseExpandedQueryLines(content, query, includeLexical);
    } catch (error) {
      console.error("OpenRouter query expansion error:", error);
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: options.model || this.rerankModelUri };
    }

    try {
      const model = options.model || this.rerankModelUri;
      const queryVectors = await this.requestEmbeddings(query, model);
      const queryEmbedding = queryVectors[0];
      if (!queryEmbedding || queryEmbedding.length === 0) {
        throw new Error("Failed to embed rerank query");
      }

      const docEmbeddings = await this.requestEmbeddings(documents.map(doc => doc.text), model);
      const scored: RerankDocumentResult[] = documents.map((doc, index) => {
        const emb = docEmbeddings[index];
        const rawCosine = emb ? cosineSimilarity(queryEmbedding, emb) : 0;
        return {
          file: doc.file,
          index,
          score: (rawCosine + 1) / 2, // Normalize cosine [-1,1] -> [0,1]
        };
      });

      scored.sort((a, b) => b.score - a.score);
      return {
        results: scored,
        model,
      };
    } catch (error) {
      console.error("OpenRouter rerank error:", error);
      return {
        results: documents.map((doc, index) => ({ file: doc.file, index, score: 0 })),
        model: options.model || this.rerankModelUri,
      };
    }
  }

  async dispose(): Promise<void> {
    // No local resources to dispose.
  }
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
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContexts: LlamaEmbeddingContext[] = [];
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContexts: Awaited<ReturnType<LlamaModel["createRankingContext"]>>[] = [];

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  // Ensure we don't load the same model/context concurrently (which can allocate duplicate VRAM).
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
   * When timer fires, models are unloaded to free memory (if no active sessions).
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
  private hasLoadedContexts(): boolean {
    return !!(this.embedContexts.length > 0 || this.rerankContexts.length > 0);
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
      // Detect available GPU types and use the best one.
      // We can't rely on gpu:"auto" — it returns false even when CUDA is available
      // (likely a binary/build config issue in node-llama-cpp).
      // @ts-expect-error node-llama-cpp API compat
      const gpuTypes = await getLlamaGpuTypes();
      // Prefer CUDA > Metal > Vulkan > CPU
      const preferred = (["cuda", "metal", "vulkan"] as const).find(g => gpuTypes.includes(g));

      let llama: Llama;
      if (preferred) {
        try {
          llama = await getLlama({ gpu: preferred, logLevel: LlamaLogLevel.error });
        } catch {
          llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
          process.stderr.write(
            `QMD Warning: ${preferred} reported available but failed to initialize. Falling back to CPU.\n`
          );
        }
      } else {
        llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
      }

      if (!llama.gpu) {
        process.stderr.write(
          "QMD Warning: no GPU acceleration, running on CPU (slow). Run 'qmd status' for details.\n"
        );
      }
      this.llama = llama;
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
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
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
   * Compute how many parallel contexts to create.
   *
   * GPU: constrained by VRAM (25% of free, capped at 8).
   * CPU: constrained by cores. Splitting threads across contexts enables
   *      true parallelism (each context runs on its own cores). Use at most
   *      half the math cores, with at least 4 threads per context.
   */
  private async computeParallelism(perContextMB: number): Promise<number> {
    const llama = await this.ensureLlama();

    if (llama.gpu) {
      try {
        const vram = await llama.getVramState();
        const freeMB = vram.free / (1024 * 1024);
        const maxByVram = Math.floor((freeMB * 0.25) / perContextMB);
        return Math.max(1, Math.min(8, maxByVram));
      } catch {
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
  private async threadsPerContext(parallelism: number): Promise<number> {
    const llama = await this.ensureLlama();
    if (llama.gpu) return 0; // GPU: let the library decide
    const cores = llama.cpuMathCores || 4;
    return Math.max(1, Math.floor(cores / parallelism));
  }

  /**
   * Load embedding contexts (lazy). Creates multiple for parallel embedding.
   * Uses promise guard to prevent concurrent context creation race condition.
   */
  private embedContextsCreatePromise: Promise<LlamaEmbeddingContext[]> | null = null;

  private async ensureEmbedContexts(): Promise<LlamaEmbeddingContext[]> {
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
            ...(threads > 0 ? { threads } : {}),
          }));
        } catch {
          if (this.embedContexts.length === 0) throw new Error("Failed to create any embedding context");
          break;
        }
      }
      this.touchActivity();
      return this.embedContexts;
    })();

    try {
      return await this.embedContextsCreatePromise;
    } finally {
      this.embedContextsCreatePromise = null;
    }
  }

  /**
   * Get a single embed context (for single-embed calls). Uses first from pool.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    const contexts = await this.ensureEmbedContexts();
    return contexts[0]!;
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
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
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
  // Chunks are max 800 tokens, so 800 + 200 + query ≈ 1100 tokens typical.
  // Use 2048 for safety margin. Still 17× less than auto (40960).
  private static readonly RERANK_CONTEXT_SIZE = 2048;

  private async ensureRerankContexts(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>[]> {
    if (this.rerankContexts.length === 0) {
      const model = await this.ensureRerankModel();
      // ~960 MB per context with flash attention at contextSize 2048
      const n = await this.computeParallelism(1000);
      const threads = await this.threadsPerContext(n);
      for (let i = 0; i < n; i++) {
        try {
          this.rerankContexts.push(await model.createRankingContext({
            contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
            flashAttention: true,
            ...(threads > 0 ? { threads } : {}),
          } as any));
        } catch {
          if (this.rerankContexts.length === 0) {
            // Flash attention might not be supported — retry without it
            try {
              this.rerankContexts.push(await model.createRankingContext({
                contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
                ...(threads > 0 ? { threads } : {}),
              }));
            } catch {
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
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    try {
      const context = await this.ensureEmbedContext();
      const embedding = await context.getEmbeddingFor(text);

      return {
        embedding: Array.from(embedding.vector) as number[],
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
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    if (texts.length === 0) return [];

    try {
      const contexts = await this.ensureEmbedContexts();
      const n = contexts.length;

      if (n === 1) {
        // Single context: sequential (no point splitting)
        const context = contexts[0]!;
        const embeddings: ({ embedding: number[]; model: string } | null)[] = [];
        for (const text of texts) {
          try {
            const embedding = await context.getEmbeddingFor(text);
            this.touchActivity();
            embeddings.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
          } catch (err) {
            console.error("Embedding error for text:", err);
            embeddings.push(null);
          }
        }
        return embeddings;
      }

      // Multiple contexts: split texts across contexts for parallel evaluation
      const chunkSize = Math.ceil(texts.length / n);
      const chunks = Array.from({ length: n }, (_, i) =>
        texts.slice(i * chunkSize, (i + 1) * chunkSize)
      );

      const chunkResults = await Promise.all(
        chunks.map(async (chunk, i) => {
          const ctx = contexts[i]!;
          const results: (EmbeddingResult | null)[] = [];
          for (const text of chunk) {
            try {
              const embedding = await ctx.getEmbeddingFor(text);
              this.touchActivity();
              results.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
            } catch (err) {
              console.error("Embedding error for text:", err);
              results.push(null);
            }
          }
          return results;
        })
      );

      return chunkResults.flat();
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    // Ensure model is loaded
    await this.ensureGenerateModel();

    // Create fresh context -> sequence -> session for each call
    const context = await this.generateModel!.createContext();
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

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
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

    const prompt = `/no_think Expand this search query: ${query}`;

    // Create fresh context for each call
    const genContext = await this.generateModel!.createContext();
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

      return parseExpandedQueryLines(result, query, includeLexical);
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    } finally {
      await genContext.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    const contexts = await this.ensureRerankContexts();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Split documents across contexts for parallel evaluation.
    // Each context has its own sequence with a lock, so parallelism comes
    // from multiple contexts evaluating different chunks simultaneously.
    const n = contexts.length;
    const chunkSize = Math.ceil(texts.length / n);
    const chunks = Array.from({ length: n }, (_, i) =>
      texts.slice(i * chunkSize, (i + 1) * chunkSize)
    ).filter(chunk => chunk.length > 0);

    const allScores = await Promise.all(
      chunks.map((chunk, i) => contexts[i]!.rankAll(query, chunk))
    );

    // Reassemble scores in original order and sort
    const flatScores = allScores.flat();
    const ranked = texts
      .map((text, i) => ({ document: text, score: flatScores[i]! }))
      .sort((a, b) => b.score - a.score);

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

  /**
   * Get device/GPU info for status display.
   * Initializes llama if not already done.
   */
  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    const llama = await this.ensureLlama();
    const gpuDevices = await llama.getGpuDeviceNames();
    let vram: { total: number; used: number; free: number } | undefined;
    if (llama.gpu) {
      try {
        const state = await llama.getVramState();
        vram = { total: state.total, used: state.used, free: state.free };
      } catch { /* no vram info */ }
    }
    return {
      gpu: llama.gpu,
      gpuOffloading: llama.supportsGpuOffloading,
      gpuDevices,
      vram,
      cpuCores: llama.cpuMathCores,
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
    // Note: llama.dispose() can hang indefinitely, so we use a timeout
    if (this.llama) {
      const disposePromise = this.llama.dispose();
      const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 1000));
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
  private llm: LLM;
  private _activeSessionCount = 0;
  private _inFlightOperations = 0;

  constructor(llm: LLM) {
    this.llm = llm;
  }

  get activeSessionCount(): number {
    return this._activeSessionCount;
  }

  get inFlightOperations(): number {
    return this._inFlightOperations;
  }

  /**
   * Returns true only when both session count and in-flight operations are 0.
   * Used by LlamaCpp to determine if idle unload is safe.
   */
  canUnload(): boolean {
    return this._activeSessionCount === 0 && this._inFlightOperations === 0;
  }

  acquire(): void {
    this._activeSessionCount++;
  }

  release(): void {
    this._activeSessionCount = Math.max(0, this._activeSessionCount - 1);
  }

  operationStart(): void {
    this._inFlightOperations++;
  }

  operationEnd(): void {
    this._inFlightOperations = Math.max(0, this._inFlightOperations - 1);
  }

  getLLM(): LLM {
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
class LLMSession implements ILLMSession {
  private manager: LLMSessionManager;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private name: string;

  constructor(manager: LLMSessionManager, options: LLMSessionOptions = {}) {
    this.manager = manager;
    this.name = options.name || "unnamed";
    this.abortController = new AbortController();

    // Link external abort signal if provided
    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
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

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  /**
   * Release the session and decrement ref count.
   * Called automatically by withLLMSession when the callback completes.
   */
  release(): void {
    if (this.released) return;
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
  private async withOperation<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.isValid) {
      throw new SessionReleasedError();
    }

    this.manager.operationStart();
    try {
      // Check abort before starting
      if (this.abortController.signal.aborted) {
        throw new SessionReleasedError(
          this.abortController.signal.reason?.message || "Session aborted"
        );
      }
      return await fn();
    } finally {
      this.manager.operationEnd();
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.withOperation(() => this.manager.getLLM().embed(text, options));
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(() => this.manager.getLLM().embedBatch(texts));
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.withOperation(() => this.manager.getLLM().expandQuery(query, options));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.withOperation(() => this.manager.getLLM().rerank(query, documents, options));
  }
}

// Session manager for the default LLM instance
let defaultSessionManager: LLMSessionManager | null = null;
let defaultLlamaCpp: LlamaCpp | null = null;
let defaultOpenRouterLLM: OpenRouterLLM | null = null;
let defaultLLM: LLM | null = null;
let defaultLLMProvider: LLMProvider | null = null;
let didWarnOpenRouterRemote = false;

/**
 * Emit the remote-provider notice once per process.
 */
function warnOpenRouterOnce(): void {
  if (didWarnOpenRouterRemote) return;
  didWarnOpenRouterRemote = true;
  process.stderr.write(
    "Notice: QMD is using OpenRouter (remote inference over HTTPS) for model operations.\n"
  );
}

/**
 * Resolve the default provider from environment.
 * Defaults to local so remote inference is always opt-in.
 */
export function getDefaultLLMProvider(): LLMProvider {
  return defaultLLMProvider ?? normalizeProvider(process.env.QMD_LLM_PROVIDER);
}

function getDefaultOpenRouterLLM(): OpenRouterLLM {
  if (!defaultOpenRouterLLM) {
    defaultOpenRouterLLM = new OpenRouterLLM();
  }
  return defaultOpenRouterLLM;
}

/**
 * Get the default LLM instance (local or OpenRouter based on QMD_LLM_PROVIDER).
 */
export function getDefaultLLM(): LLM {
  const provider = normalizeProvider(process.env.QMD_LLM_PROVIDER);
  if (defaultLLM && defaultLLMProvider === provider) {
    return defaultLLM;
  }

  if (defaultLLM && defaultLLMProvider !== provider) {
    defaultSessionManager = null;
    void defaultLLM.dispose().catch(() => {});
    defaultLLM = null;
  }

  if (provider === "openrouter") {
    warnOpenRouterOnce();
    defaultLLM = getDefaultOpenRouterLLM();
    defaultLLMProvider = "openrouter";
    return defaultLLM;
  }

  defaultLLM = getDefaultLlamaCpp();
  defaultLLMProvider = "local";
  return defaultLLM;
}

/**
 * Get the session manager for the default LLM instance.
 */
function getSessionManager(): LLMSessionManager {
  const llm = getDefaultLLM();
  if (!defaultSessionManager || defaultSessionManager.getLLM() !== llm) {
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
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const manager = getSessionManager();
  const session = new LLMSession(manager, options);

  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM(): boolean {
  if (!defaultSessionManager) return true;
  return defaultSessionManager.canUnload();
}

// =============================================================================
// Singleton accessors
// =============================================================================

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
  if (defaultLLMProvider === "local") {
    defaultLLM = llm;
    defaultSessionManager = null;
  }
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    const existing = defaultLlamaCpp;
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
    if (defaultLLM === existing) {
      defaultLLM = null;
      defaultLLMProvider = null;
      defaultSessionManager = null;
    }
  }
}

/**
 * Dispose the active default LLM instance (provider-aware).
 */
export async function disposeDefaultLLM(): Promise<void> {
  const disposed = new Set<LLM>();
  const disposeOne = async (llm: LLM | null): Promise<void> => {
    if (!llm || disposed.has(llm)) return;
    disposed.add(llm);
    await llm.dispose();
  };

  await disposeOne(defaultLLM);
  await disposeOne(defaultLlamaCpp);
  await disposeOne(defaultOpenRouterLLM);

  defaultLLM = null;
  defaultLLMProvider = null;
  defaultLlamaCpp = null;
  defaultOpenRouterLLM = null;
  defaultSessionManager = null;
}

/**
 * Test helper: clears default singleton state without disposing native resources.
 */
export function resetDefaultLLMForTests(): void {
  defaultLLM = null;
  defaultLLMProvider = null;
  defaultLlamaCpp = null;
  defaultOpenRouterLLM = null;
  defaultSessionManager = null;
  didWarnOpenRouterRemote = false;
}
