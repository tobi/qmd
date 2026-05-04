/**
 * ollama.ts - Ollama LLM provider for QMD
 *
 * Implements the LLM interface using the Ollama REST API.
 * Use this when you want to offload LLM inference to an Ollama server
 * instead of running GGUF models locally via node-llama-cpp.
 *
 * Configuration via environment variables:
 *   QMD_OLLAMA_BASE_URL       - Ollama server URL (default: http://localhost:11434)
 *   QMD_OLLAMA_EMBED_MODEL    - Embedding model name (default: nomic-embed-text)
 *   QMD_OLLAMA_GENERATE_MODEL - Generation model name (default: qwen3:1.7b)
 *   QMD_OLLAMA_RERANK_MODEL   - Reranking model name (default: qwen3:0.6b)
 */

import type {
  LLM,
  EmbeddingResult,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankResult,
  RerankDocument,
  RerankDocumentResult,
  EmbedOptions,
  GenerateOptions,
  RerankOptions,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export type OllamaConfig = {
  /** Ollama server base URL (default: http://localhost:11434) */
  baseUrl?: string;
  /** Embedding model name (default: nomic-embed-text) */
  embedModel?: string;
  /** Generation model name (default: qwen3:1.7b) */
  generateModel?: string;
  /** Reranking model name (default: qwen3:0.6b) */
  rerankModel?: string;
};

const DEFAULT_BASE_URL = "http://localhost:11434";
const DEFAULT_EMBED_MODEL = "nomic-embed-text";
const DEFAULT_GENERATE_MODEL = "qwen3:1.7b";
const DEFAULT_RERANK_MODEL = "qwen3:0.6b";

// =============================================================================
// Ollama API types
// =============================================================================

type OllamaEmbedRequest = {
  model: string;
  input: string | string[];
};

type OllamaEmbedResponse = {
  model: string;
  embeddings: number[][];
};

type OllamaGenerateRequest = {
  model: string;
  prompt: string;
  stream?: boolean;
  options?: {
    temperature?: number;
    num_predict?: number;
    top_k?: number;
    top_p?: number;
    repeat_penalty?: number;
    presence_penalty?: number;
  };
};

type OllamaGenerateResponse = {
  model: string;
  response: string;
  done: boolean;
};

type OllamaChatRequest = {
  model: string;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  stream?: boolean;
  options?: {
    temperature?: number;
    num_predict?: number;
    top_k?: number;
    top_p?: number;
  };
};

type OllamaChatResponse = {
  model: string;
  message: { role: string; content: string };
  done: boolean;
};

type OllamaShowResponse = {
  name: string;
  details: { parent_model: string; format: string; family: string; parameter_size: string };
};

// =============================================================================
// OllamaLLM Implementation
// =============================================================================

/**
 * LLM implementation backed by an Ollama server.
 *
 * Uses the Ollama REST API for embeddings, generation, and reranking.
 * No local model downloads required — Ollama handles model management.
 */
export class OllamaLLM implements LLM {
  private readonly _baseUrl: string;
  private readonly _embedModelName: string;
  private readonly _generateModelName: string;
  private readonly _rerankModelName: string;
  private disposed = false;

  constructor(config: OllamaConfig = {}) {
    this._baseUrl =
      config.baseUrl ||
      process.env.QMD_OLLAMA_BASE_URL ||
      DEFAULT_BASE_URL;
    this._embedModelName =
      config.embedModel ||
      process.env.QMD_OLLAMA_EMBED_MODEL ||
      DEFAULT_EMBED_MODEL;
    this._generateModelName =
      config.generateModel ||
      process.env.QMD_OLLAMA_GENERATE_MODEL ||
      DEFAULT_GENERATE_MODEL;
    this._rerankModelName =
      config.rerankModel ||
      process.env.QMD_OLLAMA_RERANK_MODEL ||
      DEFAULT_RERANK_MODEL;
  }

  /** Embed model name — part of the LLM interface */
  get embedModelName(): string {
    return this._embedModelName;
  }

  /** Current embed model identifier (alias for embedModelName) */
  get embedModelId(): string {
    return this._embedModelName;
  }

  /** Current generate model identifier */
  get generateModelId(): string {
    return this._generateModelName;
  }

  /** Current rerank model identifier */
  get rerankModelId(): string {
    return this._rerankModelName;
  }

  // ===========================================================================
  // Core LLM interface
  // ===========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    this.assertNotDisposed();

    const model = options.model || this._embedModelName;
    const body: OllamaEmbedRequest = {
      model,
      input: text,
    };

    try {
      const resp = await this.fetch("/api/embed", body);
      if (!resp.ok) {
        const errText = await resp.text();
        console.error(`Ollama embed error (${resp.status}): ${errText}`);
        return null;
      }

      const data = (await resp.json()) as OllamaEmbedResponse;
      if (!data.embeddings || data.embeddings.length === 0) {
        console.error("Ollama embed: no embeddings returned");
        return null;
      }

      return {
        embedding: data.embeddings[0]!,
        model: data.model || model,
      };
    } catch (error) {
      console.error("Ollama embed error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    this.assertNotDisposed();

    const model = options.model || this._embedModelName;
    const body: OllamaEmbedRequest = {
      model,
      input: texts,
    };

    try {
      const resp = await this.fetch("/api/embed", body);
      if (!resp.ok) {
        const errText = await resp.text();
        console.error(`Ollama embedBatch error (${resp.status}): ${errText}`);
        return texts.map(() => null);
      }

      const data = (await resp.json()) as OllamaEmbedResponse;
      if (!data.embeddings || data.embeddings.length === 0) {
        console.error("Ollama embedBatch: no embeddings returned");
        return texts.map(() => null);
      }

      return data.embeddings.map((embedding, i) => ({
        embedding: embedding!,
        model: data.model || model,
      }));
    } catch (error) {
      console.error("Ollama embedBatch error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    this.assertNotDisposed();

    const model = options.model || this._generateModelName;
    const maxTokens = options.maxTokens ?? 150;
    const temperature = options.temperature ?? 0.7;

    const body: OllamaGenerateRequest = {
      model,
      prompt,
      stream: false,
      options: {
        temperature,
        num_predict: maxTokens,
        top_k: 20,
        top_p: 0.8,
        presence_penalty: 0.5,
      },
    };

    try {
      const resp = await this.fetch("/api/generate", body);
      if (!resp.ok) {
        const errText = await resp.text();
        console.error(`Ollama generate error (${resp.status}): ${errText}`);
        return null;
      }

      const data = (await resp.json()) as OllamaGenerateResponse;
      return {
        text: data.response,
        model: data.model || model,
        done: data.done,
      };
    } catch (error) {
      console.error("Ollama generate error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    this.assertNotDisposed();

    try {
      const resp = await fetch(`${this._baseUrl}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: model }),
      });

      if (resp.ok) {
        const data = (await resp.json()) as OllamaShowResponse;
        return { name: data.name || model, exists: true };
      }

      return { name: model, exists: false };
    } catch {
      // Server unreachable — assume model doesn't exist locally
      return { name: model, exists: false };
    }
  }

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean; intent?: string } = {}
  ): Promise<Queryable[]> {
    this.assertNotDisposed();

    const includeLexical = options.includeLexical ?? true;
    const intent = options.intent;

    // Build the same prompt format used by LlamaCpp.expandQuery()
    const prompt = intent
      ? `/no_think Expand this search query: ${query}\nQuery intent: ${intent}`
      : `/no_think Expand this search query: ${query}`;

    const body: OllamaGenerateRequest = {
      model: this._generateModelName,
      prompt,
      stream: false,
      options: {
        temperature: 0.7,
        num_predict: 600,
        top_k: 20,
        top_p: 0.8,
        repeat_penalty: 1.0,
        presence_penalty: 0.5,
      },
    };

    try {
      const resp = await this.fetch("/api/generate", body);
      if (!resp.ok) {
        const errText = await resp.text();
        console.error(`Ollama expandQuery error (${resp.status}): ${errText}`);
        return this.fallbackQuery(query, includeLexical);
      }

      const data = (await resp.json()) as OllamaGenerateResponse;
      const result = data.response;

      // Parse lex/vec/hyde lines from response
      const lines = result.trim().split("\n");
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        if (queryTerms.length === 0) return true;
        return queryTerms.some((term) => lower.includes(term));
      };

      const queryables: Queryable[] = lines
        .map((line): Queryable | null => {
          const colonIdx = line.indexOf(":");
          if (colonIdx === -1) return null;
          const type = line.slice(0, colonIdx).trim();
          if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
          const text = line.slice(colonIdx + 1).trim();
          if (!hasQueryTerm(text)) return null;
          return { type: type as QueryType, text };
        })
        .filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
      if (filtered.length > 0) return filtered;

      // Fallback if no valid expansions were parsed
      return this.fallbackQuery(query, includeLexical);
    } catch (error) {
      console.error("Ollama expandQuery error:", error);
      return this.fallbackQuery(query, includeLexical);
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    this.assertNotDisposed();

    const model = options.model || this._rerankModelName;

    if (documents.length === 0) {
      return { results: [], model };
    }

    // Deduplicate by text before scoring
    const textToDocs = new Map<string, { file: string; index: number }[]>();
    documents.forEach((doc, index) => {
      const existing = textToDocs.get(doc.text);
      if (existing) {
        existing.push({ file: doc.file, index });
      } else {
        textToDocs.set(doc.text, [{ file: doc.file, index }]);
      }
    });

    const uniqueTexts = Array.from(textToDocs.keys());

    // Score each document in parallel using chat-based relevance judgment.
    // Ollama doesn't have a native rerank API, so we use the chat endpoint
    // to generate yes/no relevance judgments and derive scores from response tokens.
    const scores = await Promise.all(
      uniqueTexts.map(async (docText) => {
        return this.scoreDocument(query, docText, model);
      })
    );

    // Map back to per-document results
    const ranked = uniqueTexts
      .map((text, i) => ({ document: text, score: scores[i]! }))
      .sort((a, b) => b.score - a.score);

    const results: RerankDocumentResult[] = [];
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

    return { results, model };
  }

  async dispose(): Promise<void> {
    // Nothing to dispose — Ollama manages its own resources
    this.disposed = true;
  }

  // ===========================================================================
  // Reranking helpers
  // ===========================================================================

  /**
   * Score a single document's relevance to a query using Ollama's chat API.
   *
   * Sends a structured prompt asking the model to judge relevance, then
   * parses the response to derive a numeric score. The model is instructed
   * to output only "yes" or "no" with an optional confidence score.
   */
  private async scoreDocument(
    query: string,
    documentText: string,
    model: string
  ): Promise<number> {
    // Truncate very long documents to keep prompt manageable
    const maxDocChars = 2000;
    const truncatedDoc =
      documentText.length > maxDocChars
        ? documentText.slice(0, maxDocChars) + "..."
        : documentText;

    const systemPrompt =
      `You are a relevance judge. Given a query and a document, determine if the document is relevant to the query. ` +
      `Respond with ONLY a single line in this exact format: "relevance: <score>" where <score> is a number between 0.0 (not relevant) and 1.0 (highly relevant). ` +
      `Do not include any other text, explanation, or reasoning.`;

    const userPrompt = `Query: ${query}\n\nDocument:\n${truncatedDoc}`;

    const body: OllamaChatRequest = {
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      stream: false,
      options: {
        temperature: 0.1,
        num_predict: 32,
        top_k: 5,
        top_p: 0.9,
      },
    };

    try {
      const resp = await this.fetch("/api/chat", body);
      if (!resp.ok) {
        return 0.0;
      }

      const data = (await resp.json()) as OllamaChatResponse;
      const content = data.message?.content?.trim() ?? "";

      // Parse "relevance: <score>" from response
      const match = content.match(/relevance:\s*([0-9]*\.?[0-9]+)/i);
      if (match?.[1]) {
        const score = parseFloat(match[1]);
        if (Number.isFinite(score)) {
          return Math.max(0.0, Math.min(1.0, score));
        }
      }

      // Fallback: look for any number in the response
      const numMatch = content.match(/([0-9]*\.?[0-9]+)/);
      if (numMatch?.[1]) {
        const score = parseFloat(numMatch[1]);
        if (Number.isFinite(score) && score >= 0 && score <= 1) {
          return score;
        }
      }

      // Last resort: check for yes/no keywords
      const lower = content.toLowerCase();
      if (lower.includes("yes") || lower.includes("relevant")) {
        return 0.7;
      }
      if (lower.includes("no") || lower.includes("not relevant") || lower.includes("irrelevant")) {
        return 0.1;
      }

      return 0.0;
    } catch (error) {
      console.error("Ollama rerank score error:", error);
      return 0.0;
    }
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================

  /**
   * Send a POST request to the Ollama server.
   */
  private fetch(path: string, body: unknown): Promise<Response> {
    return fetch(`${this._baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  /**
   * Fallback query expansion when the LLM call fails.
   */
  private fallbackQuery(query: string, includeLexical: boolean): Queryable[] {
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter((q) => q.type !== "lex");
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error("OllamaLLM has been disposed");
    }
  }
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create a new OllamaLLM instance with optional configuration.
 */
export function createOllamaLLM(config?: OllamaConfig): OllamaLLM {
  return new OllamaLLM(config);
}
