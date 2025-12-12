/**
 * llm.ts - LLM abstraction layer for QMD
 *
 * Provides a clean interface for LLM operations with an Ollama implementation.
 * All raw fetch calls to LLM APIs should go through this module.
 */

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
  relevant: boolean;
  confidence: number;
  score: number;
  rawToken: string;
  logprob: number;
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
  size?: number;
  modifiedAt?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model: string;
  maxTokens?: number;
  temperature?: number;
  logprobs?: boolean;
  raw?: boolean;
  stop?: string[];
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model: string;
  batchSize?: number;
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
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Pull a model (download if not available)
   */
  pullModel(model: string, onProgress?: (progress: number) => void): Promise<boolean>;

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  /**
   * Expand a search query into multiple variations
   */
  expandQuery(query: string, model: string, numVariations?: number): Promise<string[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores and boolean judgments
   */
  rerank(query: string, documents: RerankDocument[], options: RerankOptions): Promise<RerankResult>;

  /**
   * Quick relevance check - returns just boolean judgments with logprobs
   * More efficient than full rerank when you just need yes/no
   */
  rerankerLogprobsCheck(query: string, documents: RerankDocument[], options: RerankOptions): Promise<RerankDocumentResult[]>;
}

// =============================================================================
// Ollama Implementation
// =============================================================================

export type OllamaConfig = {
  baseUrl?: string;
  defaultEmbedModel?: string;
  defaultGenerateModel?: string;
  defaultRerankModel?: string;
};

const DEFAULT_OLLAMA_URL = "http://localhost:11434";
const DEFAULT_EMBED_MODEL = "embeddinggemma";
const DEFAULT_GENERATE_MODEL = "qwen3:0.6b";
const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";

/**
 * Format text for embedding query
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format text for embedding document
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

/**
 * Ollama LLM implementation
 */
export class Ollama implements LLM {
  private baseUrl: string;
  private defaultEmbedModel: string;
  private defaultGenerateModel: string;
  private defaultRerankModel: string;

  constructor(config: OllamaConfig = {}) {
    this.baseUrl = config.baseUrl || process.env.OLLAMA_URL || DEFAULT_OLLAMA_URL;
    this.defaultEmbedModel = config.defaultEmbedModel || DEFAULT_EMBED_MODEL;
    this.defaultGenerateModel = config.defaultGenerateModel || DEFAULT_GENERATE_MODEL;
    this.defaultRerankModel = config.defaultRerankModel || DEFAULT_RERANK_MODEL;
  }

  /**
   * Get the base URL for this Ollama instance
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions): Promise<EmbeddingResult | null> {
    const model = options.model || this.defaultEmbedModel;
    const formatted = options.isQuery
      ? formatQueryForEmbedding(text)
      : formatDocForEmbedding(text, options.title);

    try {
      const response = await fetch(`${this.baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: formatted }),
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json() as { embeddings?: number[][] };
      if (!data.embeddings?.[0]) {
        return null;
      }

      return {
        embedding: data.embeddings[0],
        model,
      };
    } catch {
      return null;
    }
  }

  async generate(prompt: string, options: GenerateOptions): Promise<GenerateResult | null> {
    const model = options.model || this.defaultGenerateModel;

    const requestBody: Record<string, unknown> = {
      model,
      prompt,
      stream: false,
      options: {
        num_predict: options.maxTokens ?? 150,
        temperature: options.temperature ?? 0,
      },
    };

    if (options.logprobs) {
      requestBody.logprobs = true;
    }

    if (options.raw) {
      requestBody.raw = true;
    }

    if (options.stop) {
      (requestBody.options as Record<string, unknown>).stop = options.stop;
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json() as {
        response?: string;
        done?: boolean;
        logprobs?: { tokens?: string[]; token_logprobs?: number[] };
      };

      // Parse logprobs if present
      let logprobs: TokenLogProb[] | undefined;
      if (data.logprobs?.tokens && data.logprobs?.token_logprobs) {
        logprobs = data.logprobs.tokens.map((token, i) => ({
          token,
          logprob: data.logprobs!.token_logprobs![i],
        }));
      }

      return {
        text: data.response || "",
        model,
        logprobs,
        done: data.done ?? true,
      };
    } catch {
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: model }),
      });

      if (!response.ok) {
        return { name: model, exists: false };
      }

      const data = await response.json() as {
        size?: number;
        modified_at?: string;
      };

      return {
        name: model,
        exists: true,
        size: data.size,
        modifiedAt: data.modified_at,
      };
    } catch {
      return { name: model, exists: false };
    }
  }

  async pullModel(model: string, onProgress?: (progress: number) => void): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: model, stream: false }),
      });

      if (!response.ok) {
        return false;
      }

      // For non-streaming, we just wait for completion
      await response.json();
      onProgress?.(100);
      return true;
    } catch {
      return false;
    }
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, model?: string, numVariations: number = 2): Promise<string[]> {
    const useModel = model || this.defaultGenerateModel;

    const prompt = `You are a search query expander. Given a search query, generate ${numVariations} alternative queries that would help find relevant documents.

Rules:
- Use synonyms and related terminology (e.g., "craft" → "craftsmanship", "quality", "excellence")
- Rephrase to capture different angles (e.g., "engineering culture" → "technical excellence", "developer practices")
- Keep proper nouns and named concepts exactly as written (e.g., "Build a Business", "Stripe", "Shopify")
- Each variation should be 3-8 words, natural search terms
- Do NOT just append words like "search" or "find" or "documents"

Query: "${query}"

Output exactly ${numVariations} variations, one per line, no numbering or bullets:`;

    const result = await this.generate(prompt, {
      model: useModel,
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

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions
  ): Promise<RerankResult> {
    const results = await this.rerankerLogprobsCheck(query, documents, options);

    return {
      results: results.sort((a, b) => b.score - a.score),
      model: options.model || this.defaultRerankModel,
    };
  }

  async rerankerLogprobsCheck(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions
  ): Promise<RerankDocumentResult[]> {
    const model = options.model || this.defaultRerankModel;
    const batchSize = options.batchSize || 5;

    const results: RerankDocumentResult[] = [];

    // Process in batches
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map((doc) => this.rerankSingle(query, doc, model))
      );
      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Rerank a single document - internal helper
   */
  private async rerankSingle(
    query: string,
    doc: RerankDocument,
    model: string
  ): Promise<RerankDocumentResult> {
    const systemPrompt = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`;

    const instruct = `Given a search query, determine if the following document is relevant to the query. Consider both direct matches and related concepts.`;

    const docTitle = doc.title || doc.file.split("/").pop()?.replace(/\.md$/, "") || doc.file;
    const docPreview = doc.text.length > 4000 ? doc.text.substring(0, 4000) + "..." : doc.text;

    // Qwen3-reranker prompt format with empty think tags
    const prompt = `<|im_start|>system
${systemPrompt}<|im_end|>
<|im_start|>user
<Instruct>: ${instruct}
<Query>: ${query}
<Document Title>: ${docTitle}
<Document>: ${docPreview}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;

    const result = await this.generate(prompt, {
      model,
      maxTokens: 1,
      temperature: 0,
      logprobs: true,
      raw: true,
    });

    if (!result) {
      return {
        file: doc.file,
        relevant: false,
        confidence: 0,
        score: 0,
        rawToken: "",
        logprob: 0,
      };
    }

    return this.parseRerankResponse(doc.file, result);
  }

  /**
   * Parse rerank response into structured result
   */
  private parseRerankResponse(file: string, result: GenerateResult): RerankDocumentResult {
    const token = result.text.toLowerCase().trim();
    const logprob = result.logprobs?.[0]?.logprob ?? 0;
    const confidence = Math.exp(logprob);

    let relevant: boolean;
    let score: number;

    if (token.startsWith("yes")) {
      relevant = true;
      // Score: 0.5 base + up to 0.5 from confidence
      score = 0.5 + 0.5 * confidence;
    } else if (token.startsWith("no")) {
      relevant = false;
      // Score: up to 0.5 based on uncertainty (1 - confidence)
      score = 0.5 * (1 - confidence);
    } else {
      // Unknown token - neutral score
      relevant = false;
      score = 0.3;
    }

    return {
      file,
      relevant,
      confidence,
      score,
      rawToken: result.logprobs?.[0]?.token ?? token,
      logprob,
    };
  }
}

// =============================================================================
// Singleton for default Ollama instance
// =============================================================================

let defaultOllama: Ollama | null = null;

/**
 * Get the default Ollama instance (creates one if needed)
 */
export function getDefaultOllama(): Ollama {
  if (!defaultOllama) {
    defaultOllama = new Ollama();
  }
  return defaultOllama;
}

/**
 * Set a custom default Ollama instance (useful for testing)
 */
export function setDefaultOllama(ollama: Ollama | null): void {
  defaultOllama = ollama;
}
