/**
 * remote-embedding.ts - Remote embedding provider support for QMD
 *
 * Supports OpenAI-compatible and Gemini embedding APIs as alternatives
 * to local GGUF models via node-llama-cpp.
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Configuration for a remote embedding provider.
 * Parsed from the `embedding` section of index.yml.
 */
export interface RemoteEmbeddingConfig {
  provider: "openai" | "gemini";
  model: string;
  api_key?: string;
  base_url?: string;
  dimensions?: number;
}

/**
 * Result from a remote embedding call.
 */
export interface RemoteEmbeddingResult {
  embedding: number[];
  model: string;
}

// =============================================================================
// API Key Resolution
// =============================================================================

/**
 * Resolve an API key from config, supporting ${ENV_VAR} syntax and env fallbacks.
 */
export function resolveApiKey(config: RemoteEmbeddingConfig): string {
  let key = config.api_key;

  // Resolve ${ENV_VAR} references
  if (key && /^\$\{(.+)\}$/.test(key)) {
    const envVar = key.match(/^\$\{(.+)\}$/)![1]!;
    key = process.env[envVar];
  }

  // Fallback to well-known env vars
  if (!key) {
    if (config.provider === "gemini") {
      key = process.env.GOOGLE_API_KEY;
    } else if (config.provider === "openai") {
      key = process.env.OPENAI_API_KEY;
    }
  }

  if (!key) {
    const envHint = config.provider === "gemini" ? "GOOGLE_API_KEY" : "OPENAI_API_KEY";
    throw new Error(
      `No API key found for ${config.provider} embedding provider. ` +
      `Set api_key in config or export ${envHint}.`
    );
  }

  return key;
}

// =============================================================================
// Gemini Provider
// =============================================================================

const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

async function geminiEmbed(
  text: string,
  model: string,
  apiKey: string,
): Promise<number[]> {
  const url = `${GEMINI_BASE_URL}/models/${model}:embedContent`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      content: { parts: [{ text }] },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini embedding API error (${response.status}): ${body}`);
  }

  const data = await response.json() as {
    embedding: { values: number[] };
  };
  return data.embedding.values;
}

async function geminiBatchEmbed(
  texts: string[],
  model: string,
  apiKey: string,
): Promise<number[][]> {
  if (texts.length === 0) return [];
  if (texts.length === 1) {
    const result = await geminiEmbed(texts[0]!, model, apiKey);
    return [result];
  }

  const url = `${GEMINI_BASE_URL}/models/${model}:batchEmbedContents`;
  const requests = texts.map(text => ({
    model: `models/${model}`,
    content: { parts: [{ text }] },
  }));

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({ requests }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini batch embedding API error (${response.status}): ${body}`);
  }

  const data = await response.json() as {
    embeddings: { values: number[] }[];
  };
  return data.embeddings.map(e => e.values);
}

// =============================================================================
// OpenAI-compatible Provider
// =============================================================================

const OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1";

async function openaiEmbed(
  input: string | string[],
  model: string,
  apiKey: string,
  baseUrl?: string,
  dimensions?: number,
): Promise<number[][]> {
  const url = `${(baseUrl || OPENAI_DEFAULT_BASE_URL).replace(/\/+$/, '')}/embeddings`;

  const body: Record<string, unknown> = { model, input };
  if (dimensions) {
    body.dimensions = dimensions;
  }

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const respBody = await response.text();
    throw new Error(`OpenAI embedding API error (${response.status}): ${respBody}`);
  }

  const data = await response.json() as {
    data: { embedding: number[]; index: number }[];
  };

  // Sort by index to maintain input order
  return data.data
    .sort((a, b) => a.index - b.index)
    .map(d => d.embedding);
}

// =============================================================================
// Remote Embedding Provider (unified interface)
// =============================================================================

/**
 * A remote embedding provider that wraps either Gemini or OpenAI-compatible APIs.
 * Implements the same embed/embedBatch pattern as LlamaCpp for drop-in use.
 */
export class RemoteEmbeddingProvider {
  private config: RemoteEmbeddingConfig;
  private apiKey: string;

  constructor(config: RemoteEmbeddingConfig) {
    this.config = config;
    this.apiKey = resolveApiKey(config);
  }

  get provider(): string {
    return this.config.provider;
  }

  get model(): string {
    return this.config.model;
  }

  get dimensions(): number | undefined {
    return this.config.dimensions;
  }

  /**
   * Get the model identifier string used for display/storage.
   */
  get modelUri(): string {
    return `${this.config.provider}:${this.config.model}`;
  }

  /**
   * Embed a single text. Returns the embedding vector.
   */
  async embed(text: string): Promise<RemoteEmbeddingResult> {
    let embedding: number[];

    if (this.config.provider === "gemini") {
      embedding = await geminiEmbed(text, this.config.model, this.apiKey);
    } else {
      const results = await openaiEmbed(text, this.config.model, this.apiKey, this.config.base_url, this.config.dimensions);
      embedding = results[0]!;
    }

    // Truncate to configured dimensions if needed
    if (this.config.dimensions && embedding.length > this.config.dimensions) {
      embedding = embedding.slice(0, this.config.dimensions);
    }

    return { embedding, model: this.modelUri };
  }

  /**
   * Embed multiple texts efficiently using batch APIs.
   * Gemini supports up to 100 texts per batch call.
   * OpenAI supports multiple inputs in a single call.
   */
  async embedBatch(texts: string[]): Promise<RemoteEmbeddingResult[]> {
    if (texts.length === 0) return [];

    let allEmbeddings: number[][];

    if (this.config.provider === "gemini") {
      // Gemini batch API supports up to 100 items
      const GEMINI_BATCH_SIZE = 100;
      allEmbeddings = [];
      for (let i = 0; i < texts.length; i += GEMINI_BATCH_SIZE) {
        const batch = texts.slice(i, i + GEMINI_BATCH_SIZE);
        const batchResults = await geminiBatchEmbed(batch, this.config.model, this.apiKey);
        allEmbeddings.push(...batchResults);
      }
    } else {
      // OpenAI: send all at once (API handles batching internally)
      // For very large batches, chunk to avoid request size limits
      const OPENAI_BATCH_SIZE = 2048;
      allEmbeddings = [];
      for (let i = 0; i < texts.length; i += OPENAI_BATCH_SIZE) {
        const batch = texts.slice(i, i + OPENAI_BATCH_SIZE);
        const batchResults = await openaiEmbed(batch, this.config.model, this.apiKey, this.config.base_url, this.config.dimensions);
        allEmbeddings.push(...batchResults);
      }
    }

    // Truncate dimensions if configured
    if (this.config.dimensions) {
      allEmbeddings = allEmbeddings.map(emb =>
        emb.length > this.config.dimensions! ? emb.slice(0, this.config.dimensions!) : emb
      );
    }

    return allEmbeddings.map(embedding => ({
      embedding,
      model: this.modelUri,
    }));
  }
}

// =============================================================================
// Factory & Config Parsing
// =============================================================================

/**
 * Parse the embedding config from a CollectionConfig.
 * Returns null if no remote embedding is configured.
 */
export function parseEmbeddingConfig(
  config: { embedding?: RemoteEmbeddingConfig }
): RemoteEmbeddingConfig | null {
  if (!config.embedding) return null;

  const { provider, model } = config.embedding;
  if (!provider || !model) return null;

  if (provider !== "openai" && provider !== "gemini") {
    throw new Error(
      `Unsupported embedding provider: "${provider}". Supported: "openai", "gemini".`
    );
  }

  return config.embedding;
}

/**
 * Create a RemoteEmbeddingProvider from config, or return null if not configured.
 */
export function createRemoteEmbeddingProvider(
  config: { embedding?: RemoteEmbeddingConfig }
): RemoteEmbeddingProvider | null {
  const embeddingConfig = parseEmbeddingConfig(config);
  if (!embeddingConfig) return null;
  return new RemoteEmbeddingProvider(embeddingConfig);
}
