/**
 * llm-remote.ts - Remote LLM implementation for QMD
 *
 * Connects to a `qmd serve` instance over HTTP, implementing the same LLM
 * interface as LlamaCpp but without loading any models locally.
 *
 * Usage:
 *   qmd query "search terms" --server http://192.168.6.123:7832
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface RemoteLLMConfig {
  /** Base URL of the qmd serve instance, e.g. "http://192.168.6.123:7832" */
  serverUrl: string;
  /** Request timeout in ms (default: 120 000) */
  timeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

export class RemoteLLM implements LLM {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(config: RemoteLLMConfig) {
    // Normalise: strip trailing slash
    this.baseUrl = config.serverUrl.replace(/\/+$/, "");
    this.timeoutMs = config.timeoutMs ?? 120_000;
  }

  // ---- helpers ----------------------------------------------------------

  private async post<T>(path: string, body: unknown): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`qmd-server ${path} returned ${res.status}: ${text}`);
      }

      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  private async get<T>(path: string): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`qmd-server ${path} returned ${res.status}`);
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  // ---- LLM interface ----------------------------------------------------

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.post<EmbeddingResult | null>("/embed", { text, options });
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.post<(EmbeddingResult | null)[]>("/embed-batch", { texts });
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    // Generation is not exposed via serve (only used internally for query expansion)
    // expandQuery handles this end-to-end
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const health = await this.get<{ ok: boolean; models: Record<string, string> }>("/health");
      const loaded = Object.values(health.models);
      return {
        name: model,
        exists: loaded.some((m) => m.includes(model) || model.includes(m)),
      };
    } catch {
      return { name: model, exists: false };
    }
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    return this.post<Queryable[]>("/expand", { query, options });
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions,
  ): Promise<RerankResult> {
    return this.post<RerankResult>("/rerank", { query, documents });
  }

  /**
   * Tokenize remotely - falls back to char-based estimate on failure.
   */
  async tokenize(text: string): Promise<number[]> {
    try {
      const result = await this.post<{ tokens: number }>("/tokenize", { text });
      // Return a dummy token array of the right length (actual IDs don't matter for chunking)
      return new Array(result.tokens).fill(0);
    } catch {
      // Fallback: ~4 chars per token
      return new Array(Math.ceil(text.length / 4)).fill(0);
    }
  }

  async dispose(): Promise<void> {
    // Nothing to dispose - we don't own the models
  }
}
