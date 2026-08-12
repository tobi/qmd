/**
 * remote-qmd.ts - RemoteQMD: LLM-shaped client for a remote `qmd serve`
 *
 * Connects to a `qmd serve` instance over HTTP, implementing the same LLM
 * interface as LlamaCpp but without loading any models locally. This is the
 * client tier (qmd talking to a remote qmd), distinct from a remote model
 * backend (qmd talking to a model provider).
 *
 * Usage:
 *   qmd query "search terms" --remote-url http://192.168.6.123:7832
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

export interface RemoteQMDConfig {
  /** Base URL of the qmd serve instance, e.g. "http://192.168.6.123:7832" */
  serverUrl: string;
  /** Request timeout in ms (default: 300 000 — 5 minutes, generous for CPU-only ARM SBCs) */
  timeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

export class RemoteQMD implements LLM {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private _embedModelName?: string;
  private _generateModelName?: string;
  private _rerankModelName?: string;
  private healthPromise?: Promise<void>;

  constructor(config: RemoteQMDConfig) {
    // Normalise: strip trailing slash
    this.baseUrl = config.serverUrl.replace(/\/+$/, "");
    this.timeoutMs = config.timeoutMs ?? 300_000;
    // Fire-and-forget warmup so model names are populated before they're needed.
    // Sync getters return undefined until this resolves; callers already handle
    // undefined by falling back to DEFAULT_EMBED_MODEL.
    this.healthPromise = this.warmup().catch(() => undefined);
  }

  private async warmup(): Promise<void> {
    try {
      const health = await this.get<{ models: Record<string, string> }>("/health");
      this._embedModelName = health.models?.embed;
      this._generateModelName = health.models?.generate;
      this._rerankModelName = health.models?.rerank;
    } catch {
      // Server unreachable at construction time — first real request will surface the error.
    }
  }

  get embedModelName(): string | undefined {
    return this._embedModelName;
  }

  get generateModelName(): string | undefined {
    return this._generateModelName;
  }

  get rerankModelName(): string | undefined {
    return this._rerankModelName;
  }

  /**
   * Await the in-flight /health priming so model names are available
   * synchronously. Idempotent — safe to call multiple times.
   */
  async ready(): Promise<void> {
    if (this.healthPromise) await this.healthPromise;
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

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.post<(EmbeddingResult | null)[]>("/embed-batch", { texts, options });
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
