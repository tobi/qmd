/**
 * llm-remote.ts - Remote LLM backend for QMD
 *
 * Drop-in replacement for LlamaCpp that calls remote llama-server HTTP APIs
 * instead of loading models locally via node-llama-cpp.
 *
 * This enables running QMD on a low-resource machine (e.g. laptop, VPS)
 * while offloading model inference to a GPU server.
 *
 * Architecture:
 *   QMD (client)  ──HTTP──→  llama-server (GPU machine)
 *     BM25 index               embedding   (:8080)
 *     SQLite DB                 reranker    (:8081)
 *     query dispatch            generation  (:8082)
 *
 * Environment variables:
 *   QMD_REMOTE_MODE=1              Enable remote backend
 *   QMD_REMOTE_EMBED_URL           Embedding server  (default: http://localhost:8080)
 *   QMD_REMOTE_RERANK_URL          Reranker server   (default: http://localhost:8081)
 *   QMD_REMOTE_GENERATE_URL        Generation server (default: http://localhost:8082)
 *   QMD_REMOTE_API_KEY             Bearer token for all requests (optional)
 *   QMD_REMOTE_TIMEOUT_MS          Request timeout in ms (default: 30000)
 *   QMD_REMOTE_EMBED_MODEL         Model name for embeddings (default: "embed")
 *   QMD_REMOTE_GENERATE_MODEL      Model name for generation (default: "generate")
 *   QMD_REMOTE_RERANK_MODEL        Model name for reranking (default: "rerank")
 */

import {
  RerankNotSupportedError,
  type LLM,
  type EmbeddingResult,
  type EmbedOptions,
  type GenerateResult,
  type GenerateOptions,
  type Queryable,
  type QueryType,
  type RerankDocument,
  type RerankResult,
  type RerankOptions,
  type ModelInfo,
} from "./llm.js";

// =============================================================================
// Configuration (read from env at construction time, not import time)
// =============================================================================

// =============================================================================
// HTTP Client
// =============================================================================

async function httpPost<T = unknown>(url: string, body: unknown, apiKey: string, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  timer.unref?.();
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

// =============================================================================
// Response Types (llama-server OpenAI-compatible API)
// =============================================================================

interface EmbeddingResponse {
  data: Array<{ embedding: number[]; index: number }>;
  model: string;
}

interface ChatCompletionResponse {
  choices: Array<{ message: { content: string }; finish_reason: string }>;
  model?: string;
}

interface RerankResponse {
  results?: Array<{ index: number; relevance_score?: number; score?: number }>;
  data?: Array<{ index: number; relevance_score?: number; score?: number }>;
  model?: string;
}

// =============================================================================
// Remote LLM Implementation
// =============================================================================

/**
 * Remote LLM backend that delegates to llama-server HTTP APIs.
 * Implements the same LLM interface as LlamaCpp for seamless drop-in replacement.
 */
export class RemoteLlamaCpp implements LLM {
  private disposed = false;
  private embedUrl: string;
  private rerankUrl: string;
  private generateUrl: string;
  private apiKey: string;
  private timeoutMs: number;
  private embedModelName: string;
  private generateModelName: string;
  private rerankModelName: string;

  readonly embedModelUri: string;
  readonly generateModelUri: string;
  readonly rerankModelUri: string;

  constructor() {
    this.embedUrl = process.env.QMD_REMOTE_EMBED_URL || "http://localhost:8080";
    this.rerankUrl = process.env.QMD_REMOTE_RERANK_URL || "http://localhost:8081";
    this.generateUrl = process.env.QMD_REMOTE_GENERATE_URL || "http://localhost:8082";
    this.apiKey = process.env.QMD_REMOTE_API_KEY || "";
    this.timeoutMs = Number(process.env.QMD_REMOTE_TIMEOUT_MS || "30000");

    // Model names sent in API requests (override for LiteLLM or other proxies)
    this.embedModelName = process.env.QMD_REMOTE_EMBED_MODEL || "embed";
    this.generateModelName = process.env.QMD_REMOTE_GENERATE_MODEL || "generate";
    this.rerankModelName = process.env.QMD_REMOTE_RERANK_MODEL || "rerank";

    this.embedModelUri = `remote:${this.embedModelName}`;
    this.generateModelUri = `remote:${this.generateModelName}`;
    this.rerankModelUri = `remote:${this.rerankModelName}`;

    process.stderr.write(
      `[QMD] Remote LLM: embed=${this.embedUrl}, rerank=${this.rerankUrl}, generate=${this.generateUrl}` +
        `${this.apiKey ? " (auth enabled)" : ""}\n`
    );
  }

  private post<T = unknown>(url: string, body: unknown, timeoutMs?: number): Promise<T> {
    return httpPost<T>(url, body, this.apiKey, timeoutMs ?? this.timeoutMs);
  }

  // ---------------------------------------------------------------------------
  // Embedding
  // ---------------------------------------------------------------------------

  async embed(text: string, _options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const res = await this.post<EmbeddingResponse>(`${this.embedUrl}/v1/embeddings`, {
        input: text,
        model: this.embedModelName,
      });
      return { embedding: res.data[0].embedding, model: this.embedModelUri };
    } catch (e: unknown) {
      console.error("[QMD Remote] embed error:", (e as Error).message);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (!texts.length) return [];
    try {
      const res = await this.post<EmbeddingResponse>(
        `${this.embedUrl}/v1/embeddings`,
        { input: texts, model: "embed" },
        60000
      );
      // llama-server returns data[] with index field; sort by index for safety
      const sorted = [...res.data].sort((a, b) => a.index - b.index);
      return sorted.map((d) => ({
        embedding: d.embedding,
        model: this.embedModelUri,
      }));
    } catch (e: unknown) {
      console.error("[QMD Remote] embedBatch error:", (e as Error).message);
      return texts.map(() => null);
    }
  }

  // ---------------------------------------------------------------------------
  // Tokenization (byte-level fallback for chunk sizing)
  // ---------------------------------------------------------------------------

  async tokenize(text: string): Promise<readonly unknown[]> {
    return Array.from(new TextEncoder().encode(text));
  }

  async detokenize(tokens: readonly unknown[]): Promise<string> {
    return new TextDecoder().decode(Uint8Array.from(tokens as number[]));
  }

  // ---------------------------------------------------------------------------
  // Text Generation
  // ---------------------------------------------------------------------------

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    try {
      const res = await this.post<ChatCompletionResponse>(`${this.generateUrl}/v1/chat/completions`, {
        model: this.generateModelName,
        messages: [{ role: "user", content: prompt }],
        max_tokens: options?.maxTokens ?? 600,
        temperature: options?.temperature ?? 0.7,
      });
      return {
        text: res.choices[0].message.content,
        model: res.model || this.generateModelUri,
        done: true,
      };
    } catch (e: unknown) {
      console.error("[QMD Remote] generate error:", (e as Error).message);
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Query Expansion
  // ---------------------------------------------------------------------------

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;

    try {
      const prompt = `/no_think Expand this search query: ${query}`;
      const res = await this.post<ChatCompletionResponse>(
        `${this.generateUrl}/v1/chat/completions`,
        {
          model: this.generateModelName,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 600,
          temperature: 0.7,
          top_k: 20,
          top_p: 0.8,
          repeat_penalty: 1.5,
        }
      );

      const result = res.choices[0].message.content;
      const lines = result.trim().split("\n");

      const queryLower = query.toLowerCase();
      const queryTerms = queryLower
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        return queryTerms.length === 0 || queryTerms.some((t) => lower.includes(t));
      };

      const validTypes = new Set(["lex", "vec", "hyde"]);
      const queryables: Queryable[] = lines
        .map((line) => {
          const ci = line.indexOf(":");
          if (ci === -1) return null;
          const type = line.slice(0, ci).trim();
          if (!validTypes.has(type)) return null;
          const text = line.slice(ci + 1).trim();
          if (!text || !hasQueryTerm(text)) return null;
          return { type: type as QueryType, text };
        })
        .filter((q): q is Queryable => q !== null);

      const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
      if (filtered.length > 0) return filtered;
    } catch (e: unknown) {
      console.error("[QMD Remote] expandQuery error:", (e as Error).message);
    }

    return this.fallbackExpansion(query, includeLexical);
  }

  private fallbackExpansion(query: string, includeLexical: boolean): Queryable[] {
    const fb: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "vec", text: query },
    ];
    if (includeLexical) fb.unshift({ type: "lex", text: query });
    return fb;
  }

  // ---------------------------------------------------------------------------
  // Reranking
  // ---------------------------------------------------------------------------

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions
  ): Promise<RerankResult> {
    const texts = documents.map((d) => d.text);

    // Try rerank endpoints in order of preference
    const paths = ["/v1/rerank", "/rerank"];
    let lastError: unknown = null;

    for (const path of paths) {
      try {
        const res = await this.post<RerankResponse>(
          `${this.rerankUrl}${path}`,
          { model: this.rerankModelName, query, documents: texts, top_n: texts.length },
          60000
        );

        const rows = res.results || res.data || [];
        if (rows.length === 0) continue;

        const sorted = [...rows].sort(
          (a, b) => (b.relevance_score ?? b.score ?? 0) - (a.relevance_score ?? a.score ?? 0)
        );

        return {
          results: sorted.map((r) => ({
            file: documents[r.index].file,
            score: r.relevance_score ?? r.score ?? 0,
            index: r.index,
          })),
          model: this.rerankModelUri,
        };
      } catch (e: unknown) {
        lastError = e;
        const msg = (e as Error).message || "";
        // 404/405 means this path is not supported — try next
        if (msg.includes("HTTP 404") || msg.includes("HTTP 405")) continue;
        // Other errors (timeout, 500, etc.) — stop trying
        throw e;
      }
    }

    throw new RerankNotSupportedError(
      `No compatible rerank endpoint found. Last error: ${lastError}`
    );
  }

  // ---------------------------------------------------------------------------
  // Device Info
  // ---------------------------------------------------------------------------

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return {
      gpu: "remote",
      gpuOffloading: true,
      gpuDevices: [`embed: ${this.embedUrl}`, `rerank: ${this.rerankUrl}`, `generate: ${this.generateUrl}`],
      cpuCores: 0,
    };
  }

  // ---------------------------------------------------------------------------
  // Model Info & Lifecycle
  // ---------------------------------------------------------------------------

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true, path: "remote" };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
