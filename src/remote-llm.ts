/**
 * remote-llm.ts - Remote LLM provider for OpenAI-compatible embedding & reranking servers.
 *
 * Supports:
 * - POST /v1/embeddings (OpenAI-compatible)
 * - POST /v1/rerank (Cohere-compatible)
 *
 * Used with servers like omlx that serve MLX format models (e.g. bge-m3, bge-reranker).
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankOptions,
  RerankResult,
  RerankDocument,
  Queryable,
} from "./llm.js";

export type RemoteLLMConfig = {
  /** Base URL for the API (e.g. "http://localhost:8000/v1") */
  baseUrl: string;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Embedding model name (e.g. "bge-m3") */
  embedModel?: string;
  /** Reranking model name (e.g. "bge-reranker-v2-m3") */
  rerankModel?: string;
  /** Request timeout in ms for embeddings (default: 30000) */
  timeoutMs?: number;
  /** Request timeout in ms for reranking (default: 300000 = 5 min, reranking is slower) */
  rerankTimeoutMs?: number;
};

const debug = !!process.env.QMD_REMOTE_DEBUG;

export class RemoteLLM implements LLM {
  private baseUrl: string;
  private apiKey?: string;
  private embedModel: string;
  private rerankModel: string;
  private timeoutMs: number;
  private rerankTimeoutMs: number;

  readonly isRemote = true;

  constructor(config: RemoteLLMConfig) {
    // Normalize: strip trailing slash
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.embedModel = config.embedModel ?? "bge-m3";
    this.rerankModel = config.rerankModel ?? "bge-reranker-v2-m3";
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.rerankTimeoutMs = config.rerankTimeoutMs ?? 300_000;
    if (debug) {
      process.stderr.write(`[remote-llm] init baseUrl=${this.baseUrl} embed=${this.embedModel} rerank=${this.rerankModel}\n`);
    }
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  private async fetchWithTimeout(url: string, init: RequestInit, timeoutMs?: number): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs ?? this.timeoutMs);
    try {
      const resp = await fetch(url, { ...init, signal: controller.signal });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`Remote LLM error ${resp.status}: ${body}`);
      }
      return resp;
    } finally {
      clearTimeout(timeout);
    }
  }

  async embed(text: string, _options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text]);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    // Sanitize inputs to avoid remote tokenizer errors:
    // 1. Replace empty/non-string entries with a space
    // 2. Fix broken Unicode from chunk splitting (unpaired surrogates from emoji split mid-codepoint)
    const sanitized = texts.map((t, i) => {
      if (typeof t !== "string" || t.trim().length === 0) {
        if (debug) process.stderr.write(`[remote-llm] warning: empty text at index ${i}, replacing with placeholder\n`);
        return " ";
      }
      // Remove unpaired surrogates and replacement characters that crash remote tokenizers.
      // This happens when chunking splits a surrogate pair (emoji) in half.
      return t.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]|\uFFFD/g, "");
    });

    if (debug) {
      process.stderr.write(`[remote-llm] POST ${this.baseUrl}/embeddings model=${this.embedModel} texts=${sanitized.length}\n`);
    }
    const start = Date.now();

    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/embeddings`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model: this.embedModel,
          input: sanitized,
        }),
      });

      const json = await resp.json() as {
        data: { embedding: number[]; index: number }[];
        model: string;
      };

      if (debug) {
        const dim = json.data[0]?.embedding.length ?? 0;
        process.stderr.write(`[remote-llm] embed done ${Date.now() - start}ms results=${json.data.length} dim=${dim}\n`);
      }

      // Map response back to input order
      const resultMap = new Map<number, number[]>();
      for (const item of json.data) {
        resultMap.set(item.index, item.embedding);
      }

      return texts.map((_, i) => {
        const embedding = resultMap.get(i);
        if (!embedding) return null;
        return { embedding, model: json.model || this.embedModel };
      });
    } catch (error) {
      if (debug) {
        const lengths = sanitized.map((t, i) => `[${i}]:${t.length}`).join(" ");
        process.stderr.write(`[remote-llm] embed FAILED batch of ${sanitized.length}, retrying individually. lengths: ${lengths}\n`);
      }
      // Batch failed — retry each text individually to isolate bad inputs
      const results: (EmbeddingResult | null)[] = [];
      for (let i = 0; i < sanitized.length; i++) {
        try {
          const resp = await this.fetchWithTimeout(`${this.baseUrl}/embeddings`, {
            method: "POST",
            headers: this.headers(),
            body: JSON.stringify({ model: this.embedModel, input: [sanitized[i]] }),
          });
          const json = await resp.json() as { data: { embedding: number[]; index: number }[]; model: string };
          const emb = json.data[0]?.embedding;
          results.push(emb ? { embedding: emb, model: json.model || this.embedModel } : null);
        } catch (e) {
          if (debug) {
            const preview = sanitized[i]!.slice(0, 120).replace(/\n/g, "\\n");
            process.stderr.write(`[remote-llm] embed single[${i}] FAILED len=${sanitized[i]!.length} preview="${preview}": ${e}\n`);
          }
          results.push(null);
        }
      }
      return results;
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel };
    }

    const texts = documents.map((d) => d.text);

    if (debug) {
      process.stderr.write(`[remote-llm] POST ${this.baseUrl}/rerank model=${this.rerankModel} docs=${texts.length} query="${query.slice(0, 60)}"\n`);
    }
    const start = Date.now();

    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/rerank`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model: this.rerankModel,
          query,
          documents: texts,
          return_documents: false,
        }),
      }, this.rerankTimeoutMs);

      const json = await resp.json() as {
        results: { index: number; relevance_score: number }[];
      };

      if (debug) {
        const top = json.results[0];
        process.stderr.write(`[remote-llm] rerank done ${Date.now() - start}ms results=${json.results.length} top_score=${top?.relevance_score?.toFixed(4) ?? "N/A"}\n`);
      }

      const results = json.results.map((r) => ({
        file: documents[r.index]?.file ?? "",
        score: r.relevance_score,
        index: r.index,
      }));

      return { results, model: this.rerankModel };
    } catch (error) {
      console.error("Remote rerank error:", error);
      // Return all documents with score 0 as fallback
      return {
        results: documents.map((d, i) => ({ file: d.file, score: 0, index: i })),
        model: this.rerankModel,
      };
    }
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    throw new Error("RemoteLLM does not support generate(). Use local LlamaCpp for query expansion.");
  }

  async expandQuery(_query: string, _options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    throw new Error("RemoteLLM does not support expandQuery(). Use local LlamaCpp for query expansion.");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/models`, {
        method: "GET",
        headers: this.headers(),
      });
      const json = await resp.json() as { data?: { id: string }[] };
      const models = json.data ?? [];
      const exists = models.some((m) => m.id === model);
      return { name: model, exists };
    } catch {
      return { name: model, exists: false };
    }
  }

  async dispose(): Promise<void> {
    // No-op: no local resources to clean up
  }
}
