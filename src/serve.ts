/**
 * serve.ts - QMD model server
 *
 * Runs a lightweight HTTP server that exposes embedding, reranking, and query
 * expansion via a JSON API.  Designed to be started once on a host that has
 * enough RAM/GPU for the GGUF models, so that multiple QMD clients (e.g. in
 * LXC containers) can share the same loaded models over the network.
 *
 * Supports two backends:
 *   - "local"   (default) — loads GGUF models via node-llama-cpp (CPU/Vulkan)
 *   - "rkllama" — proxies to an rkllama NPU server (RK3588/RK3576)
 *
 * Usage:
 *   qmd serve [--port 7832] [--bind 0.0.0.0]
 *   qmd serve --backend rkllama [--rkllama-url http://localhost:8080]
 *
 * Endpoints:
 *   POST /embed           { text: string, options?: EmbedOptions }              -> EmbeddingResult
 *   POST /embed-batch     { texts: string[] }                                   -> EmbeddingResult[]
 *   POST /rerank          { query: string, documents: RerankDocument[] }         -> RerankResult
 *   POST /expand          { query: string, options?: ExpandOptions }             -> Queryable[]
 *   POST /tokenize        { text: string }                                      -> { tokens: number }
 *   GET  /health          -> { ok: true, models: { embed, rerank, generate } }
 */

import { createServer, type IncomingMessage, type ServerResponse } from "http";
import {
  LlamaCpp,
  type LlamaCppConfig,
  type EmbedOptions,
  type RerankDocument,
  type EmbeddingResult,
  type RerankResult,
  type RerankDocumentResult,
  type Queryable,
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
} from "./llm.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

function json(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

interface ModelBackend {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
  rerank(query: string, documents: RerankDocument[]): Promise<RerankResult>;
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]>;
  tokenize(text: string): Promise<number>;
  health(): Promise<{ models: Record<string, string> }>;
  dispose(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Local backend (node-llama-cpp)
// ---------------------------------------------------------------------------

class LocalBackend implements ModelBackend {
  private llm: LlamaCpp;
  private config: LlamaCppConfig;

  constructor(config: LlamaCppConfig = {}) {
    this.config = config;
    this.llm = new LlamaCpp(config);
  }

  async embed(text: string, options?: EmbedOptions) {
    return this.llm.embed(text, options);
  }

  async embedBatch(texts: string[]) {
    return this.llm.embedBatch(texts);
  }

  async rerank(query: string, documents: RerankDocument[]) {
    return this.llm.rerank(query, documents);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }) {
    return this.llm.expandQuery(query, options);
  }

  async tokenize(text: string) {
    const tokens = await this.llm.tokenize(text);
    return tokens?.length ?? Math.ceil(text.length / 4);
  }

  async health() {
    return {
      models: {
        embed: this.config.embedModel ?? DEFAULT_EMBED_MODEL_URI,
        rerank: this.config.rerankModel ?? DEFAULT_RERANK_MODEL_URI,
        generate: this.config.generateModel ?? DEFAULT_GENERATE_MODEL_URI,
      },
    };
  }

  async dispose() {
    await this.llm.dispose();
  }
}

// ---------------------------------------------------------------------------
// RKLLama NPU backend
// ---------------------------------------------------------------------------

class RKLlamaBackend implements ModelBackend {
  private baseUrl: string;
  private embedModel: string;
  private rerankModel: string;
  private expandModel: string;

  constructor(options: {
    url?: string;
    embedModel?: string;
    rerankModel?: string;
    expandModel?: string;
  } = {}) {
    this.baseUrl = (options.url ?? "http://localhost:8080").replace(/\/+$/, "");
    this.embedModel = options.embedModel ?? "qwen3-embedding-0.6b";
    this.rerankModel = options.rerankModel ?? "qwen3-reranker-0.6b";
    this.expandModel = options.expandModel ?? "qmd-query-expansion";
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`rkllama ${path} returned ${res.status}: ${text}`);
    }
    return (await res.json()) as T;
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    // Format text for Qwen3-Embedding (instruction-based format)
    const formattedText = options?.isQuery
      ? `Instruct: Retrieve relevant documents for the given query\nQuery: ${text}`
      : text;

    const result = await this.post<{ embeddings: number[][] }>("/api/embed", {
      model: this.embedModel,
      input: formattedText,
    });

    if (!result.embeddings || result.embeddings.length === 0) return null;

    return {
      embedding: result.embeddings[0]!,
      model: this.embedModel,
    };
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    // Send all texts in one rkllama /api/embed call — it accepts arrays
    // and returns one embedding per input, saving HTTP round-trips.
    try {
      const result = await this.post<{ embeddings: number[][] }>("/api/embed", {
        model: this.embedModel,
        input: texts,
      });

      if (!result.embeddings) return texts.map(() => null);

      return result.embeddings.map((emb) =>
        emb ? { embedding: emb, model: this.embedModel } : null,
      );
    } catch {
      // Fallback to individual embeds if batch fails
      const results: (EmbeddingResult | null)[] = [];
      for (const text of texts) {
        results.push(await this.embed(text));
      }
      return results;
    }
  }

  async rerank(query: string, documents: RerankDocument[]): Promise<RerankResult> {
    // Use rkllama's native /api/rerank endpoint which uses logit-based
    // cross-encoder scoring (softmax over yes/no token probabilities).
    // This produces accurate relevance scores directly from the NPU.
    const docTexts = documents.map((d) => d.text);

    const result = await this.post<{
      model: string;
      results: { index: number; relevance_score: number }[];
    }>("/api/rerank", {
      model: this.rerankModel,
      query,
      documents: docTexts,
    });

    // Map rkllama's response format to QMD's RerankResult format
    const results: RerankDocumentResult[] = result.results.map((r) => ({
      file: documents[r.index]?.file ?? `doc-${r.index}`,
      score: r.relevance_score,
      index: r.index,
    }));

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return { results, model: this.rerankModel };
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string },
  ): Promise<Queryable[]> {
    // The QMD query expansion model is fine-tuned to output structured lines:
    //   lex: keyword terms for BM25
    //   vec: semantic rephrasing for vector search
    //   hyde: hypothetical document for HyDE search
    // Without grammar constraints (rkllama doesn't support GBNF), we use
    // the model's training + a clear prompt to get structured output.
    const intent = options?.intent;
    const prompt = intent
      ? `/no_think Expand this search query: ${query}\nQuery intent: ${intent}`
      : `/no_think Expand this search query: ${query}`;

    const result = await this.post<{ response: string }>("/api/generate", {
      model: this.expandModel,
      prompt,
      stream: false,
      options: { temperature: 0.7, top_k: 20, top_p: 0.8, num_predict: 600 },
    });

    // Parse the response - the fine-tuned model should output lex:/vec:/hyde: lines
    const response = result.response.trim();
    const queryables: Queryable[] = [];
    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

    // Check if a generated line relates to the original query
    const hasQueryTerm = (text: string): boolean => {
      const lower = text.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some(term => lower.includes(term));
    };

    for (const line of response.split("\n")) {
      const trimmed = line.trim();
      const colonIdx = trimmed.indexOf(":");
      if (colonIdx === -1) continue;
      const type = trimmed.slice(0, colonIdx).trim().toLowerCase();
      if (type !== "lex" && type !== "vec" && type !== "hyde") continue;
      const text = trimmed.slice(colonIdx + 1).trim();
      if (!text || !hasQueryTerm(text)) continue;
      queryables.push({ type: type as "lex" | "vec" | "hyde", text });
    }

    // If the model produced valid structured output, return it
    const includeLexical = options?.includeLexical ?? true;
    const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== "lex");
    if (filtered.length > 0) return filtered;

    // Fallback: if model produced free text instead of structured lines,
    // wrap it as a vec query and add the original as lex
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter(q => q.type !== "lex");
  }

  async tokenize(text: string): Promise<number> {
    // No tokenizer endpoint in rkllama - estimate
    return Math.ceil(text.length / 4);
  }

  async health() {
    const res = await fetch(`${this.baseUrl}/api/tags`);
    const data = (await res.json()) as { models: { name: string }[] };
    const modelNames = data.models.map((m) => m.name);
    return {
      models: {
        embed: modelNames.find((n) => n.includes("embed")) ?? this.embedModel,
        rerank: modelNames.find((n) => n.includes("rerank")) ?? this.rerankModel,
        generate: modelNames.find((n) => n.includes("expansion") || n.includes("query")) ?? this.expandModel,
      },
    };
  }

  async dispose() {
    // Nothing to dispose - we don't own the rkllama process
  }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

export interface ServeOptions {
  port?: number;
  bind?: string;
  backend?: "local" | "rkllama";
  rkllamaUrl?: string;
  config?: LlamaCppConfig;
}

export async function startServer(options: ServeOptions = {}): Promise<void> {
  const port = options.port ?? 7832;
  const bind = options.bind ?? "0.0.0.0";
  const backendType = options.backend ?? "local";

  let backend: ModelBackend;

  if (backendType === "rkllama") {
    const url = options.rkllamaUrl ?? "http://localhost:8080";
    backend = new RKLlamaBackend({ url });
    console.log(`[qmd serve] Backend: rkllama NPU (${url})`);
  } else {
    backend = new LocalBackend(options.config ?? {});
    console.log(`[qmd serve] Backend: local (node-llama-cpp)`);
    console.log(`  embed:    ${options.config?.embedModel ?? DEFAULT_EMBED_MODEL_URI}`);
    console.log(`  rerank:   ${options.config?.rerankModel ?? DEFAULT_RERANK_MODEL_URI}`);
    console.log(`  generate: ${options.config?.generateModel ?? DEFAULT_GENERATE_MODEL_URI}`);
  }

  const healthInfo = await backend.health().catch(() => ({
    models: { embed: "unknown", rerank: "unknown", generate: "unknown" },
  }));
  console.log(`  Models: ${Object.values(healthInfo.models).join(", ")}`);

  const server = createServer(async (req, res) => {
    // CORS for local network
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
    const path = url.pathname;

    try {
      // ----- Health -----------------------------------------------------------
      if (path === "/health" && req.method === "GET") {
        const info = await backend.health();
        json(res, 200, { ok: true, version: "2", backend: backendType, ...info });
        return;
      }

      // Only POST below
      if (req.method !== "POST") {
        json(res, 405, { error: "Method not allowed" });
        return;
      }

      const body = JSON.parse(await readBody(req));

      // ----- Embed ------------------------------------------------------------
      if (path === "/embed") {
        const { text, options: embedOpts } = body as {
          text: string;
          options?: EmbedOptions;
        };
        if (!text) {
          json(res, 400, { error: "text is required" });
          return;
        }
        const result = await backend.embed(text, embedOpts);
        json(res, 200, result);
        return;
      }

      // ----- Embed Batch ------------------------------------------------------
      if (path === "/embed-batch") {
        const { texts } = body as { texts: string[] };
        if (!Array.isArray(texts) || texts.length === 0) {
          json(res, 400, { error: "texts array is required" });
          return;
        }
        const results = await backend.embedBatch(texts);
        json(res, 200, results);
        return;
      }

      // ----- Rerank -----------------------------------------------------------
      if (path === "/rerank") {
        const { query, documents } = body as {
          query: string;
          documents: RerankDocument[];
        };
        if (!query || !Array.isArray(documents)) {
          json(res, 400, { error: "query and documents are required" });
          return;
        }
        const result = await backend.rerank(query, documents);
        json(res, 200, result);
        return;
      }

      // ----- Expand Query -----------------------------------------------------
      if (path === "/expand") {
        const { query, options: expandOpts } = body as {
          query: string;
          options?: { context?: string; includeLexical?: boolean; intent?: string };
        };
        if (!query) {
          json(res, 400, { error: "query is required" });
          return;
        }
        const result = await backend.expandQuery(query, expandOpts);
        json(res, 200, result);
        return;
      }

      // ----- Tokenize ---------------------------------------------------------
      if (path === "/tokenize") {
        const { text } = body as { text: string };
        if (!text) {
          json(res, 400, { error: "text is required" });
          return;
        }
        const count = await backend.tokenize(text);
        json(res, 200, { tokens: count });
        return;
      }

      json(res, 404, { error: "Not found" });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[qmd serve] Error on ${path}: ${message}`);
      json(res, 500, { error: message });
    }
  });

  // Return a Promise that stays pending until the server shuts down.
  return new Promise<void>((resolve) => {
    const shutdown = async () => {
      console.log("\n[qmd serve] Shutting down...");
      server.close();
      await backend.dispose();
      resolve();
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);

    server.listen(port, bind, () => {
      console.log(`[qmd serve] Listening on http://${bind}:${port}`);
      console.log(`[qmd serve] Endpoints: /embed, /embed-batch, /rerank, /expand, /tokenize, /health`);
    });
  });
}
