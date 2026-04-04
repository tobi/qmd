/**
 * serve.ts - QMD model server
 *
 * Runs a lightweight HTTP server that exposes embedding, reranking, and query
 * expansion via a JSON API.  Designed to be started once on a host that has
 * enough RAM/GPU for the GGUF models, so that multiple QMD clients (e.g. in
 * LXC containers) can share the same loaded models over the network.
 *
 * Usage:
 *   qmd serve [--port 7832] [--bind 0.0.0.0]
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
// Server
// ---------------------------------------------------------------------------

export interface ServeOptions {
  port?: number;
  bind?: string;
  config?: LlamaCppConfig;
}

export async function startServer(options: ServeOptions = {}): Promise<void> {
  const port = options.port ?? 7832;
  const bind = options.bind ?? "0.0.0.0";

  const llm = new LlamaCpp(options.config ?? {});
  console.log(`[qmd serve] Loading models...`);
  console.log(`  embed:    ${options.config?.embedModel ?? DEFAULT_EMBED_MODEL_URI}`);
  console.log(`  rerank:   ${options.config?.rerankModel ?? DEFAULT_RERANK_MODEL_URI}`);
  console.log(`  generate: ${options.config?.generateModel ?? DEFAULT_GENERATE_MODEL_URI}`);

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
        json(res, 200, {
          ok: true,
          version: "1",
          models: {
            embed: options.config?.embedModel ?? DEFAULT_EMBED_MODEL_URI,
            rerank: options.config?.rerankModel ?? DEFAULT_RERANK_MODEL_URI,
            generate: options.config?.generateModel ?? DEFAULT_GENERATE_MODEL_URI,
          },
        });
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
        const result = await llm.embed(text, embedOpts);
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
        const results = await llm.embedBatch(texts);
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
        const result = await llm.rerank(query, documents);
        json(res, 200, result);
        return;
      }

      // ----- Expand Query -----------------------------------------------------
      if (path === "/expand") {
        const { query, options: expandOpts } = body as {
          query: string;
          options?: { context?: string; includeLexical?: boolean };
        };
        if (!query) {
          json(res, 400, { error: "query is required" });
          return;
        }
        const result = await llm.expandQuery(query, expandOpts);
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
        const tokens = await llm.tokenize(text);
        json(res, 200, { tokens: tokens?.length ?? Math.ceil(text.length / 4) });
        return;
      }

      json(res, 404, { error: "Not found" });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[qmd serve] Error on ${path}: ${message}`);
      json(res, 500, { error: message });
    }
  });

  // Graceful shutdown
  const shutdown = async () => {
    console.log("\n[qmd serve] Shutting down...");
    server.close();
    await llm.dispose();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  server.listen(port, bind, () => {
    console.log(`[qmd serve] Listening on http://${bind}:${port}`);
    console.log(`[qmd serve] Endpoints: /embed, /embed-batch, /rerank, /expand, /tokenize, /health`);
  });
}
