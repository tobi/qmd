/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Speaks MCP spec 2026-07-28 (stateless, no initialize handshake) and dual-speaks
 * 2025-era clients via the official SDK entries (`serveStdio` / `createMcpHandler`).
 */

import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "url";
import {
  createMcpHandler,
  isLegacyRequest,
  McpServer,
  ResourceTemplate,
  WebStandardStreamableHTTPServerTransport,
} from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import { existsSync } from "fs";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  getDefaultDbPath,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "../index.js";
import {
  API_VERSION,
  DAEMON_FEATURES,
  QMDWorkService,
  WorkServiceError,
  type CollectionEnsureRequest,
  type UpdateScope,
  type DaemonEmbedRequest,
  type DaemonSearchRequest,
  type DaemonSearchResult,
} from "./work-service.js";
import { getConfigPath } from "../collections.js";
import { enableProductionMode } from "../store.js";
import { checkRequestOrigin, resolveOriginGuard } from "./origin-guard.js";

const MAX_HTTP_REQUEST_BODY_BYTES = 1_048_576;
const HTTP_REQUEST_BODY_TIMEOUT_MS = 30_000;

// =============================================================================
// Types for structured content
// =============================================================================

type SearchResultItem = {
  docid: string; // Short docid (#abc123) for quick reference
  file: string;
  title: string;
  score: number;
  context: string | null;
  line: number; // Absolute line in source markdown
  snippet: string;
};

type StatusResult = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: {
    name: string;
    path: string | null;
    pattern: string | null;
    documents: number;
    lastUpdated: string;
  }[];
};

// =============================================================================
// Helper functions
// =============================================================================

/**
 * Encode a path for use in qmd:// URIs.
 * Encodes special characters but preserves forward slashes for readability.
 */
function encodeQmdPath(path: string): string {
  // Encode each path segment separately to preserve slashes
  return path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function formatSearchResults(
  results: readonly DaemonSearchResult[],
  query: string,
  intent: string | undefined,
  uriPaths: boolean,
): SearchResultItem[] {
  return results.map((result) => {
    const bestChunkPos =
      "bestChunkPos" in result ? result.bestChunkPos : result.chunkPos;
    const bestChunkLength =
      "bestChunk" in result ? result.bestChunk.length : undefined;
    const { line, snippet } = extractSnippet(
      result.body ?? "",
      query,
      300,
      bestChunkPos,
      bestChunkLength,
      intent,
    );
    return {
      docid: `#${result.docid}`,
      file: uriPaths
        ? `qmd://${encodeQmdPath(result.displayPath)}`
        : result.displayPath,
      title: result.title,
      score: Math.round(result.score * 100) / 100,
      context: result.context,
      line,
      snippet: addLineNumbers(snippet, line),
    };
  });
}

function primarySearchQuery(
  request: Pick<DaemonSearchRequest, "query" | "searches">,
): string {
  return (
    request.query ||
    request.searches?.find((search) => search.type === "lex")?.query ||
    request.searches?.find((search) => search.type === "vec")?.query ||
    request.searches?.[0]?.query ||
    ""
  );
}

/**
 * Format search results as human-readable text summary
 */
function formatSearchSummary(
  results: SearchResultItem[],
  query: string,
): string {
  if (results.length === 0) {
    return `No results found for "${query}"`;
  }
  const lines = [
    `Found ${results.length} result${results.length === 1 ? "" : "s"} for "${query}":\n`,
  ];
  for (const r of results) {
    lines.push(
      `${r.docid} ${Math.round(r.score * 100)}% ${r.file} - ${r.title}`,
    );
  }
  return lines.join("\n");
}

function getPackageVersion(): string {
  try {
    const pkgPath = join(
      dirname(fileURLToPath(import.meta.url)),
      "../../package.json",
    );
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    return pkg.version ?? "unknown";
  } catch {
    return "unknown";
  }
}

// =============================================================================
// MCP Server
// =============================================================================

/**
 * Build dynamic server instructions from actual index state.
 * Injected into the LLM's system prompt via MCP initialize (2025-era) and
 * server/discover (2026-07-28) — gives the LLM immediate context about what's
 * searchable without a tool call.
 */
async function buildInstructions(service: QMDWorkService): Promise<string> {
  let status: Awaited<ReturnType<QMDWorkService["getStatus"]>>;
  let globalCtx: string | undefined;
  try {
    status = await service.getStatus();
    globalCtx = await service.getGlobalContext();
  } catch (error) {
    if (error instanceof WorkServiceError) {
      return "QMD is temporarily unavailable while maintenance is running.";
    }
    throw error;
  }
  const lines: string[] = [];

  // --- What is this? ---
  lines.push(
    `QMD is your local search engine over ${status.totalDocuments} markdown documents.`,
  );
  if (globalCtx) lines.push(`Context: ${globalCtx}`);

  // --- What's searchable? ---
  // Emit names only — the per-collection doc counts and descriptions can run to ~1.5 KB
  // across a dozen collections, and the same info is available on demand via the `status` tool.
  if (status.collections.length > 0) {
    lines.push("");
    const names = status.collections.map((c) => c.name).join(", ");
    lines.push(`Collections (scope with \`collections\` parameter): ${names}`);
    lines.push(
      "Call the `status` tool for collection descriptions, paths, and per-collection doc counts.",
    );
  }

  // --- Capability gaps ---
  if (!status.hasVectorIndex) {
    lines.push("");
    lines.push(
      "Note: Vector embeddings are not ready; semantic searches use lexical fallback.",
    );
  } else if (status.needsEmbedding > 0) {
    lines.push("");
    lines.push(
      `Note: ${status.needsEmbedding} documents await embedding, so semantic coverage may be incomplete.`,
    );
  }

  // --- Search tool ---
  lines.push("");
  lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):");
  lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)");
  lines.push("  - type:'vec' — semantic vector search (meaning-based)");
  lines.push(
    "  - type:'hyde' — hypothetical document (write what the answer looks like)",
  );
  lines.push("");
  lines.push(
    "  Always provide `intent` on every search call to disambiguate and improve snippets.",
  );
  lines.push("");
  lines.push("Examples:");
  lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]");
  lines.push(
    "  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]",
  );
  lines.push(
    "  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]",
  );
  lines.push(
    "  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'",
  );

  // --- Retrieval workflow ---
  lines.push("");
  lines.push("Retrieval:");
  lines.push(
    "  - `get` — single document by path or docid (#abc123). Supports a line-range suffix: `file.md:100` (from line 100) or `file.md:100:40` (40 lines from line 100).",
  );
  lines.push(
    "  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`), comma-separated list, or docids (#abc123).",
  );

  // --- Non-obvious things that prevent mistakes ---
  lines.push("");
  lines.push("Tips:");
  lines.push("  - File paths in results are relative to their collection.");
  lines.push("  - Use `minScore: 0.5` to filter low-confidence results.");
  lines.push(
    "  - Results include a `context` field describing the content type.",
  );

  return lines.join("\n");
}

/**
 * Create an MCP server with all QMD tools, resources, and prompts registered.
 * Shared by both stdio and HTTP transports.
 */
async function createMcpServer(
  service: QMDWorkService,
  inflight?: InflightGate,
): Promise<McpServer> {
  // Wraps request handlers so a stdio EOF shutdown can wait for in-flight
  // work to settle before disposing the store/llm underneath it.
  const track = inflight?.track ?? (<T>(fn: T): T => fn);
  const server = new McpServer(
    { name: "qmd", version: getPackageVersion() },
    {
      instructions: await buildInstructions(service),
      // tools/list is static for the process lifetime; resources/read stays
      // uncacheable because the index can change under us.
      cacheHints: {
        "tools/list": { ttlMs: 60_000, cacheScope: "private" },
        "server/discover": { ttlMs: 60_000, cacheScope: "private" },
        "resources/read": { ttlMs: 0, cacheScope: "private" },
      },
    },
  );

  // ---------------------------------------------------------------------------
  // Resource: qmd://{path} - read-only access to documents by path
  // Note: No list() - documents are discovered via search tools
  // ---------------------------------------------------------------------------

  server.registerResource(
    "document",
    new ResourceTemplate("qmd://{+path}", { list: undefined }),
    {
      title: "QMD Document",
      description:
        "A markdown document from your QMD knowledge base. Use search tools to discover documents.",
      mimeType: "text/markdown",
    },
    track(async (uri, { path }) => {
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const pathStr = Array.isArray(path) ? path.join("/") : path || "";
      const decodedPath = decodeURIComponent(pathStr);

      // Use SDK to find document — findDocument handles collection/path resolution
      const result = await service.get(decodedPath, { includeBody: true });

      if ("error" in result) {
        return { contents: [{ uri: uri.href, text: "Document unavailable" }] };
      }

      let text = addLineNumbers(result.body || ""); // Default to line numbers
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        contents: [
          {
            uri: uri.href,
            name: result.displayPath,
            title: result.title || result.displayPath,
            mimeType: "text/markdown",
            text,
          },
        ],
      };
    }),
  );

  // ---------------------------------------------------------------------------
  // Tool: query (Primary search tool)
  // ---------------------------------------------------------------------------

  const subSearchSchema = z.object({
    type: z
      .enum(["lex", "vec", "hyde"])
      .describe(
        'lex = BM25 keywords (supports "phrase" and -negation); ' +
          "vec = semantic question; hyde = hypothetical answer passage",
      ),
    query: z
      .string()
      .describe(
        'The query text. For lex: use keywords, "quoted phrases", and -negation. ' +
          "For vec: natural language question. For hyde: 50-100 word answer passage.",
      ),
  });

  server.registerTool(
    "query",
    {
      title: "Query",
      description: `Search the knowledge base using a query document — one or more typed sub-queries combined for best recall.

Each result includes a \`line\` field with the absolute 1-indexed line of the best match in the source markdown. To read more context around a hit, call \`get(file, fromLine = max(1, line - 20), maxLines = 80, lineNumbers = true)\`.

## Query Types

**lex** — BM25 keyword search. Fast, exact, no LLM needed.
Full lex syntax:
- \`term\` — prefix match ("perf" matches "performance")
- \`"exact phrase"\` — phrase must appear verbatim
- \`-term\` or \`-"phrase"\` — exclude documents containing this

Good lex examples:
- \`"connection pool" timeout -redis\`
- \`"machine learning" -sports -athlete\`
- \`handleError async typescript\`

**vec** — Semantic vector search. Write a natural language question. Finds documents by meaning, not exact words.
- \`how does the rate limiter handle burst traffic?\`
- \`what is the tradeoff between consistency and availability?\`

**hyde** — Hypothetical document. Write 50-100 words that look like the answer. Often the most powerful for nuanced topics.
- \`The rate limiter uses a token bucket algorithm. When a client exceeds 100 req/min, subsequent requests return 429 until the window resets.\`

## Strategy

Combine types for best results. First sub-query gets 2× weight — put your strongest signal first.

| Goal | Approach |
|------|----------|
| General search (recommended) | Pass \`query\` — auto-expanded into typed variants, fused, reranked |
| Know exact term/name | \`lex\` only |
| Concept search | \`vec\` only |
| Best recall | \`lex\` + \`vec\` |
| Complex/nuanced | \`lex\` + \`vec\` + \`hyde\` |
| Unknown vocabulary | Pass \`query\` with natural language so the server auto-expands it |

## Examples

Simple lookup:
\`\`\`json
[{ "type": "lex", "query": "CAP theorem" }]
\`\`\`

Best recall on a technical topic:
\`\`\`json
[
  { "type": "lex", "query": "\\"connection pool\\" timeout -redis" },
  { "type": "vec", "query": "why do database connections time out under load" },
  { "type": "hyde", "query": "Connection pool exhaustion occurs when all connections are in use and new requests must wait. This typically happens under high concurrency when queries run longer than expected." }
]
\`\`\`

Intent-aware lex (C++ performance, not sports):
\`\`\`json
[
  { "type": "lex", "query": "\\"C++ performance\\" optimization -sports -athlete" },
  { "type": "vec", "query": "how to optimize C++ program performance" }
]
\`\`\``,
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        query: z
          .string()
          .optional()
          .describe(
            "Plain-text query, auto-expanded by the SDK into lex/vec/hyde variants, fused via " +
              "RRF and reranked. Recommended default for most searches. Mutually exclusive with 'searches'.",
          ),
        searches: z
          .array(subSearchSchema)
          .max(10)
          .optional()
          .describe(
            "Typed sub-queries to execute (lex/vec/hyde). First gets 2x weight. Use for precise " +
              "control over retrieval strategy. Mutually exclusive with 'query'.",
          ),
        limit: z
          .number()
          .optional()
          .default(10)
          .describe("Max results (default: 10)"),
        minScore: z
          .number()
          .optional()
          .default(0)
          .describe("Min relevance 0-1 (default: 0)"),
        candidateLimit: z
          .number()
          .optional()
          .describe(
            "Maximum candidates to rerank (default: 40, lower = faster but may miss results)",
          ),
        collections: z
          .array(z.string())
          .optional()
          .describe("Filter to collections (OR match)"),
        intent: z
          .string()
          .optional()
          .describe(
            "Background context to disambiguate the query. Example: query='performance', intent='web page load times and Core Web Vitals'. Does not search on its own.",
          ),
        rerank: z
          .boolean()
          .optional()
          .default(true)
          .describe(
            "Rerank results using LLM (default: true). Set to false for faster results on CPU-only machines.",
          ),
      }),
    },
    track(
      async (
        {
          query,
          searches,
          limit,
          minScore,
          candidateLimit,
          collections,
          intent,
          rerank,
        },
        ctx,
      ) => {
        // Require exactly one of `query` (plain text, auto-expanded) or `searches` (typed sub-queries).
        if (!query && (!searches || searches.length === 0)) {
          return {
            content: [
              {
                type: "text" as const,
                text: "Error: provide either 'query' (plain text) or 'searches' (typed sub-queries)",
              },
            ],
            isError: true,
          };
        }
        if (query && searches && searches.length > 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "Error: 'query' and 'searches' are mutually exclusive; provide only one",
              },
            ],
            isError: true,
          };
        }

        const outcome = await service.search(
          {
            ...(query
              ? { query }
              : { searches: (searches ?? []).map((s) => ({ ...s })) }),
            ...(collections !== undefined ? { collections } : {}),
            limit,
            minScore,
            candidateLimit,
            rerank,
            intent,
          },
          ctx.mcpReq.signal,
        );
        if (outcome.status === "unavailable") {
          return {
            content: [
              {
                type: "text" as const,
                text: `Search unavailable (${outcome.reason})`,
              },
            ],
            isError: true,
          };
        }

        const primaryQuery = primarySearchQuery({ query, searches });
        const filtered = formatSearchResults(
          outcome.results,
          primaryQuery,
          intent,
          false,
        );
        return {
          content: [
            { type: "text", text: formatSearchSummary(filtered, primaryQuery) },
          ],
          structuredContent: { results: filtered },
        };
      },
    ),
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_get (Retrieve document)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "get",
    {
      title: "Get Document",
      description:
        "Retrieve the full content of a document by its file path or docid. Use paths or docids (#abc123) from search results. Suggests similar files if not found.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        file: z
          .string()
          .describe(
            "File path or docid from search results. Supports a line-range suffix: 'pages/meeting.md:100' starts at line 100; 'pages/meeting.md:100:40' (or '#abc123:100:40') reads 40 lines from line 100.",
          ),
        fromLine: z
          .number()
          .optional()
          .describe("Start from this line number (1-indexed)"),
        maxLines: z
          .number()
          .optional()
          .describe("Maximum number of lines to return"),
        lineNumbers: z
          .boolean()
          .optional()
          .default(true)
          .describe(
            "Add line numbers to output (format: 'N: content'). On by default; set false for raw content.",
          ),
      }),
    },
    track(async ({ file, fromLine, maxLines, lineNumbers }) => {
      // Support :line and :from:count suffixes in `file` (e.g. "foo.md:120" or
      // "foo.md:120:40"). Explicit fromLine/maxLines args take precedence.
      let parsedFromLine = fromLine;
      let parsedMaxLines = maxLines;
      let lookup = file;
      const rangeMatch = lookup.match(/:(\d+):(\d+)$/);
      if (rangeMatch) {
        if (parsedFromLine === undefined)
          parsedFromLine = parseInt(rangeMatch[1]!, 10);
        if (parsedMaxLines === undefined)
          parsedMaxLines = parseInt(rangeMatch[2]!, 10);
        lookup = lookup.slice(0, -rangeMatch[0].length);
      } else {
        const colonMatch = lookup.match(/:(\d+)$/);
        if (colonMatch && colonMatch[1] && parsedFromLine === undefined) {
          parsedFromLine = parseInt(colonMatch[1], 10);
          lookup = lookup.slice(0, -colonMatch[0].length);
        }
      }
      if (parsedFromLine !== undefined)
        parsedFromLine = Math.max(1, parsedFromLine);

      const result = await service.get(lookup, { includeBody: false });

      if ("error" in result) {
        return {
          content: [{ type: "text", text: "Document unavailable" }],
          isError: true,
        };
      }

      const body =
        (await service.getDocumentBody(result.filepath, {
          fromLine: parsedFromLine,
          maxLines: parsedMaxLines,
        })) ?? "";
      let text = body;
      if (lineNumbers) {
        const startLine = parsedFromLine || 1;
        text = addLineNumbers(text, startLine);
      }
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        content: [
          {
            type: "resource",
            resource: {
              uri: `qmd://${encodeQmdPath(result.displayPath)}`,
              name: result.displayPath,
              title: result.title,
              mimeType: "text/markdown",
              text,
            },
          },
        ],
      };
    }),
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description:
        "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md'), comma-separated list, or docids. Skips files larger than maxBytes.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        pattern: z
          .string()
          .describe(
            "Glob pattern, docid, or comma-separated list of file paths/docids",
          ),
        maxLines: z.number().optional().describe("Maximum lines per file"),
        maxBytes: z
          .number()
          .optional()
          .default(DEFAULT_MULTI_GET_MAX_BYTES)
          .describe("Skip files larger than this (default: 65536 = 64KB)"),
        lineNumbers: z
          .boolean()
          .optional()
          .default(true)
          .describe(
            "Add line numbers to output (format: 'N: content'). On by default; set false for raw content.",
          ),
      }),
    },
    track(async ({ pattern, maxLines, maxBytes, lineNumbers }) => {
      const { docs, errors } = await service.multiGet(pattern, {
        includeBody: true,
        maxBytes: maxBytes || DEFAULT_MULTI_GET_MAX_BYTES,
      });

      if (docs.length === 0 && errors.length === 0) {
        return {
          content: [{ type: "text", text: "No matching documents" }],
          isError: true,
        };
      }

      const content: (
        | { type: "text"; text: string }
        | {
            type: "resource";
            resource: {
              uri: string;
              name: string;
              title?: string;
              mimeType: string;
              text: string;
            };
          }
      )[] = [];

      if (errors.length > 0) {
        content.push({ type: "text", text: "Some documents were unavailable" });
      }

      for (const result of docs) {
        if (result.skipped) {
          content.push({
            type: "text",
            text: "[SKIPPED: document exceeds the configured size limit]",
          });
          continue;
        }

        let text = result.doc.body || "";
        if (maxLines !== undefined) {
          const lines = text.split("\n");
          text = lines.slice(0, maxLines).join("\n");
          if (lines.length > maxLines) {
            text += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
          }
        }
        if (lineNumbers) {
          text = addLineNumbers(text);
        }
        if (result.doc.context) {
          text = `<!-- Context: ${result.doc.context} -->\n\n` + text;
        }

        content.push({
          type: "resource",
          resource: {
            uri: `qmd://${encodeQmdPath(result.doc.displayPath)}`,
            name: result.doc.displayPath,
            title: result.doc.title,
            mimeType: "text/markdown",
            text,
          },
        });
      }

      return { content };
    }),
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_status (Index status)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "status",
    {
      title: "Index Status",
      description:
        "Show the status of the QMD index: collections, document counts, and health information.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({}),
    },
    track(async () => {
      const status: StatusResult = await service.getStatus();

      const summary = [
        `QMD Index Status:`,
        `  Total documents: ${status.totalDocuments}`,
        `  Needs embedding: ${status.needsEmbedding}`,
        `  Vector index: ${status.hasVectorIndex ? "yes" : "no"}`,
        `  Collections: ${status.collections.length}`,
      ];

      for (const col of status.collections) {
        summary.push(`    - ${col.name}: ${col.path} (${col.documents} docs)`);
      }

      return {
        content: [{ type: "text", text: summary.join("\n") }],
        structuredContent: status,
      };
    }),
  );

  return server;
}

// =============================================================================
// Transport: stdio (default)
// =============================================================================

export type McpStartupOptions = {
  dbPath?: string;
};

/**
 * Counts running request handlers so shutdown can wait for them to settle
 * before tearing down their llm/store dependencies. The SDK aborts in-flight
 * request controllers on close, but qmd's handlers finish their current
 * store/llm work rather than observing the signal mid-operation.
 */
export type InflightGate = {
  /** Wraps a handler so the gate counts it while it runs. */
  track<T extends (...args: never[]) => unknown>(fn: T): T;
  /** Resolves once no tracked handler runs, or after an optional timeout. */
  waitForIdle(timeoutMs?: number): Promise<boolean>;
};

export function createInflightGate(): InflightGate {
  // `active` is a running-handler counter, not a closed admission barrier.
  // The barrier comes from the caller's ordering: registerStdioEofShutdown
  // runs closeServer() (which stops the transport from dispatching new
  // requests) BEFORE waitForIdle(), so by the time we wait, the only handlers
  // that can still be running are ones already dispatched — there is no source
  // of late admissions to guard against under the stdio transport.
  let active = 0;
  const waiters: Array<() => void> = [];
  return {
    track(fn) {
      const wrapped = async (...args: never[]) => {
        active += 1;
        try {
          return await fn(...args);
        } finally {
          active -= 1;
          if (active === 0) {
            while (waiters.length > 0) waiters.shift()!();
          }
        }
      };
      return wrapped as typeof fn;
    },
    waitForIdle(timeoutMs?: number): Promise<boolean> {
      if (active === 0) return Promise.resolve(true);
      return new Promise((resolve) => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const onIdle = () => {
          if (timer) clearTimeout(timer);
          resolve(true);
        };
        if (timeoutMs !== undefined) {
          timer = setTimeout(() => {
            const i = waiters.indexOf(onIdle);
            if (i >= 0) waiters.splice(i, 1);
            resolve(false);
          }, timeoutMs);
          timer.unref?.();
        }
        waiters.push(onIdle);
      });
    },
  };
}

/** Minimal stdin surface consumed by registerStdioEofShutdown, injectable for tests. */
export type StdioShutdownStdin = {
  once(event: "end" | "close", listener: () => void): unknown;
  off(event: "end" | "close", listener: () => void): unknown;
  readableEnded?: boolean;
  destroyed?: boolean;
};

export type StdioShutdownOptions = {
  /** Closes the MCP server and its transport. */
  closeServer: () => Promise<void>;
  /** Closes the SQLite store (owns disposing the per-store llama.cpp instance). */
  closeStore: () => void | Promise<void>;
  /**
   * Optional extra llama.cpp teardown, run before closeStore. The MCP store
   * disposes its own per-store LlamaCpp inside closeStore, so this is left
   * unset there; it exists for callers that own a separate instance. If
   * omitted, the step is skipped (do NOT default it to the global
   * disposeDefaultLlamaCpp — that would tear down an unrelated instance in an
   * embedded process).
   */
  disposeLlm?: () => Promise<void>;
  /** Waits for in-flight handlers to settle (see InflightGate.waitForIdle). */
  waitForIdle?: (timeoutMs: number) => Promise<boolean>;
  /** Deadline for the in-flight wait. Defaults to 5000 ms. */
  idleTimeoutMs?: number;
  /** Defaults to process.stdin. */
  stdin?: StdioShutdownStdin;
  /** Defaults to assigning process.exitCode. */
  setExitCode?: (code: number) => void;
  /** Defaults to reading process.exitCode. */
  getExitCode?: () => number | undefined;
  /** Defaults to process.stderr. */
  stderr?: {
    write(chunk: string): unknown;
    on?(event: "error", listener: (err: unknown) => void): unknown;
  };
};

/**
 * Shut the stdio MCP server down when stdin reaches EOF (#751).
 *
 * The SDK's StdioServerTransport subscribes to stdin "data"/"error" only and
 * never notices "end"/"close". When the parent MCP client dies, nothing tears
 * the process down: the warm llama.cpp model's native handles keep the event
 * loop alive, so the server reparents to PID 1, leaks RAM, and keeps the
 * SQLite index open. stdin EOF means the client is gone, so this treats it as
 * a disconnect: no new requests are accepted and nobody is left to read a
 * response — but handlers that are already running get a bounded window to
 * settle (waitForIdle) before their llm/store dependencies are torn down.
 *
 * Teardown order matters. Close the transport first so no further requests
 * are dispatched, wait for in-flight handlers, then close the store last —
 * which disposes the store's own llama.cpp instance and then the database, so
 * the dispose path cannot hit an already-closed DB. (disposeLlm is an optional
 * extra step for callers that own a separate instance; the MCP store does
 * not.) Failures are logged best-effort (the parent's death may have closed
 * stderr too) and do not stop the remaining steps. The function sets process.exitCode
 * instead of calling process.exit() so `beforeExit` still fires and
 * node-llama-cpp's auto-dispose runs before libc's static destructors —
 * process.exit() during native-addon unload has caused exit-time crashes
 * before (#59, #129; same rationale as finishSuccessfulCliCommand in the CLI).
 *
 * Returns the idempotent shutdown function: every invocation (manual, "end",
 * "close", or already-ended stdin) shares one promise, and the promise never
 * rejects.
 */
export function registerStdioEofShutdown(
  options: StdioShutdownOptions,
): () => Promise<void> {
  const stdin = options.stdin ?? process.stdin;
  const stderr = options.stderr ?? process.stderr;
  const setExitCode =
    options.setExitCode ??
    ((code: number) => {
      process.exitCode = code;
    });
  const getExitCode =
    options.getExitCode ??
    (() =>
      typeof process.exitCode === "number" ? process.exitCode : undefined);
  let shutdownPromise: Promise<void> | null = null;

  // If the parent died, its stderr pipe may be gone: writes can throw
  // synchronously or emit an async stream error. Logging must never take the
  // teardown down with it.
  stderr.on?.("error", () => {});
  const safeWrite = (chunk: string): void => {
    try {
      stderr.write(chunk);
    } catch {
      // stderr went away with the parent
    }
  };

  const performShutdown = async (): Promise<void> => {
    try {
      stdin.off("end", onStdinEof);
      stdin.off("close", onStdinEof);
    } catch {
      // an exotic stdin may throw on off(); shutdown continues regardless
    }

    // Same stderr breadcrumb style as the HTTP transport's SIGTERM/SIGINT
    // handlers; also gives tests an observable signal that the EOF path ran.
    safeWrite("Shutting down (stdin closed)...\n");

    let failed = false;
    const step = async (
      name: string,
      run: () => void | Promise<void>,
    ): Promise<void> => {
      try {
        await run();
      } catch (error) {
        failed = true;
        safeWrite(
          `QMD Warning: ${name} failed during stdio shutdown; continuing shutdown.\n`,
        );
      }
    };

    await step("server.close()", options.closeServer);
    if (options.waitForIdle) {
      await step("in-flight drain", async () => {
        const idle = await options.waitForIdle!(options.idleTimeoutMs ?? 5000);
        if (!idle) {
          safeWrite(
            "QMD Warning: in-flight request did not settle before the shutdown deadline; continuing shutdown.\n",
          );
        }
      });
    }
    if (options.disposeLlm) {
      await step("llama disposal", options.disposeLlm);
    }
    await step("store.close()", options.closeStore);

    try {
      const prior = getExitCode();
      if (failed) {
        setExitCode(1);
      } else if (prior === undefined || prior === 0) {
        setExitCode(0);
      }
      // else: keep an earlier nonzero status instead of masking it
    } catch {
      // injected setExitCode/getExitCode must not break the shutdown promise
    }
  };

  const shutdown = (): Promise<void> => (shutdownPromise ??= performShutdown());
  const onStdinEof = (): void => {
    void shutdown().catch(() => {});
  };

  stdin.once("end", onStdinEof);
  stdin.once("close", onStdinEof);

  // The parent can die between spawn and listener registration; check the
  // stream flags after subscribing so an already-ended stdin still shuts down.
  if (stdin.readableEnded || stdin.destroyed) {
    onStdinEof();
  }

  return shutdown;
}

export async function startMcpServer(
  options: McpStartupOptions = {},
): Promise<void> {
  // Opt into production mode when the MCP server is actually started, not
  // when this module is merely imported for its exports. Importing the module
  // at the top level flipped the global production flag and broke test
  // isolation for downstream suites that expect the default (development)
  // database path behaviour.
  enableProductionMode();
  const configPath = getConfigPath();
  const store = await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  const service = new QMDWorkService(store);
  const inflight = createInflightGate();
  // serveStdio dual-speaks 2026-07-28 and 2025-era clients on one connection
  // (opening exchange pins the era). A hand-wired StdioServerTransport would
  // stay 2025-only even on SDK 2.x.
  const handle = serveStdio(() => createMcpServer(service, inflight));

  // Follow the parent's lifecycle: when stdin reaches EOF the client is gone
  // and the server must exit instead of orphaning to PID 1 (#751). No
  // disposeLlm here — store.close() disposes this store's own LlamaCpp
  // instance, so passing the global disposeDefaultLlamaCpp would only risk
  // tearing down an unrelated instance in an embedded process.
  registerStdioEofShutdown({
    closeServer: () => handle.close(),
    waitForIdle: (timeoutMs) => inflight.waitForIdle(timeoutMs),
    closeStore: () => service.close(),
  });
}

// =============================================================================
// Transport: Streamable HTTP
// =============================================================================

export type HttpServerHandle = {
  httpServer: import("http").Server;
  port: number;
  stop: () => Promise<void>;
};

/**
 * Start MCP server over Streamable HTTP (JSON responses by default).
 * Binds to `options.host` (default "localhost", overridable via the QMD_HOST
 * env var) — set "0.0.0.0" to accept connections from other hosts, e.g. a
 * container liveness probe. Returns a handle for shutdown and port discovery.
 *
 * HTTP is sessionless (MCP 2026-07-28): there is no `Mcp-Session-Id`, no
 * initialize handshake, and no idle-session TTL. 2025-era clients are still
 * served per-request via the SDK's stateless legacy fallback (initialize
 * works as a standalone call; subsequent 2025 methods need a modern envelope
 * or a stdio connection). The previous session reaper (#816) is gone because
 * there are no sessions to reap.
 */
export async function startMcpHttpServer(
  port: number,
  options: {
    quiet?: boolean;
    host?: string;
    allowedOrigins?: string[];
    allowedHosts?: string[];
  } & McpStartupOptions = {},
): Promise<HttpServerHandle> {
  // See startMcpServer() for the rationale — flip production mode here so the
  // HTTP transport resolves the real database path, without leaking state into
  // callers that only import this module for its exports (e.g. tests).
  enableProductionMode();
  const configPath = getConfigPath();
  const store = await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  const service = new QMDWorkService(store);

  // Official 2026-07-28 HTTP entry: one factory, per-request instance, JSON
  // responses. Modern and legacy traffic share the same process-wide service.
  const mcpHandler = createMcpHandler(() => createMcpServer(service), {
    responseMode: "json",
  });

  async function handleLegacyMcpRequest(
    request: Request,
    parsedBody: unknown,
  ): Promise<Response> {
    if (request.method.toUpperCase() !== "POST") {
      return Response.json(
        {
          jsonrpc: "2.0",
          error: { code: -32000, message: "Method not allowed." },
          id: null,
        },
        { status: 405 },
      );
    }

    const server = await createMcpServer(service);
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    try {
      await server.connect(transport);
      return await transport.handleRequest(request, { parsedBody });
    } finally {
      await transport.close().catch(() => {});
      await server.close().catch(() => {});
    }
  }

  const startTime = Date.now();
  const quiet = options?.quiet ?? false;
  const requestControllers = new Set<AbortController>();

  /** Format timestamp for request logging */
  function ts(): string {
    return new Date().toISOString().slice(11, 23); // HH:mm:ss.SSS
  }

  function log(msg: string): void {
    if (!quiet) console.error(msg);
  }

  function nodeHeadersToWeb(nodeReq: IncomingMessage): Headers {
    const headers = new Headers();
    for (const [k, v] of Object.entries(nodeReq.headers)) {
      if (typeof v === "string") headers.set(k, v);
      else if (Array.isArray(v)) {
        for (const item of v) headers.append(k, item);
      }
    }
    return headers;
  }

  async function collectBody(req: IncomingMessage): Promise<string> {
    const declaredLength = req.headers["content-length"];
    if (declaredLength !== undefined) {
      const parsedLength = Number(declaredLength);
      if (
        !Number.isSafeInteger(parsedLength) ||
        parsedLength < 0 ||
        parsedLength > MAX_HTTP_REQUEST_BODY_BYTES
      ) {
        req.resume();
        throw new WorkServiceError("malformed");
      }
    }
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      let size = 0;
      const timer = setTimeout(() => {
        cleanup();
        req.destroy();
        reject(new WorkServiceError("malformed"));
      }, HTTP_REQUEST_BODY_TIMEOUT_MS);
      timer.unref?.();
      const cleanup = () => {
        clearTimeout(timer);
        req.removeListener("data", onData);
        req.removeListener("end", onEnd);
        req.removeListener("aborted", onAborted);
        req.removeListener("close", onClose);
        req.removeListener("error", onError);
      };
      const onData = (chunk: Buffer | string) => {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += buffer.length;
        if (size > MAX_HTTP_REQUEST_BODY_BYTES) {
          cleanup();
          req.resume();
          reject(new WorkServiceError("malformed"));
          return;
        }
        chunks.push(buffer);
      };
      const onEnd = () => {
        cleanup();
        resolve(Buffer.concat(chunks, size).toString());
      };
      const onAborted = () => {
        cleanup();
        reject(new WorkServiceError("closed"));
      };
      const onClose = () => {
        if (req.complete) return;
        cleanup();
        reject(new WorkServiceError("closed"));
      };
      const onError = () => {
        cleanup();
        reject(new WorkServiceError("malformed"));
      };
      req.on("data", onData);
      req.once("end", onEnd);
      req.once("aborted", onAborted);
      req.once("close", onClose);
      req.once("error", onError);
    });
  }

  function createRequestAbort(
    req: IncomingMessage,
    res: ServerResponse,
  ): {
    signal: AbortSignal;
    cleanup: () => void;
  } {
    const controller = new AbortController();
    requestControllers.add(controller);
    const abort = () => controller.abort();
    const onRequestClose = () => {
      if (!req.complete) abort();
    };
    const onResponseClose = () => {
      if (!res.writableEnded) abort();
    };
    req.once("aborted", abort);
    req.once("close", onRequestClose);
    res.once("close", onResponseClose);
    return {
      signal: controller.signal,
      cleanup: () => {
        req.removeListener("aborted", abort);
        req.removeListener("close", onRequestClose);
        res.removeListener("close", onResponseClose);
        requestControllers.delete(controller);
      },
    };
  }

  function writeJson(res: ServerResponse, status: number, body: unknown): void {
    if (res.destroyed || res.headersSent) return;
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(body));
  }

  function workErrorReason(error: unknown): string {
    return error instanceof WorkServiceError ? error.reason : "store_error";
  }

  function workErrorStatus(error: unknown): number {
    return error instanceof WorkServiceError && error.reason === "malformed"
      ? 400
      : 503;
  }

  function writeWorkError(res: ServerResponse, error: unknown): void {
    writeJson(res, workErrorStatus(error), {
      status: "unavailable",
      reason: workErrorReason(error),
      authoritativeEmpty: false,
    });
  }

  function parseJsonObject(rawBody: string): Record<string, unknown> {
    let value: unknown;
    try {
      value = JSON.parse(rawBody);
    } catch {
      throw new WorkServiceError("malformed");
    }
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new WorkServiceError("malformed");
    }
    return value as Record<string, unknown>;
  }

  function writeSearchOutcome(
    res: ServerResponse,
    outcome: Awaited<ReturnType<QMDWorkService["search"]>>,
    request: DaemonSearchRequest,
    uriPaths: boolean,
  ): void {
    if (outcome.status === "unavailable") {
      writeJson(res, 503, outcome);
      return;
    }
    const query = primarySearchQuery(request);
    writeJson(res, 200, {
      ...outcome,
      results: formatSearchResults(
        outcome.results,
        query,
        request.intent,
        uriPaths,
      ),
    });
  }

  const host = options.host ?? process.env.QMD_HOST ?? "localhost";
  const originGuard = resolveOriginGuard({
    host,
    ...(options.allowedOrigins
      ? { allowedOrigins: options.allowedOrigins }
      : {}),
    ...(options.allowedHosts ? { allowedHosts: options.allowedHosts } : {}),
  });

  const httpInflight = createInflightGate();
  let accepting = true;
  const httpServer = createServer(
    httpInflight.track(
      async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
        const reqStart = Date.now();
        const pathname = (nodeReq.url || "/").split("?")[0] ?? "/";
        const requestAbort = createRequestAbort(nodeReq, nodeRes);

        try {
          if (!accepting) {
            writeJson(nodeRes, 503, {
              status: "unavailable",
              reason: "closed",
              authoritativeEmpty: false,
            });
            return;
          }
          const origin = nodeReq.headers.origin;
          const hostHeader = nodeReq.headers.host;
          const verdict = checkRequestOrigin(
            {
              origin: typeof origin === "string" ? origin : undefined,
              host: typeof hostHeader === "string" ? hostHeader : undefined,
            },
            originGuard,
          );
          if (!verdict.ok) {
            writeJson(nodeRes, 403, {
              jsonrpc: "2.0",
              error: { code: -32003, message: "Forbidden: Origin not allowed" },
              id: null,
            });
            log(`${ts()} request rejected 403 (${Date.now() - reqStart}ms)`);
            return;
          }

          if (pathname === "/health" && nodeReq.method === "GET") {
            const state = await service.health();
            writeJson(nodeRes, 200, {
              status: "ok",
              version: getPackageVersion(),
              apiVersion: API_VERSION,
              features: [...DAEMON_FEATURES],
              uptime: Math.floor((Date.now() - startTime) / 1000),
              admission: state.admission,
              indexGeneration: state.indexGeneration,
            });
            log(`${ts()} health (${Date.now() - reqStart}ms)`);
            return;
          }

          if (nodeReq.method === "POST" && pathname === "/v1/search") {
            const request = parseJsonObject(
              await collectBody(nodeReq),
            ) as DaemonSearchRequest;
            const outcome = await service.search(request, requestAbort.signal);
            writeSearchOutcome(nodeRes, outcome, request, true);
            log(`${ts()} v1 search (${Date.now() - reqStart}ms)`);
            return;
          }

          if (
            nodeReq.method === "POST" &&
            (pathname === "/query" || pathname === "/search")
          ) {
            const params = parseJsonObject(await collectBody(nodeReq));
            if (!Array.isArray(params.searches))
              throw new WorkServiceError("malformed");
            const request: DaemonSearchRequest = {
              searches: params.searches as DaemonSearchRequest["searches"],
              ...(Array.isArray(params.collections)
                ? { collections: params.collections.map(String) }
                : {}),
              ...(typeof params.limit === "number"
                ? { limit: params.limit }
                : {}),
              ...(typeof params.minScore === "number"
                ? { minScore: params.minScore }
                : {}),
              ...(typeof params.candidateLimit === "number"
                ? { candidateLimit: params.candidateLimit }
                : {}),
              ...(typeof params.intent === "string"
                ? { intent: params.intent }
                : {}),
              ...(typeof params.rerank === "boolean"
                ? { rerank: params.rerank }
                : {}),
            };
            const outcome = await service.search(request, requestAbort.signal);
            writeSearchOutcome(nodeRes, outcome, request, true);
            log(`${ts()} legacy search (${Date.now() - reqStart}ms)`);
            return;
          }

          if (nodeReq.method === "POST" && pathname === "/v1/update") {
            const request = parseJsonObject(
              await collectBody(nodeReq),
            ) as UpdateScope;
            const operation = service.scheduleUpdate(request);
            writeJson(nodeRes, 202, operation);
            log(`${ts()} v1 update (${Date.now() - reqStart}ms)`);
            return;
          }

          if (nodeReq.method === "POST" && pathname === "/v1/embed") {
            const request = parseJsonObject(
              await collectBody(nodeReq),
            ) as DaemonEmbedRequest;
            const operation = service.scheduleEmbed(request);
            writeJson(nodeRes, 202, operation);
            log(`${ts()} v1 embed (${Date.now() - reqStart}ms)`);
            return;
          }

          if (
            nodeReq.method === "POST" &&
            pathname === "/v1/collections/ensure"
          ) {
            const request = parseJsonObject(
              await collectBody(nodeReq),
            ) as CollectionEnsureRequest;
            const operation = service.scheduleEnsure(request);
            writeJson(nodeRes, 202, operation);
            log(`${ts()} v1 ensure (${Date.now() - reqStart}ms)`);
            return;
          }

          if (nodeReq.method === "GET" && pathname === "/v1/collections") {
            const collections = await service.listCollections();
            writeJson(nodeRes, 200, { collections });
            log(`${ts()} v1 collections (${Date.now() - reqStart}ms)`);
            return;
          }

          if (
            nodeReq.method === "GET" &&
            pathname.startsWith("/v1/operations/")
          ) {
            let operationId: string;
            try {
              operationId = decodeURIComponent(
                pathname.slice("/v1/operations/".length),
              );
            } catch {
              throw new WorkServiceError("malformed");
            }
            const operation = service.getOperation(operationId);
            if (!operation) {
              writeJson(nodeRes, 404, {
                status: "unavailable",
                reason: "malformed",
                authoritativeEmpty: false,
              });
              return;
            }
            writeJson(nodeRes, 200, operation);
            log(`${ts()} v1 operation (${Date.now() - reqStart}ms)`);
            return;
          }

          if (pathname === "/mcp") {
            const rawBody =
              nodeReq.method !== "GET" && nodeReq.method !== "HEAD"
                ? await collectBody(nodeReq)
                : undefined;
            let parsedBody: unknown;
            if (rawBody) {
              try {
                parsedBody = JSON.parse(rawBody);
              } catch {
                parsedBody = undefined;
              }
            }
            const hostHeader =
              typeof nodeReq.headers.host === "string"
                ? nodeReq.headers.host
                : `localhost:${port}`;
            const url = `http://${hostHeader}${pathname}`;
            const request = new Request(url, {
              method: nodeReq.method || "GET",
              headers: nodeHeadersToWeb(nodeReq),
              ...(rawBody !== undefined ? { body: rawBody } : {}),
              signal: requestAbort.signal,
            });
            const response = (await isLegacyRequest(request, parsedBody))
              ? await handleLegacyMcpRequest(request, parsedBody)
              : await mcpHandler.fetch(
                  request,
                  parsedBody !== undefined ? { parsedBody } : undefined,
                );

            if (!nodeRes.destroyed) {
              nodeRes.writeHead(
                response.status,
                Object.fromEntries(response.headers),
              );
              if (response.body) {
                await pipeline(Readable.fromWeb(response.body), nodeRes);
              } else {
                nodeRes.end();
              }
            }
            log(`${ts()} mcp request (${Date.now() - reqStart}ms)`);
            return;
          }

          nodeRes.writeHead(404);
          nodeRes.end("Not Found");
        } catch (error) {
          if (requestAbort.signal.aborted) return;
          if (error instanceof WorkServiceError) {
            writeWorkError(nodeRes, error);
          } else {
            writeJson(nodeRes, 500, {
              status: "unavailable",
              reason: "store_error",
              authoritativeEmpty: false,
            });
          }
          log(`${ts()} request failed (${Date.now() - reqStart}ms)`);
        } finally {
          requestAbort.cleanup();
        }
      },
    ),
  );

  try {
    await new Promise<void>((resolve, reject) => {
      const onError = (error: Error) => {
        httpServer.removeListener("listening", onListening);
        reject(error);
      };
      const onListening = () => {
        httpServer.removeListener("error", onError);
        resolve();
      };
      httpServer.once("error", onError);
      httpServer.once("listening", onListening);
      httpServer.listen(port, host);
    });
  } catch (error) {
    await service.close().catch(() => {});
    throw error;
  }

  const actualPort = (httpServer.address() as import("net").AddressInfo).port;

  let stopPromise: Promise<void> | undefined;
  const stop = (): Promise<void> =>
    (stopPromise ??= (async () => {
      accepting = false;
      process.off("SIGTERM", onSigterm);
      process.off("SIGINT", onSigint);
      for (const controller of requestControllers) controller.abort();
      const httpClosed = new Promise<void>((resolve, reject) => {
        httpServer.close((error) => (error ? reject(error) : resolve()));
      });
      httpServer.closeAllConnections();
      let firstError: unknown;
      const step = async (run: () => Promise<unknown>): Promise<void> => {
        try {
          await run();
        } catch (error) {
          firstError ??= error;
        }
      };
      await step(async () => {
        if (!(await httpInflight.waitForIdle(5_000))) {
          throw new Error("HTTP request drain timed out");
        }
      });
      httpServer.closeAllConnections();
      await step(() => mcpHandler.close());
      await step(() => service.close());
      await step(() => httpClosed);
      if (firstError) throw firstError;
    })());

  const stopForSignal = (signal: "SIGTERM" | "SIGINT"): void => {
    log(`Shutting down (${signal})...`);
    void stop().then(
      () => {
        process.exitCode = 0;
      },
      () => {
        process.exitCode = 1;
      },
    );
  };
  const onSigterm = (): void => stopForSignal("SIGTERM");
  const onSigint = (): void => stopForSignal("SIGINT");
  process.on("SIGTERM", onSigterm);
  process.on("SIGINT", onSigint);

  log(`QMD MCP server listening on http://${host}:${actualPort}/mcp`);
  if (originGuard.disabled) {
    log(
      "Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. Only do this behind your own authenticating proxy.",
    );
  } else if (!originGuard.enforceHost) {
    log(
      `Warning: bound to ${host} with no QMD_ALLOWED_HOSTS — Host validation is off and the index is readable by anyone who can reach this port.`,
    );
  }
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (
  fileURLToPath(import.meta.url) === process.argv[1] ||
  process.argv[1]?.endsWith("/server.ts") ||
  process.argv[1]?.endsWith("/server.js")
) {
  startMcpServer().catch(() => {
    console.error("QMD MCP server failed to start.");
    process.exitCode = 1;
  });
}
