/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Speaks MCP spec 2026-07-28 (stateless, no initialize handshake) and dual-speaks
 * 2025-era clients via the official SDK entries (`serveStdio` / `createMcpHandler`).
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "url";
import { createMcpHandler, McpServer, ResourceTemplate } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import { existsSync } from "fs";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  getDefaultDbPath,
  DEFAULT_MULTI_GET_MAX_BYTES,
  type QMDStore,
  type ExpandedQuery,
  type IndexStatus,
} from "../index.js";
import { getConfigPath } from "../collections.js";
import { enableProductionMode } from "../store.js";
import { LlamaCpp } from "../llm.js";
import {
  applyProcessExitCode,
  createEofWatchdog,
  createShutdownCoordinator,
  createShutdownHold,
  formatShutdownError,
  type ShutdownTrigger,
} from "../shutdown.js";
import { checkRequestOrigin, resolveOriginGuard } from "./origin-guard.js";

// =============================================================================
// Types for structured content
// =============================================================================

type SearchResultItem = {
  docid: string;  // Short docid (#abc123) for quick reference
  file: string;
  title: string;
  score: number;
  context: string | null;
  line: number;   // Absolute line in source markdown
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
  return path.split('/').map(segment => encodeURIComponent(segment)).join('/');
}

/**
 * Format search results as human-readable text summary
 */
function formatSearchSummary(results: SearchResultItem[], query: string): string {
  if (results.length === 0) {
    return `No results found for "${query}"`;
  }
  const lines = [`Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}":\n`];
  for (const r of results) {
    lines.push(`${r.docid} ${Math.round(r.score * 100)}% ${r.file} - ${r.title}`);
  }
  return lines.join('\n');
}

function getPackageVersion(): string {
  try {
    const pkgPath = join(dirname(fileURLToPath(import.meta.url)), "../../package.json");
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
async function buildInstructions(store: QMDStore): Promise<string> {
  const status = await store.getStatus();
  const globalCtx = await store.getGlobalContext();
  const lines: string[] = [];

  // --- What is this? ---
  lines.push(`QMD is your local search engine over ${status.totalDocuments} markdown documents.`);
  if (globalCtx) lines.push(`Context: ${globalCtx}`);

  // --- What's searchable? ---
  // Emit names only — the per-collection doc counts and descriptions can run to ~1.5 KB
  // across a dozen collections, and the same info is available on demand via the `status` tool.
  if (status.collections.length > 0) {
    lines.push("");
    const names = status.collections.map(c => c.name).join(", ");
    lines.push(`Collections (scope with \`collections\` parameter): ${names}`);
    lines.push("Call the `status` tool for collection descriptions, paths, and per-collection doc counts.");
  }

  // --- Capability gaps ---
  if (!status.hasVectorIndex) {
    lines.push("");
    lines.push("Note: No vector embeddings yet. Run `qmd embed` to enable semantic search (vec/hyde).");
  } else if (status.needsEmbedding > 0) {
    lines.push("");
    lines.push(`Note: ${status.needsEmbedding} documents need embedding. Run \`qmd embed\` to update.`);
  }

  // --- Search tool ---
  lines.push("");
  lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):");
  lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)");
  lines.push("  - type:'vec' — semantic vector search (meaning-based)");
  lines.push("  - type:'hyde' — hypothetical document (write what the answer looks like)");
  lines.push("");
  lines.push("  Always provide `intent` on every search call to disambiguate and improve snippets.");
  lines.push("");
  lines.push("Examples:");
  lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]");
  lines.push("  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]");
  lines.push("  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]");
  lines.push("  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'");

  // --- Retrieval workflow ---
  lines.push("");
  lines.push("Retrieval:");
  lines.push("  - `get` — single document by path or docid (#abc123). Supports a line-range suffix: `file.md:100` (from line 100) or `file.md:100:40` (40 lines from line 100).");
  lines.push("  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`), comma-separated list, or docids (#abc123).");

  // --- Non-obvious things that prevent mistakes ---
  lines.push("");
  lines.push("Tips:");
  lines.push("  - File paths in results are relative to their collection.");
  lines.push("  - Use `minScore: 0.5` to filter low-confidence results.");
  lines.push("  - Results include a `context` field describing the content type.");

  return lines.join("\n");
}

/**
 * Create an MCP server with all QMD tools, resources, and prompts registered.
 * Shared by both stdio and HTTP transports.
 */
async function createMcpServer(store: QMDStore, inflight?: InflightGate): Promise<McpServer> {
  // Wraps request handlers so a stdio EOF shutdown can wait for in-flight
  // work to settle before disposing the store/llm underneath it.
  const track = inflight?.track ?? (<T,>(fn: T): T => fn);
  const server = new McpServer(
    { name: "qmd", version: getPackageVersion() },
    {
      instructions: await buildInstructions(store),
      // tools/list is static for the process lifetime; resources/read stays
      // uncacheable because the index can change under us.
      cacheHints: {
        "tools/list": { ttlMs: 60_000, cacheScope: "private" },
        "server/discover": { ttlMs: 60_000, cacheScope: "private" },
        "resources/read": { ttlMs: 0, cacheScope: "private" },
      },
    },
  );

  // Pre-fetch default collection names for search tools
  const defaultCollectionNames = await store.getDefaultCollectionNames();

  // ---------------------------------------------------------------------------
  // Resource: qmd://{path} - read-only access to documents by path
  // Note: No list() - documents are discovered via search tools
  // ---------------------------------------------------------------------------

  server.registerResource(
    "document",
    new ResourceTemplate("qmd://{+path}", { list: undefined }),
    {
      title: "QMD Document",
      description: "A markdown document from your QMD knowledge base. Use search tools to discover documents.",
      mimeType: "text/markdown",
    },
    track(async (uri, { path }) => {
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const pathStr = Array.isArray(path) ? path.join('/') : (path || '');
      const decodedPath = decodeURIComponent(pathStr);

      // Use SDK to find document — findDocument handles collection/path resolution
      const result = await store.get(decodedPath, { includeBody: true });

      if ("error" in result) {
        const text = result.error === "excluded_by_ignore"
          ? `Document excluded by ignore rule: ${decodedPath}\nCollection: ${result.collection}\nMatched path: ${result.path}\nIgnore rule: ${result.rule}`
          : `Document not found: ${decodedPath}`;
        return { contents: [{ uri: uri.href, text }] };
      }

      let text = addLineNumbers(result.body || "");  // Default to line numbers
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        contents: [{
          uri: uri.href,
          name: result.displayPath,
          title: result.title || result.displayPath,
          mimeType: "text/markdown",
          text,
        }],
      };
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: query (Primary search tool)
  // ---------------------------------------------------------------------------

  const subSearchSchema = z.object({
    type: z.enum(['lex', 'vec', 'hyde']).describe(
      "lex = BM25 keywords (supports \"phrase\" and -negation); " +
      "vec = semantic question; hyde = hypothetical answer passage"
    ),
    query: z.string().describe(
      "The query text. For lex: use keywords, \"quoted phrases\", and -negation. " +
      "For vec: natural language question. For hyde: 50-100 word answer passage."
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
        query: z.string().optional().describe(
          "Plain-text query, auto-expanded by the SDK into lex/vec/hyde variants, fused via " +
          "RRF and reranked. Recommended default for most searches. Mutually exclusive with 'searches'."
        ),
        searches: z.array(subSearchSchema).max(10).optional().describe(
          "Typed sub-queries to execute (lex/vec/hyde). First gets 2x weight. Use for precise " +
          "control over retrieval strategy. Mutually exclusive with 'query'."
        ),
        limit: z.number().optional().default(10).describe("Max results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Min relevance 0-1 (default: 0)"),
        candidateLimit: z.number().optional().describe(
          "Maximum candidates to rerank (default: 40, lower = faster but may miss results)"
        ),
        collections: z.array(z.string()).optional().describe("Filter to collections (OR match)"),
        intent: z.string().optional().describe(
          "Background context to disambiguate the query. Example: query='performance', intent='web page load times and Core Web Vitals'. Does not search on its own."
        ),
        rerank: z.boolean().optional().default(true).describe(
          "Rerank results using LLM (default: true). Set to false for faster results on CPU-only machines."
        ),
      }),
    },
    track(async ({ query, searches, limit, minScore, candidateLimit, collections, intent, rerank }) => {
      // Require exactly one of `query` (plain text, auto-expanded) or `searches` (typed sub-queries).
      if (!query && (!searches || searches.length === 0)) {
        return {
          content: [{ type: "text" as const, text: "Error: provide either 'query' (plain text) or 'searches' (typed sub-queries)" }],
          isError: true,
        };
      }
      if (query && searches && searches.length > 0) {
        return {
          content: [{ type: "text" as const, text: "Error: 'query' and 'searches' are mutually exclusive; provide only one" }],
          isError: true,
        };
      }

      // Use default collections if none specified
      const effectiveCollections = collections ?? defaultCollectionNames;

      // Plain `query` is auto-expanded by the SDK (expand → fuse → rerank);
      // `searches` runs the caller's typed sub-queries directly.
      const searchOptions = query
        ? { query }
        : { queries: (searches ?? []).map(s => ({ type: s.type, query: s.query })) };

      const results = await store.search({
        ...searchOptions,
        collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
        limit,
        minScore,
        candidateLimit,
        rerank,
        intent,
      });

      // Use the plain query, or the first lex/vec sub-query, for snippet extraction
      const primaryQuery = query
        || searches?.find(s => s.type === 'lex')?.query
        || searches?.find(s => s.type === 'vec')?.query
        || searches?.[0]?.query
        || "";

      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.body, primaryQuery, 300, r.bestChunkPos, r.bestChunk.length, intent);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: r.context,
          line,
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: formatSearchSummary(filtered, primaryQuery) }],
        structuredContent: { results: filtered },
      };
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_get (Retrieve document)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "get",
    {
      title: "Get Document",
      description: "Retrieve the full content of a document by its file path or docid. Use paths or docids (#abc123) from search results. Suggests similar files if not found.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        file: z.string().describe("File path or docid from search results. Supports a line-range suffix: 'pages/meeting.md:100' starts at line 100; 'pages/meeting.md:100:40' (or '#abc123:100:40') reads 40 lines from line 100."),
        fromLine: z.number().optional().describe("Start from this line number (1-indexed)"),
        maxLines: z.number().optional().describe("Maximum number of lines to return"),
        lineNumbers: z.boolean().optional().default(true).describe("Add line numbers to output (format: 'N: content'). On by default; set false for raw content."),
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
        if (parsedFromLine === undefined) parsedFromLine = parseInt(rangeMatch[1]!, 10);
        if (parsedMaxLines === undefined) parsedMaxLines = parseInt(rangeMatch[2]!, 10);
        lookup = lookup.slice(0, -rangeMatch[0].length);
      } else {
        const colonMatch = lookup.match(/:(\d+)$/);
        if (colonMatch && colonMatch[1] && parsedFromLine === undefined) {
          parsedFromLine = parseInt(colonMatch[1], 10);
          lookup = lookup.slice(0, -colonMatch[0].length);
        }
      }
      if (parsedFromLine !== undefined) parsedFromLine = Math.max(1, parsedFromLine);

      const result = await store.get(lookup, { includeBody: false });

      if ("error" in result) {
        let msg = result.error === "excluded_by_ignore"
          ? `Document excluded by ignore rule: ${file}\nCollection: ${result.collection}\nMatched path: ${result.path}\nIgnore rule: ${result.rule}`
          : `Document not found: ${file}`;
        if (result.error === "not_found" && result.similarFiles.length > 0) {
          msg += `\n\nDid you mean one of these?\n${result.similarFiles.map(s => `  - ${s}`).join('\n')}`;
        }
        return {
          content: [{ type: "text", text: msg }],
          isError: true,
        };
      }

      const body = await store.getDocumentBody(result.filepath, { fromLine: parsedFromLine, maxLines: parsedMaxLines }) ?? "";
      let text = body;
      if (lineNumbers) {
        const startLine = parsedFromLine || 1;
        text = addLineNumbers(text, startLine);
      }
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        content: [{
          type: "resource",
          resource: {
            uri: `qmd://${encodeQmdPath(result.displayPath)}`,
            name: result.displayPath,
            title: result.title,
            mimeType: "text/markdown",
            text,
          },
        }],
      };
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md'), comma-separated list, or docids. Skips files larger than maxBytes.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        pattern: z.string().describe("Glob pattern, docid, or comma-separated list of file paths/docids"),
        maxLines: z.number().optional().describe("Maximum lines per file"),
        maxBytes: z.number().optional().default(DEFAULT_MULTI_GET_MAX_BYTES).describe("Skip files larger than this (default: 65536 = 64KB)"),
        lineNumbers: z.boolean().optional().default(true).describe("Add line numbers to output (format: 'N: content'). On by default; set false for raw content."),
      }),
    },
    track(async ({ pattern, maxLines, maxBytes, lineNumbers }) => {
      const { docs, errors } = await store.multiGet(pattern, { includeBody: true, maxBytes: maxBytes || DEFAULT_MULTI_GET_MAX_BYTES });

      if (docs.length === 0 && errors.length === 0) {
        return {
          content: [{ type: "text", text: `No files matched pattern: ${pattern}` }],
          isError: true,
        };
      }

      const content: ({ type: "text"; text: string } | { type: "resource"; resource: { uri: string; name: string; title?: string; mimeType: string; text: string } })[] = [];

      if (errors.length > 0) {
        content.push({ type: "text", text: `Errors:\n${errors.join('\n')}` });
      }

      for (const result of docs) {
        if (result.skipped) {
          content.push({
            type: "text",
            text: `[SKIPPED: ${result.doc.displayPath} - ${result.skipReason}. Use 'qmd_get' with file="${result.doc.displayPath}" to retrieve.]`,
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
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_status (Index status)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "status",
    {
      title: "Index Status",
      description: "Show the status of the QMD index: collections, document counts, and health information.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({}),
    },
    track(async () => {
      const status: StatusResult = await store.getStatus();

      const summary = [
        `QMD Index Status:`,
        `  Total documents: ${status.totalDocuments}`,
        `  Needs embedding: ${status.needsEmbedding}`,
        `  Vector index: ${status.hasVectorIndex ? 'yes' : 'no'}`,
        `  Collections: ${status.collections.length}`,
      ];

      for (const col of status.collections) {
        summary.push(`    - ${col.name}: ${col.path} (${col.documents} docs)`);
      }

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: status,
      };
    })
  );

  return server;
}

// =============================================================================
// Transport: stdio (default)
// =============================================================================

export type McpStartupOptions = {
  dbPath?: string;
  store?: QMDStore;
  stdin?: StdioShutdownStdin;
  eofGraceMs?: number;
  eofHardStop?: (error: unknown) => void;
  createTransport?: (server: McpServer) => { close(): void | Promise<void> };
  /**
   * When `store` is provided, the caller retains store/LLM ownership even if
   * startup fails. When omitted, the server creates the store and must dispose
   * it (LLM then DB) before rethrowing a startup failure.
   */
};

export class ServerShuttingDownError extends Error {
  constructor() {
    super("QMD server is shutting down");
    this.name = "ServerShuttingDownError";
  }
}

/**
 * Admission gate for in-flight request handlers. Closing admission rejects
 * new work; waitForIdle has no timeout — the supervisor owns the deadline.
 */
export type InflightGate = {
  closeAdmission(): void;
  run<T>(fn: () => T | Promise<T>): Promise<T>;
  track<T extends (...args: never[]) => unknown>(fn: T): T;
  waitForIdle(): Promise<void>;
  getActiveCount(): number;
};

export function createInflightGate(): InflightGate {
  let admissionClosed = false;
  let active = 0;
  const waiters: Array<() => void> = [];

  const leave = () => {
    active -= 1;
    if (active === 0) {
      while (waiters.length > 0) waiters.shift()!();
    }
  };

  const run = async <T>(fn: () => T | Promise<T>): Promise<T> => {
    if (admissionClosed) throw new ServerShuttingDownError();

    // Single-threaded JS: no await between the closed check and increment.
    active += 1;
    try {
      return await fn();
    } finally {
      leave();
    }
  };

  return {
    closeAdmission() {
      admissionClosed = true;
    },
    run,
    track(fn) {
      return ((...args: never[]) => run(() => fn(...args))) as typeof fn;
    },
    waitForIdle() {
      if (active === 0) return Promise.resolve();
      return new Promise<void>((resolve) => waiters.push(resolve));
    },
    getActiveCount() {
      return active;
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
  stdin?: StdioShutdownStdin;
  shutdown: (trigger: ShutdownTrigger) => Promise<void>;
  stderr?: { write(chunk: string): unknown; on?(event: "error", listener: (err: unknown) => void): unknown };
};

/**
 * Thin adapter: stdin EOF becomes a coordinator trigger. Teardown order and
 * failure policy live in the shared shutdown coordinator — this function
 * must not continue disposal after a missed idle deadline.
 */
export function registerStdioEofShutdown(options: StdioShutdownOptions): () => Promise<void> {
  const stdin = options.stdin ?? process.stdin;
  const stderr = options.stderr;
  let triggered: Promise<void> | undefined;

  stderr?.on?.("error", () => {});
  const safeWrite = (chunk: string): void => {
    try {
      stderr?.write(chunk);
    } catch {
      // stderr went away with the parent
    }
  };

  const shutdown = () =>
    (triggered ??= options.shutdown({
      kind: "stdin-eof",
      exitCode: 0,
    }));

  const onEof = () => {
    try {
      stdin.off("end", onEof);
      stdin.off("close", onEof);
    } catch {
      // an exotic stdin may throw on off(); shutdown continues regardless
    }
    safeWrite("Shutting down (stdin closed)...\n");
    void shutdown().catch(() => {});
  };

  stdin.once("end", onEof);
  stdin.once("close", onEof);

  if (stdin.readableEnded || stdin.destroyed) onEof();

  return shutdown;
}

export type StdioMcpHandle = {
  shutdown(trigger?: ShutdownTrigger): Promise<void>;
};

function resolveServerLlm(store: QMDStore): LlamaCpp {
  const existing = store.internal.llm;
  if (existing) return existing;
  const llm = new LlamaCpp();
  store.internal.llm = llm;
  return llm;
}

function logShutdownError(error: unknown): void {
  try {
    process.stderr.write(`QMD shutdown failed: ${formatShutdownError(error)}\n`);
  } catch {
    // stderr may already be gone
  }
}

async function cleanupOwnedMcpStartup(options: {
  ownsStore: boolean;
  store?: QMDStore;
  llm?: LlamaCpp;
  closeTransport?: () => void | Promise<void>;
}): Promise<void> {
  try { await options.closeTransport?.(); } catch { /* best-effort */ }
  if (!options.ownsStore) return;
  try { await options.llm?.dispose(); } catch { /* best-effort */ }
  try { await options.store?.close(); } catch { /* best-effort */ }
}

export async function startMcpServer(options: McpStartupOptions = {}): Promise<StdioMcpHandle> {
  // Opt into production mode when the MCP server is actually started, not
  // when this module is merely imported for its exports. Importing the module
  // at the top level flipped the global production flag and broke test
  // isolation for downstream suites that expect the default (development)
  // database path behaviour.
  enableProductionMode();
  const ownsStore = !options.store;
  const configPath = getConfigPath();
  let store: QMDStore | undefined;
  let llm: LlamaCpp | undefined;
  let transport: { close(): void | Promise<void> } | undefined;
  try {
  store = options.store ?? await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  llm = resolveServerLlm(store);
  const inflight = createInflightGate();
  // Finish store-touching initialization before EOF can dispose ownership.
  // serveStdio's factory otherwise calls createMcpServer() (buildInstructions /
  // getDefaultCollectionNames) after shutdown registration, which would look idle.
  const mcpServer = await createMcpServer(store, inflight);
  // serveStdio dual-speaks 2026-07-28 and 2025-era clients on one connection
  // (opening exchange pins the era). A hand-wired StdioServerTransport would
  // stay 2025-only even on SDK 2.x.
  transport = options.createTransport?.(mcpServer) ?? serveStdio(async () => mcpServer);

  const startedStore = store;
  const startedLlm = llm;
  const startedTransport = transport;
  const coordinator = createShutdownCoordinator({
    closeAdmission() {
      inflight.closeAdmission();
      startedLlm.closeSessionAdmission();
    },
    stopServing: () => startedTransport.close(),
    requestAbort: (reason) => startedLlm.requestSessionAbort(reason),
    waitForInflight: () => inflight.waitForIdle(),
    waitForLlmIdle: () => startedLlm.waitForSessionIdle(),
    disposeLlm: () => startedLlm.dispose(),
    closeStore: () => startedStore.close(),
    setExitCode: applyProcessExitCode,
    logError: logShutdownError,
    holdOpen: createShutdownHold,
  });

  const eofWatchdog = createEofWatchdog({
    graceMs: options.eofGraceMs,
    hardStop: options.eofHardStop,
  });

  registerStdioEofShutdown({
    stdin: options.stdin,
    shutdown: async (trigger) => {
      if (trigger.kind === "stdin-eof") eofWatchdog.arm();
      try {
        await coordinator.shutdown(trigger);
        eofWatchdog.disarm();
      } catch (error) {
        // Leave the watchdog armed: a failed drain must not continue disposal.
        throw error;
      }
    },
    stderr: process.stderr,
  });

  return {
    shutdown: (trigger: ShutdownTrigger = { kind: "complete", exitCode: 0 }) =>
      coordinator.shutdown(trigger),
  };
  } catch (error) {
    await cleanupOwnedMcpStartup({
      ownsStore,
      store,
      llm,
      closeTransport: transport ? () => transport!.close() : undefined,
    });
    throw error;
  }
}

// =============================================================================
// Transport: Streamable HTTP
// =============================================================================

export type HttpServerHandle = {
  httpServer: import("http").Server;
  port: number;
  stop: () => Promise<void>;
  shutdown: (trigger?: ShutdownTrigger) => Promise<void>;
  inflight: InflightGate;
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
  options: ({
    quiet?: boolean;
    host?: string;
    allowedOrigins?: string[];
    allowedHosts?: string[];
    setExitCode?: (code: number) => void;
    holdOpen?: () => () => void;
    hardStop?: (error: unknown) => never;
    listen?: (server: import("http").Server, port: number, host: string) => Promise<void>;
  } & McpStartupOptions) = {},
): Promise<HttpServerHandle> {
  // See startMcpServer() for the rationale — flip production mode here so the
  // HTTP transport resolves the real database path, without leaking state into
  // callers that only import this module for its exports (e.g. tests).
  enableProductionMode();
  const ownsStore = !options.store;
  const configPath = getConfigPath();
  let store: QMDStore | undefined;
  let llm: LlamaCpp | undefined;
  let httpServer: import("http").Server | undefined;
  try {
  store = options.store ?? await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  llm = resolveServerLlm(store);
  const ownedStore = store;
  const inflight = createInflightGate();

  // Pre-fetch default collection names for REST endpoint
  const defaultCollectionNames = await ownedStore.getDefaultCollectionNames();

  // Official 2026-07-28 HTTP entry: one factory, per-request instance, JSON
  // responses (matches the previous enableJsonResponse: true). Dual-speaks
  // 2025-era traffic statelessly by default (`legacy: "stateless"`).
  // No inner inflight gate — HTTP counts each /mcp or REST request once.
  const mcpHandler = createMcpHandler(
    () => createMcpServer(ownedStore),
    { responseMode: "json" },
  );

  const startTime = Date.now();
  const quiet = options?.quiet ?? false;

  /** Format timestamp for request logging */
  function ts(): string {
    return new Date().toISOString().slice(11, 23); // HH:mm:ss.SSS
  }

  type JsonRpcLikeBody = {
    method?: unknown;
    params?: {
      name?: unknown;
      arguments?: Record<string, unknown>;
    };
  };
  type RestSearchInput = {
    type?: unknown;
    query?: unknown;
  };

  /** Extract a human-readable label from a JSON-RPC body */
  function describeRequest(body: JsonRpcLikeBody): string {
    const method = typeof body.method === "string" ? body.method : "unknown";
    if (method === "tools/call") {
      const tool = body.params?.name ?? "?";
      const args = body.params?.arguments;
      // Show query string if present, truncated
      if (args?.query) {
        const q = String(args.query).slice(0, 80);
        return `tools/call ${tool} "${q}"`;
      }
      if (args?.file) return `tools/call ${tool} ${args.file}`;
      if (args?.path) return `tools/call ${tool} ${args.path}`;
      if (args?.pattern) return `tools/call ${tool} ${args.pattern}`;
      return `tools/call ${tool}`;
    }
    return method;
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

  // Helper to collect request body
  async function collectBody(req: IncomingMessage): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks).toString();
  }

  const host = options.host ?? process.env.QMD_HOST ?? "localhost";
  const originGuard = resolveOriginGuard({
    host,
    ...(options.allowedOrigins ? { allowedOrigins: options.allowedOrigins } : {}),
    ...(options.allowedHosts ? { allowedHosts: options.allowedHosts } : {}),
  });

  httpServer = createServer(async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
    const reqStart = Date.now();
    const pathname = (nodeReq.url || "/").split("?")[0];

    try {
      // DNS-rebinding screen, ahead of routing so REST /query /search are
      // covered too — they bypass the MCP transport entirely (#881).
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
        nodeRes.writeHead(403, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32003, message: `Forbidden: ${verdict.reason}` },
          id: null,
        }));
        log(`${ts()} ${nodeReq.method} ${pathname} 403 — ${verdict.reason}`);
        return;
      }

      if (pathname === "/health" && nodeReq.method === "GET") {
        const body = JSON.stringify({ status: "ok", uptime: Math.floor((Date.now() - startTime) / 1000) });
        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(body);
        log(`${ts()} GET /health (${Date.now() - reqStart}ms)`);
        return;
      }

      // REST endpoint: POST /search — structured search without MCP protocol
      // REST endpoint: POST /query (alias: /search) — structured search without MCP protocol
      if ((pathname === "/query" || pathname === "/search") && nodeReq.method === "POST") {
        return await inflight.run(async () => {
        const rawBody = await collectBody(nodeReq);
        const params = JSON.parse(rawBody) as Record<string, unknown>;

        // Validate required fields
        if (!params.searches || !Array.isArray(params.searches)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Missing required field: searches (array)" }));
          return;
        }

        // Map to internal format
        const searches = params.searches as RestSearchInput[];
        const queries: ExpandedQuery[] = searches.map((s) => ({
          type: s.type as 'lex' | 'vec' | 'hyde',
          query: String(s.query || ""),
        }));

        // Use default collections if none specified
        const effectiveCollections = Array.isArray(params.collections) ? params.collections.map(String) : defaultCollectionNames;

        const results = await ownedStore.search({
          queries,
          collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
          limit: typeof params.limit === "number" ? params.limit : 10,
          minScore: typeof params.minScore === "number" ? params.minScore : 0,
          candidateLimit: typeof params.candidateLimit === "number" ? params.candidateLimit : undefined,
          intent: typeof params.intent === "string" ? params.intent : undefined,
          rerank: typeof params.rerank === "boolean" ? params.rerank : undefined,
        });

        // Use first lex or vec query for snippet extraction
        const primaryQuery = searches.find((s) => s.type === 'lex')?.query
          || searches.find((s) => s.type === 'vec')?.query
          || searches[0]?.query || "";

        const formatted = results.map(r => {
          const { line, snippet } = extractSnippet(r.body, String(primaryQuery), 300, r.bestChunkPos, r.bestChunk.length, typeof params.intent === "string" ? params.intent : undefined);
          return {
            docid: `#${r.docid}`,
            file: `qmd://${encodeQmdPath(r.displayPath)}`,
            title: r.title,
            score: Math.round(r.score * 100) / 100,
            context: r.context,
            line,
            snippet: addLineNumbers(snippet, line),
          };
        });

        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify({ results: formatted }));
        log(`${ts()} POST /query ${params.searches.length} queries (${Date.now() - reqStart}ms)`);
        return;
        });
      }

      if (pathname === "/mcp") {
        const rawBody = nodeReq.method !== "GET" && nodeReq.method !== "HEAD"
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
        const label = parsedBody && typeof parsedBody === "object" && parsedBody !== null
          ? describeRequest(parsedBody as JsonRpcLikeBody)
          : (nodeReq.method || "GET");
        const hostHeader = typeof nodeReq.headers.host === "string" ? nodeReq.headers.host : `localhost:${port}`;
        const url = `http://${hostHeader}${pathname}`;
        const request = new Request(url, {
          method: nodeReq.method || "GET",
          headers: nodeHeadersToWeb(nodeReq),
          ...(rawBody !== undefined ? { body: rawBody } : {}),
        });
        const response = await inflight.run(async () =>
          mcpHandler.fetch(
            request,
            parsedBody !== undefined ? { parsedBody } : undefined,
          ),
        );

        nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
        nodeRes.end(Buffer.from(await response.arrayBuffer()));
        log(`${ts()} ${nodeReq.method} /mcp ${label} (${Date.now() - reqStart}ms)`);
        return;
      }

      nodeRes.writeHead(404);
      nodeRes.end("Not Found");
    } catch (err) {
      if (err instanceof ServerShuttingDownError) {
        nodeRes.writeHead(503, {
          "Content-Type": "application/json",
          "Retry-After": "1",
          "Connection": "close",
        });
        nodeRes.end(JSON.stringify({ error: "QMD server is shutting down" }));
        return;
      }
      console.error("HTTP handler error:", err);
      nodeRes.writeHead(500);
      nodeRes.end("Internal Server Error");
    }
  });

  const ownedHttp = httpServer!;
  if (options.listen) {
    await options.listen(ownedHttp, port, host);
  } else {
    await new Promise<void>((resolve, reject) => {
      ownedHttp.on("error", reject);
      ownedHttp.listen(port, host, () => resolve());
    });
  }

  const actualPort = (ownedHttp.address() as import("net").AddressInfo).port;

  let httpClosePromise: Promise<void> | undefined;
  const closeHttpServer = (): Promise<void> =>
    (httpClosePromise ??= new Promise<void>((resolve, reject) => {
      ownedHttp.close((error?: Error) => {
        if (error) reject(error);
        else resolve();
      });
    }));

  const coordinator = createShutdownCoordinator({
    closeAdmission() {
      inflight.closeAdmission();
      llm!.closeSessionAdmission();
    },
    stopServing: async () => {
      await Promise.all([
        mcpHandler.close(),
        closeHttpServer(),
      ]);
    },
    requestAbort: (reason) => llm!.requestSessionAbort(reason),
    waitForInflight: () => inflight.waitForIdle(),
    waitForLlmIdle: () => llm!.waitForSessionIdle(),
    disposeLlm: () => llm!.dispose(),
    closeStore: () => store!.close(),
    setExitCode: options.setExitCode ?? applyProcessExitCode,
    logError: logShutdownError,
    holdOpen: options.holdOpen ?? createShutdownHold,
    ...(options.hardStop ? { hardStop: options.hardStop } : {}),
  });

  const shutdown = (trigger: ShutdownTrigger = { kind: "complete", exitCode: 0 }) =>
    coordinator.shutdown(trigger);
  const stop = () => shutdown({ kind: "complete", exitCode: 0 });

  log(`QMD MCP server listening on http://${host}:${actualPort}/mcp`);
  if (originGuard.disabled) {
    log("Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. Only do this behind your own authenticating proxy.");
  } else if (!originGuard.enforceHost) {
    log(`Warning: bound to ${host} with no QMD_ALLOWED_HOSTS — Host validation is off and the index is readable by anyone who can reach this port.`);
  }
  return { httpServer: ownedHttp, port: actualPort, stop, shutdown, inflight };
  } catch (error) {
    await cleanupOwnedMcpStartup({
      ownsStore,
      store,
      llm,
      closeTransport: httpServer
        ? () => new Promise<void>((resolve) => {
            httpServer!.close(() => resolve());
          })
        : undefined,
    });
    throw error;
  }
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer().catch(console.error);
}
