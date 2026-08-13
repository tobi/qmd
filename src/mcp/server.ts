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
  /** Resolves once no tracked handler runs, or after timeoutMs. Returns whether idle was reached. */
  waitForIdle(timeoutMs: number): Promise<boolean>;
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
    waitForIdle(timeoutMs: number): Promise<boolean> {
      if (active === 0) return Promise.resolve(true);
      return new Promise((resolve) => {
        const onIdle = () => {
          clearTimeout(timer);
          resolve(true);
        };
        const timer = setTimeout(() => {
          const i = waiters.indexOf(onIdle);
          if (i >= 0) waiters.splice(i, 1);
          resolve(false);
        }, timeoutMs);
        timer.unref?.();
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
  stderr?: { write(chunk: string): unknown; on?(event: "error", listener: (err: unknown) => void): unknown };
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
export function registerStdioEofShutdown(options: StdioShutdownOptions): () => Promise<void> {
  const stdin = options.stdin ?? process.stdin;
  const stderr = options.stderr ?? process.stderr;
  const setExitCode = options.setExitCode ?? ((code: number) => { process.exitCode = code; });
  const getExitCode = options.getExitCode ?? (() => (typeof process.exitCode === "number" ? process.exitCode : undefined));
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
    const step = async (name: string, run: () => void | Promise<void>): Promise<void> => {
      try {
        await run();
      } catch (error) {
        failed = true;
        safeWrite(
          `QMD Warning: ${name} failed during stdio shutdown (${error instanceof Error ? error.message : String(error)}); continuing shutdown.\n`
        );
      }
    };

    await step("server.close()", options.closeServer);
    if (options.waitForIdle) {
      await step("in-flight drain", async () => {
        const idle = await options.waitForIdle!(options.idleTimeoutMs ?? 5000);
        if (!idle) {
          safeWrite("QMD Warning: in-flight request did not settle before the shutdown deadline; continuing shutdown.\n");
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
  const onStdinEof = (): void => { void shutdown().catch(() => {}); };

  stdin.once("end", onStdinEof);
  stdin.once("close", onStdinEof);

  // The parent can die between spawn and listener registration; check the
  // stream flags after subscribing so an already-ended stdin still shuts down.
  if (stdin.readableEnded || stdin.destroyed) {
    onStdinEof();
  }

  return shutdown;
}

export async function startMcpServer(options: McpStartupOptions = {}): Promise<void> {
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
  const inflight = createInflightGate();
  // serveStdio dual-speaks 2026-07-28 and 2025-era clients on one connection
  // (opening exchange pins the era). A hand-wired StdioServerTransport would
  // stay 2025-only even on SDK 2.x.
  const handle = serveStdio(() => createMcpServer(store, inflight));

  // Follow the parent's lifecycle: when stdin reaches EOF the client is gone
  // and the server must exit instead of orphaning to PID 1 (#751). No
  // disposeLlm here — store.close() disposes this store's own LlamaCpp
  // instance, so passing the global disposeDefaultLlamaCpp would only risk
  // tearing down an unrelated instance in an embedded process.
  registerStdioEofShutdown({
    closeServer: () => handle.close(),
    waitForIdle: (timeoutMs) => inflight.waitForIdle(timeoutMs),
    closeStore: () => store.close(),
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
  options: ({ quiet?: boolean; host?: string; allowedOrigins?: string[]; allowedHosts?: string[] } & McpStartupOptions) = {},
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

  // Pre-fetch default collection names for REST endpoint
  const defaultCollectionNames = await store.getDefaultCollectionNames();

  // Official 2026-07-28 HTTP entry: one factory, per-request instance, JSON
  // responses (matches the previous enableJsonResponse: true). Dual-speaks
  // 2025-era traffic statelessly by default (`legacy: "stateless"`).
  const mcpHandler = createMcpHandler(
    () => createMcpServer(store),
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

  const httpServer = createServer(async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
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

        const results = await store.search({
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
        const response = await mcpHandler.fetch(
          request,
          parsedBody !== undefined ? { parsedBody } : undefined,
        );

        nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
        nodeRes.end(Buffer.from(await response.arrayBuffer()));
        log(`${ts()} ${nodeReq.method} /mcp ${label} (${Date.now() - reqStart}ms)`);
        return;
      }

      nodeRes.writeHead(404);
      nodeRes.end("Not Found");
    } catch (err) {
      console.error("HTTP handler error:", err);
      nodeRes.writeHead(500);
      nodeRes.end("Internal Server Error");
    }
  });

  await new Promise<void>((resolve, reject) => {
    httpServer.on("error", reject);
    httpServer.listen(port, host, () => resolve());
  });

  const actualPort = (httpServer.address() as import("net").AddressInfo).port;

  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    await mcpHandler.close();
    httpServer.close();
    await store.close();
  };

  process.on("SIGTERM", async () => {
    console.error("Shutting down (SIGTERM)...");
    await stop();
    process.exit(0);
  });
  process.on("SIGINT", async () => {
    console.error("Shutting down (SIGINT)...");
    await stop();
    process.exit(0);
  });

  log(`QMD MCP server listening on http://${host}:${actualPort}/mcp`);
  if (originGuard.disabled) {
    log("Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. Only do this behind your own authenticating proxy.");
  } else if (!originGuard.enforceHost) {
    log(`Warning: bound to ${host} with no QMD_ALLOWED_HOSTS — Host validation is off and the index is readable by anyone who can reach this port.`);
  }
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer().catch(console.error);
}
