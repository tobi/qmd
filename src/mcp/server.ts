/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Speaks MCP spec 2026-07-28 (stateless, no initialize handshake) and dual-speaks
 * 2025-era clients via the official SDK entries (`serveStdio` / `createMcpHandler`).
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "url";
import { createMcpHandler, McpServer, ResourceTemplate } from "@modelcontextprotocol/server";
import type { CallToolResult } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import { existsSync } from "fs";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  getDefaultDbPath,
  DEFAULT_MULTI_GET_MAX_BYTES,
  parseMetadataFilter,
  parseMetadataMatch,
  type QMDStore,
  type ExpandedQuery,
  type IndexStatus,
  type DocumentMetadata,
  type MetadataFilter,
  type MetadataMatch,
  type MetadataKeyOverview,
  type ListMetadataResult,
  MetadataBindingBudgetError,
  MetadataOptionError,
} from "../index.js";
import { getConfigPath, loadConfig } from "../collections.js";
import { DEFAULT_EMBED_MODEL_URI, DEFAULT_GENERATE_MODEL_URI, DEFAULT_RERANK_MODEL_URI } from "../llm.js";
import { decideLocalConfigGate, type SensitiveSnapshot } from "../trust.js";
import { formatMetadataKeySummaries } from "../metadata-format.js";
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
  context?: string | null;  // Folder context (omitted per-row in compact mode)
  metadata?: DocumentMetadata;  // Indexed qmd.metadata (present when non-empty)
  line: number;   // Absolute line in source markdown
  snippet: string;
};

/**
 * Validate an untrusted `filter` argument through the shared runtime
 * validator. Returns the parse error message when invalid.
 */
function validateFilterArgument(filter: unknown): { filter?: MetadataFilter; error?: string } {
  if (filter === undefined) return {};
  try {
    return { filter: parseMetadataFilter(filter) };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/** Same grammar as the filter, validated against metadata entries for discovery. */
function validateMatchArgument(match: unknown): { match?: MetadataMatch; error?: string } {
  if (match === undefined) return {};
  try {
    return { match: parseMetadataMatch(match) };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/** Pick the named fields of a JSON body that must be numbers when present. */
function readNumberFields(params: Record<string, unknown>, names: string[]): { values: Record<string, number>; error?: string } {
  const values: Record<string, number> = {};
  for (const name of names) {
    const value = params[name];
    if (value === undefined) continue;
    if (typeof value !== "number") return { values, error: `Invalid field: ${name} (must be a number)` };
    values[name] = value;
  }
  return { values };
}

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
    metadataKeyCount: number;
    metadataKeys: MetadataKeyOverview[];
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
/**
 * Opt-in compact mode: when QMD_MCP_COMPACT=1, tool responses return
 * structuredContent only and skip the duplicated human-readable text block
 * (dual-render). Default behaviour is unchanged.
 */
export function isCompactMode(): boolean {
  return process.env.QMD_MCP_COMPACT === "1";
}

/**
 * Build a tool response honouring compact mode. Non-compact keeps the
 * dual-render (text + structuredContent) that all clients understand.
 */
export function toolResponse(structured: CallToolResult["structuredContent"] & object, text: string): CallToolResult {
  // content: [] keeps the SDK type happy; structuredContent carries the payload.
  if (isCompactMode()) return { content: [], structuredContent: structured };
  return { content: [{ type: "text", text }], structuredContent: structured };
}

/**
 * Compact query responses: per-row folder `context` is highly repetitive
 * (usually one value per collection), so hoist it to a single top-level
 * `collection_context` — a string when uniform, an array when distinct.
 */
export function hoistCollectionContext<T extends { context?: string | null }>(
  rows: T[],
): { collection_context?: string | string[]; results: Omit<T, "context">[] } {
  const results = rows.map(({ context: _drop, ...rest }) => rest);
  const contexts = [...new Set(rows.map(r => r.context).filter((c): c is string => !!c))];
  if (contexts.length === 0) return { results };
  return { collection_context: contexts.length === 1 ? contexts[0]! : contexts, results };
}

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
  // Instructions use index facts, not metadata overviews. Do not rank keys on
  // initialization or on each stateless HTTP request just to discard them.
  const status = store.internal.getStatusSummary();
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
 * Cache built instructions per store. The HTTP transport builds a fresh
 * McpServer per request (sessionless MCP 2026-07-28), and buildInstructions()
 * runs a full index-status aggregation, so rebuilding it per request puts that
 * scan in front of every call. Concurrent builds share one in-flight promise,
 * and entries expire after a short TTL so index updates from other processes
 * still surface without a server restart.
 */
const INSTRUCTIONS_CACHE_TTL_MS = 60_000;
const instructionsCache = new WeakMap<QMDStore, { promise: Promise<string>; builtAt: number }>();

function getInstructions(store: QMDStore): Promise<string> {
  const cached = instructionsCache.get(store);
  // A negative age means the clock stepped back since the build; that entry is
  // not known to be fresh, so it is rebuilt rather than kept past the TTL.
  const age = cached ? Date.now() - cached.builtAt : -1;
  if (cached && age >= 0 && age < INSTRUCTIONS_CACHE_TTL_MS) {
    return cached.promise;
  }
  const promise = buildInstructions(store);
  instructionsCache.set(store, { promise, builtAt: Date.now() });
  // Drop failed builds so the next initialize retries instead of replaying
  // the cached rejection for the rest of the TTL window.
  promise.catch(() => {
    if (instructionsCache.get(store)?.promise === promise) {
      instructionsCache.delete(store);
    }
  });
  return promise;
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
      instructions: await getInstructions(store),
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
        filter: z.record(z.string(), z.unknown()).optional().describe(
          "Metadata filter (recursive JSON AST). Every returned result satisfies it. " +
          "Nodes are operator-discriminated: logical groups {operator:'and'|'or', operands:[...]}, " +
          "negation {operator:'not', operand:{...}}, and conditions {field, operator, value} with " +
          "operators eq/ne/gt/gte/lt/lte (comparison), in/nin/all (membership), contains/prefix/suffix (text), " +
          "type (value is 'string'|'number'|'boolean'), exists (presence). " +
          "Values are typed exactly (no coercion); missing keys do not match ne/nin. " +
          "Conditions with a string value may add caseInsensitive:true (ASCII folding). " +
          "Example: {\"operator\":\"and\",\"operands\":[{\"field\":\"topics\",\"operator\":\"all\",\"value\":[\"typescript\"]}," +
          "{\"field\":\"status\",\"operator\":\"ne\",\"value\":\"draft\"}]}"
        ),
        intent: z.string().optional().describe(
          "Background context to disambiguate the query. Example: query='performance', intent='web page load times and Core Web Vitals'. Does not search on its own."
        ),
        rerank: z.boolean().optional().default(true).describe(
          "Rerank results using LLM (default: true). Set to false for faster results on CPU-only machines."
        ),
      }),
    },
    track(async ({ query, searches, limit, minScore, candidateLimit, collections, filter, intent, rerank }, ctx) => {
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

      const filterValidation = validateFilterArgument(filter);
      if (filterValidation.error) {
        return {
          content: [{ type: "text" as const, text: `Error: ${filterValidation.error}` }],
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

      // The SDK's per-request abort signal is cancelled when the client goes
      // away (or the transport closes), so remote embedding/rerank stages stop
      // instead of finishing work nobody can read.
      const results = await store.search({
        ...searchOptions,
        signal: ctx.mcpReq.signal,
        collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
        filter: filterValidation.filter,
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

      const compact = isCompactMode();
      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.body, primaryQuery, 300, r.bestChunkPos, r.bestChunk.length, intent);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          // Compact: per-row context is hoisted to a top-level collection_context
          ...(compact ? {} : { context: r.context }),
          ...(Object.keys(r.metadata).length > 0 ? { metadata: r.metadata } : {}),
          line,
          snippet: addLineNumbers(snippet, line),
        };
      });

      const structured = compact ? hoistCollectionContext(filtered) : { results: filtered };
      return toolResponse(structured, formatSearchSummary(filtered, primaryQuery));
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
        if (col.metadataKeys.length > 0) {
          const keyLabels = col.metadataKeys.map(overview => `${overview.key} (${overview.types.join(" | ")})`);
          const hiddenKeys = col.metadataKeyCount - col.metadataKeys.length;
          summary.push(`      metadata keys: ${keyLabels.join(", ")}${hiddenKeys > 0 ? `, +${hiddenKeys} more` : ""}`);
        }
      }

      if (status.collections.some(col => col.metadataKeys.length > 0)) {
        summary.push(`  Metadata: call the 'metadata' tool to see values and counts before writing a 'filter'`);
      }

      return toolResponse(status, summary.join('\n'));
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_metadata (Metadata discovery)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "metadata",
    {
      title: "Metadata Discovery",
      description: `Discover which metadata keys exist, what types they hold, and how many documents share each value, so you can write a \`filter\` for the query tool from what is indexed instead of guessing.

## Mental model

\`filter\` selects WHICH documents are counted. \`match\` selects WHICH metadata entries of those documents are reported. Both take the same recursive AST as the query tool's \`filter\`. A condition tests one \`field\` of the record under evaluation: for \`filter\` the record is a document and \`field\` names one of its metadata keys, for \`match\` the record is a metadata entry and \`field\` is \`"key"\` or \`"value"\`. Every operator applies to a match except \`exists\` and \`all\`, which have no meaning for a single entry.

| match | Question answered |
|---|---|
| (none) | Which keys exist, with a window of values each |
| \`{"field":"key","operator":"eq","value":"topics"}\` | Everything about one key |
| \`{"field":"key","operator":"prefix","value":"mem-"}\` | A family of keys |
| \`{"field":"value","operator":"eq","value":"docs-team"}\` | Which keys hold this value |
| \`{"operator":"and","operands":[{"field":"key","operator":"eq","value":"priority"},{"field":"value","operator":"gte","value":3}]}\` | Values of one key above a threshold |

Add \`filter\` to any of these to see what remains after narrowing. The result then reports \`filteredDocuments\`, and every coverage count is measured against that population.

## Reading the result

Counts are documents, not values. \`keys\` is one window of \`totalKeys\` in coverage order, and \`remainingKeys\` is the exact count after it. Each key splits by type: metadata is validated per document, so a key can hold numbers in some files and strings in others, and each type reports its own \`documents\`, \`distinctValues\`, a window of \`values\`, \`remainingValues\`, and, for numbers, \`range\` (min, median, max). Page keys with \`keyOffset\` and values with \`valueOffset\`, which applies to every key in the result and so reads best after \`match\` narrows to one key.

Every value reported here can be matched with \`{field: '<metadata-key>', operator: 'eq', value}\` under the same collections.`,
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: z.object({
        collections: z.array(z.string()).optional().describe("Restrict to these collections (default: the same collections query searches)"),
        match: z.record(z.string(), z.unknown()).optional().describe(
          "Report only metadata entries matching this condition. Same recursive AST as 'filter', evaluated per entry: " +
          "a condition's field is 'key' (the entry's key name) or 'value' (its value). Default: every entry. " +
          "Example: {\"field\":\"key\",\"operator\":\"eq\",\"value\":\"topics\"}"
        ),
        filter: z.record(z.string(), z.unknown()).optional().describe(
          "Count only documents matching this metadata filter. Same recursive AST as the query tool's 'filter'. " +
          "Example: {\"field\":\"status\",\"operator\":\"eq\",\"value\":\"published\"}"
        ),
        keyLimit: z.number().int().positive().optional().default(50).describe("Keys reported (default: 50). 'remainingKeys' says how many follow the window"),
        keyOffset: z.number().int().nonnegative().optional().default(0).describe("Keys skipped before the window, in report order (default: 0)"),
        valueLimit: z.number().int().positive().optional().default(10).describe("Values reported per key and type (default: 10). 'remainingValues' says how many follow the window"),
        valueOffset: z.number().int().nonnegative().optional().default(0).describe("Values skipped per key and type before the window, in 'sort' order (default: 0)"),
        sort: z.enum(["count", "value"]).optional().default("count").describe("Order values by document count descending (default) or by value ascending"),
        minCount: z.number().int().positive().optional().default(1).describe("Hide values held by fewer documents than this (default: 1)"),
      }),
    },
    track(async ({ collections, match, filter, keyLimit, keyOffset, valueLimit, valueOffset, sort, minCount }) => {
      const matchValidation = validateMatchArgument(match);
      const filterValidation = validateFilterArgument(filter);
      const validationError = matchValidation.error ?? filterValidation.error;
      if (validationError) {
        return {
          content: [{ type: "text" as const, text: `Error: ${validationError}` }],
          isError: true,
        };
      }

      const effectiveCollections = collections ?? defaultCollectionNames;
      let result: ListMetadataResult;
      try {
        result = await store.listMetadata({
          collection: effectiveCollections.length > 0 ? effectiveCollections : undefined,
          match: matchValidation.match,
          filter: filterValidation.filter,
          keyLimit,
          keyOffset,
          valueLimit,
          valueOffset,
          sort,
          minCount,
        });
      } catch (err) {
        if (!(err instanceof MetadataBindingBudgetError)) throw err;
        return {
          content: [{ type: "text" as const, text: `Error: ${err.message}` }],
          isError: true,
        };
      }

      const text = formatMetadataKeySummaries(result, {
        showCollections: effectiveCollections.length !== 1,
        valueWindowHint: "a higher 'valueLimit' or a 'valueOffset'",
        keyWindowHint: "a higher 'keyLimit' or a 'keyOffset'",
        keyOffset,
        keyOffsetLabel: "keyOffset",
        emptyMessage: "No metadata matches. Call without match/filter to see which keys exist, or check the status tool for collections with metadata.",
      });

      return toolResponse(result, text);
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

// =============================================================================
// Lifecycle: serving -> draining -> stopped
// =============================================================================

/**
 * Admission state of an MCP transport.
 *
 * The transition order is the safety property both transports depend on, so it
 * is modelled explicitly instead of being implied by the order of teardown
 * calls: work may only touch the shared store/llm while `serving`; `draining`
 * refuses new work while already-admitted work settles; `stopped` is terminal
 * and means the store has been closed. Teardown MUST go
 * `serving -> draining -> stopped` and MUST NOT close the store before it
 * reaches `draining`, or an admitted handler keeps running against a closed
 * database.
 */
export type McpLifecycleState = "serving" | "draining" | "stopped";

export type McpLifecycle = {
  state(): McpLifecycleState;
  /**
   * Admit one unit of work, or return null once admission is closed
   * (`draining`/`stopped`). The returned release function is idempotent.
   * Callers MUST NOT touch the store when admission is refused.
   */
  admit(): (() => void) | null;
  /** `serving -> draining`. False when admission was already closed. */
  beginDrain(): boolean;
  /** `draining -> stopped`. False when called from `serving` or `stopped`. */
  finishTeardown(): boolean;
  /**
   * Resolves once no admitted work runs -- or, when `timeoutMs` is given,
   * after at most that long; the boolean reports whether idle was reached.
   * Omitting the timeout waits until idle, which is the graceful default: a
   * deadline is what interrupts work that is still using the store, so only a
   * caller that asks for one gets one.
   */
  waitForIdle(timeoutMs?: number): Promise<boolean>;
};

export function createMcpLifecycle(): McpLifecycle {
  let state: McpLifecycleState = "serving";
  let active = 0;
  const waiters: Array<() => void> = [];

  return {
    state: () => state,

    admit() {
      if (state !== "serving") return null;
      active += 1;
      let released = false;
      return () => {
        if (released) return;
        released = true;
        active -= 1;
        if (active !== 0) return;
        // Drain waiters are one-shot: they resolve now and are forgotten.
        while (waiters.length > 0) waiters.shift()!();
      };
    },

    beginDrain() {
      if (state !== "serving") return false;
      state = "draining";
      return true;
    },

    finishTeardown() {
      if (state !== "draining") return false;
      state = "stopped";
      return true;
    },

    waitForIdle(timeoutMs?: number): Promise<boolean> {
      if (active === 0) return Promise.resolve(true);
      const { promise, resolve } = Promise.withResolvers<boolean>();
      let timer: ReturnType<typeof setTimeout> | undefined;
      const onIdle = () => {
        clearTimeout(timer);
        resolve(true);
      };
      // Without a timeout the waiter stays registered and the release that
      // makes the count zero resolves it; nothing can expire it.
      if (timeoutMs !== undefined) {
        timer = setTimeout(() => {
          const i = waiters.indexOf(onIdle);
          if (i >= 0) waiters.splice(i, 1);
          resolve(false);
        }, timeoutMs);
        timer.unref?.();
      }
      waiters.push(onIdle);
      return promise;
    },
  };
}

/**
 * Default graceful-shutdown deadline.
 *
 * Both transports bound their teardown with it: admitted work gets this long
 * to finish, then the remaining requests are aborted, the sockets are severed
 * and the stop call reports a {@link McpShutdownTimeoutError}. Cleanup keeps
 * going in the background -- the store is never closed underneath an operation
 * that is still running -- so "timed out" means "reported, and abandoned to
 * finish safely", not "the listener or store was torn down under live work".
 *
 * `shutdownDrainTimeoutMs` (HTTP) and `idleTimeoutMs` (stdio) override it.
 */
export const MCP_SHUTDOWN_TIMEOUT_MS = 5000;

/** Which teardown step was running when the deadline expired. */
export type McpShutdownPhase = "drain" | "transport" | "listener" | "store";

const PHASE_SUBJECT: Record<McpShutdownPhase, string> = {
  drain: "admitted requests to settle",
  transport: "the MCP transport to close",
  listener: "the HTTP listener to close",
  store: "the store to close",
};

/**
 * Rejection of a stop/EOF shutdown that did not finish within its deadline.
 *
 * Typed so a caller can tell "still working, reported, cleanup continues"
 * apart from a failed teardown step: the abandoned work is aborted but the
 * store stays open until it actually settles.
 */
export class McpShutdownTimeoutError extends Error {
  /** Stable discriminator for callers that cannot use `instanceof`. */
  readonly code: "MCP_SHUTDOWN_TIMEOUT";
  /** The step that outlasted the deadline. */
  readonly phase: McpShutdownPhase;
  /** The deadline that expired, in milliseconds. */
  readonly timeoutMs: number;

  constructor(phase: McpShutdownPhase, timeoutMs: number) {
    super(`QMD MCP shutdown exceeded its ${timeoutMs} ms deadline while waiting for ${PHASE_SUBJECT[phase]}`);
    this.name = "McpShutdownTimeoutError";
    this.code = "MCP_SHUTDOWN_TIMEOUT";
    this.phase = phase;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Abort reason for an admitted request whose client went away (or whose
 * shutdown deadline expired). Distinct from a normal completed close so a
 * route can tell "nobody is listening any more" from "the response is done".
 */
export class McpRequestAbortedError extends Error {
  readonly code: "MCP_REQUEST_ABORTED";

  constructor(message = "The MCP client disconnected before its response completed") {
    super(message);
    this.name = "McpRequestAbortedError";
    this.code = "MCP_REQUEST_ABORTED";
  }
}

/**
 * One deadline for a whole teardown. Every step races the same expiry, so the
 * request drain, the transport close, the listener close and the store close
 * share one budget instead of each getting its own fresh one.
 */
export type ShutdownDeadline = {
  /**
   * The shared expiry promise, rejecting with a {@link McpShutdownTimeoutError}
   * naming `phase` (the step in progress when the timer fires).
   */
  at(phase: McpShutdownPhase): Promise<never>;
  /** Milliseconds left before the deadline; 0 once it passed. */
  remainingMs(): number;
  /** Clears the timer so a finished teardown never holds the process open. */
  dispose(): void;
};

export function createShutdownDeadline(timeoutMs: number = MCP_SHUTDOWN_TIMEOUT_MS): ShutdownDeadline {
  const budget = Number.isFinite(timeoutMs) && timeoutMs >= 0 ? Math.floor(timeoutMs) : MCP_SHUTDOWN_TIMEOUT_MS;
  const expiresAt = Date.now() + budget;
  let phase: McpShutdownPhase = "drain";
  let timer: ReturnType<typeof setTimeout> | undefined;

  // Referenced, deliberately not unref'd: the deadline is the one thing that
  // must fire even when the only other pending work is a promise that never
  // settles. An unref'd timer would let a process whose loop has nothing else
  // queued exit before the timeout can be reported. dispose() clears it as
  // soon as teardown finishes, so it never delays a healthy exit.
  const { promise: expiry, reject: expire } = Promise.withResolvers<never>();
  timer = setTimeout(() => expire(new McpShutdownTimeoutError(phase, budget)), budget);
  // Steps race this only while one is in progress; claiming the rejection here
  // keeps a deadline firing between steps from becoming an unhandled rejection.
  void expiry.catch(() => {});

  return {
    at(next) {
      phase = next;
      return expiry;
    },
    remainingMs: () => Math.max(0, expiresAt - Date.now()),
    dispose: () => {
      clearTimeout(timer);
      timer = undefined;
    },
  };
}

/**
 * The irreversible steps of one transport teardown, in the lifecycle order the
 * safety properties depend on:
 *
 *   beginDrain -> beginListenerClose -> drain -> closeTransport ->
 *   listenerClosed -> closeStore -> finishTeardown
 *
 * `closeTransport` and `closeStore` MUST only run once no admitted work is
 * executing: closing the transport aborts the exchanges it still tracks, and
 * closing the store disposes the LLM and the database underneath them.
 */
export type McpTeardownSteps = {
  /** `serving -> draining`; false when the lifecycle was already drained. */
  beginDrain: () => boolean;
  /** Resolves true once no admitted work runs. Omit the timeout to wait forever. */
  waitForIdle: (timeoutMs?: number) => Promise<boolean>;
  /** Tells admitted requests their result can no longer be delivered. */
  abortWork: (reason: McpShutdownTimeoutError) => void;
  /** Ends sockets a stalled exchange or a half-sent body is parked on. */
  severConnections: () => void;
  /** Stops accepting connections; `listenerClosed` reports the completion. */
  beginListenerClose: () => void;
  /** Resolves once the listener closed (every connection ended). */
  listenerClosed: () => Promise<void>;
  /** Closes the MCP transport; only valid once the drain is idle. */
  closeTransport: () => Promise<void>;
  /** Closes the store, which also disposes the LLM it owns. */
  closeStore: () => Promise<void>;
  /** `draining -> stopped`, only once the store actually closed. */
  finishTeardown: () => boolean;
};

export type McpTeardownOptions = {
  /** Overrides {@link MCP_SHUTDOWN_TIMEOUT_MS}. */
  timeoutMs?: number;
  /** Where step failures and the timeout are reported. */
  log?: (message: string) => void;
};

export type McpTeardown = {
  /**
   * Starts the teardown on first call; every caller shares one promise.
   * Resolves when the server stopped cleanly and rejects with a
   * {@link McpShutdownTimeoutError} at the deadline -- in which case cleanup
   * continues in the background.
   */
  stop: () => Promise<void>;
  /**
   * Resolves only when every teardown step actually finished (or could not be
   * retried), including a cleanup that outlived a reported timeout. Never
   * rejects. A hung store keeps it pending: "stopped" is never claimed early.
   */
  closed: Promise<void>;
};

/**
 * Tear down one transport under a single deadline.
 *
 * Until the deadline, work admitted before `beginDrain()` may finish: the
 * listener stops accepting and the drain waits. At the deadline the caller of
 * `stop()` gets a {@link McpShutdownTimeoutError} immediately, while the
 * remaining cleanup is *abandoned into the background*: admitted requests are
 * aborted (their results are undeliverable), the sockets are severed, and then
 * the unbounded wait for idle, the transport close, the listener close and the
 * store close run in the same safe order. The store is therefore never closed
 * under an active operation, and `finishTeardown()` is only reached once
 * `closeStore()` actually resolved.
 */
export function runMcpTeardown(steps: McpTeardownSteps, options: McpTeardownOptions = {}): McpTeardown {
  const timeoutMs = options.timeoutMs ?? MCP_SHUTDOWN_TIMEOUT_MS;
  const log = options.log ?? (() => {});
  const failures: unknown[] = [];
  const closed = Promise.withResolvers<void>();
  let storeClosed = false;

  const record = (name: string, error: unknown): void => {
    failures.push(error);
    log(
      `QMD Warning: ${name} failed during shutdown (${
        error instanceof Error ? error.message : String(error)
      }); continuing shutdown.`
    );
  };

  // One-shot steps: the graceful path and an abandoned cleanup share these, so
  // a step whose promise never settles is awaited again, never started twice.
  const once = (run: () => Promise<void>): (() => Promise<void>) => {
    let started: Promise<void> | null = null;
    return () => (started ??= Promise.resolve().then(run));
  };
  const closeTransport = once(() => steps.closeTransport());
  const closeStore = once(async () => {
    await steps.closeStore();
    storeClosed = true;
  });
  let listenerStarted: Promise<void> | null = null;
  const listenerClosed = (): Promise<void> => (listenerStarted ??= Promise.resolve().then(() => steps.listenerClosed()));

  const finish = (): void => {
    if (!storeClosed) return;
    try {
      steps.finishTeardown();
    } catch (error) {
      record("lifecycle teardown", error);
    }
  };

  /**
   * Run one step inside the shared deadline. A timeout propagates (it decides
   * to abandon); any other failure is recorded and teardown carries on -- the
   * store must still close.
   */
  const bounded = async (
    deadline: ShutdownDeadline,
    name: string,
    phase: McpShutdownPhase,
    run: () => Promise<void> | void,
  ): Promise<void> => {
    let work: Promise<void>;
    try {
      work = Promise.resolve().then(run);
    } catch (error) {
      record(name, error);
      return;
    }
    try {
      await Promise.race([work, deadline.at(phase)]);
    } catch (error) {
      if (error instanceof McpShutdownTimeoutError) throw error;
      record(name, error);
    }
  };

  /** Report the timeout now, then finish the same chain safely in the background. */
  const abandon = (timeout: McpShutdownTimeoutError): void => {
    log(`QMD Warning: ${timeout.message}; aborting the remaining work and finishing the shutdown in the background.`);
    try {
      steps.abortWork(timeout);
    } catch (error) {
      record("aborting admitted work", error);
    }
    try {
      steps.severConnections();
    } catch (error) {
      record("severing connections", error);
    }

    void (async () => {
      try {
        // Unbounded on purpose: the store MUST NOT close while abandoned work
        // still runs through it. If that work never settles, cleanup stops
        // here with the store open; only the standalone process's own
        // non-zero exit ends the process.
        try {
          await steps.waitForIdle();
        } catch (error) {
          record("in-flight drain", error);
        }
        await closeTransport().catch((error) => record("transport close", error));
        await listenerClosed().catch((error) => record("listener close", error));
        await closeStore().catch((error) => record("store close", error));
        finish();
      } finally {
        closed.resolve();
      }
    })();
  };

  const graceful = async (deadline: ShutdownDeadline): Promise<void> => {
    steps.beginDrain();

    // Stop accepting before waiting: nothing can enter behind the drain.
    try {
      steps.beginListenerClose();
    } catch (error) {
      record("listener close", error);
    }

    // The only window in which work admitted before beginDrain() may run. The
    // lifecycle's own timer is unref'd, so the referenced shared deadline is
    // what guarantees the bound is observed.
    const idle = await Promise.race([
      steps.waitForIdle(deadline.remainingMs()),
      deadline.at("drain"),
    ]);
    if (!idle) throw new McpShutdownTimeoutError("drain", timeoutMs);

    await bounded(deadline, "transport close", "transport", () => closeTransport());
    await bounded(deadline, "listener close", "listener", () => listenerClosed());
    await bounded(deadline, "store close", "store", () => closeStore());
    finish();
    if (failures.length > 0) throw failures[0];
  };

  let stopped: Promise<void> | null = null;
  const stop = (): Promise<void> => {
    if (stopped) return stopped;
    // The one deadline starts with the first stop(), never at construction: a
    // server that has been up longer than its budget still gets the whole
    // budget.
    const deadline = createShutdownDeadline(timeoutMs);
    stopped = (async () => {
      try {
        await graceful(deadline);
      } catch (error) {
        if (error instanceof McpShutdownTimeoutError) {
          abandon(error);
          throw error;
        }
        throw error;
      } finally {
        deadline.dispose();
      }
    })();
    // A caller may fire-and-forget stop() (a signal handler that exits next);
    // claiming the rejection here keeps it from surfacing as unhandled.
    void stopped.catch(() => {});
    void stopped.then(
      () => closed.resolve(),
      (error) => {
        // A reported timeout keeps `closed` pending until the abandoned
        // cleanup finishes; any other outcome has nothing left to run.
        if (!(error instanceof McpShutdownTimeoutError)) closed.resolve();
      },
    );
    return stopped;
  };

  return { stop, closed: closed.promise };
}

/** Where the standalone termination policy reports and how it ends the process. */
export type McpTermination = {
  /** Ends the process (`process.exit` standalone, a spy in tests). */
  exit: (code: number) => void;
  /** Where the reason for a non-zero exit is reported. */
  log: (message: string) => void;
};

/**
 * Exit-code policy for a standalone MCP server, called from its signal
 * handler.
 *
 * A clean stop exits 0. A reported timeout exits 1 *immediately*: work is
 * still running, the process is the only thing that can end it, and the
 * abandoned cleanup cannot finish before that exit. Any other teardown failure
 * also exits 1, because the store may not have closed. Embedded hosts do not
 * use this — they own their own lifetime.
 */
export async function terminateMcpServer(stop: () => Promise<void>, termination: McpTermination): Promise<void> {
  try {
    await stop();
    termination.exit(0);
  } catch (error) {
    if (error instanceof McpShutdownTimeoutError) {
      termination.log(`QMD Warning: ${error.message}; work was still running, exiting.`);
    } else {
      termination.log(
        `QMD Warning: shutdown failed (${error instanceof Error ? error.message : String(error)}); exiting.`,
      );
    }
    termination.exit(1);
  }
}

/**
 * Counts running request handlers so shutdown can wait for them to settle
 * before tearing down their llm/store dependencies, and refuses handlers the
 * transport dispatches after admission closed. The SDK aborts in-flight
 * request controllers on close, but qmd's handlers finish their current
 * store/llm work rather than observing the signal mid-operation.
 *
 * The gate owns the lifecycle state, so the admission barrier no longer
 * depends on the caller remembering to close the transport first: a handler
 * that arrives after `beginDrain()` throws instead of touching a store that
 * shutdown is about to close (the client that sent it is gone in that window).
 */
export type InflightGate = McpLifecycle & {
  /** Wraps a handler so the gate admits and counts it while it runs. */
  track<T extends (...args: never[]) => unknown>(fn: T): T;
};

export function createInflightGate(lifecycle: McpLifecycle = createMcpLifecycle()): InflightGate {
  return {
    state: () => lifecycle.state(),
    admit: () => lifecycle.admit(),
    beginDrain: () => lifecycle.beginDrain(),
    finishTeardown: () => lifecycle.finishTeardown(),
    waitForIdle: (timeoutMs?: number) => lifecycle.waitForIdle(timeoutMs),

    track(fn) {
      const wrapped = async (...args: never[]) => {
        const release = lifecycle.admit();
        if (!release) {
          throw new Error("QMD MCP server is shutting down; request refused");
        }
        try {
          return await fn(...args);
        } finally {
          release();
        }
      };
      return wrapped as typeof fn;
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
  /**
   * Handler gate shared with the transport (see InflightGate). When given,
   * shutdown closes its admission — `serving -> draining` — before
   * closeServer(), so a request the closing transport still dispatches is
   * refused instead of touching a store that teardown is about to close, and
   * marks teardown complete (`draining -> stopped`) after closeStore().
   * Also supplies waitForIdle when the caller does not inject one.
   */
  lifecycle?: McpLifecycle;
  /**
   * Waits for in-flight handlers to settle (see InflightGate.waitForIdle).
   * Call it without a timeout for the unbounded post-timeout wait.
   */
  waitForIdle?: (timeoutMs?: number) => Promise<boolean>;
  /** Deadline for the whole teardown. Defaults to {@link MCP_SHUTDOWN_TIMEOUT_MS}. */
  idleTimeoutMs?: number;
  /** Defaults to process.stdin. */
  stdin?: StdioShutdownStdin;
  /**
   * Records the outcome. Defaults to a no-op: an embedded host owns its exit
   * status, so only the standalone stdio entry passes a setter (writing
   * `process.exitCode`) and, with it, `exitOnTimeout`.
   */
  setExitCode?: (code: number) => void;
  /** Defaults to reading process.exitCode. */
  getExitCode?: () => number | undefined;
  /** Defaults to process.stderr. */
  stderr?: { write(chunk: string): unknown; on?(event: "error", listener: (err: unknown) => void): unknown };
  /**
   * Called after the outcome was recorded as 1 because the deadline expired
   * with work still running, right before the shutdown promise rejects. Only
   * the standalone stdio entry passes `process.exit(1)`: after EOF nothing can
   * consume the abandoned work, so an orphaned process would only leak (the
   * very failure #751 is about). Embedded callers leave it unset, catch the
   * rejection, and keep their own lifetime.
   */
  exitOnTimeout?: (error: McpShutdownTimeoutError) => void;
};

/**
 * The idempotent EOF shutdown plus the promise that only settles once every
 * teardown step actually finished — including cleanup abandoned after a
 * reported timeout. `closed` never rejects; the callable rejects with a
 * {@link McpShutdownTimeoutError} when the deadline expired.
 */
export type StdioShutdown = (() => Promise<void>) & { closed: Promise<void> };

/**
 * Shut the stdio MCP server down when stdin reaches EOF (#751).
 *
 * The SDK's StdioServerTransport subscribes to stdin "data"/"error" only and
 * never notices "end"/"close". When the parent MCP client dies, nothing tears
 * the process down: the warm llama.cpp model's native handles keep the event
 * loop alive, so the server reparents to PID 1, leaks RAM, and keeps the
 * SQLite index open. stdin EOF means the client is gone, so this treats it as
 * a disconnect: no new requests are accepted and nobody is left to read a
 * response — but handlers that are already running get the same bounded
 * window the HTTP transport grants (waitForIdle) before their llm/store
 * dependencies are torn down.
 *
 * Teardown order matters, and it is the lifecycle's transitions.
 * `serving -> draining` closes handler admission *and* the transport (either
 * alone is not enough: the transport can still hand us a request it read just
 * before closing), then in-flight handlers get their bounded window, then the
 * store is closed last — which disposes the store's own llama.cpp instance and
 * then the database, so the dispose path cannot hit an already-closed DB —
 * then `draining -> stopped`. (disposeLlm is an optional extra step for
 * callers that own a separate instance; the MCP store does not.) Failures are
 * logged best-effort (the parent's death may have closed stderr too) and do
 * not stop the remaining steps.
 *
 * One deadline (default {@link MCP_SHUTDOWN_TIMEOUT_MS}) bounds the whole
 * chain: the transport close, the drain, the llama disposal and the store
 * close. When it expires, the shutdown promise rejects with a
 * {@link McpShutdownTimeoutError} and the remaining cleanup continues in the
 * background — unbounded when it has to be, because the store MUST NOT close
 * underneath an operation that is still running, and the lifecycle is only
 * marked stopped once it really closed. `closed` is what reports that finish.
 *
 * This function never calls process.exit itself, and it only writes
 * `process.exitCode` through a caller-supplied `setExitCode`. The standalone
 * entry supplies both (setter plus `exitOnTimeout`), because a process that
 * outlived its deadline with work remaining can only be ended by itself;
 * writing the code rather than calling process.exit() lets `beforeExit` fire
 * and node-llama-cpp's auto-dispose run before libc's static destructors —
 * process.exit() during native-addon unload has caused exit-time crashes
 * before (#59, #129; same rationale as finishSuccessfulCliCommand in the CLI).
 *
 * Returns the idempotent shutdown function: every invocation (manual, "end",
 * "close", or already-ended stdin) shares one promise. On a reported timeout
 * that promise REJECTS with {@link McpShutdownTimeoutError} while `closed`
 * keeps tracking the background cleanup; every other outcome resolves.
 * Nothing here exits the process and, unless `setExitCode` is supplied, nothing
 * touches `process.exitCode`: the standalone entry opts into both.
 */
export function registerStdioEofShutdown(options: StdioShutdownOptions): StdioShutdown {
  const stdin = options.stdin ?? process.stdin;
  const stderr = options.stderr ?? process.stderr;
  // Default to doing nothing: an embedded host owns its exit status. The
  // standalone stdio entry passes an explicit setter.
  const setExitCode = options.setExitCode ?? (() => {});
  const getExitCode = options.getExitCode ?? (() => (typeof process.exitCode === "number" ? process.exitCode : undefined));
  const timeoutMs = options.idleTimeoutMs ?? MCP_SHUTDOWN_TIMEOUT_MS;
  const lifecycle = options.lifecycle;
  const waitForIdle = options.waitForIdle
    ?? (lifecycle ? (limit?: number) => lifecycle.waitForIdle(limit) : undefined);
  const closed = Promise.withResolvers<void>();
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

    // Close admission before the transport: the transport stops dispatching
    // once closed, but a request it already read can still reach a handler
    // afterwards, and by then the store may be closing.
    lifecycle?.beginDrain();

    // Same breadcrumb style as the HTTP transport's signal handlers; also
    // gives tests an observable signal that the EOF path ran.
    safeWrite("Shutting down (stdin closed)...\n");

    const deadline = createShutdownDeadline(timeoutMs);
    const failures: unknown[] = [];
    let storeClosed = false;

    const record = (name: string, error: unknown): void => {
      failures.push(error);
      safeWrite(
        `QMD Warning: ${name} failed during stdio shutdown (${
          error instanceof Error ? error.message : String(error)
        }); continuing shutdown.\n`
      );
    };

    // One-shot steps, shared by the graceful chain and an abandoned cleanup, so
    // a step whose promise never settles is awaited again instead of restarted.
    const once = (run: () => Promise<void>): (() => Promise<void>) => {
      let started: Promise<void> | null = null;
      return () => (started ??= Promise.resolve().then(run));
    };
    const closeServerOnce = once(() => options.closeServer());
    const disposeLlmOnce = once(async () => { await options.disposeLlm?.(); });
    const closeStoreOnce = once(async () => {
      await options.closeStore();
      storeClosed = true;
    });

    const bounded = async (name: string, phase: McpShutdownPhase, run: () => Promise<void> | void): Promise<void> => {
      let work: Promise<void>;
      try {
        work = Promise.resolve().then(run);
      } catch (error) {
        record(name, error);
        return;
      }
      try {
        await Promise.race([work, deadline.at(phase)]);
      } catch (error) {
        if (error instanceof McpShutdownTimeoutError) throw error;
        record(name, error);
      }
    };

    const finish = (): void => {
      if (!storeClosed) return;
      // Terminal transition: the store is gone, so nothing is admitted again.
      try {
        lifecycle?.finishTeardown();
      } catch (error) {
        record("lifecycle teardown", error);
      }
    };

    let timedOut: McpShutdownTimeoutError | null = null;
    try {
      await bounded("server.close()", "transport", () => closeServerOnce());
      if (waitForIdle) {
        // One shared deadline: this wait, the disposal and the store close all
        // expire together instead of each getting a fresh budget.
        const idle = await Promise.race([waitForIdle(deadline.remainingMs()), deadline.at("drain")]);
        if (!idle) throw new McpShutdownTimeoutError("drain", timeoutMs);
      }
      if (options.disposeLlm) {
        await bounded("llama disposal", "transport", () => disposeLlmOnce());
      }
      await bounded("store.close()", "store", () => closeStoreOnce());
      finish();
    } catch (error) {
      if (error instanceof McpShutdownTimeoutError) timedOut = error;
      // Every other failure was already recorded by bounded().
    } finally {
      deadline.dispose();
    }

    if (timedOut) {
      safeWrite(
        `QMD Warning: ${timedOut.message}; continuing cleanup in the background with the store still open.\n`
      );
      void (async () => {
        try {
          // Unbounded on purpose: the store must not close under an operation
          // that is still running. If that work never settles, cleanup stops
          // here — the exit code already reported the abandoned state.
          if (waitForIdle) {
            try {
              await waitForIdle();
            } catch (error) {
              record("in-flight drain", error);
            }
          }
          await closeServerOnce().catch((error) => record("server.close()", error));
          if (options.disposeLlm) await disposeLlmOnce().catch((error) => record("llama disposal", error));
          await closeStoreOnce().catch((error) => record("store.close()", error));
          finish();
        } finally {
          closed.resolve();
        }
      })();
    } else {
      closed.resolve();
    }

    try {
      const prior = getExitCode();
      if (timedOut || failures.length > 0) {
        setExitCode(1);
      } else if (prior === undefined || prior === 0) {
        setExitCode(0);
      }
      // else: keep an earlier nonzero status instead of masking it
    } catch {
      // injected setExitCode/getExitCode must not break the shutdown promise
    }
    if (timedOut) {
      options.exitOnTimeout?.(timedOut);
      // The caller gets the typed failure; `closed` already tracks the cleanup
      // that continues in the background.
      throw timedOut;
    }
  };

  const shutdown = (() => (shutdownPromise ??= performShutdown())) as StdioShutdown;
  const onStdinEof = (): void => { void shutdown().catch(() => {}); };

  stdin.once("end", onStdinEof);
  stdin.once("close", onStdinEof);

  // The parent can die between spawn and listener registration; check the
  // stream flags after subscribing so an already-ended stdin still shuts down.
  if (stdin.readableEnded || stdin.destroyed) {
    onStdinEof();
  }

  shutdown.closed = closed.promise;
  return shutdown;
}

/** Apply the non-interactive local-config trust gate before SDK model selection. */
async function createMcpStore(options: McpStartupOptions): Promise<QMDStore> {
  const dbPath = options.dbPath ?? getDefaultDbPath();
  const configPath = getConfigPath();
  if (!existsSync(configPath)) return createStore({ dbPath });

  // Read once: the approval and store must use the same config snapshot.
  const config = loadConfig();
  const snapshot: SensitiveSnapshot = {
    processors: Object.entries(config.collections)
      .filter(([, col]) => col.processor !== undefined)
      .map(([collection, col]) => ({ collection, spec: col.processor! })),
    hooks: Object.entries(config.collections)
      .filter(([, col]) => !!col.update)
      .map(([collection, col]) => ({ collection, command: col.update! })),
    paths: Object.entries(config.collections).map(([collection, col]) => ({ collection, path: col.path })),
    models: {
      embed: config.models?.embed,
      embedOptions: config.models?.embedOptions,
      rerank: config.models?.rerank,
      rerankOptions: config.models?.rerankOptions,
      generate: config.models?.generate,
    },
  };
  const decision = decideLocalConfigGate({
    configPath,
    snapshot,
    builtins: {
      embed: DEFAULT_EMBED_MODEL_URI,
      rerank: DEFAULT_RERANK_MODEL_URI,
      generate: DEFAULT_GENERATE_MODEL_URI,
    },
    isInteractive: false,
  });
  const storeConfig = { ...config };
  // Drop every configured URI and option, not the deliberate process env or
  // built-in defaults. Never mutate the loaded config or write it back to YAML.
  if (decision.action !== "run") delete storeConfig.models;
  // MCP exposes no config mutations, so inline config needs no write-through.
  return createStore({ dbPath, config: storeConfig });
}

/** Options for the stdio entry, plus who owns the process lifetime. */
export type McpStdioServerOptions = McpStartupOptions & {
  /**
   * Standalone CLI/server-main only. The EOF shutdown then records its outcome
   * in `process.exitCode` and forces `process.exit(1)` when the deadline
   * expires with work still running: after EOF nothing can consume that work,
   * and a process outliving its deadline is exactly the #751 leak.
   *
   * Omitted (embedded), a reported timeout rejects the returned `shutdown`
   * promise and cleanup continues in the background, while neither
   * `process.exitCode` nor the process itself is touched.
   */
  standalone?: boolean;
};

/**
 * What `startMcpServer` hands back so an embedder can follow the lifecycle.
 */
export type StdioServerHandle = {
  /**
   * The idempotent stdin-EOF shutdown. Rejects with a
   * {@link McpShutdownTimeoutError} when the deadline expired; resolves for
   * every other outcome.
   */
  shutdown: StdioShutdown;
  /**
   * Resolves once every teardown step actually finished, including cleanup
   * abandoned after a reported timeout. Never rejects.
   */
  closed: Promise<void>;
};

/**
 * Serve MCP over stdio until stdin reaches EOF.
 *
 * Programmatic startup never terminates the caller's process: pass
 * `standalone: true` only from a process-serving entry (the CLI, this module's
 * main block), which is the sole owner of `process.exit`.
 */
export async function startMcpServer(options: McpStdioServerOptions = {}): Promise<StdioServerHandle> {
  // Opt into production mode when the MCP server is actually started, not
  // when this module is merely imported for its exports. Importing the module
  // at the top level flipped the global production flag and broke test
  // isolation for downstream suites that expect the default (development)
  // database path behaviour.
  enableProductionMode();
  const store = await createMcpStore(options);
  const inflight = createInflightGate();
  // serveStdio dual-speaks 2026-07-28 and 2025-era clients on one connection
  // (opening exchange pins the era). A hand-wired StdioServerTransport would
  // stay 2025-only even on SDK 2.x.
  const handle = serveStdio(() => createMcpServer(store, inflight));

  // Follow the parent's lifecycle: when stdin reaches EOF the client is gone
  // and the server must shut down instead of orphaning to PID 1 (#751). No
  // disposeLlm here — store.close() disposes this store's own LlamaCpp
  // instance, so passing the global disposeDefaultLlamaCpp would only risk
  // tearing down an unrelated instance in an embedded process.
  const standby = options.standalone === true;
  const shutdown = registerStdioEofShutdown({
    // Admission and drain state live in the gate the handlers are wrapped in,
    // so shutdown closes it before the transport and drains through it.
    lifecycle: inflight,
    closeServer: () => handle.close(),
    closeStore: () => store.close(),
    // Process ownership is opted into, never implied: an embedded host decides
    // what a reported timeout or a failed step means for its own exit.
    ...(standby
      ? {
          setExitCode: (code: number) => { process.exitCode = code; },
          exitOnTimeout: () => process.exit(1),
        }
      : {}),
  });

  return { shutdown, closed: shutdown.closed };
}

// =============================================================================
// Transport: Streamable HTTP
// =============================================================================

export type HttpServerHandle = {
  httpServer: Server;
  port: number;
  /**
   * Stops the server. Every caller shares one teardown promise. Resolves when
   * the listener and the store are closed; rejects with a
   * {@link McpShutdownTimeoutError} at the deadline (default
   * {@link MCP_SHUTDOWN_TIMEOUT_MS}), in which case the remaining cleanup
   * continues in the background.
   *
   * It never calls process.exit: an embedded host decides what a reported
   * timeout means for its own lifetime.
   */
  stop: () => Promise<void>;
  /**
   * Resolves once every teardown step actually finished — including cleanup
   * abandoned after a reported timeout — so it never claims "stopped" before
   * the store is really closed. Never rejects. A hung store keeps it pending.
   */
  closed: Promise<void>;
};

/** Options for the HTTP transport: the startup options plus listener knobs. */
export type McpHttpServerOptions = McpStartupOptions & {
  quiet?: boolean;
  host?: string;
  allowedOrigins?: string[];
  allowedHosts?: string[];
  /**
   * How long the whole stop() may take: the drain of already-admitted
   * requests, the MCP transport close, the listener close and the store close
   * all share this one deadline. Defaults to {@link MCP_SHUTDOWN_TIMEOUT_MS}.
   *
   * Until the deadline, admitted work is allowed to finish. At the deadline
   * the admitted requests are aborted (their responses can no longer be
   * delivered), their sockets are severed, and stop() rejects with a
   * {@link McpShutdownTimeoutError}. The store is NOT closed underneath the
   * abandoned work: the remaining cleanup continues in the background, in the
   * same safe order, and the lifecycle only reaches `stopped` once closeStore()
   * actually resolved. A standalone entry exits non-zero at that point; an
   * embedded caller keeps running and can await `handle.closed`.
   */
  shutdownDrainTimeoutMs?: number;
  /**
   * The lifecycle every request is admitted through. Defaults to a fresh one.
   * Pass a shared lifecycle to drain the listener from an owner that knows
   * more than the HTTP transport does: after `beginDrain()` the listener
   * answers 503 instead of touching the store, and stop() still tears down
   * (the transition is idempotent).
   */
  lifecycle?: McpLifecycle;
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
 *
 * Every request is admitted through one lifecycle (see createMcpLifecycle),
 * so `stop()` drains what it admitted before closing the shared store, the
 * MCP transport, or a socket underneath one.
 *
 * `stop()` is bounded end to end. It closes admission and the listener at
 * once, lets the requests it already admitted finish until the deadline
 * (default {@link MCP_SHUTDOWN_TIMEOUT_MS}, overridable with
 * `shutdownDrainTimeoutMs`), and then closes the MCP transport, finishes the
 * listener and closes the store — all inside that one deadline. A request that
 * never completes (a client that stalls mid-body, a long-lived
 * `subscriptions/listen` stream, a hung LLM call) is aborted at the deadline:
 * `stop()` rejects with a {@link McpShutdownTimeoutError} while the abandoned
 * cleanup continues in the background, so the store is never closed under live
 * work and `stop()` never hangs the caller.
 *
 * The module installs no process signal handlers and never exits the process:
 * a standalone entry (`qmd mcp --http`) owns SIGTERM/SIGINT handling and maps a
 * reported timeout to a non-zero exit, while an embedded host decides for
 * itself and can await `handle.closed`.
 */
export async function startMcpHttpServer(
  port: number,
  options: McpHttpServerOptions = {},
): Promise<HttpServerHandle> {
  // See startMcpServer() for the rationale — flip production mode here so the
  // HTTP transport resolves the real database path, without leaking state into
  // callers that only import this module for its exports (e.g. tests).
  enableProductionMode();
  const store = await createMcpStore(options);

  // Pre-fetch default collection names for REST endpoint
  const defaultCollectionNames = await store.getDefaultCollectionNames();

  // One admission/drain barrier for every request on this listener: REST
  // routes and /mcp alike share the store that stop() closes.
  const lifecycle = options.lifecycle ?? createMcpLifecycle();

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

  /**
   * Resolve once the response can take more bytes, or once it is gone. Racing
   * the response's own lifecycle keeps a stalled subscriber from wedging the
   * write loop after its socket died.
   */
  function waitForDrain(nodeRes: ServerResponse, signal: AbortSignal): Promise<void> {
    if (signal.aborted) return Promise.resolve();
    const { promise, resolve } = Promise.withResolvers<void>();
    const settle = (): void => {
      nodeRes.off("drain", settle);
      nodeRes.off("close", settle);
      signal.removeEventListener("abort", settle);
      resolve();
    };
    nodeRes.once("drain", settle);
    nodeRes.once("close", settle);
    signal.addEventListener("abort", settle, { once: true });
    return promise;
  }

  /**
   * Copy a web Response onto a Node response with backpressure.
   *
   * `await response.arrayBuffer()` cannot be used: for `subscriptions/listen`
   * the SDK answers with an SSE stream that only ends when the subscription
   * does, so buffering waits forever and no byte ever reaches the client. Reads
   * stop as soon as the response is gone or the request is aborted, and a
   * write that cannot drain waits for 'drain' rather than queuing the rest.
   */
  async function writeWebResponse(nodeRes: ServerResponse, response: Response, signal: AbortSignal): Promise<void> {
    if (nodeRes.destroyed || nodeRes.writableEnded) {
      await response.body?.cancel(signal.reason).catch(() => {});
      return;
    }
    nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
    if (!response.body) {
      nodeRes.end();
      return;
    }

    const reader = response.body.getReader();
    const aborted = Promise.withResolvers<null>();
    const onAbort = (): void => aborted.resolve(null);
    signal.addEventListener("abort", onAbort, { once: true });
    let completed = false;
    try {
      for (;;) {
        const chunk = await Promise.race([reader.read(), aborted.promise]);
        if (chunk === null) break;
        if (chunk.done) {
          completed = true;
          break;
        }
        if (nodeRes.destroyed || nodeRes.writableEnded) break;
        if (!nodeRes.write(Buffer.from(chunk.value))) {
          await waitForDrain(nodeRes, signal);
          if (signal.aborted || nodeRes.destroyed) break;
        }
      }
    } finally {
      signal.removeEventListener("abort", onAbort);
      if (!completed) {
        // The producer may ignore cancellation; the exchange still has to be
        // told the stream is no longer wanted.
        await reader.cancel(signal.reason).catch(() => {});
      } else {
        reader.releaseLock();
      }
      if (!nodeRes.writableEnded && !nodeRes.destroyed) nodeRes.end();
    }
  }

  const host = options.host ?? process.env.QMD_HOST ?? "localhost";
  const originGuard = resolveOriginGuard({
    host,
    ...(options.allowedOrigins ? { allowedOrigins: options.allowedOrigins } : {}),
    ...(options.allowedHosts ? { allowedHosts: options.allowedHosts } : {}),
  });

  /**
   * Abort controllers for the requests this listener admitted. A client that
   * goes away and a shutdown deadline both abort here, which is what tells a
   * /mcp exchange (through its web Request's signal) and a REST search (through
   * store.search's signal) that its result is no longer wanted.
   */
  const admitted = new Set<AbortController>();
  /** True once teardown began closing the listener. */
  let closingListener = false;

  /**
   * A connection is only aborted when it closes before its response finished:
   * `finish` means the bytes were handed to the socket, so a later 'close' is
   * the normal end of a keep-alive connection rather than a disconnect.
   * IncomingMessage 'close' is deliberately not observed — it fires as soon as
   * the request body ends, which would abort a live search.
   */
  function watchForDisconnect(nodeRes: ServerResponse, abort: AbortController): void {
    nodeRes.once("close", () => {
      if (nodeRes.writableFinished) return;
      if (abort.signal.aborted) return;
      abort.abort(new McpRequestAbortedError());
    });
  }

  const httpServer = createServer(async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
    const reqStart = Date.now();
    const pathname = (nodeReq.url || "/").split("?")[0];

    // Admission gate first: once stop() has moved to draining, a request a
    // socket still delivers (one accepted before the listener closed, or
    // dispatched while the drain runs) must not read the store that teardown
    // closes after it. A refused request answers and closes its connection.
    const release = lifecycle.admit();
    if (!release) {
      nodeRes.writeHead(503, { "Content-Type": "application/json", "Connection": "close" });
      nodeRes.end(JSON.stringify({
        jsonrpc: "2.0",
        error: { code: -32000, message: "Server is shutting down" },
        id: null,
      }));
      log(`${ts()} ${nodeReq.method ?? "?"} ${pathname} 503 — shutting down`);
      return;
    }

    // The ticket covers the handler's work, not the bytes: the listener-close
    // step of teardown is what waits for a flushed response, so a slow client
    // cannot hold the drain (or the store close) hostage, while a response
    // still flushing is never severed early.
    const abort = new AbortController();
    admitted.add(abort);
    watchForDisconnect(nodeRes, abort);
    let released = false;
    const releaseAdmission = (): void => {
      if (released) return;
      released = true;
      admitted.delete(abort);
      release();
    };
    // A finished response makes its keep-alive socket idle; during teardown it
    // must go, or the listener would wait for it until the deadline.
    nodeRes.once("finish", () => {
      if (closingListener) httpServer.closeIdleConnections?.();
    });

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
        let parsedParams: unknown;
        try {
          parsedParams = JSON.parse(rawBody);
        } catch {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Invalid JSON body" }));
          return;
        }
        if (typeof parsedParams !== "object" || parsedParams === null || Array.isArray(parsedParams)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "JSON body must be an object" }));
          return;
        }
        const params = parsedParams as Record<string, unknown>;

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

        // Optional metadata filter — must be an object and a valid filter AST
        let restFilter: MetadataFilter | undefined;
        if (params.filter !== undefined) {
          if (typeof params.filter !== "object" || params.filter === null || Array.isArray(params.filter)) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: "Invalid field: filter (must be an object)" }));
            return;
          }
          const filterValidation = validateFilterArgument(params.filter);
          if (filterValidation.error) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: filterValidation.error }));
            return;
          }
          restFilter = filterValidation.filter;
        }

        // Use default collections if none specified
        const effectiveCollections = Array.isArray(params.collections) ? params.collections.map(String) : defaultCollectionNames;

        // The abort signal reaches the embedding/rerank stages, so a client
        // that hangs up or a shutdown deadline stops paying for work whose
        // result nobody will read.
        const results = await store.search({
          signal: abort.signal,
          queries,
          collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
          filter: restFilter,
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
            ...(Object.keys(r.metadata).length > 0 ? { metadata: r.metadata } : {}),
            line,
            snippet: addLineNumbers(snippet, line),
          };
        });

        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify({ results: formatted }));
        log(`${ts()} POST /query ${params.searches.length} queries (${Date.now() - reqStart}ms)`);
        return;
      }

      // REST endpoint: POST /metadata — metadata discovery, same body as the metadata tool
      if (pathname === "/metadata" && nodeReq.method === "POST") {
        const rawBody = await collectBody(nodeReq);
        let parsedParams: unknown;
        try {
          parsedParams = rawBody.trim() === "" ? {} : JSON.parse(rawBody);
        } catch {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Invalid JSON body" }));
          return;
        }
        if (typeof parsedParams !== "object" || parsedParams === null || Array.isArray(parsedParams)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "JSON body must be an object" }));
          return;
        }
        const params = parsedParams as Record<string, unknown>;

        // Optional metadata filter — must be an object and a valid filter AST
        let restFilter: MetadataFilter | undefined;
        if (params.filter !== undefined) {
          if (typeof params.filter !== "object" || params.filter === null || Array.isArray(params.filter)) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: "Invalid field: filter (must be an object)" }));
            return;
          }
          const filterValidation = validateFilterArgument(params.filter);
          if (filterValidation.error) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: filterValidation.error }));
            return;
          }
          restFilter = filterValidation.filter;
        }

        // Optional metadata match, validated the same way against entries
        let restMatch: MetadataMatch | undefined;
        if (params.match !== undefined) {
          if (typeof params.match !== "object" || params.match === null || Array.isArray(params.match)) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: "Invalid field: match (must be an object)" }));
            return;
          }
          const matchValidation = validateMatchArgument(params.match);
          if (matchValidation.error) {
            nodeRes.writeHead(400, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({ error: matchValidation.error }));
            return;
          }
          restMatch = matchValidation.match;
        }

        if (params.sort !== undefined && params.sort !== "count" && params.sort !== "value") {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Invalid field: sort (must be 'count' or 'value')" }));
          return;
        }

        if (params.collections !== undefined && !Array.isArray(params.collections)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Invalid field: collections (must be an array)" }));
          return;
        }

        // The JSON type is checked here; the store checks each number's domain.
        const numberFields = readNumberFields(params, ["keyLimit", "keyOffset", "valueLimit", "valueOffset", "minCount"]);
        if (numberFields.error) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: numberFields.error }));
          return;
        }

        // Use default collections if none specified
        const effectiveCollections = params.collections ? params.collections.map(String) : defaultCollectionNames;

        let result: ListMetadataResult;
        try {
          result = await store.listMetadata({
            collection: effectiveCollections.length > 0 ? effectiveCollections : undefined,
            match: restMatch,
            filter: restFilter,
            sort: params.sort,
            ...numberFields.values,
          });
        } catch (err) {
          if (!(err instanceof MetadataOptionError) && !(err instanceof MetadataBindingBudgetError)) throw err;
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: err.message }));
          return;
        }

        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify(result));
        log(`${ts()} POST /metadata ${result.keys.length} keys (${Date.now() - reqStart}ms)`);
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
        // The signal is what tells the SDK the exchange is cancelled when the
        // client disconnects or the shutdown deadline aborts this request; it
        // is also the only way a long-lived subscription stream learns that
        // nobody is listening any more.
        const request = new Request(url, {
          method: nodeReq.method || "GET",
          headers: nodeHeadersToWeb(nodeReq),
          ...(rawBody !== undefined ? { body: rawBody } : {}),
          signal: abort.signal,
        });
        const response = await mcpHandler.fetch(
          request,
          parsedBody !== undefined ? { parsedBody } : undefined,
        );

        await writeWebResponse(nodeRes, response, abort.signal);
        log(`${ts()} ${nodeReq.method} /mcp ${label} (${Date.now() - reqStart}ms)`);
        return;
      }

      nodeRes.writeHead(404);
      nodeRes.end("Not Found");
    } catch (err) {
      // A request whose client disconnected (or that the shutdown deadline
      // abandoned) has no socket left to answer on; anything else still
      // deserves a 500.
      if (!abort.signal.aborted && !nodeRes.headersSent && !nodeRes.writableEnded) {
        nodeRes.writeHead(500);
        nodeRes.end("Internal Server Error");
      }
    } finally {
      // The handler's work is done. Bytes still flushing belong to the
      // listener-close step of teardown, not to the drain.
      releaseAdmission();
    }
  });

  await new Promise<void>((resolve, reject) => {
    httpServer.on("error", reject);
    httpServer.listen(port, host, () => resolve());
  });

  const actualPort = (httpServer.address() as AddressInfo).port;

  // Listener completion: the callback of httpServer.close() fires once every
  // connection ended. It is resolved exactly once, by the step that issues the
  // close, and awaited under the same deadline as everything else.
  const listenerClosed = Promise.withResolvers<void>();

  /**
   * Tear the listener and the shared store down exactly once, in lifecycle
   * order (the order runMcpTeardown enforces):
   *
   * 1. `serving -> draining` closes admission, so every request from here on
   *    — including one a runtime dispatches from a socket it already accepted —
   *    is answered 503 instead of touching the store.
   * 2. Close the listener. httpServer.close() stops accepting new connections
   *    and leaves established ones alone, so an admitted request still runs;
   *    it also releases the port while that work finishes.
   * 3. Drain: admitted requests keep running until the single deadline.
   * 4. Close the MCP transport. mcpHandler.close() aborts the exchanges it
   *    still tracks, so it MUST come after the drain — closing it first aborted
   *    admitted /mcp requests, the same bug one layer up.
   * 5. Finish the listener: idle keep-alive sockets hold its 'close' open, so
   *    they are swept (again on each flushed response) before waiting.
   * 6. Close the store, then `draining -> stopped` — only if it really closed.
   *
   * On a deadline hit, runMcpTeardown aborts the admitted requests (their web
   * Request signal cancels the SDK exchange and the REST search), severs the
   * sockets, and finishes the same chain in the background: the store is never
   * closed underneath work that is still running, and `closed` only resolves
   * once it did close. Every concurrent caller shares one stop promise, so a
   * second stop() cannot close the store twice.
   */
  const teardown = runMcpTeardown(
    {
      beginDrain: () => lifecycle.beginDrain(),
      waitForIdle: (timeoutMs) => lifecycle.waitForIdle(timeoutMs),
      abortWork: (reason) => {
        for (const abort of admitted) {
          if (!abort.signal.aborted) abort.abort(reason);
        }
      },
      severConnections: () => httpServer.closeAllConnections?.(),
      beginListenerClose: () => {
        closingListener = true;
        try {
          httpServer.close(() => listenerClosed.resolve());
        } catch (error) {
          listenerClosed.resolve();
          throw error;
        } finally {
          httpServer.closeIdleConnections?.();
        }
      },
      listenerClosed: () => listenerClosed.promise,
      closeTransport: () => mcpHandler.close(),
      closeStore: () => store.close(),
      finishTeardown: () => lifecycle.finishTeardown(),
    },
    {
      ...(options.shutdownDrainTimeoutMs !== undefined ? { timeoutMs: options.shutdownDrainTimeoutMs } : {}),
      log,
    },
  );

  const stop = teardown.stop;

  log(`QMD MCP server listening on http://${host}:${actualPort}/mcp`);
  if (originGuard.disabled) {
    log("Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. Only do this behind your own authenticating proxy.");
  } else if (!originGuard.enforceHost) {
    log(`Warning: bound to ${host} with no QMD_ALLOWED_HOSTS — Host validation is off and the index is readable by anyone who can reach this port.`);
  }
  return { httpServer, port: actualPort, stop, closed: teardown.closed };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer({ standalone: true }).catch(console.error);
}
