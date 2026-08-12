/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Follows MCP spec 2025-06-18 for proper response types.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { basename, join, dirname } from "node:path";
import { fileURLToPath } from "url";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { WebStandardStreamableHTTPServerTransport }
  from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { existsSync } from "fs";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  getDefaultDbPath,
  DEFAULT_MULTI_GET_MAX_BYTES,
  type QMDStore,
  type EmbedProgress,
  type UpdateProgress,
  type ExpandedQuery,
  type IndexStatus,
} from "../index.js";
import { getConfigPath } from "../collections.js";
import {
  DEFAULT_GLOB,
  EMBED_FAILURE_BATCH_NO_VECTOR_REASON,
  EMBED_FAILURE_BATCH_SESSION_EXPIRED_PREFIX,
  EMBED_FAILURE_ERROR_RATE_REASON,
  EMBED_FAILURE_NO_VECTOR_REASON,
  EMBED_FAILURE_SESSION_EXPIRED_REASON,
  enableProductionMode,
} from "../store.js";

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

type CollectionResult = {
  name: string;
  path: string;
  pattern: string;
  documents: number;
  indexedDocuments: number;
  lastUpdated: string | null;
  includeByDefault: boolean;
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

const MAX_NODE_TIMER_MS = 2_147_483_647;
const MAX_EMBED_TIMEOUT_MINUTES = Math.floor(MAX_NODE_TIMER_MS / 60_000);

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

function toCollectionResult(
  collection: Awaited<ReturnType<QMDStore["listCollections"]>>[number]
): CollectionResult {
  return {
    name: collection.name,
    path: collection.pwd,
    pattern: collection.glob_pattern,
    documents: collection.active_count,
    indexedDocuments: collection.doc_count,
    lastUpdated: collection.last_modified,
    includeByDefault: collection.includeByDefault,
  };
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
 * Injected into the LLM's system prompt via MCP initialize response —
 * gives the LLM immediate context about what's searchable without a tool call.
 */
async function buildInstructions(
  store: QMDStore,
  collectionManagementEnabled: boolean,
): Promise<string> {
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

  if (collectionManagementEnabled) {
    lines.push("");
    lines.push("Collection management is enabled for this session:");
    lines.push("  - Use `collection_list` and `collection_show` to inspect configured collections.");
    lines.push("  - To register a directory, call `collection_add`, then `update`, inspect `needsEmbedding`, and call `embed` only when `needsEmbedding` is greater than 0.");
    lines.push("  - Use `collection_rename` to rename a collection and its indexed paths.");
    lines.push("  - Use destructive `collection_remove` to delete a collection from QMD's index; source files stay unchanged.");
    lines.push("  - Ignore patterns can be supplied to `collection_add`, but collection read tools cannot return configured ignore patterns.");
  }

  // --- Maintenance workflow ---
  lines.push("");
  lines.push("Maintenance:");
  lines.push("  - Call the `update` MCP tool to synchronize configured collections into the derived index.");
  lines.push("  - After update, call `status` and inspect `needsEmbedding`.");
  lines.push("  - Call the `embed` MCP tool only when `needsEmbedding` is greater than 0.");

  // --- Capability gaps ---
  if (!status.hasVectorIndex) {
    lines.push("");
    lines.push("Note: No vector embeddings yet; semantic search (vec/hyde) remains unavailable until pending embeddings are generated.");
  } else if (status.needsEmbedding > 0) {
    lines.push("");
    lines.push(`Note: ${status.needsEmbedding} documents currently need embedding.`);
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
  lines.push("  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`) or comma-separated list.");

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
type MaintenanceOperation =
  | "update"
  | "embed"
  | "collection_add"
  | "collection_rename"
  | "collection_remove";

type MaintenanceLease =
  | { acquired: true; release: () => void }
  | { acquired: false; active: MaintenanceOperation };

class MaintenanceCancelledError extends Error {}

const maintenanceStates = new WeakMap<
  QMDStore,
  { active?: { operation: MaintenanceOperation; leaseId: symbol } }
>();

function throwIfMaintenanceAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new MaintenanceCancelledError();
  }
}

function acquireMaintenance(
  store: QMDStore,
  operation: MaintenanceOperation
): MaintenanceLease {
  let state = maintenanceStates.get(store);
  if (!state) {
    state = {};
    maintenanceStates.set(store, state);
  }

  if (state.active) {
    return { acquired: false, active: state.active.operation };
  }

  const leaseId = Symbol(operation);
  state.active = { operation, leaseId };
  return {
    acquired: true,
    release: () => {
      if (state.active?.leaseId === leaseId) {
        state.active = undefined;
      }
    },
  };
}

function enqueueBestEffortNotification(
  pendingNotifications: Set<Promise<void>>,
  send: () => Promise<void>
): void {
  const pendingNotification = (async () => {
    try {
      await send();
    } catch {
      // Progress notifications are best-effort and never change tool results.
    }
  })();
  pendingNotifications.add(pendingNotification);
  void pendingNotification.then(() => {
    pendingNotifications.delete(pendingNotification);
  });
}

const SAFE_EMBED_FAILURE_REASONS = new Set([
  EMBED_FAILURE_NO_VECTOR_REASON,
  EMBED_FAILURE_SESSION_EXPIRED_REASON,
  EMBED_FAILURE_ERROR_RATE_REASON,
  EMBED_FAILURE_BATCH_NO_VECTOR_REASON,
]);

function sanitizeEmbedFailureReason(reason: string): string {
  if (SAFE_EMBED_FAILURE_REASONS.has(reason)) {
    return reason;
  }

  if (
    reason.startsWith(`${EMBED_FAILURE_BATCH_SESSION_EXPIRED_PREFIX}: `)
  ) {
    return EMBED_FAILURE_BATCH_SESSION_EXPIRED_PREFIX;
  }

  return "embedding backend or index write failed";
}

export type McpServerOptions = {
  enableCollectionManagement?: boolean;
};

export async function createMcpServer(
  store: QMDStore,
  options: McpServerOptions,
): Promise<McpServer> {
  const collectionManagementEnabled =
    options.enableCollectionManagement === true;
  const server = new McpServer(
    { name: "qmd", version: getPackageVersion() },
    {
      instructions: await buildInstructions(
        store,
        collectionManagementEnabled,
      ),
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
    async (uri, { path }) => {
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
    }
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
      inputSchema: {
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
      },
    },
    async ({ query, searches, limit, minScore, candidateLimit, collections, intent, rerank }) => {
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
    }
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
      inputSchema: {
        file: z.string().describe("File path or docid from search results. Supports a line-range suffix: 'pages/meeting.md:100' starts at line 100; 'pages/meeting.md:100:40' (or '#abc123:100:40') reads 40 lines from line 100."),
        fromLine: z.number().optional().describe("Start from this line number (1-indexed)"),
        maxLines: z.number().optional().describe("Maximum number of lines to return"),
        lineNumbers: z.boolean().optional().default(true).describe("Add line numbers to output (format: 'N: content'). On by default; set false for raw content."),
      },
    },
    async ({ file, fromLine, maxLines, lineNumbers }) => {
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
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        pattern: z.string().describe("Glob pattern or comma-separated list of file paths"),
        maxLines: z.number().optional().describe("Maximum lines per file"),
        maxBytes: z.number().optional().default(DEFAULT_MULTI_GET_MAX_BYTES).describe("Skip files larger than this (default: 65536 = 64KB)"),
        lineNumbers: z.boolean().optional().default(true).describe("Add line numbers to output (format: 'N: content'). On by default; set false for raw content."),
      },
    },
    async ({ pattern, maxLines, maxBytes, lineNumbers }) => {
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
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: update (Synchronize configured collections into the derived index)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "update",
    {
      title: "Update Index",
      description: "Synchronize configured Markdown collections into the derived index. Omit collections to update all configured collections. This writes only to QMD's index; source files are unchanged and configured update commands are not executed. With an MCP progress token it reports file progress. Progress notifications are best-effort and never change the tool result. Busy, cancelled, and failed runs return isError. Inspect the structured needsEmbedding count afterward and call embed only when it is greater than 0.",
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
      inputSchema: {
        collections: z.array(z.string()).min(1).optional().describe(
          "Optional non-empty list of configured collection names. Omit to update all collections."
        ),
      },
    },
    async ({ collections }, extra) => {
      if (extra.signal.aborted) {
        return {
          content: [{
            type: "text",
            text: "Update cancelled by the MCP client. Retry when ready.",
          }],
          isError: true,
        };
      }

      const maintenance = acquireMaintenance(store, "update");
      if (!maintenance.acquired) {
        return {
          content: [{
            type: "text",
            text: `QMD maintenance is busy with ${maintenance.active}. Retry after it finishes.`,
          }],
          isError: true,
        };
      }

      try {
        throwIfMaintenanceAborted(extra.signal);

        if (collections) {
          const configuredNames = new Set(
            (await store.listCollections()).map(collection => collection.name)
          );
          throwIfMaintenanceAborted(extra.signal);
          const unknownCollections = collections.filter(
            name => !configuredNames.has(name)
          );

          if (unknownCollections.length > 0) {
            return {
              content: [{
                type: "text",
                text: `Unknown collection(s): ${unknownCollections.join(", ")}. Call status to list configured collections.`,
              }],
              isError: true,
            };
          }
        }

        const progressToken = extra._meta?.progressToken;
        const pendingNotifications = new Set<Promise<void>>();
        let activeCollection: string | undefined;
        let completedFiles = 0;
        let activeCollectionTotal = 0;
        let lastProgress = 0;
        let lastTotal = 0;

        const onProgress = (info: UpdateProgress) => {
          throwIfMaintenanceAborted(extra.signal);

          if (progressToken === undefined) {
            return;
          }

          if (
            activeCollection !== undefined &&
            activeCollection !== info.collection
          ) {
            completedFiles += activeCollectionTotal;
          }
          activeCollection = info.collection;
          activeCollectionTotal = info.total;

          const progress = Math.max(lastProgress, completedFiles + info.current);
          const total = Math.max(
            lastTotal,
            completedFiles + info.total,
            progress
          );
          lastProgress = progress;
          lastTotal = total;

          enqueueBestEffortNotification(pendingNotifications, () =>
            extra.sendNotification({
              method: "notifications/progress",
              params: {
                progressToken,
                progress,
                total,
                message: `${info.collection}/${info.file}`,
              },
            })
          );
        };

        const result = await store.update({
          ...(collections === undefined ? {} : { collections }),
          onProgress,
        });
        await Promise.all(pendingNotifications);
        throwIfMaintenanceAborted(extra.signal);

        const summary =
          `Updated ${result.collections} collection(s): ${result.indexed} indexed, ` +
          `${result.updated} updated, ${result.unchanged} unchanged, ${result.removed} removed. ` +
          `${result.needsEmbedding} document(s) need embedding.`;

        return {
          content: [{ type: "text", text: summary }],
          structuredContent: result,
        };
      } catch (error) {
        if (error instanceof MaintenanceCancelledError || extra.signal.aborted) {
          return {
            content: [{
              type: "text",
              text: "Update cancelled by the MCP client. Retry when ready.",
            }],
            isError: true,
          };
        }

        return {
          content: [{
            type: "text",
            text: "Update failed. Check QMD status and server logs, then retry.",
          }],
          isError: true,
        };
      } finally {
        maintenance.release();
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: embed (Generate vector embeddings for the derived index)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "embed",
    {
      title: "Embed Documents",
      description: "Generate vector embeddings for pending indexed documents using QMD's centrally configured model. Defaults: force false and a runtime limit of 30 minutes; omit collection to process all pending collections. This writes only to the derived index and may download the configured model if it is not cached. With an MCP progress token it reports chunks, bytes, and errors. Busy, cancelled, failed, and partial-error runs return isError while preserving available counters. Per-document failure reasons are sanitized at the MCP boundary; they are categories, not raw backend errors.",
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
      inputSchema: {
        collection: z.string().min(1).optional().describe(
          "Optional configured collection name. Omit to embed pending documents from all collections."
        ),
        force: z.boolean().optional().default(false).describe(
          "Rebuild existing embeddings in the selected scope (default: false)."
        ),
        chunkStrategy: z.enum(["auto", "regex"]).optional().describe(
          "Chunking strategy. Omit to use QMD's configured default."
        ),
        maxDocsPerBatch: z.number().int().positive().optional().describe(
          "Maximum documents per embedding batch."
        ),
        maxBatchMiB: z.number().positive().optional().describe(
          "Maximum embedding batch size in MiB."
        ),
        timeoutMinutes: z.number()
          .nonnegative()
          .max(
            MAX_EMBED_TIMEOUT_MINUTES,
            `timeoutMinutes must be at most ${MAX_EMBED_TIMEOUT_MINUTES} minutes. Use 0 for no runtime limit.`
          )
          .optional()
          .default(30)
          .describe(
            `Maximum runtime in minutes (default: 30, maximum: ${MAX_EMBED_TIMEOUT_MINUTES}). Use 0 for no runtime limit.`
          ),
      },
    },
    async ({
      collection,
      force,
      chunkStrategy,
      maxDocsPerBatch,
      maxBatchMiB,
      timeoutMinutes,
    }, extra) => {
      if (extra.signal.aborted) {
        return {
          content: [{
            type: "text",
            text: "Embedding cancelled by the MCP client. Retry when ready.",
          }],
          isError: true,
        };
      }

      const maintenance = acquireMaintenance(store, "embed");
      if (!maintenance.acquired) {
        return {
          content: [{
            type: "text",
            text: `QMD maintenance is busy with ${maintenance.active}. Retry after it finishes.`,
          }],
          isError: true,
        };
      }

      try {
        throwIfMaintenanceAborted(extra.signal);

        if (collection !== undefined) {
          const configuredNames = new Set(
            (await store.listCollections()).map(configured => configured.name)
          );
          throwIfMaintenanceAborted(extra.signal);

          if (!configuredNames.has(collection)) {
            return {
              content: [{
                type: "text",
                text: `Unknown collection: ${collection}. Call status to list configured collections.`,
              }],
              isError: true,
            };
          }
        }

        const progressToken = extra._meta?.progressToken;
        const pendingNotifications = new Set<Promise<void>>();
        let lastProgress = 0;
        let lastTotal = 0;
        const onProgress = (info: EmbedProgress) => {
          throwIfMaintenanceAborted(extra.signal);

          if (progressToken === undefined) {
            return;
          }

          const progress = Math.max(lastProgress, info.chunksEmbedded);
          const total = Math.max(lastTotal, info.totalChunks, progress);
          lastProgress = progress;
          lastTotal = total;

          enqueueBestEffortNotification(pendingNotifications, () =>
            extra.sendNotification({
              method: "notifications/progress",
              params: {
                progressToken,
                progress,
                total,
                message:
                  `${info.chunksEmbedded}/${info.totalChunks} chunks; ` +
                  `${info.bytesProcessed}/${info.totalBytes} bytes; ` +
                  `${info.errors} errors`,
              },
            })
          );
        };

        const embedOptions: NonNullable<Parameters<QMDStore["embed"]>[0]> = {
          force,
          maxDurationMs: Math.round(timeoutMinutes * 60 * 1000),
          ...(collection === undefined ? {} : { collection }),
          ...(chunkStrategy === undefined ? {} : { chunkStrategy }),
          ...(maxDocsPerBatch === undefined ? {} : { maxDocsPerBatch }),
          ...(maxBatchMiB === undefined
            ? {}
            : { maxBatchBytes: maxBatchMiB * 1024 * 1024 }),
          onProgress,
        };
        const result = await store.embed(embedOptions);
        await Promise.all(pendingNotifications);
        throwIfMaintenanceAborted(extra.signal);
        const failures = result.failures ?? [];
        const structured = {
          ...result,
          failureCount: result.errors,
          failures: failures.slice(0, 20).map(failure => ({
            ...failure,
            reason: sanitizeEmbedFailureReason(failure.reason),
          })),
          failuresTruncated: failures.length > 20,
        };
        const summary =
          `Embedded ${result.docsProcessed} document(s) into ${result.chunksEmbedded} chunk(s) ` +
          `in ${result.durationMs}ms with ${result.errors} error(s).`;

        return {
          content: [{ type: "text", text: summary }],
          structuredContent: structured,
          isError: result.errors > 0,
        };
      } catch (error) {
        if (error instanceof MaintenanceCancelledError || extra.signal.aborted) {
          return {
            content: [{
              type: "text",
              text: "Embedding cancelled by the MCP client. Retry when ready.",
            }],
            isError: true,
          };
        }

        return {
          content: [{
            type: "text",
            text: "Embedding failed. Check QMD status and server logs, then retry.",
          }],
          isError: true,
        };
      } finally {
        maintenance.release();
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: collection_add (Opt-in collection configuration write)
  // ---------------------------------------------------------------------------

  if (collectionManagementEnabled) {
    server.registerTool(
      "collection_add",
      {
        title: "Add Collection",
        description: "Register a local directory as a QMD collection without indexing it. Call update afterward, then embed if documents need embeddings.",
        annotations: {
          readOnlyHint: false,
          destructiveHint: false,
          idempotentHint: false,
          openWorldHint: false,
        },
        inputSchema: {
          path: z.string().min(1).describe(
            "Existing local directory to register. Absolute paths are recommended; relative paths resolve against the server process working directory."
          ),
          name: z.string().min(1).optional().describe(
            "Optional collection name; defaults to the resolved directory basename"
          ),
          pattern: z.string().min(1).optional().default(DEFAULT_GLOB).describe(
            `Markdown glob pattern (default: ${DEFAULT_GLOB})`
          ),
          ignore: z.array(z.string().min(1)).optional().describe(
            "Optional glob patterns to exclude"
          ),
        },
      },
      async ({ path, name, pattern, ignore }) => {
        const maintenance = acquireMaintenance(store, "collection_add");
        if (!maintenance.acquired) {
          return {
            content: [{
              type: "text",
              text: `QMD maintenance is busy with ${maintenance.active}. Retry after it finishes.`,
            }],
            isError: true,
          };
        }

        try {
          let resolvedPath: string;
          try {
            resolvedPath = await realpath(path);
          } catch (error) {
            if (
              typeof error === "object" &&
              error !== null &&
              "code" in error
            ) {
              if (error.code === "ENOENT") {
                return {
                  content: [{
                    type: "text",
                    text: `Collection path does not exist: ${path}. Check the path and retry.`,
                  }],
                  isError: true,
                };
              }
              if (error.code === "EACCES") {
                return {
                  content: [{
                    type: "text",
                    text: `Collection path is not accessible: ${path}. Check directory permissions and retry.`,
                  }],
                  isError: true,
                };
              }
              if (error.code === "ENOTDIR") {
                return {
                  content: [{
                    type: "text",
                    text: `A collection path component is not a directory: ${path}. Check the path and retry.`,
                  }],
                  isError: true,
                };
              }
            }
            return {
              content: [{
                type: "text",
                text: "Could not resolve the collection path. Check access and server logs, then retry.",
              }],
              isError: true,
            };
          }

          try {
            if (!(await stat(resolvedPath)).isDirectory()) {
              return {
                content: [{
                  type: "text",
                  text: `Collection path is not a directory: ${path}. Choose a directory and retry.`,
                }],
                isError: true,
              };
            }

            const collectionName = (name ?? basename(resolvedPath)) || "root";
            const collections = await store.listCollections();
            if (collections.some(collection => collection.name === collectionName)) {
              return {
                content: [{
                  type: "text",
                  text: `Collection '${collectionName}' already exists. Choose a different name.`,
                }],
                isError: true,
              };
            }

            const duplicate = collections.find(collection =>
              collection.pwd === resolvedPath && collection.glob_pattern === pattern
            );
            if (duplicate) {
              return {
                content: [{
                  type: "text",
                  text: `Collection '${duplicate.name}' already uses this path and pattern. Call update for it or remove it first.`,
                }],
                isError: true,
              };
            }

            await store.addCollection(collectionName, {
              path: resolvedPath,
              pattern,
              ...(ignore === undefined ? {} : { ignore }),
            });

            const created = (await store.listCollections()).find(
              collection => collection.name === collectionName
            );
            if (!created) {
              return {
                content: [{
                  type: "text",
                  text: "Collection may have been registered but could not be read back. Call collection_list to check before retrying.",
                }],
                isError: true,
              };
            }

            return {
              content: [{
                type: "text",
                text: `Added collection '${collectionName}' without indexing. Call update next, then call embed if documents need embeddings.`,
              }],
              structuredContent: { collection: toCollectionResult(created) },
            };
          } catch {
            return {
              content: [{
                type: "text",
                text: "Could not add the collection. Check QMD status and server logs, then retry.",
              }],
              isError: true,
            };
          }
        } finally {
          maintenance.release();
        }
      }
    );

    server.registerTool(
      "collection_rename",
      {
        title: "Rename Collection",
        description: "Rename a configured QMD collection and its existing indexed document paths.",
        annotations: {
          readOnlyHint: false,
          destructiveHint: true,
          idempotentHint: false,
          openWorldHint: false,
        },
        inputSchema: {
          oldName: z.string().min(1).describe("Existing collection name"),
          newName: z.string().min(1).describe("New, unused collection name"),
        },
      },
      async ({ oldName, newName }) => {
        const maintenance = acquireMaintenance(store, "collection_rename");
        if (!maintenance.acquired) {
          return {
            content: [{
              type: "text",
              text: `QMD maintenance is busy with ${maintenance.active}. Retry after it finishes.`,
            }],
            isError: true,
          };
        }

        try {
          let collections: Awaited<ReturnType<QMDStore["listCollections"]>>;
          try {
            collections = await store.listCollections();
          } catch {
            return {
              content: [{
                type: "text",
                text: "Could not read the configured collections. Check QMD status and server logs, then retry.",
              }],
              isError: true,
            };
          }

          if (!collections.some(collection => collection.name === oldName)) {
            return {
              content: [{
                type: "text",
                text: `Collection '${oldName}' does not exist. Call collection_list to check the current names.`,
              }],
              isError: true,
            };
          }
          if (collections.some(collection => collection.name === newName)) {
            return {
              content: [{
                type: "text",
                text: `Collection '${newName}' already exists. Choose a different name.`,
              }],
              isError: true,
            };
          }

          try {
            const renamedSuccessfully = await store.renameCollection(oldName, newName);
            if (!renamedSuccessfully) {
              return {
                content: [{
                  type: "text",
                  text: "No collection was renamed. Call collection_list to check the current names.",
                }],
                isError: true,
              };
            }

            const renamed = (await store.listCollections()).find(
              collection => collection.name === newName
            );
            if (!renamed) {
              return {
                content: [{
                  type: "text",
                  text: "Collection may have been renamed but could not be read back. Call collection_show to verify the new name before retrying.",
                }],
                isError: true,
              };
            }

            return {
              content: [{
                type: "text",
                text: `Renamed collection '${oldName}' to '${newName}'. Existing indexed documents now use qmd://${newName}/... paths.`,
              }],
              structuredContent: { collection: toCollectionResult(renamed) },
            };
          } catch {
            return {
              content: [{
                type: "text",
                text: "Collection may have been renamed. Call collection_show to verify the new name before retrying.",
              }],
              isError: true,
            };
          }
        } finally {
          maintenance.release();
        }
      }
    );

    server.registerTool(
      "collection_remove",
      {
        title: "Remove Collection",
        description: "Remove a QMD collection and its indexed data. Also cleans up content rows that no document references anymore, across all collections. Source files remain unchanged.",
        annotations: {
          readOnlyHint: false,
          destructiveHint: true,
          idempotentHint: false,
          openWorldHint: false,
        },
        inputSchema: {
          name: z.string().min(1).describe("Existing collection name"),
        },
      },
      async ({ name }) => {
        const maintenance = acquireMaintenance(store, "collection_remove");
        if (!maintenance.acquired) {
          return {
            content: [{
              type: "text",
              text: `QMD maintenance is busy with ${maintenance.active}. Retry after it finishes.`,
            }],
            isError: true,
          };
        }

        try {
          let collections: Awaited<ReturnType<QMDStore["listCollections"]>>;
          try {
            collections = await store.listCollections();
          } catch {
            return {
              content: [{
                type: "text",
                text: "Could not read the configured collections. Check QMD status and server logs, then retry.",
              }],
              isError: true,
            };
          }

          if (!collections.some(collection => collection.name === name)) {
            return {
              content: [{
                type: "text",
                text: `Collection '${name}' does not exist. Call collection_list to check the current names.`,
              }],
              isError: true,
            };
          }

          try {
            const result = await store.removeCollection(name);
            if (!result.removed) {
              return {
                content: [{
                  type: "text",
                  text: "No collection was removed. Call collection_list to check the current names.",
                }],
                isError: true,
              };
            }

            return {
              content: [{
                type: "text",
                text: `Removed collection '${name}': deleted ${result.deletedDocs} indexed documents and cleaned ${result.cleanedHashes} globally orphaned content hashes. Source files remain unchanged.`,
              }],
              structuredContent: {
                removed: result.removed,
                deletedDocs: result.deletedDocs,
                cleanedHashes: result.cleanedHashes,
              },
            };
          } catch {
            return {
              content: [{
                type: "text",
                text: "Collection may have been removed. Call collection_list to verify before retrying.",
              }],
              isError: true,
            };
          }
        } finally {
          maintenance.release();
        }
      }
    );
  }

  // ---------------------------------------------------------------------------
  // Tools: collection_list and collection_show (Read-only collection state)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "collection_list",
    {
      title: "List Collections",
      description: "List all configured QMD collections, including ones not indexed yet; status only reports collections with active documents. Each result includes active documents and all indexed documents (active plus inactive), paths, glob patterns, last changes, and unscoped-search defaults.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {},
    },
    async () => {
      try {
        const collections = (await store.listCollections())
          .map(toCollectionResult)
          .sort((left, right) =>
            left.name < right.name ? -1 : left.name > right.name ? 1 : 0
          );
        return {
          content: [{
            type: "text",
            text: `${collections.length} configured ${collections.length === 1 ? "collection" : "collections"}.`,
          }],
          structuredContent: { collections },
        };
      } catch {
        return {
          content: [{
            type: "text",
            text: "Could not list collections. Check QMD status and server logs, then retry.",
          }],
          isError: true,
        };
      }
    }
  );

  server.registerTool(
    "collection_show",
    {
      title: "Show Collection",
      description: "Show configuration and index counts for one configured QMD collection, including active documents and all indexed documents (active plus inactive).",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        name: z.string().min(1).describe("Configured collection name"),
      },
    },
    async ({ name }) => {
      try {
        const collection = (await store.listCollections()).find(
          configured => configured.name === name
        );
        if (!collection) {
          return {
            content: [{
              type: "text",
              text: `Unknown collection: ${name}. Call collection_list to list configured collections.`,
            }],
            isError: true,
          };
        }

        const result = toCollectionResult(collection);
        return {
          content: [{
            type: "text",
            text: `Collection ${result.name}: ${result.path} (${result.documents} active of ${result.indexedDocuments} indexed documents).`,
          }],
          structuredContent: { collection: result },
        };
      } catch {
        return {
          content: [{
            type: "text",
            text: "Could not show the collection. Check QMD status and server logs, then retry.",
          }],
          isError: true,
        };
      }
    }
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
      inputSchema: {},
    },
    async () => {
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
    }
  );

  return server;
}

// =============================================================================
// Transport: stdio (default)
// =============================================================================

export type McpStartupOptions = {
  dbPath?: string;
  enableCollectionManagement?: boolean;
};

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
  const server = await createMcpServer(store, {
    enableCollectionManagement: options.enableCollectionManagement,
  });
  const transport = new StdioServerTransport();
  await server.connect(transport);
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
 * Start MCP server over Streamable HTTP (JSON responses, no SSE).
 * Binds to `options.host` (default "localhost", overridable via the QMD_HOST
 * env var) — set "0.0.0.0" to accept connections from other hosts, e.g. a
 * container liveness probe. Returns a handle for shutdown and port discovery.
 */
export async function startMcpHttpServer(
  port: number,
  options: ({ quiet?: boolean; host?: string } & McpStartupOptions) = {},
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

  // Session map: each client gets its own McpServer + Transport pair (MCP spec requirement).
  // The store is shared — it's stateless SQLite, safe for concurrent access.
  const sessions = new Map<string, WebStandardStreamableHTTPServerTransport>();

  async function createSession(): Promise<WebStandardStreamableHTTPServerTransport> {
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      enableJsonResponse: true,
      onsessioninitialized: (sessionId: string) => {
        sessions.set(sessionId, transport);
        log(`${ts()} New session ${sessionId} (${sessions.size} active)`);
      },
    });
    const server = await createMcpServer(store, {
      enableCollectionManagement: options.enableCollectionManagement,
    });
    await server.connect(transport);

    transport.onclose = () => {
      if (transport.sessionId) {
        sessions.delete(transport.sessionId);
      }
    };

    return transport;
  }

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
      if (args?.path) return `tools/call ${tool} ${args.path}`;
      if (args?.pattern) return `tools/call ${tool} ${args.pattern}`;
      return `tools/call ${tool}`;
    }
    return method;
  }

  function log(msg: string): void {
    if (!quiet) console.error(msg);
  }

  // Helper to collect request body
  async function collectBody(req: IncomingMessage): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks).toString();
  }

  const httpServer = createServer(async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
    const reqStart = Date.now();
    const pathname = nodeReq.url || "/";

    try {
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

      if (pathname === "/mcp" && nodeReq.method === "POST") {
        const rawBody = await collectBody(nodeReq);
        const body = JSON.parse(rawBody);
        const label = describeRequest(body);
        const url = `http://localhost:${port}${pathname}`;
        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(nodeReq.headers)) {
          if (typeof v === "string") headers[k] = v;
        }

        // Route to existing session or create new one on initialize
        const sessionId = headers["mcp-session-id"];
        let transport: WebStandardStreamableHTTPServerTransport;

        if (sessionId) {
          const existing = sessions.get(sessionId);
          if (!existing) {
            nodeRes.writeHead(404, { "Content-Type": "application/json" });
            nodeRes.end(JSON.stringify({
              jsonrpc: "2.0",
              error: { code: -32001, message: "Session not found" },
              id: body?.id ?? null,
            }));
            return;
          }
          transport = existing;
        } else if (isInitializeRequest(body)) {
          transport = await createSession();
        } else {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32000, message: "Bad Request: Missing session ID" },
            id: body?.id ?? null,
          }));
          return;
        }

        const request = new Request(url, { method: "POST", headers, body: rawBody });
        const response = await transport.handleRequest(request, { parsedBody: body });

        nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
        nodeRes.end(Buffer.from(await response.arrayBuffer()));
        log(`${ts()} POST /mcp ${label} (${Date.now() - reqStart}ms)`);
        return;
      }

      if (pathname === "/mcp") {
        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(nodeReq.headers)) {
          if (typeof v === "string") headers[k] = v;
        }

        // GET/DELETE must have a valid session
        const sessionId = headers["mcp-session-id"];
        if (!sessionId) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32000, message: "Bad Request: Missing session ID" },
            id: null,
          }));
          return;
        }
        const transport = sessions.get(sessionId);
        if (!transport) {
          nodeRes.writeHead(404, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32001, message: "Session not found" },
            id: null,
          }));
          return;
        }

        const url = `http://localhost:${port}${pathname}`;
        const rawBody = nodeReq.method !== "GET" && nodeReq.method !== "HEAD" ? await collectBody(nodeReq) : undefined;
        const request = new Request(url, { method: nodeReq.method || "GET", headers, ...(rawBody ? { body: rawBody } : {}) });
        const response = await transport.handleRequest(request);
        nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
        nodeRes.end(Buffer.from(await response.arrayBuffer()));
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

  const host = options.host ?? process.env.QMD_HOST ?? "localhost";
  await new Promise<void>((resolve, reject) => {
    httpServer.on("error", reject);
    httpServer.listen(port, host, () => resolve());
  });

  const actualPort = (httpServer.address() as import("net").AddressInfo).port;

  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    for (const transport of sessions.values()) {
      await transport.close();
    }
    sessions.clear();
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
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer().catch(console.error);
}
