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
import { join, dirname } from "node:path";
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
  type ExpandedQuery,
  type IndexStatus,
} from "../index.js";
import { getConfigPath } from "../collections.js";
import { enableProductionMode } from "../store.js";

enableProductionMode();

// =============================================================================
// Types for structured content
// =============================================================================

type SearchResultItem = {
  docid: string;  // Short docid (#abc123) for quick reference
  file: string;
  title: string;
  score: number;
  context: string | null;
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
 * Injected into the LLM's system prompt via MCP initialize response —
 * gives the LLM immediate context about what's searchable without a tool call.
 */
async function buildInstructions(store: QMDStore): Promise<string> {
  const status = await store.getStatus();
  const contexts = await store.listContexts();
  const globalCtx = await store.getGlobalContext();
  const lines: string[] = [];

  // --- What is this? ---
  lines.push(`QMD is your local search engine over ${status.totalDocuments} markdown documents.`);
  if (globalCtx) lines.push(`Context: ${globalCtx}`);

  // --- What's searchable? ---
  if (status.collections.length > 0) {
    lines.push("");
    lines.push("Collections (scope with `collection` parameter):");
    for (const col of status.collections) {
      // Find root context for this collection
      const rootCtx = contexts.find(c => c.collection === col.name && (c.path === "" || c.path === "/"));
      const desc = rootCtx ? ` — ${rootCtx.context}` : "";
      lines.push(`  - "${col.name}" (${col.documents} docs)${desc}`);
    }
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
  lines.push("  - `get` — single document by path or docid (#abc123). Supports line offset (`file.md:100`).");
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
async function createMcpServer(store: QMDStore): Promise<McpServer> {
  const server = new McpServer(
    { name: "qmd", version: getPackageVersion() },
    { instructions: await buildInstructions(store) },
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
        return { contents: [{ uri: uri.href, text: `Document not found: ${decodedPath}` }] };
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
| Know exact term/name | \`lex\` only |
| Concept search | \`vec\` only |
| Best recall | \`lex\` + \`vec\` |
| Complex/nuanced | \`lex\` + \`vec\` + \`hyde\` |
| Unknown vocabulary | Use a standalone natural-language query (no typed lines) so the server can auto-expand it |

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
        searches: z.array(subSearchSchema).min(1).max(10).describe(
          "Typed sub-queries to execute (lex/vec/hyde). First gets 2x weight."
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
    async ({ searches, limit, minScore, candidateLimit, collections, intent, rerank }) => {
      // Map to internal format
      const queries: ExpandedQuery[] = searches.map(s => ({
        type: s.type,
        query: s.query,
      }));

      // Use default collections if none specified
      const effectiveCollections = collections ?? defaultCollectionNames;

      const results = await store.search({
        queries,
        collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
        limit,
        minScore,
        rerank,
        intent,
      });

      // Use first lex or vec query for snippet extraction
      const primaryQuery = searches.find(s => s.type === 'lex')?.query
        || searches.find(s => s.type === 'vec')?.query
        || searches[0]?.query || "";

      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.bestChunk, primaryQuery, 300, undefined, undefined, intent);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: r.context,
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
        file: z.string().describe("File path or docid from search results (e.g., 'pages/meeting.md', '#abc123', or 'pages/meeting.md:100' to start at line 100)"),
        fromLine: z.number().optional().describe("Start from this line number (1-indexed)"),
        maxLines: z.number().optional().describe("Maximum number of lines to return"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
      },
    },
    async ({ file, fromLine, maxLines, lineNumbers }) => {
      // Support :line suffix in `file` (e.g. "foo.md:120") when fromLine isn't provided
      let parsedFromLine = fromLine;
      let lookup = file;
      const colonMatch = lookup.match(/:(\d+)$/);
      if (colonMatch && colonMatch[1] && parsedFromLine === undefined) {
        parsedFromLine = parseInt(colonMatch[1], 10);
        lookup = lookup.slice(0, -colonMatch[0].length);
      }

      const result = await store.get(lookup, { includeBody: false });

      if ("error" in result) {
        let msg = `Document not found: ${file}`;
        if (result.similarFiles.length > 0) {
          msg += `\n\nDid you mean one of these?\n${result.similarFiles.map(s => `  - ${s}`).join('\n')}`;
        }
        return {
          content: [{ type: "text", text: msg }],
          isError: true,
        };
      }

      const body = await store.getDocumentBody(result.filepath, { fromLine: parsedFromLine, maxLines }) ?? "";
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
        maxBytes: z.number().optional().default(10240).describe("Skip files larger than this (default: 10240 = 10KB)"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
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

  // ---------------------------------------------------------------------------
  // Tool: collections (List all collections)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "collections",
    {
      title: "List Collections",
      description: "List all collections with their paths, glob patterns, document counts, and last modified dates.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {},
    },
    async () => {
      const collections = await store.listCollections();

      if (collections.length === 0) {
        return {
          content: [{ type: "text", text: "No collections configured." }],
          structuredContent: { collections: [] },
        };
      }

      const lines = [`Collections (${collections.length}):`];
      for (const col of collections) {
        const pattern = col.glob_pattern || "**/*.md";
        lines.push(`  - ${col.name}: ${col.pwd} (${pattern}, ${col.doc_count} docs, last modified: ${col.last_modified || "never"})`);
      }

      return {
        content: [{ type: "text", text: lines.join('\n') }],
        structuredContent: { collections },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: add_collection (Add a new collection)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "add_collection",
    {
      title: "Add Collection",
      description: "Add a new collection by pointing it at a directory path. Optionally specify a glob pattern and ignore patterns.",
      inputSchema: {
        name: z.string().describe("Name for the new collection"),
        path: z.string().describe("Filesystem path to the collection directory"),
        pattern: z.string().optional().describe("Glob pattern for matching files (default: '**/*.md')"),
        ignore: z.array(z.string()).optional().describe("List of paths/patterns to ignore"),
      },
    },
    async ({ name, path, pattern, ignore }) => {
      await store.addCollection(name, { path, pattern, ignore });
      return {
        content: [{ type: "text", text: `Collection "${name}" added at ${path}. Run update_index to index documents.` }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: remove_collection (Remove a collection)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "remove_collection",
    {
      title: "Remove Collection",
      description: "Remove a collection by name. This removes it from the index but does not delete files on disk.",
      inputSchema: {
        name: z.string().describe("Name of the collection to remove"),
      },
    },
    async ({ name }) => {
      const removed = await store.removeCollection(name);
      if (removed) {
        return {
          content: [{ type: "text", text: `Collection "${name}" removed.` }],
        };
      }
      return {
        content: [{ type: "text", text: `Collection "${name}" not found.` }],
        isError: true,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: rename_collection (Rename a collection)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "rename_collection",
    {
      title: "Rename Collection",
      description: "Rename an existing collection.",
      inputSchema: {
        old_name: z.string().describe("Current name of the collection"),
        new_name: z.string().describe("New name for the collection"),
      },
    },
    async ({ old_name, new_name }) => {
      const renamed = await store.renameCollection(old_name, new_name);
      if (renamed) {
        return {
          content: [{ type: "text", text: `Collection "${old_name}" renamed to "${new_name}".` }],
        };
      }
      return {
        content: [{ type: "text", text: `Collection "${old_name}" not found.` }],
        isError: true,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: contexts (List all contexts)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "contexts",
    {
      title: "List Contexts",
      description: "List all path contexts across collections, plus the global context if set.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {},
    },
    async () => {
      const contexts = await store.listContexts();
      const globalCtx = await store.getGlobalContext();

      const lines: string[] = [];

      if (globalCtx) {
        lines.push(`Global context: ${globalCtx}`);
      }

      if (contexts.length === 0 && !globalCtx) {
        return {
          content: [{ type: "text", text: "No contexts configured." }],
          structuredContent: { globalContext: undefined, contexts: [] },
        };
      }

      if (contexts.length > 0) {
        lines.push(`Path contexts (${contexts.length}):`);
        for (const ctx of contexts) {
          lines.push(`  - ${ctx.collection}:${ctx.path} — ${ctx.context}`);
        }
      }

      return {
        content: [{ type: "text", text: lines.join('\n') }],
        structuredContent: { globalContext: globalCtx ?? null, contexts },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: add_context (Add context to a collection path)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "add_context",
    {
      title: "Add Context",
      description: "Add context text for a path within a collection. This helps the LLM understand what content is in that area.",
      inputSchema: {
        collection: z.string().describe("Collection name to add context to"),
        path: z.string().default("/").describe("Path prefix within the collection (default: '/')"),
        context: z.string().describe("Descriptive context text for this path"),
      },
    },
    async ({ collection, path, context }) => {
      const added = await store.addContext(collection, path, context);
      if (added) {
        return {
          content: [{ type: "text", text: `Context added for ${collection}:${path}.` }],
        };
      }
      return {
        content: [{ type: "text", text: `Failed to add context for ${collection}:${path}. Collection may not exist.` }],
        isError: true,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: remove_context (Remove context from a collection path)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "remove_context",
    {
      title: "Remove Context",
      description: "Remove context from a specific path within a collection.",
      inputSchema: {
        collection: z.string().describe("Collection name"),
        path: z.string().describe("Path prefix to remove context from"),
      },
    },
    async ({ collection, path }) => {
      const removed = await store.removeContext(collection, path);
      if (removed) {
        return {
          content: [{ type: "text", text: `Context removed from ${collection}:${path}.` }],
        };
      }
      return {
        content: [{ type: "text", text: `No context found at ${collection}:${path}.` }],
        isError: true,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: update_index (Re-index collections)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "update_index",
    {
      title: "Update Index",
      description: "Re-index collections by scanning the filesystem for new, changed, or removed files. If no collections specified, re-indexes all.",
      inputSchema: {
        collections: z.array(z.string()).optional().describe("Specific collections to re-index (default: all)"),
      },
    },
    async ({ collections }) => {
      const result = await store.update({
        ...(collections && collections.length > 0 ? { collections } : {}),
      });

      const summary = [
        `Index updated:`,
        `  Collections: ${result.collections}`,
        `  Indexed: ${result.indexed}`,
        `  Updated: ${result.updated}`,
        `  Unchanged: ${result.unchanged}`,
        `  Removed: ${result.removed}`,
        `  Needs embedding: ${result.needsEmbedding}`,
      ];

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: result,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: embed (Generate vector embeddings)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "embed",
    {
      title: "Generate Embeddings",
      description: "Generate vector embeddings for documents that need them. Requires node-llama-cpp and a model. Use force=true to re-embed all documents.",
      inputSchema: {
        force: z.boolean().optional().default(false).describe("Re-embed all documents, not just new ones (default: false)"),
        model: z.string().optional().describe("Embedding model to use (default: embeddinggemma)"),
      },
    },
    async ({ force, model }) => {
      const result = await store.embed({
        ...(force ? { force } : {}),
        ...(model ? { model } : {}),
      });

      const summary = [
        `Embedding complete:`,
        `  Documents processed: ${result.docsProcessed}`,
        `  Chunks embedded: ${result.chunksEmbedded}`,
        `  Errors: ${result.errors}`,
        `  Duration: ${(result.durationMs / 1000).toFixed(1)}s`,
      ];

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: result,
      };
    }
  );

  return server;
}

// =============================================================================
// Transport: stdio (default)
// =============================================================================

export async function startMcpServer(): Promise<void> {
  const configPath = getConfigPath();
  const store = await createStore({
    dbPath: getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  const server = await createMcpServer(store);
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
 * Binds to localhost only. Returns a handle for shutdown and port discovery.
 */
export async function startMcpHttpServer(port: number, options?: { quiet?: boolean }): Promise<HttpServerHandle> {
  const configPath = getConfigPath();
  const store = await createStore({
    dbPath: getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });

  // Pre-fetch default collection names for REST endpoint
  const defaultCollectionNames = await store.getDefaultCollectionNames();

  // Session map: each client gets its own McpServer + Transport pair (MCP spec requirement).
  // The store is shared — it's stateless SQLite, safe for concurrent access.
  const sessions = new Map<string, WebStandardStreamableHTTPServerTransport>();

  // Reindex mutex - prevents concurrent reindexing
  let reindexInProgress: Promise<unknown> | null = null;
  const isReindexing = () => reindexInProgress !== null;

  async function createSession(): Promise<WebStandardStreamableHTTPServerTransport> {
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      enableJsonResponse: true,
      onsessioninitialized: (sessionId: string) => {
        sessions.set(sessionId, transport);
        log(`${ts()} New session ${sessionId} (${sessions.size} active)`);
      },
    });
    const server = await createMcpServer(store);
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

  /** Extract a human-readable label from a JSON-RPC body */
  function describeRequest(body: any): string {
    const method = body?.method ?? "unknown";
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
        const status = await store.getStatus();
        const body = JSON.stringify({
          status: "ok",
          uptime: Math.floor((Date.now() - startTime) / 1000),
          indexedDocuments: status.totalDocuments,
          needsEmbedding: status.needsEmbedding,
          reindexInProgress: isReindexing(),
          collections: status.collections,
        });
        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(body);
        log(`${ts()} GET /health (${Date.now() - reqStart}ms)`);
        return;
      }

      // REST endpoint: GET /reindex — re-index all collections and generate embeddings
      if (pathname === "/reindex" && nodeReq.method === "GET") {
        if (isReindexing()) {
          nodeRes.writeHead(409, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Re-index already in progress" }));
          return;
        }

        const start = Date.now();
        log(`${ts()} GET /reindex (re-indexing all collections)`);

        const reindexPromise = (async () => {
          const result = await store.update();

          let embedResult;
          if (result.needsEmbedding > 0) {
            log(`${ts()} GET /reindex (embedding ${result.needsEmbedding} docs)`);
            embedResult = await store.embed();
          }

          return { result, embedResult };
        })();
        reindexInProgress = reindexPromise;

        try {
          const { result, embedResult } = await reindexPromise;

          const body = JSON.stringify({
            collections: result.collections,
            indexed: result.indexed,
            updated: result.updated,
            unchanged: result.unchanged,
            removed: result.removed,
            docsProcessed: embedResult?.docsProcessed ?? 0,
            chunksEmbedded: embedResult?.chunksEmbedded ?? 0,
            embedErrors: embedResult?.errors ?? 0,
            durationMs: Date.now() - start,
          });
          nodeRes.writeHead(200, { "Content-Type": "application/json" });
          nodeRes.end(body);
          log(`${ts()} GET /reindex complete (${Date.now() - start}ms)`);
        } finally {
          reindexInProgress = null;
        }
        return;
      }

      // REST endpoint: POST /collections — add a collection
      if (pathname === "/collections" && nodeReq.method === "POST") {
        const rawBody = await collectBody(nodeReq);
        const params = JSON.parse(rawBody);

        if (!params.name || !params.path) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Missing required fields: name, path" }));
          return;
        }

        log(`${ts()} POST /collections (name: ${params.name}, path: ${params.path})`);

        try {
          await store.addCollection(params.name, {
            path: params.path,
            pattern: params.pattern || "**/*.md",
            ignore: params.ignore,
          });

          // Trigger reindex for the new collection
          const result = await store.update({ collections: [params.name] });

          const body = JSON.stringify({
            name: params.name,
            path: params.path,
            pattern: params.pattern || "**/*.md",
            indexed: result.indexed,
            updated: result.updated,
          });
          nodeRes.writeHead(201, { "Content-Type": "application/json" });
          nodeRes.end(body);
        } catch (e: any) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: e.message }));
        }
        return;
      }

      // REST endpoint: POST /search — structured search without MCP protocol
      // REST endpoint: POST /query (alias: /search) — structured search without MCP protocol
      if ((pathname === "/query" || pathname === "/search") && nodeReq.method === "POST") {
        const rawBody = await collectBody(nodeReq);
        const params = JSON.parse(rawBody);

        // Validate required fields
        if (!params.searches || !Array.isArray(params.searches)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Missing required field: searches (array)" }));
          return;
        }

        // Map to internal format
        const queries: ExpandedQuery[] = params.searches.map((s: any) => ({
          type: s.type as 'lex' | 'vec' | 'hyde',
          query: String(s.query || ""),
        }));

        // Use default collections if none specified
        const effectiveCollections = params.collections ?? defaultCollectionNames;

        const results = await store.search({
          queries,
          collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
          limit: params.limit ?? 10,
          minScore: params.minScore ?? 0,
          intent: params.intent,
        });

        // Use first lex or vec query for snippet extraction
        const primaryQuery = params.searches.find((s: any) => s.type === 'lex')?.query
          || params.searches.find((s: any) => s.type === 'vec')?.query
          || params.searches[0]?.query || "";

        const formatted = results.map(r => {
          const { line, snippet } = extractSnippet(r.bestChunk, primaryQuery, 300);
          return {
            docid: `#${r.docid}`,
            file: r.displayPath,
            title: r.title,
            score: Math.round(r.score * 100) / 100,
            context: r.context,
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

  await new Promise<void>((resolve, reject) => {
    httpServer.on("error", reject);
    const host = process.env.QMD_MCP_HOST || "0.0.0.0";
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

  const displayHost = process.env.QMD_MCP_HOST || "0.0.0.0";
  log(`QMD MCP server listening on http://${displayHost}:${actualPort}/mcp`);
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer().catch(console.error);
}
