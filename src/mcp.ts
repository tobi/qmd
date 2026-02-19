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
import { fileURLToPath } from "url";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { WebStandardStreamableHTTPServerTransport }
  from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  hybridQuery,
  vectorSearchQuery,
  structuredSearch,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "./store.js";
import type { Store, StructuredSubSearch } from "./store.js";
import { getCollection, getGlobalContext } from "./collections.js";
import { disposeDefaultLlamaCpp } from "./llm.js";

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
    path: string;
    pattern: string;
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

// =============================================================================
// MCP Server
// =============================================================================

/**
 * Build dynamic server instructions from actual index state.
 * Injected into the LLM's system prompt via MCP initialize response —
 * gives the LLM immediate context about what's searchable without a tool call.
 */
function buildInstructions(store: Store): string {
  const status = store.getStatus();
  const lines: string[] = [];

  // --- What is this? ---
  const globalCtx = getGlobalContext();
  lines.push(`QMD is your local search engine over ${status.totalDocuments} markdown documents.`);
  if (globalCtx) lines.push(`Context: ${globalCtx}`);

  // --- What's searchable? ---
  if (status.collections.length > 0) {
    lines.push("");
    lines.push("Collections (scope with `collection` parameter):");
    for (const col of status.collections) {
      const collConfig = getCollection(col.name);
      const rootCtx = collConfig?.context?.[""] || collConfig?.context?.["/"];
      const desc = rootCtx ? ` — ${rootCtx}` : "";
      lines.push(`  - "${col.name}" (${col.documents} docs)${desc}`);
    }
  }

  // --- Capability gaps ---
  if (!status.hasVectorIndex) {
    lines.push("");
    lines.push("Note: No vector embeddings. Only `search` (BM25) is available.");
  } else if (status.needsEmbedding > 0) {
    lines.push("");
    lines.push(`Note: ${status.needsEmbedding} documents need embedding. Run \`qmd embed\` to update.`);
  }

  // --- When to use which tool (escalation ladder) ---
  // Tool schemas describe parameters; instructions describe strategy.
  lines.push("");
  lines.push("Search:");
  lines.push("  - `search` (~30ms) — BM25 keyword matching. Fast, exact terms.");
  lines.push("  - `vector_search` (~2s) — semantic search. Finds synonyms and related concepts.");
  lines.push("  - `deep_search` (~10s) — auto-expands query + reranks. Use when you don't know the exact terms.");
  lines.push("  - `structured_search` (~5s) — YOU provide the query variations. Best for complex/nuanced queries.");
  lines.push("");
  lines.push("For structured_search, pass 2-4 sub-searches:");
  lines.push("  - type:'lex' for keyword phrases (BM25)");
  lines.push("  - type:'vec' for semantic questions");
  lines.push("  - type:'hyde' for hypothetical answer snippets");

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
function createMcpServer(store: Store): McpServer {
  const server = new McpServer(
    { name: "qmd", version: "0.9.9" },
    { instructions: buildInstructions(store) },
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
      description: "A markdown document from your QMD knowledge base. Use search tools to discover documents.",
      mimeType: "text/markdown",
    },
    async (uri, { path }) => {
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const pathStr = Array.isArray(path) ? path.join('/') : (path || '');
      const decodedPath = decodeURIComponent(pathStr);

      // Parse virtual path: collection/relative/path
      const parts = decodedPath.split('/');
      const collection = parts[0] || '';
      const relativePath = parts.slice(1).join('/');

      // Find document by collection and path, join with content table
      let doc = store.db.prepare(`
        SELECT d.collection, d.path, d.title, c.doc as body
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
      `).get(collection, relativePath) as { collection: string; path: string; title: string; body: string } | null;

      // Try suffix match if exact match fails
      if (!doc) {
        doc = store.db.prepare(`
          SELECT d.collection, d.path, d.title, c.doc as body
          FROM documents d
          JOIN content c ON c.hash = d.hash
          WHERE d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${relativePath}`) as { collection: string; path: string; title: string; body: string } | null;
      }

      if (!doc) {
        return { contents: [{ uri: uri.href, text: `Document not found: ${decodedPath}` }] };
      }

      // Construct virtual path for context lookup
      const virtualPath = `qmd://${doc.collection}/${doc.path}`;
      const context = store.getContextForFile(virtualPath);

      let text = addLineNumbers(doc.body);  // Default to line numbers
      if (context) {
        text = `<!-- Context: ${context} -->\n\n` + text;
      }

      const displayName = `${doc.collection}/${doc.path}`;
      return {
        contents: [{
          uri: uri.href,
          name: displayName,
          title: doc.title || doc.path,
          mimeType: "text/markdown",
          text,
        }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_search (keyword)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "search",
    {
      title: "Keyword Search",
      description: "Search by keyword. Finds documents containing exact words and phrases in the query.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        query: z.string().describe("Search query - keywords or phrases to find"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const results = store.searchFTS(query, limit || 10, collection);
      const filtered: SearchResultItem[] = results
        .filter(r => r.score >= (minScore || 0))
        .map(r => {
          const { line, snippet } = extractSnippet(r.body || "", query, 300, r.chunkPos);
          return {
            docid: `#${r.docid}`,
            file: r.displayPath,
            title: r.title,
            score: Math.round(r.score * 100) / 100,
            context: store.getContextForFile(r.filepath),
            snippet: addLineNumbers(snippet, line),  // Default to line numbers
          };
        });

      return {
        content: [{ type: "text", text: formatSearchSummary(filtered, query) }],
        structuredContent: { results: filtered },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_vector_search (Vector semantic search)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "vector_search",
    {
      title: "Vector Search",
      description: "Search by meaning. Finds relevant documents even when they use different words than the query — handles synonyms, paraphrases, and related concepts.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0.3).describe("Minimum relevance score 0-1 (default: 0.3)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const results = await vectorSearchQuery(store, query, { collection, limit, minScore });

      if (results.length === 0) {
        // Distinguish "no embeddings" from "no matches" — check if vector table exists
        const tableExists = store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
        if (!tableExists) {
          return {
            content: [{ type: "text", text: "Vector index not found. Run 'qmd embed' first to create embeddings." }],
            isError: true,
          };
        }
      }

      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.body, query, 300);
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
        content: [{ type: "text", text: formatSearchSummary(filtered, query) }],
        structuredContent: { results: filtered },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_deep_search (Deep search with expansion + reranking)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "deep_search",
    {
      title: "Deep Search",
      description: "Deep search. Auto-expands the query into variations, searches each by keyword and meaning, and reranks for top hits across all results.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const results = await hybridQuery(store, query, { collection, limit, minScore });

      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.bestChunk, query, 300);
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
        content: [{ type: "text", text: formatSearchSummary(filtered, query) }],
        structuredContent: { results: filtered },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_structured_search (Pre-expanded queries from LLM)
  // ---------------------------------------------------------------------------

  const subSearchSchema = z.object({
    type: z.enum(['lex', 'vec', 'hyde']).describe(
      "Search type: 'lex' = BM25 keyword search (exact terms, fast), " +
      "'vec' = semantic vector search (meaning-based, finds synonyms/paraphrases), " +
      "'hyde' = hypothetical document (imagine what the answer looks like)"
    ),
    query: z.string().describe("The search query text"),
  });

  server.registerTool(
    "structured_search",
    {
      title: "Structured Search",
      description: `Execute pre-expanded search queries. Skips internal query expansion — you provide the search variations directly.

**When to use:** You're an LLM that can generate better query expansions than a small local model. Pass 2-4 sub-searches for best results.

**Search types:**
- \`lex\`: BM25 keyword search. Use short keyword phrases (2-5 terms). Good for exact terms, names, code identifiers.
- \`vec\`: Semantic vector search. Use natural language questions or descriptions. Finds documents with similar meaning even when vocabulary differs.
- \`hyde\`: Hypothetical document. Write a short passage (~50-100 words) that looks like what you're searching for. Powerful for finding conceptually similar content.

**Example:** To find CAP theorem docs, pass:
- { type: "lex", query: "CAP theorem consistency availability" }
- { type: "vec", query: "what is the tradeoff between data consistency and system availability in distributed systems" }
- { type: "hyde", query: "The CAP theorem states that a distributed system can only guarantee two of three properties: Consistency, Availability, and Partition tolerance." }`,
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        searches: z.array(subSearchSchema).min(1).max(10).describe(
          "Array of sub-searches to execute. Order matters — first search gets higher weight in fusion."
        ),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
        intent: z.string().optional().describe("(Future) Domain intent hint, e.g., 'distributed systems', 'startup finances'"),
      },
    },
    async ({ searches, limit, minScore, collection, intent }) => {
      // Map to internal format
      const subSearches: StructuredSubSearch[] = searches.map(s => ({
        type: s.type,
        query: s.query,
      }));

      const results = await structuredSearch(store, subSearches, {
        collection,
        limit,
        minScore,
        intent,
      });

      // Use first lex or vec query for snippet extraction
      const primaryQuery = searches.find(s => s.type === 'lex')?.query
        || searches.find(s => s.type === 'vec')?.query
        || searches[0]?.query || "";

      const filtered: SearchResultItem[] = results.map(r => {
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

      const result = store.findDocument(lookup, { includeBody: false });

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

      const body = store.getDocumentBody(result, parsedFromLine, maxLines) ?? "";
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
      const { docs, errors } = store.findDocuments(pattern, { includeBody: true, maxBytes: maxBytes || DEFAULT_MULTI_GET_MAX_BYTES });

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
      const status: StatusResult = store.getStatus();

      const summary = [
        `QMD Index Status:`,
        `  Total documents: ${status.totalDocuments}`,
        `  Needs embedding: ${status.needsEmbedding}`,
        `  Vector index: ${status.hasVectorIndex ? 'yes' : 'no'}`,
        `  Collections: ${status.collections.length}`,
      ];

      for (const col of status.collections) {
        summary.push(`    - ${col.path} (${col.documents} docs)`);
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

export async function startMcpServer(): Promise<void> {
  const store = createStore();
  const server = createMcpServer(store);
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
  const store = createStore();
  const mcpServer = createMcpServer(store);
  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
    enableJsonResponse: true,
  });
  await mcpServer.connect(transport);

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
        const body = JSON.stringify({ status: "ok", uptime: Math.floor((Date.now() - startTime) / 1000) });
        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(body);
        log(`${ts()} GET /health (${Date.now() - reqStart}ms)`);
        return;
      }

      // REST endpoint: POST /search — structured search without MCP protocol
      if (pathname === "/search" && nodeReq.method === "POST") {
        const rawBody = await collectBody(nodeReq);
        const params = JSON.parse(rawBody);
        
        // Validate required fields
        if (!params.searches || !Array.isArray(params.searches)) {
          nodeRes.writeHead(400, { "Content-Type": "application/json" });
          nodeRes.end(JSON.stringify({ error: "Missing required field: searches (array)" }));
          return;
        }

        // Map to internal format
        const subSearches: StructuredSubSearch[] = params.searches.map((s: any) => ({
          type: s.type as 'lex' | 'vec' | 'hyde',
          query: String(s.query || ""),
        }));

        const results = await structuredSearch(store, subSearches, {
          collection: params.collection,
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
        log(`${ts()} POST /search ${params.searches.length} queries (${Date.now() - reqStart}ms)`);
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
        const request = new Request(url, { method: "POST", headers, body: rawBody });
        const response = await transport.handleRequest(request, { parsedBody: body });
        nodeRes.writeHead(response.status, Object.fromEntries(response.headers));
        nodeRes.end(Buffer.from(await response.arrayBuffer()));
        log(`${ts()} POST /mcp ${label} (${Date.now() - reqStart}ms)`);
        return;
      }

      if (pathname === "/mcp") {
        const url = `http://localhost:${port}${pathname}`;
        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(nodeReq.headers)) {
          if (typeof v === "string") headers[k] = v;
        }
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
    httpServer.listen(port, "localhost", () => resolve());
  });

  const actualPort = (httpServer.address() as import("net").AddressInfo).port;

  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    await transport.close();
    httpServer.close();
    store.close();
    await disposeDefaultLlamaCpp();
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

  log(`QMD MCP server listening on http://localhost:${actualPort}/mcp`);
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/mcp.ts") || process.argv[1]?.endsWith("/mcp.js")) {
  startMcpServer().catch(console.error);
}
