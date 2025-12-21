#!/usr/bin/env bun
/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Follows MCP spec 2025-06-18 for proper response types.
 */

import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  createStore,
  reciprocalRankFusion,
  extractSnippet,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "./store.js";
import type { RankedResult } from "./store.js";

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
    id: number;
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

/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 */
function addLineNumbers(text: string, startLine: number = 1): string {
  const lines = text.split('\n');
  return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}

// =============================================================================
// MCP Server
// =============================================================================

export async function startMcpServer(): Promise<void> {
  // Open database once at startup - keep it open for the lifetime of the server
  const store = createStore();

  const server = new McpServer({
    name: "qmd",
    version: "1.0.0",
  });

  // ---------------------------------------------------------------------------
  // Resource: qmd://{path} - read-only access to documents by path
  // Note: No list() - documents are discovered via search tools
  // ---------------------------------------------------------------------------

  server.registerResource(
    "document",
    new ResourceTemplate("qmd://{+path}", {}),
    {
      title: "QMD Document",
      description: "A markdown document from your QMD knowledge base. Use search tools to discover documents.",
      mimeType: "text/markdown",
    },
    async (uri, { path }) => {
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const decodedPath = decodeURIComponent(path);

      // Parse virtual path: collection/relative/path
      const parts = decodedPath.split('/');
      const collection = parts[0];
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
  // Prompt: query guide
  // ---------------------------------------------------------------------------

  server.registerPrompt(
    "query",
    {
      title: "QMD Query Guide",
      description: "How to effectively search your knowledge base with QMD",
    },
    () => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `# QMD - Quick Markdown Search

QMD is your on-device search engine for markdown knowledge bases. Use it to find information across your notes, documents, and meeting transcripts.

## Available Tools

### 1. search (Fast keyword search)
Best for: Finding documents with specific keywords or phrases.
- Uses BM25 full-text search
- Fast, no LLM required
- Good for exact matches
- Use \`collection\` parameter to filter to a specific collection

### 2. vsearch (Semantic search)
Best for: Finding conceptually related content even without exact keyword matches.
- Uses vector embeddings
- Understands meaning and context
- Good for "how do I..." or conceptual queries
- Use \`collection\` parameter to filter to a specific collection

### 3. query (Hybrid search - highest quality)
Best for: Important searches where you want the best results.
- Combines keyword + semantic search
- Expands your query with variations
- Re-ranks results with LLM
- Slower but most accurate
- Use \`collection\` parameter to filter to a specific collection

### 4. get (Retrieve document)
Best for: Getting the full content of a single document you found.
- Use the file path from search results
- Supports line ranges: \`file.md:100\` or fromLine/maxLines parameters
- Suggests similar files if not found

### 5. multi_get (Retrieve multiple documents)
Best for: Getting content from multiple files at once.
- Use glob patterns: \`journals/2025-05*.md\`
- Or comma-separated: \`file1.md, file2.md\`
- Skips files over maxBytes (default 10KB) - use get for large files

### 6. status (Index info)
Shows collection info, document counts, and embedding status.

## Resources

You can also access documents directly via the \`qmd://\` URI scheme:
- List all documents: \`resources/list\`
- Read a document: \`resources/read\` with uri \`qmd://path/to/file.md\`

## Search Strategy

1. **Start with search** for quick keyword lookups
2. **Use vsearch** when keywords aren't working or for conceptual queries
3. **Use query** for important searches or when you need high confidence
4. **Use get** to retrieve a single full document
5. **Use multi_get** to batch retrieve multiple related files

## Tips

- Use \`minScore: 0.5\` to filter low-relevance results
- Use \`collection: "notes"\` to search only in a specific collection
- Check the "Context" field - it describes what kind of content the file contains
- File paths are relative to their collection (e.g., \`pages/meeting.md\`)
- For glob patterns, match on display_path (e.g., \`journals/2025-*.md\`)`,
          },
        },
      ],
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_search (BM25 full-text)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "search",
    {
      title: "Search (BM25)",
      description: "Fast keyword-based full-text search using BM25. Best for finding documents with specific words or phrases.",
      inputSchema: {
        query: z.string().describe("Search query - keywords or phrases to find"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      // Note: Collection filtering is now done post-search since collections are managed in YAML
      const results = store.searchFTS(query, limit || 10)
        .filter(r => !collection || r.collectionName === collection);
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
  // Tool: qmd_vsearch (Vector semantic search)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "vsearch",
    {
      title: "Vector Search (Semantic)",
      description: "Semantic similarity search using vector embeddings. Finds conceptually related content even without exact keyword matches. Requires embeddings (run 'qmd embed' first).",
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0.3).describe("Minimum relevance score 0-1 (default: 0.3)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const tableExists = store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
      if (!tableExists) {
        return {
          content: [{ type: "text", text: "Vector index not found. Run 'qmd embed' first to create embeddings." }],
          isError: true,
        };
      }

      // Expand query
      const queries = await store.expandQuery(query, DEFAULT_QUERY_MODEL);

      // Collect results (filter by collection after search)
      const allResults = new Map<string, { file: string; displayPath: string; title: string; body: string; score: number; docid: string }>();
      for (const q of queries) {
        const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, limit || 10)
          .then(results => results.filter(r => !collection || r.collectionName === collection));
        for (const r of vecResults) {
          const existing = allResults.get(r.filepath);
          if (!existing || r.score > existing.score) {
            allResults.set(r.filepath, { file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score, docid: r.docid });
          }
        }
      }

      const filtered: SearchResultItem[] = Array.from(allResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit || 10)
        .filter(r => r.score >= (minScore || 0.3))
        .map(r => {
          const { line, snippet } = extractSnippet(r.body || "", query, 300);
          return {
            docid: `#${r.docid}`,
            file: r.displayPath,
            title: r.title,
            score: Math.round(r.score * 100) / 100,
            context: store.getContextForFile(r.file),
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
  // Tool: qmd_query (Hybrid with reranking)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "query",
    {
      title: "Hybrid Query (Best Quality)",
      description: "Highest quality search combining BM25 + vector + query expansion + LLM reranking. Slower but most accurate. Use for important searches.",
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      // Expand query
      const queries = await store.expandQuery(query, DEFAULT_QUERY_MODEL);

      // Collect ranked lists (filter by collection after search)
      const rankedLists: RankedResult[][] = [];
      const docidMap = new Map<string, string>(); // filepath -> docid
      const hasVectors = !!store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

      for (const q of queries) {
        const ftsResults = store.searchFTS(q, 20)
          .filter(r => !collection || r.collectionName === collection);
        if (ftsResults.length > 0) {
          for (const r of ftsResults) docidMap.set(r.filepath, r.docid);
          rankedLists.push(ftsResults.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
        }
        if (hasVectors) {
          const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, 20)
            .then(results => results.filter(r => !collection || r.collectionName === collection));
          if (vecResults.length > 0) {
            for (const r of vecResults) docidMap.set(r.filepath, r.docid);
            rankedLists.push(vecResults.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
          }
        }
      }

      // RRF fusion
      const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
      const fused = reciprocalRankFusion(rankedLists, weights);
      const candidates = fused.slice(0, 30);

      // Rerank
      const reranked = await store.rerank(
        query,
        candidates.map(c => ({ file: c.file, text: c.body })),
        DEFAULT_RERANK_MODEL
      );

      // Blend scores
      const candidateMap = new Map(candidates.map(c => [c.file, { displayPath: c.displayPath, title: c.title, body: c.body }]));
      const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

      const filtered: SearchResultItem[] = reranked.map(r => {
        const rrfRank = rrfRankMap.get(r.file) || candidates.length;
        let rrfWeight: number;
        if (rrfRank <= 3) rrfWeight = 0.75;
        else if (rrfRank <= 10) rrfWeight = 0.60;
        else rrfWeight = 0.40;
        const rrfScore = 1 / rrfRank;
        const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;
        const candidate = candidateMap.get(r.file);
        const { line, snippet } = extractSnippet(candidate?.body || "", query, 300);
        return {
          docid: `#${docidMap.get(r.file) || ""}`,
          file: candidate?.displayPath || "",
          title: candidate?.title || "",
          score: Math.round(blendedScore * 100) / 100,
          context: store.getContextForFile(r.file),
          snippet: addLineNumbers(snippet, line),  // Default to line numbers
        };
      }).filter(r => r.score >= (minScore || 0)).slice(0, limit || 10);

      return {
        content: [{ type: "text", text: formatSearchSummary(filtered, query) }],
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
      if (colonMatch && parsedFromLine === undefined) {
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

  // ---------------------------------------------------------------------------
  // Connect via stdio
  // ---------------------------------------------------------------------------

  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Note: Database stays open - it will be closed when the process exits
}

// Run if this is the main module
if (import.meta.main) {
  startMcpServer().catch(console.error);
}
