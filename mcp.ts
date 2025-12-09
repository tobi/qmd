#!/usr/bin/env bun
/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
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
import { searchResultsToMcpCsv } from "./formatter.js";

export async function startMcpServer(): Promise<void> {
  // Open database once at startup - keep it open for the lifetime of the server
  const store = createStore();

  const server = new McpServer({
    name: "qmd",
    version: "1.0.0",
  });

  // Register resource template for qmd:// URIs
  // This allows clients to list and read documents via the MCP resources API
  server.registerResource(
    "document",
    new ResourceTemplate("qmd://{path}", {
      list: async () => {
        // List all indexed documents
        const docs = store.db.prepare(`
          SELECT display_path, title
          FROM documents
          WHERE active = 1
          ORDER BY modified_at DESC
          LIMIT 1000
        `).all() as { display_path: string; title: string }[];

        return {
          resources: docs.map(doc => ({
            uri: `qmd://${encodeURIComponent(doc.display_path)}`,
            name: doc.title || doc.display_path,
            mimeType: "text/markdown",
          })),
        };
      },
    }),
    {
      title: "QMD Document",
      description: "A markdown document from your QMD knowledge base",
      mimeType: "text/markdown",
    },
    async (uri, { path }) => {
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const decodedPath = decodeURIComponent(path);

      // Find document by display_path
      let doc = store.db.prepare(`SELECT filepath, display_path, body FROM documents WHERE display_path = ? AND active = 1`).get(decodedPath) as { filepath: string; display_path: string; body: string } | null;

      // Try suffix match if exact match fails
      if (!doc) {
        doc = store.db.prepare(`SELECT filepath, display_path, body FROM documents WHERE display_path LIKE ? AND active = 1 LIMIT 1`).get(`%${decodedPath}`) as { filepath: string; display_path: string; body: string } | null;
      }

      if (!doc) {
        return { contents: [{ uri: uri.href, text: `Document not found: ${decodedPath}` }] };
      }

      const context = store.getContextForFile(doc.filepath);

      let text = doc.body;
      if (context) {
        text = `<!-- Context: ${context} -->\n\n` + text;
      }

      return {
        contents: [{
          uri: uri.href,
          mimeType: "text/markdown",
          text,
        }],
      };
    }
  );

  // Register the query prompt - describes ideal usage
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

### 1. qmd_search (Fast keyword search)
Best for: Finding documents with specific keywords or phrases.
- Uses BM25 full-text search
- Fast, no LLM required
- Good for exact matches
- Use \`collection\` parameter to filter to a specific collection

### 2. qmd_vsearch (Semantic search)
Best for: Finding conceptually related content even without exact keyword matches.
- Uses vector embeddings
- Understands meaning and context
- Good for "how do I..." or conceptual queries
- Use \`collection\` parameter to filter to a specific collection

### 3. qmd_query (Hybrid search - highest quality)
Best for: Important searches where you want the best results.
- Combines keyword + semantic search
- Expands your query with variations
- Re-ranks results with LLM
- Slower but most accurate
- Use \`collection\` parameter to filter to a specific collection

### 4. qmd_get (Retrieve document)
Best for: Getting the full content of a single document you found.
- Use the file path from search results
- Supports line ranges: \`file.md:100\` or fromLine/maxLines parameters
- Suggests similar files if not found

### 5. qmd_multi_get (Retrieve multiple documents)
Best for: Getting content from multiple files at once.
- Use glob patterns: \`journals/2025-05*.md\`
- Or comma-separated: \`file1.md, file2.md\`
- Skips files over maxBytes (default 10KB) - use qmd_get for large files

### 6. qmd_status (Index info)
Shows collection info, document counts, and embedding status.

## Resources

You can also access documents directly via the \`qmd://\` URI scheme:
- List all documents: \`resources/list\`
- Read a document: \`resources/read\` with uri \`qmd://path/to/file.md\`

## Search Strategy

1. **Start with qmd_search** for quick keyword lookups
2. **Use qmd_vsearch** when keywords aren't working or for conceptual queries
3. **Use qmd_query** for important searches or when you need high confidence
4. **Use qmd_get** to retrieve a single full document
5. **Use qmd_multi_get** to batch retrieve multiple related files

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

  // Tool: search (BM25 full-text)
  server.registerTool(
    "qmd_search",
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
      // Resolve collection filter
      let collectionId: number | undefined;
      if (collection) {
        collectionId = store.getCollectionIdByName(collection) ?? undefined;
        if (collectionId === undefined) {
          return { content: [{ type: "text", text: `Error: Collection not found: ${collection}` }] };
        }
      }

      const results = store.searchFTS(query, limit || 10, collectionId);
      const filtered = results
        .filter(r => r.score >= (minScore || 0))
        .map(r => ({
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: store.getContextForFile(r.file),
          snippet: extractSnippet(r.body, query, 300, r.chunkPos).snippet,
        }));

      return {
        content: [
          {
            type: "text",
            mimeType: "text/csv",
            text: searchResultsToMcpCsv(filtered),
          },
        ],
      };
    }
  );

  // Tool: vsearch (Vector semantic search)
  server.registerTool(
    "qmd_vsearch",
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
      // Resolve collection filter
      let collectionId: number | undefined;
      if (collection) {
        collectionId = store.getCollectionIdByName(collection) ?? undefined;
        if (collectionId === undefined) {
          return { content: [{ type: "text", text: `Error: Collection not found: ${collection}` }] };
        }
      }

      const tableExists = store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
      if (!tableExists) {
        return {
          content: [{ type: "text", text: "Error: Vector index not found. Run 'qmd embed' first to create embeddings." }],
        };
      }

      // Expand query
      const queries = await store.expandQuery(query, DEFAULT_QUERY_MODEL);

      // Collect results
      const allResults = new Map<string, { file: string; displayPath: string; title: string; body: string; score: number }>();
      for (const q of queries) {
        const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, limit || 10, collectionId);
        for (const r of vecResults) {
          const existing = allResults.get(r.file);
          if (!existing || r.score > existing.score) {
            allResults.set(r.file, { file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score });
          }
        }
      }

      const filtered = Array.from(allResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit || 10)
        .filter(r => r.score >= (minScore || 0.3))
        .map(r => ({
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: store.getContextForFile(r.file),
          snippet: extractSnippet(r.body, query, 300).snippet,
        }));

      return {
        content: [
          {
            type: "text",
            mimeType: "text/csv",
            text: searchResultsToMcpCsv(filtered),
          },
        ],
      };
    }
  );

  // Tool: query (Hybrid with reranking)
  server.registerTool(
    "qmd_query",
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
      // Resolve collection filter
      let collectionId: number | undefined;
      if (collection) {
        collectionId = store.getCollectionIdByName(collection) ?? undefined;
        if (collectionId === undefined) {
          return { content: [{ type: "text", text: `Error: Collection not found: ${collection}` }] };
        }
      }

      // Expand query
      const queries = await store.expandQuery(query, DEFAULT_QUERY_MODEL);

      // Collect ranked lists
      const rankedLists: RankedResult[][] = [];
      const hasVectors = !!store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

      for (const q of queries) {
        const ftsResults = store.searchFTS(q, 20, collectionId);
        if (ftsResults.length > 0) {
          rankedLists.push(ftsResults.map(r => ({ file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score })));
        }
        if (hasVectors) {
          const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, 20, collectionId);
          if (vecResults.length > 0) {
            rankedLists.push(vecResults.map(r => ({ file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score })));
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

      const finalResults = reranked.map(r => {
        const rrfRank = rrfRankMap.get(r.file) || candidates.length;
        let rrfWeight: number;
        if (rrfRank <= 3) rrfWeight = 0.75;
        else if (rrfRank <= 10) rrfWeight = 0.60;
        else rrfWeight = 0.40;
        const rrfScore = 1 / rrfRank;
        const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;
        const candidate = candidateMap.get(r.file);
        return {
          file: candidate?.displayPath || "",
          title: candidate?.title || "",
          score: Math.round(blendedScore * 100) / 100,
          context: store.getContextForFile(r.file),
          snippet: extractSnippet(candidate?.body || "", query, 300).snippet,
        };
      }).filter(r => r.score >= (minScore || 0)).slice(0, limit || 10);

      return {
        content: [
          {
            type: "text",
            mimeType: "text/csv",
            text: searchResultsToMcpCsv(finalResults),
          },
        ],
      };
    }
  );

  // Tool: get (Retrieve document)
  server.registerTool(
    "qmd_get",
    {
      title: "Get Document",
      description: "Retrieve the full content of a document by its file path. Use paths from search results. Suggests similar files if not found.",
      inputSchema: {
        file: z.string().describe("File path from search results (e.g., 'pages/meeting.md' or 'pages/meeting.md:100' to start at line 100)"),
        fromLine: z.number().optional().describe("Start from this line number (1-indexed)"),
        maxLines: z.number().optional().describe("Maximum number of lines to return"),
      },
    },
    async ({ file, fromLine, maxLines }) => {
      const result = store.getDocument(file, fromLine, maxLines);

      if ("error" in result) {
        let msg = `Error: Document not found: ${file}`;
        if (result.similarFiles.length > 0) {
          msg += `\n\nDid you mean one of these?\n${result.similarFiles.map(s => `  - ${s}`).join('\n')}`;
        }
        return { content: [{ type: "text", text: msg }] };
      }

      let text = result.body;
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        content: [{
          type: "resource",
          resource: {
            uri: `qmd://${result.displayPath}`,
            mimeType: "text/markdown",
            text,
          },
        }],
      };
    }
  );

  // Tool: multi-get (Retrieve multiple documents)
  server.registerTool(
    "qmd_multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.",
      inputSchema: {
        pattern: z.string().describe("Glob pattern or comma-separated list of file paths"),
        maxLines: z.number().optional().describe("Maximum lines per file"),
        maxBytes: z.number().optional().default(10240).describe("Skip files larger than this (default: 10240 = 10KB)"),
      },
    },
    async ({ pattern, maxLines, maxBytes }) => {
      const { files, errors } = store.getMultipleDocuments(pattern, maxLines, maxBytes || DEFAULT_MULTI_GET_MAX_BYTES);

      if (files.length === 0 && errors.length === 0) {
        return { content: [{ type: "text", text: `No files matched pattern: ${pattern}` }] };
      }

      const content: ({ type: "text"; text: string } | { type: "resource"; resource: { uri: string; mimeType: string; text: string } })[] = [];

      if (errors.length > 0) {
        content.push({ type: "text", text: `Errors:\n${errors.join('\n')}` });
      }

      for (const file of files) {
        if (file.skipped) {
          content.push({
            type: "text",
            text: `[SKIPPED: ${file.displayPath} - ${file.skipReason}. Use 'qmd_get' with file="${file.displayPath}" to retrieve.]`,
          });
          continue;
        }

        let text = file.body;
        if (file.context) {
          text = `<!-- Context: ${file.context} -->\n\n` + text;
        }

        content.push({
          type: "resource",
          resource: {
            uri: `qmd://${file.displayPath}`,
            mimeType: "text/markdown",
            text,
          },
        });
      }

      return { content };
    }
  );

  // Tool: status (Index status)
  server.registerTool(
    "qmd_status",
    {
      title: "Index Status",
      description: "Show the status of the QMD index: collections, document counts, and health information.",
      inputSchema: {},
    },
    async () => {
      const status = store.getStatus();

      return {
        content: [{ type: "text", text: JSON.stringify(status, null, 2) }],
      };
    }
  );

  // Connect via stdio
  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Note: Database stays open - it will be closed when the process exits
}

// Run if this is the main module
if (import.meta.main) {
  startMcpServer().catch(console.error);
}
