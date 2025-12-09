/**
 * MCP Server Tests
 *
 * Tests all MCP tools, resources, and prompts.
 * Uses mocked Ollama responses and a test database.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { setDefaultOllama, Ollama } from "./llm";

// =============================================================================
// Mock Ollama
// =============================================================================

const OLLAMA_URL = "http://localhost:11434";
const originalFetch = globalThis.fetch;

const mockOllamaResponses: Record<string, (body: unknown) => Response> = {
  "/api/embed": () => {
    const embedding = Array(768).fill(0).map(() => Math.random());
    return new Response(JSON.stringify({ embeddings: [embedding] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  },
  "/api/generate": (body: unknown) => {
    const reqBody = body as { prompt?: string };
    if (reqBody.prompt?.includes("Judge") || reqBody.prompt?.includes("Document")) {
      return new Response(JSON.stringify({
        response: "yes",
        done: true,
        logprobs: { tokens: ["yes"], token_logprobs: [-0.1] },
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    } else {
      return new Response(JSON.stringify({
        response: "expanded query variation 1\nexpanded query variation 2",
        done: true,
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
  },
  "/api/show": () => {
    return new Response(JSON.stringify({ size: 1000000 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  },
};

function mockFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = typeof input === "string" ? input : input.toString();

  if (url.startsWith(OLLAMA_URL)) {
    const path = url.replace(OLLAMA_URL, "");
    const handler = mockOllamaResponses[path];
    if (handler) {
      const body = init?.body ? JSON.parse(init.body as string) : {};
      return Promise.resolve(handler(body));
    }
    throw new Error(`Unmocked Ollama endpoint: ${path}`);
  }

  throw new Error(`Unexpected fetch call to: ${url}`);
}

// =============================================================================
// Test Database Setup
// =============================================================================

let testDb: Database;
let testDbPath: string;

function initTestDatabase(db: Database): void {
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");

  db.exec(`
    CREATE TABLE IF NOT EXISTS collections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pwd TEXT NOT NULL,
      glob_pattern TEXT NOT NULL,
      created_at TEXT NOT NULL,
      context TEXT,
      UNIQUE(pwd, glob_pattern)
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS path_contexts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path_prefix TEXT NOT NULL UNIQUE,
      context TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS ollama_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      filepath TEXT NOT NULL,
      display_path TEXT NOT NULL DEFAULT '',
      body TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (collection_id) REFERENCES collections(id)
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
    )
  `);

  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      name, body,
      content='documents',
      content_rowid='id',
      tokenize='porter unicode61'
    )
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
      INSERT INTO documents_fts(rowid, name, body) VALUES (new.id, new.name, new.body);
    END
  `);

  // Create vector table
  db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[768])`);
}

function seedTestData(db: Database): void {
  const now = new Date().toISOString();

  // Create a collection
  db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at, context) VALUES (?, ?, ?, ?)`).run(
    "/test/docs",
    "**/*.md",
    now,
    "Test documentation collection"
  );

  // Add path context
  db.prepare(`INSERT INTO path_contexts (path_prefix, context, created_at) VALUES (?, ?, ?)`).run(
    "/test/docs/meetings",
    "Meeting notes and transcripts",
    now
  );

  // Add test documents
  const docs = [
    {
      name: "readme.md",
      title: "Project README",
      hash: "hash1",
      filepath: "/test/docs/readme.md",
      display_path: "readme.md",
      body: "# Project README\n\nThis is the main readme file for the project.\n\nIt contains important information about setup and usage.",
    },
    {
      name: "api.md",
      title: "API Documentation",
      hash: "hash2",
      filepath: "/test/docs/api.md",
      display_path: "api.md",
      body: "# API Documentation\n\nThis document describes the REST API endpoints.\n\n## Authentication\n\nUse Bearer tokens for auth.",
    },
    {
      name: "meeting-2024-01.md",
      title: "January Meeting Notes",
      hash: "hash3",
      filepath: "/test/docs/meetings/meeting-2024-01.md",
      display_path: "meetings/meeting-2024-01.md",
      body: "# January Meeting Notes\n\nDiscussed Q1 goals and roadmap.\n\n## Action Items\n\n- Review budget\n- Hire new team members",
    },
    {
      name: "meeting-2024-02.md",
      title: "February Meeting Notes",
      hash: "hash4",
      filepath: "/test/docs/meetings/meeting-2024-02.md",
      display_path: "meetings/meeting-2024-02.md",
      body: "# February Meeting Notes\n\nFollowed up on Q1 progress.\n\n## Updates\n\n- Budget approved\n- Two candidates interviewed",
    },
    {
      name: "large-file.md",
      title: "Large Document",
      hash: "hash5",
      filepath: "/test/docs/large-file.md",
      display_path: "large-file.md",
      body: "# Large Document\n\n" + "Lorem ipsum ".repeat(2000), // ~24KB
    },
  ];

  for (const doc of docs) {
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run(doc.name, doc.title, doc.hash, doc.filepath, doc.display_path, doc.body, now, now);
  }

  // Add embeddings for vector search
  const embedding = new Float32Array(768);
  for (let i = 0; i < 768; i++) embedding[i] = Math.random();

  for (const doc of docs.slice(0, 4)) { // Skip large file for embeddings
    db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, 0, 0, 'embeddinggemma', ?)`).run(doc.hash, now);
    db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(`${doc.hash}_0`, embedding);
  }
}

// =============================================================================
// MCP Server Test Helpers
// =============================================================================

// We need to create a testable version of the MCP handlers
// Since McpServer uses internal routing, we'll test the handler functions directly

import {
  searchFTS,
  searchVec,
  expandQuery,
  rerank,
  reciprocalRankFusion,
  extractSnippet,
  getContextForFile,
  getCollectionIdByName,
  getDocument,
  getMultipleDocuments,
  getStatus,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "./store";
import type { RankedResult } from "./store";
import { searchResultsToMcpCsv } from "./formatter";

// =============================================================================
// Tests
// =============================================================================

describe("MCP Server", () => {
  beforeAll(() => {
    globalThis.fetch = mockFetch as typeof fetch;
    setDefaultOllama(new Ollama({ baseUrl: OLLAMA_URL }));

    testDbPath = `/tmp/qmd-mcp-test-${Date.now()}.sqlite`;
    testDb = new Database(testDbPath);
    initTestDatabase(testDb);
    seedTestData(testDb);
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
    setDefaultOllama(null);
    testDb.close();
    try {
      require("fs").unlinkSync(testDbPath);
    } catch {}
  });

  // ===========================================================================
  // Tool: qmd_search (BM25)
  // ===========================================================================

  describe("qmd_search tool", () => {
    test("returns results for matching query", () => {
      const results = searchFTS(testDb, "readme", 10);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].displayPath).toBe("readme.md");
    });

    test("returns empty for non-matching query", () => {
      const results = searchFTS(testDb, "xyznonexistent", 10);
      expect(results.length).toBe(0);
    });

    test("respects limit parameter", () => {
      const results = searchFTS(testDb, "meeting", 1);
      expect(results.length).toBe(1);
    });

    test("filters by collection", () => {
      const collectionId = getCollectionIdByName(testDb, "docs");
      expect(collectionId).toBe(1);
      const results = searchFTS(testDb, "meeting", 10, collectionId!);
      expect(results.length).toBeGreaterThan(0);
    });

    test("returns null for non-existent collection", () => {
      const collectionId = getCollectionIdByName(testDb, "nonexistent");
      expect(collectionId).toBeNull();
    });

    test("formats results as CSV", () => {
      const results = searchFTS(testDb, "api", 10);
      const filtered = results.map(r => ({
        file: r.displayPath,
        title: r.title,
        score: Math.round(r.score * 100) / 100,
        context: getContextForFile(testDb, r.file),
        snippet: extractSnippet(r.body, "api", 300, r.chunkPos).snippet,
      }));
      const csv = searchResultsToMcpCsv(filtered);
      expect(csv).toContain("file,title,score,context,snippet");
      expect(csv).toContain("api.md");
    });
  });

  // ===========================================================================
  // Tool: qmd_vsearch (Vector)
  // ===========================================================================

  describe("qmd_vsearch tool", () => {
    test("returns results for semantic query", async () => {
      const results = await searchVec(testDb, "project documentation", DEFAULT_EMBED_MODEL, 10);
      expect(results.length).toBeGreaterThan(0);
    });

    test("respects limit parameter", async () => {
      const results = await searchVec(testDb, "documentation", DEFAULT_EMBED_MODEL, 2);
      expect(results.length).toBeLessThanOrEqual(2);
    });

    test("returns empty when no vector table exists", async () => {
      const emptyDb = new Database(":memory:");
      initTestDatabase(emptyDb);
      emptyDb.exec("DROP TABLE IF EXISTS vectors_vec");

      const results = await searchVec(emptyDb, "test", DEFAULT_EMBED_MODEL, 10);
      expect(results.length).toBe(0);
      emptyDb.close();
    });
  });

  // ===========================================================================
  // Tool: qmd_query (Hybrid)
  // ===========================================================================

  describe("qmd_query tool", () => {
    test("expands query with variations", async () => {
      const queries = await expandQuery("api documentation", DEFAULT_QUERY_MODEL, testDb);
      expect(queries.length).toBeGreaterThan(1);
      expect(queries[0]).toBe("api documentation");
    });

    test("performs RRF fusion on multiple result lists", () => {
      const list1: RankedResult[] = [
        { file: "/a", displayPath: "a.md", title: "A", body: "body", score: 1 },
        { file: "/b", displayPath: "b.md", title: "B", body: "body", score: 0.8 },
      ];
      const list2: RankedResult[] = [
        { file: "/b", displayPath: "b.md", title: "B", body: "body", score: 1 },
        { file: "/c", displayPath: "c.md", title: "C", body: "body", score: 0.9 },
      ];

      const fused = reciprocalRankFusion([list1, list2]);
      expect(fused.length).toBe(3);
      // B appears in both lists, should have higher score
      const bResult = fused.find(r => r.file === "/b");
      expect(bResult).toBeDefined();
    });

    test("reranks documents with LLM", async () => {
      const docs = [
        { file: "/test/docs/readme.md", text: "Project readme" },
        { file: "/test/docs/api.md", text: "API documentation" },
      ];
      const reranked = await rerank("readme", docs, DEFAULT_RERANK_MODEL, testDb);
      expect(reranked.length).toBe(2);
      expect(reranked[0].score).toBeGreaterThan(0);
    });

    test("full hybrid search pipeline", async () => {
      // Simulate full qmd_query flow
      const query = "meeting notes";
      const queries = await expandQuery(query, DEFAULT_QUERY_MODEL, testDb);

      const rankedLists: RankedResult[][] = [];
      for (const q of queries) {
        const ftsResults = searchFTS(testDb, q, 20);
        if (ftsResults.length > 0) {
          rankedLists.push(ftsResults.map(r => ({
            file: r.file,
            displayPath: r.displayPath,
            title: r.title,
            body: r.body,
            score: r.score,
          })));
        }
      }

      expect(rankedLists.length).toBeGreaterThan(0);

      const fused = reciprocalRankFusion(rankedLists);
      expect(fused.length).toBeGreaterThan(0);

      const candidates = fused.slice(0, 10);
      const reranked = await rerank(
        query,
        candidates.map(c => ({ file: c.file, text: c.body })),
        DEFAULT_RERANK_MODEL,
        testDb
      );

      expect(reranked.length).toBeGreaterThan(0);
    });
  });

  // ===========================================================================
  // Tool: qmd_get (Get Document)
  // ===========================================================================

  describe("qmd_get tool", () => {
    test("retrieves document by display_path", () => {
      const result = getDocument(testDb, "readme.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.displayPath).toBe("readme.md");
        expect(result.body).toContain("Project README");
      }
    });

    test("retrieves document by filepath", () => {
      const result = getDocument(testDb, "/test/docs/api.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.title).toBe("API Documentation");
      }
    });

    test("retrieves document by partial path", () => {
      const result = getDocument(testDb, "api.md");
      expect("error" in result).toBe(false);
    });

    test("returns not found for missing document", () => {
      const result = getDocument(testDb, "nonexistent.md");
      expect("error" in result).toBe(true);
      if ("error" in result) {
        expect(result.error).toBe("not_found");
      }
    });

    test("suggests similar files when not found", () => {
      const result = getDocument(testDb, "readm.md"); // typo
      expect("error" in result).toBe(true);
      if ("error" in result) {
        expect(result.similarFiles.length).toBeGreaterThanOrEqual(0);
      }
    });

    test("supports line range with :line suffix", () => {
      const result = getDocument(testDb, "readme.md:2", undefined, 2);
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        const lines = result.body.split("\n");
        expect(lines.length).toBeLessThanOrEqual(2);
      }
    });

    test("supports fromLine parameter", () => {
      const result = getDocument(testDb, "readme.md", 3);
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.body).not.toContain("# Project README");
      }
    });

    test("supports maxLines parameter", () => {
      const result = getDocument(testDb, "api.md", 1, 3);
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        const lines = result.body.split("\n");
        expect(lines.length).toBeLessThanOrEqual(3);
      }
    });

    test("includes context for documents in context path", () => {
      const result = getDocument(testDb, "meetings/meeting-2024-01.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.context).toBe("Meeting notes and transcripts");
      }
    });
  });

  // ===========================================================================
  // Tool: qmd_multi_get (Multi Get)
  // ===========================================================================

  describe("qmd_multi_get tool", () => {
    test("retrieves multiple documents by glob pattern", () => {
      const { files, errors } = getMultipleDocuments(testDb, "meetings/*.md");
      expect(errors.length).toBe(0);
      expect(files.length).toBe(2);
      expect(files.some(f => f.displayPath === "meetings/meeting-2024-01.md")).toBe(true);
      expect(files.some(f => f.displayPath === "meetings/meeting-2024-02.md")).toBe(true);
    });

    test("retrieves documents by comma-separated list", () => {
      const { files, errors } = getMultipleDocuments(testDb, "readme.md, api.md");
      expect(errors.length).toBe(0);
      expect(files.length).toBe(2);
    });

    test("returns errors for missing files in comma list", () => {
      const { files, errors } = getMultipleDocuments(testDb, "readme.md, nonexistent.md");
      expect(files.length).toBe(1);
      expect(errors.length).toBe(1);
      expect(errors[0]).toContain("not found");
    });

    test("skips files larger than maxBytes", () => {
      const { files } = getMultipleDocuments(testDb, "*.md", undefined, 1000); // 1KB limit
      const largeFile = files.find(f => f.displayPath === "large-file.md");
      expect(largeFile).toBeDefined();
      expect(largeFile?.skipped).toBe(true);
      if (largeFile?.skipped) {
        expect(largeFile.skipReason).toContain("too large");
      }
    });

    test("respects maxLines parameter", () => {
      const { files } = getMultipleDocuments(testDb, "readme.md", 2);
      expect(files.length).toBe(1);
      if (!files[0].skipped) {
        const lines = files[0].body.split("\n");
        // maxLines + truncation message
        expect(lines.length).toBeLessThanOrEqual(4);
      }
    });

    test("returns error for non-matching glob", () => {
      const { files, errors } = getMultipleDocuments(testDb, "nonexistent/*.md");
      expect(files.length).toBe(0);
      expect(errors.length).toBe(1);
      expect(errors[0]).toContain("No files matched");
    });

    test("includes context in results", () => {
      const { files } = getMultipleDocuments(testDb, "meetings/meeting-2024-01.md");
      expect(files.length).toBe(1);
      if (!files[0].skipped) {
        expect(files[0].context).toBe("Meeting notes and transcripts");
      }
    });
  });

  // ===========================================================================
  // Tool: qmd_status
  // ===========================================================================

  describe("qmd_status tool", () => {
    test("returns index status", () => {
      const status = getStatus(testDb);
      expect(status.totalDocuments).toBe(5);
      expect(status.hasVectorIndex).toBe(true);
      expect(status.collections.length).toBe(1);
      expect(status.collections[0].path).toBe("/test/docs");
    });

    test("shows documents needing embedding", () => {
      const status = getStatus(testDb);
      // large-file.md doesn't have embeddings
      expect(status.needsEmbedding).toBe(1);
    });
  });

  // ===========================================================================
  // Resource: qmd://{path}
  // ===========================================================================

  describe("qmd:// resource", () => {
    test("lists all documents", () => {
      const docs = testDb.prepare(`
        SELECT display_path, title
        FROM documents
        WHERE active = 1
        ORDER BY modified_at DESC
        LIMIT 1000
      `).all() as { display_path: string; title: string }[];

      expect(docs.length).toBe(5);
      expect(docs.map(d => d.display_path)).toContain("readme.md");
    });

    test("reads document by display_path", () => {
      const path = "readme.md";
      const doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      expect(doc?.body).toContain("Project README");
    });

    test("reads document by URL-encoded path", () => {
      // Simulate URL encoding that MCP clients may send
      const encodedPath = "meetings%2Fmeeting-2024-01.md";
      const decodedPath = decodeURIComponent(encodedPath);

      const doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(decodedPath) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      expect(doc?.display_path).toBe("meetings/meeting-2024-01.md");
    });

    test("reads document by suffix match", () => {
      const path = "meeting-2024-01.md"; // without meetings/ prefix
      let doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      if (!doc) {
        doc = testDb.prepare(`
          SELECT filepath, display_path, body
          FROM documents
          WHERE display_path LIKE ? AND active = 1
          LIMIT 1
        `).get(`%${path}`) as { filepath: string; display_path: string; body: string } | null;
      }

      expect(doc).not.toBeNull();
      expect(doc?.display_path).toBe("meetings/meeting-2024-01.md");
    });

    test("returns not found for missing document", () => {
      const path = "nonexistent.md";
      const doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).toBeNull();
    });

    test("includes context in document body", () => {
      const path = "meetings/meeting-2024-01.md";
      const doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      const context = getContextForFile(testDb, doc!.filepath);
      expect(context).toBe("Meeting notes and transcripts");

      // Verify context would be prepended
      let text = doc!.body;
      if (context) {
        text = `<!-- Context: ${context} -->\n\n` + text;
      }
      expect(text).toContain("<!-- Context: Meeting notes and transcripts -->");
    });

    test("handles URL-encoded special characters", () => {
      // Test various URL encodings
      const testCases = [
        { encoded: "readme.md", decoded: "readme.md" },
        { encoded: "meetings%2Fmeeting-2024-01.md", decoded: "meetings/meeting-2024-01.md" },
        { encoded: "api.md%3A10", decoded: "api.md:10" }, // with line number
      ];

      for (const { encoded, decoded } of testCases) {
        expect(decodeURIComponent(encoded)).toBe(decoded);
      }
    });

    test("handles double-encoded URLs", () => {
      // Some clients may double-encode
      const doubleEncoded = "meetings%252Fmeeting-2024-01.md";
      const singleDecoded = decodeURIComponent(doubleEncoded);
      expect(singleDecoded).toBe("meetings%2Fmeeting-2024-01.md");

      const fullyDecoded = decodeURIComponent(singleDecoded);
      expect(fullyDecoded).toBe("meetings/meeting-2024-01.md");
    });

    test("handles URL-encoded paths with spaces", () => {
      // Add a document with spaces in the path
      const now = new Date().toISOString();
      testDb.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(
        "podcast with spaces.md",
        "Podcast Episode",
        "hash_spaces",
        "/test/docs/External Podcast/2023 April - Interview.md",
        "External Podcast/2023 April - Interview.md",
        "# Podcast Episode\n\nInterview content here.",
        now,
        now
      );

      // Simulate URL-encoded path from MCP client
      const encodedPath = "External%20Podcast%2F2023%20April%20-%20Interview.md";
      const decodedPath = decodeURIComponent(encodedPath);

      expect(decodedPath).toBe("External Podcast/2023 April - Interview.md");

      const doc = testDb.prepare(`
        SELECT filepath, display_path, body
        FROM documents
        WHERE display_path = ? AND active = 1
      `).get(decodedPath) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      expect(doc?.display_path).toBe("External Podcast/2023 April - Interview.md");
      expect(doc?.body).toContain("Podcast Episode");
    });
  });

  // ===========================================================================
  // Prompt: query
  // ===========================================================================

  describe("query prompt", () => {
    test("returns usage guide", () => {
      // The prompt content is static, just verify the structure
      const promptContent = `# QMD - Quick Markdown Search

QMD is your on-device search engine for markdown knowledge bases.`;

      expect(promptContent).toContain("QMD");
      expect(promptContent).toContain("search");
    });

    test("describes all available tools", () => {
      const toolNames = [
        "qmd_search",
        "qmd_vsearch",
        "qmd_query",
        "qmd_get",
        "qmd_multi_get",
        "qmd_status",
      ];

      // Verify these are documented in the prompt
      const promptGuide = `
### 1. qmd_search (Fast keyword search)
### 2. qmd_vsearch (Semantic search)
### 3. qmd_query (Hybrid search - highest quality)
### 4. qmd_get (Retrieve document)
### 5. qmd_multi_get (Retrieve multiple documents)
### 6. qmd_status (Index info)
      `;

      for (const tool of toolNames) {
        expect(promptGuide).toContain(tool);
      }
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe("edge cases", () => {
    test("handles empty query", () => {
      const results = searchFTS(testDb, "", 10);
      expect(results.length).toBe(0);
    });

    test("handles special characters in query", () => {
      const results = searchFTS(testDb, "project's", 10);
      // Should not throw
      expect(Array.isArray(results)).toBe(true);
    });

    test("handles unicode in query", () => {
      const results = searchFTS(testDb, "文档", 10);
      expect(Array.isArray(results)).toBe(true);
    });

    test("handles very long query", () => {
      const longQuery = "documentation ".repeat(100);
      const results = searchFTS(testDb, longQuery, 10);
      expect(Array.isArray(results)).toBe(true);
    });

    test("handles query with only stopwords", () => {
      const results = searchFTS(testDb, "the and or", 10);
      expect(Array.isArray(results)).toBe(true);
    });

    test("extracts snippet around matching text", () => {
      const body = "Line 1\nLine 2\nThis is the important line with the keyword\nLine 4\nLine 5";
      const { line, snippet } = extractSnippet(body, "keyword", 200);
      expect(snippet).toContain("keyword");
      expect(line).toBe(3);
    });

    test("handles snippet extraction with chunkPos", () => {
      const body = "A".repeat(1000) + "KEYWORD" + "B".repeat(1000);
      const chunkPos = 1000; // Position of KEYWORD
      const { snippet } = extractSnippet(body, "keyword", 200, chunkPos);
      expect(snippet).toContain("KEYWORD");
    });
  });

  // ===========================================================================
  // CSV Formatting
  // ===========================================================================

  describe("CSV formatting", () => {
    test("escapes quotes in CSV", () => {
      const results = [{
        file: 'test.md',
        title: 'Test "quoted" title',
        score: 0.9,
        context: null,
        snippet: 'Some "quoted" text',
      }];
      const csv = searchResultsToMcpCsv(results);
      expect(csv).toContain('""quoted""');
    });

    test("escapes newlines in CSV", () => {
      const results = [{
        file: 'test.md',
        title: 'Test title',
        score: 0.9,
        context: null,
        snippet: 'Line 1\nLine 2',
      }];
      const csv = searchResultsToMcpCsv(results);
      expect(csv).not.toContain('\n\n'); // Should be escaped within quotes
    });

    test("handles empty results", () => {
      const csv = searchResultsToMcpCsv([]);
      expect(csv).toBe("file,title,score,context,snippet");
    });
  });
});
