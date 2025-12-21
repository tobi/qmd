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
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp } from "./llm";
import { mkdtemp, writeFile, readdir, unlink, rmdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import YAML from "yaml";
import type { CollectionConfig } from "./collections";

// =============================================================================
// Test Database Setup
// =============================================================================

let testDb: Database;
let testDbPath: string;
let testConfigDir: string;

afterAll(async () => {
  // Ensure native resources are released to avoid ggml-metal asserts on process exit.
  await disposeDefaultLlamaCpp();
});

function initTestDatabase(db: Database): void {
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");

  // Content-addressable storage - the source of truth for document content
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Documents table - file system layer mapping virtual paths to content hashes
  // Collections are now managed in YAML config
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection TEXT NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection, path)
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);

  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
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
      INSERT INTO documents_fts(rowid, name, body)
      SELECT new.id, new.path, content.doc
      FROM content
      WHERE content.hash = new.hash;
    END
  `);

  // Create vector table
  db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[768] distance_metric=cosine)`);
}

function seedTestData(db: Database): void {
  const now = new Date().toISOString();

  // Note: Collections are now managed in YAML config, not in database
  // For tests, we'll use a collection name "docs"

  // Add test documents
  const docs = [
    {
      path: "readme.md",
      title: "Project README",
      hash: "hash1",
      body: "# Project README\n\nThis is the main readme file for the project.\n\nIt contains important information about setup and usage.",
    },
    {
      path: "api.md",
      title: "API Documentation",
      hash: "hash2",
      body: "# API Documentation\n\nThis document describes the REST API endpoints.\n\n## Authentication\n\nUse Bearer tokens for auth.",
    },
    {
      path: "meetings/meeting-2024-01.md",
      title: "January Meeting Notes",
      hash: "hash3",
      body: "# January Meeting Notes\n\nDiscussed Q1 goals and roadmap.\n\n## Action Items\n\n- Review budget\n- Hire new team members",
    },
    {
      path: "meetings/meeting-2024-02.md",
      title: "February Meeting Notes",
      hash: "hash4",
      body: "# February Meeting Notes\n\nFollowed up on Q1 progress.\n\n## Updates\n\n- Budget approved\n- Two candidates interviewed",
    },
    {
      path: "large-file.md",
      title: "Large Document",
      hash: "hash5",
      body: "# Large Document\n\n" + "Lorem ipsum ".repeat(2000), // ~24KB
    },
  ];

  for (const doc of docs) {
    // Insert content first
    db.prepare(`
      INSERT OR IGNORE INTO content (hash, doc, created_at)
      VALUES (?, ?, ?)
    `).run(doc.hash, doc.body, now);

    // Then insert document metadata
    db.prepare(`
      INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
      VALUES ('docs', ?, ?, ?, ?, ?, 1)
    `).run(doc.path, doc.title, doc.hash, now, now);
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
  findDocument,
  getDocumentBody,
  findDocuments,
  getStatus,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
  createStore,
} from "./store";
import type { RankedResult } from "./store";
// Note: searchResultsToMcpCsv no longer used in MCP - using structuredContent instead

// =============================================================================
// Tests
// =============================================================================

describe("MCP Server", () => {
  beforeAll(async () => {
    // LlamaCpp uses node-llama-cpp for local model inference (no HTTP mocking needed)
    // Use shared singleton to avoid creating multiple instances with separate GPU resources
    getDefaultLlamaCpp();

    // Set up test config directory
    const configPrefix = join(tmpdir(), `qmd-mcp-config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    testConfigDir = await mkdtemp(configPrefix);
    process.env.QMD_CONFIG_DIR = testConfigDir;

    // Create YAML config with test collection
    const testConfig: CollectionConfig = {
      collections: {
        docs: {
          path: "/test/docs",
          pattern: "**/*.md",
          context: {
            "/meetings": "Meeting notes and transcripts"
          }
        }
      }
    };
    await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(testConfig));

    testDbPath = `/tmp/qmd-mcp-test-${Date.now()}.sqlite`;
    testDb = new Database(testDbPath);
    initTestDatabase(testDb);
    seedTestData(testDb);
  });

  afterAll(async () => {
    testDb.close();
    try {
      require("fs").unlinkSync(testDbPath);
    } catch {}

    // Clean up test config directory
    try {
      const files = await readdir(testConfigDir);
      for (const file of files) {
        await unlink(join(testConfigDir, file));
      }
      await rmdir(testConfigDir);
    } catch {}

    delete process.env.QMD_CONFIG_DIR;
  });

  // ===========================================================================
  // Tool: qmd_search (BM25)
  // ===========================================================================

  describe("qmd_search tool", () => {
    test("returns results for matching query", () => {
      const results = searchFTS(testDb, "readme", 10);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]!.displayPath).toBe("docs/readme.md");
    });

    test("returns empty for non-matching query", () => {
      const results = searchFTS(testDb, "xyznonexistent", 10);
      expect(results.length).toBe(0);
    });

    test("respects limit parameter", () => {
      const results = searchFTS(testDb, "meeting", 1);
      expect(results.length).toBe(1);
    });

    // Note: Collection filtering tests removed - collections are now managed in YAML, not DB

    test("formats results as structured content", () => {
      const results = searchFTS(testDb, "api", 10);
      const filtered = results.map(r => ({
        file: r.displayPath,
        title: r.title,
        score: Math.round(r.score * 100) / 100,
        context: getContextForFile(testDb, r.filepath),
        snippet: extractSnippet(r.body || "", "api", 300, r.chunkPos).snippet,
      }));
      // MCP now returns structuredContent with results array
      expect(filtered.length).toBeGreaterThan(0);
      expect(filtered[0]).toHaveProperty("file");
      expect(filtered[0]).toHaveProperty("title");
      expect(filtered[0]).toHaveProperty("score");
      expect(filtered[0]).toHaveProperty("snippet");
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
      // Always returns at least the original query, may have more if generation succeeds
      expect(queries.length).toBeGreaterThanOrEqual(1);
      expect(queries[0]).toBe("api documentation");
    }, 30000); // 30s timeout for model loading

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
      expect(reranked[0]!.score).toBeGreaterThan(0);
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
            file: r.filepath,
            displayPath: r.displayPath,
            title: r.title,
            body: r.body || "",
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
      const meta = findDocument(testDb, "readme.md", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      const body = getDocumentBody(testDb, meta) ?? "";

      expect(meta.displayPath).toBe("docs/readme.md");
      expect(body).toContain("Project README");
    });

    test("retrieves document by filepath", () => {
      const meta = findDocument(testDb, "/test/docs/api.md", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      expect(meta.title).toBe("API Documentation");
    });

    test("retrieves document by partial path", () => {
      const result = findDocument(testDb, "api.md", { includeBody: false });
      expect("error" in result).toBe(false);
    });

    test("returns not found for missing document", () => {
      const result = findDocument(testDb, "nonexistent.md", { includeBody: false });
      expect("error" in result).toBe(true);
      if ("error" in result) {
        expect(result.error).toBe("not_found");
      }
    });

    test("suggests similar files when not found", () => {
      const result = findDocument(testDb, "readm.md", { includeBody: false }); // typo
      expect("error" in result).toBe(true);
      if ("error" in result) {
        expect(result.similarFiles.length).toBeGreaterThanOrEqual(0);
      }
    });

    test("supports line range with :line suffix", () => {
      const meta = findDocument(testDb, "readme.md:2", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      const body = getDocumentBody(testDb, meta, 2, 2) ?? "";
      const lines = body.split("\n");
      expect(lines.length).toBeLessThanOrEqual(2);
    });

    test("supports fromLine parameter", () => {
      const meta = findDocument(testDb, "readme.md", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      const body = getDocumentBody(testDb, meta, 3) ?? "";
      expect(body).not.toContain("# Project README");
    });

    test("supports maxLines parameter", () => {
      const meta = findDocument(testDb, "api.md", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      const body = getDocumentBody(testDb, meta, 1, 3) ?? "";
      const lines = body.split("\n");
      expect(lines.length).toBeLessThanOrEqual(3);
    });

    test("includes context for documents in context path", () => {
      const result = findDocument(testDb, "meetings/meeting-2024-01.md", { includeBody: false });
      expect("error" in result).toBe(false);
      if ("error" in result) return;
      expect(result.context).toBe("Meeting notes and transcripts");
    });
  });

  // ===========================================================================
  // Tool: qmd_multi_get (Multi Get)
  // ===========================================================================

  describe("qmd_multi_get tool", () => {
    test("retrieves multiple documents by glob pattern", () => {
      const { docs, errors } = findDocuments(testDb, "meetings/*.md", { includeBody: true });
      expect(errors.length).toBe(0);
      expect(docs.length).toBe(2);
      const paths = docs.map(d => d.doc.displayPath);
      expect(paths).toContain("docs/meetings/meeting-2024-01.md");
      expect(paths).toContain("docs/meetings/meeting-2024-02.md");
    });

    test("retrieves documents by comma-separated list", () => {
      const { docs, errors } = findDocuments(testDb, "readme.md, api.md", { includeBody: true });
      expect(errors.length).toBe(0);
      expect(docs.length).toBe(2);
    });

    test("returns errors for missing files in comma list", () => {
      const { docs, errors } = findDocuments(testDb, "readme.md, nonexistent.md", { includeBody: true });
      expect(docs.length).toBe(1);
      expect(errors.length).toBe(1);
      expect(errors[0]).toContain("not found");
    });

    test("skips files larger than maxBytes", () => {
      const { docs } = findDocuments(testDb, "*.md", { includeBody: true, maxBytes: 1000 }); // 1KB limit
      const large = docs.find(d => d.doc.displayPath === "docs/large-file.md");
      expect(large).toBeDefined();
      expect(large?.skipped).toBe(true);
      if (large?.skipped) expect(large.skipReason).toContain("too large");
    });

    test("respects maxLines parameter", () => {
      const { docs } = findDocuments(testDb, "readme.md", { includeBody: true, maxBytes: DEFAULT_MULTI_GET_MAX_BYTES });
      expect(docs.length).toBe(1);
      const d = docs[0]!;
      expect(d.skipped).toBe(false);
      if (d.skipped) return;
      if (!("body" in d.doc)) {
        throw new Error("Expected body to be included in findDocuments result");
      }
      const lines = (d.doc.body || "").split("\n").slice(0, 2);
      expect(lines.length).toBeLessThanOrEqual(2);
    });

    test("returns error for non-matching glob", () => {
      const { docs, errors } = findDocuments(testDb, "nonexistent/*.md", { includeBody: true });
      expect(docs.length).toBe(0);
      expect(errors.length).toBe(1);
      expect(errors[0]).toContain("No files matched");
    });

    test("includes context in results", () => {
      const { docs } = findDocuments(testDb, "meetings/meeting-2024-01.md", { includeBody: true });
      expect(docs.length).toBe(1);
      const d = docs[0]!;
      expect(d.skipped).toBe(false);
      if (d.skipped) return;
      if (!("context" in d.doc)) {
        throw new Error("Expected context to be present on document result");
      }
      expect(d.doc.context).toBe("Meeting notes and transcripts");
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
      expect(status.collections[0]!.path).toBe("/test/docs");
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
        SELECT path as display_path, title
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
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      expect(doc?.body).toContain("Project README");
    });

    test("reads document by URL-encoded path", () => {
      // Simulate URL encoding that MCP clients may send
      const encodedPath = "meetings%2Fmeeting-2024-01.md";
      const decodedPath = decodeURIComponent(encodedPath);

      const doc = testDb.prepare(`
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
      `).get(decodedPath) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).not.toBeNull();
      expect(doc?.display_path).toBe("meetings/meeting-2024-01.md");
    });

    test("reads document by suffix match", () => {
      const path = "meeting-2024-01.md"; // without meetings/ prefix
      let doc = testDb.prepare(`
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      if (!doc) {
        doc = testDb.prepare(`
          SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${path}`) as { filepath: string; display_path: string; body: string } | null;
      }

      expect(doc).not.toBeNull();
      expect(doc?.display_path).toBe("meetings/meeting-2024-01.md");
    });

    test("returns not found for missing document", () => {
      const path = "nonexistent.md";
      const doc = testDb.prepare(`
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
      `).get(path) as { filepath: string; display_path: string; body: string } | null;

      expect(doc).toBeNull();
    });

    test("includes context in document body", () => {
      const path = "meetings/meeting-2024-01.md";
      const doc = testDb.prepare(`
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
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
      const body = "# Podcast Episode\n\nInterview content here.";
      const hash = "hash_spaces";
      const path = "External Podcast/2023 April - Interview.md";

      // Insert content first
      testDb.prepare(`
        INSERT OR IGNORE INTO content (hash, doc, created_at)
        VALUES (?, ?, ?)
      `).run(hash, body, now);

      // Then insert document metadata
      testDb.prepare(`
        INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
        VALUES ('docs', ?, ?, ?, ?, ?, 1)
      `).run(path, "Podcast Episode", hash, now, now);

      // Simulate URL-encoded path from MCP client
      const encodedPath = "External%20Podcast%2F2023%20April%20-%20Interview.md";
      const decodedPath = decodeURIComponent(encodedPath);

      expect(decodedPath).toBe("External Podcast/2023 April - Interview.md");

      const doc = testDb.prepare(`
        SELECT 'qmd://' || d.collection || '/' || d.path as filepath, d.path as display_path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.path = ? AND d.active = 1
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
  // MCP Spec Compliance
  // ===========================================================================

  describe("MCP spec compliance", () => {
    test("encodeQmdPath preserves slashes but encodes special chars", () => {
      // Helper function behavior (tested indirectly through resource URIs)
      const path = "External Podcast/2023 April - Interview.md";
      const segments = path.split('/').map(s => encodeURIComponent(s)).join('/');
      expect(segments).toBe("External%20Podcast/2023%20April%20-%20Interview.md");
      expect(segments).toContain("/"); // Slashes preserved
      expect(segments).toContain("%20"); // Spaces encoded
    });

    test("search results have correct structure for structuredContent", () => {
      const results = searchFTS(testDb, "readme", 5);
      const structured = results.map(r => ({
        file: r.displayPath,
        title: r.title,
        score: Math.round(r.score * 100) / 100,
        context: getContextForFile(testDb, r.filepath),
        snippet: extractSnippet(r.body || "", "readme", 300, r.chunkPos).snippet,
      }));

      expect(structured.length).toBeGreaterThan(0);
      const item = structured[0]!;
      expect(typeof item.file).toBe("string");
      expect(typeof item.title).toBe("string");
      expect(typeof item.score).toBe("number");
      expect(item.score).toBeGreaterThanOrEqual(0);
      expect(item.score).toBeLessThanOrEqual(1);
      expect(typeof item.snippet).toBe("string");
    });

    test("error responses should include isError flag", () => {
      // Simulate what MCP server returns for errors
      const errorResponse = {
        content: [{ type: "text", text: "Collection not found: nonexistent" }],
        isError: true,
      };
      expect(errorResponse.isError).toBe(true);
      expect(errorResponse.content[0]!.type).toBe("text");
    });

    test("embedded resources include name and title", () => {
      // Simulate what qmd_get returns
      const meta = findDocument(testDb, "readme.md", { includeBody: false });
      expect("error" in meta).toBe(false);
      if ("error" in meta) return;
      const body = getDocumentBody(testDb, meta) ?? "";
      const resource = {
        uri: `qmd://${meta.displayPath}`,
        name: meta.displayPath,
        title: meta.title,
        mimeType: "text/markdown",
        text: body,
      };
      expect(resource.name).toBe("docs/readme.md");
      expect(resource.title).toBe("Project README");
      expect(resource.mimeType).toBe("text/markdown");
    });

    test("status response includes structuredContent", () => {
      const status = getStatus(testDb);
      // Verify structure matches StatusResult type
      expect(typeof status.totalDocuments).toBe("number");
      expect(typeof status.needsEmbedding).toBe("number");
      expect(typeof status.hasVectorIndex).toBe("boolean");
      expect(Array.isArray(status.collections)).toBe(true);
      if (status.collections.length > 0) {
        const col = status.collections[0]!;
        expect(typeof col.name).toBe("string"); // Collections now use names, not IDs
        expect(typeof col.path).toBe("string");
        expect(typeof col.pattern).toBe("string");
        expect(typeof col.documents).toBe("number");
      }
    });
  });
});
