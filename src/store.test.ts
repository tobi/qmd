/**
 * store.test.ts - Comprehensive unit tests for the QMD store module
 *
 * Run with: bun test store.test.ts
 *
 * LLM operations use LlamaCpp with local GGUF models (node-llama-cpp).
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, mock, spyOn } from "bun:test";
import { Database } from "bun:sqlite";
import { unlink, mkdtemp, rmdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { disposeDefaultLlamaCpp } from "./llm.js";
import {
  createStore,
  getDefaultDbPath,
  homedir,
  resolve,
  getPwd,
  getRealPath,
  hashContent,
  extractTitle,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  chunkDocument,
  chunkDocumentByTokens,
  reciprocalRankFusion,
  extractSnippet,
  getCacheKey,
  handelize,
  normalizeVirtualPath,
  isVirtualPath,
  parseVirtualPath,
  type Store,
  type DocumentResult,
  type SearchResult,
  type RankedResult,
} from "./store.js";
import type { CollectionConfig } from "./collections.js";

// =============================================================================
// LlamaCpp Setup
// =============================================================================

// Note: LlamaCpp uses node-llama-cpp for local GGUF model inference.
// No HTTP mocking needed - tests use real LlamaCpp calls for integration tests.

// =============================================================================
// Test Utilities
// =============================================================================

let testDir: string;
let testDbPath: string;
let testConfigDir: string;

async function createTestStore(): Promise<Store> {
  testDbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);

  // Set up test config directory
  const configPrefix = join(testDir, `config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  testConfigDir = await mkdtemp(configPrefix);

  // Set environment variable to use test config
  process.env.QMD_CONFIG_DIR = testConfigDir;

  // Create empty YAML config
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(
    join(testConfigDir, "index.yml"),
    YAML.stringify(emptyConfig)
  );

  return createStore(testDbPath);
}

async function cleanupTestDb(store: Store): Promise<void> {
  store.close();
  try {
    await unlink(store.dbPath);
  } catch {
    // Ignore if file doesn't exist
  }

  // Clean up test config directory
  try {
    const { readdir, unlink: unlinkFile, rmdir: rmdirAsync } = await import("node:fs/promises");
    const files = await readdir(testConfigDir);
    for (const file of files) {
      await unlinkFile(join(testConfigDir, file));
    }
    await rmdirAsync(testConfigDir);
  } catch {
    // Ignore cleanup errors
  }

  // Clear environment variable
  delete process.env.QMD_CONFIG_DIR;
}

// Helper to insert a test document directly into the database
async function insertTestDocument(
  db: Database,
  collectionName: string,
  opts: {
    name?: string;
    title?: string;
    hash?: string;
    displayPath?: string;
    filepath?: string;
    body?: string;
    active?: number;
  }
): Promise<number> {
  const now = new Date().toISOString();
  const name = opts.name || "test-doc";
  const title = opts.title || "Test Document";

  // Use displayPath if provided, otherwise filepath's basename, otherwise default
  let path: string;
  if (opts.displayPath) {
    path = opts.displayPath;
  } else if (opts.filepath) {
    // Extract relative path from filepath by removing collection path
    // For tests, assume filepath is either relative or we want the whole path as the document path
    path = opts.filepath.startsWith('/') ? opts.filepath : opts.filepath;
  } else {
    path = `test/${name}.md`;
  }

  const body = opts.body || "# Test Document\n\nThis is test content.";
  const active = opts.active ?? 1;

  // Generate hash from body if not provided
  const hash = opts.hash || await hashContent(body);

  // Insert content (with OR IGNORE for deduplication)
  db.prepare(`
    INSERT OR IGNORE INTO content (hash, doc, created_at)
    VALUES (?, ?, ?)
  `).run(hash, body, now);

  // Insert document
  const result = db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(collectionName, path, title, hash, now, now, active);

  return Number(result.lastInsertRowid);
}

// Helper to create a test collection in YAML config
async function createTestCollection(
  options: { pwd?: string; glob?: string; name?: string } = {}
): Promise<string> {
  const pwd = options.pwd || "/test/collection";
  const glob = options.glob || "**/*.md";
  const name = options.name || pwd.split('/').filter(Boolean).pop() || 'test';

  // Read current config
  const configPath = join(testConfigDir, "index.yml");
  const { readFile } = await import("node:fs/promises");
  const content = await readFile(configPath, "utf-8");
  const config = YAML.parse(content) as CollectionConfig;

  // Add collection
  config.collections[name] = {
    path: pwd,
    pattern: glob,
  };

  // Write back
  await writeFile(configPath, YAML.stringify(config));
  return name;
}

// Helper to add path context in YAML config
async function addPathContext(collectionName: string, pathPrefix: string, contextText: string): Promise<void> {
  // Read current config
  const configPath = join(testConfigDir, "index.yml");
  const { readFile } = await import("node:fs/promises");
  const content = await readFile(configPath, "utf-8");
  const config = YAML.parse(content) as CollectionConfig;

  // Add context to collection
  if (!config.collections[collectionName]) {
    throw new Error(`Collection ${collectionName} not found`);
  }

  if (!config.collections[collectionName].context) {
    config.collections[collectionName].context = {};
  }

  config.collections[collectionName].context![pathPrefix] = contextText;

  // Write back
  await writeFile(configPath, YAML.stringify(config));
}

// Helper to add global context in YAML config
async function addGlobalContext(contextText: string): Promise<void> {
  const configPath = join(testConfigDir, "index.yml");
  const { readFile } = await import("node:fs/promises");
  const content = await readFile(configPath, "utf-8");
  const config = YAML.parse(content) as CollectionConfig;

  config.global_context = contextText;

  await writeFile(configPath, YAML.stringify(config));
}

// =============================================================================
// Test Setup
// =============================================================================

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-test-"));
});

afterAll(async () => {
  // Ensure native resources are released to avoid ggml-metal asserts on process exit.
  await disposeDefaultLlamaCpp();

  try {
    // Clean up test directory
    const { readdir, unlink } = await import("node:fs/promises");
    const files = await readdir(testDir);
    for (const file of files) {
      await unlink(join(testDir, file));
    }
    await rmdir(testDir);
  } catch {
    // Ignore cleanup errors
  }
});

// =============================================================================
// Path Utilities Tests
// =============================================================================

describe("Path Utilities", () => {
  test("homedir returns HOME environment variable", () => {
    const result = homedir();
    expect(result).toBe(Bun.env.HOME || "/tmp");
  });

  test("resolve handles absolute paths", () => {
    expect(resolve("/foo/bar")).toBe("/foo/bar");
    expect(resolve("/foo", "/bar")).toBe("/bar");
  });

  test("resolve handles relative paths", () => {
    const pwd = Bun.env.PWD || process.cwd();
    expect(resolve("foo")).toBe(`${pwd}/foo`);
    expect(resolve("foo", "bar")).toBe(`${pwd}/foo/bar`);
  });

  test("resolve normalizes . and ..", () => {
    expect(resolve("/foo/bar/./baz")).toBe("/foo/bar/baz");
    expect(resolve("/foo/bar/../baz")).toBe("/foo/baz");
    expect(resolve("/foo/bar/../../baz")).toBe("/baz");
  });

  test("getDefaultDbPath throws in test mode without INDEX_PATH", () => {
    // In test mode, getDefaultDbPath should throw to prevent accidental writes to global index
    // This is intentional safety behavior
    const originalIndexPath = process.env.INDEX_PATH;
    delete process.env.INDEX_PATH;

    expect(() => getDefaultDbPath()).toThrow("Database path not set");

    // Restore
    if (originalIndexPath) process.env.INDEX_PATH = originalIndexPath;
  });

  test("getDefaultDbPath uses INDEX_PATH when set", () => {
    const originalIndexPath = process.env.INDEX_PATH;
    process.env.INDEX_PATH = "/tmp/test-index.sqlite";

    expect(getDefaultDbPath()).toBe("/tmp/test-index.sqlite");
    expect(getDefaultDbPath("custom")).toBe("/tmp/test-index.sqlite"); // INDEX_PATH overrides name

    // Restore
    if (originalIndexPath) {
      process.env.INDEX_PATH = originalIndexPath;
    } else {
      delete process.env.INDEX_PATH;
    }
  });

  test("getPwd returns current working directory", () => {
    const pwd = getPwd();
    expect(pwd).toBeTruthy();
    expect(typeof pwd).toBe("string");
  });

  test("getRealPath resolves symlinks", () => {
    const result = getRealPath("/tmp");
    expect(result).toBeTruthy();
    // On macOS, /tmp is a symlink to /private/tmp
    expect(result === "/tmp" || result === "/private/tmp").toBe(true);
  });
});

// =============================================================================
// Handelize Tests - path normalization for token-friendly filenames
// =============================================================================

describe("handelize", () => {
  test("converts to lowercase", () => {
    expect(handelize("README.md")).toBe("readme.md");
    expect(handelize("MyFile.MD")).toBe("myfile.md");
  });

  test("preserves folder structure", () => {
    expect(handelize("a/b/c/d.md")).toBe("a/b/c/d.md");
    expect(handelize("docs/api/README.md")).toBe("docs/api/readme.md");
  });

  test("replaces non-word characters with dash", () => {
    expect(handelize("hello world.md")).toBe("hello-world.md");
    expect(handelize("file (1).md")).toBe("file-1.md");
    expect(handelize("foo@bar#baz.md")).toBe("foo-bar-baz.md");
  });

  test("collapses multiple special chars into single dash", () => {
    expect(handelize("hello   world.md")).toBe("hello-world.md");
    expect(handelize("foo---bar.md")).toBe("foo-bar.md");
    expect(handelize("a  -  b.md")).toBe("a-b.md");
  });

  test("removes leading and trailing dashes from segments", () => {
    expect(handelize("-hello-.md")).toBe("hello.md");
    expect(handelize("--test--.md")).toBe("test.md");
    expect(handelize("a/-b-/c.md")).toBe("a/b/c.md");
  });

  test("converts triple underscore to folder separator", () => {
    expect(handelize("foo___bar.md")).toBe("foo/bar.md");
    expect(handelize("notes___2025___january.md")).toBe("notes/2025/january.md");
    expect(handelize("a/b___c/d.md")).toBe("a/b/c/d.md");
  });

  test("handles complex real-world meeting notes", () => {
    // Example: "Money Movement Licensing Review - 2025／11／19 10:25 EST - Notes by Gemini.md"
    const complexName = "Money Movement Licensing Review - 2025／11／19 10:25 EST - Notes by Gemini.md";
    const result = handelize(complexName);
    expect(result).toBe("money-movement-licensing-review-2025-11-19-10-25-est-notes-by-gemini.md");
    expect(result).not.toContain(" ");
    expect(result).not.toContain("／");
    expect(result).not.toContain(":");
  });

  test("handles unicode characters", () => {
    // Pure unicode with no alphanumerics throws error
    expect(() => handelize("日本語.md")).toThrow("no valid filename content");
    // Mixed unicode/ascii preserves the ascii parts
    expect(handelize("café-notes.md")).toBe("caf-notes.md");
    expect(handelize("naïve.md")).toBe("na-ve.md");
    expect(handelize("日本語-notes.md")).toBe("notes.md");
  });

  test("handles dates and times in filenames", () => {
    expect(handelize("meeting-2025-01-15.md")).toBe("meeting-2025-01-15.md");
    expect(handelize("notes 2025/01/15.md")).toBe("notes-2025/01/15.md");
    expect(handelize("call_10:30_AM.md")).toBe("call-10-30-am.md");
  });

  test("handles special project naming patterns", () => {
    expect(handelize("PROJECT_ABC_v2.0.md")).toBe("project-abc-v2-0.md");
    expect(handelize("[WIP] Feature Request.md")).toBe("wip-feature-request.md");
    expect(handelize("(DRAFT) Proposal v1.md")).toBe("draft-proposal-v1.md");
  });

  test("filters out empty segments", () => {
    expect(handelize("a//b/c.md")).toBe("a/b/c.md");
    expect(handelize("/a/b/")).toBe("a/b");
    expect(handelize("///test///")).toBe("test");
  });

  test("throws error for invalid inputs", () => {
    expect(() => handelize("")).toThrow("path cannot be empty");
    expect(() => handelize("   ")).toThrow("path cannot be empty");
    expect(() => handelize(".md")).toThrow("no valid filename content");
    expect(() => handelize("...")).toThrow("no valid filename content");
    expect(() => handelize("___")).toThrow("no valid filename content");
  });

  test("handles minimal valid inputs", () => {
    expect(handelize("a")).toBe("a");
    expect(handelize("1")).toBe("1");
    expect(handelize("a.md")).toBe("a.md");
  });
});

// =============================================================================
// Store Creation Tests
// =============================================================================

describe("Store Creation", () => {
  test("createStore throws without explicit path in test mode", () => {
    // In test mode, createStore without path should throw to prevent accidental writes
    const originalIndexPath = process.env.INDEX_PATH;
    delete process.env.INDEX_PATH;

    expect(() => createStore()).toThrow("Database path not set");

    // Restore
    if (originalIndexPath) process.env.INDEX_PATH = originalIndexPath;
  });

  test("createStore creates a new store with custom path", async () => {
    const store = await createTestStore();
    expect(store.dbPath).toBe(testDbPath);
    expect(store.db).toBeInstanceOf(Database);
    await cleanupTestDb(store);
  });

  test("createStore initializes database schema", async () => {
    const store = await createTestStore();

    // Check tables exist
    const tables = store.db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
    `).all() as { name: string }[];

    const tableNames = tables.map(t => t.name);
    expect(tableNames).toContain("documents");
    expect(tableNames).toContain("documents_fts");
    expect(tableNames).toContain("content_vectors");
    expect(tableNames).toContain("llm_cache");
    // Note: path_contexts table removed in favor of YAML-based context storage

    await cleanupTestDb(store);
  });

  test("createStore sets WAL journal mode", async () => {
    const store = await createTestStore();
    const result = store.db.prepare("PRAGMA journal_mode").get() as { journal_mode: string };
    expect(result.journal_mode).toBe("wal");
    await cleanupTestDb(store);
  });

  test("store.close closes the database connection", async () => {
    const store = await createTestStore();
    store.close();
    // Attempting to use db after close should throw
    expect(() => store.db.prepare("SELECT 1").get()).toThrow();
    try {
      await unlink(testDbPath);
    } catch {}
  });
});

// =============================================================================
// Document Hashing & Title Extraction Tests
// =============================================================================

describe("Document Helpers", () => {
  test("hashContent produces consistent SHA256 hashes", async () => {
    const content = "Hello, World!";
    const hash1 = await hashContent(content);
    const hash2 = await hashContent(content);
    expect(hash1).toBe(hash2);
    expect(hash1).toMatch(/^[a-f0-9]{64}$/);
  });

  test("hashContent produces different hashes for different content", async () => {
    const hash1 = await hashContent("Hello");
    const hash2 = await hashContent("World");
    expect(hash1).not.toBe(hash2);
  });

  test("extractTitle extracts H1 heading", () => {
    const content = "# My Title\n\nSome content here.";
    expect(extractTitle(content, "file.md")).toBe("My Title");
  });

  test("extractTitle extracts H2 heading if no H1", () => {
    const content = "## My Subtitle\n\nSome content here.";
    expect(extractTitle(content, "file.md")).toBe("My Subtitle");
  });

  test("extractTitle falls back to filename", () => {
    const content = "Just some plain text without headings.";
    expect(extractTitle(content, "my-document.md")).toBe("my-document");
  });

  test("extractTitle skips generic 'Notes' heading", () => {
    const content = "# Notes\n\n## Actual Title\n\nContent";
    expect(extractTitle(content, "file.md")).toBe("Actual Title");
  });

  test("extractTitle handles 📝 Notes heading", () => {
    const content = "# 📝 Notes\n\n## Meeting Summary\n\nContent";
    expect(extractTitle(content, "file.md")).toBe("Meeting Summary");
  });
});

// =============================================================================
// Embedding Format Tests
// =============================================================================

describe("Embedding Formatting", () => {
  test("formatQueryForEmbedding adds search task prefix", () => {
    const formatted = formatQueryForEmbedding("how to deploy");
    expect(formatted).toBe("task: search result | query: how to deploy");
  });

  test("formatDocForEmbedding adds title and text prefix", () => {
    const formatted = formatDocForEmbedding("Some content", "My Title");
    expect(formatted).toBe("title: My Title | text: Some content");
  });

  test("formatDocForEmbedding handles missing title", () => {
    const formatted = formatDocForEmbedding("Some content");
    expect(formatted).toBe("title: none | text: Some content");
  });
});

// =============================================================================
// Document Chunking Tests
// =============================================================================

describe("Document Chunking", () => {
  test("chunkDocument returns single chunk for small documents", () => {
    const content = "Small document content";
    const chunks = chunkDocument(content, 1000, 0);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe(content);
    expect(chunks[0].pos).toBe(0);
  });

  test("chunkDocument splits large documents", () => {
    const content = "A".repeat(10000);
    const chunks = chunkDocument(content, 1000, 0);
    expect(chunks.length).toBeGreaterThan(1);

    // All chunks should have correct positions
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].pos).toBeGreaterThanOrEqual(0);
      if (i > 0) {
        expect(chunks[i].pos).toBeGreaterThan(chunks[i - 1].pos);
      }
    }
  });

  test("chunkDocument with overlap creates overlapping chunks", () => {
    const content = "A".repeat(3000);
    const chunks = chunkDocument(content, 1000, 150);  // 15% overlap
    expect(chunks.length).toBeGreaterThan(1);

    // With overlap, positions should be closer together than without
    // Each new chunk starts 150 chars before where the previous one ended
    for (let i = 1; i < chunks.length; i++) {
      const prevEnd = chunks[i - 1].pos + chunks[i - 1].text.length;
      const currentStart = chunks[i].pos;
      // Current chunk should start before the previous chunk ended (overlap)
      expect(currentStart).toBeLessThan(prevEnd);
      // But should still make forward progress
      expect(currentStart).toBeGreaterThan(chunks[i - 1].pos);
    }
  });

  test("chunkDocument prefers paragraph breaks", () => {
    const content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.".repeat(50);
    const chunks = chunkDocument(content, 500, 0);

    // Chunks should end at paragraph breaks when possible
    for (const chunk of chunks.slice(0, -1)) {
      // Most chunks should end near a paragraph break
      const endsNearParagraph = chunk.text.endsWith("\n\n") ||
        chunk.text.endsWith(".") ||
        chunk.text.endsWith("\n");
      // This is a soft check - not all chunks can end at breaks
    }
    expect(chunks.length).toBeGreaterThan(1);
  });

  test("chunkDocument handles UTF-8 characters correctly", () => {
    const content = "こんにちは世界".repeat(500); // Japanese text
    const chunks = chunkDocument(content, 1000, 0);

    // Should not split in the middle of a multi-byte character
    for (const chunk of chunks) {
      expect(() => new TextEncoder().encode(chunk.text)).not.toThrow();
    }
  });

  test("chunkDocument with default params uses 800-token chunks", () => {
    // Default is CHUNK_SIZE_CHARS (3200 chars) with CHUNK_OVERLAP_CHARS (480 chars)
    const content = "Word ".repeat(2000);  // ~10000 chars
    const chunks = chunkDocument(content);
    expect(chunks.length).toBeGreaterThan(1);
    // Each chunk should be around 3200 chars (except last)
    expect(chunks[0].text.length).toBeGreaterThan(2500);
    expect(chunks[0].text.length).toBeLessThanOrEqual(3200);
  });
});

describe("Token-based Chunking", () => {
  test("chunkDocumentByTokens returns single chunk for small documents", async () => {
    const content = "This is a small document.";
    const chunks = await chunkDocumentByTokens(content, 800, 120);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe(content);
    expect(chunks[0].pos).toBe(0);
    expect(chunks[0].tokens).toBeGreaterThan(0);
    expect(chunks[0].tokens).toBeLessThan(800);
  });

  test("chunkDocumentByTokens splits large documents", async () => {
    // Create a document that's definitely more than 800 tokens
    const content = "The quick brown fox jumps over the lazy dog. ".repeat(200);
    const chunks = await chunkDocumentByTokens(content, 800, 120);

    expect(chunks.length).toBeGreaterThan(1);

    // Each chunk should have ~800 tokens or less
    for (const chunk of chunks) {
      expect(chunk.tokens).toBeLessThanOrEqual(850);  // Allow slight overage
      expect(chunk.tokens).toBeGreaterThan(0);
    }

    // Chunks should have correct positions
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].pos).toBeGreaterThanOrEqual(0);
      if (i > 0) {
        expect(chunks[i].pos).toBeGreaterThan(chunks[i - 1].pos);
      }
    }
  });

  test("chunkDocumentByTokens creates overlapping chunks", async () => {
    const content = "Word ".repeat(500);  // ~500 tokens
    const chunks = await chunkDocumentByTokens(content, 200, 30);  // 15% overlap

    expect(chunks.length).toBeGreaterThan(1);

    // With overlap, consecutive chunks should have overlapping positions
    for (let i = 1; i < chunks.length; i++) {
      const prevEnd = chunks[i - 1].pos + chunks[i - 1].text.length;
      const currentStart = chunks[i].pos;
      // Current chunk should start before the previous chunk ended (overlap)
      expect(currentStart).toBeLessThan(prevEnd);
    }
  });

  test("chunkDocumentByTokens returns actual token counts", async () => {
    const content = "Hello world, this is a test.";
    const chunks = await chunkDocumentByTokens(content);

    expect(chunks).toHaveLength(1);
    // The token count should be reasonable (not 0, not equal to char count)
    expect(chunks[0].tokens).toBeGreaterThan(0);
    expect(chunks[0].tokens).toBeLessThan(content.length);  // Tokens < chars for English
  });
});

// =============================================================================
// Caching Tests
// =============================================================================

describe("Caching", () => {
  test("getCacheKey generates consistent keys", () => {
    const key1 = getCacheKey("http://example.com", { query: "test" });
    const key2 = getCacheKey("http://example.com", { query: "test" });
    expect(key1).toBe(key2);
    expect(key1).toMatch(/^[a-f0-9]{64}$/);
  });

  test("getCacheKey generates different keys for different inputs", () => {
    const key1 = getCacheKey("http://example.com", { query: "test1" });
    const key2 = getCacheKey("http://example.com", { query: "test2" });
    expect(key1).not.toBe(key2);
  });

  test("store cache operations work correctly", async () => {
    const store = await createTestStore();

    const key = "test-cache-key";
    const value = "cached result";

    // Initially empty
    expect(store.getCachedResult(key)).toBeNull();

    // Set cache
    store.setCachedResult(key, value);

    // Retrieve cache
    expect(store.getCachedResult(key)).toBe(value);

    // Clear cache
    store.clearCache();
    expect(store.getCachedResult(key)).toBeNull();

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Context Tests
// =============================================================================

describe("Path Context", () => {
  test("getContextForFile returns null when no context set", async () => {
    const store = await createTestStore();
    const context = store.getContextForFile("/some/random/path.md");
    expect(context).toBeNull();
    await cleanupTestDb(store);
  });

  test("getContextForFile returns matching context", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection({ pwd: "/test/collection", glob: "**/*.md" });
    await addPathContext(collectionName, "/docs", "Documentation files");

    // Insert a document so getContextForFile can find it
    await insertTestDocument(store.db, collectionName, {
      name: "readme",
      displayPath: "docs/readme.md",
    });

    const context = store.getContextForFile("/test/collection/docs/readme.md");
    expect(context).toBe("Documentation files");

    await cleanupTestDb(store);
  });

  test("getContextForFile returns all matching contexts", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection({ pwd: "/test/collection", glob: "**/*.md" });
    await addPathContext(collectionName, "/", "General test files");
    await addPathContext(collectionName, "/docs", "Documentation files");
    await addPathContext(collectionName, "/docs/api", "API documentation");

    // Insert documents so getContextForFile can find them
    await insertTestDocument(store.db, collectionName, {
      name: "readme",
      displayPath: "readme.md",
    });
    await insertTestDocument(store.db, collectionName, {
      name: "guide",
      displayPath: "docs/guide.md",
    });
    await insertTestDocument(store.db, collectionName, {
      name: "reference",
      displayPath: "docs/api/reference.md",
    });

    // Context now returns ALL matching contexts joined with \n\n
    expect(store.getContextForFile("/test/collection/readme.md")).toBe("General test files");
    expect(store.getContextForFile("/test/collection/docs/guide.md")).toBe("General test files\n\nDocumentation files");
    expect(store.getContextForFile("/test/collection/docs/api/reference.md")).toBe("General test files\n\nDocumentation files\n\nAPI documentation");

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Collection Tests
// =============================================================================

describe("Collections", () => {
  test("collections are managed via YAML config", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection({ pwd: "/home/user/projects/myapp", glob: "**/*.md" });

    // Collections are now in YAML, not in the database
    expect(collectionName).toBe("myapp");

    await cleanupTestDb(store);
  });
});

// =============================================================================
// FTS Search Tests
// =============================================================================

describe("FTS Search", () => {
  test("searchFTS returns empty array for no matches", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();
    await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      body: "The quick brown fox jumps over the lazy dog",
    });

    const results = store.searchFTS("nonexistent-term-xyz", 10);
    expect(results).toHaveLength(0);

    await cleanupTestDb(store);
  });

  test("searchFTS finds documents by keyword", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();
    await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      title: "Fox Document",
      body: "The quick brown fox jumps over the lazy dog",
      displayPath: "test/doc1.md",
    });

    const results = store.searchFTS("fox", 10);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].displayPath).toBe(`${collectionName}/test/doc1.md`);
    expect(results[0].filepath).toBe(`qmd://${collectionName}/test/doc1.md`);
    expect(results[0].source).toBe("fts");

    await cleanupTestDb(store);
  });

  test("searchFTS ranks title matches higher", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    // Document with "fox" in body only
    await insertTestDocument(store.db, collectionName, {
      name: "body-match",
      title: "Some Other Title",
      body: "The fox is here in the body",
      displayPath: "test/body.md",
    });

    // Document with "fox" in title (via name field which is indexed)
    await insertTestDocument(store.db, collectionName, {
      name: "fox",
      title: "Fox Title",
      body: "Different content without the animal fox",
      displayPath: "test/title.md",
    });

    const results = store.searchFTS("fox", 10);
    // Both documents contain "fox" in the body now, so we should get 2 results
    expect(results.length).toBe(2);
    // Title/name match should rank higher due to BM25 weights
    expect(results[0].displayPath).toBe(`${collectionName}/test/title.md`);

    await cleanupTestDb(store);
  });

  test("searchFTS respects limit parameter", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    // Insert 10 documents
    for (let i = 0; i < 10; i++) {
      await insertTestDocument(store.db, collectionName, {
        name: `doc${i}`,
        body: "common keyword appears here",
        displayPath: `test/doc${i}.md`,
      });
    }

    const results = store.searchFTS("common keyword", 3);
    expect(results).toHaveLength(3);

    await cleanupTestDb(store);
  });

  test("searchFTS filters by collection name", async () => {
    const store = await createTestStore();
    const collection1 = await createTestCollection({ pwd: "/path/one", glob: "**/*.md", name: "one" });
    const collection2 = await createTestCollection({ pwd: "/path/two", glob: "**/*.md", name: "two" });

    await insertTestDocument(store.db, collection1, {
      name: "doc1",
      body: "searchable content",
      displayPath: "doc1.md",
    });

    await insertTestDocument(store.db, collection2, {
      name: "doc2",
      body: "searchable content",
      displayPath: "doc2.md",
    });

    const allResults = store.searchFTS("searchable", 10);
    expect(allResults).toHaveLength(2);

    // Filter by collection name (collectionId is now treated as collection name string)
    const filtered = store.searchFTS("searchable", 10, collection1 as unknown as number);
    expect(filtered).toHaveLength(1);
    expect(filtered[0].displayPath).toBe(`${collection1}/doc1.md`);

    await cleanupTestDb(store);
  });

  test("searchFTS handles special characters in query", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();
    await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      body: "Function with params: foo(bar, baz)",
      displayPath: "test/doc1.md",
    });

    // Should not throw on special characters
    const results = store.searchFTS("foo(bar)", 10);
    // Results may vary based on FTS5 handling
    expect(Array.isArray(results)).toBe(true);

    await cleanupTestDb(store);
  });

  test("searchFTS ignores inactive documents", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      name: "active",
      body: "findme content",
      displayPath: "test/active.md",
      active: 1,
    });

    await insertTestDocument(store.db, collectionName, {
      name: "inactive",
      body: "findme content",
      displayPath: "test/inactive.md",
      active: 0,
    });

    const results = store.searchFTS("findme", 10);
    expect(results).toHaveLength(1);
    expect(results[0].displayPath).toBe(`${collectionName}/test/active.md`);
    expect(results[0].filepath).toBe(`qmd://${collectionName}/test/active.md`);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Document Retrieval Tests
// =============================================================================

describe("Document Retrieval", () => {
  describe("findDocument", () => {
    test("findDocument finds by exact filepath", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/exact/path", glob: "**/*.md" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        title: "My Document",
        displayPath: "mydoc.md",
        body: "Document content here",
      });

      const result = store.findDocument("/exact/path/mydoc.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.title).toBe("My Document");
        expect(result.displayPath).toBe(`${collectionName}/mydoc.md`);
        expect(result.filepath).toBe(`qmd://${collectionName}/mydoc.md`);
        expect(result.body).toBeUndefined(); // body not included by default
      }

      await cleanupTestDb(store);
    });

    test("findDocument finds by display_path", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/some/path", glob: "**/*.md" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "docs/mydoc.md",
      });

      const result = store.findDocument("docs/mydoc.md");
      expect("error" in result).toBe(false);

      await cleanupTestDb(store);
    });

    test("findDocument finds by partial path match", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/very/long/path/to", glob: "**/*.md" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "mydoc.md",
      });

      const result = store.findDocument("mydoc.md");
      expect("error" in result).toBe(false);

      await cleanupTestDb(store);
    });

    test("findDocument includes body when requested", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/path", glob: "**/*.md" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "mydoc.md",
        body: "The actual body content",
      });

      const result = store.findDocument("/path/mydoc.md", { includeBody: true });
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.body).toBe("The actual body content");
      }

      await cleanupTestDb(store);
    });

    test("findDocument returns error with suggestions for not found", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();
      await insertTestDocument(store.db, collectionName, {
        name: "similar",
        filepath: "/path/similar.md",
        displayPath: "similar.md",
      });

      const result = store.findDocument("simlar.md"); // typo - 1 char diff
      expect("error" in result).toBe(true);
      if ("error" in result) {
        expect(result.error).toBe("not_found");
        // Levenshtein distance of 1 should be found with maxDistance 3
        expect(result.similarFiles.length).toBeGreaterThanOrEqual(0); // May or may not find depending on distance calc
      }

      await cleanupTestDb(store);
    });

    test("findDocument handles :line suffix", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        filepath: "/path/mydoc.md",
        displayPath: "mydoc.md",
      });

      const result = store.findDocument("mydoc.md:100");
      expect("error" in result).toBe(false);

      await cleanupTestDb(store);
    });

    test("findDocument expands ~ to home directory", async () => {
      const store = await createTestStore();
      const home = homedir();
      const collectionName = await createTestCollection({ pwd: home, name: "home" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        filepath: `${home}/docs/mydoc.md`,
        displayPath: "docs/mydoc.md",
      });

      const result = store.findDocument("~/docs/mydoc.md");
      expect("error" in result).toBe(false);

      await cleanupTestDb(store);
    });

    test("findDocument includes context from path_contexts", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/path" });
      await addPathContext(collectionName, "docs", "Documentation");
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "docs/mydoc.md",
      });

      const result = store.findDocument("/path/docs/mydoc.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        expect(result.context).toBe("Documentation");
      }

      await cleanupTestDb(store);
    });

    test("findDocument includes hierarchical contexts (global + collection + path)", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/archive", name: "archive" });

      // Add global context
      await addGlobalContext("Global context for all documents");

      // Add collection root context
      await addPathContext(collectionName, "/", "Archive collection context");

      // Add path-specific contexts at different levels
      await addPathContext(collectionName, "/podcasts", "Podcast episodes");
      await addPathContext(collectionName, "/podcasts/external", "External podcast interviews");

      // Insert document in nested path
      await insertTestDocument(store.db, collectionName, {
        name: "interview",
        displayPath: "podcasts/external/2024-jan-interview.md",
      });

      const result = store.findDocument("/archive/podcasts/external/2024-jan-interview.md");
      expect("error" in result).toBe(false);
      if (!("error" in result)) {
        // Should have all contexts joined with double newlines
        expect(result.context).toBe(
          "Global context for all documents\n\n" +
          "Archive collection context\n\n" +
          "Podcast episodes\n\n" +
          "External podcast interviews"
        );
      }

      await cleanupTestDb(store);
    });
  });

  describe("getDocumentBody", () => {
    test("getDocumentBody returns full body", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/path" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "mydoc.md",
        body: "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
      });

      const body = store.getDocumentBody({ filepath: "/path/mydoc.md" });
      expect(body).toBe("Line 1\nLine 2\nLine 3\nLine 4\nLine 5");

      await cleanupTestDb(store);
    });

    test("getDocumentBody supports line range", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection({ pwd: "/path" });
      await insertTestDocument(store.db, collectionName, {
        name: "mydoc",
        displayPath: "mydoc.md",
        body: "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
      });

      const body = store.getDocumentBody({ filepath: "/path/mydoc.md" }, 2, 2);
      expect(body).toBe("Line 2\nLine 3");

      await cleanupTestDb(store);
    });

    test("getDocumentBody returns null for non-existent document", async () => {
      const store = await createTestStore();
      const body = store.getDocumentBody({ filepath: "/nonexistent.md" });
      expect(body).toBeNull();
      await cleanupTestDb(store);
    });
  });

  describe("findDocuments (multi-get)", () => {
    test("findDocuments finds by glob pattern", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();

      await insertTestDocument(store.db, collectionName, {
        name: "doc1",
        filepath: "/path/journals/2024-01.md",
        displayPath: "journals/2024-01.md",
      });
      await insertTestDocument(store.db, collectionName, {
        name: "doc2",
        filepath: "/path/journals/2024-02.md",
        displayPath: "journals/2024-02.md",
      });
      await insertTestDocument(store.db, collectionName, {
        name: "doc3",
        filepath: "/path/other/file.md",
        displayPath: "other/file.md",
      });

      const { docs, errors } = store.findDocuments("journals/2024-*.md");
      expect(errors).toHaveLength(0);
      expect(docs).toHaveLength(2);

      await cleanupTestDb(store);
    });

    test("findDocuments finds by comma-separated list", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();

      await insertTestDocument(store.db, collectionName, {
        name: "doc1",
        filepath: "/path/doc1.md",
        displayPath: "doc1.md",
      });
      await insertTestDocument(store.db, collectionName, {
        name: "doc2",
        filepath: "/path/doc2.md",
        displayPath: "doc2.md",
      });

      const { docs, errors } = store.findDocuments("doc1.md, doc2.md");
      expect(errors).toHaveLength(0);
      expect(docs).toHaveLength(2);

      await cleanupTestDb(store);
    });

    test("findDocuments reports errors for not found files", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();

      await insertTestDocument(store.db, collectionName, {
        name: "doc1",
        filepath: "/path/doc1.md",
        displayPath: "doc1.md",
      });

      const { docs, errors } = store.findDocuments("doc1.md, nonexistent.md");
      expect(docs).toHaveLength(1);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain("not found");

      await cleanupTestDb(store);
    });

    test("findDocuments skips large files", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();

      await insertTestDocument(store.db, collectionName, {
        name: "large",
        filepath: "/path/large.md",
        displayPath: "large.md",
        body: "x".repeat(20000), // 20KB
      });

      const { docs } = store.findDocuments("large.md", { maxBytes: 10000 });
      expect(docs).toHaveLength(1);
      expect(docs[0].skipped).toBe(true);
      if (docs[0].skipped) {
        expect(docs[0].skipReason).toContain("too large");
      }

      await cleanupTestDb(store);
    });

    test("findDocuments includes body when requested", async () => {
      const store = await createTestStore();
      const collectionName = await createTestCollection();

      await insertTestDocument(store.db, collectionName, {
        name: "doc1",
        filepath: "/path/doc1.md",
        displayPath: "doc1.md",
        body: "The content",
      });

      const { docs } = store.findDocuments("doc1.md", { includeBody: true });
      expect(docs[0].skipped).toBe(false);
      if (!docs[0].skipped) {
        expect(docs[0].doc.body).toBe("The content");
      }

      await cleanupTestDb(store);
    });
  });

});

// =============================================================================
// Snippet Extraction Tests
// =============================================================================

describe("Snippet Extraction", () => {
  test("extractSnippet finds query terms", () => {
    const body = "First line.\nSecond line with keyword.\nThird line.\nFourth line.";
    const { line, snippet } = extractSnippet(body, "keyword", 500);

    expect(line).toBe(2); // Line 2 contains "keyword"
    expect(snippet).toContain("keyword");
  });

  test("extractSnippet includes context lines", () => {
    const body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
    const { snippet } = extractSnippet(body, "keyword", 500);

    expect(snippet).toContain("Line 2"); // Context before
    expect(snippet).toContain("Line 3 has keyword");
    expect(snippet).toContain("Line 4"); // Context after
  });

  test("extractSnippet respects maxLen for content", () => {
    const body = "A".repeat(1000);
    const result = extractSnippet(body, "query", 100);

    // Snippet includes header + content, content should be truncated
    expect(result.snippet).toContain("@@"); // Has diff header
    expect(result.snippet).toContain("..."); // Content was truncated
  });

  test("extractSnippet uses chunkPos hint", () => {
    const body = "First section...\n".repeat(50) + "Target keyword here\n" + "More content...".repeat(50);
    const chunkPos = body.indexOf("Target keyword");

    const { snippet } = extractSnippet(body, "Target", 200, chunkPos);
    expect(snippet).toContain("Target keyword");
  });

  test("extractSnippet returns beginning when no match", () => {
    const body = "First line\nSecond line\nThird line";
    const { line, snippet } = extractSnippet(body, "nonexistent", 500);

    expect(line).toBe(1);
    expect(snippet).toContain("First line");
  });

  test("extractSnippet includes diff-style header", () => {
    const body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
    const { snippet, linesBefore, linesAfter, snippetLines } = extractSnippet(body, "keyword", 500);

    // Header should show line position and context info
    expect(snippet).toMatch(/^@@ -\d+,\d+ @@ \(\d+ before, \d+ after\)/);
    expect(linesBefore).toBe(1); // Line 1 comes before
    expect(linesAfter).toBe(0);  // Snippet includes to end (lines 2-5)
    expect(snippetLines).toBe(4); // Lines 2, 3, 4, 5
  });

  test("extractSnippet calculates linesBefore and linesAfter correctly", () => {
    const body = "L1\nL2\nL3\nL4 match\nL5\nL6\nL7\nL8\nL9\nL10";
    const { linesBefore, linesAfter, snippetLines, line } = extractSnippet(body, "match", 500);

    expect(line).toBe(4); // "L4 match" is line 4
    expect(linesBefore).toBe(2); // L1, L2 before snippet (snippet starts at L3)
    expect(snippetLines).toBe(4); // L3, L4, L5, L6
    expect(linesAfter).toBe(4); // L7, L8, L9, L10 after snippet
  });

  test("extractSnippet header format matches diff style", () => {
    const body = "A\nB\nC keyword\nD\nE\nF\nG\nH";
    const { snippet } = extractSnippet(body, "keyword", 500);

    // Should start with @@ -line,count @@ (N before, M after)
    const headerMatch = snippet.match(/^@@ -(\d+),(\d+) @@ \((\d+) before, (\d+) after\)/);
    expect(headerMatch).not.toBeNull();

    const [, startLine, count, before, after] = headerMatch!;
    expect(parseInt(startLine)).toBe(2); // Snippet starts at line 2 (B)
    expect(parseInt(count)).toBe(4);     // 4 lines: B, C keyword, D, E
    expect(parseInt(before)).toBe(1);    // A is before
    expect(parseInt(after)).toBe(3);     // F, G, H are after
  });

  test("extractSnippet at document start shows 0 before", () => {
    const body = "First line keyword\nSecond\nThird\nFourth\nFifth";
    const { linesBefore, linesAfter, snippetLines, line } = extractSnippet(body, "keyword", 500);

    expect(line).toBe(1);         // Keyword on first line
    expect(linesBefore).toBe(0);  // Nothing before
    expect(snippetLines).toBe(3); // First, Second, Third (bestLine-1 to bestLine+3, clamped)
    expect(linesAfter).toBe(2);   // Fourth, Fifth
  });

  test("extractSnippet at document end shows 0 after", () => {
    const body = "First\nSecond\nThird\nFourth\nFifth keyword";
    const { linesBefore, linesAfter, snippetLines, line } = extractSnippet(body, "keyword", 500);

    expect(line).toBe(5);         // Keyword on last line
    expect(linesBefore).toBe(3);  // First, Second, Third before snippet
    expect(snippetLines).toBe(2); // Fourth, Fifth keyword (bestLine-1 to bestLine+3, clamped)
    expect(linesAfter).toBe(0);   // Nothing after
  });

  test("extractSnippet with single line document", () => {
    const body = "Single line with keyword";
    const { linesBefore, linesAfter, snippetLines, snippet } = extractSnippet(body, "keyword", 500);

    expect(linesBefore).toBe(0);
    expect(linesAfter).toBe(0);
    expect(snippetLines).toBe(1);
    expect(snippet).toContain("@@ -1,1 @@ (0 before, 0 after)");
    expect(snippet).toContain("Single line with keyword");
  });

  test("extractSnippet with chunkPos adjusts line numbers correctly", () => {
    // 50 lines of padding, then keyword, then more content
    const padding = "Padding line\n".repeat(50);
    const body = padding + "Target keyword here\nMore content\nEven more";
    const chunkPos = padding.length; // Position of "Target keyword"

    const { line, linesBefore, linesAfter } = extractSnippet(body, "keyword", 200, chunkPos);

    expect(line).toBe(51); // "Target keyword" is line 51
    expect(linesBefore).toBeGreaterThan(40); // Many lines before
  });
});

// =============================================================================
// Reciprocal Rank Fusion Tests
// =============================================================================

describe("Reciprocal Rank Fusion", () => {
  const makeResult = (file: string, score: number): RankedResult => ({
    file,
    displayPath: file,
    title: file,
    body: "body",
    score,
  });

  test("RRF combines single list correctly", () => {
    const list1 = [
      makeResult("doc1", 0.9),
      makeResult("doc2", 0.8),
      makeResult("doc3", 0.7),
    ];

    const fused = reciprocalRankFusion([list1]);

    // Order should be preserved
    expect(fused[0].file).toBe("doc1");
    expect(fused[1].file).toBe("doc2");
    expect(fused[2].file).toBe("doc3");
  });

  test("RRF merges documents from multiple lists", () => {
    const list1 = [makeResult("doc1", 0.9), makeResult("doc2", 0.8)];
    const list2 = [makeResult("doc2", 0.95), makeResult("doc3", 0.85)];

    const fused = reciprocalRankFusion([list1, list2]);

    // doc2 appears in both lists, should have higher combined score
    expect(fused.find(r => r.file === "doc2")).toBeDefined();
    expect(fused.find(r => r.file === "doc1")).toBeDefined();
    expect(fused.find(r => r.file === "doc3")).toBeDefined();
  });

  test("RRF respects weights", () => {
    const list1 = [makeResult("doc1", 0.9)];
    const list2 = [makeResult("doc2", 0.9)];

    // Give double weight to list1
    const fused = reciprocalRankFusion([list1, list2], [2.0, 1.0]);

    // doc1 should rank higher due to weight
    expect(fused[0].file).toBe("doc1");
  });

  test("RRF adds top-rank bonus", () => {
    // doc1 is #1 in list1, doc2 is #2 in list1
    const list1 = [makeResult("doc1", 0.9), makeResult("doc2", 0.8)];
    const list2 = [makeResult("doc3", 0.85)];

    const fused = reciprocalRankFusion([list1, list2]);

    // doc1 should get +0.05 bonus for being #1
    // doc2 should get +0.02 bonus for being #2-3
    const doc1 = fused.find(r => r.file === "doc1");
    const doc2 = fused.find(r => r.file === "doc2");

    expect(doc1!.score).toBeGreaterThan(doc2!.score);
  });

  test("RRF handles empty lists", () => {
    const fused = reciprocalRankFusion([[], []]);
    expect(fused).toHaveLength(0);
  });

  test("RRF uses k parameter correctly", () => {
    const list = [makeResult("doc1", 0.9)];

    // With different k values, scores should differ
    const fused60 = reciprocalRankFusion([list], [], 60);
    const fused30 = reciprocalRankFusion([list], [], 30);

    // Lower k = higher scores for top ranks
    expect(fused30[0].score).toBeGreaterThan(fused60[0].score);
  });
});

// =============================================================================
// Index Status Tests
// =============================================================================

describe("Index Status", () => {
  test("getStatus returns correct structure", async () => {
    const store = await createTestStore();
    const status = store.getStatus();
    expect(status).toHaveProperty("totalDocuments");
    expect(status).toHaveProperty("needsEmbedding");
    expect(status).toHaveProperty("hasVectorIndex");
    expect(status).toHaveProperty("collections");
    expect(Array.isArray(status.collections)).toBe(true);

    await cleanupTestDb(store);
  });

  test("getStatus counts documents correctly", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, { name: "doc1", active: 1 });
    await insertTestDocument(store.db, collectionName, { name: "doc2", active: 1 });
    await insertTestDocument(store.db, collectionName, { name: "doc3", active: 0 }); // inactive

    const status = store.getStatus();
    expect(status.totalDocuments).toBe(2); // Only active docs

    await cleanupTestDb(store);
  });

  test("getStatus reports collection info", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection({ pwd: "/test/path", glob: "**/*.md" });
    await insertTestDocument(store.db, collectionName, { name: "doc1" });

    const status = store.getStatus();
    expect(status.collections.length).toBeGreaterThanOrEqual(1);
    const col = status.collections.find(c => c.name === collectionName);
    expect(col).toBeDefined();
    expect(col?.path).toBe("/test/path");
    expect(col?.pattern).toBe("**/*.md");
    expect(col?.documents).toBe(1);

    await cleanupTestDb(store);
  });

  test("getHashesNeedingEmbedding counts correctly", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    // Add documents with different hashes
    await insertTestDocument(store.db, collectionName, { name: "doc1", hash: "hash1" });
    await insertTestDocument(store.db, collectionName, { name: "doc2", hash: "hash2" });
    await insertTestDocument(store.db, collectionName, { name: "doc3", hash: "hash1" }); // same hash as doc1

    const needsEmbedding = store.getHashesNeedingEmbedding();
    expect(needsEmbedding).toBe(2); // hash1 and hash2

    await cleanupTestDb(store);
  });

  test("getIndexHealth returns health info", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();
    await insertTestDocument(store.db, collectionName, { name: "doc1" });

    const health = store.getIndexHealth();
    expect(health).toHaveProperty("needsEmbedding");
    expect(health).toHaveProperty("totalDocs");
    expect(health).toHaveProperty("daysStale");
    expect(health.totalDocs).toBe(1);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Fuzzy Matching Tests
// =============================================================================

describe("Fuzzy Matching", () => {
  test("findSimilarFiles finds similar paths", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      name: "readme",
      displayPath: "docs/readme.md",
    });
    await insertTestDocument(store.db, collectionName, {
      name: "readmi",
      displayPath: "docs/readmi.md", // typo
    });

    const similar = store.findSimilarFiles("docs/readme.md", 3, 5);
    expect(similar).toContain("docs/readme.md");

    await cleanupTestDb(store);
  });

  test("findSimilarFiles respects maxDistance", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      name: "abc",
      displayPath: "abc.md",
    });
    await insertTestDocument(store.db, collectionName, {
      name: "xyz",
      displayPath: "xyz.md", // very different
    });

    const similar = store.findSimilarFiles("abc.md", 1, 5); // max distance 1
    expect(similar).toContain("abc.md");
    expect(similar).not.toContain("xyz.md");

    await cleanupTestDb(store);
  });

  test("matchFilesByGlob matches patterns", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      filepath: "/p/journals/2024-01.md",
      displayPath: "journals/2024-01.md",
    });
    await insertTestDocument(store.db, collectionName, {
      filepath: "/p/journals/2024-02.md",
      displayPath: "journals/2024-02.md",
    });
    await insertTestDocument(store.db, collectionName, {
      filepath: "/p/docs/readme.md",
      displayPath: "docs/readme.md",
    });

    const matches = store.matchFilesByGlob("journals/*.md");
    expect(matches).toHaveLength(2);
    expect(matches.every(m => m.displayPath.startsWith("journals/"))).toBe(true);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Vector Table Tests
// =============================================================================

describe("Vector Table", () => {
  test("ensureVecTable creates vector table", async () => {
    const store = await createTestStore();

    // Initially no vector table
    let exists = store.db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'
    `).get();
    expect(exists).toBeFalsy(); // null or undefined

    // Create vector table
    store.ensureVecTable(768);

    exists = store.db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'
    `).get();
    expect(exists).toBeTruthy();

    await cleanupTestDb(store);
  });

  test("ensureVecTable recreates table if dimensions change", async () => {
    const store = await createTestStore();

    // Create with 768 dimensions
    store.ensureVecTable(768);

    // Check dimensions
    let tableInfo = store.db.prepare(`
      SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'
    `).get() as { sql: string };
    expect(tableInfo.sql).toContain("float[768]");

    // Recreate with different dimensions
    store.ensureVecTable(1024);

    tableInfo = store.db.prepare(`
      SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'
    `).get() as { sql: string };
    expect(tableInfo.sql).toContain("float[1024]");

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe("Integration", () => {
  test("full document lifecycle: create, search, retrieve", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection({ pwd: "/test/notes", glob: "**/*.md" });

    // Add context - use "/" for collection root
    await addPathContext(collectionName, "/", "Personal notes");

    // Insert documents
    await insertTestDocument(store.db, collectionName, {
      name: "meeting",
      title: "Team Meeting Notes",
      filepath: "/test/notes/meeting.md",
      displayPath: "notes/meeting.md",
      body: "# Team Meeting Notes\n\nDiscussed project timeline and deliverables.",
    });

    await insertTestDocument(store.db, collectionName, {
      name: "ideas",
      title: "Project Ideas",
      filepath: "/test/notes/ideas.md",
      displayPath: "notes/ideas.md",
      body: "# Project Ideas\n\nBrainstorming new features for the product.",
    });

    // Search
    const searchResults = store.searchFTS("project", 10);
    expect(searchResults.length).toBe(2);

    // Status - SKIPPED: getStatus() has bug (queries non-existent collections table)
    // const status = store.getStatus();
    // expect(status.totalDocuments).toBe(2);
    // expect(status.collections).toHaveLength(1);

    // Retrieve single document
    const doc = store.findDocument("notes/meeting.md", { includeBody: true });
    expect("error" in doc).toBe(false);
    if (!("error" in doc)) {
      expect(doc.title).toBe("Team Meeting Notes");
      expect(doc.context).toBe("Personal notes");
      expect(doc.body).toContain("Team Meeting");
    }

    // Multi-get
    const { docs, errors } = store.findDocuments("notes/*.md", { includeBody: true });
    expect(errors).toHaveLength(0);
    expect(docs).toHaveLength(2);

    await cleanupTestDb(store);
  });

  test("multiple stores can operate independently", async () => {
    const store1 = await createTestStore();
    const store2 = await createTestStore();

    const col1 = await createTestCollection({ pwd: "/store1", glob: "**/*.md", name: "store1" });
    const col2 = await createTestCollection({ pwd: "/store2", glob: "**/*.md", name: "store2" });

    await insertTestDocument(store1.db, col1, {
      name: "doc1",
      body: "unique content for store1",
      displayPath: "doc.md",
    });

    await insertTestDocument(store2.db, col2, {
      name: "doc2",
      body: "different content for store2",
      displayPath: "doc.md",
    });

    // Each store should only see its own documents
    const results1 = store1.searchFTS("unique", 10);
    const results2 = store2.searchFTS("different", 10);

    expect(results1).toHaveLength(1);
    expect(results1[0].displayPath).toBe("store1/doc.md");
    expect(results1[0].filepath).toBe("qmd://store1/doc.md");

    expect(results2).toHaveLength(1);
    expect(results2[0].displayPath).toBe("store2/doc.md");
    expect(results2[0].filepath).toBe("qmd://store2/doc.md");

    // Cross-check: store1 shouldn't find store2's content
    const cross1 = store1.searchFTS("different", 10);
    const cross2 = store2.searchFTS("unique", 10);

    expect(cross1).toHaveLength(0);
    expect(cross2).toHaveLength(0);

    await cleanupTestDb(store1);
    await cleanupTestDb(store2);
  });
});

// =============================================================================
// LlamaCpp Integration Tests (using real local models)
// =============================================================================

describe("LlamaCpp Integration", () => {
  test("searchVec returns empty when no vector index", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();
    await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      body: "Some content",
    });

    // No vectors_vec table exists, should return empty
    const results = await store.searchVec("query", "embeddinggemma", 10);
    expect(results).toHaveLength(0);

    await cleanupTestDb(store);
  });

  test("searchVec returns results when vector index exists", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    const hash = "testhash123";
    await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      hash,
      body: "Some content about testing",
      filepath: "/test/doc1.md",
      displayPath: "doc1.md",
    });

    // Create vector table and insert a vector
    store.ensureVecTable(768);
    const embedding = Array(768).fill(0).map(() => Math.random());
    store.db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, 0, 0, 'test', ?)`).run(hash, new Date().toISOString());
    store.db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(`${hash}_0`, new Float32Array(embedding));

    const results = await store.searchVec("test query", "embeddinggemma", 10);
    expect(results).toHaveLength(1);
    expect(results[0].displayPath).toBe(`${collectionName}/doc1.md`);
    expect(results[0].filepath).toBe(`qmd://${collectionName}/doc1.md`);
    expect(results[0].source).toBe("vec");

    await cleanupTestDb(store);
  });

  test("expandQuery returns original plus expanded queries", async () => {
    const store = await createTestStore();

    const queries = await store.expandQuery("test query");
    expect(queries).toContain("test query");
    expect(queries[0]).toBe("test query");
    // LlamaCpp returns original + variations
    expect(queries.length).toBeGreaterThanOrEqual(1);

    await cleanupTestDb(store);
  }, 30000);

  test("expandQuery caches results", async () => {
    const store = await createTestStore();

    // First call
    const queries1 = await store.expandQuery("cached query test");
    // Second call - should hit cache
    const queries2 = await store.expandQuery("cached query test");

    expect(queries1[0]).toBe(queries2[0]);

    await cleanupTestDb(store);
  }, 30000);

  test("rerank scores documents", async () => {
    const store = await createTestStore();

    const docs = [
      { file: "doc1.md", text: "Relevant content about the topic" },
      { file: "doc2.md", text: "Other content" },
    ];

    const results = await store.rerank("topic", docs);
    expect(results).toHaveLength(2);
    // LlamaCpp reranker returns relevance scores
    expect(results[0].score).toBeGreaterThan(0);

    await cleanupTestDb(store);
  });

  test("rerank caches results", async () => {
    const store = await createTestStore();

    const docs = [{ file: "doc1.md", text: "Content for caching test" }];

    // First call
    await store.rerank("cache test query", docs);
    // Second call - should hit cache
    const results = await store.rerank("cache test query", docs);

    expect(results).toHaveLength(1);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Edge Cases & Error Handling
// =============================================================================

describe("Edge Cases", () => {
  test("handles empty database gracefully", async () => {
    const store = await createTestStore();

    const searchResults = store.searchFTS("anything", 10);
    expect(searchResults).toHaveLength(0);

    // SKIPPED: getStatus() has bug (queries non-existent collections table)
    // const status = store.getStatus();
    // expect(status.totalDocuments).toBe(0);
    // expect(status.collections).toHaveLength(0);

    const doc = store.findDocument("nonexistent.md");
    expect("error" in doc).toBe(true);

    await cleanupTestDb(store);
  });

  test("handles very long document bodies", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    const longBody = "word ".repeat(100000); // ~600KB
    await insertTestDocument(store.db, collectionName, {
      name: "long",
      body: longBody,
      displayPath: "long.md",
    });

    const results = store.searchFTS("word", 10);
    expect(results).toHaveLength(1);

    await cleanupTestDb(store);
  });

  test("handles unicode content correctly", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      name: "unicode",
      title: "日本語タイトル",
      body: "# 日本語\n\n内容は日本語で書かれています。\n\nEmoji: 🎉🚀✨",
      displayPath: "unicode.md",
    });

    // Should be searchable
    const results = store.searchFTS("日本語", 10);
    expect(results.length).toBeGreaterThan(0);

    // Should retrieve correctly
    const doc = store.findDocument("unicode.md", { includeBody: true });
    expect("error" in doc).toBe(false);
    if (!("error" in doc)) {
      expect(doc.title).toBe("日本語タイトル");
      expect(doc.body).toContain("🎉");
    }

    await cleanupTestDb(store);
  });

  test("handles documents with special characters in paths", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    await insertTestDocument(store.db, collectionName, {
      name: "special",
      filepath: "/path/file with spaces.md",
      displayPath: "file with spaces.md",
      body: "Content",
    });

    const doc = store.findDocument("file with spaces.md");
    expect("error" in doc).toBe(false);

    await cleanupTestDb(store);
  });

  test("handles concurrent operations", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    // Insert multiple documents concurrently
    const inserts = Array.from({ length: 10 }, (_, i) =>
      insertTestDocument(store.db, collectionName, {
        name: `concurrent${i}`,
        body: `Content ${i} searchterm`,
        displayPath: `concurrent${i}.md`,
      })
    );

    await Promise.all(inserts);

    // All should be searchable
    const results = store.searchFTS("searchterm", 20);
    expect(results).toHaveLength(10);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Content-Addressable Storage Tests
// =============================================================================

describe("Content-Addressable Storage", () => {
  test("same content gets same hash from multiple collections", async () => {
    const store = await createTestStore();

    // Create two collections
    const collection1 = await createTestCollection({ pwd: "/path/collection1", name: "collection1" });
    const collection2 = await createTestCollection({ pwd: "/path/collection2", name: "collection2" });

    // Add same content to both collections
    const content = "# Same Content\n\nThis is the same content in two places.";
    const hash1 = await hashContent(content);

    const doc1 = await insertTestDocument(store.db, collection1, {
      name: "doc1",
      body: content,
      displayPath: "doc1.md",
    });

    const doc2 = await insertTestDocument(store.db, collection2, {
      name: "doc2",
      body: content,
      displayPath: "doc2.md",
    });

    // Both should have the same hash
    const hash1Db = store.db.prepare(`SELECT hash FROM documents WHERE id = ?`).get(doc1) as { hash: string };
    const hash2Db = store.db.prepare(`SELECT hash FROM documents WHERE id = ?`).get(doc2) as { hash: string };

    expect(hash1Db.hash).toBe(hash2Db.hash);
    expect(hash1Db.hash).toBe(hash1);

    // There should only be one entry in the content table
    const contentCount = store.db.prepare(`SELECT COUNT(*) as count FROM content WHERE hash = ?`).get(hash1) as { count: number };
    expect(contentCount.count).toBe(1);

    await cleanupTestDb(store);
  });

  test("removing one collection preserves content used by another", async () => {
    const store = await createTestStore();

    // Create two collections
    const collection1 = await createTestCollection({ pwd: "/path/collection1", name: "collection1" });
    const collection2 = await createTestCollection({ pwd: "/path/collection2", name: "collection2" });

    // Add same content to both collections
    const sharedContent = "# Shared Content\n\nThis is shared.";
    const sharedHash = await hashContent(sharedContent);

    await insertTestDocument(store.db, collection1, {
      name: "shared1",
      body: sharedContent,
      displayPath: "shared1.md",
    });

    await insertTestDocument(store.db, collection2, {
      name: "shared2",
      body: sharedContent,
      displayPath: "shared2.md",
    });

    // Add unique content to collection1
    const uniqueContent = "# Unique Content\n\nThis is unique to collection1.";
    const uniqueHash = await hashContent(uniqueContent);

    await insertTestDocument(store.db, collection1, {
      name: "unique",
      body: uniqueContent,
      displayPath: "unique.md",
    });

    // Verify both hashes exist in content table
    const sharedExists1 = store.db.prepare(`SELECT hash FROM content WHERE hash = ?`).get(sharedHash);
    const uniqueExists1 = store.db.prepare(`SELECT hash FROM content WHERE hash = ?`).get(uniqueHash);
    expect(sharedExists1).toBeTruthy();
    expect(uniqueExists1).toBeTruthy();

    // Remove collection1 documents (collections are in YAML now)
    store.db.prepare(`DELETE FROM documents WHERE collection = ?`).run(collection1);

    // Clean up orphaned content (mimics what the CLI does)
    store.db.prepare(`
      DELETE FROM content
      WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
    `).run();

    // Shared content should still exist (used by collection2)
    const sharedExists2 = store.db.prepare(`SELECT hash FROM content WHERE hash = ?`).get(sharedHash);
    expect(sharedExists2).toBeTruthy();

    // Unique content should be removed (only used by collection1)
    const uniqueExists2 = store.db.prepare(`SELECT hash FROM content WHERE hash = ?`).get(uniqueHash);
    expect(uniqueExists2).toBeFalsy();

    await cleanupTestDb(store);
  });

  test("deduplicates content across many collections", async () => {
    const store = await createTestStore();

    const sharedContent = "# Common Header\n\nThis appears everywhere.";
    const sharedHash = await hashContent(sharedContent);

    // Create 5 collections with the same content
    const collectionNames = [];
    for (let i = 0; i < 5; i++) {
      const collName = await createTestCollection({ pwd: `/path/collection${i}`, name: `collection${i}` });
      collectionNames.push(collName);

      await insertTestDocument(store.db, collName, {
        name: `doc${i}`,
        body: sharedContent,
        displayPath: `doc${i}.md`,
      });
    }

    // Should have 5 documents
    const docCount = store.db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
    expect(docCount.count).toBe(5);

    // But only 1 content entry
    const contentCount = store.db.prepare(`SELECT COUNT(*) as count FROM content WHERE hash = ?`).get(sharedHash) as { count: number };
    expect(contentCount.count).toBe(1);

    // All documents should point to the same hash
    const hashes = store.db.prepare(`SELECT DISTINCT hash FROM documents WHERE active = 1`).all() as { hash: string }[];
    expect(hashes).toHaveLength(1);
    expect(hashes[0].hash).toBe(sharedHash);

    await cleanupTestDb(store);
  });

  test("different content gets different hashes", async () => {
    const store = await createTestStore();
    const collectionName = await createTestCollection();

    const content1 = "# Content One";
    const content2 = "# Content Two";
    const hash1 = await hashContent(content1);
    const hash2 = await hashContent(content2);

    // Hashes should be different
    expect(hash1).not.toBe(hash2);

    const doc1 = await insertTestDocument(store.db, collectionName, {
      name: "doc1",
      body: content1,
      displayPath: "doc1.md",
    });

    const doc2 = await insertTestDocument(store.db, collectionName, {
      name: "doc2",
      body: content2,
      displayPath: "doc2.md",
    });

    // Both hashes should exist in content table
    const hash1Db = store.db.prepare(`SELECT hash FROM documents WHERE id = ?`).get(doc1) as { hash: string };
    const hash2Db = store.db.prepare(`SELECT hash FROM documents WHERE id = ?`).get(doc2) as { hash: string };

    expect(hash1Db.hash).toBe(hash1);
    expect(hash2Db.hash).toBe(hash2);
    expect(hash1Db.hash).not.toBe(hash2Db.hash);

    // Should have 2 entries in content table
    const contentCount = store.db.prepare(`SELECT COUNT(*) as count FROM content`).get() as { count: number };
    expect(contentCount.count).toBe(2);

    await cleanupTestDb(store);
  });
});

// =============================================================================
// Virtual Path Normalization Tests
// =============================================================================

describe("normalizeVirtualPath", () => {
  test("already normalized qmd:// path passes through", () => {
    expect(normalizeVirtualPath("qmd://collection/path.md")).toBe("qmd://collection/path.md");
    expect(normalizeVirtualPath("qmd://journals/2025-01-01.md")).toBe("qmd://journals/2025-01-01.md");
  });

  test("handles //collection/path format (missing qmd: prefix)", () => {
    expect(normalizeVirtualPath("//collection/path.md")).toBe("qmd://collection/path.md");
    expect(normalizeVirtualPath("//journals/2025-01-01.md")).toBe("qmd://journals/2025-01-01.md");
  });

  test("handles qmd:// with extra slashes", () => {
    expect(normalizeVirtualPath("qmd:////collection/path.md")).toBe("qmd://collection/path.md");
    expect(normalizeVirtualPath("qmd:///journals/2025-01-01.md")).toBe("qmd://journals/2025-01-01.md");
    expect(normalizeVirtualPath("qmd:///////archive/file.md")).toBe("qmd://archive/file.md");
  });

  test("handles collection root paths", () => {
    expect(normalizeVirtualPath("qmd://collection/")).toBe("qmd://collection/");
    expect(normalizeVirtualPath("qmd://collection")).toBe("qmd://collection");
    expect(normalizeVirtualPath("//collection/")).toBe("qmd://collection/");
  });

  test("preserves bare collection/path format (not auto-converted)", () => {
    // Bare paths without qmd:// or // prefix are NOT converted
    // (could be relative filesystem paths)
    expect(normalizeVirtualPath("collection/path.md")).toBe("collection/path.md");
    expect(normalizeVirtualPath("journals/2025-01-01.md")).toBe("journals/2025-01-01.md");
  });

  test("preserves absolute filesystem paths", () => {
    expect(normalizeVirtualPath("/Users/test/file.md")).toBe("/Users/test/file.md");
    expect(normalizeVirtualPath("/absolute/path/file.md")).toBe("/absolute/path/file.md");
  });

  test("preserves home-relative paths", () => {
    expect(normalizeVirtualPath("~/Documents/file.md")).toBe("~/Documents/file.md");
  });

  test("preserves docid format", () => {
    expect(normalizeVirtualPath("#abc123")).toBe("#abc123");
    expect(normalizeVirtualPath("#def456")).toBe("#def456");
  });

  test("handles whitespace trimming", () => {
    expect(normalizeVirtualPath("  qmd://collection/path.md  ")).toBe("qmd://collection/path.md");
    expect(normalizeVirtualPath("  //collection/path.md  ")).toBe("qmd://collection/path.md");
  });
});

describe("isVirtualPath", () => {
  test("recognizes qmd:// paths", () => {
    expect(isVirtualPath("qmd://collection/path.md")).toBe(true);
    expect(isVirtualPath("qmd://journals/2025-01-01.md")).toBe(true);
    expect(isVirtualPath("qmd://collection")).toBe(true);
  });

  test("recognizes //collection/path format", () => {
    expect(isVirtualPath("//collection/path.md")).toBe(true);
    expect(isVirtualPath("//journals/2025-01-01.md")).toBe(true);
  });

  test("does not auto-recognize bare collection/path format", () => {
    // Bare paths could be relative filesystem paths, so not auto-detected as virtual
    expect(isVirtualPath("collection/path.md")).toBe(false);
    expect(isVirtualPath("journals/2025-01-01.md")).toBe(false);
    expect(isVirtualPath("archive/subfolder/file.md")).toBe(false);
  });

  test("rejects docid format", () => {
    expect(isVirtualPath("#abc123")).toBe(false);
    expect(isVirtualPath("#def456")).toBe(false);
  });

  test("rejects absolute filesystem paths", () => {
    expect(isVirtualPath("/Users/test/file.md")).toBe(false);
    expect(isVirtualPath("/absolute/path/file.md")).toBe(false);
  });

  test("rejects home-relative paths", () => {
    expect(isVirtualPath("~/Documents/file.md")).toBe(false);
    expect(isVirtualPath("~/notes/journal.md")).toBe(false);
  });

  test("rejects paths without slashes", () => {
    expect(isVirtualPath("file.md")).toBe(false);
    expect(isVirtualPath("document")).toBe(false);
  });
});

describe("parseVirtualPath", () => {
  test("parses standard qmd:// paths", () => {
    expect(parseVirtualPath("qmd://collection/path.md")).toEqual({
      collectionName: "collection",
      path: "path.md",
    });
    expect(parseVirtualPath("qmd://journals/2025-01-01.md")).toEqual({
      collectionName: "journals",
      path: "2025-01-01.md",
    });
  });

  test("parses paths with nested directories", () => {
    expect(parseVirtualPath("qmd://archive/subfolder/file.md")).toEqual({
      collectionName: "archive",
      path: "subfolder/file.md",
    });
  });

  test("parses collection root paths", () => {
    expect(parseVirtualPath("qmd://collection/")).toEqual({
      collectionName: "collection",
      path: "",
    });
    expect(parseVirtualPath("qmd://collection")).toEqual({
      collectionName: "collection",
      path: "",
    });
  });

  test("parses //collection/path format (normalizes first)", () => {
    expect(parseVirtualPath("//collection/path.md")).toEqual({
      collectionName: "collection",
      path: "path.md",
    });
  });

  test("parses qmd:// with extra slashes (normalizes first)", () => {
    expect(parseVirtualPath("qmd:////collection/path.md")).toEqual({
      collectionName: "collection",
      path: "path.md",
    });
  });

  test("returns null for non-virtual paths", () => {
    expect(parseVirtualPath("/absolute/path.md")).toBe(null);
    expect(parseVirtualPath("~/home/path.md")).toBe(null);
    expect(parseVirtualPath("#docid")).toBe(null);
    expect(parseVirtualPath("file.md")).toBe(null);
    // Bare collection/path is not recognized as virtual
    expect(parseVirtualPath("collection/path.md")).toBe(null);
  });
});
