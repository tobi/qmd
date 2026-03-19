/**
 * sdk.test.ts - Unit tests for the QMD SDK (library mode)
 *
 * Tests the public API exposed via `@tobilu/qmd` (src/index.ts).
 * Uses inline config (no YAML files) to verify the SDK works self-contained.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { existsSync, writeFileSync, mkdirSync, readFileSync } from "node:fs";
import YAML from "yaml";
import {
  createStore,
  type QMDStore,
  type CollectionConfig,
  type StoreOptions,
  type UpdateProgress,
  type SearchOptions,
  type LexSearchOptions,
  type VectorSearchOptions,
  type ExpandQueryOptions,
} from "../src/index.js";
import { setDefaultLlamaCpp } from "../src/llm.js";

// =============================================================================
// Test Helpers
// =============================================================================

let testDir: string;
let docsDir: string;
let notesDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-sdk-test-"));
  docsDir = join(testDir, "docs");
  notesDir = join(testDir, "notes");

  // Create test directories with sample markdown files
  await mkdir(docsDir, { recursive: true });
  await mkdir(notesDir, { recursive: true });

  await writeFile(join(docsDir, "readme.md"), "# Getting Started\n\nThis is the getting started guide for the project.\n");
  await writeFile(join(docsDir, "auth.md"), "# Authentication\n\nAuthentication uses JWT tokens for session management.\nUsers log in with email and password.\n");
  await writeFile(join(docsDir, "api.md"), "# API Reference\n\n## Endpoints\n\n### POST /login\nAuthenticate a user.\n\n### GET /users\nList all users.\n");
  await writeFile(join(notesDir, "meeting-2025-01.md"), "# January Planning Meeting\n\nDiscussed Q1 roadmap and resource allocation.\n");
  await writeFile(join(notesDir, "meeting-2025-02.md"), "# February Standup\n\nReviewed sprint progress. Authentication feature is on track.\n");
  await writeFile(join(notesDir, "ideas.md"), "# Project Ideas\n\n- Build a search engine\n- Create a knowledge base\n- Implement vector search\n");
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {
    // Ignore cleanup errors
  }
});

function freshDbPath(): string {
  return join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

function setStoreDocumentTimestamps(
  store: QMDStore,
  path: string,
  timestamps: { modifiedAt?: string; createdAt?: string }
): void {
  const db = store.internal.db;
  if (timestamps.modifiedAt) {
    db.prepare(`UPDATE documents SET modified_at = ? WHERE path = ?`).run(timestamps.modifiedAt, path);
  }
  if (timestamps.createdAt) {
    db.prepare(`UPDATE documents SET created_at = ? WHERE path = ?`).run(timestamps.createdAt, path);
  }
}

// =============================================================================
// Constructor Tests
// =============================================================================

describe("createStore", () => {
  test("creates store with inline config", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    expect(store).toBeDefined();
    expect(store.dbPath).toBeTruthy();
    expect(store.internal).toBeDefined();
    await store.close();
  });

  test("creates store with YAML config file", async () => {
    const configPath = join(testDir, "test-config.yml");
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = await createStore({
      dbPath: freshDbPath(),
      configPath,
    });

    expect(store).toBeDefined();
    await store.close();
  });

  test("throws if dbPath is missing", async () => {
    await expect(
      createStore({ dbPath: "", config: { collections: {} } })
    ).rejects.toThrow("dbPath is required");
  });

  test("opens with just dbPath (DB-only mode)", async () => {
    const store = await createStore({ dbPath: freshDbPath() } as StoreOptions);
    expect(store).toBeDefined();
    // No collections yet — fresh DB
    const collections = await store.listCollections();
    expect(collections).toEqual([]);
    await store.close();
  });

  test("throws if both configPath and config are provided", async () => {
    await expect(
      createStore({
        dbPath: freshDbPath(),
        configPath: "/some/path.yml",
        config: { collections: {} },
      })
    ).rejects.toThrow("Provide either configPath or config, not both");
  });

  test("creates database file on disk", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(existsSync(dbPath)).toBe(true);
    await store.close();
  });

  test("store.dbPath matches the provided path", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(store.dbPath).toBe(dbPath);
    await store.close();
  });
});

// =============================================================================
// Collection Management Tests
// =============================================================================

describe("collection management", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("addCollection adds a collection to inline config", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });

    const collections = await store.listCollections();
    const names = collections.map(c => c.name);
    expect(names).toContain("docs");
  });

  test("addCollection with default pattern", async () => {
    await store.addCollection("notes", { path: notesDir });

    const collections = await store.listCollections();
    expect(collections.find(c => c.name === "notes")).toBeDefined();
  });

  test("removeCollection removes existing collection", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    const removed = await store.removeCollection("docs");

    expect(removed).toBe(true);
    const collections = await store.listCollections();
    expect(collections.map(c => c.name)).not.toContain("docs");
  });

  test("removeCollection returns false for non-existent collection", async () => {
    const removed = await store.removeCollection("nonexistent");
    expect(removed).toBe(false);
  });

  test("renameCollection renames a collection", async () => {
    await store.addCollection("old-name", { path: docsDir, pattern: "**/*.md" });
    const renamed = await store.renameCollection("old-name", "new-name");

    expect(renamed).toBe(true);
    const names = (await store.listCollections()).map(c => c.name);
    expect(names).toContain("new-name");
    expect(names).not.toContain("old-name");
  });

  test("renameCollection returns false for non-existent source", async () => {
    const renamed = await store.renameCollection("nonexistent", "new-name");
    expect(renamed).toBe(false);
  });

  test("renameCollection throws if target exists", async () => {
    await store.addCollection("a", { path: docsDir, pattern: "**/*.md" });
    await store.addCollection("b", { path: notesDir, pattern: "**/*.md" });

    await expect(store.renameCollection("a", "b")).rejects.toThrow("already exists");
  });

  test("listCollections returns empty array for empty config", async () => {
    const collections = await store.listCollections();
    expect(collections).toEqual([]);
  });

  test("multiple collections can be added", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    await store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });

    const names = (await store.listCollections()).map(c => c.name);
    expect(names).toContain("docs");
    expect(names).toContain("notes");
    expect(names).toHaveLength(2);
  });
});

// =============================================================================
// Context Management Tests
// =============================================================================

describe("context management", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("addContext adds context to a collection path", async () => {
    const added = await store.addContext("docs", "/auth", "Authentication docs");
    expect(added).toBe(true);

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
  });

  test("addContext returns false for non-existent collection", async () => {
    const added = await store.addContext("nonexistent", "/path", "Some context");
    expect(added).toBe(false);
  });

  test("removeContext removes existing context", async () => {
    await store.addContext("docs", "/auth", "Authentication docs");
    const removed = await store.removeContext("docs", "/auth");

    expect(removed).toBe(true);
    const contexts = await store.listContexts();
    expect(contexts.find(c => c.path === "/auth")).toBeUndefined();
  });

  test("removeContext returns false for non-existent context", async () => {
    const removed = await store.removeContext("docs", "/nonexistent");
    expect(removed).toBe(false);
  });

  test("setGlobalContext sets and retrieves global context", async () => {
    await store.setGlobalContext("Global knowledge base");
    const global = await store.getGlobalContext();

    expect(global).toBe("Global knowledge base");
  });

  test("setGlobalContext with undefined clears it", async () => {
    await store.setGlobalContext("Some context");
    await store.setGlobalContext(undefined);
    const global = await store.getGlobalContext();

    expect(global).toBeUndefined();
  });

  test("listContexts includes global context", async () => {
    await store.setGlobalContext("Global context");
    const contexts = await store.listContexts();

    expect(contexts).toContainEqual({
      collection: "*",
      path: "/",
      context: "Global context",
    });
  });

  test("listContexts returns contexts across multiple collections", async () => {
    await store.addContext("docs", "/", "Documentation");
    await store.addContext("notes", "/", "Personal notes");

    const contexts = await store.listContexts();
    expect(contexts.filter(c => c.path === "/")).toHaveLength(2);
  });

  test("multiple contexts on same collection", async () => {
    await store.addContext("docs", "/auth", "Auth docs");
    await store.addContext("docs", "/api", "API docs");

    const contexts = (await store.listContexts()).filter(c => c.collection === "docs");
    expect(contexts).toHaveLength(2);
    expect(contexts.map(c => c.path).sort()).toEqual(["/api", "/auth"]);
  });

  test("addContext overwrites existing context for same path", async () => {
    await store.addContext("docs", "/auth", "Old context");
    await store.addContext("docs", "/auth", "New context");

    const contexts = (await store.listContexts()).filter(c => c.path === "/auth");
    expect(contexts).toHaveLength(1);
    expect(contexts[0]!.context).toBe("New context");
  });
});

// =============================================================================
// Inline Config Isolation Tests
// =============================================================================

describe("inline config isolation", () => {
  test("inline config does not write any files to disk", async () => {
    const configDir = join(testDir, "should-not-exist");
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });
    await store.addContext("docs", "/", "Documentation");

    expect(existsSync(configDir)).toBe(false);
    await store.close();
  });

  test("inline config mutations persist within session", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    await store.addContext("docs", "/", "My docs");

    // Verify the mutations are visible
    const collections = await store.listCollections();
    expect(collections.map(c => c.name)).toContain("docs");

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/",
      context: "My docs",
    });

    await store.close();
  });

  test("two stores with different inline configs are independent", async () => {
    const store1 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Close first store (resets config source)
    await store1.close();

    const store2 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = (await store2.listCollections()).map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    await store2.close();
  });
});

// =============================================================================
// YAML Config File Tests
// =============================================================================

describe("YAML config file mode", () => {
  test("loads collections from YAML file", async () => {
    const configPath = join(testDir, `config-${Date.now()}.yml`);
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
        notes: { path: notesDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    const names = (await store.listCollections()).map(c => c.name);

    expect(names).toContain("docs");
    expect(names).toContain("notes");
    await store.close();
  });

  test("addCollection persists to YAML file", async () => {
    const configPath = join(testDir, `config-persist-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({ collections: {} }));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    await store.addCollection("newcol", { path: docsDir, pattern: "**/*.md" });
    await store.close();

    // Read the YAML file directly and verify
    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections).toHaveProperty("newcol");
    expect(parsed.collections.newcol!.path).toBe(docsDir);
  });

  test("context persists to YAML file", async () => {
    const configPath = join(testDir, `config-ctx-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({
      collections: { docs: { path: docsDir, pattern: "**/*.md" } },
    }));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    await store.addContext("docs", "/api", "API documentation");
    await store.close();

    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections.docs!.context).toEqual({ "/api": "API documentation" });
  });

  test("non-existent config file returns empty collections", async () => {
    const configPath = join(testDir, "nonexistent-config.yml");
    const store = await createStore({ dbPath: freshDbPath(), configPath });
    const collections = await store.listCollections();

    expect(collections).toEqual([]);
    await store.close();
  });
});

// =============================================================================
// Search Tests (BM25 - no LLM needed)
// =============================================================================

describe("searchLex (BM25)", () => {
  let store: QMDStore;
  let dbPath: string;

  beforeAll(async () => {
    dbPath = join(testDir, "search-test.sqlite");
    store = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    // Index documents manually using internal store
    const now = new Date().toISOString();
    const { internal } = store;
    const fs = require("fs");

    // Index docs collection
    for (const file of ["readme.md", "auth.md", "api.md"]) {
      const fullPath = join(docsDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("docs", `qmd://docs/${file}`, title, hash, now, now);
    }

    // Index notes collection
    for (const file of ["meeting-2025-01.md", "meeting-2025-02.md", "ideas.md"]) {
      const fullPath = join(notesDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("notes", `qmd://notes/${file}`, title, hash, now, now);
    }
  });

  afterAll(async () => {
    await store.close();
  });

  test("searchLex returns results for matching query", async () => {
    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);
  });

  test("searchLex results have expected shape", async () => {
    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    const result = results[0]!;
    expect(result).toHaveProperty("filepath");
    expect(result).toHaveProperty("score");
    expect(result).toHaveProperty("title");
    expect(result).toHaveProperty("docid");
    expect(result).toHaveProperty("collectionName");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThan(0);
  });

  test("searchLex respects limit option", async () => {
    const results = await store.searchLex("meeting", { limit: 1 });
    expect(results.length).toBeLessThanOrEqual(1);
  });

  test("searchLex with collection filter", async () => {
    const results = await store.searchLex("authentication", { collection: "notes" });
    for (const r of results) {
      expect(r.collectionName).toBe("notes");
    }
  });

  test("searchLex returns empty for non-matching query", async () => {
    const results = await store.searchLex("xyznonexistentterm123");
    expect(results).toHaveLength(0);
  });

  test("searchLex finds documents across collections", async () => {
    const results = await store.searchLex("authentication", { limit: 10 });
    const collections = new Set(results.map(r => r.collectionName));
    // Auth appears in both docs/auth.md and notes/meeting-2025-02.md
    expect(collections.size).toBeGreaterThanOrEqual(1);
  });
});

// =============================================================================
// Unified search() API Tests
// =============================================================================

describe("search (unified API)", () => {
  let store: QMDStore;

  beforeAll(async () => {
    store = await createStore({
      dbPath: join(testDir, "unified-search-test.sqlite"),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });
    await store.update();
  });

  afterAll(async () => {
    await store.close();
  });

  test("search() requires query or queries", async () => {
    await expect(store.search({} as SearchOptions)).rejects.toThrow("requires either 'query' or 'queries'");
  });

  test("search() with pre-expanded queries and rerank:false", async () => {
    const results = await store.search({
      queries: [
        { type: "lex", query: "authentication JWT" },
        { type: "lex", query: "login session" },
      ],
      rerank: false,
    });
    expect(results.length).toBeGreaterThan(0);
  });

  // Tests below use search({ query: ... }) which triggers LLM query expansion
  describe.skipIf(!!process.env.CI)("with LLM query expansion", () => {
    test("search() with query and rerank:false returns results", async () => {
      const results = await store.search({ query: "authentication", rerank: false });
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty("file");
      expect(results[0]).toHaveProperty("score");
      expect(results[0]).toHaveProperty("title");
      expect(results[0]).toHaveProperty("bestChunk");
      expect(results[0]).toHaveProperty("docid");
    });

    test("search() with intent and rerank:false returns results", async () => {
      const results = await store.search({
        query: "meeting",
        intent: "quarterly planning and roadmap",
        rerank: false,
      });
      expect(results.length).toBeGreaterThan(0);
    });

    test("search() with collection filter", async () => {
      const results = await store.search({
        query: "authentication",
        collection: "docs",
        rerank: false,
      });
      for (const r of results) {
        expect(r.file).toMatch(/^qmd:\/\/docs\//);
      }
    });

    test("search() with collections filter", async () => {
      const results = await store.search({
        query: "authentication",
        collections: ["docs"],
        rerank: false,
      });
      for (const r of results) {
        expect(r.file).toMatch(/^qmd:\/\/docs\//);
      }
    });

    test("search() with limit", async () => {
      const results = await store.search({ query: "meeting", limit: 1, rerank: false });
      expect(results.length).toBeLessThanOrEqual(1);
    });

    test("search() returns empty for non-matching query", async () => {
      const results = await store.search({ query: "xyznonexistentterm123", rerank: false });
      expect(results).toHaveLength(0);
    });
  });
});

// =============================================================================
// Document Retrieval Tests
// =============================================================================

describe("get and multiGet", () => {
  let store: QMDStore;

  beforeAll(async () => {
    store = await createStore({
      dbPath: join(testDir, "get-test.sqlite"),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Index documents
    const now = new Date().toISOString();
    const { internal } = store;
    const fs = require("fs");

    for (const file of ["readme.md", "auth.md", "api.md"]) {
      const fullPath = join(docsDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("docs", `qmd://docs/${file}`, title, hash, now, now);
    }
  });

  afterAll(async () => {
    await store.close();
  });

  test("get retrieves a document by path", async () => {
    const result = await store.get("qmd://docs/auth.md");

    expect("error" in result).toBe(false);
    if (!("error" in result)) {
      expect(result.title).toBe("Authentication");
      expect(result.collectionName).toBe("docs");
    }
  });

  test("get with includeBody returns body content", async () => {
    const result = await store.get("qmd://docs/auth.md", { includeBody: true });

    if (!("error" in result)) {
      expect(result.body).toBeDefined();
      expect(result.body).toContain("JWT tokens");
    }
  });

  test("get returns not_found for missing document", async () => {
    const result = await store.get("qmd://docs/nonexistent.md");

    expect("error" in result).toBe(true);
    if ("error" in result) {
      expect(result.error).toBe("not_found");
    }
  });

  test("get by docid", async () => {
    // First get a document to find its docid
    const doc = await store.get("qmd://docs/readme.md");
    if (!("error" in doc)) {
      const byDocid = await store.get(`#${doc.docid}`);
      expect("error" in byDocid).toBe(false);
      if (!("error" in byDocid)) {
        expect(byDocid.docid).toBe(doc.docid);
      }
    }
  });

  test("multiGet retrieves multiple documents", async () => {
    const { docs, errors } = await store.multiGet("qmd://docs/*.md");
    expect(docs.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Index Health Tests
// =============================================================================

describe("index health", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("getStatus returns valid structure", async () => {
    const status = await store.getStatus();

    expect(status).toHaveProperty("totalDocuments");
    expect(status).toHaveProperty("needsEmbedding");
    expect(status).toHaveProperty("hasVectorIndex");
    expect(status).toHaveProperty("collections");
    expect(typeof status.totalDocuments).toBe("number");
  });

  test("getIndexHealth returns valid structure", async () => {
    const health = await store.getIndexHealth();

    expect(health).toHaveProperty("needsEmbedding");
    expect(health).toHaveProperty("totalDocs");
    expect(typeof health.needsEmbedding).toBe("number");
    expect(typeof health.totalDocs).toBe("number");
  });

  test("fresh store has zero documents", async () => {
    const status = await store.getStatus();
    expect(status.totalDocuments).toBe(0);
  });
});

// =============================================================================
// Update Tests
// =============================================================================

describe("update", () => {
  test("indexes files and returns correct stats", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update();

    expect(result.collections).toBe(1);
    expect(result.indexed).toBe(3); // readme.md, auth.md, api.md
    expect(result.updated).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.removed).toBe(0);
    expect(typeof result.needsEmbedding).toBe("number");

    await store.close();
  });

  test("second update shows unchanged files", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    const result = await store.update();

    expect(result.indexed).toBe(0);
    expect(result.unchanged).toBe(3);

    await store.close();
  });

  test("update with onProgress callback fires", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const progress: UpdateProgress[] = [];
    await store.update({
      onProgress: (info) => progress.push(info),
    });

    expect(progress.length).toBeGreaterThan(0);
    expect(progress[0]!.collection).toBe("docs");
    expect(progress[0]!.current).toBeGreaterThanOrEqual(1);
    expect(progress[0]!.total).toBe(3);

    await store.close();
  });

  test("update with collection filter", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update({ collections: ["docs"] });

    expect(result.collections).toBe(1);
    expect(result.indexed).toBe(3); // Only docs

    await store.close();
  });

  test("update multiple collections", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update();

    expect(result.collections).toBe(2);
    expect(result.indexed).toBe(6); // 3 docs + 3 notes

    await store.close();
  });

  test("documents are searchable after update", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();

    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    await store.close();
  });

  test("searchLex filters by modifiedAfter", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    setStoreDocumentTimestamps(store, "auth.md", { modifiedAt: "2026-03-15T12:00:00.000Z" });
    setStoreDocumentTimestamps(store, "api.md", { modifiedAt: "2026-02-01T12:00:00.000Z" });

    const results = await store.searchLex("authentication", {
      modifiedAfter: "2026-03-10T00:00:00Z",
    });

    expect(results.map(r => r.displayPath)).toEqual(["docs/auth.md"]);

    await store.close();
  });

  test("unified search filters by createdAfter", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    setStoreDocumentTimestamps(store, "meeting-2025-02.md", { createdAt: "2026-03-15T12:00:00.000Z" });
    setStoreDocumentTimestamps(store, "meeting-2025-01.md", { createdAt: "2026-02-01T12:00:00.000Z" });

    const results = await store.search({
      queries: [{ type: "lex", query: "February" }],
      rerank: false,
      createdAfter: "2026-03-10T00:00:00Z",
    });

    expect(results.map(r => r.displayPath)).toEqual(["notes/meeting-2025-02.md"]);

    await store.close();
  });

  test("searchLex uses inclusive modifiedAfter boundary", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    setStoreDocumentTimestamps(store, "auth.md", { modifiedAt: "2026-03-10T00:00:00.000Z" });

    const results = await store.searchLex("authentication", {
      modifiedAfter: "2026-03-10T00:00:00.000Z",
    });

    expect(results.map(r => r.displayPath)).toContain("docs/auth.md");

    await store.close();
  });

  test("searchLex applies modifiedAfter and createdAfter together", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    setStoreDocumentTimestamps(store, "meeting-2025-02.md", {
      modifiedAt: "2026-03-15T12:00:00.000Z",
      createdAt: "2026-03-15T12:00:00.000Z",
    });
    setStoreDocumentTimestamps(store, "meeting-2025-01.md", {
      modifiedAt: "2026-03-15T12:00:00.000Z",
      createdAt: "2026-02-01T12:00:00.000Z",
    });

    const results = await store.searchLex("meeting", {
      modifiedAfter: "2026-03-10T00:00:00Z",
      createdAfter: "2026-03-10T00:00:00Z",
    });

    expect(results.map(r => r.displayPath)).toEqual(["notes/meeting-2025-02.md"]);

    await store.close();
  });

  test("searchLex rejects invalid timestamp filters", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    await expect(store.searchLex("authentication", {
      modifiedAfter: "not-a-date",
    })).rejects.toThrow("Invalid modifiedAfter timestamp");

    await store.close();
  });

  test("searchVector filters by modifiedAfter", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = {
      async embed(_text: string) {
        return { embedding: new Array(768).fill(0.5), model: "fake-embed" };
      },
      async embedBatch(texts: string[]) {
        return texts.map(() => ({
          embedding: new Array(768).fill(0.5),
          model: "fake-embed",
        }));
      },
    };

    setDefaultLlamaCpp({
      async tokenize(text: string) {
        return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
      },
      async embed(_text: string) {
        return { embedding: new Array(768).fill(0.5), model: "fake-embed" };
      },
    } as any);
    store.internal.llm = fakeLlm as any;

    try {
      await store.update();
      await store.embed();
      setStoreDocumentTimestamps(store, "auth.md", { modifiedAt: "2026-03-15T12:00:00.000Z" });
      setStoreDocumentTimestamps(store, "api.md", { modifiedAt: "2026-02-01T12:00:00.000Z" });

      const results = await store.searchVector("authentication", {
        modifiedAfter: "2026-03-10T00:00:00Z",
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results.map(r => r.displayPath)).toContain("docs/auth.md");
      expect(results.map(r => r.displayPath)).not.toContain("docs/api.md");
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });
});

describe("embed", () => {
  function createFakeTokenizer() {
    return {
      async tokenize(text: string) {
        return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
      },
    };
  }

  function createFakeEmbedLlm() {
    const embedBatchCalls: string[][] = [];
    return {
      embedBatchCalls,
      async embed(_text: string) {
        return { embedding: [0.1, 0.2, 0.3], model: "fake-embed" };
      },
      async embedBatch(texts: string[]) {
        embedBatchCalls.push([...texts]);
        return texts.map((_text, index) => ({
          embedding: [index + 1, index + 2, index + 3],
          model: "fake-embed",
        }));
      },
    };
  }

  test("store.embed forwards batch limit options", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = createFakeEmbedLlm();
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = fakeLlm as any;

    try {
      await store.update();
      const result = await store.embed({
        maxDocsPerBatch: 1,
        maxBatchBytes: 1024 * 1024,
      });

      expect(fakeLlm.embedBatchCalls).toHaveLength(3);
      expect(fakeLlm.embedBatchCalls.map(call => call.length)).toEqual([1, 1, 1]);
      expect(result.docsProcessed).toBe(3);
      expect(result.chunksEmbedded).toBe(3);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.embed rejects invalid batch limits", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    try {
      await expect(store.embed({ maxDocsPerBatch: 0 })).rejects.toThrow("maxDocsPerBatch");
      await expect(store.embed({ maxBatchBytes: 0 })).rejects.toThrow("maxBatchBytes");
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });
});

// =============================================================================
// Lifecycle Tests
// =============================================================================

describe("lifecycle", () => {
  test("close() is async and does not throw", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // close() should return a promise
    const result = store.close();
    expect(result).toBeInstanceOf(Promise);
    await result;
  });

  test("close() makes subsequent operations throw", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    await store.close();

    // Database operations should fail after close
    await expect(store.getStatus()).rejects.toThrow();
  });

  test("multiple stores can coexist with different databases", async () => {
    const store1 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Note: since config source is module-level, we close store1 first
    await store1.close();

    const store2 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = (await store2.listCollections()).map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    await store2.close();
  });
});

// =============================================================================
// Config Initialization Tests
// =============================================================================

describe("config initialization", () => {
  test("inline config with global_context is preserved", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        global_context: "System knowledge base",
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const global = await store.getGlobalContext();
    expect(global).toBe("System knowledge base");
    await store.close();
  });

  test("inline config with pre-existing contexts is preserved", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            context: { "/auth": "Authentication docs" },
          },
        },
      },
    });

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
    await store.close();
  });

  test("inline config with empty collections object works", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    expect(await store.listCollections()).toEqual([]);
    expect(await store.listContexts()).toEqual([]);
    await store.close();
  });

  test("inline config with multiple collection options", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            ignore: ["drafts/**"],
            includeByDefault: true,
          },
          notes: {
            path: notesDir,
            pattern: "**/*.md",
            includeByDefault: false,
          },
        },
      },
    });

    const collections = await store.listCollections();
    expect(collections).toHaveLength(2);
    await store.close();
  });
});

// =============================================================================
// Type Export Tests (compile-time checks, runtime verification)
// =============================================================================

describe("type exports", () => {
  test("StoreOptions type is usable", () => {
    const opts: StoreOptions = {
      dbPath: "/tmp/test.sqlite",
      config: { collections: {} },
    };
    expect(opts.dbPath).toBe("/tmp/test.sqlite");
  });

  test("CollectionConfig type is usable", () => {
    const config: CollectionConfig = {
      global_context: "test",
      collections: {
        test: { path: "/tmp", pattern: "**/*.md" },
      },
    };
    expect(config.collections).toHaveProperty("test");
  });

  test("QMDStore type exposes expected methods", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // Verify all methods exist
    expect(typeof store.search).toBe("function");
    expect(typeof store.searchLex).toBe("function");
    expect(typeof store.searchVector).toBe("function");
    expect(typeof store.expandQuery).toBe("function");
    expect(typeof store.get).toBe("function");
    expect(typeof store.multiGet).toBe("function");
    expect(typeof store.addCollection).toBe("function");
    expect(typeof store.removeCollection).toBe("function");
    expect(typeof store.renameCollection).toBe("function");
    expect(typeof store.listCollections).toBe("function");
    expect(typeof store.addContext).toBe("function");
    expect(typeof store.removeContext).toBe("function");
    expect(typeof store.setGlobalContext).toBe("function");
    expect(typeof store.getGlobalContext).toBe("function");
    expect(typeof store.listContexts).toBe("function");
    expect(typeof store.getStatus).toBe("function");
    expect(typeof store.getIndexHealth).toBe("function");
    expect(typeof store.update).toBe("function");
    expect(typeof store.embed).toBe("function");
    expect(typeof store.close).toBe("function");

    await store.close();
  });
});

// =============================================================================
// DB-Only Mode Tests (self-contained store)
// =============================================================================

describe("DB-only mode", () => {
  test("reopen store with just dbPath after config+update session", async () => {
    const dbPath = freshDbPath();

    // Session 1: create store with config, update, close
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
        global_context: "Test knowledge base",
      },
    });

    await store1.update();

    // Verify documents indexed
    const status1 = await store1.getStatus();
    expect(status1.totalDocuments).toBe(6);
    await store1.close();

    // Session 2: reopen with just dbPath — no config
    const store2 = await createStore({ dbPath } as StoreOptions);

    // Collections should still be available
    const collections = await store2.listCollections();
    expect(collections.map(c => c.name).sort()).toEqual(["docs", "notes"]);

    // Search should still work
    const results = await store2.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    // Global context should still be available
    const globalCtx = await store2.getGlobalContext();
    expect(globalCtx).toBe("Test knowledge base");

    // Contexts from collections should persist
    const status2 = await store2.getStatus();
    expect(status2.totalDocuments).toBe(6);

    await store2.close();
  });

  test("config sync populates store_collections table", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            context: { "/auth": "Auth documentation" },
          },
        },
      },
    });

    // Verify collections are in the DB via listCollections
    const collections = await store.listCollections();
    expect(collections).toHaveLength(1);
    expect(collections[0]!.name).toBe("docs");
    expect(collections[0]!.pwd).toBe(docsDir);

    // Verify contexts are accessible
    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Auth documentation",
    });

    await store.close();
  });

  test("config hash skip: second init with same config skips sync", async () => {
    const dbPath = freshDbPath();
    const config = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
    };

    // First init — syncs config
    const store1 = await createStore({ dbPath, config });
    await store1.close();

    // Second init with same config — should skip sync (no-op, but should not error)
    const store2 = await createStore({ dbPath, config });
    const collections = await store2.listCollections();
    expect(collections).toHaveLength(1);
    expect(collections[0]!.name).toBe("docs");
    await store2.close();
  });

  test("DB-only mode supports collection mutations", async () => {
    const dbPath = freshDbPath();

    // Session 1: create with config
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
    await store1.close();

    // Session 2: reopen DB-only, add a collection
    const store2 = await createStore({ dbPath } as StoreOptions);
    await store2.addCollection("notes", { path: notesDir, pattern: "**/*.md" });

    const names = (await store2.listCollections()).map(c => c.name).sort();
    expect(names).toEqual(["docs", "notes"]);

    await store2.close();

    // Session 3: reopen DB-only again, verify both collections persist
    const store3 = await createStore({ dbPath } as StoreOptions);
    const names3 = (await store3.listCollections()).map(c => c.name).sort();
    expect(names3).toEqual(["docs", "notes"]);
    await store3.close();
  });

  test("DB-only mode supports context mutations", async () => {
    const dbPath = freshDbPath();

    // Session 1: create with config
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
    await store1.addContext("docs", "/api", "API docs");
    await store1.setGlobalContext("Global context");
    await store1.close();

    // Session 2: reopen DB-only
    const store2 = await createStore({ dbPath } as StoreOptions);

    const contexts = await store2.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/api",
      context: "API docs",
    });
    expect(contexts).toContainEqual({
      collection: "*",
      path: "/",
      context: "Global context",
    });

    await store2.close();
  });
});
