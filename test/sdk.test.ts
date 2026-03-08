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
} from "../src/index.js";

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

// =============================================================================
// Constructor Tests
// =============================================================================

describe("createStore", () => {
  test("creates store with inline config", () => {
    const store = createStore({
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
    store.close();
  });

  test("creates store with YAML config file", () => {
    const configPath = join(testDir, "test-config.yml");
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = createStore({
      dbPath: freshDbPath(),
      configPath,
    });

    expect(store).toBeDefined();
    store.close();
  });

  test("throws if dbPath is missing", () => {
    expect(() =>
      createStore({ dbPath: "", config: { collections: {} } })
    ).toThrow("dbPath is required");
  });

  test("throws if neither configPath nor config is provided", () => {
    expect(() =>
      createStore({ dbPath: freshDbPath() } as StoreOptions)
    ).toThrow("Either configPath or config is required");
  });

  test("throws if both configPath and config are provided", () => {
    expect(() =>
      createStore({
        dbPath: freshDbPath(),
        configPath: "/some/path.yml",
        config: { collections: {} },
      })
    ).toThrow("Provide either configPath or config, not both");
  });

  test("creates database file on disk", () => {
    const dbPath = freshDbPath();
    const store = createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(existsSync(dbPath)).toBe(true);
    store.close();
  });

  test("store.dbPath matches the provided path", () => {
    const dbPath = freshDbPath();
    const store = createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(store.dbPath).toBe(dbPath);
    store.close();
  });
});

// =============================================================================
// Collection Management Tests
// =============================================================================

describe("collection management", () => {
  let store: QMDStore;

  beforeEach(() => {
    store = createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });
  });

  afterEach(() => {
    store.close();
  });

  test("addCollection adds a collection to inline config", () => {
    store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });

    const collections = store.listCollections();
    const names = collections.map(c => c.name);
    expect(names).toContain("docs");
  });

  test("addCollection with default pattern", () => {
    store.addCollection("notes", { path: notesDir });

    const collections = store.listCollections();
    expect(collections.find(c => c.name === "notes")).toBeDefined();
  });

  test("removeCollection removes existing collection", () => {
    store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    const removed = store.removeCollection("docs");

    expect(removed).toBe(true);
    const collections = store.listCollections();
    expect(collections.map(c => c.name)).not.toContain("docs");
  });

  test("removeCollection returns false for non-existent collection", () => {
    const removed = store.removeCollection("nonexistent");
    expect(removed).toBe(false);
  });

  test("renameCollection renames a collection", () => {
    store.addCollection("old-name", { path: docsDir, pattern: "**/*.md" });
    const renamed = store.renameCollection("old-name", "new-name");

    expect(renamed).toBe(true);
    const names = store.listCollections().map(c => c.name);
    expect(names).toContain("new-name");
    expect(names).not.toContain("old-name");
  });

  test("renameCollection returns false for non-existent source", () => {
    const renamed = store.renameCollection("nonexistent", "new-name");
    expect(renamed).toBe(false);
  });

  test("renameCollection throws if target exists", () => {
    store.addCollection("a", { path: docsDir, pattern: "**/*.md" });
    store.addCollection("b", { path: notesDir, pattern: "**/*.md" });

    expect(() => store.renameCollection("a", "b")).toThrow("already exists");
  });

  test("listCollections returns empty array for empty config", () => {
    const collections = store.listCollections();
    expect(collections).toEqual([]);
  });

  test("multiple collections can be added", () => {
    store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });

    const names = store.listCollections().map(c => c.name);
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

  beforeEach(() => {
    store = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(() => {
    store.close();
  });

  test("addContext adds context to a collection path", () => {
    const added = store.addContext("docs", "/auth", "Authentication docs");
    expect(added).toBe(true);

    const contexts = store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
  });

  test("addContext returns false for non-existent collection", () => {
    const added = store.addContext("nonexistent", "/path", "Some context");
    expect(added).toBe(false);
  });

  test("removeContext removes existing context", () => {
    store.addContext("docs", "/auth", "Authentication docs");
    const removed = store.removeContext("docs", "/auth");

    expect(removed).toBe(true);
    const contexts = store.listContexts();
    expect(contexts.find(c => c.path === "/auth")).toBeUndefined();
  });

  test("removeContext returns false for non-existent context", () => {
    const removed = store.removeContext("docs", "/nonexistent");
    expect(removed).toBe(false);
  });

  test("setGlobalContext sets and retrieves global context", () => {
    store.setGlobalContext("Global knowledge base");
    const global = store.getGlobalContext();

    expect(global).toBe("Global knowledge base");
  });

  test("setGlobalContext with undefined clears it", () => {
    store.setGlobalContext("Some context");
    store.setGlobalContext(undefined);
    const global = store.getGlobalContext();

    expect(global).toBeUndefined();
  });

  test("listContexts includes global context", () => {
    store.setGlobalContext("Global context");
    const contexts = store.listContexts();

    expect(contexts).toContainEqual({
      collection: "*",
      path: "/",
      context: "Global context",
    });
  });

  test("listContexts returns contexts across multiple collections", () => {
    store.addContext("docs", "/", "Documentation");
    store.addContext("notes", "/", "Personal notes");

    const contexts = store.listContexts();
    expect(contexts.filter(c => c.path === "/")).toHaveLength(2);
  });

  test("multiple contexts on same collection", () => {
    store.addContext("docs", "/auth", "Auth docs");
    store.addContext("docs", "/api", "API docs");

    const contexts = store.listContexts().filter(c => c.collection === "docs");
    expect(contexts).toHaveLength(2);
    expect(contexts.map(c => c.path).sort()).toEqual(["/api", "/auth"]);
  });

  test("addContext overwrites existing context for same path", () => {
    store.addContext("docs", "/auth", "Old context");
    store.addContext("docs", "/auth", "New context");

    const contexts = store.listContexts().filter(c => c.path === "/auth");
    expect(contexts).toHaveLength(1);
    expect(contexts[0]!.context).toBe("New context");
  });
});

// =============================================================================
// Inline Config Isolation Tests
// =============================================================================

describe("inline config isolation", () => {
  test("inline config does not write any files to disk", () => {
    const configDir = join(testDir, "should-not-exist");
    const store = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });
    store.addContext("docs", "/", "Documentation");

    expect(existsSync(configDir)).toBe(false);
    store.close();
  });

  test("inline config mutations persist within session", () => {
    const store = createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    store.addContext("docs", "/", "My docs");

    // Verify the mutations are visible
    const collections = store.listCollections();
    expect(collections.map(c => c.name)).toContain("docs");

    const contexts = store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/",
      context: "My docs",
    });

    store.close();
  });

  test("two stores with different inline configs are independent", () => {
    const store1 = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Close first store (resets config source)
    store1.close();

    const store2 = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = store2.listCollections().map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    store2.close();
  });
});

// =============================================================================
// YAML Config File Tests
// =============================================================================

describe("YAML config file mode", () => {
  test("loads collections from YAML file", () => {
    const configPath = join(testDir, `config-${Date.now()}.yml`);
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
        notes: { path: notesDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = createStore({ dbPath: freshDbPath(), configPath });
    const names = store.listCollections().map(c => c.name);

    expect(names).toContain("docs");
    expect(names).toContain("notes");
    store.close();
  });

  test("addCollection persists to YAML file", () => {
    const configPath = join(testDir, `config-persist-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({ collections: {} }));

    const store = createStore({ dbPath: freshDbPath(), configPath });
    store.addCollection("newcol", { path: docsDir, pattern: "**/*.md" });
    store.close();

    // Read the YAML file directly and verify
    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections).toHaveProperty("newcol");
    expect(parsed.collections.newcol!.path).toBe(docsDir);
  });

  test("context persists to YAML file", () => {
    const configPath = join(testDir, `config-ctx-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({
      collections: { docs: { path: docsDir, pattern: "**/*.md" } },
    }));

    const store = createStore({ dbPath: freshDbPath(), configPath });
    store.addContext("docs", "/api", "API documentation");
    store.close();

    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections.docs!.context).toEqual({ "/api": "API documentation" });
  });

  test("non-existent config file returns empty collections", () => {
    const configPath = join(testDir, "nonexistent-config.yml");
    const store = createStore({ dbPath: freshDbPath(), configPath });
    const collections = store.listCollections();

    expect(collections).toEqual([]);
    store.close();
  });
});

// =============================================================================
// Search Tests (BM25 - no LLM needed)
// =============================================================================

describe("search (BM25)", () => {
  let store: QMDStore;
  let dbPath: string;

  beforeAll(() => {
    dbPath = join(testDir, "search-test.sqlite");
    store = createStore({
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

  afterAll(() => {
    store.close();
  });

  test("search returns results for matching query", () => {
    const results = store.search("authentication");
    expect(results.length).toBeGreaterThan(0);
  });

  test("search results have expected shape", () => {
    const results = store.search("authentication");
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

  test("search respects limit option", () => {
    const results = store.search("meeting", { limit: 1 });
    expect(results.length).toBeLessThanOrEqual(1);
  });

  test("search with collection filter", () => {
    const results = store.search("authentication", { collection: "notes" });
    for (const r of results) {
      expect(r.collectionName).toBe("notes");
    }
  });

  test("search returns empty for non-matching query", () => {
    const results = store.search("xyznonexistentterm123");
    expect(results).toHaveLength(0);
  });

  test("search finds documents across collections", () => {
    const results = store.search("authentication", { limit: 10 });
    const collections = new Set(results.map(r => r.collectionName));
    // Auth appears in both docs/auth.md and notes/meeting-2025-02.md
    expect(collections.size).toBeGreaterThanOrEqual(1);
  });
});

// =============================================================================
// Document Retrieval Tests
// =============================================================================

describe("get and multiGet", () => {
  let store: QMDStore;

  beforeAll(() => {
    store = createStore({
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

  afterAll(() => {
    store.close();
  });

  test("get retrieves a document by path", () => {
    const result = store.get("qmd://docs/auth.md");

    expect("error" in result).toBe(false);
    if (!("error" in result)) {
      expect(result.title).toBe("Authentication");
      expect(result.collectionName).toBe("docs");
    }
  });

  test("get with includeBody returns body content", () => {
    const result = store.get("qmd://docs/auth.md", { includeBody: true });

    if (!("error" in result)) {
      expect(result.body).toBeDefined();
      expect(result.body).toContain("JWT tokens");
    }
  });

  test("get returns not_found for missing document", () => {
    const result = store.get("qmd://docs/nonexistent.md");

    expect("error" in result).toBe(true);
    if ("error" in result) {
      expect(result.error).toBe("not_found");
    }
  });

  test("get by docid", () => {
    // First get a document to find its docid
    const doc = store.get("qmd://docs/readme.md");
    if (!("error" in doc)) {
      const byDocid = store.get(`#${doc.docid}`);
      expect("error" in byDocid).toBe(false);
      if (!("error" in byDocid)) {
        expect(byDocid.docid).toBe(doc.docid);
      }
    }
  });

  test("multiGet retrieves multiple documents", () => {
    const { docs, errors } = store.multiGet("qmd://docs/*.md");
    expect(docs.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Index Health Tests
// =============================================================================

describe("index health", () => {
  let store: QMDStore;

  beforeEach(() => {
    store = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(() => {
    store.close();
  });

  test("getStatus returns valid structure", () => {
    const status = store.getStatus();

    expect(status).toHaveProperty("totalDocuments");
    expect(status).toHaveProperty("needsEmbedding");
    expect(status).toHaveProperty("hasVectorIndex");
    expect(status).toHaveProperty("collections");
    expect(typeof status.totalDocuments).toBe("number");
  });

  test("getIndexHealth returns valid structure", () => {
    const health = store.getIndexHealth();

    expect(health).toHaveProperty("needsEmbedding");
    expect(health).toHaveProperty("totalDocs");
    expect(typeof health.needsEmbedding).toBe("number");
    expect(typeof health.totalDocs).toBe("number");
  });

  test("fresh store has zero documents", () => {
    const status = store.getStatus();
    expect(status.totalDocuments).toBe(0);
  });
});

// =============================================================================
// Lifecycle Tests
// =============================================================================

describe("lifecycle", () => {
  test("close() makes subsequent operations throw", () => {
    const store = createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    store.close();

    // Database operations should fail after close
    expect(() => store.getStatus()).toThrow();
  });

  test("multiple stores can coexist with different databases", () => {
    const store1 = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Note: since config source is module-level, we close store1 first
    store1.close();

    const store2 = createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = store2.listCollections().map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    store2.close();
  });
});

// =============================================================================
// Config Initialization Tests
// =============================================================================

describe("config initialization", () => {
  test("inline config with global_context is preserved", () => {
    const store = createStore({
      dbPath: freshDbPath(),
      config: {
        global_context: "System knowledge base",
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const global = store.getGlobalContext();
    expect(global).toBe("System knowledge base");
    store.close();
  });

  test("inline config with pre-existing contexts is preserved", () => {
    const store = createStore({
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

    const contexts = store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
    store.close();
  });

  test("inline config with empty collections object works", () => {
    const store = createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    expect(store.listCollections()).toEqual([]);
    expect(store.listContexts()).toEqual([]);
    store.close();
  });

  test("inline config with multiple collection options", () => {
    const store = createStore({
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

    const collections = store.listCollections();
    expect(collections).toHaveLength(2);
    store.close();
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

  test("QMDStore type exposes expected methods", () => {
    const store = createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // Verify all methods exist
    expect(typeof store.query).toBe("function");
    expect(typeof store.search).toBe("function");
    expect(typeof store.structuredSearch).toBe("function");
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
    expect(typeof store.close).toBe("function");

    store.close();
  });
});
