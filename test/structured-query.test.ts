/**
 * Structured Query Tests
 *
 * Unit tests: normalizeQuery, hasCallerExpansions, type contracts.
 * Integration tests: hybridQuery routing — verifies structured queries skip
 * LLM expansion and route caller fields to the right search backends.
 *
 * Integration tests use a real SQLite store with FTS (no vector index) to
 * avoid LLM/embedding dependencies while testing the routing logic.
 */

import { describe, test, expect, vi, beforeAll, afterAll } from "vitest";
import { mkdtemp, writeFile, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import {
  createStore,
  normalizeQuery,
  hasCallerExpansions,
  hashContent,
  hybridQuery,
  type Store,
  type StructuredQuery,
} from "../src/store";
import type { CollectionConfig } from "../src/collections";

// =============================================================================
// Test helpers — lightweight store setup for integration tests
// =============================================================================

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-structured-test-"));
});

afterAll(async () => {
  try {
    const { rm } = await import("node:fs/promises");
    await rm(testDir, { recursive: true, force: true });
  } catch { /* ignore */ }
});

async function createTestStore(): Promise<Store> {
  const dbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;

  const config: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(config));

  return createStore(dbPath);
}

async function addCollection(store: Store, name: string): Promise<string> {
  const configPath = join(process.env.QMD_CONFIG_DIR!, "index.yml");
  const { readFile } = await import("node:fs/promises");
  const config = YAML.parse(await readFile(configPath, "utf-8")) as CollectionConfig;
  config.collections[name] = { path: `/test/${name}`, pattern: "**/*.md" };
  await writeFile(configPath, YAML.stringify(config));
  return name;
}

async function addDoc(
  store: Store, collection: string,
  title: string, body: string, path?: string,
): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(body);
  const docPath = path || `${title.toLowerCase().replace(/\s+/g, "-")}.md`;

  store.db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, body, now);
  store.db.prepare(`INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, 1)`)
    .run(collection, docPath, title, hash, now, now);
}

async function cleanup(store: Store): Promise<void> {
  store.close();
  try { await unlink(store.dbPath); } catch { /* ignore */ }
  delete process.env.QMD_CONFIG_DIR;
}

// =============================================================================
// normalizeQuery
// =============================================================================

describe("normalizeQuery", () => {
  test("string becomes { text } object", () => {
    expect(normalizeQuery("performance")).toEqual({ text: "performance" });
  });

  test("structured query with populated fields passes through", () => {
    const sq: StructuredQuery = {
      text: "performance",
      keywords: ["TTFB", "core web vitals"],
      concepts: ["frontend rendering optimization"],
      passage: "Reducing time-to-first-byte requires optimizing the critical rendering path.",
    };
    const result = normalizeQuery(sq);
    expect(result).toEqual(sq);
  });

  test("minimal structured query (text only) passes through", () => {
    expect(normalizeQuery({ text: "performance" })).toEqual({ text: "performance" });
  });

  test("strips empty keywords array", () => {
    const result = normalizeQuery({ text: "test", keywords: [] });
    expect(result).toEqual({ text: "test" });
    expect(result.keywords).toBeUndefined();
  });

  test("strips empty concepts array", () => {
    const result = normalizeQuery({ text: "test", concepts: [] });
    expect(result).toEqual({ text: "test" });
    expect(result.concepts).toBeUndefined();
  });

  test("strips empty passage string", () => {
    const result = normalizeQuery({ text: "test", passage: "" });
    expect(result).toEqual({ text: "test" });
    expect(result.passage).toBeUndefined();
  });

  test("keeps non-empty fields, strips empty ones", () => {
    const result = normalizeQuery({
      text: "test",
      keywords: ["TTFB"],
      concepts: [],
      passage: "",
    });
    expect(result).toEqual({ text: "test", keywords: ["TTFB"] });
  });
});

// =============================================================================
// hasCallerExpansions
// =============================================================================

describe("hasCallerExpansions", () => {
  test("returns false for string-normalized query (text only)", () => {
    expect(hasCallerExpansions({ text: "performance" })).toBe(false);
  });

  test("returns false for empty arrays", () => {
    expect(hasCallerExpansions({ text: "performance", keywords: [], concepts: [] })).toBe(false);
  });

  test("returns false for undefined fields", () => {
    expect(hasCallerExpansions({ text: "performance", keywords: undefined, concepts: undefined, passage: undefined })).toBe(false);
  });

  test("returns true when keywords present", () => {
    expect(hasCallerExpansions({ text: "performance", keywords: ["TTFB"] })).toBe(true);
  });

  test("returns true when concepts present", () => {
    expect(hasCallerExpansions({ text: "performance", concepts: ["frontend rendering"] })).toBe(true);
  });

  test("returns true when passage present", () => {
    expect(hasCallerExpansions({
      text: "performance",
      passage: "Reducing TTFB requires edge caching.",
    })).toBe(true);
  });

  test("returns true when all fields present", () => {
    expect(hasCallerExpansions({
      text: "performance",
      keywords: ["TTFB"],
      concepts: ["rendering"],
      passage: "A passage about performance.",
    })).toBe(true);
  });

  test("returns false for empty passage string", () => {
    expect(hasCallerExpansions({ text: "performance", passage: "" })).toBe(false);
  });
});

// =============================================================================
// StructuredQuery type shape
// =============================================================================

describe("StructuredQuery type contracts", () => {
  test("text is the only required field", () => {
    const sq: StructuredQuery = { text: "test" };
    expect(normalizeQuery(sq).text).toBe("test");
  });

  test("keywords is string array", () => {
    const sq: StructuredQuery = { text: "test", keywords: ["a", "b", "c"] };
    expect(sq.keywords).toHaveLength(3);
  });

  test("concepts is string array", () => {
    const sq: StructuredQuery = { text: "test", concepts: ["semantic phrase one", "semantic phrase two"] };
    expect(sq.concepts).toHaveLength(2);
  });

  test("passage is a single string", () => {
    const sq: StructuredQuery = {
      text: "test",
      passage: "A hypothetical document passage about the topic at hand.",
    };
    expect(typeof sq.passage).toBe("string");
  });
});

// =============================================================================
// hybridQuery routing integration tests
//
// Uses real SQLite FTS (no vector index) to verify routing without needing
// LLM models for embedding. expandQuery is spied/mocked to verify call behavior.
// =============================================================================

describe("hybridQuery routing", () => {
  test("structured query with keywords skips LLM expansion", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Web Vitals", "Core web vitals measure TTFB and LCP for page load performance");
    await addDoc(store, coll, "Team Health", "Team health is about trust and psychological safety");

    const expandSpy = vi.spyOn(store, "expandQuery");

    const results = await hybridQuery(store, {
      text: "performance",
      keywords: ["TTFB", "core web vitals"],
    }, { limit: 5 });

    // expandQuery must NOT be called — caller provided their own expansion
    expect(expandSpy).not.toHaveBeenCalled();
    // FTS on "performance" (initial probe) + "TTFB" + "core web vitals" should
    // find the web vitals doc
    expect(results.length).toBeGreaterThan(0);
    expect(results.some(r => r.title === "Web Vitals")).toBe(true);

    expandSpy.mockRestore();
    await cleanup(store);
  });

  test("plain string query calls LLM expansion", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Fox Doc", "The quick brown fox jumps over the lazy dog");

    // Mock expandQuery to avoid needing the actual LLM
    const expandSpy = vi.spyOn(store, "expandQuery").mockResolvedValue([
      { type: "lex", text: "quick fox" },
    ]);

    await hybridQuery(store, "fox", { limit: 5 });

    // expandQuery MUST be called — no caller expansions, no strong signal
    expect(expandSpy).toHaveBeenCalledWith("fox", undefined);

    expandSpy.mockRestore();
    await cleanup(store);
  });

  test("structured query with only passage skips expansion", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Scaling", "Database sharding enables horizontal scaling");

    const expandSpy = vi.spyOn(store, "expandQuery");

    // passage alone should trigger caller-expansion path
    const results = await hybridQuery(store, {
      text: "scaling",
      passage: "Horizontal scaling through database sharding and read replicas",
    }, { limit: 5 });

    expect(expandSpy).not.toHaveBeenCalled();
    // Without vector index, only FTS on "scaling" runs (passage needs embeddings)
    // But the routing decision is what we're testing — expandQuery was skipped
    expect(results.length).toBeGreaterThan(0);

    expandSpy.mockRestore();
    await cleanup(store);
  });

  test("empty expansion fields fall through to LLM expansion", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Test Doc", "Some content for testing");

    const expandSpy = vi.spyOn(store, "expandQuery").mockResolvedValue([]);

    // Empty arrays + empty passage = no caller expansions after normalization
    await hybridQuery(store, {
      text: "testing",
      keywords: [],
      concepts: [],
      passage: "",
    }, { limit: 5 });

    // normalizeQuery strips empties, so this falls through to LLM path
    expect(expandSpy).toHaveBeenCalled();

    expandSpy.mockRestore();
    await cleanup(store);
  });

  test("onExpand hook does not fire for structured queries", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Doc", "Content about performance metrics");

    const onExpand = vi.fn();
    vi.spyOn(store, "expandQuery");

    await hybridQuery(store, {
      text: "performance",
      keywords: ["metrics"],
    }, { limit: 5, hooks: { onExpand } });

    expect(onExpand).not.toHaveBeenCalled();

    await cleanup(store);
  });

  test("onExpand hook fires for string queries with LLM expansion", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Doc", "Content about performance metrics");

    const onExpand = vi.fn();
    vi.spyOn(store, "expandQuery").mockResolvedValue([
      { type: "lex", text: "metrics benchmarks" },
      { type: "vec", text: "performance measurement" },
    ]);

    await hybridQuery(store, "performance", { limit: 5, hooks: { onExpand } });

    expect(onExpand).toHaveBeenCalledWith("performance", [
      { type: "lex", text: "metrics benchmarks" },
      { type: "vec", text: "performance measurement" },
    ]);

    await cleanup(store);
  });

  test("keywords route to FTS and influence results", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    // Two docs — only one matches the keyword expansion
    await addDoc(store, coll, "TTFB Guide", "Time to first byte optimization reduces latency");
    await addDoc(store, coll, "Team Trust", "Building trust in engineering teams requires candor");

    vi.spyOn(store, "expandQuery");

    const results = await hybridQuery(store, {
      text: "performance",
      keywords: ["latency", "time to first byte"],
    }, { limit: 5 });

    // The keyword "latency" should boost the TTFB doc
    const titles = results.map(r => r.title);
    expect(titles).toContain("TTFB Guide");

    await cleanup(store);
  });

  test("intent disables strong-signal bypass for structured queries", async () => {
    const store = await createTestStore();
    const coll = await addCollection(store, "routing");

    await addDoc(store, coll, "Exact Match", "A very specific unique term zephyr appears here zephyr zephyr");

    const expandSpy = vi.spyOn(store, "expandQuery");

    // Structured query with intent — strong signal should be disabled
    // (callerExpansions = true already disables it, but test the combination)
    await hybridQuery(store, {
      text: "zephyr",
      keywords: ["wind patterns"],
    }, { limit: 5, intent: "meteorology" });

    expect(expandSpy).not.toHaveBeenCalled();

    expandSpy.mockRestore();
    await cleanup(store);
  });
});
