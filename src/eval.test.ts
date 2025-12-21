/**
 * Evaluation Tests for QMD Search Quality
 *
 * Tests search quality against synthetic documents with known-answer queries.
 * Validates that search improvements don't regress quality.
 *
 * Three test suites:
 * 1. BM25 (FTS) - lexical search baseline
 * 2. Vector Search - semantic search with embeddings
 * 3. Hybrid (RRF) - combined lexical + vector with rank fusion
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, rmSync, readFileSync, readdirSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import Database from "bun:sqlite";

// Set INDEX_PATH before importing store to prevent using global index
const tempDir = mkdtempSync(join(tmpdir(), "qmd-eval-"));
process.env.INDEX_PATH = join(tempDir, "eval.sqlite");

import {
  createStore,
  searchFTS,
  searchVec,
  insertDocument,
  insertContent,
  insertEmbedding,
  chunkDocumentByTokens,
  reciprocalRankFusion,
  DEFAULT_EMBED_MODEL,
  type RankedResult,
} from "./store";
import { getDefaultLlamaCpp, formatDocForEmbedding, disposeDefaultLlamaCpp } from "./llm";

// Eval queries with expected documents
const evalQueries: {
  query: string;
  expectedDoc: string;
  difficulty: "easy" | "medium" | "hard" | "fusion";
}[] = [
  // EASY: Exact keyword matches
  { query: "API versioning", expectedDoc: "api-design", difficulty: "easy" },
  { query: "Series A fundraising", expectedDoc: "fundraising", difficulty: "easy" },
  { query: "CAP theorem", expectedDoc: "distributed-systems", difficulty: "easy" },
  { query: "overfitting machine learning", expectedDoc: "machine-learning", difficulty: "easy" },
  { query: "remote work VPN", expectedDoc: "remote-work", difficulty: "easy" },
  { query: "Project Phoenix retrospective", expectedDoc: "product-launch", difficulty: "easy" },

  // MEDIUM: Semantic/conceptual queries
  { query: "how to structure REST endpoints", expectedDoc: "api-design", difficulty: "medium" },
  { query: "raising money for startup", expectedDoc: "fundraising", difficulty: "medium" },
  { query: "consistency vs availability tradeoffs", expectedDoc: "distributed-systems", difficulty: "medium" },
  { query: "how to prevent models from memorizing data", expectedDoc: "machine-learning", difficulty: "medium" },
  { query: "working from home guidelines", expectedDoc: "remote-work", difficulty: "medium" },
  { query: "what went wrong with the launch", expectedDoc: "product-launch", difficulty: "medium" },

  // HARD: Vague, partial memory, indirect
  { query: "nouns not verbs", expectedDoc: "api-design", difficulty: "hard" },
  { query: "Sequoia investor pitch", expectedDoc: "fundraising", difficulty: "hard" },
  { query: "Raft algorithm leader election", expectedDoc: "distributed-systems", difficulty: "hard" },
  { query: "F1 score precision recall", expectedDoc: "machine-learning", difficulty: "hard" },
  { query: "quarterly team gathering travel", expectedDoc: "remote-work", difficulty: "hard" },
  { query: "beta program 47 bugs", expectedDoc: "product-launch", difficulty: "hard" },

  // FUSION: Multi-signal queries that need both lexical AND semantic matching
  // These should have weak individual scores but strong combined RRF scores
  { query: "how much runway before running out of money", expectedDoc: "fundraising", difficulty: "fusion" },
  { query: "datacenter replication sync strategy", expectedDoc: "distributed-systems", difficulty: "fusion" },
  { query: "splitting data for training and testing", expectedDoc: "machine-learning", difficulty: "fusion" },
  { query: "JSON response codes error messages", expectedDoc: "api-design", difficulty: "fusion" },
  { query: "video calls camera async messaging", expectedDoc: "remote-work", difficulty: "fusion" },
  { query: "CI/CD pipeline testing coverage", expectedDoc: "product-launch", difficulty: "fusion" },
];

// Helper to check if result matches expected doc
function matchesExpected(filepath: string, expectedDoc: string): boolean {
  return filepath.toLowerCase().includes(expectedDoc);
}

// Helper to calculate hit rate
function calcHitRate(
  queries: typeof evalQueries,
  searchFn: (query: string) => { filepath: string }[],
  topK: number
): number {
  let hits = 0;
  for (const { query, expectedDoc } of queries) {
    const results = searchFn(query).slice(0, topK);
    if (results.some(r => matchesExpected(r.filepath, expectedDoc))) hits++;
  }
  return hits / queries.length;
}

// =============================================================================
// BM25 (Lexical) Tests - Fast, no model loading needed
// =============================================================================

describe("BM25 Search (FTS)", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;

  beforeAll(() => {
    store = createStore();
    db = store.db;

    // Load and index eval documents
    const evalDocsDir = join(import.meta.dir, "../test/eval-docs");
    const files = readdirSync(evalDocsDir).filter(f => f.endsWith(".md"));

    for (const file of files) {
      const content = readFileSync(join(evalDocsDir, file), "utf-8");
      const title = content.split("\n")[0]?.replace(/^#\s*/, "") || file;
      const hash = Bun.hash(content).toString(16).slice(0, 12);
      const now = new Date().toISOString();

      insertContent(db, hash, content, now);
      insertDocument(db, "eval-docs", file, title, hash, now, now);
    }
  });

  afterAll(() => {
    store.close();
  });

  test("easy queries: ≥80% Hit@3", () => {
    const easyQueries = evalQueries.filter(q => q.difficulty === "easy");
    const hitRate = calcHitRate(easyQueries, q => searchFTS(db, q, 5), 3);
    expect(hitRate).toBeGreaterThanOrEqual(0.8);
  });

  test("medium queries: ≥15% Hit@3 (BM25 struggles with semantic)", () => {
    const mediumQueries = evalQueries.filter(q => q.difficulty === "medium");
    const hitRate = calcHitRate(mediumQueries, q => searchFTS(db, q, 5), 3);
    expect(hitRate).toBeGreaterThanOrEqual(0.15);
  });

  test("hard queries: ≥15% Hit@5 (BM25 baseline)", () => {
    const hardQueries = evalQueries.filter(q => q.difficulty === "hard");
    const hitRate = calcHitRate(hardQueries, q => searchFTS(db, q, 5), 5);
    expect(hitRate).toBeGreaterThanOrEqual(0.15);
  });

  test("overall Hit@3 ≥40% (BM25 baseline)", () => {
    const hitRate = calcHitRate(evalQueries, q => searchFTS(db, q, 5), 3);
    expect(hitRate).toBeGreaterThanOrEqual(0.4);
  });
});

// =============================================================================
// Vector Search Tests - Requires embedding model
// =============================================================================

describe("Vector Search", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;
  let hasEmbeddings = false;

  beforeAll(async () => {
    store = createStore();
    db = store.db;

    // Check if embeddings already exist (from previous test run)
    const vecTable = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
    ).get();

    if (vecTable) {
      const count = db.prepare(`SELECT COUNT(*) as cnt FROM vectors_vec`).get() as { cnt: number };
      if (count.cnt > 0) {
        hasEmbeddings = true;
        return;
      }
    }

    // Generate embeddings for test documents
    const llm = getDefaultLlamaCpp();
    store.ensureVecTable(768); // embeddinggemma uses 768 dimensions

    const evalDocsDir = join(import.meta.dir, "../test/eval-docs");
    const files = readdirSync(evalDocsDir).filter(f => f.endsWith(".md"));

    for (const file of files) {
      const content = readFileSync(join(evalDocsDir, file), "utf-8");
      const hash = Bun.hash(content).toString(16).slice(0, 12);
      const title = content.split("\n")[0]?.replace(/^#\s*/, "") || file;

      // Chunk and embed
      const chunks = await chunkDocumentByTokens(content);
      for (let seq = 0; seq < chunks.length; seq++) {
        const chunk = chunks[seq];
        if (!chunk) continue;
        const formatted = formatDocForEmbedding(chunk.text, title);
        const result = await llm.embed(formatted, { model: DEFAULT_EMBED_MODEL, isQuery: false });
        if (result?.embedding) {
          // Convert to Float32Array for sqlite-vec
          const embedding = new Float32Array(result.embedding);
          const now = new Date().toISOString();
          insertEmbedding(db, hash, seq, chunk.pos, embedding, DEFAULT_EMBED_MODEL, now);
        }
      }
    }
    hasEmbeddings = true;
  }, 120000); // 2 minute timeout for embedding generation

  afterAll(() => {
    store.close();
  });

  // Note: Don't dispose here - Hybrid tests also use llama.
  // Dispose happens in the global afterAll.

  test("easy queries: ≥60% Hit@3 (vector should match keywords too)", async () => {
    if (!hasEmbeddings) return; // Skip if embedding failed

    const easyQueries = evalQueries.filter(q => q.difficulty === "easy");
    let hits = 0;
    for (const { query, expectedDoc } of easyQueries) {
      const results = await searchVec(db, query, DEFAULT_EMBED_MODEL, 5);
      if (results.slice(0, 3).some(r => matchesExpected(r.filepath, expectedDoc))) hits++;
    }
    expect(hits / easyQueries.length).toBeGreaterThanOrEqual(0.6);
  }, 60000);

  test("medium queries: ≥40% Hit@3 (vector excels at semantic)", async () => {
    if (!hasEmbeddings) return;

    const mediumQueries = evalQueries.filter(q => q.difficulty === "medium");
    let hits = 0;
    for (const { query, expectedDoc } of mediumQueries) {
      const results = await searchVec(db, query, DEFAULT_EMBED_MODEL, 5);
      if (results.slice(0, 3).some(r => matchesExpected(r.filepath, expectedDoc))) hits++;
    }
    // Vector search should do better on semantic queries than BM25
    expect(hits / mediumQueries.length).toBeGreaterThanOrEqual(0.4);
  }, 60000);

  test("hard queries: ≥30% Hit@5 (vector helps with vague queries)", async () => {
    if (!hasEmbeddings) return;

    const hardQueries = evalQueries.filter(q => q.difficulty === "hard");
    let hits = 0;
    for (const { query, expectedDoc } of hardQueries) {
      const results = await searchVec(db, query, DEFAULT_EMBED_MODEL, 5);
      if (results.some(r => matchesExpected(r.filepath, expectedDoc))) hits++;
    }
    expect(hits / hardQueries.length).toBeGreaterThanOrEqual(0.3);
  }, 60000);

  test("overall Hit@3 ≥50% (vector baseline)", async () => {
    if (!hasEmbeddings) return;

    let hits = 0;
    for (const { query, expectedDoc } of evalQueries) {
      const results = await searchVec(db, query, DEFAULT_EMBED_MODEL, 5);
      if (results.slice(0, 3).some(r => matchesExpected(r.filepath, expectedDoc))) hits++;
    }
    expect(hits / evalQueries.length).toBeGreaterThanOrEqual(0.5);
  }, 60000);
});

// =============================================================================
// Hybrid Search (RRF) Tests - Combines BM25 + Vector
// =============================================================================

describe("Hybrid Search (RRF)", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;
  let hasVectors = false;

  beforeAll(() => {
    store = createStore();
    db = store.db;
    // Check if vectors exist
    const vecTable = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
    ).get();
    if (vecTable) {
      const count = db.prepare(`SELECT COUNT(*) as cnt FROM vectors_vec`).get() as { cnt: number };
      hasVectors = count.cnt > 0;
    }
  });

  afterAll(() => {
    store.close();
  });

  // Helper: run hybrid search with RRF fusion
  async function hybridSearch(query: string, limit: number = 10): Promise<RankedResult[]> {
    const rankedLists: RankedResult[][] = [];

    // FTS results
    const ftsResults = searchFTS(db, query, 20);
    if (ftsResults.length > 0) {
      rankedLists.push(ftsResults.map(r => ({
        file: r.filepath,
        displayPath: r.displayPath,
        title: r.title,
        body: r.body || "",
        score: r.score
      })));
    }

    // Vector results
    const vecResults = await searchVec(db, query, DEFAULT_EMBED_MODEL, 20);
    if (vecResults.length > 0) {
      rankedLists.push(vecResults.map(r => ({
        file: r.filepath,
        displayPath: r.displayPath,
        title: r.title,
        body: r.body || "",
        score: r.score
      })));
    }

    if (rankedLists.length === 0) return [];

    // Apply RRF fusion
    const fused = reciprocalRankFusion(rankedLists);
    return fused.slice(0, limit);
  }

  test("easy queries: ≥80% Hit@3 (hybrid should match BM25)", async () => {
    const easyQueries = evalQueries.filter(q => q.difficulty === "easy");
    let hits = 0;
    for (const { query, expectedDoc } of easyQueries) {
      const results = await hybridSearch(query);
      if (results.slice(0, 3).some(r => matchesExpected(r.file, expectedDoc))) hits++;
    }
    expect(hits / easyQueries.length).toBeGreaterThanOrEqual(0.8);
  }, 60000);

  test("medium queries: ≥50% Hit@3 with vectors, ≥15% without", async () => {
    const mediumQueries = evalQueries.filter(q => q.difficulty === "medium");
    let hits = 0;
    for (const { query, expectedDoc } of mediumQueries) {
      const results = await hybridSearch(query);
      if (results.slice(0, 3).some(r => matchesExpected(r.file, expectedDoc))) hits++;
    }
    // With vectors: hybrid should outperform both BM25 (15%) and vector (40%)
    // Without vectors: hybrid is just BM25, so use BM25 threshold
    const threshold = hasVectors ? 0.5 : 0.15;
    expect(hits / mediumQueries.length).toBeGreaterThanOrEqual(threshold);
  }, 60000);

  test("hard queries: ≥35% Hit@5 with vectors, ≥15% without", async () => {
    const hardQueries = evalQueries.filter(q => q.difficulty === "hard");
    let hits = 0;
    for (const { query, expectedDoc } of hardQueries) {
      const results = await hybridSearch(query);
      if (results.some(r => matchesExpected(r.file, expectedDoc))) hits++;
    }
    const threshold = hasVectors ? 0.35 : 0.15;
    expect(hits / hardQueries.length).toBeGreaterThanOrEqual(threshold);
  }, 60000);

  test("fusion queries: ≥50% Hit@3 (RRF combines weak signals)", async () => {
    if (!hasVectors) return; // Fusion requires both methods

    const fusionQueries = evalQueries.filter(q => q.difficulty === "fusion");
    let hybridHits = 0;
    let bm25Hits = 0;
    let vecHits = 0;

    for (const { query, expectedDoc } of fusionQueries) {
      // Hybrid results
      const hybridResults = await hybridSearch(query);
      if (hybridResults.slice(0, 3).some(r => matchesExpected(r.file, expectedDoc))) hybridHits++;

      // BM25 results for comparison
      const bm25Results = searchFTS(db, query, 5);
      if (bm25Results.slice(0, 3).some(r => matchesExpected(r.filepath, expectedDoc))) bm25Hits++;

      // Vector results for comparison
      const vecResults = await searchVec(db, query, DEFAULT_EMBED_MODEL, 5);
      if (vecResults.slice(0, 3).some(r => matchesExpected(r.filepath, expectedDoc))) vecHits++;
    }

    const hybridRate = hybridHits / fusionQueries.length;
    const bm25Rate = bm25Hits / fusionQueries.length;
    const vecRate = vecHits / fusionQueries.length;

    // Fusion should achieve at least 50% on these multi-signal queries
    expect(hybridRate).toBeGreaterThanOrEqual(0.5);

    // Fusion should outperform or match the best individual method
    expect(hybridRate).toBeGreaterThanOrEqual(Math.max(bm25Rate, vecRate));
  }, 60000);

  test("overall Hit@3 ≥60% with vectors, ≥40% without", async () => {
    // Filter out fusion queries for overall score (they're tested separately)
    const standardQueries = evalQueries.filter(q => q.difficulty !== "fusion");
    let hits = 0;
    for (const { query, expectedDoc } of standardQueries) {
      const results = await hybridSearch(query);
      if (results.slice(0, 3).some(r => matchesExpected(r.file, expectedDoc))) hits++;
    }
    const threshold = hasVectors ? 0.6 : 0.4;
    expect(hits / standardQueries.length).toBeGreaterThanOrEqual(threshold);
  }, 60000);
});

// =============================================================================
// Cleanup
// =============================================================================

afterAll(async () => {
  // Ensure native resources are released to avoid ggml-metal asserts on process exit.
  await disposeDefaultLlamaCpp();
  rmSync(tempDir, { recursive: true, force: true });
});
