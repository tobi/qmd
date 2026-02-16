/**
 * BM25-only evaluation tests (unit layer).
 *
 * This is a fast suite copied from the BM25 block in `models/eval.test.ts`.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import type { Database } from "../src/db.js";
import { createHash } from "crypto";
import { fileURLToPath } from "url";

import {
  createStore,
  searchFTS,
  insertDocument,
  insertContent,
} from "../src/store";

// Set INDEX_PATH before importing store to prevent using global index
const tempDir = mkdtempSync(join(tmpdir(), "qmd-eval-unit-"));
process.env.INDEX_PATH = join(tempDir, "eval-unit.sqlite");

afterAll(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

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

function matchesExpected(filepath: string, expectedDoc: string): boolean {
  return filepath.toLowerCase().includes(expectedDoc);
}

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

describe("BM25 Search (FTS)", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;

  beforeAll(() => {
    store = createStore();
    db = store.db;

    // Load and index eval documents
    const evalDocsDir = join(dirname(fileURLToPath(import.meta.url)), "eval-docs");
    const files = readdirSync(evalDocsDir).filter(f => f.endsWith(".md"));

    for (const file of files) {
      const content = readFileSync(join(evalDocsDir, file), "utf-8");
      const title = content.split("\n")[0]?.replace(/^#\s*/, "") || file;
      const hash = createHash("sha256").update(content).digest("hex").slice(0, 12);
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
