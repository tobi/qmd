/**
 * Evaluation Tests for QMD Search Quality
 *
 * Tests search quality against synthetic documents with known-answer queries.
 * Validates that search improvements don't regress quality.
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
  getDb,
  closeDb,
  searchFTS,
  insertDocument,
  insertContent,
} from "./store";

// Eval queries with expected documents
const evalQueries: {
  query: string;
  expectedDoc: string;
  difficulty: "easy" | "medium" | "hard";
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
];

describe("Search Quality Evaluation", () => {
  let db: Database;

  beforeAll(() => {
    // Initialize database (INDEX_PATH already set at top of file)
    db = getDb();

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
    closeDb();
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe("BM25 Search (FTS)", () => {
    test("easy queries: ≥80% Hit@3", () => {
      const easyQueries = evalQueries.filter(q => q.difficulty === "easy");
      let hits = 0;

      for (const { query, expectedDoc } of easyQueries) {
        const results = searchFTS(db, query, 5);
        const top3 = results.slice(0, 3);
        const found = top3.some(r => r.filepath.toLowerCase().includes(expectedDoc));
        if (found) hits++;
      }

      const hitRate = hits / easyQueries.length;
      expect(hitRate).toBeGreaterThanOrEqual(0.8);
    });

    test("medium queries: ≥15% Hit@3 (BM25 struggles with semantic)", () => {
      const mediumQueries = evalQueries.filter(q => q.difficulty === "medium");
      let hits = 0;

      for (const { query, expectedDoc } of mediumQueries) {
        const results = searchFTS(db, query, 5);
        const top3 = results.slice(0, 3);
        const found = top3.some(r => r.filepath.toLowerCase().includes(expectedDoc));
        if (found) hits++;
      }

      const hitRate = hits / mediumQueries.length;
      // BM25 alone struggles with semantic queries - baseline is low
      expect(hitRate).toBeGreaterThanOrEqual(0.15);
    });

    test("hard queries: ≥15% Hit@5 (BM25 baseline)", () => {
      const hardQueries = evalQueries.filter(q => q.difficulty === "hard");
      let hits = 0;

      for (const { query, expectedDoc } of hardQueries) {
        const results = searchFTS(db, query, 5);
        const found = results.some(r => r.filepath.toLowerCase().includes(expectedDoc));
        if (found) hits++;
      }

      const hitRate = hits / hardQueries.length;
      // BM25 alone really struggles with vague queries
      expect(hitRate).toBeGreaterThanOrEqual(0.15);
    });
  });

  describe("Overall Quality", () => {
    test("overall Hit@3 ≥40% (BM25 baseline)", () => {
      let hits = 0;

      for (const { query, expectedDoc } of evalQueries) {
        const results = searchFTS(db, query, 5);
        const top3 = results.slice(0, 3);
        const found = top3.some(r => r.filepath.toLowerCase().includes(expectedDoc));
        if (found) hits++;
      }

      const hitRate = hits / evalQueries.length;
      // BM25 alone: ~40% is baseline, hybrid should be higher
      expect(hitRate).toBeGreaterThanOrEqual(0.4);
    });
  });
});
