/**
 * wiki-bench v0 — BM25/FTS regression suite.
 *
 * Indexes the vendored wiki corpus under test/wiki-bench-docs/ and scores
 * exact + alias queries from src/bench/fixtures/wiki-v0.json using searchFTS
 * + scoreResults. Labels come from registry/frontmatter only (no LLM judge).
 *
 * Measured BM25 baseline (wiki-bench-v0, lex/FTS only):
 *   exact mean recall@3:  1.0000 (n=20)
 *   alias mean recall@3:  0.9000 (n=10)
 *   exact+alias mean MRR: 0.8811 (n=30)
 * Floors kept at 0.85 / 0.80 / 0.70 (no relaxation needed).
 * Known BM25 weak spot: alias-shopify (entity drowned by source mentions).
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import {
  mkdtempSync,
  rmSync,
  readFileSync,
  readdirSync,
  statSync,
} from "fs";
import { join, dirname, relative } from "path";
import { tmpdir } from "os";
import { createHash } from "crypto";
import { fileURLToPath } from "url";
import type { Database } from "../src/db.js";
import {
  createStore,
  searchFTS,
  insertDocument,
  insertContent,
} from "../src/store";
import { scoreResults } from "../src/bench/score.js";
import type { BenchmarkFixture, BenchmarkQuery } from "../src/bench/types.js";

const tempDir = mkdtempSync(join(tmpdir(), "qmd-wiki-bench-"));
process.env.INDEX_PATH = join(tempDir, "wiki-bench.sqlite");

afterAll(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

const __dirname = dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = join(__dirname, "wiki-bench-docs");
const FIXTURE_PATH = join(__dirname, "..", "src", "bench", "fixtures", "wiki-v0.json");
const COLLECTION = "wiki-bench";

/**
 * Floors from measured BM25 baseline (exact R@3=1.00, alias R@3=0.90, MRR=0.88).
 * Kept high enough to catch regressions; not relaxed.
 */
const FLOOR_EXACT_RECALL_AT_3 = 0.85;
const FLOOR_ALIAS_RECALL_AT_3 = 0.80;
const FLOOR_COMBINED_MRR = 0.70;

function walkMarkdownFiles(dir: string, base: string = dir): string[] {
  const out: string[] = [];
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, ent.name);
    if (ent.isDirectory()) {
      out.push(...walkMarkdownFiles(full, base));
    } else if (ent.isFile() && ent.name.endsWith(".md")) {
      out.push(relative(base, full));
    }
  }
  return out.sort();
}

function extractTitle(content: string, fallback: string): string {
  const fm = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (fm) {
    const m = fm[1]!.match(/^title:\s*(.+)$/m);
    if (m) {
      const raw = m[1]!.trim();
      // Strip matching quotes
      const quoted = raw.match(/^["'](.*)["']$/);
      if (quoted) return quoted[1]!.trim() || fallback;
      return raw || fallback;
    }
  }
  const heading = content.match(/^#\s+(.+)$/m);
  if (heading?.[1]) return heading[1]!.trim();
  return fallback;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

type Scored = {
  id: string;
  type: string;
  recall_at_3: number;
  mrr: number;
  top_files: string[];
};

function scoreQuery(db: Database, q: BenchmarkQuery): Scored {
  const results = searchFTS(db, q.query, 10, COLLECTION);
  const files = results.map(r => r.filepath);
  const scores = scoreResults(files, q.expected_files, Math.max(q.expected_in_top_k, 3));
  return {
    id: q.id,
    type: q.type,
    recall_at_3: scores.recall_at_3,
    mrr: scores.mrr,
    top_files: files.slice(0, 3),
  };
}

describe("wiki-bench BM25 (FTS)", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;
  let fixture: BenchmarkFixture;
  let exactQueries: BenchmarkQuery[];
  let aliasQueries: BenchmarkQuery[];
  let exactScores: Scored[];
  let aliasScores: Scored[];

  beforeAll(() => {
    store = createStore();
    db = store.db;

    const files = walkMarkdownFiles(DOCS_DIR);
    expect(files.length).toBe(34);

    for (const rel of files) {
      const full = join(DOCS_DIR, rel);
      expect(statSync(full).size).toBeGreaterThan(0);
      const content = readFileSync(full, "utf-8");
      const title = extractTitle(content, rel);
      const hash = createHash("sha256").update(content).digest("hex").slice(0, 12);
      const now = new Date().toISOString();
      insertContent(db, hash, content, now);
      // filepath = relative path (e.g. concepts/trust-battery.md)
      insertDocument(db, COLLECTION, rel, title, hash, now, now);
    }

    fixture = JSON.parse(readFileSync(FIXTURE_PATH, "utf-8")) as BenchmarkFixture;
    exactQueries = fixture.queries.filter(q => q.type === "exact");
    aliasQueries = fixture.queries.filter(q => q.type === "alias");
    expect(exactQueries.length).toBeGreaterThan(0);
    expect(aliasQueries.length).toBeGreaterThan(0);

    exactScores = exactQueries.map(q => scoreQuery(db, q));
    aliasScores = aliasQueries.map(q => scoreQuery(db, q));

    const exactR3 = mean(exactScores.map(s => s.recall_at_3));
    const aliasR3 = mean(aliasScores.map(s => s.recall_at_3));
    const combinedMrr = mean([...exactScores, ...aliasScores].map(s => s.mrr));

    // Print measured baseline for PR / floor tuning
    console.log("\n[wiki-bench] measured BM25 metrics:");
    console.log(`  exact mean recall@3 = ${exactR3.toFixed(4)} (n=${exactScores.length})`);
    console.log(`  alias mean recall@3 = ${aliasR3.toFixed(4)} (n=${aliasScores.length})`);
    console.log(`  exact+alias mean MRR = ${combinedMrr.toFixed(4)}`);
    for (const s of [...exactScores, ...aliasScores]) {
      if (s.recall_at_3 < 1 || s.mrr < 1) {
        console.log(`  miss/partial ${s.id}: r@3=${s.recall_at_3.toFixed(2)} mrr=${s.mrr.toFixed(2)} top=${JSON.stringify(s.top_files)}`);
      }
    }
  });

  afterAll(() => {
    store.close();
  });

  test("indexes 34 non-empty wiki pages", () => {
    const count = (db.prepare(
      `SELECT COUNT(*) as n FROM documents WHERE collection = ? AND active = 1`
    ).get(COLLECTION) as { n: number }).n;
    expect(count).toBe(34);
  });

  test("exact: mean recall_at_3 meets floor", () => {
    const exactR3 = mean(exactScores.map(s => s.recall_at_3));
    expect(exactR3).toBeGreaterThanOrEqual(FLOOR_EXACT_RECALL_AT_3);
  });

  test("alias: mean recall_at_3 meets floor", () => {
    const aliasR3 = mean(aliasScores.map(s => s.recall_at_3));
    expect(aliasR3).toBeGreaterThanOrEqual(FLOOR_ALIAS_RECALL_AT_3);
  });

  test("exact+alias: mean MRR meets floor", () => {
    const combinedMrr = mean([...exactScores, ...aliasScores].map(s => s.mrr));
    expect(combinedMrr).toBeGreaterThanOrEqual(FLOOR_COMBINED_MRR);
  });
});
