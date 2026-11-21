/**
 * store-cjk-bigram-fts.test.ts — Regression tests for CJK query bigram fallback (#976).
 *
 * CJK runs are indexed as single-character tokens (unicode61 has no CJK word
 * segmentation), so buildFTS5Query()'s exact-phrase wrapping means a CJK term
 * only matches a fully contiguous run. A query containing one multi-character
 * CJK run that does not appear contiguously in the corpus therefore returned
 * zero results even when the remaining terms matched — the all-or-nothing
 * behavior reported in #976.
 *
 * The fix: for CJK runs of 3+ characters, buildFTS5Query() now emits an OR
 * group of overlapping bigram phrases (`"数据库教程"` → `("数据" OR "据库" OR
 * "库教" OR "教程")`), so the term matches on any partial overlap while per-
 * term AND semantics, negation, and bm25 ranking of longer shared runs are
 * preserved. 1-2 character terms keep the exact phrase.
 *
 * Run with: bun test test/store-cjk-bigram-fts.test.ts
 *        or: pnpm test:node test/store-cjk-bigram-fts.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { unlink, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import { createStore } from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

// =============================================================================
// Fixtures / helpers
// =============================================================================

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-cjk-bigram-"));
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {
    // ignore
  }
});

function freshDbPath(): string {
  return join(testDir, `bigram-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

async function setEmptyConfig(): Promise<void> {
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(emptyConfig));
}

function createBaseSchema(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);
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
  db.exec(`
    CREATE TABLE IF NOT EXISTS store_config (
      key TEXT PRIMARY KEY,
      value TEXT
    )
  `);
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      filepath, title, body,
      tokenize='porter unicode61'
    )
  `);
}

function seedDocument(
  db: Database,
  opts: { id: number; collection: string; path: string; title: string; body: string }
): void {
  const hash = `hash-${opts.id}`;
  const now = new Date().toISOString();
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, opts.body, now);
  db.prepare(`
    INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(opts.id, opts.collection, opts.path, opts.title, hash, now, now, 1);
}

const DOCS = [
  { id: 1, path: "db-ml.md", body: "这是一篇介绍数据库和机器学习的文章。" },
  { id: 2, path: "cooking.md", body: "红烧肉的做法与烹饪技巧。" },
  { id: 3, path: "interview.md", body: "机器学习面试常见问题与面试技巧汇总。" },
  { id: 4, path: "coupon.md", body: "学生优惠券发放公告。" },
];

describe("buildFTS5Query — CJK bigram OR fallback (#976)", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
    const seed = openDatabase(dbPath);
    createBaseSchema(seed);
    for (const d of DOCS) {
      seedDocument(seed, { id: d.id, collection: "zh", path: d.path, title: d.path, body: d.body });
    }
    seed.close();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("a CJK run absent from the corpus no longer voids the query (#976)", () => {
    const store = createStore(dbPath);
    try {
      // 数据库教程 appears nowhere as a contiguous run: doc 1 has 数据库 and
      // separately …机器学习的文章; nothing contains 库教 or 教程 adjacent to
      // the run. Pre-fix this query returned zero results; with the bigram
      // fallback the 数据/据库 overlap still finds doc 1.
      const hits = store.searchFTS("数据库教程", 10, "zh");
      expect(hits.map(h => h.displayPath)).toEqual(["zh/db-ml.md"]);
    } finally {
      store.close();
    }
  });

  test("exact contiguous runs still match", () => {
    const store = createStore(dbPath);
    try {
      // Contiguous run in 学生优惠券 matches through the same bigram group.
      const coupon = store.searchFTS("优惠券", 10, "zh");
      expect(coupon.map(h => h.displayPath)).toEqual(["zh/coupon.md"]);

      // Contiguous 4-char run matches in both docs holding it.
      const ml = store.searchFTS("机器学习", 10, "zh");
      expect(ml.map(h => h.displayPath).sort()).toEqual(["zh/db-ml.md", "zh/interview.md"]);
    } finally {
      store.close();
    }
  });

  test("per-term AND semantics are preserved", () => {
    const store = createStore(dbPath);
    try {
      // Both docs hold 机器学习, but only interview.md also holds 面试.
      const hits = store.searchFTS("机器学习 面试", 10, "zh");
      expect(hits.map(h => h.displayPath)).toEqual(["zh/interview.md"]);
    } finally {
      store.close();
    }
  });

  test("negated CJK terms still exclude via the bigram group", () => {
    const store = createStore(dbPath);
    try {
      // doc 1 holds the 数据 overlap, so the negated group must exclude it;
      // interview.md holds 机器学习 without any 数据库教程 bigram and remains.
      const hits = store.searchFTS("机器学习 -数据库教程", 10, "zh");
      expect(hits.map(h => h.displayPath)).toEqual(["zh/interview.md"]);
    } finally {
      store.close();
    }
  });

  test("unrelated corpora stay un-matched", () => {
    const store = createStore(dbPath);
    try {
      const hits = store.searchFTS("数据库教程", 10, "zh");
      expect(hits.map(h => h.displayPath)).not.toContain("zh/cooking.md");
      expect(hits.map(h => h.displayPath)).not.toContain("zh/coupon.md");
    } finally {
      store.close();
    }
  });
});
