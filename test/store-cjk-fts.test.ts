/**
 * store-cjk-fts.test.ts — Regression tests for the CJK FTS migration fix.
 *
 * Covers rebuildFTSForCjkNormalization()'s batched, shadow-table rebuild
 * (src/store.ts). The original implementation cleared documents_fts up front,
 * loaded every document body into memory via a single unbounded .all() inside
 * one giant transaction, and left FTS empty on a mid-rebuild crash.
 *
 * The row-scan strategy has since gone through two fixes for two distinct
 * failure modes, both still enforced here:
 *   - OOM (the original bug): never materialize every active document's body
 *     into one JS array at once. First fixed via a streaming .iterate()
 *     cursor; now enforced via bounded, keyset-paginated .all() batches
 *     (`WHERE id > ? ORDER BY id LIMIT ?`) — each call materializes at most
 *     one bounded batch, then the batch is discarded before the next SELECT.
 *   - "database is busy" (a later regression): a better-sqlite3 .iterate()
 *     cursor left open while a transaction begins on the same connection
 *     raises "database is busy". Bounded LIMIT/OFFSET-style batches instead
 *     fully finalize each SELECT before the insert transaction starts, so the
 *     connection is never busy with two concurrent statements.
 * Both fixes share the same shadow-table structure:
 *   - builds a separate documents_fts_rebuild shadow table in bounded batches,
 *   - atomically swaps it in only after a complete pass,
 *   - drops a lingering shadow table from a prior crashed run on the next run.
 *
 * These tests exercise the migration through createStore()/openDatabase() on a
 * pre-seeded DB that omits the fts_cjk_normalized_version marker (forcing a
 * rebuild on open), plus a structural assertion that the body scan is bounded
 * (LIMIT + keyset pagination) rather than a single unbounded .all() over every
 * active document. They also cover a brand-new empty store and a legacy
 * fts5(name, body, content='documents') schema, which used to throw
 * `no such column: T.name` during rebuildFTSForCjkNormalization (#792).
 *
 * Run with: bun test test/store-cjk-fts.test.ts
 *        or: pnpm test:node test/store-cjk-fts.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { unlink, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import { createStore } from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

// =============================================================================
// Fixtures / helpers
// =============================================================================

const FTS_CJK_NORMALIZED_VERSION = "1";

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-cjk-fts-"));
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {
    // ignore
  }
});

/** Allocate a fresh DB path inside the shared temp dir. */
function freshDbPath(): string {
  return join(testDir, `cjk-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

/**
 * Point QMD config resolution at a throwaway empty-collections config so
 * createStore() doesn't reach into a real workspace.
 */
async function setEmptyConfig(): Promise<void> {
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(emptyConfig));
}

/**
 * Create the minimal subset of QMD schema directly (no triggers) so we can seed
 * documents/content WITHOUT auto-populating documents_fts, then control the FTS
 * shadow state by hand. Mirrors src/store.ts initializeDatabase() column defs.
 */
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

/** Insert one (content, document) pair. Does NOT touch documents_fts. */
function seedDocument(
  db: Database,
  opts: { id: number; collection: string; path: string; title: string; body: string; active?: number }
): void {
  const hash = `hash-${opts.id}`;
  const now = new Date().toISOString();
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, opts.body, now);
  db.prepare(`
    INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(opts.id, opts.collection, opts.path, opts.title, hash, now, now, opts.active ?? 1);
}

function ftsRowCount(db: Database): number {
  const row = db.prepare(`SELECT count(*) as n FROM documents_fts`).get() as { n: number };
  return Number(row.n);
}

function activeDocCount(db: Database): number {
  const row = db.prepare(`SELECT count(*) as n FROM documents WHERE active = 1`).get() as { n: number };
  return Number(row.n);
}

// =============================================================================
// Test 1 — structural: rebuild scans source rows in bounded, paginated batches
// =============================================================================

describe("rebuildFTSForCjkNormalization — bounded source scan", () => {
  test("body scan is bounded (keyset LIMIT pagination), never one unbounded .all()", () => {
    const __filename = fileURLToPath(import.meta.url);
    const storeSrc = readFileSync(
      join(dirname(__filename), "..", "src", "store.ts"),
      "utf8"
    );

    // Isolate the migration function body so the assertion can't be satisfied
    // by an unrelated .all()/.iterate() elsewhere in the 4000-line file.
    const startIdx = storeSrc.indexOf("function rebuildFTSForCjkNormalization(");
    expect(startIdx).toBeGreaterThan(-1);
    // initializeDatabase() is the next top-level function after the migration.
    const endIdx = storeSrc.indexOf("function initializeDatabase(", startIdx);
    expect(endIdx).toBeGreaterThan(startIdx);
    const fnBodyRaw = storeSrc.slice(startIdx, endIdx);

    // Strip line + block comments so the assertions match real executed code,
    // not narrative comments that mention old/alternate behavior.
    const fnBody = fnBodyRaw
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n")
      .map(line => line.replace(/\/\/.*$/, ""))
      .join("\n");

    // The guarded invariant is "bounded memory", not a specific better-sqlite3
    // API. Any .all() used for the source-body scan must be paired with a
    // LIMIT and a keyset cursor (id > ?), so each call only ever materializes
    // one bounded batch — never every active document's body at once (the
    // original OOM bug) — and each SELECT fully finalizes before the insert
    // transaction begins (the "database is busy" regression this replaced a
    // streaming .iterate() cursor to fix, since holding a cursor open on the
    // same connection while starting a transaction raises "busy").
    expect(fnBody).toMatch(/LIMIT\s*\?/);
    expect(fnBody).toMatch(/d\.id\s*>\s*\?/);
    expect(fnBody).toMatch(/ORDER BY d\.id/);
    // Sanity: it builds into a shadow table and atomically swaps it in via
    // INSERT INTO … SELECT (not ALTER TABLE … RENAME, which triggers SQLite
    // 3.25+ re-validation of dependent trigger bodies).
    expect(fnBody).toContain("documents_fts_rebuild");
    expect(fnBody).toContain("INSERT INTO documents_fts");
    // DELETE FROM documents_fts on leftover content-external FTS compiles as
    // SELECT T.name FROM documents (#792). Repair must run inside this
    // function *before* that DELETE (and before the version-stamp early
    // return), not only in initializeDatabase.
    const ensureIdx = fnBody.indexOf("ensureDocumentsFtsSchema");
    const deleteIdx = fnBody.indexOf("DELETE FROM documents_fts");
    expect(ensureIdx).toBeGreaterThan(-1);
    expect(deleteIdx).toBeGreaterThan(ensureIdx);
  });
});

// =============================================================================
// Test 2 — large library migrates fully without OOM and indexes every active doc
// =============================================================================

describe("rebuildFTSForCjkNormalization — large-library full migration", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("opens successfully and indexes all active docs (50 × ≥10KB bodies)", async () => {
    const DOC_COUNT = 60;
    const bigBody = "lorem ipsum dolor sit amet ".repeat(450); // ~12 KB
    expect(bigBody.length).toBeGreaterThanOrEqual(10 * 1024);

    // Seed a DB that predates the CJK migration: data present, FTS NOT yet
    // normalized, and the version marker absent → next open must rebuild.
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      for (let i = 1; i <= DOC_COUNT; i++) {
        seedDocument(seed, {
          id: i,
          collection: "big",
          path: `doc-${i}.md`,
          title: `Document ${i}`,
          body: `${bigBody} unique-token-${i}`,
        });
      }
      // No fts_cjk_normalized_version row, documents_fts left empty.
      expect(ftsRowCount(seed)).toBe(0);
      seed.close();
    }

    // Open through the real path → triggers rebuildFTSForCjkNormalization().
    const store = createStore(dbPath);
    try {
      const db = store.db;
      // Migration ran to completion and stamped the version marker.
      const ver = db.prepare(
        `SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`
      ).get() as { value?: string } | undefined;
      expect(ver?.value).toBe(FTS_CJK_NORMALIZED_VERSION);

      // FTS rowcount equals active doc count — every active doc got indexed
      // (crossing the BATCH_SIZE=500 boundary is fine; this is the smaller
      // multi-batch correctness check).
      expect(ftsRowCount(db)).toBe(activeDocCount(db));
      expect(ftsRowCount(db)).toBe(DOC_COUNT);

      // And the index actually serves a query for a per-doc unique token.
      const hits = store.searchFTS("unique-token-42", 10, "big");
      expect(hits.length).toBe(1);
      expect(hits[0]!.displayPath).toBe("big/doc-42.md");

      // No leftover shadow table after the swap.
      const shadow = db.prepare(
        `SELECT name FROM sqlite_master WHERE name = 'documents_fts_rebuild'`
      ).get();
      // bun:sqlite .get() returns null, better-sqlite3 returns undefined for no-row.
      expect(shadow ?? undefined).toBeUndefined();
    } finally {
      store.close();
    }
  });

  test("inactive docs are excluded from the rebuilt index", async () => {
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      seedDocument(seed, { id: 1, collection: "c", path: "a.md", title: "A", body: "alpha active doc", active: 1 });
      seedDocument(seed, { id: 2, collection: "c", path: "b.md", title: "B", body: "beta inactive doc", active: 0 });
      seedDocument(seed, { id: 3, collection: "c", path: "d.md", title: "C", body: "gamma active doc", active: 1 });
      seed.close();
    }
    const store = createStore(dbPath);
    try {
      expect(ftsRowCount(store.db)).toBe(activeDocCount(store.db));
      expect(ftsRowCount(store.db)).toBe(2);
      expect(store.searchFTS("beta", 10).length).toBe(0); // inactive not indexed
      expect(store.searchFTS("alpha", 10).length).toBe(1);
    } finally {
      store.close();
    }
  });
});

// =============================================================================
// Test 3 — crashed prior run leaves a shadow table; existing FTS is preserved
// =============================================================================

describe("rebuildFTSForCjkNormalization — recovery from a crashed prior run", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("lingering documents_fts_rebuild is dropped; live FTS not destroyed and ends correct", async () => {
    const DOC_COUNT = 5;
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      for (let i = 1; i <= DOC_COUNT; i++) {
        seedDocument(seed, {
          id: i,
          collection: "c",
          path: `doc-${i}.md`,
          title: `Doc ${i}`,
          body: `searchable body content row-${i}`,
        });
      }

      // Populate the LIVE documents_fts normally (as a pre-migration index
      // would be). This must survive until the swap.
      const ins = seed.prepare(
        `INSERT INTO documents_fts(rowid, filepath, title, body) VALUES (?, ?, ?, ?)`
      );
      for (let i = 1; i <= DOC_COUNT; i++) {
        ins.run(i, `c/doc-${i}.md`, `Doc ${i}`, `searchable body content row-${i}`);
      }
      expect(ftsRowCount(seed)).toBe(DOC_COUNT);

      // Simulate a crash mid-rebuild: a half-built shadow table lingers, with
      // a poisoned/partial row that must NOT survive into the final index.
      seed.exec(`
        CREATE VIRTUAL TABLE documents_fts_rebuild USING fts5(
          filepath, title, body,
          tokenize='porter unicode61'
        )
      `);
      seed.prepare(
        `INSERT INTO documents_fts_rebuild(rowid, filepath, title, body) VALUES (?, ?, ?, ?)`
      ).run(999, "c/poison.md", "Poison", "leftover-from-crashed-run garbage");

      // Version marker absent → migration runs on next open.
      seed.close();
    }

    // Live FTS must still be intact on disk before reopen (sanity: reopen raw).
    {
      const check = openDatabase(dbPath);
      expect(ftsRowCount(check)).toBe(DOC_COUNT); // existing FTS NOT destroyed
      check.close();
    }

    const store = createStore(dbPath);
    try {
      const db = store.db;

      // Migration completed and stamped the version.
      const ver = db.prepare(
        `SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`
      ).get() as { value?: string } | undefined;
      expect(ver?.value).toBe(FTS_CJK_NORMALIZED_VERSION);

      // Final index rowcount is correct (== active docs), NOT inflated by the
      // leftover poison row from the crashed run.
      expect(ftsRowCount(db)).toBe(activeDocCount(db));
      expect(ftsRowCount(db)).toBe(DOC_COUNT);

      // The poison row was dropped with the stale shadow table.
      expect(store.searchFTS("leftover-from-crashed-run", 10).length).toBe(0);

      // Real rows are searchable.
      expect(store.searchFTS("searchable", 10).length).toBe(DOC_COUNT);

      // Stale shadow table is gone.
      const shadow = db.prepare(
        `SELECT name FROM sqlite_master WHERE name = 'documents_fts_rebuild'`
      ).get();
      // bun:sqlite .get() returns null, better-sqlite3 returns undefined for no-row.
      expect(shadow ?? undefined).toBeUndefined();
    } finally {
      store.close();
    }
  });
});

// =============================================================================
// Test 4 — CJK search works after migration
// =============================================================================

describe("rebuildFTSForCjkNormalization — CJK query after migration", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("a Chinese term matches the doc that migration normalized", async () => {
    // 数据库 = "database", 机器学习 = "machine learning"
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      seedDocument(seed, {
        id: 1,
        collection: "zh",
        path: "db.md",
        title: "数据库指南",
        body: "这是一篇关于数据库和机器学习的文章。",
      });
      seedDocument(seed, {
        id: 2,
        collection: "zh",
        path: "other.md",
        title: "无关文档",
        body: "这篇文章谈论烹饪和旅行。",
      });
      seed.close();
    }

    const store = createStore(dbPath);
    try {
      // Without CJK normalization in the rebuild, unicode61 would index the run
      // as one un-segmentable token and this phrase query would miss.
      const hits = store.searchFTS("数据库", 10, "zh");
      expect(hits.length).toBe(1);
      expect(hits[0]!.displayPath).toBe("zh/db.md");

      // A CJK term only present in doc 1 also resolves to doc 1.
      const ml = store.searchFTS("机器学习", 10, "zh");
      expect(ml.length).toBe(1);
      expect(ml[0]!.displayPath).toBe("zh/db.md");
    } finally {
      store.close();
    }
  });
});

// =============================================================================
// Test 5 — Latin BM25 search still works after migration (regression guard)
// =============================================================================

describe("rebuildFTSForCjkNormalization — Latin search regression guard", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("normal English BM25 search is unaffected and ranks the best match first", async () => {
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      seedDocument(seed, {
        id: 1,
        collection: "en",
        path: "vector.md",
        title: "Vector Search",
        body: "Vector search uses embeddings to find semantically similar documents. embeddings embeddings embeddings",
      });
      seedDocument(seed, {
        id: 2,
        collection: "en",
        path: "keyword.md",
        title: "Keyword Search",
        body: "Keyword search uses an inverted index over terms with occasional embeddings mention.",
      });
      seedDocument(seed, {
        id: 3,
        collection: "en",
        path: "cooking.md",
        title: "Cooking",
        body: "A recipe for soup with no relation to information retrieval.",
      });
      seed.close();
    }

    const store = createStore(dbPath);
    try {
      // Prefix term match (Latin path through buildFTS5Query).
      const hits = store.searchFTS("embeddings", 10, "en");
      // Both vector.md and keyword.md mention embeddings; cooking.md does not.
      const paths = hits.map(h => h.displayPath).sort();
      expect(paths).toEqual(["en/keyword.md", "en/vector.md"]);

      // BM25 ranks the embeddings-heavy doc first.
      expect(hits[0]!.displayPath).toBe("en/vector.md");

      // A unique Latin term resolves to exactly one doc.
      const recipe = store.searchFTS("recipe", 10, "en");
      expect(recipe.length).toBe(1);
      expect(recipe[0]!.displayPath).toBe("en/cooking.md");

      // Multi-term AND query.
      const both = store.searchFTS("keyword inverted", 10, "en");
      expect(both.length).toBe(1);
      expect(both[0]!.displayPath).toBe("en/keyword.md");
    } finally {
      store.close();
    }
  });
});


// =============================================================================
// Test 6 — legacy fts5(name, body, content='documents') must not crash open (#792)
// =============================================================================

describe("rebuildFTSForCjkNormalization — legacy external-content FTS schema (#792)", () => {
  let dbPath: string;

  beforeEach(async () => {
    await setEmptyConfig();
    dbPath = freshDbPath();
  });

  afterEach(async () => {
    try {
      await unlink(dbPath);
    } catch {
      // ignore
    }
  });

  test("createStore on a brand-new empty file stamps the CJK version", async () => {
    const store = createStore(dbPath);
    try {
      const ver = store.db.prepare(
        `SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`
      ).get() as { value?: string } | undefined;
      expect(ver?.value).toBe(FTS_CJK_NORMALIZED_VERSION);
      expect(ftsRowCount(store.db)).toBe(0);

      const sqlRow = store.db.prepare(
        `SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'`
      ).get() as { sql?: string } | undefined;
      const sql = (sqlRow?.sql ?? "").toLowerCase();
      expect(sql).toContain("filepath");
      expect(sql).toContain("title");
      expect(sql).not.toMatch(/content\s*=/);
    } finally {
      store.close();
    }
  });

  test("opens a DB whose documents_fts still uses name/body + content='documents'", async () => {
    {
      const seed = openDatabase(dbPath);
      seed.exec(`
        CREATE TABLE IF NOT EXISTS content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
      `);
      seed.exec(`
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
      seed.exec(`
        CREATE TABLE IF NOT EXISTS store_config (
          key TEXT PRIMARY KEY,
          value TEXT
        )
      `);
      // The schema that produced `no such column: T.name` on CJK rebuild:
      // FTS5 external-content maps column `name` onto documents.name.
      seed.exec(`
        CREATE VIRTUAL TABLE documents_fts USING fts5(
          name, body,
          content='documents',
          content_rowid='id',
          tokenize='porter unicode61'
        )
      `);
      seed.exec(`
        CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
          INSERT INTO documents_fts(rowid, name, body)
          SELECT new.id, new.path, content.doc
          FROM content
          WHERE content.hash = new.hash;
        END
      `);
      seedDocument(seed, {
        id: 1,
        collection: "docs",
        path: "readme.md",
        title: "Project README",
        body: "legacy fts canary token UNIQUE_KEYWORD_XYZ",
      });
      seed.close();
    }

    const store = createStore(dbPath);
    try {
      const sqlRow = store.db.prepare(
        `SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'`
      ).get() as { sql?: string } | undefined;
      const sql = (sqlRow?.sql ?? "").toLowerCase();
      expect(sql).toContain("filepath");
      expect(sql).toContain("title");
      expect(sql).not.toMatch(/content\s*=/);

      const ver = store.db.prepare(
        `SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`
      ).get() as { value?: string } | undefined;
      expect(ver?.value).toBe(FTS_CJK_NORMALIZED_VERSION);

      const hits = store.searchFTS("UNIQUE_KEYWORD_XYZ", 10, "docs");
      expect(hits.length).toBe(1);
      expect(hits[0]!.displayPath).toBe("docs/readme.md");
    } finally {
      store.close();
    }
  });

  test("repairs leftover name/body FTS even when the CJK version is already stamped", async () => {
    {
      const seed = openDatabase(dbPath);
      seed.exec(`
        CREATE TABLE IF NOT EXISTS content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
      `);
      seed.exec(`
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
      seed.exec(`
        CREATE TABLE IF NOT EXISTS store_config (
          key TEXT PRIMARY KEY,
          value TEXT
        )
      `);
      seed.exec(`
        CREATE VIRTUAL TABLE documents_fts USING fts5(
          name, body,
          content='documents',
          content_rowid='id',
          tokenize='porter unicode61'
        )
      `);
      seed.prepare(`
        INSERT INTO store_config(key, value) VALUES ('fts_cjk_normalized_version', ?)
      `).run(FTS_CJK_NORMALIZED_VERSION);
      seedDocument(seed, {
        id: 1,
        collection: "docs",
        path: "readme.md",
        title: "Project README",
        body: "stamped-legacy canary STAMPED_LEGACY_XYZ",
      });
      seed.close();
    }

    const store = createStore(dbPath);
    try {
      const sqlRow = store.db.prepare(
        `SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'`
      ).get() as { sql?: string } | undefined;
      const sql = (sqlRow?.sql ?? "").toLowerCase();
      expect(sql).toContain("filepath");
      expect(sql).not.toMatch(/content\s*=/);

      const cols = store.db.prepare(`PRAGMA table_info(documents_fts)`).all() as { name: string }[];
      const names = cols.map(c => c.name);
      expect(names).toContain("filepath");
      expect(names).not.toContain("name");

      const hits = store.searchFTS("STAMPED_LEGACY_XYZ", 10, "docs");
      expect(hits.length).toBe(1);
      expect(hits[0]!.displayPath).toBe("docs/readme.md");
    } finally {
      store.close();
    }
  });
});
