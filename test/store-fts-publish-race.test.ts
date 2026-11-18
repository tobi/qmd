/**
 * store-fts-publish-race.test.ts — the CJK FTS rebuild publishing a corpus that
 * other connections are still writing (src/store.ts:
 * rebuildFTSForCjkNormalization and its publish transaction).
 *
 * The rebuild reads documents into a per-process shadow table in batches with
 * no lock held, then publishes under BEGIN IMMEDIATE. A writer that commits in
 * that window used to be lost by the blind DELETE+INSERT: an updated document
 * got its pre-update text back, a new document ended up with no row at all, and
 * a deactivated one could keep a row. The published state carries the one-time
 * version stamp, so a bad publish is never repaired afterwards.
 *
 * The publish now diffs a per-row identity snapshot (taken while reading)
 * against the documents table inside the publish transaction and re-reads only
 * what changed. These tests drive that window deterministically with the
 * internal before-publish seam — the hook runs after the shadow is complete and
 * before the publish transaction opens — and commit through a second connection
 * exactly there, which is what a concurrent `qmd update` does.
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { unlink, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import { createStore, _setFtsRebuildBeforePublishHookForTesting } from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

const FTS_CJK_NORMALIZED_VERSION = "1";

let testDir: string;
let dbPath: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-fts-publish-race-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(async () => {
  dbPath = join(testDir, `race-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  // Point config resolution at throwaway empty collections.
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(emptyConfig));
});

afterEach(async () => {
  _setFtsRebuildBeforePublishHookForTesting(null);
  delete process.env.QMD_CONFIG_DIR;
  try {
    await unlink(dbPath);
  } catch {
    // ignore
  }
});

/**
 * Minimal pre-migration schema: documents + content, documents_fts present but
 * empty. No CJK version stamp, so the next open must rebuild; the FTS sync
 * triggers are installed by the opening store before its rebuild, so writers
 * that commit during the rebuild maintain live FTS exactly like production.
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

function leftoverRebuildTables(db: Database): string[] {
  const rows = db.prepare(`
    SELECT name FROM sqlite_master
    WHERE name LIKE 'documents_fts_rebuild%' OR name LIKE 'documents_fts_snapshot%'
  `).all() as { name: string }[];
  return rows.map(row => row.name);
}

function cjkVersion(db: Database): string | undefined {
  const row = db.prepare(`SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`)
    .get() as { value?: string } | undefined;
  return row?.value;
}

/** Run a write against the same index from a second connection. */
function writeFromSecondConnection(write: (writer: Database) => void): void {
  const writer = openDatabase(dbPath);
  try {
    writer.exec(`BEGIN IMMEDIATE`);
    write(writer);
    writer.exec(`COMMIT`);
  } catch (error) {
    if (writer.inTransaction) writer.exec(`ROLLBACK`);
    throw error;
  } finally {
    writer.close();
  }
}

describe("CJK FTS rebuild — concurrent writer during the shadow build", () => {
  test("updates, inserts and deactivations committed before the publish are not lost", () => {
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      seedDocument(seed, {
        id: 1, collection: "c", path: "a.md", title: "Alpha",
        body: "# Alpha\n\nalphaneedle lives only in the first document",
      });
      seedDocument(seed, {
        id: 2, collection: "c", path: "b.md", title: "Beta",
        body: "# Beta\n\nstaleneedle lives only in the old body",
      });
      seedDocument(seed, {
        id: 3, collection: "c", path: "c.md", title: "Gamma",
        body: "# Gamma\n\nretiredneedle belongs to the removed document",
      });
      seed.close();
    }

    let sawWriter = false;
    _setFtsRebuildBeforePublishHookForTesting(() => {
      sawWriter = true;
      writeFromSecondConnection(writer => {
        const now = new Date().toISOString();
        // Doc 2 was edited on disk: new content row, new hash/title on its
        // document. The documents_au trigger also writes the raw new body into
        // live documents_fts — which the publish's DELETE would throw away.
        writer.prepare(`INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
          .run("hash-2b", "# Rewritten\n\nfreshneedle 数据库指南", now);
        writer.prepare(`UPDATE documents SET hash = ?, title = ?, modified_at = ? WHERE id = 2`)
          .run("hash-2b", "Rewritten", now);
        // Doc 4 is a brand-new file.
        writer.prepare(`INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
          .run("hash-4", "# Newcomer\n\nnewcomerneedle arrived mid-rebuild", now);
        writer.prepare(`
          INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
          VALUES (4, 'c', 'd.md', 'Newcomer', 'hash-4', ?, ?, 1)
        `).run(now, now);
        // Doc 3's file was deleted, so its document is deactivated.
        writer.prepare(`UPDATE documents SET active = 0 WHERE id = 3`).run();
      });
    });

    const store = createStore(dbPath);
    try {
      expect(sawWriter).toBe(true);
      const db = store.db;

      // The published index matches the corpus at commit time, not at read time.
      expect(activeDocCount(db)).toBe(3);
      expect(ftsRowCount(db)).toBe(3);

      expect(store.searchFTS("staleneedle", 10)).toHaveLength(0);
      expect(store.searchFTS("freshneedle", 10).map(r => r.displayPath)).toEqual(["c/b.md"]);
      expect(store.searchFTS("newcomerneedle", 10).map(r => r.displayPath)).toEqual(["c/d.md"]);
      expect(store.searchFTS("retiredneedle", 10)).toHaveLength(0);
      expect(store.searchFTS("alphaneedle", 10).map(r => r.displayPath)).toEqual(["c/a.md"]);

      // The re-read row is normalized like every other rebuilt row: the raw row
      // the concurrent writer's trigger wrote would not match a CJK phrase.
      expect(store.searchFTS("数据库", 10).map(r => r.displayPath)).toEqual(["c/b.md"]);

      expect(cjkVersion(db)).toBe(FTS_CJK_NORMALIZED_VERSION);
      expect(leftoverRebuildTables(db)).toEqual([]);
    } finally {
      store.close();
    }
  });

  test("a document deleted or activated during the rebuild ends up indexed exactly once", () => {
    {
      const seed = openDatabase(dbPath);
      createBaseSchema(seed);
      seedDocument(seed, {
        id: 1, collection: "c", path: "a.md", title: "Alpha",
        body: "# Alpha\n\nalphaneedle stays",
      });
      seedDocument(seed, {
        id: 2, collection: "c", path: "b.md", title: "Doomed",
        body: "# Doomed\n\ndoomedneedle is deleted mid-rebuild",
      });
      // Inactive when the shadow reads its id range, so it is never read.
      seedDocument(seed, {
        id: 3, collection: "c", path: "c.md", title: "Revived",
        body: "# Revived\n\nrevivedneedle comes back mid-rebuild",
        active: 0,
      });
      seed.close();
    }

    _setFtsRebuildBeforePublishHookForTesting(() => {
      writeFromSecondConnection(writer => {
        writer.prepare(`DELETE FROM documents WHERE id = 2`).run();
        writer.prepare(`UPDATE documents SET active = 1 WHERE id = 3`).run();
      });
    });

    const store = createStore(dbPath);
    try {
      const db = store.db;

      expect(activeDocCount(db)).toBe(2);
      expect(ftsRowCount(db)).toBe(2);

      expect(store.searchFTS("doomedneedle", 10)).toHaveLength(0);
      expect(store.searchFTS("alphaneedle", 10).map(r => r.displayPath)).toEqual(["c/a.md"]);
      expect(store.searchFTS("revivedneedle", 10).map(r => r.displayPath)).toEqual(["c/c.md"]);
      expect(cjkVersion(db)).toBe(FTS_CJK_NORMALIZED_VERSION);
    } finally {
      store.close();
    }
  });

  test("without a concurrent writer the rebuild publishes every active document", () => {
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
          body: `# Doc ${i}\n\nuniquetoken${i} body content`,
        });
      }
      seed.close();
    }

    let sawHook = false;
    _setFtsRebuildBeforePublishHookForTesting(() => {
      sawHook = true;
    });

    const store = createStore(dbPath);
    try {
      const db = store.db;
      expect(sawHook).toBe(true);
      expect(ftsRowCount(db)).toBe(DOC_COUNT);
      expect(ftsRowCount(db)).toBe(activeDocCount(db));
      for (let i = 1; i <= DOC_COUNT; i++) {
        expect(store.searchFTS(`uniquetoken${i}`, 10).map(r => r.displayPath)).toEqual([`c/doc-${i}.md`]);
      }
      expect(cjkVersion(db)).toBe(FTS_CJK_NORMALIZED_VERSION);
      expect(leftoverRebuildTables(db)).toEqual([]);
    } finally {
      store.close();
    }

    // A later open sees the stamp and does not rebuild again.
    const reopened = createStore(dbPath);
    try {
      expect(ftsRowCount(reopened.db)).toBe(DOC_COUNT);
    } finally {
      reopened.close();
    }
  });
});
