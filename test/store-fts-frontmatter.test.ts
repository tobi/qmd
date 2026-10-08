/**
 * documents_fts indexes leading YAML frontmatter as its own `head` column:
 * splitter, 3-column store rebuild, ranking, triggers, older writers, rename.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import YAML from "yaml";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import { createStore, renameCollection } from "../src/store.js";
import { splitFrontmatter } from "../src/metadata.js";
import { VECTOR_PARTITION_VERSION } from "../src/store-migrations.js";
import type { CollectionConfig } from "../src/collections.js";

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-fts-frontmatter-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true }).catch(() => {});
});

beforeEach(async () => {
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(emptyConfig));
});

function freshDbPath(): string {
  return join(testDir, `fm-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

/**
 * A store as an older qmd left it: base tables, a 3-column documents_fts with
 * the CJK stamp set (so only the missing `head` column can trigger a rebuild),
 * and no rows in FTS.
 */
function createThreeColumnStore(dbPath: string, docs: { id: number; path: string; doc: string }[]): void {
  const db = openDatabase(dbPath);
  db.exec(`CREATE TABLE content (hash TEXT PRIMARY KEY, doc TEXT NOT NULL, created_at TEXT NOT NULL)`);
  db.exec(`
    CREATE TABLE documents (
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
  db.exec(`CREATE TABLE store_config (key TEXT PRIMARY KEY, value TEXT)`);
  db.exec(`CREATE VIRTUAL TABLE documents_fts USING fts5(filepath, title, body, tokenize='porter unicode61')`);
  db.prepare(`INSERT INTO store_config (key, value) VALUES ('fts_cjk_normalized_version', '1')`).run();
  const now = new Date().toISOString();
  const insertContent = db.prepare(`INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)`);
  const insertDoc = db.prepare(`
    INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, 'notes', ?, ?, ?, ?, ?, 1)
  `);
  for (const d of docs) {
    insertContent.run(`hash-${d.id}`, d.doc, now);
    insertDoc.run(d.id, d.path, `Doc ${d.id}`, `hash-${d.id}`, now, now);
  }
  db.close();
}

function ftsColumns(db: Database): string[] {
  return (db.prepare(`PRAGMA table_info(documents_fts)`).all() as { name: string }[]).map(c => c.name);
}

function matchCount(db: Database, match: string): number {
  const row = db.prepare(`SELECT count(*) AS n FROM documents_fts WHERE documents_fts MATCH ?`).get(match) as { n: number };
  return Number(row.n);
}

// =============================================================================
// splitFrontmatter
// =============================================================================

describe("splitFrontmatter", () => {
  test.each([
    ["LF", "---\ntitle: A\n---\nBody\n", "title: A\n", "Body\n"],
    ["CRLF", "---\r\ntitle: A\r\n---\r\nBody\r\n", "title: A\r\n", "Body\r\n"],
    ["BOM", "﻿---\ntitle: A\n---\nBody", "title: A\n", "Body"],
    ["`...` closer", "---\ntitle: A\n...\nBody", "title: A\n", "Body"],
    ["closer at EOF", "---\ntitle: A\n---", "title: A\n", ""],
    ["trailing spaces on delimiters", "--- \ntitle: A\n---\t\nBody", "title: A\n", "Body"],
    ["empty frontmatter", "---\n---\nBody", "", "Body"],
  ])("splits %s", (_name, content, head, body) => {
    expect(splitFrontmatter(content, "a.md")).toEqual({ head, body });
  });

  test.each([
    ["no closing delimiter", "---\ntitle: A\nBody"],
    ["`---` only later in the file (a rule)", "# Title\n\n---\n\nBody\n---\n"],
    ["`----` is not a delimiter", "----\ntitle: A\n----\nBody"],
    ["text after the opening `---`", "--- x\ntitle: A\n---\nBody"],
    ["no frontmatter", "# Title\n\nBody"],
  ])("leaves the document whole when there is %s", (_name, content) => {
    expect(splitFrontmatter(content, "a.md")).toEqual({ head: "", body: content });
  });

  test("only splits frontmatter file types", () => {
    const content = "---\ntitle: A\n---\nBody";
    expect(splitFrontmatter(content, "a.txt")).toEqual({ head: "", body: content });
    expect(splitFrontmatter(content, "a.MDX").head).toBe("title: A\n");
  });

  test("a closing `---` ends the block even inside what reads as a value", () => {
    // Same rule as metadata extraction: the first delimiter line closes.
    expect(splitFrontmatter("---\na: |\n  x\n---\nBody", "a.md")).toEqual({ head: "a: |\n  x\n", body: "Body" });
  });
});

// =============================================================================
// Migration of a 3-column store
// =============================================================================

describe("documents_fts head column — migration", () => {
  test("rebuilds a 3-column store across batch boundaries; frontmatter lands in head only", () => {
    const dbPath = freshDbPath();
    const docs = Array.from({ length: 1201 }, (_, i) => ({
      id: i + 1,
      path: `doc-${i + 1}.md`,
      doc: `---\ndescription: frontmatterterm${i + 1}\n---\n# Doc ${i + 1}\n\nbodyterm${i + 1} prose.\n`,
    }));
    createThreeColumnStore(dbPath, docs);

    const store = createStore(dbPath);
    try {
      const db = store.db;
      expect(ftsColumns(db)).toEqual(["filepath", "title", "body", "head"]);
      expect(Number((db.prepare(`SELECT count(*) AS n FROM documents_fts`).get() as { n: number }).n)).toBe(1201);
      for (const id of [1, 500, 501, 1000, 1201]) {
        expect(matchCount(db, `head : frontmatterterm${id}`)).toBe(1);
        expect(matchCount(db, `body : frontmatterterm${id}`)).toBe(0);
        expect(matchCount(db, `body : bodyterm${id}`)).toBe(1);
      }
      const leftovers = db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'documents_fts_rebuild%'`
      ).all();
      expect(leftovers).toEqual([]);
      const triggers = (db.prepare(`SELECT name FROM sqlite_master WHERE type = 'trigger'`).all() as { name: string }[])
        .map(t => t.name).sort();
      expect(triggers).toEqual(expect.arrayContaining(["documents_ad", "documents_ai", "documents_au"]));
      expect(store.searchFTS("frontmatterterm777", 5).map(r => r.filepath)).toEqual(["qmd://notes/doc-777.md"]);
    } finally {
      store.close();
    }

    // A second open finds the current schema and leaves FTS alone.
    const db = openDatabase(dbPath);
    db.prepare(`DELETE FROM documents_fts WHERE rowid = 1`).run();
    db.close();
    const reopened = createStore(dbPath);
    try {
      expect(Number((reopened.db.prepare(`SELECT count(*) AS n FROM documents_fts`).get() as { n: number }).n)).toBe(1200);
    } finally {
      reopened.close();
    }
  });

  test("does not renumber user_version, so the vector layout step still runs on old stores", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.md", doc: "---\ntags: x\n---\nBody" }]);
    const store = createStore(dbPath);
    try {
      expect(Number((store.db.prepare(`PRAGMA user_version`).get() as { user_version: number }).user_version))
        .toBe(VECTOR_PARTITION_VERSION);
    } finally {
      store.close();
    }
  });
});

// =============================================================================
// Ranking, triggers, older writers
// =============================================================================

describe("documents_fts head column — behaviour", () => {
  test("a term in frontmatter outranks the same term once in a long body", () => {
    const dbPath = freshDbPath();
    const filler = "Unrelated words about something else entirely. ".repeat(40);
    createThreeColumnStore(dbPath, [
      { id: 1, path: "prose.md", doc: `# Notes\n\n${filler}It mentions reconciliation once.\n` },
      { id: 2, path: "tagged.md", doc: `---\ntags: [reconciliation]\n---\n# Ledger\n\n${filler}\n` },
    ]);
    const store = createStore(dbPath);
    try {
      expect(store.searchFTS("reconciliation", 5).map(r => r.filepath)).toEqual([
        "qmd://notes/tagged.md",
        "qmd://notes/prose.md",
      ]);
    } finally {
      store.close();
    }
  });

  test("the sync triggers still index a direct write, with the whole document in body", () => {
    const dbPath = freshDbPath();
    createStore(dbPath).close();
    const db = openDatabase(dbPath);
    try {
      const now = new Date().toISOString();
      db.prepare(`INSERT INTO content (hash, doc, created_at) VALUES ('h1', ?, ?)`).run("---\ntags: triggerterm\n---\nBody", now);
      db.prepare(`
        INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
        VALUES ('notes', 'direct.md', 'Direct', 'h1', ?, ?, 1)
      `).run(now, now);
      expect(matchCount(db, `body : triggerterm`)).toBe(1);
    } finally {
      db.close();
    }
  });

  test("a 3-column insert (older binary) works and the schema is not rebuilt on the next open", () => {
    const dbPath = freshDbPath();
    createStore(dbPath).close();
    const db = openDatabase(dbPath);
    db.prepare(`INSERT INTO documents_fts(rowid, filepath, title, body) VALUES (99, 'notes/old.md', 'Old', 'oldwriter')`).run();
    db.close();
    const store = createStore(dbPath);
    try {
      expect(matchCount(store.db, "oldwriter")).toBe(1);
    } finally {
      store.close();
    }
  });
});

describe("documents_fts head column — upgrades and rewrites", () => {
  test("a 3-column store at the current user_version keeps its triggers after the rebuild", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.md", doc: "---\ntags: uvterm\n---\nBody" }]);
    const seed = openDatabase(dbPath);
    seed.exec(`PRAGMA user_version = ${VECTOR_PARTITION_VERSION}`);
    seed.close();
    const store = createStore(dbPath);
    try {
      expect(matchCount(store.db, "head : uvterm")).toBe(1);
      const triggers = (store.db.prepare(`SELECT name FROM sqlite_master WHERE type = 'trigger'`).all() as { name: string }[])
        .map(t => t.name);
      expect(triggers).toEqual(expect.arrayContaining(["documents_ad", "documents_ai", "documents_au"]));
    } finally {
      store.close();
    }
  });

  test("a store an older binary repopulated (CJK stamp only, head empty) is rebuilt", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.md", doc: "---\ntags: oldrebuild\n---\nBody" }]);
    createStore(dbPath).close();
    // What an older binary leaves behind after winning the publish race.
    const db = openDatabase(dbPath);
    db.exec(`DELETE FROM documents_fts`);
    db.prepare(`INSERT INTO documents_fts(rowid, filepath, title, body) VALUES (1, 'notes/a.md', 'Doc 1', ?)`)
      .run("---\ntags: oldrebuild\n---\nBody");
    db.prepare(`DELETE FROM store_config WHERE key = 'fts_head_version'`).run();
    db.close();
    const store = createStore(dbPath);
    try {
      expect(matchCount(store.db, "head : oldrebuild")).toBe(1);
      expect(matchCount(store.db, "body : oldrebuild")).toBe(0);
    } finally {
      store.close();
    }
  });

  test("renaming a collection keeps frontmatter in head", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.md", doc: "---\ntags: renameterm\n---\nBody" }]);
    const store = createStore(dbPath);
    try {
      store.db.prepare(`INSERT OR IGNORE INTO store_collections (name, path, pattern) VALUES ('notes', '/tmp/notes', '**/*.md')`).run();
      renameCollection(store.db, "notes", "renamed");
      expect(matchCount(store.db, "head : renameterm")).toBe(1);
      expect(matchCount(store.db, "body : renameterm")).toBe(0);
    } finally {
      store.close();
    }
  });

  test("CJK frontmatter is normalized into head", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.md", doc: "---\ntitle: 数据库索引\n---\nBody" }]);
    const store = createStore(dbPath);
    try {
      expect(store.searchFTS("数据库", 5).map(r => r.filepath)).toEqual(["qmd://notes/a.md"]);
    } finally {
      store.close();
    }
  });

  test("a non-Markdown file keeps a leading `---` block in body", () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, [{ id: 1, path: "a.txt", doc: "---\ntags: txtterm\n---\nBody" }]);
    const store = createStore(dbPath);
    try {
      expect(matchCount(store.db, "body : txtterm")).toBe(1);
      expect(matchCount(store.db, "head : txtterm")).toBe(0);
    } finally {
      store.close();
    }
  });
});

// =============================================================================
// Concurrent first opens of a 3-column store
// =============================================================================

const thisDir = dirname(fileURLToPath(import.meta.url));
const workerScript = join(thisDir, "_helpers", "store-init-worker.ts");
const tsxCli = join(thisDir, "..", "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

function runWorker(dbPath: string, startAtMs: number): Promise<{ code: number | null; stderr: string }> {
  const args = isBunRuntime ? [workerScript, dbPath, String(startAtMs)] : [tsxCli, workerScript, dbPath, String(startAtMs)];
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("close", (code) => resolve({ code, stderr }));
  });
}

describe("documents_fts head column — concurrent migration", () => {
  test("parallel first opens of a 3-column store all succeed and publish one complete index", async () => {
    const dbPath = freshDbPath();
    createThreeColumnStore(dbPath, Array.from({ length: 600 }, (_, i) => ({
      id: i + 1,
      path: `doc-${i + 1}.md`,
      doc: `---\ntags: t${i + 1}\n---\nBody ${i + 1}\n`,
    })));
    const startAtMs = Date.now() + 1000;
    const results = await Promise.all(Array.from({ length: 6 }, () => runWorker(dbPath, startAtMs)));
    const failed = results.filter(r => r.code !== 0).map(r => r.stderr.trim());
    expect(failed).toEqual([]);

    const db = openDatabase(dbPath);
    try {
      expect(ftsColumns(db)).toEqual(["filepath", "title", "body", "head"]);
      expect(Number((db.prepare(`SELECT count(*) AS n FROM documents_fts`).get() as { n: number }).n)).toBe(600);
      expect(matchCount(db, "head : t600")).toBe(1);
    } finally {
      db.close();
    }
  }, 60_000);
});
