/**
 * The orphan-vector scans in `qmd update`, `qmd cleanup` and the cleanup
 * preview must look each vector row's document up by hash. Without
 * `sqlite_stat1`, SQLite 3.54 otherwise picks idx_documents_collection
 * (collection, active), which matches every active document in the
 * collection, so the scan costs documents × vector rows: 692 s inside
 * cleanup's write transaction on a 23k-document index.
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cleanupOrphanedVectors, createStore, previewCleanup, type Store } from "../src/store.js";
import type { Database } from "../src/db.js";

const MODEL = "plan-model";
const DOCS = 40;

let dir: string | undefined;
let store: Store | undefined;

afterEach(async () => {
  store?.close();
  store = undefined;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = undefined;
});

async function seededStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-orphan-plan-"));
  const s = createStore(join(dir, "index.sqlite"));
  s.ensureVecTable(3);
  const now = new Date().toISOString();
  s.db.transaction(() => {
    for (let i = 0; i < DOCS; i++) {
      const hash = `doc${String(i).padStart(4, "0")}`;
      s.insertContent(hash, `# ${hash}\n\nbody ${hash}`, now);
      s.insertDocument("notes", `${hash}.md`, hash, hash, now, now);
      s.insertEmbedding(hash, 0, 0, new Float32Array([1, i / DOCS, 0]), MODEL, now, 1);
      // Every fourth document goes inactive, so its vector row is an orphan.
      if (i % 4 === 0) s.deactivateDocument("notes", `${hash}.md`);
    }
  })();
  return s;
}

/** The store's database with the SQL of every prepared statement recorded. */
function recordStatements(db: Database, seen: string[]): Database {
  return new Proxy(db, {
    get(target, property) {
      if (property === "prepare") {
        return (sql: string) => {
          seen.push(sql);
          return target.prepare(sql);
        };
      }
      const member = target[property as keyof Database];
      return typeof member === "function" ? member.bind(target) : member;
    },
  });
}

/**
 * Query-plan lines for the documents access (SEARCH or SCAN of `d`) in the one
 * vector_rows orphan scan the call prepared.
 */
function documentLookups(db: Database, run: (db: Database) => unknown): string[] {
  const seen: string[] = [];
  run(recordStatements(db, seen));
  const scans = seen.filter(sql =>
    /\bFROM\s+vector_rows\s+vr\b/.test(sql) && /NOT EXISTS\s*\(\s*SELECT 1 FROM documents d\b/.test(sql));
  expect(scans).toHaveLength(1);
  const plan = (db.prepare(`EXPLAIN QUERY PLAN ${scans[0]}`).all() as { detail: string }[]).map(row => row.detail);
  const lookups = plan.filter(detail => /^(SCAN|SEARCH)\s+(TABLE\s+)?d\b/.test(detail));
  expect(lookups.length, plan.join(" | ")).toBeGreaterThan(0);
  return lookups;
}

describe("orphan-vector scans look documents up by hash", () => {
  for (const statistics of [false, true]) {
    const label = statistics ? "with sqlite_stat1" : "without sqlite_stat1";

    test(`qmd cleanup's preview ${label}`, async () => {
      const s = (store = await seededStore());
      if (statistics) s.db.exec("ANALYZE");
      for (const lookup of documentLookups(s.db, db => previewCleanup(db))) {
        expect(lookup).toContain("idx_documents_hash (hash=?)");
      }
    });

    test(`qmd update's cleanupOrphanedVectors ${label}`, async () => {
      const s = (store = await seededStore());
      if (statistics) s.db.exec("ANALYZE");
      for (const lookup of documentLookups(s.db, db => cleanupOrphanedVectors(db))) {
        expect(lookup).toContain("idx_documents_hash (hash=?)");
      }
      // The fix changes the plan, not the answer: the ten orphans still go.
      const rows = (s.db.prepare(`SELECT COUNT(*) AS c FROM vector_rows`).get() as { c: number }).c;
      expect(rows).toBe(DOCS - DOCS / 4);
    });
  }
});
