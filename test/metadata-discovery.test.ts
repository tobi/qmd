/**
 * metadata-discovery.test.ts - Store-level metadata discovery: key summaries,
 * per-type value windows, the match over metadata entries, and scope and gate
 * agreement with filtered search.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  insertContent,
  insertDocument,
  hashContent,
  searchFTS,
  getStatus,
  getStatusSummary,
  type Store,
} from "../src/store.js";
import {
  countDocumentsPendingMetadata,
  countDocumentsWithMetadata,
  countMetadataKeys,
  listMetadata,
  listMetadataCollectionSummaries,
  METADATA_SQL_BINDING_BUDGET,
  MetadataBindingBudgetError,
  MetadataOptionError,
  replaceDocumentMetadata,
  type ListMetadataOptions,
  type MetadataKeySummary,
} from "../src/metadata-store.js";
import type { Database, SQLiteValue } from "../src/db.js";
import { compileMetadataFilter, compileMetadataMatch, MetadataFilterError, type MetadataFilter, type MetadataMatch } from "../src/metadata-filter.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-discovery-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  const dbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  store = createStore(dbPath);
});

afterEach(() => {
  store.close();
});

let documentCounter = 0;

/** Insert an active document with extracted metadata. Every body contains "doc" so FTS can reach it. */
async function insertMetadataDoc(collection: string, metadata: DocumentMetadata): Promise<number> {
  documentCounter += 1;
  const path = `doc-${documentCounter}.md`;
  const content = `# doc ${documentCounter}\n\nbody of doc ${documentCounter}\n`;
  const now = new Date().toISOString();
  const hash = await hashContent(content);
  insertContent(store.db, hash, content, now);
  const documentId = insertDocument(store.db, collection, path, path, hash, now, now);
  replaceDocumentMetadata(store.db, documentId, { metadata, extractionVersion: METADATA_EXTRACTION_VERSION });
  return documentId;
}

/** The store's database with every prepared statement's SQL reported before it runs. */
function observeStatements(db: Database, onPrepare: (sql: string) => void, onAll?: (sql: string, params: SQLiteValue[]) => void): Database {
  return new Proxy(db, {
    get(target, property) {
      if (property === "prepare") {
        return (sql: string) => {
          onPrepare(sql);
          const statement = target.prepare(sql);
          if (!onAll) return statement;

          return new Proxy(statement, {
            get(prepared, member) {
              if (member === "all") return (...params: SQLiteValue[]) => {
                onAll(sql, params);
                return prepared.all(...params);
              };
              const value = prepared[member as keyof typeof prepared];
              return typeof value === "function" ? value.bind(prepared) : value;
            },
          });
        };
      }
      // Bun's Database keeps private fields, so its methods must run against the real instance.
      const member = target[property as keyof Database];
      return typeof member === "function" ? member.bind(target) : member;
    },
  });
}

function planOf(sql: string): string[] {
  const placeholders = Array.from(sql.matchAll(/\?/g), () => null);
  const rows = store.db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...placeholders) as { detail: string }[];
  return rows.map(row => row.detail);
}

function summaryOf(keys: MetadataKeySummary[], key: string): MetadataKeySummary {
  const summary = keys.find(candidate => candidate.key === key);
  if (!summary) throw new Error(`key ${key} missing from ${keys.map(candidate => candidate.key).join(", ")}`);
  return summary;
}

function valuesOf(keys: MetadataKeySummary[], key: string): [string | number | boolean, number][] {
  return summaryOf(keys, key).types[0]!.values.map(count => [count.value, count.documents]);
}

describe("listMetadata counting", () => {
  test("counts documents, not values, and reports coverage per key", async () => {
    await insertMetadataDoc("notes", { topics: ["a", "b"], status: "draft" });
    await insertMetadataDoc("notes", { topics: ["a"], status: "published" });
    await insertMetadataDoc("notes", { status: "published" });

    const result = listMetadata(store.db);

    expect(result.documents).toBe(3);
    expect(result.filteredDocuments).toBeUndefined();
    expect(result.keys.map(summary => summary.key)).toEqual(["status", "topics"]);

    const topics = summaryOf(result.keys, "topics");
    expect(topics.documents).toBe(2);
    expect(topics.types).toHaveLength(1);
    expect(topics.types[0]!.multiValued).toBe(true);
    expect(topics.types[0]!.distinctValues).toBe(2);
    expect(valuesOf(result.keys, "topics")).toEqual([["a", 2], ["b", 1]]);

    const status = summaryOf(result.keys, "status");
    expect(status.documents).toBe(3);
    expect(status.types[0]!.multiValued).toBe(false);
    expect(valuesOf(result.keys, "status")).toEqual([["published", 2], ["draft", 1]]);
  });

  test("returns an empty key list when nothing has metadata", async () => {
    await insertMetadataDoc("notes", {});

    const result = listMetadata(store.db);

    expect(result).toEqual({ documents: 1, totalKeys: 0, keys: [], remainingKeys: 0 });
  });

  test("applies the extraction gate filtered search applies", async () => {
    await insertMetadataDoc("notes", { status: "visible" });
    const pendingId = await insertMetadataDoc("notes", { status: "pending" });
    const erroredId = await insertMetadataDoc("notes", { status: "errored" });
    const staleId = await insertMetadataDoc("notes", { status: "stale" });
    const inactiveId = await insertMetadataDoc("notes", { status: "inactive" });

    store.db.prepare(`DELETE FROM document_metadata WHERE document_id = ?`).run(pendingId);
    store.db.prepare(`UPDATE document_metadata SET extraction_error = 'boom' WHERE document_id = ?`).run(erroredId);
    store.db.prepare(`UPDATE document_metadata SET extraction_version = ? WHERE document_id = ?`).run(METADATA_EXTRACTION_VERSION - 1, staleId);
    store.db.prepare(`UPDATE documents SET active = 0 WHERE id = ?`).run(inactiveId);

    const result = listMetadata(store.db);

    // The denominator counts active documents whether or not they are extracted.
    expect(result.documents).toBe(4);
    expect(valuesOf(result.keys, "status")).toEqual([["visible", 1]]);
  });
});

describe("listMetadata scope and filter", () => {
  beforeEach(async () => {
    await insertMetadataDoc("notes", { status: "published", priority: 3 });
    await insertMetadataDoc("notes", { status: "draft", priority: 1 });
    await insertMetadataDoc("work", { status: "published", priority: 5 });
    await insertMetadataDoc("work", { status: "archived" });
  });

  test("undefined collection means every collection", () => {
    const result = listMetadata(store.db);

    expect(result.documents).toBe(4);
    expect(summaryOf(result.keys, "status").documents).toBe(4);
    expect(summaryOf(result.keys, "status").types[0]!.collections).toEqual(["notes", "work"]);
  });

  test("a single collection scopes counts and the denominator", () => {
    const result = listMetadata(store.db, { collection: "notes" });

    expect(result.documents).toBe(2);
    expect(valuesOf(result.keys, "status")).toEqual([["draft", 1], ["published", 1]]);
    expect(summaryOf(result.keys, "status").types[0]!.collections).toEqual(["notes"]);
  });

  test("a collection list scopes to exactly those collections", async () => {
    await insertMetadataDoc("other", { status: "elsewhere" });

    const result = listMetadata(store.db, { collection: ["notes", "work"] });

    expect(result.documents).toBe(4);
    expect(valuesOf(result.keys, "status").map(([value]) => value)).not.toContain("elsewhere");
  });

  test("an unknown collection yields an empty scope", () => {
    const result = listMetadata(store.db, { collection: "missing" });

    expect(result).toEqual({ documents: 0, totalKeys: 0, keys: [], remainingKeys: 0 });
  });

  test("filter narrows which documents are counted and reports how many pass", () => {
    const result = listMetadata(store.db, {
      filter: { field: "status", operator: "eq", value: "published" },
    });

    expect(result.documents).toBe(4);
    expect(result.filteredDocuments).toBe(2);
    expect(summaryOf(result.keys, "priority").documents).toBe(2);
    expect(valuesOf(result.keys, "priority")).toEqual([[3, 1], [5, 1]]);
    expect(valuesOf(result.keys, "status")).toEqual([["published", 2]]);
  });

  test("filter composes with the same AST search accepts", () => {
    const result = listMetadata(store.db, {
      filter: {
        operator: "and",
        operands: [
          { field: "status", operator: "eq", value: "published" },
          { field: "priority", operator: "gte", value: 4 },
        ],
      },
    });

    expect(result.filteredDocuments).toBe(1);
    expect(summaryOf(result.keys, "status").types[0]!.collections).toEqual(["work"]);
  });
});

describe("listMetadata match", () => {
  beforeEach(async () => {
    await insertMetadataDoc("notes", { topics: ["typescript", "sqlite"], owner: "docs-team", "mem-kind": "fact", priority: 3, reviewed: true });
    await insertMetadataDoc("notes", { topics: ["typescript", "PostgreSQL"], owner: "search-team", reviewers: ["docs-team", "security-team"], "mem-scope": "user", priority: 10, reviewed: false });
  });

  function keysOf(match: MetadataMatch): string[] {
    return listMetadata(store.db, { match }).keys.map(summary => summary.key);
  }

  test("eq on the key field reports only that key", () => {
    const result = listMetadata(store.db, { match: { field: "key", operator: "eq", value: "topics" } });

    expect(result.keys.map(summary => summary.key)).toEqual(["topics"]);
    expect(valuesOf(result.keys, "topics")).toEqual([["typescript", 2], ["PostgreSQL", 1], ["sqlite", 1]]);
  });

  test("text operators on the key field select a family of keys", () => {
    expect(keysOf({ field: "key", operator: "prefix", value: "mem-" })).toEqual(["mem-kind", "mem-scope"]);
    expect(keysOf({ field: "key", operator: "suffix", value: "-scope" })).toEqual(["mem-scope"]);
    expect(keysOf({ field: "key", operator: "contains", value: "review" })).toEqual(["reviewed", "reviewers"]);
  });

  test("membership on the key field answers which of several names exist", () => {
    expect(keysOf({ field: "key", operator: "in", value: ["tags", "topics", "labels"] })).toEqual(["topics"]);
    expect(keysOf({ field: "key", operator: "nin", value: ["topics", "owner", "priority", "reviewed"] })).toEqual(["mem-kind", "mem-scope", "reviewers"]);
  });

  test("a match nothing satisfies yields an empty key list and keeps the denominator", () => {
    const result = listMetadata(store.db, { match: { field: "key", operator: "prefix", value: "missing-" } });

    expect(result.keys).toEqual([]);
    expect(result.documents).toBe(2);
  });

  test("eq on the value field is a reverse lookup across keys", () => {
    const result = listMetadata(store.db, { match: { field: "value", operator: "eq", value: "docs-team" } });

    expect(result.keys.map(summary => summary.key)).toEqual(["owner", "reviewers"]);
    expect(summaryOf(result.keys, "owner").documents).toBe(1);
    expect(summaryOf(result.keys, "owner").types[0]!.distinctValues).toBe(1);
    expect(valuesOf(result.keys, "reviewers")).toEqual([["docs-team", 1]]);
  });

  test("a key and a value condition compose with and, keeping the key's array-ness", () => {
    const result = listMetadata(store.db, {
      match: {
        operator: "and",
        operands: [
          { field: "key", operator: "eq", value: "topics" },
          { field: "value", operator: "prefix", value: "type" },
        ],
      },
    });

    expect(valuesOf(result.keys, "topics")).toEqual([["typescript", 2]]);
    expect(summaryOf(result.keys, "topics").types[0]!.remainingValues).toBe(0);
    // Array-ness describes the key, not the matched entries.
    expect(summaryOf(result.keys, "topics").types[0]!.multiValued).toBe(true);
  });

  test("values match by their own type, never through a text form", () => {
    expect(keysOf({ field: "value", operator: "eq", value: 10 })).toEqual(["priority"]);
    expect(keysOf({ field: "value", operator: "eq", value: "10" })).toEqual([]);
    expect(keysOf({ field: "value", operator: "eq", value: true })).toEqual(["reviewed"]);
    expect(keysOf({ field: "value", operator: "contains", value: "true" })).toEqual([]);
    expect(keysOf({ field: "value", operator: "prefix", value: "1" })).toEqual([]);
  });

  test("ordered comparisons on the value field select values and narrow the range", () => {
    const result = listMetadata(store.db, {
      match: {
        operator: "and",
        operands: [
          { field: "key", operator: "eq", value: "priority" },
          { field: "value", operator: "gte", value: 5 },
        ],
      },
    });

    expect(valuesOf(result.keys, "priority")).toEqual([[10, 1]]);
    expect(summaryOf(result.keys, "priority").types[0]!.range).toEqual({ min: 10, median: 10, max: 10 });
    expect(summaryOf(result.keys, "priority").documents).toBe(1);
  });

  test("type on the value field selects keys by the type they hold", () => {
    expect(keysOf({ field: "value", operator: "type", value: "boolean" })).toEqual(["reviewed"]);
    expect(keysOf({ field: "value", operator: "type", value: "number" })).toEqual(["priority"]);
    expect(keysOf({ operator: "not", operand: { field: "value", operator: "type", value: "string" } })).toEqual(["priority", "reviewed"]);
  });

  test("type on the value field reports one side of a key whose documents disagree", async () => {
    await insertMetadataDoc("work", { priority: "high" });

    const result = listMetadata(store.db, {
      match: {
        operator: "and",
        operands: [
          { field: "key", operator: "eq", value: "priority" },
          { field: "value", operator: "type", value: "number" },
        ],
      },
    });

    const priority = summaryOf(result.keys, "priority");
    expect(priority.types.map(typeSummary => typeSummary.type)).toEqual(["number"]);
    expect(priority.documents).toBe(2);
    expect(priority.types[0]!.collections).toEqual(["notes"]);
  });

  test("membership and negation on the value field exclude known values", () => {
    const result = listMetadata(store.db, {
      match: {
        operator: "and",
        operands: [
          { field: "key", operator: "eq", value: "topics" },
          { operator: "not", operand: { field: "value", operator: "in", value: ["typescript"] } },
        ],
      },
    });

    expect(valuesOf(result.keys, "topics")).toEqual([["PostgreSQL", 1], ["sqlite", 1]]);
    expect(keysOf({ field: "value", operator: "nin", value: ["docs-team", "search-team", "security-team", "typescript", "sqlite", "PostgreSQL", "fact", "user"] })).toEqual([]);
  });

  test("or composes across the key and value fields", () => {
    const result = listMetadata(store.db, {
      match: {
        operator: "or",
        operands: [
          { field: "key", operator: "eq", value: "owner" },
          { field: "value", operator: "eq", value: "docs-team" },
        ],
      },
    });

    expect(result.keys.map(summary => summary.key)).toEqual(["owner", "reviewers"]);
    // Every owner entry satisfies the first branch, only one reviewers entry the second.
    expect(valuesOf(result.keys, "owner")).toEqual([["docs-team", 1], ["search-team", 1]]);
    expect(valuesOf(result.keys, "reviewers")).toEqual([["docs-team", 1]]);
  });

  test("caseInsensitive folds ASCII on either field", () => {
    expect(keysOf({ field: "value", operator: "contains", value: "sql" })).toEqual(["topics"]);
    expect(valuesOf(listMetadata(store.db, { match: { field: "value", operator: "contains", value: "sql" } }).keys, "topics")).toEqual([["sqlite", 1]]);
    expect(valuesOf(listMetadata(store.db, { match: { field: "value", operator: "contains", value: "sql", caseInsensitive: true } }).keys, "topics")).toEqual([["PostgreSQL", 1], ["sqlite", 1]]);
    expect(keysOf({ field: "key", operator: "eq", value: "TOPICS", caseInsensitive: true })).toEqual(["topics"]);
  });

  test("a number or boolean operand against the key field matches nothing", () => {
    expect(keysOf({ field: "key", operator: "eq", value: 3 })).toEqual([]);
    expect(keysOf({ field: "key", operator: "gte", value: 0 })).toEqual([]);
    expect(keysOf({ field: "key", operator: "in", value: [true] })).toEqual([]);
    // The key field is always a string, so `type string` on it is every key.
    const everyKey = listMetadata(store.db).keys.map(summary => summary.key);
    expect(keysOf({ field: "key", operator: "type", value: "string" })).toEqual(everyKey);
    expect(keysOf({ field: "key", operator: "type", value: "number" })).toEqual([]);
  });

  test("match composes with filter: filter counts documents, match selects entries", () => {
    const result = listMetadata(store.db, {
      match: { field: "key", operator: "eq", value: "topics" },
      filter: { field: "reviewed", operator: "eq", value: true },
    });

    expect(result.filteredDocuments).toBe(1);
    expect(valuesOf(result.keys, "topics")).toEqual([["sqlite", 1], ["typescript", 1]]);
  });

  test("rejects conditions that have no meaning for a metadata entry", () => {
    const cases: [unknown, RegExp][] = [
      [{ field: "topics", operator: "eq", value: "x" }, /^Invalid metadata match at \$: 'topics' is not a field of a metadata entry, expected 'key' or 'value'/],
      [{ field: "value", operator: "exists", value: true }, /'exists' has no meaning for a single metadata entry/],
      [{ field: "value", operator: "all", value: ["a"] }, /'all' has no meaning for a single metadata entry/],
      [{ operator: "and", operands: [{ field: "key", operator: "eq", value: "a" }, { field: "nope", operator: "eq", value: 1 }] }, /at \$\.operands\[1\]:/],
      [{ field: "value", operator: "eq", value: 3, caseInsensitive: true }, /^Invalid metadata match at \$\.caseInsensitive:/],
    ];

    for (const [match, expected] of cases) {
      expect(() => listMetadata(store.db, { match: match as MetadataMatch })).toThrow(expected);
      expect(() => listMetadata(store.db, { match: match as MetadataMatch })).toThrow(MetadataFilterError);
    }
  });
});

describe("listMetadata key window", () => {
  // Coverage: e 5, d 4, c 3, b 2, a 1. Report order is e, d, c, b, a.
  beforeEach(async () => {
    const keyNames = ["a", "b", "c", "d", "e"];
    for (const [index, key] of keyNames.entries()) {
      for (let count = 0; count <= index; count += 1) await insertMetadataDoc("notes", { [key]: "x" });
    }
  });

  function keysOf(options: ListMetadataOptions = {}): string[] {
    return listMetadata(store.db, options).keys.map(summary => summary.key);
  }

  test("reports every key with no remainder when the vocabulary fits the window", () => {
    const result = listMetadata(store.db);

    expect(result.keys.map(summary => summary.key)).toEqual(["e", "d", "c", "b", "a"]);
    expect(result.totalKeys).toBe(5);
    expect(result.remainingKeys).toBe(0);
  });

  test("keyLimit windows keys in report order and reports the exact remainder", () => {
    const result = listMetadata(store.db, { keyLimit: 2 });

    expect(result.keys.map(summary => summary.key)).toEqual(["e", "d"]);
    expect(result.totalKeys).toBe(5);
    expect(result.remainingKeys).toBe(3);
  });

  test("keyOffset pages through the vocabulary without gaps or repeats", () => {
    const pages = [0, 2, 4].map(keyOffset => listMetadata(store.db, { keyLimit: 2, keyOffset }));

    expect(pages.map(page => page.keys.map(summary => summary.key))).toEqual([["e", "d"], ["c", "b"], ["a"]]);
    expect(pages.map(page => page.remainingKeys)).toEqual([3, 1, 0]);
    expect(pages.every(page => page.totalKeys === 5)).toBe(true);
  });

  test("an offset past the end reports no keys and keeps the total", () => {
    const result = listMetadata(store.db, { keyOffset: 10 });

    expect(result.keys).toEqual([]);
    expect(result.totalKeys).toBe(5);
    expect(result.remainingKeys).toBe(0);
  });

  test("an infinite keyLimit removes the window", () => {
    expect(keysOf({ keyLimit: Infinity })).toEqual(["e", "d", "c", "b", "a"]);
  });

  test("the window applies to the keys the match admits", () => {
    const result = listMetadata(store.db, { match: { field: "key", operator: "in", value: ["a", "c", "e"] }, keyLimit: 2 });

    expect(result.keys.map(summary => summary.key)).toEqual(["e", "c"]);
    expect(result.totalKeys).toBe(3);
    expect(result.remainingKeys).toBe(1);
  });

  test("a key summary inside the window is complete", async () => {
    await insertMetadataDoc("work", { e: ["y", "z"], d: 1 });

    const result = listMetadata(store.db, { keyLimit: 1 });
    const e = summaryOf(result.keys, "e");

    expect(e.documents).toBe(6);
    expect(e.types[0]!.multiValued).toBe(true);
    expect(e.types[0]!.distinctValues).toBe(3);
    expect(e.types[0]!.collections).toEqual(["notes", "work"]);
    expect(result.keys).toHaveLength(1);
  });

  test("collection provenance uses UTF-8 ordering without SQLite 3.44 aggregate syntax", async () => {
    for (const collection of ["𐀀", "", "notes", "B"]) {
      await insertMetadataDoc(collection, { provenance: "shared" });
    }
    const observed = observeStatements(store.db, sql => {
      expect(sql).not.toMatch(/json_group_array\([^)]*\bORDER\s+BY/iu);
    });
    const report = listMetadata(observed, { match: { field: "key", operator: "eq", value: "provenance" } });
    expect(summaryOf(report.keys, "provenance").types[0]!.collections).toEqual(["B", "notes", "", "𐀀"]);
  });

  test("pages are prefixes of the full result under one collation, on both readers", async () => {
    // Equal coverage, so order falls to the key name. BINARY collation puts
    // uppercase before "_", "_" before lowercase, and lowercase before "é".
    await insertMetadataDoc("work", { _id: 1, a: 1, B: 1, "é": 1, Z: 1 });
    const names = ["B", "Z", "_id", "a", "é"];

    const fullResult = listMetadata(store.db, { collection: "work" }).keys.map(summary => summary.key);
    const pages = [0, 2, 4].flatMap(keyOffset =>
      listMetadata(store.db, { collection: "work", keyLimit: 2, keyOffset }).keys.map(summary => summary.key));

    expect(fullResult).toEqual(names);
    expect(pages).toEqual(names);
    expect(listMetadataCollectionSummaries(store.db).get("work")!.keys.map(overview => overview.key)).toEqual(names);
    expect(listMetadataCollectionSummaries(store.db, 3).get("work")!.keys.map(overview => overview.key)).toEqual(names.slice(0, 3));
  });

  test("ranks keys once per report and groups values once for both totals and windows", () => {
    const statements: string[] = [];
    const observed = observeStatements(store.db, sql => statements.push(sql));
    const options: ListMetadataOptions[] = [
      { keyLimit: 5, valueLimit: 3 },
      { collection: "notes", keyLimit: 5, valueLimit: 3 },
      { filter: { field: "e", operator: "prefix", value: "x" }, keyLimit: 5 },
      {},
    ];
    // One statement ranks the keys, but the planner may still evaluate its
    // CTE once per joined row unless it is materialized. Check both, with
    // and without table statistics, which change the planner's choice.
    for (const statistics of [false, true]) {
      if (statistics) store.db.exec("ANALYZE");
      for (const option of options) {
        statements.length = 0;
        listMetadata(observed, option);
        const rankingStatements = statements.filter(sql => sql.includes("selected_keys AS"));
        expect(rankingStatements).toHaveLength(1);
        expect(statements.filter(sql => sql.includes("GROUP BY mv.key, mv.value_type, mv.text_value"))).toHaveLength(1);
        if (option.filter) expect(statements.some(sql => sql.includes("d.id IN (SELECT mv.document_id"))).toBe(true);
        const plan = planOf(rankingStatements[0]!);
        expect(plan.some(step => step.includes("MATERIALIZE selected_keys")), rankingStatements[0]).toBe(true);
        expect(plan.some(step => step.includes("CO-ROUTINE selected_keys")), rankingStatements[0]).toBe(false);
      }
    }

    statements.length = 0;
    listMetadataCollectionSummaries(observed, 10);
    expect(statements).toHaveLength(1);
    expect(statements[0]).toContain("mv.ordinal = 0");
    expect(statements[0]).not.toContain("number_value");
  });
});

describe("metadata overviews", () => {
  test("batched collection overviews agree with full reports through lifecycle changes", async () => {
    await insertMetadataDoc("notes", { tags: ["a", "b", "c"], priority: 1 });
    const changed = await insertMetadataDoc("notes", { tags: 7, priority: "high" });
    await insertMetadataDoc("work", { tags: true, "é": "x", "😀": "x" });
    const stale = await insertMetadataDoc("work", { old: "hidden" });
    const inactive = await insertMetadataDoc("work", { deleted: "hidden" });
    store.db.prepare("UPDATE document_metadata SET extraction_version = 0 WHERE document_id = ?").run(stale);
    store.db.prepare("UPDATE documents SET active = 0 WHERE id = ?").run(inactive);

    for (const phase of ["initial", "replaced", "renamed", "deleted"] as const) {
      if (phase === "replaced") replaceDocumentMetadata(store.db, changed, { metadata: { tags: [true, false] }, extractionVersion: METADATA_EXTRACTION_VERSION });
      if (phase === "renamed") store.db.prepare("UPDATE documents SET collection = 'renamed' WHERE collection = 'notes'").run();
      if (phase === "deleted") store.db.prepare("DELETE FROM documents WHERE id = ?").run(changed);
      for (const keyLimit of [1, 10, Infinity]) {
        const statements: string[] = [];
        const observed = observeStatements(store.db, sql => statements.push(sql));
        const overviews = listMetadataCollectionSummaries(observed, keyLimit);
        expect(statements).toHaveLength(1);
        for (const collection of ["notes", "renamed", "work", "missing"]) {
          const report = listMetadata(store.db, { collection, keyLimit });
          const expected = { totalKeys: report.totalKeys, keys: report.keys.map(key => ({ key: key.key, documents: key.documents, types: key.types.map(type => type.type) })) };
          expect(overviews.get(collection) ?? { totalKeys: 0, keys: [] }).toEqual(expected);
        }
      }
    }
  });
});

describe("entry text predicates", () => {
  test("positive, negated, and compound matches partition empty and nonempty strings", async () => {
    const values = ["", "draft", "PUBLISHED", "abc\u0000XYZ", "\u0000", "😀XYZ", 7, true];
    for (const value of values) await insertMetadataDoc("notes", { label: value });

    for (const operator of ["contains", "prefix", "suffix"] as const) {
      for (const operand of ["ft", "PUB", "XYZ", "\u0000", "😀", "nomatch"]) {
        for (const caseInsensitive of [false, true]) {
          const fold = (text: string) => caseInsensitive ? text.replace(/[A-Z]/gu, letter => letter.toLowerCase()) : text;
          const expected = values.map(value => {
            if (typeof value !== "string") return false;
            const text = fold(value);
            const needle = fold(operand);
            return operator === "contains" ? text.includes(needle) : operator === "prefix" ? text.startsWith(needle) : text.endsWith(needle);
          });
          const condition = { field: "value", operator, value: operand, caseInsensitive } as const;
          const predicates: MetadataMatch[] = [
            condition,
            { operator: "not", operand: condition },
            { operator: "and", operands: [{ field: "key", operator: "eq", value: "label" }, { operator: "not", operand: condition }] },
            { operator: "or", operands: [condition, { operator: "not", operand: condition }] },
          ];
          const expectations = [expected, expected.map(value => !value), expected.map(value => !value), values.map(() => true)];
          for (const [index, predicate] of predicates.entries()) {
            const compiled = compileMetadataMatch(predicate, "mv");
            const rows = store.db.prepare(`SELECT ${compiled.sql} AS matched FROM document_metadata_values mv ORDER BY document_id`).all(...compiled.params) as { matched: number }[];
            expect(rows.map(row => row.matched)).toEqual(expectations[index]!.map(Number));
            const report = listMetadata(store.db, { match: predicate, valueLimit: Infinity });
            expect(report.keys[0]?.documents ?? 0).toBe(expectations[index]!.filter(Boolean).length);
          }

          const filter: MetadataFilter = { operator: "not", operand: { ...condition, field: "label" } };
          for (const scope of ["candidates", "corpus"] as const) {
            const compiled = compileMetadataFilter(filter, "d", scope);
            const rows = store.db.prepare(`SELECT ${compiled.sql} AS matched FROM documents d ORDER BY id`).all(...compiled.params) as { matched: number }[];
            expect(rows.map(row => row.matched)).toEqual(expected.map(value => Number(!value)));
          }
        }
      }
    }
  });
});

describe("listMetadata consistency", () => {
  test("reads the whole report from one snapshot while another connection writes", async () => {
    const documentId = await insertMetadataDoc("notes", { priority: 1 });
    const writer = createStore(store.db.prepare("PRAGMA database_list").all().map(row => (row as { file: string }).file)[0]!);

    try {
      let rewritten = false;
      const observed = observeStatements(store.db, sql => {
        // Commit a change from the second connection after the type statistics
        // (min and max) have been read and before the medians and values are,
        // the interleaving that would report a range no document ever had.
        if (rewritten || !sql.includes("PARTITION BY mv.key ORDER BY mv.number_value")) return;
        rewritten = true;
        replaceDocumentMetadata(writer.db, documentId, { metadata: { priority: 100 }, extractionVersion: METADATA_EXTRACTION_VERSION });
      });

      const priority = summaryOf(listMetadata(observed).keys, "priority").types[0]!;

      expect(rewritten).toBe(true);
      expect(priority.range).toEqual({ min: 1, median: 1, max: 1 });
      expect(priority.values).toEqual([{ value: 1, documents: 1 }]);
      expect(summaryOf(listMetadata(store.db).keys, "priority").types[0]!.values).toEqual([{ value: 100, documents: 1 }]);
    } finally {
      writer.close();
    }
  });
});

describe("listMetadata options", () => {
  test("rejects a window or ordering option outside its domain", () => {
    const cases: [ListMetadataOptions, RegExp][] = [
      [{ keyLimit: 0 }, /^Invalid keyLimit: expected a positive integer or Infinity, received 0$/],
      [{ keyLimit: 1.5 }, /^Invalid keyLimit: expected a positive integer or Infinity, received 1.5$/],
      [{ keyOffset: -1 }, /^Invalid keyOffset: expected a non-negative integer, received -1$/],
      [{ valueLimit: -Infinity }, /^Invalid valueLimit: expected a positive integer or Infinity, received -Infinity$/],
      [{ valueOffset: 0.5 }, /^Invalid valueOffset: expected a non-negative integer, received 0.5$/],
      [{ minCount: 0 }, /^Invalid minCount: expected a positive integer, received 0$/],
      [{ sort: "size" as "count" }, /^Invalid sort: expected 'count' or 'value', received "size"$/],
    ];

    for (const [options, expected] of cases) {
      expect(() => listMetadata(store.db, options)).toThrow(expected);
      expect(() => listMetadata(store.db, options)).toThrow(MetadataOptionError);
    }
  });

  test("names the offending option on the error", () => {
    try {
      listMetadata(store.db, { keyOffset: -1 });
      throw new Error("expected listMetadata to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(MetadataOptionError);
      expect((error as MetadataOptionError).option).toBe("keyOffset");
    }
  });

  test("rejects integers SQLite cannot bind as INTEGER, and binds the largest it can", async () => {
    await insertMetadataDoc("notes", { status: "x" });
    const largest = Number.MAX_SAFE_INTEGER;

    expect(() => listMetadata(store.db, { keyLimit: 1e30 })).toThrow(MetadataOptionError);
    expect(() => listMetadata(store.db, { keyOffset: 2 ** 53 })).toThrow(/^Invalid keyOffset: expected a non-negative integer, received 9007199254740992$/);
    expect(() => listMetadata(store.db, { valueOffset: 1e30 })).toThrow(MetadataOptionError);
    expect(() => listMetadata(store.db, { minCount: 1e30 })).toThrow(MetadataOptionError);

    expect(listMetadata(store.db, { keyLimit: largest, keyOffset: largest }).keys).toEqual([]);
    expect(summaryOf(listMetadata(store.db, { valueLimit: largest, valueOffset: largest }).keys, "status").types[0]!.values).toEqual([]);
    expect(summaryOf(listMetadata(store.db, { valueLimit: largest }).keys, "status").types[0]!.values).toHaveLength(1);
  });

  test("the production value window includes its exact integer endpoint past 2^53", async () => {
    await insertMetadataDoc("notes", { label: "x" });
    let windowChecked = false;
    const observed = observeStatements(store.db, () => {}, (sql, params) => {
      const windowSql = /\((rank > \? AND rank <= .*)\) AS in_window/u.exec(sql)?.[1];
      if (!windowSql) return;
      windowChecked = true;

      // Execute the actual emitted predicate and bindings on a tiny exact-rank
      // fixture. Reading the rank itself as a JS number would round it again.
      const rows = store.db.prepare(`
        WITH ranks(rank) AS (VALUES (9007199254740991), (9007199254740992), (9007199254740993), (9007199254740994))
        SELECT CAST(rank AS TEXT) AS rank FROM ranks WHERE ${windowSql} ORDER BY rank
      `).all(...params.slice(-3));
      expect(rows).toEqual([{ rank: "9007199254740992" }, { rank: "9007199254740993" }]);
    });

    listMetadata(observed, { valueOffset: Number.MAX_SAFE_INTEGER, valueLimit: 2 });
    expect(windowChecked).toBe(true);
  });

  test("rejects a filter and match whose bindings together exceed the statement budget", async () => {
    await insertMetadataDoc("notes", { status: "x" });
    const members = Array.from({ length: 64 }, (_, index) => `v${index}`);
    // 7 groups of 31 conditions: 225 nodes, inside every parser limit.
    const wideFilter: MetadataFilter = {
      operator: "and",
      operands: Array.from({ length: 7 }, () => ({
        operator: "and" as const,
        operands: Array.from({ length: 31 }, () => ({ field: "status", operator: "all" as const, value: members })),
      })),
    };
    const wideMatch: MetadataMatch = {
      operator: "or",
      operands: Array.from({ length: 7 }, () => ({
        operator: "or" as const,
        operands: Array.from({ length: 31 }, () => ({ field: "value" as const, operator: "in" as const, value: members })),
      })),
    };

    // 217 conditions binding field and value per member: 27,776, under budget alone.
    expect(listMetadata(store.db, { filter: wideFilter }).totalKeys).toBe(0);
    expect(listMetadata(store.db, { match: wideMatch }).totalKeys).toBe(0);

    try {
      listMetadata(store.db, { filter: wideFilter, match: wideMatch });
      throw new Error("expected listMetadata to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(MetadataBindingBudgetError);
      const budgetError = error as MetadataBindingBudgetError;
      expect(budgetError.budget).toBe(METADATA_SQL_BINDING_BUDGET);
      expect(budgetError.bindings).toBeGreaterThan(METADATA_SQL_BINDING_BUDGET);
      expect(budgetError.message).toMatch(/^filter and match together bind \d+ SQL parameters, over the budget of 30000\. Shorten their membership lists\.$/);
    }
  });
});

describe("listMetadata value window", () => {
  beforeEach(async () => {
    const tags = ["a", "a", "a", "b", "b", "c", "d", "d", "d", "d"];
    for (const tag of tags) await insertMetadataDoc("notes", { tag });
  });

  test("defaults to count order with a stable tiebreak", () => {
    expect(valuesOf(listMetadata(store.db).keys, "tag")).toEqual([["d", 4], ["a", 3], ["b", 2], ["c", 1]]);
  });

  test("sort by value orders ascending", () => {
    expect(valuesOf(listMetadata(store.db, { sort: "value" }).keys, "tag")).toEqual([["a", 3], ["b", 2], ["c", 1], ["d", 4]]);
  });

  test("valueLimit windows values and reports the exact remainder", () => {
    const tag = summaryOf(listMetadata(store.db, { valueLimit: 2 }).keys, "tag").types[0]!;

    expect(tag.values.map(count => count.value)).toEqual(["d", "a"]);
    expect(tag.distinctValues).toBe(4);
    expect(tag.remainingValues).toBe(2);
  });

  test("an infinite valueLimit removes the window", () => {
    const tag = summaryOf(listMetadata(store.db, { valueLimit: Infinity }).keys, "tag").types[0]!;

    expect(tag.values).toHaveLength(4);
    expect(tag.remainingValues).toBe(0);
  });

  test("valueOffset pages through the values without gaps or repeats", () => {
    const pages = [0, 2, 4].map(valueOffset =>
      summaryOf(listMetadata(store.db, { valueLimit: 2, valueOffset }).keys, "tag").types[0]!);

    expect(pages.map(tag => tag.values.map(count => count.value))).toEqual([["d", "a"], ["b", "c"], []]);
    expect(pages.map(tag => tag.remainingValues)).toEqual([2, 0, 0]);
    expect(pages.every(tag => tag.distinctValues === 4)).toBe(true);
  });

  test("valueOffset follows the requested sort", () => {
    const tag = summaryOf(listMetadata(store.db, { sort: "value", valueLimit: 2, valueOffset: 1 }).keys, "tag").types[0]!;

    expect(tag.values.map(count => count.value)).toEqual(["b", "c"]);
  });

  test("minCount drops the tail from the values, distinct count, and remainder", () => {
    const tag = summaryOf(listMetadata(store.db, { minCount: 3, valueLimit: 1 }).keys, "tag").types[0]!;

    expect(tag.values).toEqual([{ value: "d", documents: 4 }]);
    expect(tag.distinctValues).toBe(2);
    expect(tag.remainingValues).toBe(1);
    // Coverage still counts every document holding the key.
    expect(tag.documents).toBe(10);
  });

  test("a minCount nothing meets leaves the key with an empty window", () => {
    const tag = summaryOf(listMetadata(store.db, { minCount: 99 }).keys, "tag").types[0]!;

    expect(tag.values).toEqual([]);
    expect(tag.distinctValues).toBe(0);
    expect(tag.remainingValues).toBe(0);
  });
});

describe("listMetadata numbers", () => {
  test("reports min, median, and max for an odd count", async () => {
    for (const priority of [5, 1, 3]) await insertMetadataDoc("notes", { priority });

    const priority = summaryOf(listMetadata(store.db).keys, "priority").types[0]!;

    expect(priority.range).toEqual({ min: 1, median: 3, max: 5 });
    expect(priority.values.map(count => count.value)).toEqual([1, 3, 5]);
  });

  test("averages the middle values for an even count", async () => {
    for (const priority of [1, 2, 3, 10]) await insertMetadataDoc("notes", { priority });

    expect(summaryOf(listMetadata(store.db).keys, "priority").types[0]!.range).toEqual({ min: 1, median: 2.5, max: 10 });
  });

  test("the even-count median is finite and correctly rounded across the double range", async () => {
    const cases: [number, number, number][] = [
      [1e308, 1.5e308, 1.25e308],
      [-1.5e308, -1e308, -1.25e308],
      [-1e308, 1e308, 0],
      [Number.MIN_VALUE, 2 * Number.MIN_VALUE, 2 * Number.MIN_VALUE],
      [-2 * Number.MIN_VALUE, -Number.MIN_VALUE, -2 * Number.MIN_VALUE],
      [Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE],
    ];

    for (const [index, [lower, upper]] of cases.entries()) {
      await insertMetadataDoc(`c${index}`, { n: lower });
      await insertMetadataDoc(`c${index}`, { n: upper });
    }

    for (const [index, [lower, upper, median]] of cases.entries()) {
      const n = summaryOf(listMetadata(store.db, { collection: `c${index}` }).keys, "n").types[0]!;
      expect(n.range, `${lower}, ${upper}`).toEqual({ min: lower, median, max: upper });
    }
  });

  test("median counts every value row, including array elements", async () => {
    await insertMetadataDoc("notes", { scores: [1, 1, 1] });
    await insertMetadataDoc("notes", { scores: [9] });

    const scores = summaryOf(listMetadata(store.db).keys, "scores").types[0]!;

    expect(scores.range).toEqual({ min: 1, median: 1, max: 9 });
    expect(scores.values).toEqual([{ value: 1, documents: 1 }, { value: 9, documents: 1 }]);
  });

  test("strings and booleans carry no range", async () => {
    await insertMetadataDoc("notes", { status: "x", reviewed: true });

    const result = listMetadata(store.db);

    expect(summaryOf(result.keys, "status").types[0]!.range).toBeUndefined();
    expect(summaryOf(result.keys, "reviewed").types[0]!.range).toBeUndefined();
  });
});

describe("listMetadata type conflicts and attribution", () => {
  test("splits a key by type and partitions its documents", async () => {
    await insertMetadataDoc("notes", { priority: 3 });
    await insertMetadataDoc("notes", { priority: 1 });
    await insertMetadataDoc("work", { priority: "high" });

    const priority = summaryOf(listMetadata(store.db).keys, "priority");

    expect(priority.documents).toBe(3);
    expect(priority.types.map(typeSummary => [typeSummary.type, typeSummary.documents])).toEqual([["number", 2], ["string", 1]]);
    expect(priority.types[0]!.range).toEqual({ min: 1, median: 2, max: 3 });
    expect(priority.types[0]!.collections).toEqual(["notes"]);
    expect(priority.types[1]!.collections).toEqual(["work"]);
    expect(priority.types[1]!.values).toEqual([{ value: "high", documents: 1 }]);
  });

  test("orders equally covered types by name", async () => {
    await insertMetadataDoc("notes", { flag: true });
    await insertMetadataDoc("notes", { flag: "yes" });

    expect(summaryOf(listMetadata(store.db).keys, "flag").types.map(typeSummary => typeSummary.type)).toEqual(["boolean", "string"]);
  });

  test("orders keys by coverage descending, then name", async () => {
    await insertMetadataDoc("notes", { zeta: 1, alpha: 1, mid: 1 });
    await insertMetadataDoc("notes", { zeta: 1, alpha: 1 });
    await insertMetadataDoc("notes", { zeta: 1 });

    expect(listMetadata(store.db).keys.map(summary => summary.key)).toEqual(["zeta", "alpha", "mid"]);
  });
});

describe("listMetadata agrees with filtered search", () => {
  test("every reported value is reachable through an eq filter under the same scope", async () => {
    await insertMetadataDoc("notes", { topics: ["a", "b"], priority: 3, reviewed: true, owner: "docs-team" });
    await insertMetadataDoc("notes", { topics: ["b"], priority: 1.5, reviewed: false });
    await insertMetadataDoc("work", { topics: ["c"], priority: 3 });
    const pendingId = await insertMetadataDoc("notes", { topics: ["ghost"] });
    store.db.prepare(`DELETE FROM document_metadata WHERE document_id = ?`).run(pendingId);

    const scopes: ListMetadataOptions[] = [{}, { collection: "notes" }, { collection: ["notes", "work"] }];
    for (const scope of scopes) {
      const result = listMetadata(store.db, { ...scope, valueLimit: Infinity });
      expect(result.keys.length).toBeGreaterThan(0);

      for (const summary of result.keys) {
        for (const typeSummary of summary.types) {
          for (const count of typeSummary.values) {
            const hits = searchFTS(store.db, "doc", 100, scope.collection, { field: summary.key, operator: "eq", value: count.value });
            expect(hits, `${summary.key} = ${String(count.value)}`).toHaveLength(count.documents);
          }
        }
      }
    }
  });
});

describe("status view helpers", () => {
  beforeEach(async () => {
    await insertMetadataDoc("notes", { status: "published", priority: 3 });
    await insertMetadataDoc("notes", { status: "draft" });
    await insertMetadataDoc("notes", {});
    await insertMetadataDoc("work", { priority: "high", source: "jira" });
    const pendingId = await insertMetadataDoc("work", { status: "ghost" });
    store.db.prepare(`DELETE FROM document_metadata WHERE document_id = ?`).run(pendingId);
  });

  test("listMetadataCollectionSummaries reports names, coverage, and types per collection in coverage order", () => {
    const overviews = listMetadataCollectionSummaries(store.db);
    expect(overviews.get("notes")).toEqual({ totalKeys: 2, keys: [
      { key: "status", documents: 2, types: ["string"] },
      { key: "priority", documents: 1, types: ["number"] },
    ] });
    expect(overviews.get("work")).toEqual({ totalKeys: 2, keys: [
      { key: "priority", documents: 1, types: ["string"] },
      { key: "source", documents: 1, types: ["string"] },
    ] });
    expect(overviews.has("missing")).toBe(false);

    const windowed = listMetadataCollectionSummaries(store.db, 1);
    expect(windowed.get("notes")).toEqual({ totalKeys: 2, keys: [{ key: "status", documents: 2, types: ["string"] }] });
    expect(listMetadataCollectionSummaries(store.db, Infinity).get("notes")!.keys).toHaveLength(2);
  });

  test("countMetadataKeys reports the vocabulary size in scope", () => {
    expect(countMetadataKeys(store.db)).toBe(3);
    expect(countMetadataKeys(store.db, ["work"])).toBe(2);
    expect(countMetadataKeys(store.db, ["missing"])).toBe(0);
  });

  test("countDocumentsWithMetadata counts extracted documents declaring a key", () => {
    expect(countDocumentsWithMetadata(store.db)).toBe(3);
    expect(countDocumentsWithMetadata(store.db, ["notes"])).toBe(2);
  });

  test("status batches metadata overviews and initialization reads no metadata values", () => {
    const statements: string[] = [];
    const observed = observeStatements(store.db, sql => statements.push(sql));
    const status = getStatus(observed);
    expect(status.collections).toHaveLength(2);
    expect(status.collections.find(collection => collection.name === "notes")?.metadataKeyCount).toBe(2);
    expect(statements.filter(sql => sql.includes("document_metadata_values"))).toHaveLength(1);

    statements.length = 0;
    const summary = getStatusSummary(observed);
    expect(summary.totalDocuments).toBe(status.totalDocuments);
    expect(summary.pendingMetadata).toBe(status.pendingMetadata);
    expect(summary.collections.map(collection => collection.name)).toEqual(status.collections.map(collection => collection.name));
    expect(summary.collections[0]).not.toHaveProperty("metadataKeys");
    expect(statements.some(sql => sql.includes("document_metadata_values"))).toBe(false);
  });

  test("countDocumentsPendingMetadata accepts a collection scope", () => {
    expect(countDocumentsPendingMetadata(store.db)).toBe(1);
    expect(countDocumentsPendingMetadata(store.db, ["notes"])).toBe(0);
    expect(countDocumentsPendingMetadata(store.db, ["work"])).toBe(1);
  });
});
