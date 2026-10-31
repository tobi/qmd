/**
 * metadata-filter.test.ts - Filter AST validation and parameterized SQL
 * compilation, including end-to-end predicate semantics against SQLite.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import {
  parseMetadataFilter,
  parseMetadataMatch,
  compileMetadataFilter,
  type FilterScope,
  compileMetadataMatch,
  MetadataFilterError,
  METADATA_FILTER_LIMITS,
  type MetadataFilter,
} from "../src/metadata-filter.js";
import { initializeMetadataSchema, replaceDocumentMetadata } from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";

// =============================================================================
// Validation
// =============================================================================

describe("parseMetadataFilter", () => {
  test("accepts every condition operator shape", () => {
    const conditions: unknown[] = [
      { field: "status", operator: "eq", value: "published" },
      { field: "status", operator: "ne", value: "draft" },
      { field: "priority", operator: "gt", value: 3 },
      { field: "priority", operator: "gte", value: 3 },
      { field: "priority", operator: "lt", value: 10 },
      { field: "name", operator: "lte", value: "m" },
      { field: "topics", operator: "in", value: ["a", "b"] },
      { field: "topics", operator: "nin", value: [1, 2] },
      { field: "flags", operator: "all", value: [true, false] },
      { field: "status", operator: "exists", value: false },
      { field: "topics", operator: "contains", value: "vec" },
      { field: "topics", operator: "prefix", value: "sql" },
      { field: "owner", operator: "suffix", value: "-team" },
      { field: "priority", operator: "type", value: "number" },
    ];
    for (const condition of conditions) {
      expect(parseMetadataFilter(condition)).toEqual(condition);
    }
  });

  test("accepts caseInsensitive on string operands only", () => {
    const accepted: unknown[] = [
      { field: "status", operator: "eq", value: "Published", caseInsensitive: true },
      { field: "status", operator: "ne", value: "Draft", caseInsensitive: false },
      { field: "name", operator: "gt", value: "M", caseInsensitive: true },
      { field: "topics", operator: "in", value: ["SQL", "TypeScript"], caseInsensitive: true },
      { field: "topics", operator: "all", value: ["SQL"], caseInsensitive: true },
      { field: "topics", operator: "contains", value: "SQL", caseInsensitive: true },
      { field: "topics", operator: "prefix", value: "SQL", caseInsensitive: true },
      { field: "topics", operator: "suffix", value: "SQL", caseInsensitive: true },
    ];
    for (const condition of accepted) {
      expect(parseMetadataFilter(condition)).toEqual(condition);
    }

    const rejected: [unknown, RegExp][] = [
      [{ field: "priority", operator: "eq", value: 3, caseInsensitive: true }, /at \$\.caseInsensitive:.*string values only/],
      [{ field: "reviewed", operator: "in", value: [true], caseInsensitive: true }, /string values only/],
      [{ field: "reviewed", operator: "exists", value: true, caseInsensitive: true }, /does not apply to 'exists'/],
      [{ field: "priority", operator: "type", value: "number", caseInsensitive: true }, /does not apply to 'type'/],
      [{ field: "status", operator: "eq", value: "x", caseInsensitive: "yes" }, /must be a boolean/],
    ];
    for (const [input, expected] of rejected) {
      expect(() => parseMetadataFilter(input)).toThrow(expected);
    }
  });

  test("accepts nested groups and negation", () => {
    const filter = {
      operator: "and",
      operands: [
        { field: "topics", operator: "all", value: ["typescript", "programming"] },
        {
          operator: "or",
          operands: [
            { field: "status", operator: "eq", value: "published" },
            { operator: "not", operand: { field: "audience", operator: "eq", value: "internal" } },
          ],
        },
      ],
    };
    expect(parseMetadataFilter(filter)).toEqual(filter);
  });

  test("canonicalizes membership arrays by de-duplicating", () => {
    const parsed = parseMetadataFilter({ field: "topics", operator: "in", value: ["a", "b", "a"] });
    expect(parsed).toEqual({ field: "topics", operator: "in", value: ["a", "b"] });
  });

  test("rejects invalid node shapes with the failing JSON path", () => {
    const cases: [unknown, RegExp][] = [
      ["not-an-object", /at \$:.*must be an object/],
      [{ field: "a", value: 1 }, /at \$:.*missing 'operator'/],
      [{ operator: "equal", field: "a", value: 1 }, /unknown operator 'equal'/],
      [{ operator: "and", operands: [] }, /non-empty 'operands'/],
      [{ operator: "and", operands: "nope" }, /'operands' array/],
      [{ operator: "not", operands: [{ field: "a", operator: "eq", value: 1 }] }, /unknown property 'operands'/],
      [{ operator: "not" }, /exactly one 'operand'/],
      [{ operator: "and", operands: [{ operator: "eq" }] }, /at \$\.operands\[0\]/],
      [{ operator: "eq", value: 1 }, /non-empty string 'field'/],
      [{ operator: "eq", field: "a" }, /requires a 'value'/],
      [{ operator: "eq", field: "a", value: 1, extra: true }, /unknown property 'extra'/],
      [{ operator: "and", operands: [{ field: "a", operator: "eq", value: 1 }], field: "a" }, /unknown property 'field'/],
      [{ operator: "eq", key: "a", value: 1 }, /unknown property 'key'/],
      [{ operator: "gt", field: "a", value: true }, /string or number value/],
      [{ operator: "eq", field: "a", value: NaN }, /finite/],
      [{ operator: "eq", field: "a", value: { nested: 1 } }, /string, number, or boolean/],
      [{ operator: "in", field: "a", value: "x" }, /array value/],
      [{ operator: "in", field: "a", value: [] }, /non-empty array/],
      [{ operator: "in", field: "a", value: [1, "two"] }, /homogeneous array/],
      [{ operator: "exists", field: "a", value: "yes" }, /boolean value/],
      [{ operator: "contains", field: "a", value: "" }, /at \$\.value:.*'contains' requires a non-empty string/],
      [{ operator: "prefix", field: "a", value: 3 }, /'prefix' requires a non-empty string/],
      [{ operator: "suffix", field: "a", value: ["x"] }, /string, number, or boolean/],
      [{ operator: "type", field: "a", value: "integer" }, /at \$\.value:.*'type' requires one of: string, number, boolean/],
      [{ operator: "type", field: "a", value: 1 }, /'type' requires one of/],
    ];

    for (const [input, expected] of cases) {
      expect(() => parseMetadataFilter(input)).toThrow(expected);
      expect(() => parseMetadataFilter(input)).toThrow(MetadataFilterError);
    }
  });

  test("rejects excessive depth, node count, and operand count", () => {
    let deepFilter: unknown = { field: "a", operator: "eq", value: 1 };
    for (let i = 0; i <= METADATA_FILTER_LIMITS.maxDepth; i++) {
      deepFilter = { operator: "not", operand: deepFilter };
    }
    expect(() => parseMetadataFilter(deepFilter)).toThrow(/nesting depth/);

    const condition = { field: "a", operator: "eq", value: 1 };
    const wideGroup = {
      operator: "or",
      operands: Array.from({ length: METADATA_FILTER_LIMITS.maxGroupOperands + 1 }, () => condition),
    };
    expect(() => parseMetadataFilter(wideGroup)).toThrow(/operands/);

    const manyNodes = {
      operator: "and",
      operands: Array.from({ length: METADATA_FILTER_LIMITS.maxGroupOperands }, () => ({
        operator: "and",
        operands: Array.from({ length: METADATA_FILTER_LIMITS.maxGroupOperands }, () => condition),
      })),
    };
    expect(() => parseMetadataFilter(manyNodes)).toThrow(/nodes/);

    const manyValues = {
      field: "a",
      operator: "in",
      value: Array.from({ length: METADATA_FILTER_LIMITS.maxMembershipValues + 1 }, (_, i) => i),
    };
    expect(() => parseMetadataFilter(manyValues)).toThrow(/values/);
  });
});

describe("parseMetadataMatch", () => {
  test("accepts the filter grammar with 'key' and 'value' as the condition fields", () => {
    const match = {
      operator: "and",
      operands: [
        { field: "key", operator: "prefix", value: "mem-" },
        { operator: "or", operands: [
          { field: "value", operator: "gte", value: 3 },
          { field: "value", operator: "type", value: "boolean" },
          { operator: "not", operand: { field: "value", operator: "in", value: ["Draft"], caseInsensitive: true } },
        ] },
      ],
    };
    expect(parseMetadataMatch(match)).toEqual(match);
  });

  test("rejects other fields and the two set operators, naming the match", () => {
    const cases: [unknown, RegExp][] = [
      [{ field: "topics", operator: "eq", value: "x" }, /^Invalid metadata match at \$: 'topics' is not a field of a metadata entry, expected 'key' or 'value'$/],
      [{ field: "value", operator: "exists", value: true }, /^Invalid metadata match at \$: 'exists' has no meaning for a single metadata entry$/],
      [{ field: "key", operator: "all", value: ["a"] }, /'all' has no meaning for a single metadata entry/],
      [{ operator: "not", operand: { field: "value", operator: "eq" } }, /^Invalid metadata match at \$\.operand: 'eq' requires a 'value'$/],
      ["nope", /^Invalid metadata match at \$: each filter node must be an object$/],
    ];
    for (const [input, expected] of cases) {
      expect(() => parseMetadataMatch(input)).toThrow(expected);
      expect(() => parseMetadataMatch(input)).toThrow(MetadataFilterError);
    }
    // The filter keeps its own name.
    expect(() => parseMetadataFilter("nope")).toThrow(/^Invalid metadata filter at \$:/);
  });
});

describe("compileMetadataMatch", () => {
  test("compiles to a predicate over the row alias with every operand bound", () => {
    const compiled = compileMetadataMatch(parseMetadataMatch({
      operator: "and",
      operands: [
        { field: "key", operator: "eq", value: "k'; --" },
        { field: "value", operator: "suffix", value: "V'; --", caseInsensitive: true },
        { field: "value", operator: "nin", value: [1, 2] },
      ],
    }), "row");

    expect(compiled.sql).not.toContain("'; --");
    expect(compiled.sql).not.toContain("EXISTS");
    expect(compiled.sql).toContain("row.key = ?");
    expect(compiled.sql).toContain("lower(row.text_value)");
    expect(compiled.sql).toContain("row.number_value NOT IN (?, ?)");
    // The suffix binds its operand's UTF-8 byte length ahead of the operand.
    expect(compiled.params).toEqual(["k'; --", 6, "v'; --", 1, 2]);
  });
});

// =============================================================================
// SQL compilation and semantics
// =============================================================================

describe("compileMetadataFilter semantics", () => {
  let db: Database;

  beforeEach(() => {
    db = openDatabase(":memory:");
    db.exec(`
      CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
      )
    `);
    initializeMetadataSchema(db);
  });

  afterEach(() => {
    db.close();
  });

  function insertDoc(path: string, metadata: DocumentMetadata): number {
    const result = db.prepare(`INSERT INTO documents (path) VALUES (?)`).run(path);
    const documentId = Number(result.lastInsertRowid);
    replaceDocumentMetadata(db, documentId, { metadata, extractionVersion: METADATA_EXTRACTION_VERSION });
    return documentId;
  }

  function matchPathsIn(filter: MetadataFilter, scope: FilterScope): string[] {
    const compiled = compileMetadataFilter(parseMetadataFilter(filter), "d", scope);
    const rows = db.prepare(`
      SELECT d.path FROM documents d
      JOIN document_metadata dm ON dm.document_id = d.id
        AND dm.extraction_version = ${METADATA_EXTRACTION_VERSION}
        AND dm.extraction_error IS NULL
      WHERE d.active = 1 AND ${compiled.sql}
      ORDER BY d.path
    `).all(...compiled.params) as { path: string }[];
    return rows.map(row => row.path);
  }

  /** Both scopes are one semantics in two SQL forms, so every case checks they agree. */
  function matchPaths(filter: MetadataFilter): string[] {
    const candidates = matchPathsIn(filter, "candidates");
    expect(matchPathsIn(filter, "corpus"), `corpus scope for ${JSON.stringify(filter)}`).toEqual(candidates);
    return candidates;
  }

  test("eq matches each scalar type exactly, without coercion", () => {
    insertDoc("str.md", { status: "published" });
    insertDoc("num.md", { status: 1 });
    insertDoc("bool.md", { status: true });

    expect(matchPaths({ field: "status", operator: "eq", value: "published" })).toEqual(["str.md"]);
    expect(matchPaths({ field: "status", operator: "eq", value: 1 })).toEqual(["num.md"]);
    expect(matchPaths({ field: "status", operator: "eq", value: true })).toEqual(["bool.md"]);
    expect(matchPaths({ field: "status", operator: "eq", value: "1" })).toEqual([]);
  });

  test("ne requires key presence and matching type", () => {
    insertDoc("draft.md", { status: "draft" });
    insertDoc("published.md", { status: "published" });
    insertDoc("missing.md", { other: "x" });
    insertDoc("typed.md", { status: 1 });

    expect(matchPaths({ field: "status", operator: "ne", value: "draft" })).toEqual(["published.md"]);
  });

  test("ordered comparisons match numbers and binary-ordered strings", () => {
    insertDoc("low.md", { priority: 1 });
    insertDoc("mid.md", { priority: 3 });
    insertDoc("high.md", { priority: 7 });
    insertDoc("alpha.md", { name: "alpha" });
    insertDoc("zulu.md", { name: "zulu" });

    expect(matchPaths({ field: "priority", operator: "gte", value: 3 })).toEqual(["high.md", "mid.md"]);
    expect(matchPaths({ field: "priority", operator: "lt", value: 3 })).toEqual(["low.md"]);
    expect(matchPaths({ field: "name", operator: "gt", value: "alpha" })).toEqual(["zulu.md"]);
    // Type mismatch: no string 'priority' values exist.
    expect(matchPaths({ field: "priority", operator: "gte", value: "3" })).toEqual([]);
  });

  test("in, nin, and all evaluate membership over value sets", () => {
    insertDoc("ts.md", { topics: ["typescript", "programming"] });
    insertDoc("go.md", { topics: ["go"] });
    insertDoc("none.md", { other: "x" });

    expect(matchPaths({ field: "topics", operator: "in", value: ["typescript", "rust"] })).toEqual(["ts.md"]);
    expect(matchPaths({ field: "topics", operator: "nin", value: ["typescript", "rust"] })).toEqual(["go.md"]);
    expect(matchPaths({ field: "topics", operator: "all", value: ["typescript", "programming"] })).toEqual(["ts.md"]);
    expect(matchPaths({ field: "topics", operator: "all", value: ["typescript", "rust"] })).toEqual([]);
  });

  test("exists matches presence and absence", () => {
    insertDoc("has.md", { status: "ok" });
    insertDoc("hasnt.md", { other: "x" });

    expect(matchPaths({ field: "status", operator: "exists", value: true })).toEqual(["has.md"]);
    expect(matchPaths({ field: "status", operator: "exists", value: false })).toEqual(["hasnt.md"]);
  });

  test("and, or, and not compose recursively", () => {
    insertDoc("a.md", { status: "published", priority: 5 });
    insertDoc("b.md", { status: "published", priority: 1 });
    insertDoc("c.md", { status: "draft", priority: 9 });

    expect(matchPaths({
      operator: "and",
      operands: [
        { field: "status", operator: "eq", value: "published" },
        { field: "priority", operator: "gte", value: 3 },
      ],
    })).toEqual(["a.md"]);

    expect(matchPaths({
      operator: "or",
      operands: [
        { field: "priority", operator: "gte", value: 9 },
        { field: "priority", operator: "lte", value: 1 },
      ],
    })).toEqual(["b.md", "c.md"]);

    expect(matchPaths({
      operator: "not",
      operand: { field: "status", operator: "eq", value: "draft" },
    })).toEqual(["a.md", "b.md"]);
  });

  test("independent conditions on one array key match different elements", () => {
    insertDoc("wide.md", { priority: [1, 20] });
    insertDoc("narrow.md", { priority: [5] });

    const rangeFilter: MetadataFilter = {
      operator: "and",
      operands: [
        { field: "priority", operator: "gte", value: 3 },
        { field: "priority", operator: "lt", value: 10 },
      ],
    };
    // Document-level semantics: [1, 20] satisfies both conditions via
    // different elements — no implicit same-element fusion.
    expect(matchPaths(rangeFilter)).toEqual(["narrow.md", "wide.md"]);
  });

  test("text operators match any string element and never a number or boolean", () => {
    insertDoc("sqlite.md", { topics: ["sqlite", "search"] });
    insertDoc("vec.md", { topics: ["sqlite-vec", "embeddings"] });
    insertDoc("pg.md", { topics: "postgres" });
    insertDoc("num.md", { topics: 42 });
    insertDoc("bool.md", { topics: true });

    expect(matchPaths({ field: "topics", operator: "prefix", value: "sql" })).toEqual(["sqlite.md", "vec.md"]);
    expect(matchPaths({ field: "topics", operator: "suffix", value: "vec" })).toEqual(["vec.md"]);
    expect(matchPaths({ field: "topics", operator: "contains", value: "arch" })).toEqual(["sqlite.md"]);
    expect(matchPaths({ field: "topics", operator: "contains", value: "4" })).toEqual([]);
    expect(matchPaths({ field: "topics", operator: "prefix", value: "tru" })).toEqual([]);
  });

  test("prefix and suffix compare whole characters and never overrun the value", () => {
    insertDoc("exact.md", { code: "abc" });
    insertDoc("emoji.md", { code: "\u{1F600}abc" });

    // The operand equal to the value is both its prefix and its suffix.
    expect(matchPaths({ field: "code", operator: "prefix", value: "abc" })).toEqual(["exact.md"]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "abc" })).toEqual(["emoji.md", "exact.md"]);
    // An operand longer than the value cannot match either end.
    expect(matchPaths({ field: "code", operator: "prefix", value: "abcd" })).toEqual([]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "zabc" })).toEqual([]);
    // Astral characters count as one character on both sides.
    expect(matchPaths({ field: "code", operator: "prefix", value: "\u{1F600}a" })).toEqual(["emoji.md"]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "\u{1F600}abc" })).toEqual(["emoji.md"]);
  });

  test("text operators see the whole value past an embedded NUL", () => {
    insertDoc("nul.md", { code: "abc\u0000XYZ" });
    insertDoc("plain.md", { code: "abc" });

    expect(matchPaths({ field: "code", operator: "suffix", value: "XYZ" })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "abc" })).toEqual(["plain.md"]);
    expect(matchPaths({ field: "code", operator: "prefix", value: "abc\u0000XYZ" })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "abc\u0000XYZ" })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "prefix", value: "abc\u0000" })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "contains", value: "\u0000X" })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "suffix", value: "xyz", caseInsensitive: true })).toEqual(["nul.md"]);
    expect(matchPaths({ field: "code", operator: "eq", value: "abc\u0000XYZ" })).toEqual(["nul.md"]);
  });

  test("type matches the stored type of a key's values", () => {
    insertDoc("num.md", { priority: 3 });
    insertDoc("nums.md", { priority: [1, 2] });
    insertDoc("str.md", { priority: "high" });
    insertDoc("bool.md", { priority: true });
    insertDoc("none.md", { other: 1 });

    expect(matchPaths({ field: "priority", operator: "type", value: "number" })).toEqual(["num.md", "nums.md"]);
    expect(matchPaths({ field: "priority", operator: "type", value: "string" })).toEqual(["str.md"]);
    expect(matchPaths({ field: "priority", operator: "type", value: "boolean" })).toEqual(["bool.md"]);
    expect(matchPaths({
      operator: "not",
      operand: { field: "priority", operator: "type", value: "string" },
    })).toEqual(["bool.md", "none.md", "num.md", "nums.md"]);
  });

  test("caseInsensitive folds ASCII letters on both sides for every string operator", () => {
    insertDoc("upper.md", { status: "PUBLISHED", topics: ["PostgreSQL", "MySQL"] });
    insertDoc("lower.md", { status: "published", topics: ["sqlite"] });
    insertDoc("draft.md", { status: "Draft", topics: ["Search"] });

    expect(matchPaths({ field: "status", operator: "eq", value: "Published" })).toEqual([]);
    expect(matchPaths({ field: "status", operator: "eq", value: "Published", caseInsensitive: true })).toEqual(["lower.md", "upper.md"]);
    expect(matchPaths({ field: "status", operator: "ne", value: "published", caseInsensitive: true })).toEqual(["draft.md"]);
    expect(matchPaths({ field: "status", operator: "in", value: ["DRAFT"], caseInsensitive: true })).toEqual(["draft.md"]);
    expect(matchPaths({ field: "status", operator: "nin", value: ["DRAFT"], caseInsensitive: true })).toEqual(["lower.md", "upper.md"]);
    expect(matchPaths({ field: "topics", operator: "all", value: ["postgresql", "mysql"], caseInsensitive: true })).toEqual(["upper.md"]);
    expect(matchPaths({ field: "topics", operator: "contains", value: "sql", caseInsensitive: true })).toEqual(["lower.md", "upper.md"]);
    expect(matchPaths({ field: "topics", operator: "prefix", value: "postgres", caseInsensitive: true })).toEqual(["upper.md"]);
    expect(matchPaths({ field: "topics", operator: "suffix", value: "SQL", caseInsensitive: true })).toEqual(["upper.md"]);
    // Lexical comparison folds too: "Draft" sorts before "published" once lowered.
    expect(matchPaths({ field: "status", operator: "lt", value: "M", caseInsensitive: true })).toEqual(["draft.md"]);
  });

  test("caseInsensitive leaves non-ASCII letters exact", () => {
    insertDoc("upper.md", { city: "\u00C9VORA" });
    insertDoc("lower.md", { city: "\u00E9vora" });

    // The ASCII part folds and the accented initial does not, so each
    // spelling matches itself only.
    expect(matchPaths({ field: "city", operator: "eq", value: "\u00C9vora", caseInsensitive: true })).toEqual(["upper.md"]);
    expect(matchPaths({ field: "city", operator: "eq", value: "\u00E9VORA", caseInsensitive: true })).toEqual(["lower.md"]);
  });

  test("boolean values round-trip through membership operators", () => {
    insertDoc("flagged.md", { reviewed: true });
    insertDoc("unflagged.md", { reviewed: false });

    expect(matchPaths({ field: "reviewed", operator: "in", value: [true] })).toEqual(["flagged.md"]);
    expect(matchPaths({ field: "reviewed", operator: "nin", value: [true] })).toEqual(["unflagged.md"]);
  });

  test("SQL injection payloads in keys and values stay data", () => {
    insertDoc("safe.md", { "key'; DROP TABLE documents; --": "v'; DROP TABLE documents; --" });

    expect(matchPaths({
      field: "key'; DROP TABLE documents; --",
      operator: "eq",
      value: "v'; DROP TABLE documents; --",
    })).toEqual(["safe.md"]);

    // Table survived.
    expect((db.prepare(`SELECT COUNT(*) as c FROM documents`).get() as { c: number }).c).toBe(1);
  });

  test("every condition seeks by value or by document, with and without statistics", () => {
    // The covering indexes are (key, <value>, document_id). An equality on the
    // value column is one probe by value. Everything else must seek the
    // document's rows through the primary key, or the planner walks the key's
    // whole value range once per document. `ne` and `nin` pair one of each.
    // ANALYZE must not change any of it (below about 32 rows, statistics
    // make a full scan of the tiny index the cheaper plan, correctly).
    for (let index = 0; index < 64; index++) {
      insertDoc(`d${index}.md`, { status: index % 2 ? "published" : "draft", priority: index, topics: [`t${index}`] });
    }

    // SQLite 3.43 labels the same point probe USING INDEX, without COVERING.
    const byValue = "INDEX idx_metadata_";
    const byDocument = "INDEX sqlite_autoindex_document_metadata_values_1 (document_id=?)";
    const expectations: [MetadataFilter, string][] = [
      [{ field: "status", operator: "eq", value: "published" }, byValue],
      [{ field: "topics", operator: "in", value: ["t1", "t2"] }, byValue],
      [{ field: "topics", operator: "all", value: ["t1"] }, byValue],
      [{ field: "status", operator: "eq", value: "PUBLISHED", caseInsensitive: true }, byDocument],
      [{ field: "priority", operator: "gt", value: 3 }, byDocument],
      [{ field: "priority", operator: "lte", value: 3 }, byDocument],
      [{ field: "status", operator: "ne", value: "draft" }, byDocument],
      [{ field: "topics", operator: "nin", value: ["t1"] }, byDocument],
      [{ field: "status", operator: "exists", value: true }, byDocument],
      [{ field: "status", operator: "type", value: "string" }, byDocument],
      [{ field: "topics", operator: "prefix", value: "t" }, byDocument],
      [{ field: "topics", operator: "contains", value: "1" }, byDocument],
      [{ field: "topics", operator: "suffix", value: "1", caseInsensitive: true }, byDocument],
    ];

    for (const statistics of [false, true]) {
      if (statistics) db.exec("ANALYZE");
      for (const [filter, seek] of expectations) {
        const compiled = compileMetadataFilter(parseMetadataFilter(filter), "d");
        const plan = (db.prepare(`EXPLAIN QUERY PLAN SELECT COUNT(*) FROM documents d WHERE ${compiled.sql}`)
          .all(...compiled.params) as { detail: string }[])
          .map(row => row.detail)
          .filter(detail => detail.includes(" mv "));
        const label = `${filter.operator}${"caseInsensitive" in filter ? " folded" : ""}${statistics ? " with statistics" : ""}`;
        expect(plan.some(detail => detail.includes(seek)), `${label}: ${plan.join(" | ")}`).toBe(true);
        expect(plan.some(detail => detail.includes("key=?") && !detail.includes("document_id=?")), `${label} walks a key range: ${plan.join(" | ")}`).toBe(false);
      }
    }
  });

  test("the corpus scope compiles each condition to one uncorrelated document set", () => {
    for (let index = 0; index < 64; index++) {
      insertDoc(`d${index}.md`, { status: index % 2 ? "published" : "draft", priority: index });
    }
    const filter = parseMetadataFilter({
      operator: "and",
      operands: [
        { field: "priority", operator: "gt", value: 3 },
        { operator: "not", operand: { field: "status", operator: "exists", value: false } },
      ],
    });
    const compiled = compileMetadataFilter(filter, "d", "corpus");
    expect(compiled.sql).toBe(
      "(d.id IN (SELECT mv.document_id FROM document_metadata_values mv WHERE mv.key = ? AND mv.value_type = 'number' AND mv.number_value > ?)" +
      " AND NOT NOT d.id IN (SELECT mv.document_id FROM document_metadata_values mv WHERE mv.key = ?))",
    );
    expect(compiled.params).toEqual(["priority", 3, "status"]);

    for (const statistics of [false, true]) {
      if (statistics) db.exec("ANALYZE");
      const plan = (db.prepare(`EXPLAIN QUERY PLAN SELECT COUNT(*) FROM documents d WHERE ${compiled.sql}`)
        .all(...compiled.params) as { detail: string }[])
        .map(row => row.detail);
      // The document sets are built once (LIST SUBQUERY), by covering index, and probed per document.
      expect(plan.filter(step => step.includes("LIST SUBQUERY"))).toHaveLength(2);
      expect(plan.some(step => step.includes(" EXISTS ")), plan.join(" | ")).toBe(false);
      expect(plan.some(step => step.includes("idx_metadata_number_lookup (key=? AND number_value>?)")), plan.join(" | ")).toBe(true);
    }
  });

  test("compiled SQL never interpolates user keys or values", () => {
    const filter = parseMetadataFilter({
      operator: "and",
      operands: [
        { field: "key'; --", operator: "eq", value: "value'; --" },
        { field: "topics", operator: "in", value: ["a'; --"] },
        { field: "topics", operator: "contains", value: "c'; --" },
        { field: "topics", operator: "suffix", value: "S'; --", caseInsensitive: true },
      ],
    });
    const compiled = compileMetadataFilter(filter, "d");
    expect(compiled.sql).not.toContain("'; --");
    expect(compiled.params).toContain("key'; --");
    expect(compiled.params).toContain("value'; --");
    expect(compiled.params).toContain("a'; --");
    expect(compiled.params).toContain("c'; --");
    expect(compiled.params).toContain("s'; --");
  });
});
