/**
 * metadata-search.test.ts - Metadata filtering across FTS, vector, and
 * structured search. Vector tests use precomputed embeddings so no models
 * are downloaded.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  searchFTS,
  searchVec,
  structuredSearch,
  insertContent,
  insertDocument,
  insertEmbedding,
  hashContent,
  DEFAULT_EMBED_MODEL,
  type Store,
} from "../src/store.js";
import { replaceDocumentMetadata, syncDocumentMetadata } from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";
import { parseMetadataFilter, type MetadataFilter } from "../src/metadata-filter.js";
import type { Database, SQLiteValue } from "../src/db.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-search-"));
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

async function insertDoc(
  collection: string,
  path: string,
  body: string,
  metadata?: DocumentMetadata,
): Promise<{ documentId: number; hash: string }> {
  const now = new Date().toISOString();
  const hash = await hashContent(body);
  insertContent(store.db, hash, body, now);
  const documentId = insertDocument(store.db, collection, path, path, hash, now, now);

  if (metadata !== undefined) {
    replaceDocumentMetadata(store.db, documentId, { metadata, extractionVersion: METADATA_EXTRACTION_VERSION });
  }
  return { documentId, hash };
}

describe("searchFTS with metadata filter", () => {
  test("returns only documents matching the filter, with metadata attached", async () => {
    await insertDoc("notes", "published.md", "# Auth\n\nauthentication flow details", { status: "published" });
    await insertDoc("notes", "draft.md", "# Auth draft\n\nauthentication flow draft", { status: "draft" });
    await insertDoc("notes", "plain.md", "# Auth notes\n\nauthentication flow notes", {});

    const unfiltered = searchFTS(store.db, "authentication", 10);
    expect(unfiltered.length).toBe(3);

    const filtered = searchFTS(store.db, "authentication", 10, undefined, {
      field: "status", operator: "eq", value: "published",
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/published.md"]);
    expect(filtered[0]!.metadata).toEqual({ status: "published" });
  });

  test("finds a matching document ranked below the unfiltered candidate window", async () => {
    // limit=5 made the old filtered window 50; every noise doc outranks the target.
    for (let i = 0; i < 50; i++) {
      await insertDoc("notes", `noise-${i}.md`, `# N${i}\n\nalpha alpha alpha`, { status: "draft" });
    }
    await insertDoc(
      "notes",
      "target.md",
      `# Target\n\n${"Unrelated prose. ".repeat(40)}One weaker mention of alpha.`,
      { status: "published" },
    );

    const filtered = searchFTS(store.db, "alpha", 5, undefined, {
      field: "status", operator: "eq", value: "published",
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/target.md"]);
  });

  test("excludes pending, stale, and errored documents from filtered search", async () => {
    const { documentId: erroredId } = await insertDoc("notes", "errored.md", "# One\n\ncommon term");
    await insertDoc("notes", "pending.md", "# Two\n\ncommon term");
    const { documentId: staleId } = await insertDoc("notes", "stale.md", "# Three\n\ncommon term");
    await insertDoc("notes", "extracted.md", "# Four\n\ncommon term", {});

    replaceDocumentMetadata(store.db, erroredId, {
      metadata: {},
      error: "boom",
      extractionVersion: METADATA_EXTRACTION_VERSION,
    });
    replaceDocumentMetadata(store.db, staleId, {
      metadata: {},
      extractionVersion: METADATA_EXTRACTION_VERSION - 1,
    });

    // An unprocessed document must not satisfy `exists: false`.
    const filtered = searchFTS(store.db, "common", 10, undefined, {
      field: "status", operator: "exists", value: false,
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/extracted.md"]);

    // Unfiltered search still sees every document.
    expect(searchFTS(store.db, "common", 10).length).toBe(4);
  });

  test("composes multi-collection scope with one metadata filter", async () => {
    await insertDoc("notes", "a.md", "# A\n\nshared topic", { status: "published" });
    await insertDoc("docs", "b.md", "# B\n\nshared topic", { status: "published" });
    await insertDoc("docs", "c.md", "# C\n\nshared topic", { status: "draft" });
    await insertDoc("other", "d.md", "# D\n\nshared topic", { status: "published" });

    const filtered = searchFTS(store.db, "shared", 10, ["notes", "docs"], {
      field: "status", operator: "eq", value: "published",
    });
    expect(filtered.map(r => r.displayPath).sort()).toEqual(["docs/b.md", "notes/a.md"]);
  });

  test("nested filters apply through FTS", async () => {
    await insertDoc("notes", "match.md", "# M\n\nfilter target", {
      topics: ["typescript", "programming"], status: "published", priority: 5,
    });
    await insertDoc("notes", "wrong-topic.md", "# W\n\nfilter target", {
      topics: ["cooking"], status: "published", priority: 5,
    });
    await insertDoc("notes", "low-priority.md", "# L\n\nfilter target", {
      topics: ["typescript", "programming"], status: "draft", priority: 1,
    });

    const filter: MetadataFilter = {
      operator: "and",
      operands: [
        { field: "topics", operator: "all", value: ["typescript", "programming"] },
        {
          operator: "or",
          operands: [
            { field: "status", operator: "eq", value: "published" },
            { field: "priority", operator: "gte", value: 3 },
          ],
        },
      ],
    };
    const filtered = searchFTS(store.db, "filter target", 10, undefined, filter);
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/match.md"]);
  });

  test("selective filters may under-fill but never return an ineligible row", async () => {
    for (let i = 0; i < 20; i++) {
      await insertDoc("notes", `doc-${i}.md`, `# Doc ${i}\n\nrepeated keyword body ${i}`, {
        status: i === 0 ? "published" : "draft",
      });
    }

    const filtered = searchFTS(store.db, "repeated keyword", 5, undefined, {
      field: "status", operator: "eq", value: "published",
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/doc-0.md"]);
  });
});

describe("searchVec with metadata filter", () => {
  const model = DEFAULT_EMBED_MODEL;
  const queryEmbedding = [1, 0, 0];

  async function insertEmbeddedDoc(
    collection: string,
    path: string,
    body: string,
    embedding: number[],
    metadata?: DocumentMetadata,
  ): Promise<void> {
    const { hash } = await insertDoc(collection, path, body, metadata);
    insertEmbedding(store.db, hash, 0, 0, new Float32Array(embedding), model, new Date().toISOString(), 1);
  }

  test("exact scan over the eligible set returns only matching documents", async () => {
    store.ensureVecTable(3);
    await insertEmbeddedDoc("notes", "close-draft.md", "# Close draft", [1, 0, 0], { status: "draft" });
    await insertEmbeddedDoc("notes", "far-published.md", "# Far published", [0, 1, 0], { status: "published" });

    const unfiltered = await searchVec(store.db, "q", model, 10, undefined, undefined, queryEmbedding);
    expect(unfiltered.map(r => r.displayPath)).toEqual(["notes/close-draft.md", "notes/far-published.md"]);

    const filtered = await searchVec(
      store.db, "q", model, 10, undefined, undefined, queryEmbedding, undefined,
      { field: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/far-published.md"]);
    expect(filtered[0]!.metadata).toEqual({ status: "published" });
  });

  test("a near-ceiling filter fits both vector lookup paths and still excludes shared nonmatching copies", async () => {
    store.ensureVecTable(3);
    const body = "# Book\n\nDeterministic vector fixture";
    const { hash } = await insertDoc("notes", "published.md", body, { eligible: true });
    await insertDoc("notes", "draft-copy.md", body, { eligible: false });
    const timestamp = new Date().toISOString();

    store.db.transaction(() => {
      for (let sequence = 0; sequence < 20_000; sequence++) {
        insertEmbedding(store.db, hash, sequence, 0, new Float32Array([1, sequence / 20_001, 0]), model, timestamp, 20_001);
      }
    })();

    // 256 nodes, 64 distinct members per wide leaf: a valid filter with
    // 31,490 bindings. Candidate IDs must not exhaust the remaining budget.
    const members = Array.from({ length: 64 }, (_, index) => `value-${index}`);
    const leaves = Array.from({ length: 246 }, () => ({ field: "absent", operator: "all", value: members }));
    const groups = Array.from({ length: 8 }, (_, index) => ({ operator: "or", operands: leaves.slice(index * 32, (index + 1) * 32) }));
    const metadataFilter = parseMetadataFilter({ operator: "or", operands: [{ field: "eligible", operator: "eq", value: true }, ...groups] });

    // At 20,000 eligible chunks the exact path returns up to limit * 3 IDs.
    const exactResults = await searchVec(store.db, "q", model, 500, "notes", undefined, queryEmbedding, undefined, metadataFilter);
    expect(exactResults.map(result => result.displayPath)).toEqual(["notes/published.md"]);

    // The extra chunk crosses into capped global lookup. Its 4,096 candidate
    // IDs used to push the final document lookup past Node's variable limit.
    insertEmbedding(store.db, hash, 20_000, 0, new Float32Array([1, 1, 0]), model, timestamp, 20_001);
    const fallbackResults = await searchVec(store.db, "q", model, 137, "notes", undefined, queryEmbedding, undefined, metadataFilter);
    expect(fallbackResults.map(result => result.displayPath)).toEqual(["notes/published.md"]);
  });

  test("returns empty when no documents are eligible", async () => {
    store.ensureVecTable(3);
    await insertEmbeddedDoc("notes", "doc.md", "# Doc", [1, 0, 0], { status: "draft" });

    const filtered = await searchVec(
      store.db, "q", model, 10, undefined, undefined, queryEmbedding, undefined,
      { field: "status", operator: "eq", value: "published" },
    );
    expect(filtered).toEqual([]);
  });

  test("shared content hash returns only the matching document path", async () => {
    store.ensureVecTable(3);
    const now = new Date().toISOString();
    const body = "# Shared body";
    const hash = await hashContent(body);
    insertContent(store.db, hash, body, now);

    const publishedId = insertDocument(store.db, "notes", "published-copy.md", "t", hash, now, now);
    const draftId = insertDocument(store.db, "docs", "draft-copy.md", "t", hash, now, now);
    replaceDocumentMetadata(store.db, publishedId, {
      metadata: { status: "published" }, extractionVersion: METADATA_EXTRACTION_VERSION,
    });
    replaceDocumentMetadata(store.db, draftId, {
      metadata: { status: "draft" }, extractionVersion: METADATA_EXTRACTION_VERSION,
    });
    insertEmbedding(store.db, hash, 0, 0, new Float32Array([1, 0, 0]), model, now, 1);

    const filtered = await searchVec(
      store.db, "q", model, 10, undefined, undefined, queryEmbedding, undefined,
      { field: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/published-copy.md"]);
  });

  test("composes collection scope with the metadata filter", async () => {
    store.ensureVecTable(3);
    await insertEmbeddedDoc("notes", "a.md", "# A", [1, 0, 0], { status: "published" });
    await insertEmbeddedDoc("docs", "b.md", "# B", [0.9, 0.1, 0], { status: "published" });

    const filtered = await searchVec(
      store.db, "q", model, 10, "docs", undefined, queryEmbedding, undefined,
      { field: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["docs/b.md"]);
  });

  const eligibleOnly: MetadataFilter = { field: "eligible", operator: "eq", value: true };

  /** An eligible document and an ineligible copy of its content in one collection, one chunk per vector. */
  async function insertLongDocumentWithExcludedCopy(vectors: number[][]): Promise<void> {
    const body = "# Many chunks";
    const { hash } = await insertDoc("book", "many-chunks.md", body, { eligible: true });
    await insertDoc("book", "excluded-copy.md", body, { eligible: false });
    const now = new Date().toISOString();
    vectors.forEach((vector, seq) => insertEmbedding(store.db, hash, seq, seq * 100, new Float32Array(vector), model, now, vectors.length));
  }

  test("a document with many close chunks does not starve another eligible document", async () => {
    store.ensureVecTable(3);
    await insertLongDocumentWithExcludedCopy(Array.from({ length: 20 }, () => [1, 0, 0]));
    await insertEmbeddedDoc("book", "second-document.md", "# Second document", [0, 1, 0], { eligible: true });

    const results = await searchVec(store.db, "q", model, 2, undefined, undefined, queryEmbedding, undefined, eligibleOnly);
    expect(results.map(r => r.displayPath)).toEqual(["book/many-chunks.md", "book/second-document.md"]);
  });

  test.each([20, 450])("the best chunk of a %i-chunk document survives deduplication", async (chunks) => {
    store.ensureVecTable(3);
    await insertLongDocumentWithExcludedCopy(
      Array.from({ length: chunks }, (_, seq) => (seq === chunks - 1 ? [1, 0, 0] : [0, 0, 1])),
    );
    await insertEmbeddedDoc("book", "second-document.md", "# Second document", [-1, 0.1, 0], { eligible: true });

    const results = await searchVec(store.db, "q", model, 2, undefined, undefined, queryEmbedding, undefined, eligibleOnly);
    expect(results.map(r => r.displayPath)).toEqual(["book/many-chunks.md", "book/second-document.md"]);
    expect(results[0]!.chunkPos).toBe((chunks - 1) * 100);
  });

  test("a filter admitting more than 20,000 chunks still returns its nearest eligible documents", async () => {
    store.ensureVecTable(3);
    store.db.exec("BEGIN");
    for (let i = 0; i < 20_001; i++) {
      await insertEmbeddedDoc("book", `eligible-${i}.md`, `# Eligible ${i}`, [0, 1, 0], { eligible: true });
    }
    for (let i = 0; i < 200; i++) {
      await insertEmbeddedDoc("book", `closer-${i}.md`, `# Closer ${i}`, [1, 0, 0], { eligible: false });
    }
    store.db.exec("COMMIT");

    const filtered = await searchVec(
      store.db, "q", model, 5, "book", undefined, queryEmbedding, undefined,
      { field: "eligible", operator: "eq", value: true },
    );
    expect(filtered).toHaveLength(5);
    expect(filtered.every(r => r.metadata.eligible === true)).toBe(true);
  }, 120_000);

  test("a filtered vector scan binds no list of eligible rows, so none is held on the heap", async () => {
    store.ensureVecTable(3);
    // One eligible document of 2,000 chunks, and closer ineligible documents.
    const { hash } = await insertDoc("book", "long-eligible.md", "# Long eligible", { eligible: true });
    const now = new Date().toISOString();
    store.db.transaction(() => {
      for (let seq = 0; seq < 2_000; seq++) insertEmbedding(store.db, hash, seq, seq, new Float32Array([0, 1, 0]), model, now, 2_000);
    })();
    for (let i = 0; i < 50; i++) {
      await insertEmbeddedDoc("book", `closer-${i}.md`, `# Closer ${i}`, [1, 0, 0], { eligible: false });
    }
    // Every string bound to a vector scan: a JSON list of eligible rowids would show here.
    const bound: string[] = [];
    const recording: Database = {
      prepare: (sql: string) => {
        const real = store.db.prepare(sql);
        if (!sql.includes("MATCH")) return real;
        return {
          ...real,
          run: real.run.bind(real),
          get: real.get.bind(real),
          iterate: real.iterate.bind(real),
          all: (...params: SQLiteValue[]) => {
            for (const param of params) if (typeof param === "string" && param.startsWith("[")) bound.push(param);
            return real.all(...params);
          },
        };
      },
      transaction: (fn) => store.db.transaction(fn),
      exec: (sql: string) => store.db.exec(sql),
      loadExtension: (path: string) => store.db.loadExtension(path),
      close: () => store.db.close(),
    };

    for (const scope of ["book", undefined]) {
      const results = await searchVec(recording, "q", model, 5, scope, undefined, queryEmbedding, undefined, eligibleOnly);
      expect(results.map(r => r.displayPath)).toEqual(["book/long-eligible.md"]);
    }
    expect(bound).toEqual([]);
  });

  test("shared content hash within one collection returns only the matching document path", async () => {
    store.ensureVecTable(3);
    const body = "# Shared body";
    const { hash } = await insertDoc("notes", "published-copy.md", body, { status: "published" });
    await insertDoc("notes", "draft-copy.md", body, { status: "draft" });
    insertEmbedding(store.db, hash, 0, 0, new Float32Array([1, 0, 0]), model, new Date().toISOString(), 1);

    const filtered = await searchVec(
      store.db, "q", model, 10, "notes", undefined, queryEmbedding, undefined,
      { field: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/published-copy.md"]);
  });
});

describe("structuredSearch with metadata filter", () => {
  test("lex-only structured search filters and attaches metadata", async () => {
    // Ensure metadata comes from real frontmatter extraction end to end.
    const publishedBody = "---\nqmd:\n  metadata:\n    status: published\n---\n\n# Pub\n\nstructured keyword";
    const { documentId: publishedId } = await insertDoc("notes", "published.md", publishedBody);
    syncDocumentMetadata(store.db, publishedId, publishedBody, "published.md");

    const draftBody = "---\nqmd:\n  metadata:\n    status: draft\n---\n\n# Draft\n\nstructured keyword";
    const { documentId: draftId } = await insertDoc("notes", "draft.md", draftBody);
    syncDocumentMetadata(store.db, draftId, draftBody, "draft.md");

    const results = await structuredSearch(store, [{ type: "lex", query: "structured keyword" }], {
      filter: { field: "status", operator: "eq", value: "published" },
      skipRerank: true,
    });

    expect(results.map(r => r.displayPath)).toEqual(["notes/published.md"]);
    expect(results[0]!.metadata).toEqual({ status: "published" });
  });

  test("unfiltered structured search attaches metadata to every result", async () => {
    await insertDoc("notes", "a.md", "# A\n\nmeta keyword", { topics: ["x"] });
    await insertDoc("notes", "b.md", "# B\n\nmeta keyword");

    const results = await structuredSearch(store, [{ type: "lex", query: "meta keyword" }], {
      skipRerank: true,
    });

    expect(results.length).toBe(2);
    const byPath = new Map(results.map(r => [r.displayPath, r.metadata]));
    expect(byPath.get("notes/a.md")).toEqual({ topics: ["x"] });
    expect(byPath.get("notes/b.md")).toEqual({});
  });
});

describe("searchFTS with a collection scope and a metadata filter together", () => {
  const published: MetadataFilter = { field: "status", operator: "eq", value: "published" };

  test("returns the in-scope document the filter admits, past both the window and a stronger draft", async () => {
    // Every noise document outranks both small-collection documents globally,
    // and the draft outranks the target inside the small collection.
    for (let i = 0; i < 50; i++) {
      await insertDoc("large", `noise-${i}.md`, `# N${i}\n\nalpha alpha alpha`, { status: "published" });
    }
    await insertDoc("small", "draft.md", "# Draft\n\nalpha alpha", { status: "draft" });
    await insertDoc(
      "small",
      "target.md",
      `# Target\n\n${"Unrelated prose. ".repeat(40)}One weaker mention of alpha.`,
      { status: "published" },
    );

    const results = searchFTS(store.db, "alpha", 1, "small", published);
    expect(results.map(r => r.displayPath)).toEqual(["small/target.md"]);
  });

  test("keeps the 256 KiB body cap on the scoped path", async () => {
    await insertDoc("small", "long.md", `# Long\n\nalpha ${"z".repeat(300 * 1024)}`, { status: "published" });

    const results = searchFTS(store.db, "alpha", 5, "small", published);
    expect(results).toHaveLength(1);
    expect(results[0]!.body!.length).toBe(262_144);
  });
});
