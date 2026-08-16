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
import type { MetadataFilter } from "../src/metadata-filter.js";

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
      key: "status", operator: "eq", value: "published",
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/published.md"]);
    expect(filtered[0]!.metadata).toEqual({ status: "published" });
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
      key: "status", operator: "exists", value: false,
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
      key: "status", operator: "eq", value: "published",
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
        { key: "topics", operator: "all", value: ["typescript", "programming"] },
        {
          operator: "or",
          operands: [
            { key: "status", operator: "eq", value: "published" },
            { key: "priority", operator: "gte", value: 3 },
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
      key: "status", operator: "eq", value: "published",
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
      { key: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/far-published.md"]);
    expect(filtered[0]!.metadata).toEqual({ status: "published" });
  });

  test("returns empty when no documents are eligible", async () => {
    store.ensureVecTable(3);
    await insertEmbeddedDoc("notes", "doc.md", "# Doc", [1, 0, 0], { status: "draft" });

    const filtered = await searchVec(
      store.db, "q", model, 10, undefined, undefined, queryEmbedding, undefined,
      { key: "status", operator: "eq", value: "published" },
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
      { key: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["notes/published-copy.md"]);
  });

  test("composes collection scope with the metadata filter", async () => {
    store.ensureVecTable(3);
    await insertEmbeddedDoc("notes", "a.md", "# A", [1, 0, 0], { status: "published" });
    await insertEmbeddedDoc("docs", "b.md", "# B", [0.9, 0.1, 0], { status: "published" });

    const filtered = await searchVec(
      store.db, "q", model, 10, "docs", undefined, queryEmbedding, undefined,
      { key: "status", operator: "eq", value: "published" },
    );
    expect(filtered.map(r => r.displayPath)).toEqual(["docs/b.md"]);
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
      filter: { key: "status", operator: "eq", value: "published" },
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
