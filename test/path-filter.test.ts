import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  searchFTS,
  structuredSearch,
  __testPathPrefixHelpers,
  type Store,
} from "../src/store.js";

const stores: Store[] = [];
const dirs: string[] = [];

async function makeStore(): Promise<Store> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-path-filter-"));
  dirs.push(dir);
  const store = createStore(join(dir, "test.sqlite"));
  stores.push(store);
  return store;
}

async function addDoc(store: Store, path: string, body: string, title = path, originalPath?: string): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(`${path}\n${body}`);
  insertContent(store.db, hash, body, now);
  insertDocument(store.db, "docs", path, title, hash, now, now, originalPath);
}

afterEach(async () => {
  while (stores.length) stores.pop()!.close();
  while (dirs.length) await rm(dirs.pop()!, { recursive: true, force: true });
});

describe("searchFTS pathPrefix", () => {
  test("returns all docs without filter", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");
    await addDoc(store, "other/b.md", "fcose graph layout");

    const results = searchFTS(store.db, "fcose", 10, "docs");
    expect(results.map(r => r.filepath).sort()).toEqual([
      "qmd://docs/other/b.md",
      "qmd://docs/stack/a.md",
    ]);
  });

  test("filters by folder-prefix and normalizes slashes/whitespace", async () => {
    const store = await makeStore();
    await addDoc(store, "foo/bar/a.md", "fcose graph layout");
    await addDoc(store, "foo/baz/b.md", "fcose graph layout");

    const results = searchFTS(store.db, "fcose", 10, "docs", "  /foo\\bar/  ");
    expect(results.map(r => r.filepath)).toEqual(["qmd://docs/foo/bar/a.md"]);
  });

  test("root and empty path mean no filter", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");
    await addDoc(store, "other/b.md", "fcose graph layout");

    expect(searchFTS(store.db, "fcose", 10, "docs", "/")).toHaveLength(2);
    expect(searchFTS(store.db, "fcose", 10, "docs", "   ")).toHaveLength(2);
  });

  test("range predicate respects folder boundary", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");
    await addDoc(store, "stack0/b.md", "fcose graph layout");
    await addDoc(store, "stack-monitor/c.md", "fcose graph layout");

    const results = searchFTS(store.db, "fcose", 10, "docs", "stack");
    expect(results.map(r => r.filepath)).toEqual(["qmd://docs/stack/a.md"]);
  });

  test("range predicate is case-sensitive", async () => {
    const store = await makeStore();
    await addDoc(store, "Stack/a.md", "fcose graph layout");
    await addDoc(store, "stack/b.md", "fcose graph layout");

    const results = searchFTS(store.db, "fcose", 10, "docs", "stack");
    expect(results.map(r => r.filepath)).toEqual(["qmd://docs/stack/b.md"]);
  });

  test("uses original paths for case-sensitive folder filtering and result paths", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout", "stack/a.md", "stack/a.md");
    await addDoc(store, "stack/d.md", "fcose graph layout", "Stack/d.md", "Stack/d.md");
    await addDoc(store, "stack/sub/e.md", "fcose graph layout", "stack/sub/e.md", "stack/sub/e.md");

    expect(searchFTS(store.db, "fcose", 10, "docs", "stack").map(r => r.filepath).sort()).toEqual([
      "qmd://docs/stack/a.md",
      "qmd://docs/stack/sub/e.md",
    ]);
    expect(searchFTS(store.db, "fcose", 10, "docs", "Stack").map(r => r.filepath)).toEqual([
      "qmd://docs/Stack/d.md",
    ]);
  });

  test("folder-only design: file-like path matches no documents", async () => {
    const store = await makeStore();
    await addDoc(store, "README.md", "fcose graph layout");

    expect(searchFTS(store.db, "fcose", 10, "docs", "README.md")).toEqual([]);
  });

  test("SQL injection probe is parameterized", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");

    const results = searchFTS(store.db, "fcose", 10, "docs", "'; DROP TABLE documents;--");
    expect(results).toEqual([]);
    expect(searchFTS(store.db, "fcose", 10, "docs", "stack")).toHaveLength(1);
  });

  test("lexicographicSuccessor invariant", () => {
    expect(__testPathPrefixHelpers.lexicographicSuccessor("stack/")).toBe("stack0");
    expect(() => __testPathPrefixHelpers.lexicographicSuccessor("stack")).toThrow(/must end with '\/'/);
  });
});

describe("structuredSearch pathPrefix", () => {
  test("passes pathPrefix through structured lex search", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");
    await addDoc(store, "other/b.md", "fcose graph layout");

    const results = await structuredSearch(
      store,
      [{ type: "lex", query: "fcose" }],
      { collections: ["docs"], pathPrefix: "stack", skipRerank: true, explain: true }
    );

    expect(results).toHaveLength(1);
    expect(results[0]!.file).toBe("qmd://docs/stack/a.md");
    expect(results[0]!.explain?.scoreType).toBe("rrf-position");
    expect(results[0]!.explain?.backendSources).toEqual(["bm25"]);
  });
});
