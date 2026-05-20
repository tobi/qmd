import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  structuredSearch,
  type Store,
} from "../src/store.js";

const stores: Store[] = [];
const dirs: string[] = [];

async function makeStore(): Promise<Store> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-score-breakdown-"));
  dirs.push(dir);
  const store = createStore(join(dir, "test.sqlite"));
  stores.push(store);
  return store;
}

async function addDoc(store: Store, path: string, body: string, title = path): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(`${path}\n${body}`);
  insertContent(store.db, hash, body, now);
  insertDocument(store.db, "docs", path, title, hash, now, now);
}

afterEach(async () => {
  while (stores.length) stores.pop()!.close();
  while (dirs.length) await rm(dirs.pop()!, { recursive: true, force: true });
});

describe("score breakdown", () => {
  test("rerank:false exposes rrf-position and per-result bm25 backend", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");

    const results = await structuredSearch(
      store,
      [{ type: "lex", query: "fcose" }],
      { collections: ["docs"], skipRerank: true, explain: true }
    );

    expect(results[0]!.score).toBe(1);
    expect(results[0]!.explain?.scoreType).toBe("rrf-position");
    expect(results[0]!.explain?.backendSources).toEqual(["bm25"]);
    expect(results[0]!.explain?.ftsScores.length).toBeGreaterThan(0);
    expect(results[0]!.explain?.vectorScores).toEqual([]);
  });

  test("rerank:true exposes rerank-blend without calling a real LLM", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");
    store.rerank = async (_query, docs) => docs.map(doc => ({ file: doc.file, score: 0.5 }));

    const results = await structuredSearch(
      store,
      [{ type: "lex", query: "fcose" }],
      { collections: ["docs"], skipRerank: false, explain: true }
    );

    expect(results[0]!.explain?.scoreType).toBe("rerank-blend");
    expect(results[0]!.explain?.backendSources).toEqual(["bm25"]);
    expect(results[0]!.explain?.rerankScore).toBe(0.5);
    expect(results[0]!.score).toBe(results[0]!.explain?.blendedScore);
  });

  test("score is unrounded when explain payload is available", async () => {
    const store = await makeStore();
    await addDoc(store, "stack/a.md", "fcose graph layout");

    const results = await structuredSearch(
      store,
      [{ type: "lex", query: "fcose" }],
      { collections: ["docs"], skipRerank: true, explain: true }
    );

    expect(results[0]!.explain).toBeDefined();
    expect(results[0]!.score).toBe(results[0]!.explain!.blendedScore);
  });
});
