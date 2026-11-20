/**
 * findSimilarByEmbedding against the partitioned vector layout.
 */

import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, insertContent, insertDocument, findSimilarByEmbedding, type Store } from "../src/store.js";

let store: Store | undefined;
let dir: string | undefined;

afterEach(async () => {
  store?.close();
  store = undefined;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = undefined;
});

async function seeded(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-similar-"));
  const s = createStore(join(dir, "index.sqlite"));
  store = s;
  s.ensureVecTable(3);
  const now = new Date().toISOString();
  const add = (collection: string, path: string, hash: string, vector: number[]) => {
    insertContent(s.db, hash, `# ${path}\n\nbody of ${path}`, now);
    insertDocument(s.db, collection, path, path, hash, now, now);
    s.insertEmbedding(hash, 0, 0, new Float32Array(vector), "test-model", now, 1);
  };
  add("notes", "source.md", "aaaa11112222", [1, 0, 0]);
  add("notes", "near.md", "bbbb11112222", [0.9, 0.1, 0]);
  add("notes", "far.md", "cccc11112222", [0, 1, 0]);
  add("work", "other.md", "dddd11112222", [0.8, 0.2, 0]);
  return s;
}

describe("findSimilarByEmbedding", () => {
  test("ranks other documents by the source's stored embedding and leaves the source out", async () => {
    const s = await seeded();
    const results = findSimilarByEmbedding(s.db, "notes/source.md", 5);

    expect(results.map((r) => r.displayPath)).toEqual(["notes/near.md", "work/other.md", "notes/far.md"]);
    expect(results[0]!.score).toBeGreaterThan(results[2]!.score);
    expect(results[0]!.source).toBe("vec");
    expect(results[0]!.body).toContain("body of near.md");
  });

  test("resolves a docid, and honours limit and collection scope", async () => {
    const s = await seeded();

    expect(findSimilarByEmbedding(s.db, "#aaaa11", 1).map((r) => r.displayPath)).toEqual(["notes/near.md"]);
    expect(findSimilarByEmbedding(s.db, "notes/source.md", 5, "work").map((r) => r.displayPath)).toEqual(["work/other.md"]);
    expect(findSimilarByEmbedding(s.db, "notes/source.md", 5, ["notes", "work"])).toHaveLength(3);
  });

  test("returns nothing for an unknown document or scope", async () => {
    const s = await seeded();

    expect(findSimilarByEmbedding(s.db, "notes/missing.md", 5)).toEqual([]);
    expect(findSimilarByEmbedding(s.db, "notes/source.md", 5, "absent")).toEqual([]);
  });

  test("returns nothing before anything is embedded", async () => {
    dir = await mkdtemp(join(tmpdir(), "qmd-similar-"));
    store = createStore(join(dir, "index.sqlite"));

    expect(findSimilarByEmbedding(store.db, "notes/source.md", 5)).toEqual([]);
  });
});
