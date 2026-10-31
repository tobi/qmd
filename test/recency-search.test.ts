/**
 * The recency boost through the search pipeline, without an LLM.
 */

import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, insertContent, insertDocument, structuredSearch, type Store } from "../src/store.js";

let store: Store | undefined;
let dir: string | undefined;

afterEach(async () => {
  store?.close();
  store = undefined;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = undefined;
});

const DAY = 24 * 60 * 60 * 1000;

async function seeded(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-recency-"));
  const s = createStore(join(dir, "index.sqlite"));
  store = s;
  const add = (path: string, hash: string, ageDays: number) => {
    const at = new Date(Date.now() - ageDays * DAY).toISOString();
    insertContent(s.db, hash, `# ${path}\n\nzebrafinch migration notes`, at);
    insertDocument(s.db, "notes", path, path, hash, at, at);
  };
  add("alpha-old.md", "aaaa11112222", 3650);
  add("beta-new.md", "bbbb11112222", 0);
  return s;
}

const searches = [{ type: "lex" as const, query: "zebrafinch" }];

describe("structuredSearch with recency", () => {
  test("decays the older document and ranks the newer one first", async () => {
    const s = await seeded();
    const results = await structuredSearch(s, searches, {
      skipRerank: true,
      explain: true,
      recency: { halfLife: 30, weight: 0.6 },
    });

    expect(results.map((r) => r.displayPath)).toEqual(["notes/beta-new.md", "notes/alpha-old.md"]);
    expect(results[0]!.explain!.recencyDecay).toBeCloseTo(1, 2);
    expect(results[1]!.explain!.recencyDecay).toBeCloseTo(0.4, 2);
  });

  test("leaves scores alone when recency is not requested", async () => {
    const s = await seeded();
    const results = await structuredSearch(s, searches, { skipRerank: true, explain: true });

    expect(results).toHaveLength(2);
    for (const result of results) expect(result.explain!.recencyDecay).toBeUndefined();
  });
});
