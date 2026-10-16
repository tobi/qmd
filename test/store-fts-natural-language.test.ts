/**
 * store-fts-natural-language.test.ts - lex queries written as questions.
 *
 * A lex query is parsed as keywords: every term ANDed, each as a prefix. A
 * question asks for every one of its words, function words included, so it
 * matches only a document that happens to hold them all. A query that reads as
 * a question (it contains a stopword or ends with "?", and uses no quotes or
 * -negation) is ORed over its content words instead, without prefix
 * expansion, and bm25 ranks the matches. Keyword queries and the explicit
 * syntax keep their parse.
 *
 * Run with: bun test test/store-fts-natural-language.test.ts
 *        or: pnpm test:node test/store-fts-natural-language.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, afterEach } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  syncConfigToDb,
  type Store,
} from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

const DOCS: Record<string, { title: string; body: string }> = {
  "rf.md": {
    title: "RF feedback",
    body: "Storage ring RF frequency feedback loop and how its gains are tuned.",
  },
  "orbit.md": {
    title: "Orbit correction",
    body: "Orbit correction procedure for the storage ring, run by multi-agent control.",
  },
  "fox.md": {
    title: "Pangram",
    body: "The quick brown fox jumps over the lazy dog.",
  },
};

let testDir: string;
let currentStore: Store | null = null;

async function createDocStore(): Promise<Store> {
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const config: CollectionConfig = {
    collections: { docs: { path: "/test/docs", pattern: "**/*.md" } },
  };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(config));
  const store = createStore(join(testDir, `nl-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
  currentStore = store;
  syncConfigToDb(store.db, config);
  const now = new Date().toISOString();
  for (const [path, { title, body }] of Object.entries(DOCS)) {
    const hash = await hashContent(body);
    insertContent(store.db, hash, body, now);
    insertDocument(store.db, "docs", path, title, hash, now, now);
  }
  return store;
}

const paths = (store: Store, query: string) => store.searchFTS(query, 10).map(r => r.displayPath);

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-fts-natural-language-"));
});

afterEach(() => {
  currentStore?.close();
  currentStore = null;
  delete process.env.QMD_CONFIG_DIR;
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

describe("natural-language lex queries", () => {
  test("a question whose words no single document holds still finds the best match", async () => {
    const store = await createDocStore();
    // "tuned" is not in the RF document, and nothing holds "what": the AND parse returns nothing.
    const hits = paths(store, "what is the storage ring RF frequency tuned to?");
    expect(hits[0]).toBe("docs/rf.md");
    expect(hits).toContain("docs/orbit.md");
    expect(hits).not.toContain("docs/fox.md");
  });

  test("bm25 ranks the document holding more of the content words first", async () => {
    const store = await createDocStore();
    expect(paths(store, "how is orbit correction run for the storage ring")[0]).toBe("docs/orbit.md");
  });

  test("function words do not match on their own", async () => {
    const store = await createDocStore();
    // Only "brown" is a content word; "the"/"is"/"of" would otherwise match every document.
    expect(paths(store, "what is the colour of the brown animal")).toEqual(["docs/fox.md"]);
  });

  test("content words are not prefix-expanded", async () => {
    const store = await createDocStore();
    expect(paths(store, "where is the correct procedure")).toEqual(["docs/orbit.md"]);
    expect(paths(store, "where is the corr")).toEqual([]);
  });

  test("hyphenated words stay phrases", async () => {
    const store = await createDocStore();
    expect(paths(store, "who runs the multi-agent control?")).toEqual(["docs/orbit.md"]);
  });

  test("a query of stopwords only keeps the keyword parse", async () => {
    const store = await createDocStore();
    expect(paths(store, "the").sort()).toEqual(["docs/fox.md", "docs/orbit.md"]);
  });
});

describe("keyword queries and explicit syntax are unchanged", () => {
  test("keywords without a stopword stay ANDed prefixes", async () => {
    const store = await createDocStore();
    expect(paths(store, "storage feed")).toEqual(["docs/rf.md"]);
    expect(paths(store, "storage fox")).toEqual([]);
  });

  test("a quoted phrase keeps the keyword parse even with stopwords", async () => {
    const store = await createDocStore();
    expect(paths(store, '"the storage ring" orbit')).toEqual(["docs/orbit.md"]);
  });

  test("negation keeps the keyword parse even with stopwords", async () => {
    const store = await createDocStore();
    expect(paths(store, "the storage -orbit")).toEqual([]);
    expect(paths(store, "storage ring -orbit")).toEqual(["docs/rf.md"]);
  });
});
