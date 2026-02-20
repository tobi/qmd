import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, hashContent, isSqliteVecAvailable, type Store } from "../src/store.js";

let testDir: string;
let configDir: string;
let store: Store;
let hasSqliteVec = false;

async function insertVectorDoc(
  store: Store,
  collection: string,
  path: string,
  body: string,
  embedding: number[],
): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(`${collection}:${path}:${body}`);

  store.db.prepare(`
    INSERT INTO content (hash, doc, created_at)
    VALUES (?, ?, ?)
  `).run(hash, body, now);

  store.db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, 1)
  `).run(collection, path, path.replace(/\.md$/, ""), hash, now, now);

  store.db.prepare(`
    INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
    VALUES (?, 0, 0, 'test', ?)
  `).run(hash, now);

  store.db.prepare(`
    INSERT INTO vectors_vec (hash_seq, embedding)
    VALUES (?, ?)
  `).run(`${hash}_0`, new Float32Array(embedding));
}

describe("searchVec multi-collection filtering", () => {
  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-searchvec-multi-"));
    configDir = join(testDir, "config");
    await mkdir(configDir, { recursive: true });

    await writeFile(
      join(configDir, "index.yml"),
      YAML.stringify({
        collections: {
          "target-a": { path: "/tmp/target-a", pattern: "**/*.md" },
          "target-b": { path: "/tmp/target-b", pattern: "**/*.md" },
          noisy: { path: "/tmp/noisy", pattern: "**/*.md" },
        },
      })
    );

    process.env.QMD_CONFIG_DIR = configDir;
    store = createStore(join(testDir, "index.sqlite"));
    hasSqliteVec = isSqliteVecAvailable();
    if (hasSqliteVec) {
      store.ensureVecTable(3);
    }
  });

  afterEach(async () => {
    if (store) {
      store.close();
    }
    delete process.env.QMD_CONFIG_DIR;
    if (testDir) {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("supports collection IN filters for repeated -c semantics", async () => {
    if (!hasSqliteVec) {
      return;
    }

    await insertVectorDoc(store, "target-a", "a.md", "dominator target a", [0.99, 0.01, 0]);
    await insertVectorDoc(store, "target-b", "b.md", "dominator target b", [0.98, 0.02, 0]);

    // Create many highly similar vectors in an unrequested collection to mimic top-k domination.
    for (let i = 0; i < 60; i++) {
      await insertVectorDoc(store, "noisy", `noise-${i}.md`, `dominator noisy ${i}`, [1, 0, 0]);
    }

    const results = await store.searchVec(
      "dominator",
      "embeddinggemma",
      10,
      ["target-a", "target-b"],
      undefined,
      [1, 0, 0],
    );

    const collections = new Set(results.map(r => r.collectionName));

    expect(collections.has("target-a")).toBe(true);
    expect(collections.has("target-b")).toBe(true);
    expect(collections.has("noisy")).toBe(false);
  });
});
