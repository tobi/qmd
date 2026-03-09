import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import postgres from "postgres";
import YAML from "yaml";

const TEST_POSTGRES_URL = "postgresql://teleclawd@localhost/qmd_test";
const RUN_POSTGRES_TESTS = process.env.QMD_ENABLE_POSTGRES_TESTS === "1";

let testConfigDir = "";

async function ensureTestDatabase(): Promise<void> {
  const admin = postgres("postgresql://teleclawd@localhost/postgres", {
    max: 1,
    idle_timeout: 1,
    connect_timeout: 5,
  });

  try {
    await admin.unsafe(`CREATE DATABASE qmd_test`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (!message.includes("already exists") && !message.includes("duplicate_database")) {
      throw err;
    }
  } finally {
    await admin.end({ timeout: 5 });
  }

  const sql = postgres(TEST_POSTGRES_URL, {
    max: 1,
    idle_timeout: 1,
    connect_timeout: 5,
  });
  try {
    await sql.unsafe(`CREATE EXTENSION IF NOT EXISTS vector`);
  } finally {
    await sql.end({ timeout: 5 });
  }
}

async function resetDatabase(): Promise<void> {
  const sql = postgres(TEST_POSTGRES_URL, {
    max: 1,
    idle_timeout: 1,
    connect_timeout: 5,
  });

  try {
    await sql.unsafe(`DROP TABLE IF EXISTS vectors CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS content_vectors CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS llm_cache CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS documents CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS content CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS path_contexts CASCADE`);
    await sql.unsafe(`DROP TABLE IF EXISTS collections CASCADE`);
    await sql.unsafe(`CREATE EXTENSION IF NOT EXISTS vector`);
  } finally {
    await sql.end({ timeout: 5 });
  }
}

async function setupConfigDir(): Promise<void> {
  testConfigDir = await mkdtemp(join(tmpdir(), "qmd-pg-config-"));
  process.env.QMD_CONFIG_DIR = testConfigDir;
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify({ collections: {} }), "utf-8");
}

async function cleanupConfigDir(): Promise<void> {
  if (testConfigDir) {
    await rm(testConfigDir, { recursive: true, force: true });
  }
  delete process.env.QMD_CONFIG_DIR;
}

async function importStoreModule() {
  vi.resetModules();
  process.env.QMD_BACKEND = "postgres";
  process.env.QMD_POSTGRES_URL = TEST_POSTGRES_URL;
  return await import("../src/store.js");
}

describe.skipIf(!RUN_POSTGRES_TESTS)("Postgres backend integration", () => {
  beforeAll(async () => {
    await ensureTestDatabase();
  });

  beforeEach(async () => {
    await resetDatabase();
    await setupConfigDir();
  });

  afterEach(async () => {
    await cleanupConfigDir();
    delete process.env.QMD_BACKEND;
    delete process.env.QMD_POSTGRES_URL;
  });

  test("createStore initializes postgres schema and pgvector index", async () => {
    const { createStore } = await importStoreModule();
    const store = createStore();

    try {
      expect(store.backend).toBe("postgres");
      store.ensureVecTable(3);

      const table = store.db.prepare(`SELECT to_regclass(current_schema() || '.vectors') AS name`).get() as {
        name: string | null;
      } | null;
      expect(table?.name).toBeTruthy();

      const indexes = store.db.prepare(`
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = current_schema() AND tablename = 'vectors'
      `).all() as { indexname: string }[];
      expect(indexes.some((idx) => idx.indexname === "idx_vectors_embedding_hnsw")).toBe(true);
    } finally {
      store.close();
    }
  });

  test("searchFTS uses postgres tsvector index", async () => {
    const { createStore } = await importStoreModule();
    const store = createStore();

    try {
      const now = new Date().toISOString();
      const body = "postgres vector index with tsvector ranking";
      const hash = "hash-fts-1";

      store.insertContent(hash, body, now);
      store.insertDocument("notes", "pg/fts.md", "Postgres FTS", hash, now, now);

      const results = store.searchFTS("postgres ranking", 10);
      expect(results.length).toBe(1);
      expect(results[0]?.displayPath).toBe("notes/pg/fts.md");
      expect(results[0]?.score).toBeGreaterThan(0);
    } finally {
      store.close();
    }
  });

  test("searchVec uses pgvector distance ordering", async () => {
    const { createStore, DEFAULT_EMBED_MODEL } = await importStoreModule();
    const store = createStore();

    try {
      store.ensureVecTable(3);
      const now = new Date().toISOString();

      const hash1 = "hash-vec-1";
      const hash2 = "hash-vec-2";

      store.insertContent(hash1, "alpha semantic content", now);
      store.insertDocument("notes", "pg/a.md", "Alpha", hash1, now, now);
      store.insertEmbedding(hash1, 0, 0, new Float32Array([1, 0, 0]), "test", now);

      store.insertContent(hash2, "beta semantic content", now);
      store.insertDocument("notes", "pg/b.md", "Beta", hash2, now, now);
      store.insertEmbedding(hash2, 0, 0, new Float32Array([0, 1, 0]), "test", now);

      const results = await store.searchVec(
        "semantic query",
        DEFAULT_EMBED_MODEL,
        5,
        undefined,
        undefined,
        [1, 0, 0],
      );

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]?.displayPath).toBe("notes/pg/a.md");
      expect(results[0]?.score).toBeGreaterThan(results[1]?.score ?? -1);
    } finally {
      store.close();
    }
  });
});
