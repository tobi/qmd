/**
 * store-concurrency.test.ts - concurrent schema-init safety
 *
 * Reproduces cross-process races in cold store initialization: WAL migration,
 * FTS virtual-table CREATE (`table documents_fts already exists` on Bun/macOS),
 * FTS sync trigger rebuild, and CJK FTS normalization shadow-table rebuild.
 */
import { describe, test, expect } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { openDatabase } from "../src/db.ts";
import { createStore, syncConfigToDb } from "../src/store.ts";
import type { CollectionConfig } from "../src/collections.ts";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const workerScript = join(thisDir, "_helpers", "store-init-worker.ts");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const WORKERS = isBunRuntime ? (process.platform === "darwin" ? 16 : 12) : 6;
// A single unlucky schedule can miss the FTS CREATE race that Bun/macOS CI
// hits. Two cold trials keep the test tight without blowing the 60s budget.
const COLD_TRIALS = 2;

type WorkerResult = { code: number | null; stderr: string };

function runWorker(dbPath: string, startAtMs: number): Promise<WorkerResult> {
  const args = isBunRuntime
    ? [workerScript, dbPath, String(startAtMs)]
    : [tsxCli, workerScript, dbPath, String(startAtMs)];
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("close", (code) => resolve({ code, stderr }));
  });
}

async function openConcurrently(dbPath: string, n: number): Promise<WorkerResult[]> {
  const startAtMs = Date.now() + 1000;
  return Promise.all(Array.from({ length: n }, () => runWorker(dbPath, startAtMs)));
}

function expectAllSucceeded(results: WorkerResult[]): void {
  const failed = results.filter(r => r.code !== 0);
  const joined = failed.map(r => r.stderr.trim()).join("\n---\n");
  // On failure the joined worker stderr is surfaced by the assertion below.
  expect(joined).toBe("");
  expect(joined).not.toMatch(/already exists/i);
  expect(failed).toHaveLength(0);
}

function expectSchemaIntact(dbPath: string): void {
  const db = openDatabase(dbPath);
  try {
    const triggers = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'trigger'`)
      .all() as { name: string }[];
    expect(new Set(triggers.map(t => t.name))).toEqual(
      new Set(["documents_ai", "documents_ad", "documents_au"])
    );

    const fts = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'`)
      .get();
    expect(fts).toBeTruthy();

    const versionRow = db.prepare(`PRAGMA user_version`).get() as Record<string, number>;
    expect(Object.values(versionRow)[0]).toBeGreaterThanOrEqual(1);

    const cjkVersion = db
      .prepare(`SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'`)
      .get() as { value?: string } | undefined;
    expect(cjkVersion?.value).toBe("1");

    const leakedShadow = db
      .prepare(`SELECT name FROM sqlite_master WHERE name LIKE 'documents_fts_rebuild%'`)
      .all() as { name: string }[];
    expect(leakedShadow).toHaveLength(0);
  } finally {
    db.close();
  }
}

describe("concurrent store initialization", () => {
  test("FTS table create is serialized (IF NOT EXISTS + BEGIN IMMEDIATE)", () => {
    const storeSrc = readFileSync(join(projectRoot, "src", "store.ts"), "utf8");
    const startIdx = storeSrc.indexOf("const DOCUMENTS_FTS_DDL");
    const ensureIdx = storeSrc.indexOf("function ensureDocumentsFtsSchema(");
    const endIdx = storeSrc.indexOf("function cjkRebuildVersion(", ensureIdx);
    expect(startIdx).toBeGreaterThan(-1);
    expect(ensureIdx).toBeGreaterThan(startIdx);
    expect(endIdx).toBeGreaterThan(ensureIdx);

    const createBody = storeSrc.slice(startIdx, ensureIdx)
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n")
      .map(line => line.replace(/\/\/.*$/, ""))
      .join("\n");
    const ensureBody = storeSrc.slice(ensureIdx, endIdx)
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n")
      .map(line => line.replace(/\/\/.*$/, ""))
      .join("\n");

    expect(createBody).toMatch(/CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts/);
    expect(createBody).toContain("isAlreadyExistsError");
    expect(ensureBody).toContain("BEGIN IMMEDIATE");
    expect(ensureBody).toContain("createDocumentsFtsTable");
    // initializeDatabase must not autocommit-create the FTS table outside the lock.
    const initStart = storeSrc.indexOf("function initializeDatabase(");
    const initEnd = storeSrc.indexOf("function rowToNamedCollection(", initStart);
    const initBody = storeSrc.slice(initStart, initEnd);
    expect(initBody).not.toMatch(/CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts/);
    expect(initBody).toContain("ensureDocumentsFtsSchema");
  });

  test("cold database: N processes initialize without colliding on FTS setup", async () => {
    for (let trial = 0; trial < COLD_TRIALS; trial++) {
      const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
      const dbPath = join(dir, "index.sqlite");
      try {
        const results = await openConcurrently(dbPath, WORKERS);
        expectAllSucceeded(results);
        expectSchemaIntact(dbPath);
      } finally {
        await rm(dir, { recursive: true, force: true });
      }
    }
  }, 60_000);

  test("existing database: N processes reopen without rebuilding triggers", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      const [seed] = await openConcurrently(dbPath, 1);
      expect(seed!.code).toBe(0);

      const results = await openConcurrently(dbPath, WORKERS);
      expectAllSucceeded(results);
      expectSchemaIntact(dbPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);

  test("opening an initialized store does not take the write lock (#1133)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-1133-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      // Seed: the first open initializes the schema.
      const seed = createStore(dbPath);
      seed.close();

      // Hold the write lock from a second connection, as `qmd update` does.
      const writer = openDatabase(dbPath);
      writer.exec("BEGIN IMMEDIATE");
      writer.prepare(`INSERT INTO llm_cache (hash, result, created_at) VALUES ('x', 'y', 'z')`).run();
      try {
        // A read-only open must not wait for the writer: the schema is current.
        process.env.QMD_SQLITE_BUSY_TIMEOUT = "3000";
        try {
          const started = Date.now();
          const store = createStore(dbPath);
          try {
            expect(Date.now() - started).toBeLessThan(3000);
            expect(store.getStatusSummary().totalDocuments).toBe(0);
          } finally {
            store.close();
          }
        } finally {
          delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
        }
      } finally {
        writer.exec("ROLLBACK");
        writer.close();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);

  test("syncing an unchanged config does not take the write lock (#1144)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-1144-"));
    const dbPath = join(dir, "index.sqlite");
    const config: CollectionConfig = { collections: {} };
    try {
      // Seed: the first open initializes the schema, then stamp the config
      // hash the way the CLI's getStore() does (createStore + syncConfigToDb).
      const seed = createStore(dbPath);
      syncConfigToDb(seed.db, config);
      seed.close();

      // Hold the write lock from a second connection, as `qmd update` does.
      const writer = openDatabase(dbPath);
      writer.exec("BEGIN IMMEDIATE");
      writer.prepare(`INSERT INTO llm_cache (hash, result, created_at) VALUES ('x', 'y', 'z')`).run();
      try {
        // The stored hash already matches this config, so syncing it must not
        // wait for the writer: no transaction is needed at all.
        process.env.QMD_SQLITE_BUSY_TIMEOUT = "3000";
        try {
          const started = Date.now();
          const db = openDatabase(dbPath);
          try {
            syncConfigToDb(db, config);
            expect(Date.now() - started).toBeLessThan(3000);
            const row = db
              .prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`)
              .get() as { value: string };
            expect(row.value).toBe(createHash("sha256").update(JSON.stringify(config)).digest("hex"));
          } finally {
            db.close();
          }
        } finally {
          delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
        }
      } finally {
        writer.exec("ROLLBACK");
        writer.close();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
