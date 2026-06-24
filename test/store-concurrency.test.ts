/**
 * store-concurrency.test.ts - concurrent schema-init safety
 *
 * Reproduces cross-process races in cold store initialization: WAL migration,
 * FTS sync trigger rebuild, and CJK FTS normalization shadow-table rebuild.
 */
import { describe, test, expect } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { openDatabase } from "../src/db.ts";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const workerScript = join(thisDir, "_helpers", "store-init-worker.ts");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const WORKERS = isBunRuntime ? (process.platform === "darwin" ? 16 : 12) : 6;

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
  // On failure the joined worker stderr is surfaced by the assertion below.
  expect(failed.map(r => r.stderr.trim()).join("\n---\n")).toBe("");
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
  test("cold database: N processes initialize without colliding on FTS setup", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      const results = await openConcurrently(dbPath, WORKERS);
      expectAllSucceeded(results);
      expectSchemaIntact(dbPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);

  test("existing database: N processes reopen without rebuilding triggers", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      const [seed] = await openConcurrently(dbPath, 1);
      expect(seed.code).toBe(0);

      const results = await openConcurrently(dbPath, WORKERS);
      expectAllSucceeded(results);
      expectSchemaIntact(dbPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
