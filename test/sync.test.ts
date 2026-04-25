/**
 * sync.test.ts - Tests for the syncCollection() mtime-first differential sync
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, writeFile, mkdir, rm, unlink, rename, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { statSync, writeFileSync } from "node:fs";
import YAML from "yaml";
import {
  createStore,
  syncCollection,
  reindexCollection,
  hashContent,
  syncConfigToDb,
  type Store,
  type SyncResult,
} from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

// =============================================================================
// Test Helpers
// =============================================================================

let testDir: string;
let configDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-sync-test-"));
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {}
});

async function createTestEnv() {
  const dir = await mkdtemp(join(testDir, "coll-"));
  const dbPath = join(testDir, `db-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;

  const config: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(config));

  const store = createStore(dbPath);
  return { dir, store };
}

async function cleanupEnv(store: Store) {
  store.close();
  delete process.env.QMD_CONFIG_DIR;
}

// =============================================================================
// syncCollection Tests
// =============================================================================

describe("syncCollection", () => {
  test("indexes new files on first sync", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "hello.md"), "# Hello\n\nWorld\n");
    await writeFile(join(dir, "goodbye.md"), "# Goodbye\n\nSee you later\n");

    const result = await syncCollection(store, dir, "**/*.md", "test");

    expect(result.added).toBe(2);
    expect(result.updated).toBe(0);
    expect(result.removed).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.filesScanned).toBe(2);

    await cleanupEnv(store);
  });

  test("skips unchanged files by mtime", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "stable.md"), "# Stable\n\nThis doesn't change\n");

    // First sync to populate index
    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(1);

    // Second sync — file hasn't changed
    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.added).toBe(0);
    expect(second.updated).toBe(0);
    expect(second.unchanged).toBe(1);
    expect(second.skippedMtime).toBe(1);

    await cleanupEnv(store);
  });

  test("detects modified files via mtime then hash", async () => {
    const { dir, store } = await createTestEnv();
    const filePath = join(dir, "changing.md");

    await writeFile(filePath, "# Version 1\n\nOriginal content\n");

    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(1);

    // Modify the file — ensure mtime changes (add small delay)
    await new Promise(r => setTimeout(r, 50));
    await writeFile(filePath, "# Version 2\n\nUpdated content\n");

    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.updated).toBe(1);
    expect(second.unchanged).toBe(0);

    await cleanupEnv(store);
  });

  test("detects mtime change but hash unchanged (touch)", async () => {
    const { dir, store } = await createTestEnv();
    const filePath = join(dir, "touched.md");

    await writeFile(filePath, "# Touched\n\nContent stays the same\n");

    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(1);

    // Touch the file — changes mtime but not content
    await new Promise(r => setTimeout(r, 50));
    const now = new Date();
    await utimes(filePath, now, now);

    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.updated).toBe(0);
    expect(second.unchanged).toBe(1);
    expect(second.skippedHash).toBe(1);
    expect(second.skippedMtime).toBe(0);

    await cleanupEnv(store);
  });

  test("removes deleted files", async () => {
    const { dir, store } = await createTestEnv();
    const filePath = join(dir, "ephemeral.md");

    await writeFile(filePath, "# Temporary\n\nWill be deleted\n");
    await writeFile(join(dir, "keeper.md"), "# Keeper\n\nStays around\n");

    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(2);

    // Delete one file
    await unlink(filePath);

    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.removed).toBe(1);
    expect(second.unchanged).toBe(1);

    await cleanupEnv(store);
  });

  test("detects renamed files and reuses embeddings", async () => {
    const { dir, store } = await createTestEnv();
    const content = "# Renameable\n\nThis file will be renamed but content stays the same\n";
    const oldPath = join(dir, "old-name.md");
    const newPath = join(dir, "new-name.md");

    await writeFile(oldPath, content);

    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(1);

    // Rename the file
    await rename(oldPath, newPath);

    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.renamed).toBe(1);
    expect(second.added).toBe(0);
    expect(second.removed).toBe(0);

    await cleanupEnv(store);
  });

  test("handles new + deleted + unchanged in same sync", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "stays.md"), "# Stays\n\nPermanent file\n");
    await writeFile(join(dir, "goes.md"), "# Goes\n\nWill be deleted\n");

    const first = await syncCollection(store, dir, "**/*.md", "test");
    expect(first.added).toBe(2);

    // Delete one, add another
    await unlink(join(dir, "goes.md"));
    await writeFile(join(dir, "arrives.md"), "# Arrives\n\nBrand new file\n");

    const second = await syncCollection(store, dir, "**/*.md", "test");
    expect(second.added).toBe(1);
    expect(second.removed).toBe(1);
    expect(second.unchanged).toBe(1);

    await cleanupEnv(store);
  });

  test("skips empty files", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "empty.md"), "");
    await writeFile(join(dir, "whitespace.md"), "   \n\n  \n");
    await writeFile(join(dir, "real.md"), "# Real\n\nHas content\n");

    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.added).toBe(1);
    expect(result.filesScanned).toBe(3);

    await cleanupEnv(store);
  });

  test("skips hidden files and directories", async () => {
    const { dir, store } = await createTestEnv();

    await mkdir(join(dir, ".hidden"), { recursive: true });
    await writeFile(join(dir, ".hidden", "secret.md"), "# Secret\n");
    await writeFile(join(dir, ".dotfile.md"), "# Dotfile\n");
    await writeFile(join(dir, "visible.md"), "# Visible\n\nNormal file\n");

    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.added).toBe(1);

    await cleanupEnv(store);
  });

  test("respects ignore patterns", async () => {
    const { dir, store } = await createTestEnv();

    await mkdir(join(dir, "drafts"), { recursive: true });
    await writeFile(join(dir, "drafts", "wip.md"), "# WIP\n\nWork in progress\n");
    await writeFile(join(dir, "published.md"), "# Published\n\nDone\n");

    const result = await syncCollection(store, dir, "**/*.md", "test", {
      ignorePatterns: ["drafts/**"],
    });
    expect(result.added).toBe(1);

    await cleanupEnv(store);
  });

  test("handles subdirectories", async () => {
    const { dir, store } = await createTestEnv();

    await mkdir(join(dir, "sub", "deep"), { recursive: true });
    await writeFile(join(dir, "root.md"), "# Root\n");
    await writeFile(join(dir, "sub", "child.md"), "# Child\n");
    await writeFile(join(dir, "sub", "deep", "leaf.md"), "# Leaf\n");

    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.added).toBe(3);
    expect(result.filesScanned).toBe(3);

    await cleanupEnv(store);
  });

  test("first sync after migration populates disk_mtime for all docs", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "legacy.md"), "# Legacy\n\nExisting doc\n");

    // Simulate pre-migration: use reindexCollection (which sets disk_mtime),
    // then NULL it out to simulate legacy state
    await reindexCollection(store, dir, "**/*.md", "test");
    store.db.prepare(`UPDATE documents SET disk_mtime = NULL WHERE collection = ?`).run("test");

    // Verify it's NULL
    const before = store.db.prepare(
      `SELECT disk_mtime FROM documents WHERE collection = ? AND active = 1`
    ).get("test") as { disk_mtime: string | null };
    expect(before.disk_mtime).toBeNull();

    // Sync should read the file (mtime check fails due to NULL) and populate disk_mtime
    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.unchanged).toBe(1);
    expect(result.skippedHash).toBe(1); // mtime changed (null→value), hash same

    const after = store.db.prepare(
      `SELECT disk_mtime FROM documents WHERE collection = ? AND active = 1`
    ).get("test") as { disk_mtime: string | null };
    expect(after.disk_mtime).not.toBeNull();

    await cleanupEnv(store);
  });

  test("sync results match expectations for mixed operations", async () => {
    const { dir, store } = await createTestEnv();

    // Initial state: 3 files
    await writeFile(join(dir, "a.md"), "# File A\n");
    await writeFile(join(dir, "b.md"), "# File B\n");
    await writeFile(join(dir, "c.md"), "# File C\n");

    await syncCollection(store, dir, "**/*.md", "test");

    // Modify b, delete c, add d, rename a→e (same content)
    await new Promise(r => setTimeout(r, 50));
    await writeFile(join(dir, "b.md"), "# File B Updated\n");
    await unlink(join(dir, "c.md"));
    await writeFile(join(dir, "d.md"), "# File D\n");
    await rename(join(dir, "a.md"), join(dir, "e.md"));

    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.updated).toBe(1);   // b.md content changed
    expect(result.removed).toBe(1);   // c.md deleted
    expect(result.added).toBe(1);     // d.md new
    expect(result.renamed).toBe(1);   // a.md → e.md

    await cleanupEnv(store);
  });

  test("fires onProgress callback", async () => {
    const { dir, store } = await createTestEnv();

    await writeFile(join(dir, "one.md"), "# One\n");
    await writeFile(join(dir, "two.md"), "# Two\n");

    const phases: string[] = [];
    await syncCollection(store, dir, "**/*.md", "test", {
      onProgress: (info) => {
        if (!phases.includes(info.phase)) {
          phases.push(info.phase);
        }
      },
    });

    expect(phases).toContain("scanning");
    expect(phases).toContain("processing");

    await cleanupEnv(store);
  });

  test("cleans up orphaned content after deletions", async () => {
    const { dir, store } = await createTestEnv();

    const uniqueContent = `# Unique ${Date.now()}\n\nContent that will be orphaned\n`;
    await writeFile(join(dir, "orphan.md"), uniqueContent);

    await syncCollection(store, dir, "**/*.md", "test");

    // Verify content exists
    const hash = await hashContent(uniqueContent);
    const contentBefore = store.db.prepare(
      `SELECT hash FROM content WHERE hash = ?`
    ).get(hash) as { hash: string } | undefined;
    expect(contentBefore).toBeDefined();

    // Delete the file and sync
    await unlink(join(dir, "orphan.md"));
    const result = await syncCollection(store, dir, "**/*.md", "test");
    expect(result.removed).toBe(1);
    expect(result.orphanedCleaned).toBeGreaterThanOrEqual(1);

    await cleanupEnv(store);
  });
});
