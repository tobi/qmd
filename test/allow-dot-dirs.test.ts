/**
 * Integration tests for the per-collection `allowDotDirs` config.
 *
 * Hidden (dot-prefixed) files/dirs are skipped by default. `allowDotDirs` opts
 * specific dot-dirs back into indexing, while every other dot-prefixed segment
 * (.git, .cache, …) stays excluded.
 */
import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/index.js";

let root: string;

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "qmd-allow-dot-dirs-"));
  await mkdir(join(root, "docs"), { recursive: true });
  await mkdir(join(root, ".aidocs", "sessions"), { recursive: true });
  await mkdir(join(root, ".git"), { recursive: true });
  await writeFile(join(root, "docs", "readme.md"), "# Readme\n\nPublic guide.\n");
  await writeFile(join(root, ".aidocs", "sessions", "agentnote.md"), "# Session note\n\nAgent session summary.\n");
  await writeFile(join(root, ".git", "gitinternal.md"), "# Git internals\n\nShould never be indexed.\n");
});

afterAll(async () => {
  try {
    await rm(root, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }
});

function freshDb(): string {
  return join(root, `idx-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

async function indexedPaths(allowDotDirs?: string[]): Promise<string> {
  const store = await createStore({
    dbPath: freshDb(),
    config: {
      collections: {
        proj: { path: root, pattern: "**/*.md", ...(allowDotDirs ? { allowDotDirs } : {}) },
      },
    },
  });
  await store.update();
  const rows = (store.internal as { db: { prepare(sql: string): { all(): unknown[] } } }).db
    .prepare("SELECT path FROM documents WHERE active = 1")
    .all() as { path: string }[];
  await store.close();
  return rows.map(r => r.path).join("\n");
}

describe("allowDotDirs", () => {
  test("dot-dirs are skipped by default (unchanged behavior)", async () => {
    const paths = await indexedPaths();
    expect(paths).toContain("readme");
    expect(paths).not.toContain("agentnote"); // .aidocs/ not indexed
    expect(paths).not.toContain("gitinternal"); // .git/ not indexed
  });

  test("listed dot-dirs are indexed; other dot-dirs stay excluded", async () => {
    const paths = await indexedPaths([".aidocs"]);
    expect(paths).toContain("readme");
    expect(paths).toContain("agentnote"); // .aidocs/sessions/agentnote.md indexed
    expect(paths).not.toContain("gitinternal"); // .git/ still excluded
  });
});
