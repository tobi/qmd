import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  insertContext,
  upsertStoreCollection,
  getStoreContexts,
} from "../src/store.js";

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-insert-context-"));
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {
    // ignore
  }
});

function freshDbPath(): string {
  return join(testDir, `ctx-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

describe("insertContext", () => {
  test("looks up store_collections by name and writes path context", () => {
    const store = createStore(freshDbPath());
    try {
      upsertStoreCollection(store.db, "docs", { path: "/tmp/docs" });
      insertContext(store.db, "docs", "/api", "API documentation");
      expect(getStoreContexts(store.db)).toEqual([
        { collection: "docs", path: "/api", context: "API documentation" },
      ]);
    } finally {
      store.close();
    }
  });

  test("overwrites existing context for the same path prefix", () => {
    const store = createStore(freshDbPath());
    try {
      upsertStoreCollection(store.db, "docs", { path: "/tmp/docs" });
      insertContext(store.db, "docs", "/api", "old");
      insertContext(store.db, "docs", "/api", "new");
      expect(getStoreContexts(store.db)).toEqual([
        { collection: "docs", path: "/api", context: "new" },
      ]);
    } finally {
      store.close();
    }
  });

  test("throws a collection-not-found error, not no such column: id", () => {
    const store = createStore(freshDbPath());
    try {
      expect(() => insertContext(store.db, "missing", "/api", "nope")).toThrow(
        "Collection 'missing' not found",
      );
    } finally {
      store.close();
    }
  });

  test("store_collections has no id column", () => {
    const store = createStore(freshDbPath());
    try {
      const cols = store.db.prepare("PRAGMA table_info(store_collections)").all() as { name: string }[];
      expect(cols.map((c) => c.name)).not.toContain("id");
      expect(cols.map((c) => c.name)).toContain("name");
    } finally {
      store.close();
    }
  });
});
