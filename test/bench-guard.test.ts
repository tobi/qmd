/**
 * Bench collection guardrails (#716).
 *
 * `qmd bench` used to run every fixture query against a missing/empty
 * collection and print a wall of 0.00. These tests lock the fail-fast
 * checks and the all-zero stderr warning.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeFileSync } from "node:fs";
import {
  assertBenchCollectionReady,
  benchSummaryAllZero,
  allZeroBenchWarning,
  runBenchmark,
} from "../src/bench/bench.js";
import { createStore } from "../src/index.js";

describe("assertBenchCollectionReady", () => {
  test("throws when the named collection is missing", () => {
    expect(() => assertBenchCollectionReady([], "eval-docs")).toThrow(/Collection not found: eval-docs/);
    expect(() => assertBenchCollectionReady(
      [{ name: "notes", active_count: 3 }],
      "eval-docs",
    )).toThrow(/Available: notes/);
  });

  test("throws when the named collection has no indexed documents", () => {
    expect(() => assertBenchCollectionReady(
      [{ name: "eval-docs", active_count: 0 }],
      "eval-docs",
    )).toThrow(/has no indexed documents/);
  });

  test("throws when no collection is named and nothing is indexed", () => {
    expect(() => assertBenchCollectionReady([])).toThrow(/No indexed documents found/);
    expect(() => assertBenchCollectionReady(
      [{ name: "notes", active_count: 0 }],
    )).toThrow(/No indexed documents found/);
  });

  test("allows a named collection that has documents", () => {
    expect(() => assertBenchCollectionReady(
      [{ name: "eval-docs", active_count: 6 }],
      "eval-docs",
    )).not.toThrow();
  });

  test("allows an unnamed bench when any collection has documents", () => {
    expect(() => assertBenchCollectionReady(
      [{ name: "notes", active_count: 2 }],
    )).not.toThrow();
  });
});

describe("benchSummaryAllZero", () => {
  const zero = {
    avg_precision: 0,
    avg_recall: 0,
    avg_recall_at_1: 0,
    avg_recall_at_3: 0,
    avg_recall_at_5: 0,
    avg_mrr: 0,
    avg_f1: 0,
    avg_latency_ms: 12,
  };

  test("is true when every backend scored zero", () => {
    expect(benchSummaryAllZero({ bm25: zero, vector: { ...zero } })).toBe(true);
  });

  test("is false when any backend has a hit", () => {
    expect(benchSummaryAllZero({
      bm25: { ...zero, avg_recall: 0.5, avg_mrr: 1 },
      vector: zero,
    })).toBe(false);
  });

  test("is false for an empty summary", () => {
    expect(benchSummaryAllZero({})).toBe(false);
  });
});

describe("allZeroBenchWarning", () => {
  test("points at qmd ls for the fixture collection", () => {
    expect(allZeroBenchWarning("eval-docs")).toContain("qmd ls eval-docs");
    expect(allZeroBenchWarning()).toContain("qmd ls");
  });
});

describe("runBenchmark collection guard", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "qmd-bench-guard-"));
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  function writeFixture(collection: string, query = "API versioning", expected = "api.md") {
    const fixturePath = join(dir, "fixture.json");
    writeFileSync(fixturePath, JSON.stringify({
      description: "guardrail fixture",
      version: 1,
      collection,
      queries: [{
        id: "exact-api",
        query,
        type: "exact",
        description: "keyword",
        expected_files: [expected],
        expected_in_top_k: 1,
      }],
    }));
    return fixturePath;
  }

  test("errors before searching when the fixture collection does not exist", async () => {
    const fixturePath = writeFixture("eval-docs");
    const dbPath = join(dir, "index.sqlite");
    await expect(runBenchmark(fixturePath, {
      json: true,
      dbPath,
      backends: ["bm25"],
      config: { collections: { notes: { path: dir, pattern: "**/*.md" } } },
    })).rejects.toThrow(/Collection not found: eval-docs/);
  });

  test("errors before searching when the collection exists but has no documents", async () => {
    const docs = join(dir, "docs");
    await mkdir(docs, { recursive: true });
    const fixturePath = writeFixture("docs");
    const dbPath = join(dir, "index.sqlite");
    await expect(runBenchmark(fixturePath, {
      json: true,
      dbPath,
      backends: ["bm25"],
      config: { collections: { docs: { path: docs, pattern: "**/*.md" } } },
    })).rejects.toThrow(/has no indexed documents/);
  });

  test("runs bm25 against an indexed collection", async () => {
    const docs = join(dir, "docs");
    await mkdir(docs, { recursive: true });
    await writeFile(join(docs, "api.md"), "# API versioning\n\nUse /v1 and /v2 endpoints.\n");
    const dbPath = join(dir, "index.sqlite");
    const config = { collections: { docs: { path: docs, pattern: "**/*.md" } } };
    const store = await createStore({ dbPath, config });
    await store.update();
    await store.close();

    const fixturePath = writeFixture("docs", "API versioning", "api.md");
    const stderr: string[] = [];
    const origWrite = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
      stderr.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString());
      return origWrite(chunk as string, ...(rest as []));
    }) as typeof process.stderr.write;
    try {
      const result = await runBenchmark(fixturePath, {
        json: true,
        dbPath,
        backends: ["bm25"],
        config,
      });
      expect(result.summary.bm25?.avg_recall).toBeGreaterThan(0);
      expect(stderr.join("")).not.toContain("All benchmark scores were 0.00");
    } finally {
      process.stderr.write = origWrite;
    }
  });

  test("warns on stderr when every backend scores zero", async () => {
    const docs = join(dir, "docs");
    await mkdir(docs, { recursive: true });
    await writeFile(join(docs, "api.md"), "# API versioning\n\nUse /v1 and /v2 endpoints.\n");
    const dbPath = join(dir, "index.sqlite");
    const config = { collections: { docs: { path: docs, pattern: "**/*.md" } } };
    const store = await createStore({ dbPath, config });
    await store.update();
    await store.close();

    const fixturePath = writeFixture("docs", "zzzz-no-such-token-in-corpus", "missing.md");
    const stderr: string[] = [];
    const origWrite = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
      stderr.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString());
      return origWrite(chunk as string, ...(rest as []));
    }) as typeof process.stderr.write;
    try {
      const result = await runBenchmark(fixturePath, {
        json: true,
        dbPath,
        backends: ["bm25"],
        config,
      });
      expect(result.summary.bm25?.avg_recall).toBe(0);
      expect(stderr.join("")).toContain("All benchmark scores were 0.00");
      expect(stderr.join("")).toContain("qmd ls docs");
    } finally {
      process.stderr.write = origWrite;
    }
  });
});
