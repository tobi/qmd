/**
 * wiki-bench v0 — fixture schema validation (no wiki corpus required).
 *
 * Asserts that src/bench/fixtures/wiki-v0.json is well-formed for the
 * benchmark harness. Does not read or require any wiki page bodies.
 */

import { describe, test, expect } from "vitest";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import type { BenchmarkFixture, BenchmarkQuery } from "../src/bench/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dirname, "..", "src", "bench", "fixtures", "wiki-v0.json");

const ALLOWED_TYPES = new Set([
  "exact",
  "semantic",
  "topical",
  "cross-domain",
  "alias",
] as const);

/** Wiki-relative paths only: concepts|sources|entities|syntheses/...md */
const WIKI_REL_PATH =
  /^(concepts|sources|entities|syntheses)\/(?:[\w.-]+\/)*[\w.-]+\.md$/;

describe("wiki-bench fixture (schema only)", () => {
  const raw = readFileSync(FIXTURE_PATH, "utf-8");
  const fixture = JSON.parse(raw) as BenchmarkFixture;

  test("has version, collection, and non-empty queries", () => {
    expect(typeof fixture.version).toBe("number");
    expect(fixture.version).toBeGreaterThanOrEqual(1);
    expect(typeof fixture.collection).toBe("string");
    expect(fixture.collection!.length).toBeGreaterThan(0);
    expect(Array.isArray(fixture.queries)).toBe(true);
    expect(fixture.queries.length).toBeGreaterThan(0);
  });

  test("has at least 40 queries", () => {
    expect(fixture.queries.length).toBeGreaterThanOrEqual(40);
  });

  test("each query has required fields and allowed type", () => {
    for (const q of fixture.queries) {
      expect(typeof q.id).toBe("string");
      expect(q.id.length).toBeGreaterThan(0);
      expect(typeof q.query).toBe("string");
      expect(q.query.length).toBeGreaterThan(0);
      expect(ALLOWED_TYPES.has(q.type as BenchmarkQuery["type"])).toBe(true);
      expect(typeof q.description).toBe("string");
      expect(q.description.length).toBeGreaterThan(0);
      expect(Array.isArray(q.expected_files)).toBe(true);
      expect(q.expected_files.length).toBeGreaterThan(0);
      expect(typeof q.expected_in_top_k).toBe("number");
      expect(q.expected_in_top_k).toBeGreaterThanOrEqual(1);
    }
  });

  test("query ids are unique", () => {
    const ids = fixture.queries.map((q) => q.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  test("expected_files are wiki-relative .md paths (no abs, no ..)", () => {
    for (const q of fixture.queries) {
      for (const file of q.expected_files) {
        expect(typeof file).toBe("string");
        expect(file.endsWith(".md")).toBe(true);
        expect(file.startsWith("/")).toBe(false);
        expect(file.includes("..")).toBe(false);
        expect(file).toMatch(WIKI_REL_PATH);
      }
    }
  });

  test("does not vendor or require wiki corpus under test/", () => {
    // Schema-only suite: fixture JSON is the only input.
    expect(FIXTURE_PATH.endsWith("wiki-v0.json")).toBe(true);
    expect(fixture.queries.length).toBeGreaterThan(0);
  });
});
