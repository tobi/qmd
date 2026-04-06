/**
 * Tests for the benchmark scoring functions.
 */

import { describe, test, expect } from "vitest";
import { normalizePath, pathsMatch, scoreResults } from "../src/bench/score.js";

describe("normalizePath", () => {
  test("lowercases path", () => {
    expect(normalizePath("Resources/Concepts/Context Engineering.md"))
      .toBe("resources/concepts/context engineering.md");
  });

  test("strips qmd:// prefix", () => {
    expect(normalizePath("qmd://collection/docs/readme.md"))
      .toBe("docs/readme.md");
  });

  test("strips leading/trailing slashes", () => {
    expect(normalizePath("/docs/readme.md/")).toBe("docs/readme.md");
  });

  test("handles plain filename", () => {
    expect(normalizePath("readme.md")).toBe("readme.md");
  });
});

describe("pathsMatch", () => {
  test("exact match", () => {
    expect(pathsMatch("docs/readme.md", "docs/readme.md")).toBe(true);
  });

  test("case-insensitive match", () => {
    expect(pathsMatch("Docs/README.md", "docs/readme.md")).toBe(true);
  });

  test("suffix match (result is longer)", () => {
    expect(pathsMatch("/full/path/docs/readme.md", "docs/readme.md")).toBe(true);
  });

  test("suffix match (expected is longer)", () => {
    expect(pathsMatch("readme.md", "docs/readme.md")).toBe(true);
  });

  test("qmd:// prefix handled", () => {
    expect(pathsMatch("qmd://col/docs/readme.md", "docs/readme.md")).toBe(true);
  });

  test("different files don't match", () => {
    expect(pathsMatch("docs/readme.md", "docs/other.md")).toBe(false);
  });
});

describe("scoreResults", () => {
  test("perfect score: all expected in top-k", () => {
    const result = scoreResults(
      ["a.md", "b.md", "c.md"],
      ["a.md", "b.md"],
      2,
    );
    expect(result.precision_at_k).toBe(1);
    expect(result.recall).toBe(1);
    expect(result.mrr).toBe(1);
    expect(result.f1).toBe(1);
    expect(result.hits_at_k).toBe(2);
  });

  test("zero score: none found", () => {
    const result = scoreResults(
      ["x.md", "y.md", "z.md"],
      ["a.md", "b.md"],
      2,
    );
    expect(result.precision_at_k).toBe(0);
    expect(result.recall).toBe(0);
    expect(result.mrr).toBe(0);
    expect(result.f1).toBe(0);
    expect(result.hits_at_k).toBe(0);
  });

  test("partial: found outside top-k", () => {
    const result = scoreResults(
      ["x.md", "y.md", "a.md"],
      ["a.md"],
      1,
    );
    expect(result.precision_at_k).toBe(0); // not in top-1
    expect(result.recall).toBe(1); // found somewhere
    expect(result.mrr).toBeCloseTo(1 / 3); // rank 3
    expect(result.hits_at_k).toBe(0);
  });

  test("MRR: first relevant at rank 2", () => {
    const result = scoreResults(
      ["x.md", "a.md", "b.md"],
      ["a.md", "b.md"],
      3,
    );
    expect(result.mrr).toBeCloseTo(0.5); // 1/2
  });

  test("empty results", () => {
    const result = scoreResults([], ["a.md"], 1);
    expect(result.precision_at_k).toBe(0);
    expect(result.recall).toBe(0);
    expect(result.mrr).toBe(0);
  });

  test("empty expected", () => {
    const result = scoreResults(["a.md"], [], 1);
    expect(result.precision_at_k).toBe(0);
    expect(result.recall).toBe(0);
  });
});
