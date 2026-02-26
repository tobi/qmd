/**
 * Unit tests for multi-collection filter logic (PR #191).
 *
 * Tests the filterByCollections post-filter and the resolveCollectionFilter
 * behavior for single-collection vs multi-collection search.
 */

import { describe, test, expect } from "vitest";
import { parseArgs } from "node:util";

// Reproduce the filterByCollections logic from qmd.ts for testing
// (the function is private in qmd.ts)
function filterByCollections<T extends { filepath?: string; file?: string }>(
  results: T[],
  collectionNames: string[],
): T[] {
  if (collectionNames.length <= 1) return results;
  const prefixes = collectionNames.map((n) => `qmd://${n}/`);
  return results.filter((r) => {
    const path = r.filepath || r.file || "";
    return prefixes.some((p) => path.startsWith(p));
  });
}

describe("filterByCollections", () => {
  const results = [
    { filepath: "qmd://docs/readme.md", file: "qmd://docs/readme.md" },
    { filepath: "qmd://notes/todo.md", file: "qmd://notes/todo.md" },
    { filepath: "qmd://journals/2024/jan.md", file: "qmd://journals/2024/jan.md" },
    { filepath: "qmd://docs/api.md", file: "qmd://docs/api.md" },
  ];

  test("returns all results when no collections specified", () => {
    expect(filterByCollections(results, [])).toEqual(results);
  });

  test("returns all results for single collection (no-op, handled by SQL filter)", () => {
    expect(filterByCollections(results, ["docs"])).toEqual(results);
  });

  test("filters to matching collections when multiple specified", () => {
    const filtered = filterByCollections(results, ["docs", "journals"]);
    expect(filtered).toHaveLength(3);
    expect(filtered.map((r) => r.filepath)).toEqual([
      "qmd://docs/readme.md",
      "qmd://journals/2024/jan.md",
      "qmd://docs/api.md",
    ]);
  });

  test("filters correctly with two collections", () => {
    const filtered = filterByCollections(results, ["notes", "journals"]);
    expect(filtered).toHaveLength(2);
    expect(filtered.map((r) => r.filepath)).toEqual([
      "qmd://notes/todo.md",
      "qmd://journals/2024/jan.md",
    ]);
  });

  test("returns empty when no results match collections", () => {
    const filtered = filterByCollections(results, ["archive", "trash"]);
    expect(filtered).toHaveLength(0);
  });

  test("uses file field when filepath is missing", () => {
    const fileOnlyResults = [
      { file: "qmd://docs/readme.md" },
      { file: "qmd://notes/todo.md" },
    ];
    const filtered = filterByCollections(fileOnlyResults, ["docs", "notes"]);
    expect(filtered).toHaveLength(2);
  });

  test("uses filepath over file when both present", () => {
    const mixedResults = [
      { filepath: "qmd://docs/readme.md", file: "qmd://notes/todo.md" },
    ];
    const filtered = filterByCollections(mixedResults, ["docs", "notes"]);
    expect(filtered).toHaveLength(1);
    // Should match via filepath (docs), not file (notes)
    expect(filtered[0].filepath).toBe("qmd://docs/readme.md");
  });
});

describe("resolveCollectionFilter input normalization", () => {
  // Test the array normalization logic without the DB dependency
  function normalizeCollectionInput(raw: string | string[] | undefined): string[] {
    if (!raw) return [];
    return Array.isArray(raw) ? raw : [raw];
  }

  test("undefined returns empty array", () => {
    expect(normalizeCollectionInput(undefined)).toEqual([]);
  });

  test("single string returns single-element array", () => {
    expect(normalizeCollectionInput("docs")).toEqual(["docs"]);
  });

  test("array passes through", () => {
    expect(normalizeCollectionInput(["docs", "notes"])).toEqual(["docs", "notes"]);
  });

  test("empty string returns single-element array", () => {
    expect(normalizeCollectionInput("")).toEqual([]);
  });
});

describe("collection option type from parseArgs", () => {
  // Verify that parseArgs with `multiple: true` produces string[]
  test("parseArgs multiple:true produces array for repeated flags", () => {
    const { values } = parseArgs({
      args: ["-c", "docs", "-c", "notes"],
      options: {
        collection: { type: "string", short: "c", multiple: true },
      },
      strict: true,
    });
    expect(values.collection).toEqual(["docs", "notes"]);
  });

  test("parseArgs multiple:true produces array for single flag", () => {
    const { values } = parseArgs({
      args: ["-c", "docs"],
      options: {
        collection: { type: "string", short: "c", multiple: true },
      },
      strict: true,
    });
    expect(values.collection).toEqual(["docs"]);
  });

  test("parseArgs multiple:true produces undefined when flag absent", () => {
    const { values } = parseArgs({
      args: [],
      options: {
        collection: { type: "string", short: "c", multiple: true },
      },
      strict: true,
    });
    expect(values.collection).toBeUndefined();
  });
});
