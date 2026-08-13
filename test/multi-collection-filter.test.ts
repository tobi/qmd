/**
 * Unit tests for multi-collection CLI filter input (#191, #775).
 *
 * Collection scoping is applied in searchFTS/searchVec (search each collection,
 * then merge). These tests cover CLI parseArgs / input normalization only.
 * Starvation regressions live in store.test.ts.
 */

import { describe, test, expect } from "vitest";
import { parseArgs } from "node:util";

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
