/**
 * Query-expansion bound/dedup policy (pure logic, no model or runtime dependency).
 */

import { describe, test, expect } from "vitest";
import {
  sanitizeQueryExpansions,
  normalizeExpansionText,
  MAX_QUERY_EXPANSIONS,
  MAX_QUERY_EXPANSIONS_PER_TYPE,
} from "../src/query-expansion.js";

type Expansion = { type: string; text: string };

function sanitize(items: Expansion[], original = "an unrelated original query"): Expansion[] {
  return sanitizeQueryExpansions(items, (item) => item.text, original);
}

describe("normalizeExpansionText", () => {
  test("trims, collapses internal whitespace, and lowercases", () => {
    expect(normalizeExpansionText("  Auth \t  Setup \n")).toBe("auth setup");
  });
});

describe("sanitizeQueryExpansions", () => {
  test("leaves a normal small expansion untouched", () => {
    const small: Expansion[] = [
      { type: "lex", text: "oauth2 login" },
      { type: "vec", text: "how to configure authentication" },
      { type: "hyde", text: "Information about authentication setup" },
    ];

    expect(sanitize(small)).toEqual(small);
  });

  test("collapses a repeated-hyde dump to the per-type cap", () => {
    const repeated: Expansion[] = Array.from({ length: 26 }, () => ({
      type: "hyde",
      text: "Information about authentication",
    }));
    const uniqueTail: Expansion[] = [
      { type: "hyde", text: "A second hyde document" },
      { type: "hyde", text: "A third hyde document" },
    ];

    const result = sanitize([...repeated, ...uniqueTail]);

    expect(result).toEqual([
      { type: "hyde", text: "Information about authentication" },
      { type: "hyde", text: "A second hyde document" },
    ]);
    expect(result).toHaveLength(MAX_QUERY_EXPANSIONS_PER_TYPE);
  });

  test("keeps stable source order while enforcing 2 per type and 6 total", () => {
    const oversized: Expansion[] = [
      { type: "hyde", text: "hyde-a" },
      { type: "lex", text: "lex-a" },
      { type: "vec", text: "vec-a" },
      { type: "hyde", text: "hyde-b" },
      { type: "lex", text: "lex-b" },
      { type: "vec", text: "vec-b" },
      { type: "hyde", text: "hyde-c" },
      { type: "lex", text: "lex-c" },
      { type: "vec", text: "vec-c" },
    ];

    const result = sanitize(oversized);

    expect(result).toEqual([
      { type: "hyde", text: "hyde-a" },
      { type: "lex", text: "lex-a" },
      { type: "vec", text: "vec-a" },
      { type: "hyde", text: "hyde-b" },
      { type: "lex", text: "lex-b" },
      { type: "vec", text: "vec-b" },
    ]);
    expect(result).toHaveLength(MAX_QUERY_EXPANSIONS);
  });

  test("never exceeds the total cap even when every type is under its own cap", () => {
    // 2 of each type is exactly 6 — one more of any type must not slip past.
    const result = sanitize([
      { type: "lex", text: "l1" },
      { type: "lex", text: "l2" },
      { type: "vec", text: "v1" },
      { type: "vec", text: "v2" },
      { type: "hyde", text: "h1" },
      { type: "hyde", text: "h2" },
      { type: "lex", text: "l3" },
    ]);

    expect(result).toHaveLength(MAX_QUERY_EXPANSIONS);
    expect(result.map((r) => r.text)).not.toContain("l3");
  });

  test("deduplicates case-insensitively after trimming and collapsing whitespace", () => {
    const result = sanitize([
      { type: "vec", text: "  Foo   Bar  " },
      { type: "vec", text: "foo bar" },
      { type: "vec", text: "FOO\tBAR" },
    ]);

    expect(result).toEqual([{ type: "vec", text: "  Foo   Bar  " }]);
  });

  test("keeps the same normalized text under different types", () => {
    // lex routes to FTS, vec/hyde route to vector — not redundant work.
    const mixed: Expansion[] = [
      { type: "lex", text: "  Auth   Setup  " },
      { type: "vec", text: "auth setup" },
      { type: "hyde", text: "AUTH SETUP" },
    ];

    expect(sanitize(mixed)).toEqual(mixed);
  });

  test("rejects unknown types and text that normalizes to empty", () => {
    const result = sanitize([
      { type: "sql", text: "SELECT 1" },
      { type: "", text: "no type" },
      { type: "vec", text: "   " },
      { type: "hyde", text: "\t\n" },
      { type: "lex", text: "kept term" },
    ]);

    expect(result).toEqual([{ type: "lex", text: "kept term" }]);
  });

  test("excludes normalized restatements of the original query before capping", () => {
    // The original is always searched directly, so a variant that merely
    // restates it must not consume one of the six slots.
    const result = sanitizeQueryExpansions(
      [
        { type: "vec", text: "  QUERY " },
        { type: "vec", text: "A" },
        { type: "vec", text: "B" },
      ],
      (item) => item.text,
      "query",
    );

    expect(result).toEqual([
      { type: "vec", text: "A" },
      { type: "vec", text: "B" },
    ]);
  });

  test("excludes the original for every type, not just vec", () => {
    const result = sanitize(
      [
        { type: "lex", text: "Auth   Setup" },
        { type: "hyde", text: "auth setup" },
        { type: "vec", text: " AUTH SETUP " },
        { type: "vec", text: "something else" },
      ],
      "auth setup",
    );

    expect(result).toEqual([{ type: "vec", text: "something else" }]);
  });

  test("returns an empty list when every variant is unusable", () => {
    expect(sanitize([{ type: "hyde", text: "  " }, { type: "nope", text: "x" }])).toEqual([]);
  });
});
