/**
 * MCP multi_get glob guard (HIMMEL-7227).
 *
 * picomatch/braces have no patched release for their ReDoS / resource
 * exhaustion advisories, so MCP patterns are allowlisted before they reach
 * picomatch. These tests cover the pure validator; the wire path is covered in
 * mcp.test.ts.
 */

import { describe, test, expect } from "vitest";
import picomatch from "picomatch";
import {
  validateMcpGlobPattern,
  MCP_GLOB_MAX_LENGTH,
  MCP_GLOB_MAX_WILDCARD_RUNS,
  MCP_GLOB_MAX_CLASSES,
  MCP_COMMA_LIST_MAX_ENTRIES,
  MCP_COMMA_LIST_MAX_LENGTH,
} from "../src/mcp/glob-guard.js";

describe("validateMcpGlobPattern", () => {
  test.each([
    "journals/2025-05*.md",
    "docs/**/*.md",
    "meetings/*.md",
    "docs/readme?.md",
    "docs/note-[0-9].md",
    "docs/[abc]*.md",
    "notes/日本語*.md",
    "my notes/it's & mine=1%.md",
    "readme.md, api.md",
    "readme.md,api.md",
    "notes (1).md, notes (2).md",
    "#abc123",
    "#abc123, #def456",
    "qmd://docs/meetings/*.md",
  ])("accepts documented shape %s", (pattern) => {
    expect(validateMcpGlobPattern(pattern)).toBeNull();
  });

  test.each([
    ["brace expansion", "docs/{readme,changelog}.md"],
    ["brace range", "docs/{1..9999}.md"],
    ["nested braces", "{a,{b,{c,d}}}"],
    ["lone open brace", "docs/{a.md"],
    ["lone close brace", "docs/a}.md"],
    ["brace in a comma list", "readme.md, {a,b}.md"],
    ["extglob @(", "docs/@(a|b).md"],
    ["extglob !(", "docs/!(a|b).md"],
    ["extglob +(", "docs/+(a|aa).md"],
    ["extglob *(", "docs/*(a|aa).md"],
    ["extglob ?(", "docs/?(a|aa).md"],
    ["plain group with a quantifier", "([a-z0-9./:_-]+)+Q"],
    ["plain group with alternation", "(a|a)+Q"],
    ["alternation bar", "docs/a|b.md"],
    ["plus after a class", "docs/[a-z]+Q"],
    ["plus after a group", "docs/(ab)+Q"],
    ["plus on its own", "docs/a+b.md"],
    ["class with a plus inside", "docs/[a+]Q"],
    ["class with a group inside", "docs/[(a)]Q"],
    ["negated class", "docs/[!a]Q"],
    ["caret class", "docs/[^a]Q"],
    ["empty class", "docs/[]Q"],
    ["unterminated class", "docs/[abc"],
    ["stray close bracket", "docs/abc]"],
    ["leading negation", "!docs/*.md"],
    ["backslash escape", "docs/a\\*b"],
    ["class repeating a plus", "[a-z./]+[a-z./]+[a-z./]+[a-z./]+[a-z./]+Q"],
  ])("refuses %s", (_label, pattern) => {
    expect(validateMcpGlobPattern(pattern)).toMatch(/not supported/i);
  });

  test("the refusal names what is allowed", () => {
    expect(validateMcpGlobPattern("(a)")).toMatch(/Allowed: letters, digits/);
  });

  test("refuses a glob longer than the cap", () => {
    expect(MCP_GLOB_MAX_LENGTH).toBe(256);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_LENGTH))).toMatch(/too long/i);
    expect(validateMcpGlobPattern("*".repeat(MCP_GLOB_MAX_LENGTH + 1))).toMatch(/too long/i);
    expect(validateMcpGlobPattern("*".repeat(MCP_GLOB_MAX_LENGTH))).toBeNull();
  });

  test("does not cap a long comma-separated list of plain paths", () => {
    const list = Array.from({ length: 40 }, (_, i) => `notes/file-${i}.md`).join(", ");
    expect(list.length).toBeGreaterThan(MCP_GLOB_MAX_LENGTH);
    expect(validateMcpGlobPattern(list)).toBeNull();
  });

  test("bounds a plain comma list by entry count and total size", () => {
    expect(MCP_COMMA_LIST_MAX_ENTRIES).toBe(200);
    expect(MCP_COMMA_LIST_MAX_LENGTH).toBe(16 * 1024);
    const list = (n: number) => Array.from({ length: n }, (_, i) => `n/f${i}.md`).join(",");
    expect(validateMcpGlobPattern(list(MCP_COMMA_LIST_MAX_ENTRIES))).toBeNull();
    expect(validateMcpGlobPattern(list(MCP_COMMA_LIST_MAX_ENTRIES + 1))).toMatch(/too many entries/i);
    // empty entries do not count toward the cap
    expect(validateMcpGlobPattern(list(MCP_COMMA_LIST_MAX_ENTRIES) + ",,, ,")).toBeNull();
  });

  test("bounds the total size of a plain comma list at exactly 16 KiB", () => {
    // entries of 255 chars joined by commas, the last one padded so the list is exactly `total` chars
    const exact = (total: number) => {
      const parts: string[] = [];
      let used = 0;
      while (total - used > MCP_GLOB_MAX_LENGTH) {
        parts.push("a".repeat(MCP_GLOB_MAX_LENGTH - 1));
        used += MCP_GLOB_MAX_LENGTH;
      }
      parts.push("a".repeat(total - used));
      return parts.join(",");
    };
    expect(exact(MCP_COMMA_LIST_MAX_LENGTH).length).toBe(MCP_COMMA_LIST_MAX_LENGTH);
    expect(validateMcpGlobPattern(exact(MCP_COMMA_LIST_MAX_LENGTH))).toBeNull();
    expect(validateMcpGlobPattern(exact(MCP_COMMA_LIST_MAX_LENGTH + 1))).toMatch(/list too long/i);
  });

  test("caps every entry of a plain comma list at the glob length", () => {
    // findDocuments runs a levenshtein scan over every path for each unresolved entry,
    // so one huge entry is as costly as a huge docid
    expect(validateMcpGlobPattern(["a".repeat(MCP_GLOB_MAX_LENGTH), "b.md"].join(","))).toBeNull();
    expect(validateMcpGlobPattern(["b.md", "a".repeat(MCP_GLOB_MAX_LENGTH)].join(", "))).toBeNull();
    expect(validateMcpGlobPattern(["a".repeat(MCP_GLOB_MAX_LENGTH + 1), "b.md"].join(","))).toMatch(/entry too long/i);
    expect(validateMcpGlobPattern("a".repeat(16000) + ",b.md")).toMatch(/entry too long/i);
    // the entry is measured as findDocuments sees it, trimmed
    expect(validateMcpGlobPattern([" ".repeat(50) + "a".repeat(MCP_GLOB_MAX_LENGTH), "b.md"].join(","))).toBeNull();
  });

  test("refuses more than the allowed runs of * (the plain-star backtracking blow-up)", () => {
    expect(MCP_GLOB_MAX_WILDCARD_RUNS).toBe(3);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS + 1) + "b")).toMatch(/wildcard/i);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b")).toBeNull();
    // a class between two stars does not merge them: this is 6 runs, not 3
    expect(validateMcpGlobPattern("*[a-z]*[a-z]*[a-z]*-*-*Q")).toMatch(/wildcard/i);
    expect(validateMcpGlobPattern("*[a]*[a]*[a]*b")).toMatch(/wildcard/i);
    expect(validateMcpGlobPattern("*[a]*[a]*[a]b")).toBeNull();
    // ** is one run, so the documented recursive shapes stay legal
    expect(validateMcpGlobPattern("docs/**/2025-*/*.md")).toBeNull();
  });

  test("refuses more than the allowed number of [...] classes", () => {
    expect(MCP_GLOB_MAX_CLASSES).toBe(3);
    expect(validateMcpGlobPattern("[a]".repeat(MCP_GLOB_MAX_CLASSES + 1))).toMatch(/classes/i);
    expect(validateMcpGlobPattern("[a]".repeat(MCP_GLOB_MAX_CLASSES))).toBeNull();
  });

  test("known ReDoS shapes are refused instantly, never handed to picomatch", () => {
    const evil = ["+(a|aa)".repeat(30) + "!", "([a-z0-9./:_-]|[a-z0-9./:_-])+Q", "([a-z0-9./:_-]+)+Q"];
    for (const pattern of evil) {
      const start = performance.now();
      expect(validateMcpGlobPattern(pattern)).not.toBeNull();
      expect(performance.now() - start).toBeLessThan(1000);
    }
  });

  test("the costliest accepted shapes match a long path in bounded time", () => {
    const path = "a".repeat(300);
    const worstAccepted = [
      "*[a]*[a]*[a]b",
      "a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b",
      "*a*a*a" + "[a]".repeat(MCP_GLOB_MAX_CLASSES) + "b",
      "a?".repeat(100) + "b",
    ];
    for (const pattern of worstAccepted) {
      expect(validateMcpGlobPattern(pattern)).toBeNull();
      const start = performance.now();
      picomatch(pattern)(path);
      // measured at ~16 ms; the bound is generous so a loaded CI box cannot flake it,
      // and still far below the multi-second blow-up the guard exists to prevent
      expect(performance.now() - start).toBeLessThan(5000);
    }
  });
});
