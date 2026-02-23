/**
 * structured-search.test.ts - Tests for structured search functionality
 *
 * Tests cover:
 * - CLI query parser (parseStructuredQuery)
 * - StructuredSubSearch type validation
 * - Basic structuredSearch function behavior
 *
 * Run with: bun test structured-search.test.ts
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  structuredSearch,
  validateSemanticQuery,
  validateLexQuery,
  type StructuredSubSearch,
  type Store,
} from "../src/store.js";
import { disposeDefaultLlamaCpp } from "../src/llm.js";

// =============================================================================
// parseStructuredQuery Tests (CLI Parser)
// =============================================================================

function parseStructuredQuery(query: string): StructuredSubSearch[] | null {
  const rawLines = query.split('\n').map((line, idx) => ({
    raw: line,
    trimmed: line.trim(),
    number: idx + 1,
  })).filter(line => line.trimmed.length > 0);

  if (rawLines.length === 0) return null;

  const prefixRe = /^(lex|vec|hyde):\s*/i;
  const expandRe = /^expand:\s*/i;
  const typed: StructuredSubSearch[] = [];

  for (const line of rawLines) {
    if (expandRe.test(line.trimmed)) {
      if (rawLines.length > 1) {
        throw new Error(`Line ${line.number} starts with expand:, but query documents cannot mix expand with typed lines. Submit a single expand query instead.`);
      }
      const text = line.trimmed.replace(expandRe, '').trim();
      if (!text) {
        throw new Error('expand: query must include text.');
      }
      return null;
    }

    const match = line.trimmed.match(prefixRe);
    if (match) {
      const type = match[1]!.toLowerCase() as 'lex' | 'vec' | 'hyde';
      const text = line.trimmed.slice(match[0].length).trim();
      if (!text) {
        throw new Error(`Line ${line.number} (${type}:) must include text.`);
      }
      if (/\r|\n/.test(text)) {
        throw new Error(`Line ${line.number} (${type}:) contains a newline. Keep each query on a single line.`);
      }
      typed.push({ type, query: text, line: line.number });
      continue;
    }

    if (rawLines.length === 1) {
      return null;
    }

    throw new Error(`Line ${line.number} is missing a lex:/vec:/hyde: prefix. Each line in a query document must start with one.`);
  }

  return typed.length > 0 ? typed : null;
}

describe("parseStructuredQuery", () => {
  describe("plain queries (returns null for normal expansion)", () => {
    test("single line without prefix", () => {
      expect(parseStructuredQuery("CAP theorem")).toBeNull();
      expect(parseStructuredQuery("distributed systems")).toBeNull();
    });

    test("explicit expand line treated as plain query", () => {
      expect(parseStructuredQuery("expand: error handling best practices")).toBeNull();
    });

    test("empty queries", () => {
      expect(parseStructuredQuery("")).toBeNull();
      expect(parseStructuredQuery("   ")).toBeNull();
      expect(parseStructuredQuery("\n\n")).toBeNull();
    });
  });

  describe("single prefixed queries", () => {
    test("lex: prefix", () => {
      const result = parseStructuredQuery("lex: CAP theorem");
      expect(result).toEqual([{ type: "lex", query: "CAP theorem", line: 1 }]);
    });

    test("vec: prefix", () => {
      const result = parseStructuredQuery("vec: what is the CAP theorem");
      expect(result).toEqual([{ type: "vec", query: "what is the CAP theorem", line: 1 }]);
    });

    test("hyde: prefix", () => {
      const result = parseStructuredQuery("hyde: The CAP theorem states that...");
      expect(result).toEqual([{ type: "hyde", query: "The CAP theorem states that...", line: 1 }]);
    });

    test("uppercase prefix", () => {
      expect(parseStructuredQuery("LEX: keywords")).toEqual([{ type: "lex", query: "keywords", line: 1 }]);
      expect(parseStructuredQuery("VEC: question")).toEqual([{ type: "vec", query: "question", line: 1 }]);
      expect(parseStructuredQuery("HYDE: passage")).toEqual([{ type: "hyde", query: "passage", line: 1 }]);
    });

    test("mixed case prefix", () => {
      expect(parseStructuredQuery("Lex: test")).toEqual([{ type: "lex", query: "test", line: 1 }]);
      expect(parseStructuredQuery("VeC: test")).toEqual([{ type: "vec", query: "test", line: 1 }]);
    });
  });

  describe("multiple prefixed queries", () => {
    test("lex + vec", () => {
      const result = parseStructuredQuery("lex: keywords\nvec: natural language");
      expect(result).toEqual([
        { type: "lex", query: "keywords", line: 1 },
        { type: "vec", query: "natural language", line: 2 },
      ]);
    });

    test("all three types", () => {
      const result = parseStructuredQuery("lex: keywords\nvec: question\nhyde: hypothetical doc");
      expect(result).toEqual([
        { type: "lex", query: "keywords", line: 1 },
        { type: "vec", query: "question", line: 2 },
        { type: "hyde", query: "hypothetical doc", line: 3 },
      ]);
    });

    test("duplicate types allowed", () => {
      const result = parseStructuredQuery("lex: term1\nlex: term2\nlex: term3");
      expect(result).toEqual([
        { type: "lex", query: "term1", line: 1 },
        { type: "lex", query: "term2", line: 2 },
        { type: "lex", query: "term3", line: 3 },
      ]);
    });

    test("order preserved", () => {
      const result = parseStructuredQuery("hyde: passage\nvec: question\nlex: keywords");
      expect(result).toEqual([
        { type: "hyde", query: "passage", line: 1 },
        { type: "vec", query: "question", line: 2 },
        { type: "lex", query: "keywords", line: 3 },
      ]);
    });
  });

  describe("mixed plain and prefixed", () => {
    test("plain line with prefixed lines throws helpful error", () => {
      expect(() => parseStructuredQuery("plain keywords\nvec: semantic question"))
        .toThrow(/missing a lex:\/vec:\/hyde:/);
    });

    test("plain line prepended before other prefixed throws", () => {
      expect(() => parseStructuredQuery("keywords\nhyde: passage\nvec: question"))
        .toThrow(/missing a lex:\/vec:\/hyde:/);
    });
  });

  describe("error cases", () => {
    test("multiple plain lines throws", () => {
      expect(() => parseStructuredQuery("line one\nline two")).toThrow(/missing a lex:\/vec:\/hyde:/);
    });

    test("three plain lines throws", () => {
      expect(() => parseStructuredQuery("a\nb\nc")).toThrow(/missing a lex:\/vec:\/hyde:/);
    });

    test("mixing expand: with other lines throws", () => {
      expect(() => parseStructuredQuery("expand: question\nlex: keywords"))
        .toThrow(/cannot mix expand with typed lines/);
    });

    test("expand: without text throws", () => {
      expect(() => parseStructuredQuery("expand:   ")).toThrow(/must include text/);
    });

    test("typed line without text throws", () => {
      expect(() => parseStructuredQuery("lex:   \nvec: real")).toThrow(/must include text/);
    });
  });

  describe("whitespace handling", () => {
    test("empty lines ignored", () => {
      const result = parseStructuredQuery("lex: keywords\n\nvec: question\n");
      expect(result).toEqual([
        { type: "lex", query: "keywords", line: 1 },
        { type: "vec", query: "question", line: 3 },
      ]);
    });

    test("whitespace-only lines ignored", () => {
      const result = parseStructuredQuery("lex: keywords\n   \nvec: question");
      expect(result).toEqual([
        { type: "lex", query: "keywords", line: 1 },
        { type: "vec", query: "question", line: 3 },
      ]);
    });

    test("leading/trailing whitespace trimmed from lines", () => {
      const result = parseStructuredQuery("  lex: keywords  \n  vec: question  ");
      expect(result).toEqual([
        { type: "lex", query: "keywords", line: 1 },
        { type: "vec", query: "question", line: 2 },
      ]);
    });

    test("internal whitespace preserved in query", () => {
      const result = parseStructuredQuery("lex:   multiple   spaces   ");
      expect(result).toEqual([{ type: "lex", query: "multiple   spaces", line: 1 }]);
    });

    test("empty prefix value throws", () => {
      expect(() => parseStructuredQuery("lex: \nvec: actual query")).toThrow(/must include text/);
    });

    test("only empty prefix values throws", () => {
      expect(() => parseStructuredQuery("lex: \nvec: \nhyde: ")).toThrow(/must include text/);
    });
  });

  describe("edge cases", () => {
    test("colon in query text preserved", () => {
      const result = parseStructuredQuery("lex: time: 12:30 PM");
      expect(result).toEqual([{ type: "lex", query: "time: 12:30 PM", line: 1 }]);
    });

    test("prefix-like text in query preserved", () => {
      const result = parseStructuredQuery("vec: what does lex: mean");
      expect(result).toEqual([{ type: "vec", query: "what does lex: mean", line: 1 }]);
    });

    test("newline in hyde passage (as single line)", () => {
      // If user wants actual newlines in hyde, they need to escape or use multiline syntax
      const result = parseStructuredQuery("hyde: The answer is X. It means Y.");
      expect(result).toEqual([{ type: "hyde", query: "The answer is X. It means Y.", line: 1 }]);
    });
  });
});

// =============================================================================
// StructuredSubSearch Type Tests
// =============================================================================

describe("StructuredSubSearch type", () => {
  test("accepts lex type", () => {
    const search: StructuredSubSearch = { type: "lex", query: "test" };
    expect(search.type).toBe("lex");
    expect(search.query).toBe("test");
  });

  test("accepts vec type", () => {
    const search: StructuredSubSearch = { type: "vec", query: "test" };
    expect(search.type).toBe("vec");
    expect(search.query).toBe("test");
  });

  test("accepts hyde type", () => {
    const search: StructuredSubSearch = { type: "hyde", query: "test" };
    expect(search.type).toBe("hyde");
    expect(search.query).toBe("test");
  });
});

// =============================================================================
// structuredSearch Function Tests
// =============================================================================

describe("structuredSearch", () => {
  let testDir: string;
  let store: Store;

  beforeAll(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-structured-test-"));
    const testDbPath = join(testDir, "test.sqlite");
    const testConfigDir = await mkdtemp(join(testDir, "config-"));
    process.env.QMD_CONFIG_DIR = testConfigDir;
    store = createStore(testDbPath);
  });

  afterAll(async () => {
    store.close();
    await disposeDefaultLlamaCpp();
    if (testDir) {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("returns empty array for empty searches", async () => {
    const results = await structuredSearch(store, []);
    expect(results).toEqual([]);
  });

  test("returns empty array when no documents match", async () => {
    const results = await structuredSearch(store, [
      { type: "lex", query: "nonexistent-term-xyz123" }
    ]);
    expect(results).toEqual([]);
  });

  test("accepts all search types without error", async () => {
    // These may return empty results but should not throw
    await expect(structuredSearch(store, [{ type: "lex", query: "test" }])).resolves.toBeDefined();
    // vec and hyde require embeddings, so just test lex
  });

  test("respects limit option", async () => {
    const results = await structuredSearch(store, [
      { type: "lex", query: "test" }
    ], { limit: 5 });
    expect(results.length).toBeLessThanOrEqual(5);
  });

  test("respects minScore option", async () => {
    const results = await structuredSearch(store, [
      { type: "lex", query: "test" }
    ], { minScore: 0.5 });
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.5);
    }
  });

  test("throws when lex query contains newline characters", async () => {
    await expect(structuredSearch(store, [
      { type: "lex", query: "foo\nbar", line: 3 }
    ])).rejects.toThrow(/Line 3 \(lex\):/);
  });

  test("throws when lex query has unmatched quote", async () => {
    await expect(structuredSearch(store, [
      { type: "lex", query: "\"unfinished phrase", line: 2 }
    ])).rejects.toThrow(/unmatched double quote/);
  });
});

// =============================================================================
// FTS Query Syntax Tests
// =============================================================================

describe("lex query syntax", () => {
  // Note: These test via CLI behavior since buildFTS5Query is not exported

  describe("validateSemanticQuery", () => {

    test("accepts plain natural language", () => {
      expect(validateSemanticQuery("how does error handling work")).toBeNull();
      expect(validateSemanticQuery("what is the CAP theorem")).toBeNull();
    });

    test("rejects negation syntax", () => {
      expect(validateSemanticQuery("performance -sports")).toContain("Negation");
      expect(validateSemanticQuery('-"exact phrase"')).toContain("Negation");
    });


    test("accepts hyde-style hypothetical answers", () => {
      expect(validateSemanticQuery(
        "The CAP theorem states that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance."
      )).toBeNull();
    });
  });

  describe("validateLexQuery", () => {
    test("accepts basic lex query", () => {
      expect(validateLexQuery("auth token")).toBeNull();
    });

    test("rejects newline", () => {
      expect(validateLexQuery("foo\nbar")).toContain("single line");
    });

    test("rejects unmatched quote", () => {
      expect(validateLexQuery("\"unfinished")).toContain("unmatched");
    });
  });
});

// =============================================================================
// buildFTS5Query Tests (lex parser)
// =============================================================================

describe("buildFTS5Query (lex parser)", () => {
  // Mirror the function for unit testing
  function sanitizeFTS5Term(term: string): string {
    return term.replace(/[^\p{L}\p{N}']/gu, '').toLowerCase();
  }

  function buildFTS5Query(query: string): string | null {
    const positive: string[] = [];
    const negative: string[] = [];
    let i = 0;
    const s = query.trim();

    while (i < s.length) {
      while (i < s.length && /\s/.test(s[i]!)) i++;
      if (i >= s.length) break;
      const negated = s[i] === '-';
      if (negated) i++;

      if (s[i] === '"') {
        const start = i + 1; i++;
        while (i < s.length && s[i] !== '"') i++;
        const phrase = s.slice(start, i).trim();
        i++;
        if (phrase.length > 0) {
          const sanitized = phrase.split(/\s+/).map((t: string) => sanitizeFTS5Term(t)).filter((t: string) => t).join(' ');
          if (sanitized) (negated ? negative : positive).push(`"${sanitized}"`);
        }
      } else {
        const start = i;
        while (i < s.length && !/[\s"]/.test(s[i]!)) i++;
        const term = s.slice(start, i);
        const sanitized = sanitizeFTS5Term(term);
        if (sanitized) (negated ? negative : positive).push(`"${sanitized}"*`);
      }
    }

    if (positive.length === 0 && negative.length === 0) return null;
    if (positive.length === 0) return null;

    let result = positive.join(' AND ');
    for (const neg of negative) result = `${result} NOT ${neg}`;
    return result;
  }

  test("plain terms → prefix match with AND", () => {
    expect(buildFTS5Query("foo bar")).toBe('"foo"* AND "bar"*');
  });

  test("single term", () => {
    expect(buildFTS5Query("performance")).toBe('"performance"*');
  });

  test("quoted phrase → exact match (no prefix)", () => {
    expect(buildFTS5Query('"machine learning"')).toBe('"machine learning"');
  });

  test("quoted phrase with mixed case sanitized", () => {
    expect(buildFTS5Query('"C++ performance"')).toBe('"c performance"');
  });

  test("negation of term", () => {
    expect(buildFTS5Query("performance -sports")).toBe('"performance"* NOT "sports"*');
  });

  test("negation of phrase", () => {
    expect(buildFTS5Query('performance -"sports athlete"')).toBe('"performance"* NOT "sports athlete"');
  });

  test("multiple negations", () => {
    expect(buildFTS5Query("performance -sports -athlete")).toBe('"performance"* NOT "sports"* NOT "athlete"*');
  });

  test("quoted positive + negation", () => {
    expect(buildFTS5Query('"machine learning" -sports -athlete')).toBe('"machine learning" NOT "sports"* NOT "athlete"*');
  });

  test("intent-aware C++ performance example", () => {
    const result = buildFTS5Query('"C++ performance" optimization -sports -athlete');
    expect(result).toContain('NOT "sports"*');
    expect(result).toContain('NOT "athlete"*');
    expect(result).toContain('"optimization"*');
  });

  test("only negations with no positives → null (can't search)", () => {
    expect(buildFTS5Query("-sports -athlete")).toBeNull();
  });

  test("empty string → null", () => {
    expect(buildFTS5Query("")).toBeNull();
    expect(buildFTS5Query("   ")).toBeNull();
  });

  test("special chars in terms stripped", () => {
    expect(buildFTS5Query("hello!world")).toBe('"helloworld"*');
  });
});
