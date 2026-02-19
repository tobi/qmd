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
  type StructuredSubSearch,
  type Store,
} from "../src/store.js";
import { disposeDefaultLlamaCpp } from "../src/llm.js";
import {
  clearApiEmbeddingScope,
  setApiEmbeddingScopeFromCurrentEnv,
} from "../src/vector-scope-guard.js";

// =============================================================================
// parseStructuredQuery Tests (CLI Parser)
// =============================================================================

/**
 * Parse structured search query syntax.
 * This is a copy of the function from qmd.ts for isolated testing.
 */
function parseStructuredQuery(query: string): StructuredSubSearch[] | null {
  const lines = query.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return null;

  const prefixRe = /^(lex|vec|hyde):\s*/i;
  const searches: StructuredSubSearch[] = [];
  const plainLines: string[] = [];

  for (const line of lines) {
    const match = line.match(prefixRe);
    if (match) {
      const type = match[1]!.toLowerCase() as 'lex' | 'vec' | 'hyde';
      const text = line.slice(match[0].length).trim();
      if (text.length > 0) {
        searches.push({ type, query: text });
      }
    } else {
      plainLines.push(line);
    }
  }

  // All plain lines, no prefixes -> null (use normal expansion)
  if (searches.length === 0 && plainLines.length === 1) {
    return null;
  }

  // Multiple plain lines without prefixes -> ambiguous, error
  if (plainLines.length > 1) {
    throw new Error("Ambiguous query: multiple lines without lex:/vec:/hyde: prefix.");
  }

  // Mix of prefixed and one plain line -> treat plain as lex
  if (plainLines.length === 1) {
    searches.unshift({ type: 'lex', query: plainLines[0]! });
  }

  return searches.length > 0 ? searches : null;
}

describe("parseStructuredQuery", () => {
  describe("plain queries (returns null for normal expansion)", () => {
    test("single line without prefix", () => {
      expect(parseStructuredQuery("CAP theorem")).toBeNull();
      expect(parseStructuredQuery("distributed systems")).toBeNull();
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
      expect(result).toEqual([{ type: "lex", query: "CAP theorem" }]);
    });

    test("vec: prefix", () => {
      const result = parseStructuredQuery("vec: what is the CAP theorem");
      expect(result).toEqual([{ type: "vec", query: "what is the CAP theorem" }]);
    });

    test("hyde: prefix", () => {
      const result = parseStructuredQuery("hyde: The CAP theorem states that...");
      expect(result).toEqual([{ type: "hyde", query: "The CAP theorem states that..." }]);
    });

    test("uppercase prefix", () => {
      expect(parseStructuredQuery("LEX: keywords")).toEqual([{ type: "lex", query: "keywords" }]);
      expect(parseStructuredQuery("VEC: question")).toEqual([{ type: "vec", query: "question" }]);
      expect(parseStructuredQuery("HYDE: passage")).toEqual([{ type: "hyde", query: "passage" }]);
    });

    test("mixed case prefix", () => {
      expect(parseStructuredQuery("Lex: test")).toEqual([{ type: "lex", query: "test" }]);
      expect(parseStructuredQuery("VeC: test")).toEqual([{ type: "vec", query: "test" }]);
    });
  });

  describe("multiple prefixed queries", () => {
    test("lex + vec", () => {
      const result = parseStructuredQuery("lex: keywords\nvec: natural language");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "vec", query: "natural language" },
      ]);
    });

    test("all three types", () => {
      const result = parseStructuredQuery("lex: keywords\nvec: question\nhyde: hypothetical doc");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "vec", query: "question" },
        { type: "hyde", query: "hypothetical doc" },
      ]);
    });

    test("duplicate types allowed", () => {
      const result = parseStructuredQuery("lex: term1\nlex: term2\nlex: term3");
      expect(result).toEqual([
        { type: "lex", query: "term1" },
        { type: "lex", query: "term2" },
        { type: "lex", query: "term3" },
      ]);
    });

    test("order preserved", () => {
      const result = parseStructuredQuery("hyde: passage\nvec: question\nlex: keywords");
      expect(result).toEqual([
        { type: "hyde", query: "passage" },
        { type: "vec", query: "question" },
        { type: "lex", query: "keywords" },
      ]);
    });
  });

  describe("mixed plain and prefixed", () => {
    test("single plain line with prefixed lines -> plain becomes lex first", () => {
      const result = parseStructuredQuery("plain keywords\nvec: semantic question");
      expect(result).toEqual([
        { type: "lex", query: "plain keywords" },
        { type: "vec", query: "semantic question" },
      ]);
    });

    test("plain line prepended before other prefixed", () => {
      const result = parseStructuredQuery("keywords\nhyde: passage\nvec: question");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "hyde", query: "passage" },
        { type: "vec", query: "question" },
      ]);
    });
  });

  describe("error cases", () => {
    test("multiple plain lines throws", () => {
      expect(() => parseStructuredQuery("line one\nline two")).toThrow("Ambiguous query");
    });

    test("three plain lines throws", () => {
      expect(() => parseStructuredQuery("a\nb\nc")).toThrow("Ambiguous query");
    });
  });

  describe("whitespace handling", () => {
    test("empty lines ignored", () => {
      const result = parseStructuredQuery("lex: keywords\n\nvec: question\n");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "vec", query: "question" },
      ]);
    });

    test("whitespace-only lines ignored", () => {
      const result = parseStructuredQuery("lex: keywords\n   \nvec: question");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "vec", query: "question" },
      ]);
    });

    test("leading/trailing whitespace trimmed from lines", () => {
      const result = parseStructuredQuery("  lex: keywords  \n  vec: question  ");
      expect(result).toEqual([
        { type: "lex", query: "keywords" },
        { type: "vec", query: "question" },
      ]);
    });

    test("internal whitespace preserved in query", () => {
      const result = parseStructuredQuery("lex:   multiple   spaces   ");
      expect(result).toEqual([{ type: "lex", query: "multiple   spaces" }]);
    });

    test("empty prefix value skipped", () => {
      const result = parseStructuredQuery("lex: \nvec: actual query");
      expect(result).toEqual([{ type: "vec", query: "actual query" }]);
    });

    test("only empty prefix values returns null", () => {
      const result = parseStructuredQuery("lex: \nvec: \nhyde: ");
      expect(result).toBeNull();
    });
  });

  describe("edge cases", () => {
    test("colon in query text preserved", () => {
      const result = parseStructuredQuery("lex: time: 12:30 PM");
      expect(result).toEqual([{ type: "lex", query: "time: 12:30 PM" }]);
    });

    test("prefix-like text in query preserved", () => {
      const result = parseStructuredQuery("vec: what does lex: mean");
      expect(result).toEqual([{ type: "vec", query: "what does lex: mean" }]);
    });

    test("newline in hyde passage (as single line)", () => {
      // If user wants actual newlines in hyde, they need to escape or use multiline syntax
      const result = parseStructuredQuery("hyde: The answer is X. It means Y.");
      expect(result).toEqual([{ type: "hyde", query: "The answer is X. It means Y." }]);
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

  test("applies API scope guard on structured query path", async () => {
    const originalBackend = process.env.QMD_LLM_BACKEND;
    const originalEmbedBaseUrl = process.env.QMD_EMBED_BASE_URL;
    const originalEmbedModel = process.env.QMD_EMBED_MODEL;

    try {
      process.env.QMD_LLM_BACKEND = "api";
      process.env.QMD_EMBED_BASE_URL = "https://api.openai.com/v1";
      process.env.QMD_EMBED_MODEL = "text-embedding-3-small";
      setApiEmbeddingScopeFromCurrentEnv(store.db);

      process.env.QMD_LLM_BACKEND = "local";
      await expect(structuredSearch(store, [{ type: "lex", query: "test" }]))
        .rejects.toThrow("current backend is local");
    } finally {
      clearApiEmbeddingScope(store.db);

      if (originalBackend === undefined) delete process.env.QMD_LLM_BACKEND;
      else process.env.QMD_LLM_BACKEND = originalBackend;

      if (originalEmbedBaseUrl === undefined) delete process.env.QMD_EMBED_BASE_URL;
      else process.env.QMD_EMBED_BASE_URL = originalEmbedBaseUrl;

      if (originalEmbedModel === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = originalEmbedModel;
    }
  });
});

// =============================================================================
// FTS Query Syntax Tests
// =============================================================================

describe("lex query syntax", () => {
  // Note: These test via CLI behavior since buildFTS5Query is not exported

  describe("validateSemanticQuery", () => {
    // Import the validation function
    const { validateSemanticQuery } = require("../src/store.js");

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
