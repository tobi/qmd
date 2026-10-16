import { describe, expect, test } from "vitest";
import {
  normalizeCjkForFTS,
  parseLexicalQuery,
  sanitizeFTS5Term,
} from "../src/lexical-query.js";
import {
  boundedPassageWindow,
  fitPassageWindow,
  isUtf16Boundary,
  locateLexical,
  parsePositiveLexicalAnchors,
  selectionWindowLocation,
  validatePassageBudget,
  vectorChunkLocation,
  type DocumentLocationRef,
} from "../src/search-locations.js";

const REF: DocumentLocationRef = {
  uri: "qmd://threads/thread-orbit.md",
  contentHash: "sha256:orbit-thread-v3",
};

describe("lexical query grammar", () => {
  test("preserves source clauses and compiles QMD's FTS expressions", () => {
    expect(parseLexicalQuery(
      'orbit -sports "refresh token" -"legacy tenant" pause-and-resume 認証 apply_secrets !!!',
    )).toEqual([
      { text: "orbit", negated: false, match: "prefix", ftsExpression: '"orbit"*' },
      { text: "sports", negated: true, match: "prefix", ftsExpression: '"sports"*' },
      {
        text: "refresh token",
        negated: false,
        match: "phrase",
        ftsExpression: '"refresh token"',
      },
      {
        text: "legacy tenant",
        negated: true,
        match: "phrase",
        ftsExpression: '"legacy tenant"',
      },
      {
        text: "pause-and-resume",
        negated: false,
        match: "phrase",
        ftsExpression: '"pause and resume"',
      },
      { text: "認証", negated: false, match: "phrase", ftsExpression: '"認 証"' },
      {
        text: "apply_secrets",
        negated: false,
        match: "prefix",
        ftsExpression: '"apply_secrets"*',
      },
    ]);
  });

  test("keeps the established CJK and term sanitization output", () => {
    expect(normalizeCjkForFTS("Auth認証한")).toBe("Auth 認 証 한 ");
    expect(sanitizeFTS5Term("@Tobi/QMD's_apply!")).toBe("tobiqmd's_apply");
  });
});

describe("positive lexical anchors", () => {
  test("keeps positive phrases and literals and excludes negative clauses", () => {
    expect(parsePositiveLexicalAnchors('orbit -sports "refresh token" -"legacy tenant"')).toEqual([
      { kind: "literal", text: "orbit", tokens: ["orbit"], match: "prefix" },
      {
        kind: "phrase",
        text: "refresh token",
        tokens: ["refresh", "token"],
        match: "phrase",
      },
    ]);
  });

  test("uses the FTS compound and CJK phrase grammar", () => {
    expect(parsePositiveLexicalAnchors("pause-and-resume 認証"))
      .toEqual([
        {
          kind: "phrase",
          text: "pause-and-resume",
          tokens: ["pause", "and", "resume"],
          match: "phrase",
        },
        { kind: "phrase", text: "認証", tokens: ["認", "証"], match: "phrase" },
      ]);
  });
});

describe("lexical locations", () => {
  test("reports the actual UTF-16 span of an exact positive phrase", () => {
    const body = "sports happened first. The Orbit refresh token expires after rotation.";
    const result = locateLexical(
      body,
      '-sports "refresh token"',
      REF,
      { maxUtf8Bytes: 40 },
    );

    expect(result?.location).toEqual({
      ...REF,
      kind: "lexical_exact",
      anchorKind: "phrase",
      matchedText: "refresh token",
      startUtf16: body.indexOf("refresh token"),
      endUtf16: body.indexOf("refresh token") + "refresh token".length,
    });
    expect(result?.passage.text).toContain("refresh token");
    expect(result?.passage.anchorClipped).toBe(false);
  });

  test("labels tokenizer-equivalent punctuation as an approximate window", () => {
    const body = "The worker uses pause and resume delivery for the Orbit command.";
    const result = locateLexical(body, '"pause-and-resume"', REF, { maxUtf8Bytes: 36 });

    expect(result?.location.kind).toBe("lexical_approximate");
    expect(result?.location).toMatchObject({
      ...REF,
      reason: "stem_or_tokenizer",
      queryText: "pause-and-resume",
      startUtf16: body.indexOf("pause and resume"),
      endUtf16: body.indexOf("pause and resume") + "pause and resume".length,
    });
    expect(result?.passage.text).toContain("pause and resume");
  });

  test("labels a Porter-style stem match as approximate", () => {
    const body = "The Orbit worker runs after the command is recorded.";
    const result = locateLexical(body, "running", REF, { maxUtf8Bytes: 30 });

    expect(result?.location.kind).toBe("lexical_approximate");
    expect(result?.passage.text).toContain("runs");
  });

  test("a negative-only query yields no source anchor", () => {
    expect(locateLexical("legacy tenant", '-legacy -"old tenant"', REF, {
      maxUtf8Bytes: 64,
    })).toBeNull();
  });

  test("an oversized exact literal preserves the location and marks the passage clipped", () => {
    const literal = "authorization";
    const result = locateLexical(literal, literal, REF, {
      maxUtf8Bytes: 5,
      maxUnicodeScalars: 5,
    });

    expect(result?.location).toMatchObject({
      kind: "lexical_exact",
      startUtf16: 0,
      endUtf16: literal.length,
      matchedText: literal,
    });
    expect(result?.passage).toEqual({
      startUtf16: 0,
      endUtf16: 5,
      text: "autho",
      utf8Bytes: 5,
      unicodeScalars: 5,
      anchorClipped: true,
    });
  });

  test("preserves reference validation precedence", () => {
    expect(() => locateLexical(
      "body",
      "body",
      { uri: "", contentHash: "" },
      { maxUtf8Bytes: -1 },
    )).toThrow("uri must contain at least one character");
  });
});

describe("bounded passage windows", () => {
  test("stops at both body edges with remaining byte budget", () => {
    expect(boundedPassageWindow(
      "Orbit",
      { startUtf16: 0, endUtf16: 5 },
      { maxUtf8Bytes: 64 },
    )).toEqual({
      startUtf16: 0,
      endUtf16: 5,
      text: "Orbit",
      utf8Bytes: 5,
      unicodeScalars: 5,
      anchorClipped: false,
    });
  });

  test("includes a late anchor at an offset beyond normal document limits", () => {
    const body = `${"a".repeat(280_025)}Orbit command${"z".repeat(100)}`;
    const anchor = {
      startUtf16: 280_025,
      endUtf16: 280_025 + "Orbit command".length,
    };
    const passage = boundedPassageWindow(body, anchor, { maxUtf8Bytes: 31 });

    expect(passage.startUtf16).toBeGreaterThan(279_990);
    expect(passage.endUtf16).toBeGreaterThan(anchor.endUtf16);
    expect(passage.text).toContain("Orbit command");
    expect(passage.utf8Bytes).toBeLessThanOrEqual(31);
    expect(passage.anchorClipped).toBe(false);
  });

  test("keeps every boundary between complete Unicode scalars", () => {
    const body = "left😀Orbit😀right";
    const startUtf16 = body.indexOf("Orbit");
    const passage = boundedPassageWindow(
      body,
      { startUtf16, endUtf16: startUtf16 + "Orbit".length },
      { maxUtf8Bytes: 13, maxUnicodeScalars: 7 },
    );

    expect(passage.text).toBe("😀Orbit😀");
    expect(passage.utf8Bytes).toBe(13);
    expect(passage.unicodeScalars).toBe(7);
    expect(passage.text.charCodeAt(0)).toBe(0xd83d);
    expect(passage.text.charCodeAt(passage.text.length - 1)).toBe(0xde00);
  });

  test("clips before an astral or three-byte scalar that exceeds the byte budget", () => {
    const body = "😀€x";

    expect(boundedPassageWindow(body, { startUtf16: 0, endUtf16: body.length }, {
      maxUtf8Bytes: 3,
    })).toEqual({
      startUtf16: 0,
      endUtf16: 0,
      text: "",
      utf8Bytes: 0,
      unicodeScalars: 0,
      anchorClipped: true,
    });
    expect(boundedPassageWindow(body, { startUtf16: 0, endUtf16: body.length }, {
      maxUtf8Bytes: 6,
    })).toMatchObject({
      startUtf16: 0,
      endUtf16: 2,
      text: "😀",
      utf8Bytes: 4,
      unicodeScalars: 1,
      anchorClipped: true,
    });
  });

  test("validates copied budgets and UTF-16 scalar boundaries", () => {
    const budget = { maxUtf8Bytes: 12, maxUnicodeScalars: 7 };
    const validated = validatePassageBudget(budget);

    expect(validated).toEqual(budget);
    expect(validated).not.toBe(budget);
    expect(isUtf16Boundary("a😀b", 1)).toBe(true);
    expect(isUtf16Boundary("a😀b", 2)).toBe(false);
    expect(isUtf16Boundary("a😀b", 3)).toBe(true);
    expect(isUtf16Boundary("a😀b", 1.5)).toBe(false);
    expect(isUtf16Boundary("a😀b", 5)).toBe(false);
  });
});

describe("reranker passage fitting", () => {
  test("retains the full anchor when the anchor fits the reranker", () => {
    const body = "0123456789ANCHORabcdefghij";
    const anchor = { startUtf16: 10, endUtf16: 16 };
    const passage = fitPassageWindow(body, anchor, { maxUtf8Bytes: body.length }, {
      maxDocumentTokens: 6,
      countTokens: text => text.length,
    });

    expect(passage.text).toBe("ANCHOR");
    expect(passage.anchorClipped).toBe(false);
    expect(body.slice(passage.startUtf16, passage.endUtf16)).toBe(passage.text);
  });

  test("retains a measured fit when token counts are nonmonotonic", () => {
    const body = "0123456789ANCHORabcdefghij";
    const anchor = { startUtf16: 10, endUtf16: 16 };
    const countTokens = (text: string): number => text.length === 11 ? 6 : text.length;
    const passage = fitPassageWindow(body, anchor, { maxUtf8Bytes: body.length }, {
      maxDocumentTokens: 6,
      countTokens,
    });

    expect(passage.text).toHaveLength(11);
    expect(passage.text).toContain("ANCHOR");
    expect(countTokens(passage.text)).toBeLessThanOrEqual(6);
    expect(body.slice(passage.startUtf16, passage.endUtf16)).toBe(passage.text);
  });

  test("returns an honest clipped anchor for a zero-token document budget", () => {
    const body = "leftANCHORright";
    const passage = fitPassageWindow(
      body,
      { startUtf16: 4, endUtf16: 10 },
      { maxUtf8Bytes: body.length },
      { maxDocumentTokens: 0, countTokens: text => text.length },
    );

    expect(passage).toEqual({
      startUtf16: 4,
      endUtf16: 4,
      text: "",
      utf8Bytes: 0,
      unicodeScalars: 0,
      anchorClipped: true,
    });
  });

  test("honors the caller scalar cap while clipping an oversized multibyte anchor", () => {
    const body = "x😀€abcy";
    const passage = fitPassageWindow(
      body,
      { startUtf16: 1, endUtf16: 7 },
      { maxUtf8Bytes: 100, maxUnicodeScalars: 3 },
      { maxDocumentTokens: 2, countTokens: text => Array.from(text).length },
    );

    expect(passage).toEqual({
      startUtf16: 1,
      endUtf16: 4,
      text: "😀€",
      utf8Bytes: 7,
      unicodeScalars: 2,
      anchorClipped: true,
    });
    expect(isUtf16Boundary(body, passage.endUtf16)).toBe(true);
    expect(body.slice(passage.startUtf16, passage.endUtf16)).toBe(passage.text);
  });
});

describe("vector and selection locations", () => {
  test("keeps a late vector start exact and leaves its end unknown", () => {
    const body = `${"😀".repeat(9_001)}Orbit command`;
    const startUtf16 = body.indexOf("Orbit command");

    expect(startUtf16).toBe(18_002);
    expect(vectorChunkLocation(REF, startUtf16, null)).toEqual({
      ...REF,
      kind: "vector_chunk_start",
      startUtf16: 18_002,
      endUtf16: null,
      chunkSeq: null,
    });
  });

  test("preserves keyword-intent selection provenance", () => {
    expect(selectionWindowLocation(REF, { startUtf16: 41, endUtf16: 83 })).toEqual({
      ...REF,
      kind: "selection_window",
      origin: "keyword_intent",
      startUtf16: 41,
      endUtf16: 83,
    });
  });
});
