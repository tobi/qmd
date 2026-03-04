/**
 * cjk-fts5.test.ts — Tests for CJK bigram splitting and FTS5 query building.
 *
 * Tests the REAL exported functions from store.ts (not local copies).
 * Covers: bigram splitting, script boundary detection, punctuation pre-split,
 * surrogate pair safety, truncation consistency, and sanitization edge cases.
 *
 * Run with: bun test cjk-fts5.test.ts
 */

import { describe, test, expect } from "vitest";
import { sanitizeFTS5Term, splitCJKBigrams, buildFTS5Query } from "../src/store";

// =============================================================================
// sanitizeFTS5Term
// =============================================================================

describe("sanitizeFTS5Term", () => {
  test("lowercase and strip punctuation", () => {
    expect(sanitizeFTS5Term("Hello!")).toBe("hello");
    expect(sanitizeFTS5Term("API_v2")).toBe("apiv2");
  });

  test("preserve apostrophes in contractions", () => {
    expect(sanitizeFTS5Term("don't")).toBe("don't");
    expect(sanitizeFTS5Term("it's")).toBe("it's");
  });

  test("trim leading/trailing apostrophes", () => {
    expect(sanitizeFTS5Term("'hello'")).toBe("hello");
    expect(sanitizeFTS5Term("'''test'''")).toBe("test");
    expect(sanitizeFTS5Term("'")).toBe("");
  });

  test("preserve CJK characters", () => {
    expect(sanitizeFTS5Term("飞书")).toBe("飞书");
    expect(sanitizeFTS5Term("API接口")).toBe("api接口");
  });

  test("empty and pure-punctuation inputs", () => {
    expect(sanitizeFTS5Term("")).toBe("");
    expect(sanitizeFTS5Term("!!!")).toBe("");
    expect(sanitizeFTS5Term("@#$")).toBe("");
  });
});

// =============================================================================
// splitCJKBigrams
// =============================================================================

describe("splitCJKBigrams", () => {
  // ── Pure Latin (no splitting) ──
  test("Latin terms returned as-is", () => {
    expect(splitCJKBigrams("hello")).toEqual(["hello"]);
    expect(splitCJKBigrams("api")).toEqual(["api"]);
    expect(splitCJKBigrams("a")).toEqual(["a"]);
    expect(splitCJKBigrams("")).toEqual([""]);
  });

  // ── Pure CJK — basic bigram splitting ──
  test("1-char CJK returns as-is", () => {
    expect(splitCJKBigrams("中")).toEqual(["中"]);
  });

  test("2-char CJK returns as-is (single bigram)", () => {
    expect(splitCJKBigrams("飞书")).toEqual(["飞书"]);
  });

  test("3-char CJK → 2 bigrams", () => {
    expect(splitCJKBigrams("飞书消")).toEqual(["飞书", "书消"]);
  });

  test("4-char CJK → 3 bigrams", () => {
    expect(splitCJKBigrams("飞书消息")).toEqual(["飞书", "书消", "消息"]);
  });

  test("5-char CJK → 4 bigrams (max without truncation)", () => {
    expect(splitCJKBigrams("飞书消息丢")).toEqual(["飞书", "书消", "消息", "息丢"]);
  });

  // ── Truncation consistency (#7) ──
  test("6-char CJK → truncated to 4 bigrams (first 2 + last 2)", () => {
    // "飞书消息丢失" → 5 bigrams → truncated
    expect(splitCJKBigrams("飞书消息丢失")).toEqual(["飞书", "书消", "息丢", "丢失"]);
  });

  test("7-char CJK → also truncated to 4 (no discontinuity)", () => {
    // "飞书消息丢失修" → 6 bigrams → truncated to 4
    expect(splitCJKBigrams("飞书消息丢失修")).toEqual(["飞书", "书消", "丢失", "失修"]);
  });

  test("10-char CJK → truncated to 4", () => {
    // "飞书消息丢失修复方案" → 9 bigrams → first 2 + last 2
    expect(splitCJKBigrams("飞书消息丢失修复方案")).toEqual(["飞书", "书消", "复方", "方案"]);
  });

  // ── Script boundary splitting (mixed CJK+Latin/digits) ──
  test("Latin+CJK → split at boundary", () => {
    expect(splitCJKBigrams("api接口")).toEqual(["api", "接口"]);
  });

  test("CJK+Latin → split at boundary (reversed)", () => {
    expect(splitCJKBigrams("接口api")).toEqual(["接口", "api"]);
  });

  test("Latin+CJK+Latin → 3 segments", () => {
    expect(splitCJKBigrams("http请求api")).toEqual(["http", "请求", "api"]);
  });

  test("CJK+Latin+CJK → 3 segments", () => {
    expect(splitCJKBigrams("请求http连接")).toEqual(["请求", "http", "连接"]);
  });

  test("digit+CJK → split at boundary", () => {
    expect(splitCJKBigrams("2026年计划")).toEqual(["2026", "年计", "计划"]);
  });

  test("short Latin+short CJK", () => {
    expect(splitCJKBigrams("v2版本")).toEqual(["v2", "版本"]);
  });

  test("long CJK segment in mixed term still gets bigram split", () => {
    // "IFRS会计准则" → ["ifrs", bigrams of "会计准则"]
    // But note: splitCJKBigrams receives already-sanitized (lowercase) terms
    expect(splitCJKBigrams("ifrs会计准则")).toEqual(["ifrs", "会计", "计准", "准则"]);
  });

  // ── Halfwidth Katakana (#3) ──
  test("fullwidth katakana treated as CJK", () => {
    expect(splitCJKBigrams("テスト")).toEqual(["テス", "スト"]);
  });

  test("halfwidth katakana treated as CJK", () => {
    expect(splitCJKBigrams("ﾃｽﾄ")).toEqual(["ﾃｽ", "ｽﾄ"]);
  });

  test("hiragana treated as CJK", () => {
    expect(splitCJKBigrams("こんにちは")).toEqual(["こん", "んに", "にち", "ちは"]);
  });

  // ── Hangul (#8) ──
  test("hangul syllables treated as CJK", () => {
    expect(splitCJKBigrams("한국어")).toEqual(["한국", "국어"]);
  });

  test("hangul compatibility jamo treated as CJK", () => {
    expect(splitCJKBigrams("ㄱㄴㄷ")).toEqual(["ㄱㄴ", "ㄴㄷ"]);
  });

  // ── Surrogate pair safety (#2) ──
  test("extension B character (U+20000) not corrupted", () => {
    // 𠀀 = U+20000, encoded as surrogate pair in JS
    // Not in BMP CJK ranges → treated as non-CJK → returned as-is
    expect(splitCJKBigrams("𠀀")).toEqual(["𠀀"]);
  });

  test("mixed BMP CJK + extension B handled cleanly", () => {
    // "中𠀀国" → CJK detected but not pure → script boundary split
    // With 'u' flag, 𠀀 is treated as single code point (non-CJK)
    expect(splitCJKBigrams("中𠀀国")).toEqual(["中", "𠀀", "国"]);
  });

  test("pure BMP CJK bigrams use code point iteration", () => {
    // Verify [...term] spread works correctly for BMP characters
    expect(splitCJKBigrams("一二三")).toEqual(["一二", "二三"]);
  });
});

// =============================================================================
// buildFTS5Query
// =============================================================================

describe("buildFTS5Query", () => {
  // ── Basic English ──
  test("plain terms → prefix match with AND", () => {
    expect(buildFTS5Query("gateway restart")).toBe('"gateway"* AND "restart"*');
  });

  test("single term", () => {
    expect(buildFTS5Query("performance")).toBe('"performance"*');
  });

  test("quoted phrase → exact match", () => {
    expect(buildFTS5Query('"machine learning"')).toBe('"machine learning"');
  });

  test("negation", () => {
    expect(buildFTS5Query("performance -sports")).toBe('"performance"* NOT "sports"*');
  });

  test("multiple negations", () => {
    expect(buildFTS5Query("perf -a -b")).toBe('"perf"* NOT "a"* NOT "b"*');
  });

  // ── CJK bigram integration ──
  test("2-char CJK → single bigram", () => {
    expect(buildFTS5Query("飞书")).toBe('"飞书"*');
  });

  test("4-char CJK → 3 bigram AND chain", () => {
    expect(buildFTS5Query("飞书消息")).toBe('"飞书"* AND "书消"* AND "消息"*');
  });

  test("mixed CJK+Latin with space", () => {
    expect(buildFTS5Query("gateway 重启")).toBe('"gateway"* AND "重启"*');
  });

  // ── CJK punctuation pre-split (#6) ──
  test("Chinese comma splits terms", () => {
    expect(buildFTS5Query("飞书，消息")).toBe('"飞书"* AND "消息"*');
  });

  test("Chinese period splits terms", () => {
    expect(buildFTS5Query("飞书。消息")).toBe('"飞书"* AND "消息"*');
  });

  test("Chinese exclamation splits terms", () => {
    expect(buildFTS5Query("你好！世界")).toBe('"你好"* AND "世界"*');
  });

  test("Chinese question mark splits terms", () => {
    expect(buildFTS5Query("什么？意思")).toBe('"什么"* AND "意思"*');
  });

  test("Japanese corner brackets split terms", () => {
    expect(buildFTS5Query("飞书「消息」")).toBe('"飞书"* AND "消息"*');
  });

  test("Japanese white corner brackets split terms", () => {
    expect(buildFTS5Query("飞书『消息』")).toBe('"飞书"* AND "消息"*');
  });

  test("fullwidth slash splits terms", () => {
    expect(buildFTS5Query("飞书／消息")).toBe('"飞书"* AND "消息"*');
  });

  test("fullwidth pipe splits terms", () => {
    expect(buildFTS5Query("飞书｜消息")).toBe('"飞书"* AND "消息"*');
  });

  test("fullwidth braces split terms", () => {
    expect(buildFTS5Query("飞书｛消息｝")).toBe('"飞书"* AND "消息"*');
  });

  test("angle brackets split terms", () => {
    expect(buildFTS5Query("飞书〈消息〉")).toBe('"飞书"* AND "消息"*');
  });

  test("fullwidth space splits terms", () => {
    expect(buildFTS5Query("飞书　消息")).toBe('"飞书"* AND "消息"*');
  });

  test("multiple mixed punctuation", () => {
    expect(buildFTS5Query("飞书，消息。测试！")).toBe('"飞书"* AND "消息"* AND "测试"*');
  });

  // ── Edge cases ──
  test("null for empty/whitespace", () => {
    expect(buildFTS5Query("")).toBeNull();
    expect(buildFTS5Query("   ")).toBeNull();
  });

  test("null for only punctuation", () => {
    expect(buildFTS5Query("，。！？")).toBeNull();
  });

  test("null for only negative terms", () => {
    expect(buildFTS5Query("-hello -world")).toBeNull();
  });

  test("unclosed quote handled gracefully", () => {
    // '"hello' — unclosed quote, parsed as phrase to end of string
    const result = buildFTS5Query('"hello');
    expect(result).not.toBeNull();
  });

  // ── Security ──
  test("SQL-like input safely sanitized", () => {
    const result = buildFTS5Query("'; DROP TABLE --");
    // Quotes and symbols stripped by sanitizeFTS5Term, only "drop" and "table" survive
    expect(result).toBe('"drop"* AND "table"*');
  });

  test("angle brackets stripped", () => {
    const result = buildFTS5Query("<script>alert(1)</script>");
    expect(result).toBe('"scriptalert1script"*');
  });
});
