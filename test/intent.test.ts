/**
 * Intent Parameter Unit Tests
 *
 * Tests the intent-aware pipeline logic:
 * - extractSnippet with intent-derived terms
 * - chunk selection scoring with intent
 * - strong-signal bypass when intent is present
 *
 * These are pure logic tests — no LLM or database required.
 */

import { describe, test, expect } from "vitest";
import { extractSnippet, extractIntentTerms, INTENT_WEIGHT_CHUNK } from "../src/store";

// =============================================================================
// extractSnippet with intent
// =============================================================================

describe("extractSnippet with intent", () => {
  // Each section contains "performance" so the query score is tied (1.0 each).
  // Intent terms (INTENT_WEIGHT_SNIPPET) then break the tie toward the relevant section.
  const body = [
    "# Notes on Various Topics",
    "",
    "## Web Performance Section",
    "Web performance means optimizing page load times and Core Web Vitals.",
    "Reduce latency, improve rendering speed, and measure performance budgets.",
    "",
    "## Team Performance Section",
    "Team performance depends on trust, psychological safety, and feedback.",
    "Build culture where performance reviews drive growth not fear.",
    "",
    "## Health Performance Section",
    "Health performance comes from consistent exercise, sleep, and endurance.",
    "Track fitness metrics, optimize recovery, and monitor healthspan.",
  ].join("\n");

  test("without intent, anchors on query terms only", () => {
    const result = extractSnippet(body, "performance", { maxLen: 500 });
    // "performance" appears in title and multiple sections — should anchor on first match
    expect(result.snippet).toContain("Performance");
  });

  test("with web-perf intent, prefers web performance section", () => {
    const result = extractSnippet(body, "performance", { maxLen: 500,
      intent: "Looking for notes about web performance, latency, and page load times" });
    expect(result.snippet).toMatch(/latency|page.*load|Core Web Vitals/i);
  });

  test("with health intent, prefers health section", () => {
    const result = extractSnippet(body, "performance", { maxLen: 500,
      intent: "Looking for notes about personal health, fitness, and endurance" });
    expect(result.snippet).toMatch(/health|fitness|endurance|exercise/i);
  });

  test("with team intent, prefers team section", () => {
    const result = extractSnippet(body, "performance", { maxLen: 500,
      intent: "Looking for notes about building high-performing teams and culture" });
    expect(result.snippet).toMatch(/team|culture|trust|feedback/i);
  });

  test("intent does not override strong query match", () => {
    // Query "Core Web Vitals" is very specific — intent shouldn't pull away from it
    const result = extractSnippet(body, "Core Web Vitals", { maxLen: 500,
      intent: "Looking for notes about health and fitness" });
    expect(result.snippet).toContain("Core Web Vitals");
  });

  test("absent intent produces same result as undefined", () => {
    const withoutIntent = extractSnippet(body, "performance", { maxLen: 500 });
    const withUndefined = extractSnippet(body, "performance", { maxLen: 500, intent: undefined });
    expect(withoutIntent.line).toBe(withUndefined.line);
    expect(withoutIntent.snippet).toBe(withUndefined.snippet);
  });
});

// =============================================================================
// Intent keyword extraction (used in chunk selection)
// =============================================================================

describe("intent keyword extraction logic", () => {
  // Mirrors the chunk selection scoring in hybridQuery, using the shared
  // extractIntentTerms helper and INTENT_WEIGHT_CHUNK constant.
  function scoreChunk(text: string, query: string, intent?: string): number {
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    const intentTerms = intent ? extractIntentTerms(intent) : [];
    const lower = text.toLowerCase();
    const qScore = queryTerms.reduce((acc, term) => acc + (lower.includes(term) ? 1 : 0), 0);
    const iScore = intentTerms.reduce((acc, term) => acc + (lower.includes(term) ? INTENT_WEIGHT_CHUNK : 0), 0);
    return qScore + iScore;
  }

  const chunks = [
    "Web performance: optimize page load times, reduce latency, improve rendering pipeline.",
    "Team performance: build trust, give feedback, set clear expectations for the group.",
    "Health performance: exercise regularly, sleep 8 hours, manage stress for endurance.",
  ];

  test("without intent, all chunks score equally on 'performance'", () => {
    const scores = chunks.map(c => scoreChunk(c, "performance"));
    // All contain "performance", so all score 1
    expect(scores[0]).toBe(scores[1]);
    expect(scores[1]).toBe(scores[2]);
  });

  test("with web intent, web chunk scores highest", () => {
    const intent = "looking for notes about page load times and latency optimization";
    const scores = chunks.map(c => scoreChunk(c, "performance", intent));
    expect(scores[0]).toBeGreaterThan(scores[1]!);
    expect(scores[0]).toBeGreaterThan(scores[2]!);
  });

  test("with health intent, health chunk scores highest", () => {
    const intent = "looking for notes about exercise, sleep, and endurance";
    const scores = chunks.map(c => scoreChunk(c, "performance", intent));
    expect(scores[2]).toBeGreaterThan(scores[0]!);
    expect(scores[2]).toBeGreaterThan(scores[1]!);
  });

  test("intent terms have lower weight than query terms (1.0)", () => {
    const intent = "looking for latency";
    // Chunk 0 has "performance" (query: 1.0) + "latency" (intent: INTENT_WEIGHT_CHUNK) = 1.5
    const withBoth = scoreChunk(chunks[0]!, "performance", intent);
    const queryOnly = scoreChunk(chunks[0]!, "performance");
    expect(withBoth).toBe(queryOnly + INTENT_WEIGHT_CHUNK);
  });

  test("stop words are filtered, short domain terms survive", () => {
    const intent = "the art of web performance";
    // "the" (stop word), "art" (survives), "of" (stop word),
    // "web" (survives), "performance" (survives)
    // Chunk 0 contains "Web" + "performance" → 2 intent hits
    // Chunks 1,2 contain only "performance" → 1 intent hit
    const scores = chunks.map(c => scoreChunk(c, "test", intent));
    expect(scores[0]).toBe(INTENT_WEIGHT_CHUNK * 2); // "web" + "performance"
    expect(scores[1]).toBe(INTENT_WEIGHT_CHUNK);      // "performance" only
    expect(scores[2]).toBe(INTENT_WEIGHT_CHUNK);      // "performance" only
  });

  test("extractIntentTerms filters stop words and punctuation", () => {
    // "looking", "for", "notes", "about" are stop words
    expect(extractIntentTerms("looking for notes about latency optimization"))
      .toEqual(["latency", "optimization"]);
    // "what", "is", "the", "to", "find" are stop words; "best", "way" survive
    expect(extractIntentTerms("what is the best way to find"))
      .toEqual(["best", "way"]);
    // Short domain terms survive (>1 char, not stop words)
    expect(extractIntentTerms("web performance latency page load times"))
      .toEqual(["web", "performance", "latency", "page", "load", "times"]);
    // Acronyms survive — the whole point of >1 vs >3
    expect(extractIntentTerms("API design for LLM agents"))
      .toEqual(["api", "design", "llm", "agents"]);
    // Surrounding punctuation stripped, internal hyphens preserved
    expect(extractIntentTerms("personal health, fitness, and endurance"))
      .toEqual(["personal", "health", "fitness", "endurance"]);
    expect(extractIntentTerms("self-hosted real-time (decision-making)"))
      .toEqual(["self-hosted", "real-time", "decision-making"]);
  });
});

// =============================================================================
// Strong-signal bypass with intent
// =============================================================================

describe("strong-signal bypass logic", () => {
  // Mirrors the logic in hybridQuery:
  // const hasStrongSignal = !intent && topScore >= 0.85 && gap >= 0.15
  function hasStrongSignal(topScore: number, secondScore: number, intent?: string): boolean {
    return !intent
      && topScore >= 0.85
      && (topScore - secondScore) >= 0.15;
  }

  test("strong signal detected without intent", () => {
    expect(hasStrongSignal(0.90, 0.70)).toBe(true);
  });

  test("strong signal bypassed when intent provided", () => {
    expect(hasStrongSignal(0.90, 0.70, "looking for health performance")).toBe(false);
  });

  test("weak signal not affected by intent", () => {
    expect(hasStrongSignal(0.50, 0.45)).toBe(false);
    expect(hasStrongSignal(0.50, 0.45, "some intent")).toBe(false);
  });

  test("close scores not strong even without intent", () => {
    expect(hasStrongSignal(0.90, 0.80)).toBe(false); // gap < 0.15
  });
});
