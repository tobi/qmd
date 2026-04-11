/**
 * decay.test.ts - Unit tests for Ebbinghaus memory decay
 *
 * Run with: bun test test/decay.test.ts
 */

import { describe, test, expect } from "vitest";
import {
  computeStrength,
  isValidCategory,
  BASE_LAMBDA,
  CATEGORIES,
  PRUNE_THRESHOLD,
  type Category,
} from "../src/decay.js";

/** Helper: create an ISO date string N days before the given timestamp. */
function daysAgo(days: number, now: number = Date.now()): string {
  return new Date(now - days * 24 * 60 * 60 * 1000).toISOString();
}

const NOW = Date.now();

describe("computeStrength", () => {
  test("new document (0 days) has strength ≈ importance", () => {
    const createdAt = new Date(NOW).toISOString();
    for (const cat of CATEGORIES) {
      const strength = computeStrength(0.5, cat, createdAt, 0, NOW);
      expect(strength).toBeCloseTo(0.5, 2);
    }
  });

  test("importance=1, day 0, no recalls → strength ≈ 1.0", () => {
    const strength = computeStrength(1.0, "fact", daysAgo(0, NOW), 0, NOW);
    expect(strength).toBeCloseTo(1.0, 2);
  });

  test("importance=0 always returns 0 regardless of other params", () => {
    expect(computeStrength(0, "strategy", daysAgo(0, NOW), 0, NOW)).toBe(0);
    expect(computeStrength(0, "fact", daysAgo(10, NOW), 5, NOW)).toBe(0);
    expect(computeStrength(0, "failure", daysAgo(100, NOW), 100, NOW)).toBe(0);
  });

  test("strength decays over time", () => {
    const day0 = computeStrength(0.5, "fact", daysAgo(0, NOW), 0, NOW);
    const day5 = computeStrength(0.5, "fact", daysAgo(5, NOW), 0, NOW);
    const day20 = computeStrength(0.5, "fact", daysAgo(20, NOW), 0, NOW);
    expect(day0).toBeGreaterThan(day5);
    expect(day5).toBeGreaterThan(day20);
  });

  test("different categories decay at different rates", () => {
    const days = 10;
    const strengths = CATEGORIES.map(cat =>
      computeStrength(0.5, cat, daysAgo(days, NOW), 0, NOW)
    );
    // strategy decays slowest, failure decays fastest
    expect(strengths[0]).toBeGreaterThan(strengths[1]!); // strategy > fact
    expect(strengths[1]).toBeGreaterThan(strengths[2]!); // fact > assumption
    expect(strengths[2]).toBeGreaterThan(strengths[3]!); // assumption > failure
  });

  test("recall_count increases strength", () => {
    const noRecalls = computeStrength(0.5, "fact", daysAgo(10, NOW), 0, NOW);
    const someRecalls = computeStrength(0.5, "fact", daysAgo(10, NOW), 3, NOW);
    const manyRecalls = computeStrength(0.5, "fact", daysAgo(10, NOW), 10, NOW);
    expect(someRecalls).toBeGreaterThan(noRecalls);
    expect(manyRecalls).toBeGreaterThan(someRecalls);
    // recall multiplier: (1 + recall_count * 0.2)
    expect(someRecalls / noRecalls).toBeCloseTo(1.6, 5); // 1 + 3*0.2
    expect(manyRecalls / noRecalls).toBeCloseTo(3.0, 5); // 1 + 10*0.2
  });

  test("strategy doc with importance=0.5 drops below 0.05 after ~38 days", () => {
    // λ_eff = 0.10 * (1 - 0.5*0.8) = 0.06
    // strength = 0.5 * e^(-0.06 * days)
    // 0.05 = 0.5 * e^(-0.06 * days) → e^(-0.06*d) = 0.1 → d ≈ 38.4
    const at35 = computeStrength(0.5, "strategy", daysAgo(35, NOW), 0, NOW);
    const at42 = computeStrength(0.5, "strategy", daysAgo(42, NOW), 0, NOW);
    expect(at35).toBeGreaterThan(PRUNE_THRESHOLD);
    expect(at42).toBeLessThan(PRUNE_THRESHOLD);
  });

  test("failure category decays very fast", () => {
    // λ_eff = 0.35 * (1 - 0.5*0.8) = 0.21
    // strength at day 10 = 0.5 * e^(-0.21*10) ≈ 0.5 * 0.1225 ≈ 0.061
    const at10 = computeStrength(0.5, "failure", daysAgo(10, NOW), 0, NOW);
    expect(at10).toBeLessThan(0.1);
    const at15 = computeStrength(0.5, "failure", daysAgo(15, NOW), 0, NOW);
    expect(at15).toBeLessThan(PRUNE_THRESHOLD);
  });

  test("high importance slows decay (importance=1 vs importance=0.5)", () => {
    const highImp = computeStrength(1.0, "fact", daysAgo(20, NOW), 0, NOW);
    const lowImp = computeStrength(0.5, "fact", daysAgo(20, NOW), 0, NOW);
    // importance=1 has λ_eff = 0.16*0.2 = 0.032 → strength = e^(-0.032*20) ≈ 0.527
    // importance=0.5 has λ_eff = 0.16*0.6 = 0.096 → strength = 0.5*e^(-0.096*20) ≈ 0.074
    expect(highImp).toBeGreaterThan(lowImp);
    expect(highImp).toBeGreaterThan(0.4);
    expect(lowImp).toBeLessThan(0.15);
  });

  test("strength can exceed 1.0 with high recall count", () => {
    // importance=1, day 0, recallCount=10 → 1 * 1 * (1 + 10*0.2) = 3.0
    const strength = computeStrength(1.0, "fact", daysAgo(0, NOW), 10, NOW);
    expect(strength).toBeCloseTo(3.0, 2);
  });

  test("effective lambda formula: λ_eff = baseLambda * (1 - importance * 0.8)", () => {
    // With importance=0: λ_eff = baseLambda * 1.0 → full decay
    // With importance=1: λ_eff = baseLambda * 0.2 → slow decay
    const fullDecay = computeStrength(0.3, "fact", daysAgo(10, NOW), 0, NOW);
    const slowDecay = computeStrength(0.9, "fact", daysAgo(10, NOW), 0, NOW);
    // Ratio of decay factors: the higher importance one should decay less
    const fullLambda = BASE_LAMBDA.fact * (1 - 0.3 * 0.8);
    const slowLambda = BASE_LAMBDA.fact * (1 - 0.9 * 0.8);
    const expectedFull = 0.3 * Math.exp(-fullLambda * 10);
    const expectedSlow = 0.9 * Math.exp(-slowLambda * 10);
    expect(fullDecay).toBeCloseTo(expectedFull, 10);
    expect(slowDecay).toBeCloseTo(expectedSlow, 10);
  });
});

describe("isValidCategory", () => {
  test("valid categories return true", () => {
    for (const cat of CATEGORIES) {
      expect(isValidCategory(cat)).toBe(true);
    }
  });

  test("invalid categories return false", () => {
    expect(isValidCategory("")).toBe(false);
    expect(isValidCategory("unknown")).toBe(false);
    expect(isValidCategory("STRATEGY")).toBe(false);
    expect(isValidCategory("Strategy")).toBe(false);
  });
});

describe("constants", () => {
  test("BASE_LAMBDA values are ordered: strategy < fact < assumption < failure", () => {
    expect(BASE_LAMBDA.strategy).toBeLessThan(BASE_LAMBDA.fact);
    expect(BASE_LAMBDA.fact).toBeLessThan(BASE_LAMBDA.assumption);
    expect(BASE_LAMBDA.assumption).toBeLessThan(BASE_LAMBDA.failure);
  });

  test("all BASE_LAMBDA values are positive", () => {
    for (const cat of CATEGORIES) {
      expect(BASE_LAMBDA[cat]).toBeGreaterThan(0);
    }
  });

  test("PRUNE_THRESHOLD is a small positive number", () => {
    expect(PRUNE_THRESHOLD).toBeGreaterThan(0);
    expect(PRUNE_THRESHOLD).toBeLessThan(0.5);
  });

  test("CATEGORIES has exactly 4 entries", () => {
    expect(CATEGORIES).toHaveLength(4);
  });
});
