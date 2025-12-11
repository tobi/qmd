import { describe, test, expect } from "bun:test";
import { formatETA, formatTimeAgo, formatBytes, formatScore } from "./formatters.ts";

/**
 * Test suite for formatting utility functions
 * Tests formatBytes, formatScore, formatTimeAgo, formatETA
 */

describe("formatETA", () => {

  test("formats seconds correctly", () => {
    expect(formatETA(0)).toBe("0s");
    expect(formatETA(30)).toBe("30s");
    expect(formatETA(59)).toBe("59s");
  });

  test("formats minutes and seconds", () => {
    expect(formatETA(60)).toBe("1m 0s");
    expect(formatETA(90)).toBe("1m 30s");
    expect(formatETA(119)).toBe("1m 59s");
    expect(formatETA(3599)).toBe("59m 59s");
  });

  test("formats hours and minutes", () => {
    expect(formatETA(3600)).toBe("1h 0m");
    expect(formatETA(3660)).toBe("1h 1m");
    expect(formatETA(7200)).toBe("2h 0m");
    expect(formatETA(7320)).toBe("2h 2m");
  });
});

describe("formatTimeAgo", () => {

  test("formats seconds ago", () => {
    const now = Date.now();
    const date = new Date(now - 30 * 1000);
    expect(formatTimeAgo(date)).toBe("30s ago");
  });

  test("formats minutes ago", () => {
    const now = Date.now();
    const date = new Date(now - 5 * 60 * 1000);
    expect(formatTimeAgo(date)).toBe("5m ago");
  });

  test("formats hours ago", () => {
    const now = Date.now();
    const date = new Date(now - 3 * 60 * 60 * 1000);
    expect(formatTimeAgo(date)).toBe("3h ago");
  });

  test("formats days ago", () => {
    const now = Date.now();
    const date = new Date(now - 2 * 24 * 60 * 60 * 1000);
    expect(formatTimeAgo(date)).toBe("2d ago");
  });
});

describe("formatBytes", () => {
  test("formats bytes to human readable", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1048576)).toBe("1.0 MB");
    expect(formatBytes(1073741824)).toBe("1.0 GB");
  });
});

describe("formatScore", () => {
  // Note: formatScore returns colored output when TTY is available
  // For tests, NO_COLOR env var should disable colors
  test("formats scores as percentages", () => {
    const result100 = formatScore(1.0);
    const result86 = formatScore(0.856);
    const result0 = formatScore(0.0);

    // Check that percentage values are correct (ignoring color codes)
    expect(result100).toContain("100%");
    expect(result86).toContain("86%");
    expect(result0).toContain("  0%");
  });
});

/**
 * Edge case tests
 */
describe("Edge Cases", () => {
  test("handles negative seconds gracefully", () => {
    // Should return "0s" or handle gracefully
    const result = formatETA(-10);
    expect(typeof result).toBe("string");
  });

  test("handles very large numbers", () => {
    const result = formatETA(86400); // 1 day in seconds
    expect(result).toBe("24h 0m");
  });

  test("handles fractional seconds", () => {
    expect(formatETA(1.5)).toBe("2s"); // Rounds
    expect(formatETA(59.9)).toBe("60s");
  });
});
