/**
 * embed-config.test.ts - Tests for embed configuration helpers
 *
 * Run with: bun test embed-config.test.ts
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { getMaxEmbedFileBytes } from "./qmd.js";
import { DEFAULT_MAX_EMBED_FILE_BYTES } from "./store.js";

describe("getMaxEmbedFileBytes", () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env.QMD_MAX_EMBED_FILE_BYTES;
    delete process.env.QMD_MAX_EMBED_FILE_BYTES;
  });

  afterEach(() => {
    if (originalEnv !== undefined) {
      process.env.QMD_MAX_EMBED_FILE_BYTES = originalEnv;
    } else {
      delete process.env.QMD_MAX_EMBED_FILE_BYTES;
    }
  });

  test("returns default when env var is unset", () => {
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
    expect(getMaxEmbedFileBytes()).toBe(5 * 1024 * 1024);
  });

  test("respects valid numeric env var", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "1048576"; // 1MB
    expect(getMaxEmbedFileBytes()).toBe(1048576);
  });

  test("respects large values", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "10485760"; // 10MB
    expect(getMaxEmbedFileBytes()).toBe(10485760);
  });

  test("floors fractional values to integer", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "1500.7";
    expect(getMaxEmbedFileBytes()).toBe(1500);
  });

  test("falls back to default for non-numeric string", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "abc";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });

  test("falls back to default for empty string", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });

  test("falls back to default for zero", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "0";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });

  test("falls back to default for negative value", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "-100";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });

  test("falls back to default for Infinity", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "Infinity";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });

  test("falls back to default for NaN", () => {
    process.env.QMD_MAX_EMBED_FILE_BYTES = "NaN";
    expect(getMaxEmbedFileBytes()).toBe(DEFAULT_MAX_EMBED_FILE_BYTES);
  });
});
