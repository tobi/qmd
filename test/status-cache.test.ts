/**
 * Tests for the opt-in TTL cache on store.getStatus().
 *
 * Default behavior (no options, or ttlMs=0) must remain unchanged: every call
 * re-runs the underlying queries. When a caller passes ttlMs > 0, repeat calls
 * within that window must return the cached snapshot.
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach, vi } from "vitest";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { createStore, enableProductionMode, type Store } from "../src/store";

describe("getStatus opt-in TTL cache", () => {
  const baseDir = join(tmpdir(), `qmd-status-cache-${process.pid}`);
  let store: Store;

  beforeAll(() => {
    enableProductionMode();
    if (existsSync(baseDir)) rmSync(baseDir, { recursive: true, force: true });
    mkdirSync(baseDir, { recursive: true });
  });

  afterAll(() => {
    try { store?.close(); } catch { /* noop */ }
    if (existsSync(baseDir)) rmSync(baseDir, { recursive: true, force: true });
  });

  beforeEach(() => {
    try { store?.close(); } catch { /* noop */ }
    const dbPath = join(baseDir, `${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
    store = createStore(dbPath);
  });

  test("default call (no options) does not cache — each call runs a fresh query", () => {
    const prepareSpy = vi.spyOn(store.db, "prepare");
    store.getStatus();
    const firstPrepareCount = prepareSpy.mock.calls.length;
    expect(firstPrepareCount).toBeGreaterThan(0);

    store.getStatus();
    expect(prepareSpy.mock.calls.length).toBeGreaterThan(firstPrepareCount);
    prepareSpy.mockRestore();
  });

  test("ttlMs > 0 caches the result for subsequent calls within the window", () => {
    vi.useFakeTimers();
    try {
      const prepareSpy = vi.spyOn(store.db, "prepare");
      const first = store.getStatus({ ttlMs: 1000 });
      const firstPrepareCount = prepareSpy.mock.calls.length;
      expect(firstPrepareCount).toBeGreaterThan(0);

      vi.advanceTimersByTime(500);
      const second = store.getStatus({ ttlMs: 1000 });
      expect(second).toBe(first);
      expect(prepareSpy.mock.calls.length).toBe(firstPrepareCount);

      vi.advanceTimersByTime(600);
      const third = store.getStatus({ ttlMs: 1000 });
      expect(third).not.toBe(first);
      expect(prepareSpy.mock.calls.length).toBeGreaterThan(firstPrepareCount);
      prepareSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });

  test("ttlMs=0 falls back to fresh-each-call semantics", () => {
    vi.useFakeTimers();
    try {
      const prepareSpy = vi.spyOn(store.db, "prepare");
      store.getStatus({ ttlMs: 0 });
      const firstPrepareCount = prepareSpy.mock.calls.length;

      store.getStatus({ ttlMs: 0 });
      expect(prepareSpy.mock.calls.length).toBeGreaterThan(firstPrepareCount);
      prepareSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });

  test("a fresh (uncached) call invalidates the previous cache", () => {
    vi.useFakeTimers();
    try {
      const cached = store.getStatus({ ttlMs: 60_000 });

      const prepareSpy = vi.spyOn(store.db, "prepare");
      const fresh = store.getStatus();
      expect(fresh).not.toBe(cached);
      const freshPrepareCount = prepareSpy.mock.calls.length;
      expect(freshPrepareCount).toBeGreaterThan(0);

      const refreshed = store.getStatus({ ttlMs: 60_000 });
      expect(refreshed).not.toBe(cached);
      expect(prepareSpy.mock.calls.length).toBeGreaterThan(freshPrepareCount);
      prepareSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });
});
