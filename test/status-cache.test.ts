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

import {
  createStore,
  enableProductionMode,
  _resetProductionModeForTesting,
  type Store,
} from "../src/store";

describe("getStatus opt-in TTL cache", () => {
  const baseDir = join(tmpdir(), `qmd-status-cache-${process.pid}`);
  let store: Store;

  beforeAll(() => {
    // enableProductionMode() toggles a module-global flag. Reset it in
    // afterAll so later test files in the same vitest process aren't
    // affected (fileParallelism is off, so state leaks linearly).
    enableProductionMode();
    if (existsSync(baseDir)) rmSync(baseDir, { recursive: true, force: true });
    mkdirSync(baseDir, { recursive: true });
  });

  afterAll(() => {
    try { store?.close(); } catch { /* noop */ }
    if (existsSync(baseDir)) rmSync(baseDir, { recursive: true, force: true });
    _resetProductionModeForTesting();
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

  test("each caller's ttlMs is authoritative against the data's age", () => {
    // If caller A populates the cache with a long TTL, caller B asking for a
    // short TTL must get fresh data once the data's age exceeds B's window —
    // regardless of A's window. The cache stores fetchedAt, not expiresAt.
    vi.useFakeTimers();
    try {
      const prepareSpy = vi.spyOn(store.db, "prepare");
      // Caller A seeds the cache with a 60s TTL.
      const a = store.getStatus({ ttlMs: 60_000 });
      const aPrepare = prepareSpy.mock.calls.length;
      expect(aPrepare).toBeGreaterThan(0);

      // 2 seconds later, caller B asks for at-most-1s-old data. Cached value
      // is 2s old, so B must trigger a fresh query.
      vi.advanceTimersByTime(2_000);
      const b = store.getStatus({ ttlMs: 1_000 });
      expect(b).not.toBe(a);
      expect(prepareSpy.mock.calls.length).toBeGreaterThan(aPrepare);

      // Caller C asks for at-most-60s-old data right after B. The cache was
      // just refreshed by B, so C should reuse it.
      const cPrepare = prepareSpy.mock.calls.length;
      const c = store.getStatus({ ttlMs: 60_000 });
      expect(c).toBe(b);
      expect(prepareSpy.mock.calls.length).toBe(cPrepare);
      prepareSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });

  test("fetchedAt is stamped after the query completes, not before", () => {
    // If the expensive getStatus query takes real wall-clock time, the cached
    // snapshot's recorded fetchedAt must reflect when the data became
    // available. Stamping pre-query would age the cache by the query duration
    // and can negate a caller's TTL window on the very next call.
    //
    // Simulated via a Date.now spy whose returned "clock" advances by 100ms
    // on every call. Inside store.getStatus(), the cache stamp is the last
    // Date.now() the function executes, so with the fix the cached fetchedAt
    // is strictly greater than 0 (the virtual clock at entry). A buggy
    // pre-query stamp would be 0, which this test would detect by observing
    // that a follow-up caller with a ttlMs just shorter than the elapsed
    // time receives a cached snapshot rather than a fresh one.
    let clock = 0;
    const nowSpy = vi.spyOn(Date, "now").mockImplementation(() => {
      const value = clock;
      clock += 100;
      return value;
    });
    try {
      const first = store.getStatus({ ttlMs: 10_000 });
      // At this point `clock` has advanced by 100ms * (number of Date.now
      // calls during the store.getStatus body). fetchedAt was the value of
      // `clock` at the moment of the post-query stamp — strictly > 0.

      // Call again immediately; the next Date.now() return is `clock`, so
      // the observed age is `clock - fetchedAt` = 100ms (one tick).
      const prepareSpy = vi.spyOn(store.db, "prepare");
      const second = store.getStatus({ ttlMs: 200 });
      expect(second).toBe(first);
      expect(prepareSpy.mock.calls.length).toBe(0);
      prepareSpy.mockRestore();
    } finally {
      nowSpy.mockRestore();
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
