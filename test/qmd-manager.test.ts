/**
 * qmd-manager.test.ts — Unit tests for QmdManager
 *
 * QmdManager is a high-level orchestrator that wraps QMDStore with:
 *   - configurable similarity thresholds (minScore, limit)
 *   - pluggable similarity engine (LlamaCpp)
 *   - pluggable storage (SQLite Database)
 *   - inactivity timer for automatic resource cleanup
 *
 * All three dependencies (timer, storage, similarity engine) are injected
 * so tests run fully in-memory without touching the filesystem or GPU.
 *
 * Run with: bun test qmd-manager.test.ts
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import type { QmdManager, QmdManagerOptions } from "../src/qmd-manager.js";

// =============================================================================
// Dependency Mocks
// =============================================================================

/**
 * Mock LlamaCpp (similarity engine).
 * Mirrors the shape used by the real LlamaCpp in src/llm.ts:
 *   - embed()  → number[][]
 *   - rerank() → scored candidates
 *   - dispose()
 */
function makeMockSimilarityEngine() {
  return {
    embed: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3]]),
    rerank: vi.fn().mockResolvedValue([]),
    dispose: vi.fn().mockResolvedValue(undefined),
    // Inactivity-timer hook — checked by QmdManager before auto-dispose
    canUnload: vi.fn().mockReturnValue(true),
  };
}

/**
 * Mock SQLite Database (storage layer).
 * Mirrors the minimal interface QmdManager uses from src/db.ts.
 */
function makeMockStorage() {
  return {
    prepare: vi.fn().mockReturnValue({
      get: vi.fn().mockReturnValue(undefined),
      all: vi.fn().mockReturnValue([]),
      run: vi.fn(),
    }),
    exec: vi.fn(),
    close: vi.fn(),
  };
}

/**
 * Mock timer — replaces the real inactivity setTimeout/clearTimeout so
 * tests can advance time without waiting.
 *
 * The returned object exposes `fire()` to manually trigger the callback,
 * which lets tests assert cleanup behaviour without `vi.useFakeTimers`.
 */
function makeMockTimer() {
  let callback: (() => void) | null = null;
  let scheduled = false;

  return {
    schedule: vi.fn((fn: () => void, _ms: number) => {
      callback = fn;
      scheduled = true;
    }),
    cancel: vi.fn(() => {
      callback = null;
      scheduled = false;
    }),
    /** Manually trigger the scheduled callback (simulates timeout firing). */
    fire() {
      if (callback) callback();
    },
    get isScheduled() {
      return scheduled;
    },
  };
}

// =============================================================================
// Factory Helper
// =============================================================================

/**
 * Instantiate a QmdManager with fully-mocked dependencies.
 *
 * @param overrides  Partial QmdManagerOptions merged on top of safe defaults.
 *
 * Default thresholds:
 *   minScore          = 0.0   (accept all results)
 *   limit             = 10
 *   inactivityTimeoutMs = 300_000 (5 min)
 */
async function createTestManager(overrides: Partial<QmdManagerOptions> = {}): Promise<{
  manager: QmdManager;
  similarityEngine: ReturnType<typeof makeMockSimilarityEngine>;
  storage: ReturnType<typeof makeMockStorage>;
  timer: ReturnType<typeof makeMockTimer>;
}> {
  const similarityEngine = overrides.similarityEngine ?? makeMockSimilarityEngine();
  const storage = overrides.storage ?? makeMockStorage();
  const timer = overrides.timer ?? makeMockTimer();

  const { QmdManager } = await import("../src/qmd-manager.js");

  const manager = new QmdManager({
    minScore: 0.0,
    limit: 10,
    inactivityTimeoutMs: 300_000,
    ...overrides,
    similarityEngine: similarityEngine as any,
    storage: storage as any,
    timer: timer as any,
  });

  return { manager, similarityEngine, storage, timer };
}

// =============================================================================
// Lifecycle
// =============================================================================

describe("QmdManager — lifecycle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("constructs without error given valid options", async () => {
    const { manager } = await createTestManager();
    expect(manager).toBeDefined();
  });

  test("close() disposes the similarity engine", async () => {
    const { manager, similarityEngine } = await createTestManager();
    await manager.close();
    expect(similarityEngine.dispose).toHaveBeenCalledOnce();
  });

  test("close() closes the storage", async () => {
    const { manager, storage } = await createTestManager();
    await manager.close();
    expect(storage.close).toHaveBeenCalledOnce();
  });

  test("close() cancels any pending inactivity timer", async () => {
    const { manager, timer } = await createTestManager();
    await manager.close();
    expect(timer.cancel).toHaveBeenCalled();
  });
});

// =============================================================================
// Inactivity Timer
// =============================================================================

describe("QmdManager — inactivity timer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("schedules inactivity timer after construction when timeout > 0", async () => {
    const { timer } = await createTestManager({ inactivityTimeoutMs: 5_000 });
    expect(timer.schedule).toHaveBeenCalledWith(expect.any(Function), 5_000);
  });

  test("does not schedule timer when inactivityTimeoutMs is 0", async () => {
    const { timer } = await createTestManager({ inactivityTimeoutMs: 0 });
    expect(timer.schedule).not.toHaveBeenCalled();
  });

  test("timer fires → calls dispose on similarity engine when idle", async () => {
    const { timer, similarityEngine } = await createTestManager({ inactivityTimeoutMs: 5_000 });
    // Simulate the timer firing while nothing is in-flight
    similarityEngine.canUnload.mockReturnValue(true);
    timer.fire();
    expect(similarityEngine.dispose).toHaveBeenCalled();
  });

  test("timer fires → skips dispose when similarity engine is busy", async () => {
    const { timer, similarityEngine } = await createTestManager({ inactivityTimeoutMs: 5_000 });
    similarityEngine.canUnload.mockReturnValue(false);
    timer.fire();
    expect(similarityEngine.dispose).not.toHaveBeenCalled();
  });
});

// =============================================================================
// Threshold Options
// =============================================================================

describe("QmdManager — threshold options", () => {
  test("accepts default threshold options", async () => {
    const { manager } = await createTestManager();
    expect(manager.options.minScore).toBe(0.0);
    expect(manager.options.limit).toBe(10);
  });

  test("accepts custom minScore threshold", async () => {
    const { manager } = await createTestManager({ minScore: 0.75 });
    expect(manager.options.minScore).toBe(0.75);
  });

  test("accepts custom result limit", async () => {
    const { manager } = await createTestManager({ limit: 25 });
    expect(manager.options.limit).toBe(25);
  });

  test("rejects minScore outside [0, 1]", async () => {
    await expect(createTestManager({ minScore: -0.1 })).rejects.toThrow();
    await expect(createTestManager({ minScore: 1.1 })).rejects.toThrow();
  });

  test("rejects limit <= 0", async () => {
    await expect(createTestManager({ limit: 0 })).rejects.toThrow();
    await expect(createTestManager({ limit: -1 })).rejects.toThrow();
  });
});

// =============================================================================
// Search Integration (wired through mocked dependencies)
// =============================================================================

describe("QmdManager — search", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("search() calls similarity engine embed with the query", async () => {
    const { manager, similarityEngine } = await createTestManager();
    await manager.search("authentication flow");
    expect(similarityEngine.embed).toHaveBeenCalledWith(
      expect.stringContaining("authentication flow"),
    );
  });

  test("search() filters results below minScore threshold", async () => {
    const { manager, similarityEngine } = await createTestManager({ minScore: 0.8 });
    // Mock returns a mix of high and low-score docs
    similarityEngine.rerank.mockResolvedValue([
      { path: "high.md", score: 0.9 },
      { path: "low.md", score: 0.5 },
    ]);

    const results = await manager.search("query");
    expect(results.every((r) => r.score >= 0.8)).toBe(true);
  });

  test("search() respects the limit option", async () => {
    const { manager, similarityEngine } = await createTestManager({ limit: 3 });
    similarityEngine.rerank.mockResolvedValue([
      { path: "a.md", score: 0.9 },
      { path: "b.md", score: 0.85 },
      { path: "c.md", score: 0.8 },
      { path: "d.md", score: 0.75 },
    ]);

    const results = await manager.search("query");
    expect(results.length).toBeLessThanOrEqual(3);
  });

  test("search() returns empty array when similarity engine finds nothing", async () => {
    const { manager, similarityEngine } = await createTestManager();
    similarityEngine.rerank.mockResolvedValue([]);
    const results = await manager.search("unknown topic");
    expect(results).toEqual([]);
  });

  test("search() resets the inactivity timer after each call", async () => {
    const { manager, timer } = await createTestManager({ inactivityTimeoutMs: 5_000 });
    const callsBefore = (timer.schedule as ReturnType<typeof vi.fn>).mock.calls.length;
    await manager.search("query");
    const callsAfter = (timer.schedule as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(callsAfter).toBeGreaterThan(callsBefore);
  });
});
