/**
 * Unit tests for ModalBackend JS client.
 *
 * Tests the ModalBackend class with mocked Modal SDK, covering:
 * - embed(), generate(), rerank(), ping() method calls
 * - Retry logic for connection errors
 * - Immediate throw for non-connection errors
 * - Lazy initialization
 */

import { describe, test, expect, vi, beforeEach, afterAll } from "vitest";
import { writeFileSync, unlinkSync, mkdtempSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

// Create a temp directory with a fake .modal.toml for tests that need auth
const tempDir = mkdtempSync(join(tmpdir(), "modal-test-"));
const fakeTomlPath = join(tempDir, ".modal.toml");
writeFileSync(fakeTomlPath, "[default]\ntoken_id = fake\ntoken_secret = fake\n");

// Path that does not exist (for missing-toml test)
const missingTomlPath = join(tempDir, "nonexistent", ".modal.toml");

afterAll(() => {
  try { unlinkSync(fakeTomlPath); } catch { /* ignore */ }
});

// Mock the "modal" package
vi.mock("modal", () => {
  const mockRemote = vi.fn();
  const mockMethod = vi.fn(() => ({ remote: mockRemote }));
  const mockInstance = vi.fn(() => Promise.resolve({ method: mockMethod }));
  const mockFromName = vi.fn(() =>
    Promise.resolve({ instance: mockInstance }),
  );
  const MockModalClient = vi.fn(() => ({
    cls: { fromName: mockFromName },
  }));

  return {
    ModalClient: MockModalClient,
    _mockRemote: mockRemote,
    _mockMethod: mockMethod,
    _mockInstance: mockInstance,
    _mockFromName: mockFromName,
    _MockModalClient: MockModalClient,
  };
});

// Import after mocks are set up
import {
  ModalBackend,
  withRetry,
  isConnectionError,
} from "../src/modal.js";

// Access mock internals
async function getMocks() {
  const modal = await import("modal");
  return {
    ModalClient: (modal as any)._MockModalClient as ReturnType<typeof vi.fn>,
    mockFromName: (modal as any)._mockFromName as ReturnType<typeof vi.fn>,
    mockInstance: (modal as any)._mockInstance as ReturnType<typeof vi.fn>,
    mockMethod: (modal as any)._mockMethod as ReturnType<typeof vi.fn>,
    mockRemote: (modal as any)._mockRemote as ReturnType<typeof vi.fn>,
  };
}

/** Create a ModalBackend pointed at the fake toml file. */
function createBackend(): ModalBackend {
  return new ModalBackend({ tomlPath: fakeTomlPath });
}

beforeEach(async () => {
  const { ModalClient, mockFromName, mockInstance, mockMethod, mockRemote } =
    await getMocks();
  ModalClient.mockClear();
  mockFromName.mockClear();
  mockInstance.mockClear();
  mockMethod.mockClear();
  mockRemote.mockClear();

  // Re-setup default return values after clear
  mockMethod.mockReturnValue({ remote: mockRemote });
  mockInstance.mockReturnValue(Promise.resolve({ method: mockMethod }));
  mockFromName.mockReturnValue(Promise.resolve({ instance: mockInstance }));
});

// ============================================================================
// isConnectionError
// ============================================================================

describe("isConnectionError", () => {
  test("detects ECONNREFUSED", () => {
    const err = new Error("connect ECONNREFUSED 127.0.0.1:443");
    expect(isConnectionError(err)).toBe(true);
  });

  test("detects ETIMEDOUT", () => {
    const err = new Error("connect ETIMEDOUT 10.0.0.1:443");
    expect(isConnectionError(err)).toBe(true);
  });

  test("detects ECONNRESET", () => {
    const err = new Error("read ECONNRESET");
    expect(isConnectionError(err)).toBe(true);
  });

  test("detects 'unavailable' (gRPC status)", () => {
    const err = new Error("14 UNAVAILABLE: Connection dropped");
    expect(isConnectionError(err)).toBe(true);
  });

  test("detects 'deadline exceeded' (gRPC status)", () => {
    const err = new Error("4 DEADLINE_EXCEEDED: Deadline exceeded");
    expect(isConnectionError(err)).toBe(true);
  });

  test("returns false for auth errors", () => {
    const err = new Error("Authentication failed: invalid token");
    expect(isConnectionError(err)).toBe(false);
  });

  test("returns false for not-found errors", () => {
    const err = new Error("Function not found: qmd-inference");
    expect(isConnectionError(err)).toBe(false);
  });
});

// ============================================================================
// withRetry
// ============================================================================

describe("withRetry", () => {
  test("returns result on first success", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await withRetry(fn, 3, 0);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test("retries on connection error and succeeds on second attempt", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("connect ECONNREFUSED"))
      .mockResolvedValueOnce("ok");
    const result = await withRetry(fn, 3, 0);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  test("throws after maxAttempts connection errors", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new Error("connect ECONNREFUSED 127.0.0.1:443"));
    await expect(withRetry(fn, 3, 0)).rejects.toThrow(
      /ECONNREFUSED/,
    );
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test("throws immediately on non-connection error without retrying", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new Error("Authentication failed"));
    await expect(withRetry(fn, 3, 0)).rejects.toThrow("Authentication failed");
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

// ============================================================================
// ModalBackend
// ============================================================================

describe("ModalBackend", () => {
  test("lazy initialization: client not created until first call", async () => {
    const { ModalClient } = await getMocks();
    const _backend = createBackend();
    expect(ModalClient).not.toHaveBeenCalled();
  });

  test("throws if ~/.modal.toml is missing", async () => {
    const backend = new ModalBackend({ tomlPath: missingTomlPath });
    await expect(backend.ping()).rejects.toThrow(/\.modal\.toml/);
  });

  describe("embed", () => {
    test("calls remote with correct args and returns vectors", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      const vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
      mockRemote.mockResolvedValue(vectors);

      const backend = createBackend();
      const result = await backend.embed(["hello", "world"]);

      expect(mockMethod).toHaveBeenCalledWith("embed");
      expect(mockRemote).toHaveBeenCalledWith(
        [["hello", "world"]],
      );
      expect(result).toEqual(vectors);
    });
  });

  describe("tokenize", () => {
    test("calls remote with texts and returns token IDs", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      const tokenIds = [[1, 2, 3], [4, 5, 6, 7]];
      mockRemote.mockResolvedValue(tokenIds);

      const backend = createBackend();
      const result = await backend.tokenize(["hello world", "goodbye"]);

      expect(mockMethod).toHaveBeenCalledWith("tokenize");
      expect(mockRemote).toHaveBeenCalledWith([["hello world", "goodbye"]]);
      expect(result).toEqual(tokenIds);
    });

    test("throws on connection error after retries", async () => {
      const { mockRemote } = await getMocks();
      mockRemote.mockRejectedValue(new Error("connect ECONNREFUSED"));

      const backend = createBackend();
      await expect(backend.tokenize(["test"])).rejects.toThrow(
        /Modal .* not reachable/,
      );
    });
  });

  describe("generate", () => {
    test("calls remote with prompt, grammar, maxTokens, model", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      mockRemote.mockResolvedValue("generated text");

      const backend = createBackend();
      const result = await backend.generate(
        "test prompt",
        "root ::= [a-z]+",
        100,
        "expand",
      );

      expect(mockMethod).toHaveBeenCalledWith("generate");
      expect(mockRemote).toHaveBeenCalledWith(
        ["test prompt", "root ::= [a-z]+", 100, "expand"],
      );
      expect(result).toBe("generated text");
    });

    test("passes null grammar correctly", async () => {
      const { mockRemote } = await getMocks();
      mockRemote.mockResolvedValue("text");

      const backend = createBackend();
      await backend.generate("prompt", null, 50);

      expect(mockRemote).toHaveBeenCalledWith(
        ["prompt", null, 50, "expand"],
      );
    });
  });

  describe("rerank", () => {
    test("calls remote with query and texts, returns scores", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      const scores = [0.9, 0.3, 0.7];
      mockRemote.mockResolvedValue(scores);

      const backend = createBackend();
      const result = await backend.rerank("search query", [
        "doc1",
        "doc2",
        "doc3",
      ]);

      expect(mockMethod).toHaveBeenCalledWith("rerank");
      expect(mockRemote).toHaveBeenCalledWith(
        ["search query", ["doc1", "doc2", "doc3"]],
      );
      expect(result).toEqual(scores);
    });
  });

  describe("ping", () => {
    test("calls remote and returns true", async () => {
      const { mockRemote, mockMethod } = await getMocks();
      mockRemote.mockResolvedValue(true);

      const backend = createBackend();
      const result = await backend.ping();

      expect(mockMethod).toHaveBeenCalledWith("ping");
      expect(mockRemote).toHaveBeenCalledWith();
      expect(result).toBe(true);
    });
  });

  describe("retry logic", () => {
    test("retries on connection error, succeeds on second attempt", async () => {
      const { mockRemote } = await getMocks();
      mockRemote
        .mockRejectedValueOnce(new Error("connect ECONNREFUSED"))
        .mockResolvedValueOnce(true);

      const backend = createBackend();
      const result = await backend.ping();

      expect(result).toBe(true);
      expect(mockRemote).toHaveBeenCalledTimes(2);
    });

    test("throws after 3 connection errors with clear message", async () => {
      const { mockRemote } = await getMocks();
      mockRemote.mockRejectedValue(
        new Error("connect ECONNREFUSED 127.0.0.1:443"),
      );

      const backend = createBackend();
      await expect(backend.ping()).rejects.toThrow(
        /Modal .* not reachable/,
      );
    });

    test("non-connection errors throw immediately without retry", async () => {
      const { mockRemote } = await getMocks();
      mockRemote.mockRejectedValue(new Error("Not found: qmd-inference"));

      const backend = createBackend();
      await expect(backend.ping()).rejects.toThrow("Not found: qmd-inference");
      expect(mockRemote).toHaveBeenCalledTimes(1);
    });
  });

  describe("dispose", () => {
    test("is a no-op and does not throw", () => {
      const backend = createBackend();
      expect(() => backend.dispose()).not.toThrow();
    });
  });

  describe("connection reuse", () => {
    test("reuses client and cls instance across calls", async () => {
      const { ModalClient, mockRemote } = await getMocks();
      mockRemote.mockResolvedValue(true);

      const backend = createBackend();
      await backend.ping();
      await backend.ping();

      // ModalClient constructor called only once
      expect(ModalClient).toHaveBeenCalledTimes(1);
    });
  });
});
