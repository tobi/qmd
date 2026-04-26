import { describe, expect, test, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const state = vi.hoisted(() => ({ nodeLlamaImported: false }));

vi.mock("node-llama-cpp", () => {
  state.nodeLlamaImported = true;
  return {
    LlamaLogLevel: { error: "error" },
    LlamaChatSession: vi.fn(),
    getLlama: vi.fn(async () => {
      throw new Error("node-llama-cpp should not be initialized for no-op embed");
    }),
    resolveModelFile: vi.fn(async () => {
      throw new Error("node-llama-cpp should not resolve models for no-op embed");
    }),
  };
});

describe("lazy node-llama-cpp loading", () => {
  test("no-op embed does not import node-llama-cpp", async () => {
    const { createStore } = await import("../src/index.js");
    const dir = mkdtempSync(join(tmpdir(), "qmd-lazy-"));

    try {
      const store = await createStore({
        dbPath: join(dir, "index.sqlite"),
        config: { collections: {} },
      });

      const result = await store.embed();
      await store.close();

      expect(result).toEqual({ docsProcessed: 0, chunksEmbedded: 0, errors: 0, durationMs: 0 });
      expect(state.nodeLlamaImported).toBe(false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
