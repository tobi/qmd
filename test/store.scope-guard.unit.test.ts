import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, type Store } from "../src/store.js";
import {
  clearApiEmbeddingScope,
  getVectorScopeGuardMessage,
  setApiEmbeddingScopeFromCurrentEnv,
} from "../src/vector-scope-guard.js";

describe("Vector scope guard (API metadata)", () => {
  let testDir: string;
  let store: Store;

  const originalBackend = process.env.QMD_LLM_BACKEND;
  const originalEmbedBaseUrl = process.env.QMD_EMBED_BASE_URL;
  const originalEmbedModel = process.env.QMD_EMBED_MODEL;

  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-scope-guard-"));
    store = createStore(join(testDir, "index.sqlite"));

    delete process.env.QMD_LLM_BACKEND;
    delete process.env.QMD_EMBED_BASE_URL;
    delete process.env.QMD_EMBED_MODEL;
  });

  afterEach(async () => {
    store.close();
    await rm(testDir, { recursive: true, force: true });

    if (originalBackend === undefined) delete process.env.QMD_LLM_BACKEND;
    else process.env.QMD_LLM_BACKEND = originalBackend;

    if (originalEmbedBaseUrl === undefined) delete process.env.QMD_EMBED_BASE_URL;
    else process.env.QMD_EMBED_BASE_URL = originalEmbedBaseUrl;

    if (originalEmbedModel === undefined) delete process.env.QMD_EMBED_MODEL;
    else process.env.QMD_EMBED_MODEL = originalEmbedModel;
  });

  test("local backend with no api metadata does not block vector paths", () => {
    process.env.QMD_LLM_BACKEND = "local";
    const message = getVectorScopeGuardMessage(store.db);
    expect(message).toBeNull();
  });

  test("local backend blocks when api metadata exists", () => {
    process.env.QMD_LLM_BACKEND = "api";
    process.env.QMD_EMBED_BASE_URL = "https://api.openai.com/v1";
    process.env.QMD_EMBED_MODEL = "text-embedding-3-small";
    setApiEmbeddingScopeFromCurrentEnv(store.db);

    process.env.QMD_LLM_BACKEND = "local";
    const message = getVectorScopeGuardMessage(store.db);
    expect(message).toContain("current backend is local");
    expect(message).toContain("qmd embed -f");
  });

  test("api backend blocks legacy vectors when api metadata is missing", () => {
    process.env.QMD_LLM_BACKEND = "api";
    clearApiEmbeddingScope(store.db);

    store.ensureVecTable(3);
    store.insertEmbedding(
      "hash-1",
      0,
      0,
      new Float32Array([0.1, 0.2, 0.3]),
      "legacy-model",
      new Date().toISOString()
    );

    const message = getVectorScopeGuardMessage(store.db);
    expect(message).toContain("legacy/ambiguous");
    expect(message).toContain("qmd embed -f");
  });

  test("api backend allows matching stored scope", () => {
    process.env.QMD_LLM_BACKEND = "api";
    process.env.QMD_EMBED_BASE_URL = "https://api.openai.com/v1";
    process.env.QMD_EMBED_MODEL = "text-embedding-3-small";
    setApiEmbeddingScopeFromCurrentEnv(store.db);

    const message = getVectorScopeGuardMessage(store.db);
    expect(message).toBeNull();
  });

  test("api backend blocks mismatched stored scope", () => {
    process.env.QMD_LLM_BACKEND = "api";
    process.env.QMD_EMBED_BASE_URL = "https://api.openai.com/v1";
    process.env.QMD_EMBED_MODEL = "text-embedding-3-small";
    setApiEmbeddingScopeFromCurrentEnv(store.db);

    process.env.QMD_EMBED_MODEL = "text-embedding-3-large";
    const message = getVectorScopeGuardMessage(store.db);
    expect(message).toContain("scope mismatch");
    expect(message).toContain("Stored scope");
    expect(message).toContain("Current scope");
  });
});
