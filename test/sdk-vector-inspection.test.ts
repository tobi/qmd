/**
 * The SDK exposes deep vector inspection as an explicit diagnostic. Its
 * existing health call keeps the same cheap result and model default.
 */
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, type QMDStore } from "../src/index.js";
import { getEmbeddingFingerprint } from "../src/store.js";

const SELECTED_MODEL = "hf:test/sdk-vector-inspection/selected.gguf";
const OVERRIDE_MODEL = "hf:test/sdk-vector-inspection/override.gguf";

interface RuntimeVectorInspector {
  inspectVectorIndex(options?: { model?: unknown }): Promise<unknown>;
}

let testDir: string;
let docsDir: string;
let sdkStore: QMDStore | undefined;

beforeEach(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-sdk-vector-inspection-"));
  docsDir = join(testDir, "docs");
  await mkdir(docsDir);
  sdkStore = undefined;
});

afterEach(async () => {
  vi.restoreAllMocks();
  await sdkStore?.close();
  await rm(testDir, { recursive: true, force: true });
});

async function openStore(): Promise<QMDStore> {
  const store = await createStore({
    dbPath: join(testDir, "index.sqlite"),
    config: {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
      models: { embed: SELECTED_MODEL },
    },
  });
  sdkStore = store;
  return store;
}

function seedIndexedDocument(store: QMDStore, model: string): void {
  const now = new Date().toISOString();
  const hash = `hash-${model}`;
  store.internal.ensureVecTable(3);
  store.internal.insertContent(hash, "# Indexed document\n\nVector inspection fixture.", now);
  store.internal.insertDocument("docs", "indexed.md", "Indexed document", hash, now, now);
  store.internal.insertEmbedding(
    hash,
    0,
    0,
    new Float32Array([1, 0, 0]),
    model,
    now,
    1,
    getEmbeddingFingerprint(model),
  );
}

describe("QMDStore.inspectVectorIndex", () => {
  test("defaults to the SDK-selected embedding model and preserves cheap health", async () => {
    const store = await openStore();
    seedIndexedDocument(store, SELECTED_MODEL);
    const expectedHealth = { needsEmbedding: 0, totalDocs: 1, daysStale: 0 };

    expect(await store.getIndexHealth()).toEqual(expectedHealth);
    expect(await store.inspectVectorIndex()).toEqual({
      model: SELECTED_MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(SELECTED_MODEL),
      partitionState: "checked",
      activeDocuments: 1,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
    expect(await store.getIndexHealth()).toEqual(expectedHealth);
  });

  test("uses an explicit model and its matching fingerprint", async () => {
    const store = await openStore();
    seedIndexedDocument(store, OVERRIDE_MODEL);
    const expectedDefaultHealth = { needsEmbedding: 1, totalDocs: 1, daysStale: 0 };

    expect(await store.getIndexHealth()).toEqual(expectedDefaultHealth);
    expect(await store.inspectVectorIndex({ model: OVERRIDE_MODEL })).toEqual({
      model: OVERRIDE_MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(OVERRIDE_MODEL),
      partitionState: "checked",
      activeDocuments: 1,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
    expect(await store.getIndexHealth()).toEqual(expectedDefaultHealth);
  });

  test("rejects invalid models before delegating to the database inspector", async () => {
    const store = await openStore();
    const inspectSpy = vi.spyOn(store.internal, "inspectVectorIndex");
    const runtimeStore: RuntimeVectorInspector = store;

    await expect(store.inspectVectorIndex({ model: "" })).rejects.toThrow("model must be a non-empty string");
    await expect(store.inspectVectorIndex({ model: "   " })).rejects.toThrow("model must be a non-empty string");
    await expect(runtimeStore.inspectVectorIndex({ model: 42 }))
      .rejects.toThrow("model must be a non-empty string");
    expect(inspectSpy).not.toHaveBeenCalled();
  });
});
