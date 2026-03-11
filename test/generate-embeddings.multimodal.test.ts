import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, test, vi } from "vitest";
import { createStore, generateEmbeddings } from "../src/store.js";

type MockInput = string | { text?: string; filePath?: string };

const stores: Array<{ dbPath: string; close: () => void }> = [];
const dirs: string[] = [];

afterEach(async () => {
  vi.restoreAllMocks();

  for (const store of stores.splice(0)) {
    store.close();
    await rm(store.dbPath, { force: true });
  }

  for (const dir of dirs.splice(0)) {
    await rm(dir, { recursive: true, force: true });
  }
});

async function setupMultimodalDoc(contentType: "image" | "pdf", filename: string, body: string, title: string): Promise<{
  store: ReturnType<typeof createStore>;
  absolutePath: string;
  collectionName: string;
}> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-embed-mm-"));
  dirs.push(dir);
  const dbPath = join(dir, "index.sqlite");
  const store = createStore(dbPath);
  stores.push({ dbPath, close: () => store.close() });

  const absolutePath = join(dir, filename);
  await mkdir(dirname(absolutePath), { recursive: true });
  const bytes = contentType === "pdf"
    ? Buffer.from("%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nendobj\n")
    : Buffer.from([137, 80, 78, 71]);
  await writeFile(absolutePath, bytes);

  const hash = createHash("sha256").update(bytes).digest("hex");
  const now = new Date().toISOString();
  const collectionName = "docs";

  store.db.prepare(`
    INSERT INTO store_collections (name, path, pattern)
    VALUES (?, ?, ?)
  `).run(collectionName, dir, "**/*");

  store.db.prepare(`
    INSERT INTO content (hash, doc, created_at)
    VALUES (?, ?, ?)
  `).run(hash, body, now);

  store.db.prepare(`
    INSERT INTO documents (collection, path, title, hash, content_type, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
  `).run(collectionName, filename, title, hash, contentType, now, now);

  return { store, absolutePath, collectionName };
}

describe("generateEmbeddings multimodal inputs", () => {
  test("image embeddings include text context and file part", async () => {
    const { store, absolutePath } = await setupMultimodalDoc(
      "image",
      "receipts/invoice.png",
      "[image] receipts/invoice.png (image/png)\nInvoice for ACME Corp. Total due: $400.",
      "March Invoice"
    );

    const firstInputs: MockInput[] = [];
    const batchInputs: MockInput[][] = [];
    (store as any).llm = {
      embed: vi.fn(async (input: MockInput) => {
        firstInputs.push(input);
        return { embedding: [0.1, 0.2, 0.3], model: "mock-model" };
      }),
      embedBatch: vi.fn(async (inputs: MockInput[]) => {
        batchInputs.push(inputs);
        return inputs.map(() => ({ embedding: [0.1, 0.2, 0.3], model: "mock-model" }));
      }),
      generate: vi.fn(async () => null),
      modelExists: vi.fn(async () => ({ name: "mock", exists: true })),
      expandQuery: vi.fn(async () => []),
      rerank: vi.fn(async () => ({ results: [], model: "mock" })),
      dispose: vi.fn(async () => {}),
    };

    const result = await generateEmbeddings(store);

    expect(result.chunksEmbedded).toBe(1);
    expect(result.errors).toBe(0);

    const input = firstInputs[0] as Exclude<MockInput, string>;
    expect(typeof input).toBe("object");
    expect(input.filePath).toBe(absolutePath);
    expect(input.text).toContain("File: receipts/invoice.png");
    expect(input.text).toContain("Title: invoice");
    expect(input.text).toContain("Body: [image] receipts/invoice.png (image/png)");
    expect(input.text).toContain("Type: image");
    expect(batchInputs[0]?.length).toBe(1);
  });

  test("pdf embeddings include text context and file part", async () => {
    const { store, absolutePath } = await setupMultimodalDoc(
      "pdf",
      "reports/q1.pdf",
      "[pdf] reports/q1.pdf (application/pdf)\nQ1 financial summary.",
      "Q1 Report"
    );

    const firstInputs: MockInput[] = [];
    (store as any).llm = {
      embed: vi.fn(async (input: MockInput) => {
        firstInputs.push(input);
        return { embedding: [0.1, 0.2, 0.3], model: "mock-model" };
      }),
      embedBatch: vi.fn(async (inputs: MockInput[]) => {
        return inputs.map(() => ({ embedding: [0.1, 0.2, 0.3], model: "mock-model" }));
      }),
      generate: vi.fn(async () => null),
      modelExists: vi.fn(async () => ({ name: "mock", exists: true })),
      expandQuery: vi.fn(async () => []),
      rerank: vi.fn(async () => ({ results: [], model: "mock" })),
      dispose: vi.fn(async () => {}),
    };

    const result = await generateEmbeddings(store);

    expect(result.chunksEmbedded).toBe(1);
    expect(result.errors).toBe(0);

    const input = firstInputs[0] as Exclude<MockInput, string>;
    expect(typeof input).toBe("object");
    expect(input.filePath).toBe(absolutePath);
    expect(input.text).toContain("File: reports/q1.pdf");
    expect(input.text).toContain("Title: q1");
    expect(input.text).toContain("Body: [pdf] reports/q1.pdf (application/pdf)");
    expect(input.text).toContain("Type: pdf");
  });
});
