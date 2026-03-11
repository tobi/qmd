import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  GoogleAIEmbedder,
  GOOGLE_EMBED_BATCH_LIMIT,
  parseGeminiDimensionsFromEnv,
  type EmbedInput,
} from "../src/google-embed.js";

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  delete process.env.QMD_EMBED_DIMENSIONS;
});

describe("GoogleAIEmbedder", () => {
  test("embeds text and truncates with matryoshka dimensions", async () => {
    process.env.QMD_EMBED_DIMENSIONS = "768";
    expect(parseGeminiDimensionsFromEnv()).toBe(768);

    const values = Array.from({ length: 1024 }, (_, i) => i / 1024);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      embedding: { values },
    }), { status: 200 }));

    const embedder = new GoogleAIEmbedder("test-key");
    const result = await embedder.embed("hello world", { taskType: "RETRIEVAL_DOCUMENT" });

    expect(result).not.toBeNull();
    expect(result?.embedding.length).toBe(768);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(String(url)).toContain(":embedContent?key=test-key");
    const body = JSON.parse(String(init?.body));
    expect(body.taskType).toBe("RETRIEVAL_DOCUMENT");
    expect(body.outputDimensionality).toBe(768);
    expect(body.content.parts[0].text).toBe("hello world");
  });

  test("embeds multimodal input with inline image data", async () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-google-embed-"));
    const file = join(dir, "image.png");
    writeFileSync(file, Buffer.from([137, 80, 78, 71]));

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      embedding: { values: Array.from({ length: 3072 }, () => 0.1) },
    }), { status: 200 }));

    const embedder = new GoogleAIEmbedder("test-key", 3072);
    const result = await embedder.embed({ text: "caption", filePath: file }, { taskType: "SEMANTIC_SIMILARITY" });

    expect(result).not.toBeNull();
    const [, init] = fetchSpy.mock.calls[0]!;
    const body = JSON.parse(String(init?.body));
    expect(body.taskType).toBe("SEMANTIC_SIMILARITY");
    expect(body.content.parts).toHaveLength(2);
    expect(body.content.parts[1].inlineData.mimeType).toBe("image/png");

    rmSync(dir, { recursive: true, force: true });
  });

  test("batch embeds in chunks of 100", async () => {
    const inputs: EmbedInput[] = Array.from({ length: GOOGLE_EMBED_BATCH_LIMIT + 1 }, (_, i) => `doc-${i}`);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      const body = JSON.parse(String(init?.body));
      const count = body.requests.length;
      return new Response(JSON.stringify({
        embeddings: Array.from({ length: count }, () => ({ values: Array.from({ length: 3072 }, () => 0.2) })),
      }), { status: 200 });
    });

    const embedder = new GoogleAIEmbedder("test-key", 3072);
    const results = await embedder.embedBatch(inputs, { taskType: "RETRIEVAL_QUERY" });

    expect(results).toHaveLength(GOOGLE_EMBED_BATCH_LIMIT + 1);
    expect(results.every(r => r?.embedding.length === 3072)).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  test("retries retryable HTTP responses and respects retry-after", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("rate limited", {
        status: 429,
        headers: { "retry-after": "0" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        embedding: { values: Array.from({ length: 3072 }, () => 0.3) },
      }), { status: 200 }));

    const embedder = new GoogleAIEmbedder("test-key", 3072);
    const result = await embedder.embed("retry me");

    expect(result).not.toBeNull();
    expect(result?.embedding.length).toBe(3072);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  test("retries transient network errors", async () => {
    vi.useFakeTimers();
    const fetchSpy = vi.spyOn(globalThis, "fetch")
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        embedding: { values: Array.from({ length: 3072 }, () => 0.4) },
      }), { status: 200 }));

    const embedder = new GoogleAIEmbedder("test-key", 3072);
    const pending = embedder.embed("network retry");
    await vi.advanceTimersByTimeAsync(500);
    const result = await pending;

    expect(result).not.toBeNull();
    expect(result?.embedding.length).toBe(3072);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  test("batch embed tolerates invalid multimodal input", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      const body = JSON.parse(String(init?.body));
      expect(body.requests).toHaveLength(1);
      return new Response(JSON.stringify({
        embeddings: [{ values: Array.from({ length: 3072 }, () => 0.5) }],
      }), { status: 200 });
    });

    const embedder = new GoogleAIEmbedder("test-key", 3072);
    const results = await embedder.embedBatch(
      ["ok-text", { filePath: "/tmp/not-supported.txt" }],
      { taskType: "RETRIEVAL_DOCUMENT" }
    );

    expect(results).toHaveLength(2);
    expect(results[0]?.embedding.length).toBe(3072);
    expect(results[1]).toBeNull();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});
