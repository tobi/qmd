import { afterEach, describe, expect, test, vi } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import {
  getDefaultEnvironment,
  StdioClientTransport,
} from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  ProgressNotificationSchema,
  type ProgressNotification,
} from "@modelcontextprotocol/sdk/types.js";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { QMDStore } from "../src/index.js";
import { createMcpServer } from "../src/mcp/server.js";

const openHarnesses: Array<{
  client: Client;
  server: Awaited<ReturnType<typeof createMcpServer>>;
}> = [];

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function createStoreDouble(
  updateImplementation?: (
    options?: Parameters<QMDStore["update"]>[0]
  ) => ReturnType<QMDStore["update"]>,
  embedImplementation?: (
    options?: Parameters<QMDStore["embed"]>[0]
  ) => ReturnType<QMDStore["embed"]>,
) {
  const update = vi.fn(updateImplementation ?? (async () => ({
    collections: 2,
    indexed: 3,
    updated: 4,
    unchanged: 5,
    removed: 6,
    needsEmbedding: 7,
  })));
  const embed = vi.fn(embedImplementation ?? (async () => ({
    docsProcessed: 3,
    chunksEmbedded: 4,
    errors: 0,
    durationMs: 5,
    failures: [],
  })));

  const store = {
    getStatus: async () => ({
      totalDocuments: 0,
      needsEmbedding: 0,
      hasVectorIndex: false,
      collections: [
        {
          name: "docs",
          path: "/docs",
          pattern: "**/*.md",
          documents: 0,
          lastUpdated: "",
        },
        {
          name: "notes",
          path: "/notes",
          pattern: "**/*.md",
          documents: 0,
          lastUpdated: "",
        },
      ],
    }),
    getGlobalContext: async () => undefined,
    getDefaultCollectionNames: async () => ["docs", "notes"],
    listCollections: async () => [
      {
        name: "docs",
        pwd: "/docs",
        glob_pattern: "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      },
      {
        name: "notes",
        pwd: "/notes",
        glob_pattern: "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      },
    ],
    update,
    embed,
  } as unknown as QMDStore;

  return { store, update, embed };
}

async function createHarness(store: QMDStore) {
  const server = await createMcpServer(store);
  const client = new Client({ name: "qmd-maintenance-test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  await server.connect(serverTransport);
  await client.connect(clientTransport);

  openHarnesses.push({ client, server });
  return { client, server, serverTransport };
}

function rejectProgressNotifications(serverTransport: InMemoryTransport): void {
  const originalSend = serverTransport.send.bind(serverTransport);
  serverTransport.send = async (message, options) => {
    if (
      "method" in message &&
      message.method === "notifications/progress"
    ) {
      throw new Error("progress transport unavailable");
    }

    await originalSend(message, options);
  };
}

function getFirstText(result: unknown): string {
  const content = (result as {
    content: Array<{ type: string; text?: string }>;
  }).content;
  expect(content[0]?.type).toBe("text");
  return content[0]?.text ?? "";
}

afterEach(async () => {
  await Promise.all(openHarnesses.splice(0).map(async ({ client, server }) => {
    await client.close();
    await server.close();
  }));
});

describe("MCP maintenance tools", () => {
  test("stdio tools/list exposes both maintenance tools", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-mcp-stdio-"));
    const client = new Client({
      name: "qmd-stdio-maintenance-test",
      version: "1.0.0",
    });
    const transport = new StdioClientTransport({
      command: "node",
      args: ["--import", "tsx", "src/cli/qmd.ts", "mcp"],
      cwd: process.cwd(),
      env: {
        ...getDefaultEnvironment(),
        INDEX_PATH: join(testDir, "index.sqlite"),
        QMD_CONFIG_DIR: testDir,
        LLAMA_LOG_LEVEL: "error",
        GGML_LOG_LEVEL: "error",
        GGML_BACKEND_SILENT: "1",
      },
      stderr: "pipe",
    });

    try {
      await client.connect(transport);
      const { tools } = await client.listTools();
      const toolNames = tools.map(tool => tool.name);

      expect(toolNames).toContain("update");
      expect(toolNames).toContain("embed");
    } finally {
      await client.close();
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("server instructions recommend the safe MCP maintenance workflow", async () => {
    const { store } = createStoreDouble();
    store.getStatus = async () => ({
      totalDocuments: 3,
      needsEmbedding: 2,
      hasVectorIndex: true,
      collections: [],
    });
    const { client } = await createHarness(store);

    const instructions = client.getInstructions() ?? "";

    expect(instructions).toContain("Call the `update` MCP tool");
    expect(instructions).toContain("inspect `needsEmbedding`");
    expect(instructions).toContain(
      "Call the `embed` MCP tool only when `needsEmbedding` is greater than 0"
    );
    expect(instructions).not.toContain("qmd embed");
  });

  test("tools/list exposes update as a local idempotent write", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store);

    const { tools } = await client.listTools();
    const updateTool = tools.find((tool) => tool.name === "update");

    expect(updateTool).toBeDefined();
    expect(updateTool?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    });
    expect(updateTool?.inputSchema.properties).toHaveProperty("collections");
    expect(
      (updateTool?.inputSchema.properties?.collections as { minItems?: number })
        .minItems
    ).toBe(1);
    expect(updateTool?.description).toMatch(/derived index/i);
    expect(updateTool?.description).toMatch(/progress/i);
    expect(updateTool?.description).toMatch(/best-effort/i);
    expect(updateTool?.description).toMatch(/never change the tool result/i);
    expect(updateTool?.description).toMatch(/busy|cancel/i);
    expect(updateTool?.description).toMatch(/needsEmbedding/);
  });

  test("tools/list exposes embed without caller-controlled model or paths", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store);

    const { tools } = await client.listTools();
    const embedTool = tools.find((tool) => tool.name === "embed");

    expect(embedTool).toBeDefined();
    expect(embedTool?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    });
    expect(embedTool?.inputSchema.properties).not.toHaveProperty("model");
    expect(embedTool?.inputSchema.properties).not.toHaveProperty("dbPath");
    expect(embedTool?.inputSchema.properties).not.toHaveProperty("configPath");
    expect(embedTool?.inputSchema.properties).not.toHaveProperty("path");
    expect(embedTool?.description).toMatch(/derived index/i);
    expect(embedTool?.description).toMatch(/30 minutes/i);
    expect(embedTool?.description).toMatch(/progress/i);
    expect(embedTool?.description).toMatch(/partial|busy|cancel/i);
  });

  test("embed validates and converts every supported option for the Store", async () => {
    const { store, embed } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {
        collection: "docs",
        force: true,
        chunkStrategy: "regex",
        maxDocsPerBatch: 12,
        maxBatchMiB: 1.5,
        timeoutMinutes: 0.25,
      },
    });

    expect(embed).toHaveBeenCalledWith({
      collection: "docs",
      force: true,
      chunkStrategy: "regex",
      maxDocsPerBatch: 12,
      maxBatchBytes: 1.5 * 1024 * 1024,
      maxDurationMs: 0.25 * 60 * 1000,
      onProgress: expect.any(Function),
    });
    expect(result.structuredContent).toMatchObject({
      docsProcessed: 3,
      chunksEmbedded: 4,
      errors: 0,
      durationMs: 5,
      failureCount: 0,
      failures: [],
      failuresTruncated: false,
    });
  });

  test("embed rejects an unknown collection before calling the Store", async () => {
    const { store, embed } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: { collection: "missing" },
    });

    expect(result.isError).toBe(true);
    expect(getFirstText(result)).toContain("missing");
    expect(embed).not.toHaveBeenCalled();
  });

  test("embed forwards chunk progress with bytes, errors, and the MCP token", async () => {
    const { store } = createStoreDouble(undefined, async (options) => {
      options?.onProgress?.({
        chunksEmbedded: 1,
        totalChunks: 3,
        bytesProcessed: 100,
        totalBytes: 300,
        errors: 0,
        failures: [],
      });
      options?.onProgress?.({
        chunksEmbedded: 2,
        totalChunks: 3,
        bytesProcessed: 200,
        totalBytes: 300,
        errors: 1,
        failures: [],
      });

      return {
        docsProcessed: 2,
        chunksEmbedded: 2,
        errors: 1,
        durationMs: 10,
        failures: [],
      };
    });
    const { client } = await createHarness(store);
    const progress: ProgressNotification["params"][] = [];

    client.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      progress.push(notification.params);
    });

    await client.callTool({
      name: "embed",
      arguments: {},
      _meta: { progressToken: 42 },
    });

    expect(progress.map(item => item.progressToken)).toEqual([42, 42]);
    expect(progress.map(item => item.progress)).toEqual([1, 2]);
    expect(progress.map(item => item.total)).toEqual([3, 3]);
    expect(progress.map(item => item.message)).toEqual([
      "1/3 chunks; 100/300 bytes; 0 errors",
      "2/3 chunks; 200/300 bytes; 1 errors",
    ]);
  });

  test("embed ignores rejected progress notifications after Store success", async () => {
    const { store } = createStoreDouble(undefined, async (options) => {
      options?.onProgress?.({
        chunksEmbedded: 2,
        totalChunks: 3,
        bytesProcessed: 200,
        totalBytes: 300,
        errors: 0,
        failures: [],
      });

      return {
        docsProcessed: 2,
        chunksEmbedded: 2,
        errors: 0,
        durationMs: 10,
        failures: [],
      };
    });
    const { client, serverTransport } = await createHarness(store);
    rejectProgressNotifications(serverTransport);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
      _meta: { progressToken: "rejected-embed-progress" },
    });

    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toMatchObject({
      docsProcessed: 2,
      chunksEmbedded: 2,
      errors: 0,
      durationMs: 10,
      failureCount: 0,
      failures: [],
      failuresTruncated: false,
    });
  });

  test("embed handles rejected progress notifications when the Store throws", async () => {
    const { store } = createStoreDouble(undefined, async (options) => {
      options?.onProgress?.({
        chunksEmbedded: 1,
        totalChunks: 2,
        bytesProcessed: 100,
        totalBytes: 200,
        errors: 0,
        failures: [],
      });
      throw new Error("embedding failed");
    });
    const { client, serverTransport } = await createHarness(store);
    rejectProgressNotifications(serverTransport);
    const unhandledRejections: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => {
      unhandledRejections.push(reason);
    };
    process.on("unhandledRejection", onUnhandledRejection);

    try {
      const result = await client.callTool({
        name: "embed",
        arguments: {},
        _meta: { progressToken: "rejected-embed-progress" },
      });
      await new Promise<void>((resolve) => setImmediate(resolve));

      expect(result.isError).toBe(true);
      expect(unhandledRejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
    }
  });

  test("embed converts Store exceptions into a safe tool error", async () => {
    const { store, update } = createStoreDouble(undefined, async () => {
      throw new Error("model failed with SECRET_MODEL_KEY\ninternal stack detail");
    });
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
    });
    const text = getFirstText(result);

    expect(result.isError).toBe(true);
    expect(text).toContain("Embedding failed");
    expect(text).not.toContain("SECRET_MODEL_KEY");
    expect(text).not.toContain("stack");

    const afterError = await client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterError.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();
  });

  test("embed applies the 30-minute default and preserves zero as unlimited", async () => {
    const { store, embed } = createStoreDouble();
    const { client } = await createHarness(store);

    await client.callTool({ name: "embed", arguments: {} });
    await client.callTool({
      name: "embed",
      arguments: { timeoutMinutes: 0 },
    });

    expect(embed.mock.calls[0]?.[0]).toEqual({
      force: false,
      maxDurationMs: 30 * 60 * 1000,
      onProgress: expect.any(Function),
    });
    expect(embed.mock.calls[1]?.[0]).toEqual({
      force: false,
      maxDurationMs: 0,
      onProgress: expect.any(Function),
    });
  });

  test("embed treats a zero-work result as success", async () => {
    const { store } = createStoreDouble(undefined, async () => ({
      docsProcessed: 0,
      chunksEmbedded: 0,
      errors: 0,
      durationMs: 0,
      failures: [],
    }));
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
    });

    expect(result.isError).toBe(false);
    expect(result.structuredContent).toMatchObject({
      docsProcessed: 0,
      chunksEmbedded: 0,
      errors: 0,
      durationMs: 0,
      failureCount: 0,
      failures: [],
      failuresTruncated: false,
    });
  });

  test("embed masks raw failure reasons while preserving diagnostic fields", async () => {
    const { store } = createStoreDouble(undefined, async () => ({
      docsProcessed: 1,
      chunksEmbedded: 0,
      errors: 1,
      durationMs: 5,
      failures: [{
        path: "docs/private.md",
        hash: "hash-private",
        seq: 7,
        attempts: 3,
        reason: "https://user:SECRET_TOKEN@example.test/model failed",
      }],
    }));
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
    });
    const text = getFirstText(result);
    const structuredText = JSON.stringify(result.structuredContent);

    expect(text).not.toContain("SECRET_TOKEN");
    expect(text).not.toContain("example.test");
    expect(structuredText).not.toContain("SECRET_TOKEN");
    expect(structuredText).not.toContain("example.test");
    expect(result.structuredContent).toMatchObject({
      failures: [{
        path: "docs/private.md",
        hash: "hash-private",
        seq: 7,
        attempts: 3,
        reason: "embedding backend or index write failed",
      }],
    });
  });

  test("embed preserves safe Store reasons and sanitizes an expired batch detail", async () => {
    const safeReasons = [
      "embedding returned no vector",
      "LLM session expired before embedding chunk",
      "embedding aborted because error rate was too high",
      "batch embedding returned no vector",
    ];
    const failures = [
      ...safeReasons.map((reason, index) => ({
        path: `docs/safe-${index}.md`,
        hash: `hash-safe-${index}`,
        seq: index,
        attempts: 1,
        reason,
      })),
      {
        path: "docs/expired-batch.md",
        hash: "hash-expired-batch",
        seq: 4,
        attempts: 3,
        reason:
          "batch failed and session expired: https://user:SECRET_TOKEN@example.test/model",
      },
    ];
    const { store } = createStoreDouble(undefined, async () => ({
      docsProcessed: 5,
      chunksEmbedded: 0,
      errors: 5,
      durationMs: 5,
      failures,
    }));
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
    });
    const structured = result.structuredContent as {
      failures: Array<{ reason: string }>;
    };

    expect(structured.failures.map(failure => failure.reason)).toEqual([
      ...safeReasons,
      "batch failed and session expired",
    ]);
    expect(JSON.stringify(result.structuredContent)).not.toContain("SECRET_TOKEN");
    expect(JSON.stringify(result.structuredContent)).not.toContain("example.test");
  });

  test("embed derives truncation only from available failure details", async () => {
    const { store: shortStore } = createStoreDouble(undefined, async () => ({
      docsProcessed: 25,
      chunksEmbedded: 0,
      errors: 25,
      durationMs: 5,
      failures: [{
        path: "docs/only-detail.md",
        hash: "hash-only",
        seq: 1,
        attempts: 3,
        reason: "failed",
      }],
    }));
    const { client: shortClient } = await createHarness(shortStore);
    const shortResult = await shortClient.callTool({
      name: "embed",
      arguments: {},
    });

    expect(shortResult.structuredContent).toMatchObject({
      failureCount: 25,
      failuresTruncated: false,
    });
    expect(
      (shortResult.structuredContent as { failures: unknown[] }).failures
    ).toHaveLength(1);

    const { store: missingStore } = createStoreDouble(undefined, async () => ({
      docsProcessed: 25,
      chunksEmbedded: 0,
      errors: 25,
      durationMs: 5,
    }));
    const { client: missingClient } = await createHarness(missingStore);
    const missingResult = await missingClient.callTool({
      name: "embed",
      arguments: {},
    });

    expect(missingResult.structuredContent).toMatchObject({
      failureCount: 25,
      failures: [],
      failuresTruncated: false,
    });
  });

  test("embed keeps partial results while limiting failure details to 20", async () => {
    const failures = Array.from({ length: 25 }, (_, index) => ({
      path: `doc-${index}.md`,
      hash: `hash-${index}`,
      seq: index,
      attempts: 3,
      reason: "failed",
    }));
    const { store, update } = createStoreDouble(undefined, async () => ({
      docsProcessed: 25,
      chunksEmbedded: 10,
      errors: 25,
      durationMs: 50,
      failures,
    }));
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "embed",
      arguments: {},
    });
    const structured = result.structuredContent as {
      docsProcessed: number;
      chunksEmbedded: number;
      failureCount: number;
      failures: unknown[];
      failuresTruncated: boolean;
    };

    expect(result.isError).toBe(true);
    expect(structured.docsProcessed).toBe(25);
    expect(structured.chunksEmbedded).toBe(10);
    expect(structured.failureCount).toBe(25);
    expect(structured.failures).toHaveLength(20);
    expect(structured.failuresTruncated).toBe(true);

    const afterPartialFailure = await client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterPartialFailure.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();
  });

  test("embed rejects timer overflow and accepts the maximum safe duration", async () => {
    const maxTimerMinutes = 35_791;
    const { store, embed } = createStoreDouble();
    const { client } = await createHarness(store);

    const overflow = await client.callTool({
      name: "embed",
      arguments: { timeoutMinutes: maxTimerMinutes + 0.25 },
    });

    expect(overflow.isError).toBe(true);
    expect(getFirstText(overflow)).toContain("Use 0 for no runtime limit");
    expect(embed).not.toHaveBeenCalled();

    const boundary = await client.callTool({
      name: "embed",
      arguments: { timeoutMinutes: maxTimerMinutes },
    });

    expect(boundary.isError).not.toBe(true);
    expect(embed).toHaveBeenCalledOnce();
    expect(embed.mock.calls[0]?.[0]).toEqual({
      force: false,
      maxDurationMs: 2_147_460_000,
      onProgress: expect.any(Function),
    });
  });

  test("embed rejects invalid numeric and chunk options before calling the Store", async () => {
    const { store, embed } = createStoreDouble();
    const { client } = await createHarness(store);
    const invalidArguments = [
      { maxDocsPerBatch: 0 },
      { maxDocsPerBatch: 1.5 },
      { maxBatchMiB: 0 },
      { maxBatchMiB: Number.POSITIVE_INFINITY },
      { timeoutMinutes: -1 },
      { chunkStrategy: "semantic" },
    ];

    for (const arguments_ of invalidArguments) {
      const result = await client.callTool({
        name: "embed",
        arguments: arguments_,
      });
      expect(result.isError).toBe(true);
    }

    expect(embed).not.toHaveBeenCalled();
  });

  test("a running update blocks embed across sessions but not read tools", async () => {
    const started = deferred();
    const release = deferred();
    const { store, embed } = createStoreDouble(async () => {
      started.resolve();
      await release.promise;
      return {
        collections: 2,
        indexed: 0,
        updated: 0,
        unchanged: 0,
        removed: 0,
        needsEmbedding: 0,
      };
    });
    const first = await createHarness(store);
    const second = await createHarness(store);

    const updateCall = first.client.callTool({
      name: "update",
      arguments: {},
    });
    await started.promise;

    try {
      const statusResult = await second.client.callTool({
        name: "status",
        arguments: {},
      });
      const busyResult = await second.client.callTool({
        name: "embed",
        arguments: {},
      });

      expect(statusResult.isError).not.toBe(true);
      expect(busyResult.isError).toBe(true);
      expect(getFirstText(busyResult)).toContain("busy");
      expect(embed).not.toHaveBeenCalled();
    } finally {
      release.resolve();
      await updateCall;
    }

    const afterRelease = await second.client.callTool({
      name: "embed",
      arguments: {},
    });
    expect(afterRelease.isError).toBe(false);
    expect(embed).toHaveBeenCalledOnce();
  });

  test("a running embed blocks update across sessions and releases afterward", async () => {
    const started = deferred();
    const release = deferred();
    const { store, update } = createStoreDouble(undefined, async () => {
      started.resolve();
      await release.promise;
      return {
        docsProcessed: 0,
        chunksEmbedded: 0,
        errors: 0,
        durationMs: 0,
        failures: [],
      };
    });
    const first = await createHarness(store);
    const second = await createHarness(store);

    const embedCall = first.client.callTool({
      name: "embed",
      arguments: {},
    });
    await started.promise;

    try {
      const busyResult = await second.client.callTool({
        name: "update",
        arguments: {},
      });

      expect(busyResult.isError).toBe(true);
      expect(getFirstText(busyResult)).toContain("busy");
      expect(update).not.toHaveBeenCalled();
    } finally {
      release.resolve();
      await embedCall;
    }

    const afterRelease = await second.client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterRelease.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();
  });

  test("an MCP abort without a progress token stops embed and sends no notification", async () => {
    const started = deferred();
    const continueWork = deferred();
    const finished = deferred();
    let continuedPastBoundary = false;
    const { store, update } = createStoreDouble(undefined, async (options) => {
      started.resolve();
      try {
        await continueWork.promise;
        options?.onProgress?.({
          chunksEmbedded: 1,
          totalChunks: 2,
          bytesProcessed: 100,
          totalBytes: 200,
          errors: 0,
          failures: [],
        });
        continuedPastBoundary = true;
        return {
          docsProcessed: 1,
          chunksEmbedded: 1,
          errors: 0,
          durationMs: 1,
          failures: [],
        };
      } finally {
        finished.resolve();
      }
    });
    const first = await createHarness(store);
    const second = await createHarness(store);
    const progress: ProgressNotification["params"][] = [];
    first.client.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      progress.push(notification.params);
    });
    const abortController = new AbortController();

    const embedCall = first.client.callTool(
      { name: "embed", arguments: {} },
      undefined,
      { signal: abortController.signal },
    ).then(
      () => ({ resolved: true }),
      () => ({ resolved: false }),
    );

    await started.promise;
    abortController.abort();
    continueWork.resolve();
    await finished.promise;

    expect((await embedCall).resolved).toBe(false);
    expect(continuedPastBoundary).toBe(false);
    expect(progress).toEqual([]);

    await new Promise<void>((resolve) => setImmediate(resolve));
    const afterAbort = await second.client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterAbort.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();
  });

  test("an MCP abort without a progress token stops update and sends no notification", async () => {
    const started = deferred();
    const continueWork = deferred();
    const finished = deferred();
    let continuedPastBoundary = false;
    const { store, embed } = createStoreDouble(async (options) => {
      started.resolve();
      try {
        await continueWork.promise;
        options?.onProgress?.({
          collection: "docs",
          current: 1,
          total: 2,
          file: "first.md",
        });
        continuedPastBoundary = true;
        return {
          collections: 1,
          indexed: 1,
          updated: 0,
          unchanged: 0,
          removed: 0,
          needsEmbedding: 1,
        };
      } finally {
        finished.resolve();
      }
    });
    const first = await createHarness(store);
    const second = await createHarness(store);
    const progress: ProgressNotification["params"][] = [];
    first.client.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      progress.push(notification.params);
    });
    const abortController = new AbortController();

    const updateCall = first.client.callTool(
      { name: "update", arguments: {} },
      undefined,
      { signal: abortController.signal },
    ).then(
      () => ({ resolved: true }),
      () => ({ resolved: false }),
    );

    await started.promise;
    abortController.abort();
    continueWork.resolve();
    await finished.promise;

    expect((await updateCall).resolved).toBe(false);
    expect(continuedPastBoundary).toBe(false);
    expect(progress).toEqual([]);

    await new Promise<void>((resolve) => setImmediate(resolve));
    const afterAbort = await second.client.callTool({
      name: "embed",
      arguments: {},
    });
    expect(afterAbort.isError).not.toBe(true);
    expect(embed).toHaveBeenCalledOnce();
  });

  test("an MCP abort stops embed at the next progress boundary and releases the lock", async () => {
    const started = deferred();
    const continueWork = deferred();
    const finished = deferred();
    let continuedPastBoundary = false;
    const { store, update } = createStoreDouble(undefined, async (options) => {
      started.resolve();
      try {
        await continueWork.promise;
        options?.onProgress?.({
          chunksEmbedded: 1,
          totalChunks: 2,
          bytesProcessed: 100,
          totalBytes: 200,
          errors: 0,
          failures: [],
        });
        continuedPastBoundary = true;
        return {
          docsProcessed: 1,
          chunksEmbedded: 1,
          errors: 0,
          durationMs: 1,
          failures: [],
        };
      } finally {
        finished.resolve();
      }
    });
    const first = await createHarness(store);
    const second = await createHarness(store);
    const abortController = new AbortController();

    const embedCall = first.client.callTool(
      {
        name: "embed",
        arguments: {},
        _meta: { progressToken: "cancelled-embed" },
      },
      undefined,
      { signal: abortController.signal },
    ).then(
      () => ({ resolved: true }),
      () => ({ resolved: false }),
    );

    await started.promise;
    abortController.abort();
    continueWork.resolve();
    await finished.promise;

    const callResult = await embedCall;
    expect(callResult.resolved).toBe(false);
    expect(continuedPastBoundary).toBe(false);

    await new Promise<void>((resolve) => setImmediate(resolve));
    const afterAbort = await second.client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterAbort.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();
  });

  test("an MCP abort during validation prevents the Store write and releases the lock", async () => {
    const validationStarted = deferred();
    const continueValidation = deferred();
    const validationReturned = deferred();
    const { store, update } = createStoreDouble();
    store.listCollections = async () => {
      validationStarted.resolve();
      await continueValidation.promise;
      validationReturned.resolve();
      return [{
        name: "docs",
        pwd: "/docs",
        glob_pattern: "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      }];
    };
    const first = await createHarness(store);
    const second = await createHarness(store);
    const abortController = new AbortController();

    const updateCall = first.client.callTool(
      {
        name: "update",
        arguments: { collections: ["docs"] },
      },
      undefined,
      { signal: abortController.signal },
    ).catch(() => undefined);

    await validationStarted.promise;
    abortController.abort();
    continueValidation.resolve();
    await validationReturned.promise;
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(update).not.toHaveBeenCalled();
    await updateCall;

    const afterAbort = await second.client.callTool({
      name: "embed",
      arguments: {},
    });
    expect(afterAbort.isError).not.toBe(true);
  });

  test("update rejects every unknown collection before calling the store", async () => {
    const { store, update } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "update",
      arguments: { collections: ["missing", "also-missing"] },
    });

    expect(result.isError).toBe(true);
    const text = getFirstText(result);
    expect(text).toContain("also-missing");
    expect(text).toContain("missing");
    expect(update).not.toHaveBeenCalled();
  });

  test("update forwards file progress with the original MCP token", async () => {
    const { store } = createStoreDouble(async (options) => {
      options?.onProgress?.({
        collection: "docs",
        current: 1,
        total: 2,
        file: "first.md",
      });
      options?.onProgress?.({
        collection: "docs",
        current: 2,
        total: 2,
        file: "second.md",
      });

      return {
        collections: 1,
        indexed: 2,
        updated: 0,
        unchanged: 0,
        removed: 0,
        needsEmbedding: 2,
      };
    });
    const { client } = await createHarness(store);
    const progress: ProgressNotification["params"][] = [];

    client.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      progress.push(notification.params);
    });

    await client.callTool({
      name: "update",
      arguments: {},
      _meta: { progressToken: "update-progress" },
    });

    expect(progress.map(item => item.progressToken)).toEqual([
      "update-progress",
      "update-progress",
    ]);
    expect(progress.map(item => item.progress)).toEqual([1, 2]);
    expect(progress.map(item => item.total)).toEqual([2, 2]);
    expect(progress.map(item => item.message)).toEqual([
      "docs/first.md",
      "docs/second.md",
    ]);
  });

  test("update ignores rejected progress notifications after Store success", async () => {
    const { store } = createStoreDouble(async (options) => {
      options?.onProgress?.({
        collection: "docs",
        current: 1,
        total: 1,
        file: "first.md",
      });

      return {
        collections: 1,
        indexed: 1,
        updated: 2,
        unchanged: 3,
        removed: 4,
        needsEmbedding: 5,
      };
    });
    const { client, serverTransport } = await createHarness(store);
    rejectProgressNotifications(serverTransport);

    const result = await client.callTool({
      name: "update",
      arguments: {},
      _meta: { progressToken: "rejected-update-progress" },
    });

    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toEqual({
      collections: 1,
      indexed: 1,
      updated: 2,
      unchanged: 3,
      removed: 4,
      needsEmbedding: 5,
    });
  });

  test("update handles rejected progress notifications when the Store throws", async () => {
    const { store } = createStoreDouble(async (options) => {
      options?.onProgress?.({
        collection: "docs",
        current: 1,
        total: 1,
        file: "first.md",
      });
      throw new Error("update failed");
    });
    const { client, serverTransport } = await createHarness(store);
    rejectProgressNotifications(serverTransport);
    const unhandledRejections: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => {
      unhandledRejections.push(reason);
    };
    process.on("unhandledRejection", onUnhandledRejection);

    try {
      const result = await client.callTool({
        name: "update",
        arguments: {},
        _meta: { progressToken: "rejected-update-progress" },
      });
      await new Promise<void>((resolve) => setImmediate(resolve));

      expect(result.isError).toBe(true);
      expect(unhandledRejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
    }
  });

  test("update converts Store exceptions into a safe tool error", async () => {
    const { store, embed } = createStoreDouble(async () => {
      throw new Error("database failed with SECRET_TOKEN\ninternal stack detail");
    });
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "update",
      arguments: {},
    });
    const text = getFirstText(result);

    expect(result.isError).toBe(true);
    expect(text).toContain("Update failed");
    expect(text).not.toContain("SECRET_TOKEN");
    expect(text).not.toContain("stack");

    const afterError = await client.callTool({
      name: "embed",
      arguments: {},
    });
    expect(afterError.isError).toBe(false);
    expect(embed).toHaveBeenCalledOnce();
  });

  test("update forwards an explicit collection selection unchanged", async () => {
    const { store, update } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "update",
      arguments: { collections: ["notes", "docs"] },
    });

    expect(update).toHaveBeenCalledWith({
      collections: ["notes", "docs"],
      onProgress: expect.any(Function),
    });
    expect(result.structuredContent).toEqual({
      collections: 2,
      indexed: 3,
      updated: 4,
      unchanged: 5,
      removed: 6,
      needsEmbedding: 7,
    });
    expect(getFirstText(result)).toContain("7 document(s) need embedding");
  });

  test("update without collections updates all and emits no unsolicited progress", async () => {
    const { store, update } = createStoreDouble();
    const { client } = await createHarness(store);
    const progress: ProgressNotification["params"][] = [];

    client.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      progress.push(notification.params);
    });

    const result = await client.callTool({
      name: "update",
      arguments: {},
    });

    expect(update).toHaveBeenCalledWith({
      onProgress: expect.any(Function),
    });
    expect(result.structuredContent).toMatchObject({
      collections: 2,
      indexed: 3,
      updated: 4,
      unchanged: 5,
      removed: 6,
      needsEmbedding: 7,
    });
    expect(progress).toEqual([]);
  });

  test("update rejects an explicitly empty collection list before calling the store", async () => {
    const { store, update } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "update",
      arguments: { collections: [] },
    });
    const text = getFirstText(result);

    expect(result.isError).toBe(true);
    expect(text).toContain("collections");
    expect(update).not.toHaveBeenCalled();
  });
});
