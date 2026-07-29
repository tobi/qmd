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
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
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

  const addCollection = vi.fn(async (
    _name: string,
    _options: { path: string; pattern?: string; ignore?: string[] },
  ) => undefined);
  const renameCollection = vi.fn(async (
    _oldName: string,
    _newName: string,
  ) => true);
  const removeCollection = vi.fn(async (_name: string) => ({
    removed: true,
    deletedDocs: 3,
    cleanedHashes: 2,
  }));

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
    addCollection,
    renameCollection,
    removeCollection,
    update,
    embed,
  } as unknown as QMDStore;

  return {
    store,
    addCollection,
    renameCollection,
    removeCollection,
    update,
    embed,
  };
}

async function createHarness(
  store: QMDStore,
  options: { enableCollectionManagement?: boolean } = {},
) {
  const server = await createMcpServer(store, options);
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
  test("collection management registration depends only on the operator opt-in", async () => {
    const namesFor = async (options: { enableCollectionManagement?: boolean }) => {
      const { store } = createStoreDouble();
      const { client } = await createHarness(store, options);
      return (await client.listTools()).tools.map(tool => tool.name).sort();
    };

    const disabled = await namesFor({});
    const enabled = await namesFor({ enableCollectionManagement: true });

    expect(disabled).not.toContain("collection_add");
    expect(enabled.filter(name => !disabled.includes(name))).toEqual([
      "collection_add",
      "collection_remove",
      "collection_rename",
    ]);
  });

  test("tools/list exposes collection_add as an opt-in local non-destructive write", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const tool = (await client.listTools()).tools.find(
      candidate => candidate.name === "collection_add"
    );

    expect(tool?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    });
    expect(Object.keys(tool?.inputSchema.properties ?? {}).sort()).toEqual([
      "ignore",
      "name",
      "path",
      "pattern",
    ]);
    expect(tool?.inputSchema.required).toEqual(["path"]);
    expect(tool?.description).toMatch(/without indexing.*update.*embed/i);
  });

  test("collection_add resolves paths, persists options, and returns the show contract", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-add-"));
    const realDirectory = join(testDir, "docs");
    const linkedDirectory = join(testDir, "linked-docs");
    await mkdir(realDirectory);
    await symlink(realDirectory, linkedDirectory);
    const { store, addCollection, update, embed } = createStoreDouble();
    const configured: Awaited<ReturnType<QMDStore["listCollections"]>> = [];
    store.listCollections = async () => configured;
    addCollection.mockImplementation(async (name, options) => {
      configured.push({
        name,
        pwd: options.path,
        glob_pattern: options.pattern ?? "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      });
    });
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    try {
      const addResult = await client.callTool({
        name: "collection_add",
        arguments: {
          path: linkedDirectory,
          name: "manual",
          pattern: "guides/**/*.md",
          ignore: ["drafts/**", "private/**"],
        },
      });
      const showResult = await client.callTool({
        name: "collection_show",
        arguments: { name: "manual" },
      });

      expect(addResult.structuredContent).toEqual(showResult.structuredContent);
      expect(addResult.structuredContent).toEqual({
        collection: {
          name: "manual",
          path: realDirectory,
          pattern: "guides/**/*.md",
          documents: 0,
          indexedDocuments: 0,
          lastUpdated: null,
          includeByDefault: true,
        },
      });
      expect(addCollection).toHaveBeenCalledWith("manual", {
        path: realDirectory,
        pattern: "guides/**/*.md",
        ignore: ["drafts/**", "private/**"],
      });
      expect(getFirstText(addResult)).toMatch(/without indexing.*update.*embed/i);
      expect(update).not.toHaveBeenCalled();
      expect(embed).not.toHaveBeenCalled();
    } finally {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("collection_add derives the CLI name and default pattern", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-defaults-"));
    const directory = join(testDir, "docs");
    await mkdir(directory);
    const { store, addCollection } = createStoreDouble();
    const configured: Awaited<ReturnType<QMDStore["listCollections"]>> = [];
    store.listCollections = async () => configured;
    addCollection.mockImplementation(async (name, options) => {
      configured.push({
        name,
        pwd: options.path,
        glob_pattern: options.pattern ?? "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      });
    });
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    try {
      const result = await client.callTool({
        name: "collection_add",
        arguments: { path: directory },
      });

      expect(result.structuredContent).toMatchObject({
        collection: {
          name: "docs",
          path: directory,
          pattern: "**/*.md",
        },
      });
      expect(addCollection).toHaveBeenCalledWith("docs", {
        path: directory,
        pattern: "**/*.md",
      });
    } finally {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("collection_add rejects invalid paths and collisions before the Store write", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-add-errors-"));
    const directory = join(testDir, "docs");
    const file = join(testDir, "file.md");
    await mkdir(directory);
    await writeFile(file, "not a directory");
    const { store, addCollection } = createStoreDouble();
    store.listCollections = async () => [{
      name: "existing",
      pwd: directory,
      glob_pattern: "**/*.md",
      doc_count: 0,
      active_count: 0,
      last_modified: null,
      includeByDefault: true,
    }];
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    try {
      const missing = await client.callTool({
        name: "collection_add",
        arguments: { path: join(testDir, "missing") },
      });
      const notDirectory = await client.callTool({
        name: "collection_add",
        arguments: { path: file },
      });
      const nameCollision = await client.callTool({
        name: "collection_add",
        arguments: { path: directory, name: "existing", pattern: "other/*.md" },
      });
      const pathCollision = await client.callTool({
        name: "collection_add",
        arguments: { path: directory, name: "other" },
      });

      expect(getFirstText(missing)).toMatch(/does not exist.*check.*retry/i);
      expect(getFirstText(notDirectory)).toMatch(/not a directory.*choose.*retry/i);
      expect(getFirstText(nameCollision)).toMatch(/existing.*already exists.*different name/i);
      expect(getFirstText(pathCollision)).toMatch(/existing.*path and pattern.*update.*remove/i);
      expect([missing, notDirectory, nameCollision, pathCollision])
        .toSatisfy(results => results.every(result => result.isError === true));
      expect(addCollection).not.toHaveBeenCalled();
    } finally {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("collection_add warns against blind retry when read-back fails", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-readback-"));
    const { store, addCollection } = createStoreDouble();
    store.listCollections = async () => [];
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    try {
      const result = await client.callTool({
        name: "collection_add",
        arguments: { path: testDir, name: "unconfirmed" },
      });

      expect(addCollection).toHaveBeenCalledOnce();
      expect(result.isError).toBe(true);
      expect(getFirstText(result)).toMatch(/may have been registered.*collection_list.*before retrying/i);
    } finally {
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("collection_add is not callable without explicit enablement", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "collection_add",
      arguments: { path: "/docs" },
    });

    expect(result.isError).toBe(true);
    expect(getFirstText(result)).toMatch(/tool collection_add not found/i);
  });

  test("tools/list exposes collection_rename as an opt-in destructive local write", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const tool = (await client.listTools()).tools.find(
      candidate => candidate.name === "collection_rename"
    );

    expect(tool?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: false,
      openWorldHint: false,
    });
    expect(Object.keys(tool?.inputSchema.properties ?? {}).sort()).toEqual([
      "newName",
      "oldName",
    ]);
    expect(tool?.inputSchema.required).toEqual(["oldName", "newName"]);
  });

  test("collection_rename updates indexed paths and returns the show contract", async () => {
    const { store, renameCollection } = createStoreDouble();
    const configured: Awaited<ReturnType<QMDStore["listCollections"]>> = [{
      name: "old-name",
      pwd: "/docs",
      glob_pattern: "**/*.md",
      doc_count: 3,
      active_count: 3,
      last_modified: "2026-07-29T12:00:00.000Z",
      includeByDefault: true,
    }];
    store.listCollections = async () => configured;
    renameCollection.mockImplementation(async (oldName, newName) => {
      const collection = configured.find(candidate => candidate.name === oldName);
      if (!collection) return false;
      collection.name = newName;
      return true;
    });
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const result = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "old-name", newName: "new-name" },
    });

    expect(renameCollection).toHaveBeenCalledWith("old-name", "new-name");
    expect(result.structuredContent).toEqual({
      collection: {
        name: "new-name",
        path: "/docs",
        pattern: "**/*.md",
        documents: 3,
        indexedDocuments: 3,
        lastUpdated: "2026-07-29T12:00:00.000Z",
        includeByDefault: true,
      },
    });
    expect(getFirstText(result)).toMatch(/old-name.*new-name.*qmd:\/\/new-name\//i);
  });

  test("collection_rename rejects unknown sources and occupied targets before the Store write", async () => {
    const { store, renameCollection } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const unknown = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "missing", newName: "new-name" },
    });
    const occupied = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "notes" },
    });

    expect(unknown.isError).toBe(true);
    expect(getFirstText(unknown)).toMatch(/missing.*does not exist.*collection_list/i);
    expect(occupied.isError).toBe(true);
    expect(getFirstText(occupied)).toMatch(/notes.*already exists.*different name/i);
    expect(renameCollection).not.toHaveBeenCalled();
  });

  test("collection_rename reports unconfirmed outcomes without recommending a blind retry", async () => {
    const { store, renameCollection } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    renameCollection.mockResolvedValueOnce(false);
    const unchanged = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "manual" },
    });

    renameCollection.mockResolvedValueOnce(true);
    const unreadable = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "manual" },
    });

    renameCollection.mockRejectedValueOnce(new Error("sensitive store failure"));
    const ambiguous = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "manual" },
    });

    expect(getFirstText(unchanged)).toMatch(/no collection was renamed.*collection_list/i);
    expect(getFirstText(unreadable)).toMatch(/may have been renamed.*collection_show.*before retrying/i);
    expect(getFirstText(ambiguous)).toMatch(/may have been renamed.*collection_show.*before retrying/i);
    expect(getFirstText(ambiguous)).not.toContain("sensitive store failure");
    expect([unchanged, unreadable, ambiguous])
      .toSatisfy(results => results.every(result => result.isError === true));
  });

  test("collection_rename reports a prevalidation read failure as safe to retry", async () => {
    const { store, renameCollection } = createStoreDouble();
    store.listCollections = vi.fn().mockRejectedValueOnce(
      new Error("sensitive read failure")
    );
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const result = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "manual" },
    });

    expect(result.isError).toBe(true);
    expect(getFirstText(result)).toMatch(/could not read.*status.*logs.*retry/i);
    expect(getFirstText(result)).not.toMatch(/may have been renamed|sensitive/i);
    expect(renameCollection).not.toHaveBeenCalled();
  });

  test("tools/list exposes collection_remove as an opt-in destructive local write", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const tool = (await client.listTools()).tools.find(
      candidate => candidate.name === "collection_remove"
    );

    expect(tool?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: false,
      openWorldHint: false,
    });
    expect(Object.keys(tool?.inputSchema.properties ?? {})).toEqual(["name"]);
    expect(tool?.inputSchema.required).toEqual(["name"]);
    expect(tool?.description).toMatch(/source files.*unchanged/i);
  });

  test("collection_remove reports document deletion and global orphan cleanup", async () => {
    const { store, removeCollection } = createStoreDouble();
    const configured = await store.listCollections();
    store.listCollections = async () => configured;
    removeCollection.mockImplementation(async (name) => {
      const index = configured.findIndex(collection => collection.name === name);
      if (index < 0) {
        return { removed: false, deletedDocs: 0, cleanedHashes: 0 };
      }
      configured.splice(index, 1);
      return { removed: true, deletedDocs: 7, cleanedHashes: 5 };
    });
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const result = await client.callTool({
      name: "collection_remove",
      arguments: { name: "docs" },
    });

    expect(removeCollection).toHaveBeenCalledWith("docs");
    expect(result.structuredContent).toEqual({
      removed: true,
      deletedDocs: 7,
      cleanedHashes: 5,
    });
    expect(getFirstText(result)).toMatch(
      /removed.*docs.*7.*documents.*5.*globally orphaned.*source files.*unchanged/i
    );
  });

  test("collection_remove rejects an unknown collection before the Store write", async () => {
    const { store, removeCollection } = createStoreDouble();
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const result = await client.callTool({
      name: "collection_remove",
      arguments: { name: "missing" },
    });

    expect(result.isError).toBe(true);
    expect(getFirstText(result)).toMatch(/missing.*does not exist.*collection_list/i);
    expect(removeCollection).not.toHaveBeenCalled();
  });

  test("collection_remove distinguishes safe prevalidation failures from ambiguous mutations", async () => {
    const { store, removeCollection } = createStoreDouble();
    store.listCollections = vi.fn()
      .mockRejectedValueOnce(new Error("sensitive read failure"))
      .mockResolvedValue([
        {
          name: "docs",
          pwd: "/docs",
          glob_pattern: "**/*.md",
          doc_count: 0,
          active_count: 0,
          last_modified: null,
          includeByDefault: true,
        },
      ]);
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const unreadable = await client.callTool({
      name: "collection_remove",
      arguments: { name: "docs" },
    });

    removeCollection.mockRejectedValueOnce(new Error("sensitive write failure"));
    const ambiguous = await client.callTool({
      name: "collection_remove",
      arguments: { name: "docs" },
    });

    expect(getFirstText(unreadable)).toMatch(/could not read.*status.*logs.*retry/i);
    expect(getFirstText(unreadable)).not.toMatch(/may have been removed|sensitive/i);
    expect(getFirstText(ambiguous)).toMatch(/may have been removed.*collection_list.*before retrying/i);
    expect(getFirstText(ambiguous)).not.toContain("sensitive write failure");
    expect([unreadable, ambiguous])
      .toSatisfy(results => results.every(result => result.isError === true));
  });

  test("collection_remove reports a failed Store result as unchanged", async () => {
    const { store, removeCollection } = createStoreDouble();
    removeCollection.mockResolvedValueOnce({
      removed: false,
      deletedDocs: 0,
      cleanedHashes: 0,
    });
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const result = await client.callTool({
      name: "collection_remove",
      arguments: { name: "docs" },
    });

    expect(result.isError).toBe(true);
    expect(getFirstText(result)).toMatch(/no collection was removed.*collection_list/i);
  });

  test("tools/list exposes read-only collection tools", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store);

    const { tools } = await client.listTools();
    const collectionList = tools.find((tool) => tool.name === "collection_list");
    const collectionShow = tools.find((tool) => tool.name === "collection_show");

    expect(collectionList?.annotations).toEqual({
      readOnlyHint: true,
      openWorldHint: false,
    });
    expect(collectionShow?.annotations).toEqual({
      readOnlyHint: true,
      openWorldHint: false,
    });
    expect(collectionList?.inputSchema.properties).toEqual({});
    expect(collectionShow?.inputSchema.required).toEqual(["name"]);
  });

  test("collection_list normalizes count semantics and sorts configured collections", async () => {
    const { store } = createStoreDouble();
    store.listCollections = async () => [
      {
        name: "notes",
        pwd: "/notes",
        glob_pattern: "notes/**/*.md",
        doc_count: 5,
        active_count: 3,
        last_modified: "2026-07-29T12:00:00.000Z",
        includeByDefault: false,
      },
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
        name: "Ärger",
        pwd: "/aerger",
        glob_pattern: "**/*.md",
        doc_count: 1,
        active_count: 1,
        last_modified: null,
        includeByDefault: false,
      },
    ];
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "collection_list",
      arguments: {},
    });

    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toEqual({
      collections: [
        {
          name: "docs",
          path: "/docs",
          pattern: "**/*.md",
          documents: 0,
          indexedDocuments: 0,
          lastUpdated: null,
          includeByDefault: true,
        },
        {
          name: "notes",
          path: "/notes",
          pattern: "notes/**/*.md",
          documents: 3,
          indexedDocuments: 5,
          lastUpdated: "2026-07-29T12:00:00.000Z",
          includeByDefault: false,
        },
        {
          name: "Ärger",
          path: "/aerger",
          pattern: "**/*.md",
          documents: 1,
          indexedDocuments: 1,
          lastUpdated: null,
          includeByDefault: false,
        },
      ],
    });
    expect(getFirstText(result)).toMatch(/3 configured collections/i);
  });

  test("collection_show returns the same wrapped collection contract", async () => {
    const { store } = createStoreDouble();
    store.listCollections = async () => [{
      name: "notes",
      pwd: "/notes",
      glob_pattern: "notes/**/*.md",
      doc_count: 5,
      active_count: 3,
      last_modified: "2026-07-29T12:00:00.000Z",
      includeByDefault: false,
    }];
    const { client } = await createHarness(store);

    const listResult = await client.callTool({
      name: "collection_list",
      arguments: {},
    });
    const showResult = await client.callTool({
      name: "collection_show",
      arguments: { name: "notes" },
    });
    const listCollection = (listResult.structuredContent as {
      collections: unknown[];
    }).collections[0];

    expect(showResult.isError).not.toBe(true);
    expect(showResult.structuredContent).toEqual({
      collection: listCollection,
    });
    expect(showResult.structuredContent).toEqual({
      collection: {
        name: "notes",
        path: "/notes",
        pattern: "notes/**/*.md",
        documents: 3,
        indexedDocuments: 5,
        lastUpdated: "2026-07-29T12:00:00.000Z",
        includeByDefault: false,
      },
    });
    expect(getFirstText(showResult)).toMatch(/notes.*\/notes.*3 active of 5 indexed documents/i);
  });

  test("collection_show rejects unknown names with a useful tool error", async () => {
    const { store } = createStoreDouble();
    const { client } = await createHarness(store);

    const result = await client.callTool({
      name: "collection_show",
      arguments: { name: "missing" },
    });

    expect(result.isError).toBe(true);
    expect(result.structuredContent).toBeUndefined();
    expect(getFirstText(result)).toMatch(/missing.*collection_list/i);
  });

  test("collection tools mask Store exceptions", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-mask-"));
    const { store } = createStoreDouble();
    store.listCollections = async () => {
      throw new Error("secret database path and stacktrace");
    };
    const { client } = await createHarness(store);
    const { client: enabledClient } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    try {
      const listResult = await client.callTool({
        name: "collection_list",
        arguments: {},
      });
      const showResult = await client.callTool({
        name: "collection_show",
        arguments: { name: "docs" },
      });
      const addResult = await enabledClient.callTool({
        name: "collection_add",
        arguments: { path: testDir },
      });

      expect(listResult.isError).toBe(true);
      expect(showResult.isError).toBe(true);
      expect(addResult.isError).toBe(true);
      expect(getFirstText(listResult)).not.toMatch(/secret|database path|stacktrace/i);
      expect(getFirstText(showResult)).not.toMatch(/secret|database path|stacktrace/i);
      expect(getFirstText(addResult)).not.toMatch(/secret|database path|stacktrace/i);
    } finally {
      await rm(testDir, { recursive: true, force: true });
    }
  });

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
      expect(toolNames).toContain("collection_list");
      expect(toolNames).toContain("collection_show");
      expect(toolNames).not.toContain("collection_add");
    } finally {
      await client.close();
      await rm(testDir, { recursive: true, force: true });
    }
  });

  test("stdio start option enables collection management", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-mcp-stdio-enabled-"));
    const client = new Client({
      name: "qmd-stdio-collection-test",
      version: "1.0.0",
    });
    const transport = new StdioClientTransport({
      command: "node",
      args: [
        "--import",
        "tsx",
        "src/cli/qmd.ts",
        "mcp",
        "--enable-collection-management",
      ],
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
      const toolNames = (await client.listTools()).tools.map(tool => tool.name);
      expect(toolNames).toContain("collection_add");
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
    const { client: managementClient } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const instructions = client.getInstructions() ?? "";
    const managementInstructions = managementClient.getInstructions() ?? "";

    expect(instructions).toContain("Call the `update` MCP tool");
    expect(instructions).toContain("inspect `needsEmbedding`");
    expect(instructions).toContain(
      "Call the `embed` MCP tool only when `needsEmbedding` is greater than 0"
    );
    expect(instructions).not.toContain("qmd embed");
    expect(instructions).not.toMatch(
      /collection_add|collection_rename|collection_remove/
    );
    expect(managementInstructions).toMatch(
      /collection_add.*update.*needsEmbedding.*embed/is
    );
    expect(managementInstructions).toMatch(
      /collection_rename.*collection_remove/is
    );
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

  test("a running update blocks embed and every collection write but not read tools", async () => {
    const started = deferred();
    const release = deferred();
    const {
      store,
      addCollection,
      renameCollection,
      removeCollection,
      embed,
    } = createStoreDouble(async () => {
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
    const second = await createHarness(store, {
      enableCollectionManagement: true,
    });

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
      const busyAdd = await second.client.callTool({
        name: "collection_add",
        arguments: { path: "/does/not/need/to/exist" },
      });
      const busyRename = await second.client.callTool({
        name: "collection_rename",
        arguments: { oldName: "docs", newName: "manual" },
      });
      const busyRemove = await second.client.callTool({
        name: "collection_remove",
        arguments: { name: "docs" },
      });

      expect(statusResult.isError).not.toBe(true);
      expect([busyResult, busyAdd, busyRename, busyRemove])
        .toSatisfy(results => results.every(result =>
          result.isError === true && getFirstText(result).includes("busy")
        ));
      expect(embed).not.toHaveBeenCalled();
      expect(addCollection).not.toHaveBeenCalled();
      expect(renameCollection).not.toHaveBeenCalled();
      expect(removeCollection).not.toHaveBeenCalled();
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

  test("a running collection_add blocks writes across sessions while reads stay available", async () => {
    const testDir = await mkdtemp(join(tmpdir(), "qmd-collection-lock-"));
    const started = deferred();
    const release = deferred();
    const { store, addCollection, update, embed } = createStoreDouble();
    const configured: Awaited<ReturnType<QMDStore["listCollections"]>> = [];
    store.listCollections = vi.fn(async () => configured);
    addCollection.mockImplementation(async (name, options) => {
      started.resolve();
      await release.promise;
      configured.push({
        name,
        pwd: options.path,
        glob_pattern: options.pattern ?? "**/*.md",
        doc_count: 0,
        active_count: 0,
        last_modified: null,
        includeByDefault: true,
      });
    });
    const first = await createHarness(store, {
      enableCollectionManagement: true,
    });
    const second = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const addCall = first.client.callTool({
      name: "collection_add",
      arguments: { path: testDir, name: "shared" },
    });
    await started.promise;

    try {
      const busyAdd = await second.client.callTool({
        name: "collection_add",
        arguments: { path: testDir, name: "shared" },
      });
      expect(store.listCollections).toHaveBeenCalledTimes(1);

      const listResult = await second.client.callTool({
        name: "collection_list",
        arguments: {},
      });
      const showResult = await second.client.callTool({
        name: "collection_show",
        arguments: { name: "shared" },
      });
      const busyUpdate = await second.client.callTool({
        name: "update",
        arguments: {},
      });
      const busyEmbed = await second.client.callTool({
        name: "embed",
        arguments: {},
      });

      expect(listResult.isError).not.toBe(true);
      expect(getFirstText(showResult)).not.toContain("busy");
      expect([busyAdd, busyUpdate, busyEmbed])
        .toSatisfy(results => results.every(result =>
          result.isError === true && getFirstText(result).includes("busy")
        ));
      expect(addCollection).toHaveBeenCalledOnce();
      expect(update).not.toHaveBeenCalled();
      expect(embed).not.toHaveBeenCalled();
    } finally {
      release.resolve();
      expect((await addCall).isError).not.toBe(true);
    }

    const collision = await second.client.callTool({
      name: "collection_add",
      arguments: { path: testDir, name: "shared" },
    });
    expect(collision.isError).toBe(true);
    expect(getFirstText(collision)).toMatch(/shared.*already exists/i);
    expect(addCollection).toHaveBeenCalledOnce();

    const afterRelease = await second.client.callTool({
      name: "update",
      arguments: {},
    });
    expect(afterRelease.isError).not.toBe(true);
    expect(update).toHaveBeenCalledOnce();

    await rm(testDir, { recursive: true, force: true });
  });

  test("collection write validation and Store errors release the maintenance lock", async () => {
    const { store, renameCollection, update } = createStoreDouble();
    renameCollection.mockRejectedValueOnce(new Error("rename failed"));
    const { client } = await createHarness(store, {
      enableCollectionManagement: true,
    });

    const invalid = await client.callTool({
      name: "collection_remove",
      arguments: { name: "missing" },
    });
    const storeError = await client.callTool({
      name: "collection_rename",
      arguments: { oldName: "docs", newName: "manual" },
    });
    const afterErrors = await client.callTool({
      name: "update",
      arguments: {},
    });

    expect(invalid.isError).toBe(true);
    expect(storeError.isError).toBe(true);
    expect(afterErrors.isError).not.toBe(true);
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
