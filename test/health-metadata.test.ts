import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, renameSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, type QMDStore, type HealthStatus, type HealthMetadata, type CollectionConfig } from "../src/index.js";
import { environmentIdentity, metadataIdentity, resolveIndexContext } from "../src/index-context.js";
import { getConfigPath, setConfigSource } from "../src/collections.js";
import { getEmbeddingFingerprint, insertContent, insertDocument, syncConfigToDb } from "../src/store.js";
import { LlamaCpp, resolveEmbedContextSize, resolveRerankContextSize, setNodeLlamaCppModuleForTest } from "../src/llm.js";
import { createMcpServer } from "../src/mcp/server.js";
import {
  InMemoryTransport,
  type McpServer, type JSONRPCRequest, type JSONRPCResultResponse, type InitializeResult, type ListToolsResult, type CallToolResult,
} from "@modelcontextprotocol/server";

let root: string;
let configPath: string;
let docs: string;
let stores: QMDStore[];
let modelCalls: number;
let savedEnv: Record<string, string | undefined>;
let savedFetch: typeof globalThis.fetch;
let modelExistsSpy: { mockRestore(): void } | undefined;

function restoreEnvironment() {
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

function configuration(model = "fixture:embedding-a", included = true): CollectionConfig {
  return {
    models: { embed: model, generate: "fixture:generate", rerank: "fixture:rerank" },
    global_context: "Fixture global context",
    collections: included ? { notes: {
      path: docs, pattern: "**/*.md",
      context: { "/": "Fixture collection context", "/sub": "Fixture subcontext" },
    } } : {},
  };
}

function metadata(snapshot: HealthStatus): HealthMetadata {
  if ("unavailable" in snapshot.health) throw new Error("Expected an owned native snapshot");
  return snapshot.health;
}

async function open(name = "index") {
  const store = await createStore({ dbPath: join(root, `${name}.sqlite`), configPath });
  stores.push(store);
  return store;
}

function logicalState(store: QMDStore) {
  const db = store.internal.db;
  return JSON.stringify({
    changes: db.prepare("SELECT total_changes() AS n").get(),
    schemaVersion: db.prepare("PRAGMA schema_version").get(),
    schema: db.prepare("SELECT type, name, sql FROM sqlite_master ORDER BY name").all(),
    collections: db.prepare("SELECT * FROM store_collections ORDER BY name").all(),
    config: db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    content: db.prepare("SELECT * FROM content ORDER BY hash").all(),
    documents: db.prepare("SELECT * FROM documents ORDER BY id").all(),
    vectors: db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    llmCache: db.prepare("SELECT * FROM llm_cache ORDER BY hash").all(),
  });
}
function useLegacyVectorSchema(store: QMDStore) {
  const db = store.internal.db;
  const indexes = db.prepare(
    "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'content_vectors'",
  ).all<{ name: string; sql: string | null }>();
  for (const index of indexes) {
    if (index.sql?.includes("total_chunks")) {
      db.exec(`DROP INDEX "${index.name.replaceAll('"', '""')}"`);
    }
  }
  db.exec("ALTER TABLE content_vectors DROP COLUMN total_chunks");
}


beforeEach(() => {
  savedEnv = {
    QMD_CONFIG_DIR: process.env.QMD_CONFIG_DIR,
    INDEX_PATH: process.env.INDEX_PATH,
    QMD_EMBED_MODEL: process.env.QMD_EMBED_MODEL,
    QMD_EMBED_CONTEXT_SIZE: process.env.QMD_EMBED_CONTEXT_SIZE,
    QMD_RERANK_CONTEXT_SIZE: process.env.QMD_RERANK_CONTEXT_SIZE,
    XDG_CACHE_HOME: process.env.XDG_CACHE_HOME,
  };
  savedFetch = globalThis.fetch;
  modelExistsSpy = undefined;
  root = mkdtempSync(join(tmpdir(), "qmd-health-metadata-"));
  docs = join(root, "documents");
  configPath = join(root, "index.yml");
  mkdirSync(docs);
  stores = [];
  modelCalls = 0;
  writeFileSync(configPath, YAML.stringify(configuration()));
  process.env.QMD_CONFIG_DIR = root;
  process.env.INDEX_PATH = "";
  setNodeLlamaCppModuleForTest({
    getLlama: async () => { modelCalls++; throw new Error("No model work in health fixtures"); },
    resolveModelFile: async () => { modelCalls++; throw new Error("No model downloads in health fixtures"); },
    LlamaChatSession: class {
      async prompt(): Promise<string> { modelCalls++; throw new Error("No inference in health fixtures"); }
    },
    LlamaLogLevel: { error: 0 },
  });
  modelExistsSpy = vi.spyOn(LlamaCpp.prototype, "modelExists").mockImplementation(async () => {
    modelCalls++;
    throw new Error("No model file inspection in health fixtures");
  });
});

afterEach(async () => {
  try {
    for (const store of stores) await store.close();
  } finally {
    setConfigSource();
    setNodeLlamaCppModuleForTest(null);
    restoreEnvironment();
    modelExistsSpy?.mockRestore();
    globalThis.fetch = savedFetch;
    rmSync(root, { recursive: true, force: true });
    expect(modelCalls).toBe(0);
  }
});

describe("native health metadata", () => {
  test("includes configured-empty collections without SQL/config writes or model work", async () => {
    const store = await open();
    const sqlBefore = logicalState(store);
    const yamlBefore = readFileSync(configPath);
    const snapshot = await store.getHealthSnapshot();
    const health = metadata(snapshot);
    expect(snapshot.collections).toEqual([]); // Existing document-derived status is unchanged.
    expect(health.catalog).toEqual({ complete: true, entries: [{
      name: metadataIdentity("notes"), path: metadataIdentity(docs), documents: 0,
    }] });
    expect(health.active).toEqual(health.current);
    expect(health.countModel).toBe(health.active.embeddingModel);
    expect(health.pendingEmbeddings).toBe(0);
    expect(logicalState(store)).toBe(sqlBefore);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
    expect(JSON.stringify(health)).not.toContain("fixture:embedding-a");
    expect(JSON.stringify(health)).not.toContain(root);
    expect(JSON.stringify(health)).not.toContain("Fixture global context");
    expect(JSON.stringify(health)).not.toContain("Fixture collection context");
  });

  test("keeps old-model backlog bound to the actual store after a config model change", async () => {
    const store = await open();
    const db = store.internal.db;
    const timestamp = "2026-01-01T00:00:00.000Z";
    insertContent(db, "fixture-hash", "# Synthetic document", timestamp);
    insertDocument(db, "notes", "note.md", "Synthetic", "fixture-hash", timestamp, timestamp);
    db.prepare(`INSERT INTO content_vectors
      (hash, seq, pos, model, embed_fingerprint, embedded_at, total_chunks)
      VALUES (?, 0, 0, ?, ?, ?, 1)`).run(
      "fixture-hash", "fixture:embedding-a", getEmbeddingFingerprint("fixture:embedding-a"), timestamp,
    );
    expect(metadata(await store.getHealthSnapshot()).pendingEmbeddings).toBe(0);
    writeFileSync(configPath, YAML.stringify(configuration("fixture:embedding-b")));
    const health = metadata(await store.getHealthSnapshot());
    expect(health.active.embeddingModel).toBe(metadataIdentity("fixture:embedding-a"));
    expect(health.current.embeddingModel).toBe(metadataIdentity("fixture:embedding-b"));
    expect(health.countModel).toBe(health.active.embeddingModel);
    expect(health.active.configuration).not.toBe(health.current.configuration);
  });

  test("remote providers leave native counts available without claiming a version 1 binding", async () => {
    const config = configuration("https://embed.invalid/v1#fixture");
    writeFileSync(configPath, YAML.stringify(config));
    const store = await open();
    expect(await store.getHealthSnapshot()).toMatchObject({
      needsEmbedding: 0,
      health: { schemaVersion: 1, unavailable: true },
    });
    writeFileSync(configPath, YAML.stringify(configuration()));
    expect((await store.getHealthSnapshot()).health).toEqual({ schemaVersion: 1, unavailable: true });
    expect(modelCalls).toBe(0);
  });

  test.each(["collection", "global"] as const)("binds actual %s context until an owner refresh", async field => {
    const store = await open();
    for (const text of ["Changed fixture context", undefined]) {
      const original = metadata(await store.getHealthSnapshot());
      expect(original.active).toEqual(original.current);
      const contextBefore = store.internal.getContextForPath("notes", "note.md");
      const edited = configuration();
      if (field === "collection") {
        edited.collections.notes!.context = text === undefined ? undefined : { "/": text };
      } else {
        edited.global_context = text;
      }
      writeFileSync(configPath, YAML.stringify(edited));
      const sqlBefore = logicalState(store);
      const yamlBefore = readFileSync(configPath);
      const stale = metadata(await store.getHealthSnapshot());
      expect(stale.active).toEqual(original.active);
      expect(stale.current.configuration).not.toBe(stale.active.configuration);
      expect(stale.current.embeddingModel).toBe(stale.active.embeddingModel);
      expect(store.internal.getContextForPath("notes", "note.md")).toBe(contextBefore);
      expect(logicalState(store)).toBe(sqlBefore);
      expect(readFileSync(configPath)).toEqual(yamlBefore);

      // Only the owner refresh writes; a health read must never perform this sync.
      syncConfigToDb(store.internal.db, edited);
      const refreshedSql = logicalState(store);
      const refreshed = metadata(await store.getHealthSnapshot());
      expect(refreshed.active).toEqual(refreshed.current);
      expect(refreshed.active.configuration).toBe(stale.current.configuration);
      expect(store.internal.getContextForPath("notes", "note.md")).not.toBe(contextBefore);
      expect(logicalState(store)).toBe(refreshedSql);
      expect(readFileSync(configPath)).toEqual(yamlBefore);
    }
  });

  test("binds contexts in inline and DB-only stores using native DB state", async () => {
    const config = configuration();
    delete config.models;
    const inline = await createStore({ dbPath: join(root, "inline.sqlite"), config });
    stores.push(inline);
    const inlineHealth = metadata(await inline.getHealthSnapshot());
    expect(inlineHealth.active).toEqual(inlineHealth.current);
    const reopened = await createStore({ dbPath: inline.dbPath });
    stores.push(reopened);
    const before = logicalState(reopened);
    const health = metadata(await reopened.getHealthSnapshot());
    expect(health.active).toEqual(health.current);
    expect(health.active.configuration).toBe(inlineHealth.active.configuration);
    expect(await reopened.getGlobalContext()).toBe(config.global_context);
    expect(logicalState(reopened)).toBe(before);
  });

  test("ignores YAML formatting and fields outside the health binding", async () => {
    const store = await open();
    const original = metadata(await store.getHealthSnapshot());
    const edited = configuration();
    edited.editor_uri = "fixture://editor/{path}";
    edited.collections.notes!.update = "must-not-run";
    writeFileSync(configPath, `# Comment outside the binding\n${YAML.stringify(edited)}`);
    const before = logicalState(store);
    const yamlBefore = readFileSync(configPath);
    const health = metadata(await store.getHealthSnapshot());
    expect(health.active).toEqual(original.active);
    expect(health.current).toEqual(original.current);
    expect(logicalState(store)).toBe(before);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
  });

  test("reports pending documents for the active model, not coverage by another model", async () => {
    const store = await open();
    const db = store.internal.db;
    const timestamp = "2026-01-01T00:00:00.000Z";
    insertContent(db, "fixture-hash", "# Synthetic document", timestamp);
    insertDocument(db, "notes", "note.md", "Synthetic", "fixture-hash", timestamp, timestamp);
    db.prepare(`INSERT INTO content_vectors
      (hash, seq, pos, model, embed_fingerprint, embedded_at, total_chunks)
      VALUES (?, 0, 0, ?, ?, ?, 1)`).run(
      "fixture-hash", "fixture:other-model", getEmbeddingFingerprint("fixture:other-model"), timestamp,
    );
    const health = metadata(await store.getHealthSnapshot());
    expect(health.pendingEmbeddings).toBe(1);
    expect(health.countModel).toBe(metadataIdentity("fixture:embedding-a"));
  });

  test("does not resurrect a removed collection from the old live DB catalog", async () => {
    const store = await open();
    writeFileSync(configPath, YAML.stringify(configuration("fixture:embedding-a", false)));
    const before = logicalState(store);
    const health = metadata(await store.getHealthSnapshot());
    expect(health.catalog.entries).toEqual([]);
    expect(health.current.configuration).not.toBe(health.active.configuration);
    expect(logicalState(store)).toBe(before);
  });

  test("reads its owned config without changing another store's global config source", async () => {
    const otherConfig = join(root, "other.yml");
    writeFileSync(otherConfig, YAML.stringify(configuration("fixture:embedding-b", false)));
    setConfigSource({ configPath: otherConfig });
    const first = await open("first");
    const second = await createStore({ dbPath: join(root, "second.sqlite"), configPath: otherConfig });
    stores.push(second);
    const [firstSnapshot, secondSnapshot] = await Promise.all([
      first.getHealthSnapshot(), second.getHealthSnapshot(),
    ]);
    expect(metadata(firstSnapshot).current.embeddingModel).toBe(metadataIdentity("fixture:embedding-a"));
    expect(metadata(secondSnapshot).current.embeddingModel).toBe(metadataIdentity("fixture:embedding-b"));
    expect(metadata(firstSnapshot).catalog.entries[0]?.documents).toBe(0);
    expect(getConfigPath()).toBe(otherConfig);
  });

  test("refuses an inherited environment change, missing config, or replaced index", async () => {
    const store = await open();
    process.env.QMD_EMBED_MODEL = "fixture:changed-environment";
    expect((await store.getHealthSnapshot()).health).toEqual({ schemaVersion: 1, unavailable: true });
    restoreEnvironment();
    process.env.QMD_CONFIG_DIR = root;
    process.env.INDEX_PATH = "";
    unlinkSync(configPath);
    expect((await store.getHealthSnapshot()).health).toEqual({ schemaVersion: 1, unavailable: true });
    writeFileSync(configPath, YAML.stringify(configuration()));
    renameSync(store.dbPath, join(root, "original.sqlite"));
    writeFileSync(store.dbPath, "different file");
    const health = metadata(await store.getHealthSnapshot());
    expect(health.active.index).not.toBe(health.current.index);
  });

  test("fingerprints the exact QMD_EMBED_CONTEXT_SIZE value and refuses drift", async () => {
    process.env.QMD_EMBED_CONTEXT_SIZE = "2048";
    const store = await open();
    const original = metadata(await store.getHealthSnapshot());
    expect(original.environment.names).toContain("QMD_EMBED_CONTEXT_SIZE");
    const sqlBefore = logicalState(store);
    const yamlBefore = readFileSync(configPath);
    for (const value of ["4096", "02048"]) {
      process.env.QMD_EMBED_CONTEXT_SIZE = value;
      expect((await store.getHealthSnapshot()).health).toEqual({ schemaVersion: 1, unavailable: true });
    }
    process.env.QMD_EMBED_CONTEXT_SIZE = "2048";
    expect(metadata(await store.getHealthSnapshot())).toEqual(original);
    expect(logicalState(store)).toBe(sqlBefore);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
  });

  test.each(["embed", "rerank"] as const)("binds module-captured %s context size when env changes before store creation", async kind => {
    const baseline = await open("before-context-change");
    const original = metadata(await baseline.getHealthSnapshot());
    const native = baseline.internal.llm!;
    const captured = kind === "embed" ? native.embedContextSize : native.rerankContextSize;
    const fresh = captured === 1024 ? 2048 : 1024;
    process.env[kind === "embed" ? "QMD_EMBED_CONTEXT_SIZE" : "QMD_RERANK_CONTEXT_SIZE"] = String(fresh);

    const store = await open("after-context-change");
    expect(store.internal.llm!.embedContextSize).toBe(native.embedContextSize);
    expect(store.internal.llm!.rerankContextSize).toBe(native.rerankContextSize);
    expect(kind === "embed" ? resolveEmbedContextSize() : resolveRerankContextSize()).toBe(fresh);
    const sqlBefore = logicalState(store);
    const yamlBefore = readFileSync(configPath);
    const health = metadata(await store.getHealthSnapshot());
    expect(health.environment.digest).toBe(environmentIdentity());
    expect(health.active.configuration).toBe(original.active.configuration);
    expect(health.active.configuration).not.toBe(health.current.configuration);
    expect(health.active.embeddingModel).toBe(health.current.embeddingModel);
    expect(logicalState(store)).toBe(sqlBefore);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
  });

  test.each([
    ["8192", 8192, 8192],
    ["1024suffix", 1024, 1024],
    ["0", 2048, 4096],
    ["-1", 2048, 4096],
    ["invalid", 2048, 4096],
    [undefined, 2048, 4096],
  ] as const)("preserves native context-size parsing for %s", (value, embed, rerank) => {
    for (const key of ["QMD_EMBED_CONTEXT_SIZE", "QMD_RERANK_CONTEXT_SIZE"]) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    expect(resolveEmbedContextSize()).toBe(embed);
    expect(resolveRerankContextSize()).toBe(rerank);
  });

  test("refuses legacy schema instead of performing lazy migration during health", async () => {
    const store = await open();
    useLegacyVectorSchema(store);
    const before = logicalState(store);
    await expect(store.getHealthSnapshot()).rejects.toThrow(/total_chunks/);
    expect(logicalState(store)).toBe(before);
  });

  test("equivalent context-map ordering retains the active binding without writes", async () => {
    const store = await open();
    const original = metadata(await store.getHealthSnapshot());
    const reordered = configuration();
    reordered.collections.notes!.context = Object.fromEntries(
      Object.entries(reordered.collections.notes!.context!).reverse(),
    );
    writeFileSync(configPath, YAML.stringify(reordered));
    const sqlBefore = logicalState(store);
    const yamlBefore = readFileSync(configPath);
    const snapshot = metadata(await store.getHealthSnapshot());
    expect(snapshot.active).toEqual(original.active);
    expect(snapshot.current).toEqual(snapshot.active);
    expect(logicalState(store)).toBe(sqlBefore);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
  });

  test.each(["model", "collection", "global"] as const)("rejects %s configuration drift during the SQLite snapshot", async field => {
    const store = await open();
    const db = store.internal.db;
    const prepare = db.prepare.bind(db);
    const sqlBefore = logicalState(store);
    const edited = configuration();
    if (field === "model") edited.models!.embed = "fixture:changed-mid-read";
    if (field === "collection") edited.collections.notes!.context = { "/": "Changed mid-read" };
    if (field === "global") edited.global_context = "Changed mid-read";
    const yamlAfter = YAML.stringify(edited);
    let changed = false;
    db.prepare = sql => {
      if (!changed && sql.includes("COUNT(DISTINCT d.hash)")) {
        changed = true;
        writeFileSync(configPath, yamlAfter);
      }
      return prepare(sql);
    };
    try {
      expect((await store.getHealthSnapshot()).health).toEqual({ schemaVersion: 1, unavailable: true });
    } finally { db.prepare = prepare; }
    expect(changed).toBe(true);
    expect(logicalState(store)).toBe(sqlBefore);
    expect(readFileSync(configPath, "utf8")).toBe(yamlAfter);
  });

  test("reads active collection and global contexts in the same SQLite snapshot", async () => {
    const store = await open();
    const owner = await createStore({ dbPath: store.dbPath });
    stores.push(owner);
    const original = metadata(await store.getHealthSnapshot());
    const edited = configuration();
    edited.global_context = "Refreshed global context";
    edited.collections.notes!.context = { "/": "Refreshed collection context" };
    const yamlBefore = readFileSync(configPath);
    const db = store.internal.db;
    const prepare = db.prepare.bind(db);
    let refreshed = false;
    db.prepare = sql => {
      if (!refreshed && sql.includes("COUNT(DISTINCT d.hash)")) {
        refreshed = true;
        owner.internal.db.transaction(() => syncConfigToDb(owner.internal.db, edited))();
      }
      return prepare(sql);
    };
    try {
      const snapshot = metadata(await store.getHealthSnapshot());
      expect(snapshot.active).toEqual(original.active);
      expect(snapshot.current).toEqual(original.current);
    } finally { db.prepare = prepare; }
    expect(refreshed).toBe(true);
    const sqlAfterRefresh = logicalState(store);
    const next = metadata(await store.getHealthSnapshot());
    expect(next.active.configuration).not.toBe(original.active.configuration);
    expect(next.current).toEqual(original.current);
    expect(logicalState(store)).toBe(sqlAfterRefresh);
    expect(readFileSync(configPath)).toEqual(yamlBefore);
  });

  test.each(["current", "legacy"] as const)("actual MCP initialize, tools/list, and status are model-free and read-only with %s schema", async schema => {
    process.env.INDEX_PATH = join(root, "index.sqlite");
    // MCP uses an inline, trust-checked snapshot of its file and must still
    // compare with that file on later reads, without enabling write-through.
    const store = await createStore({
      dbPath: process.env.INDEX_PATH,
      config: configuration(),
      configSourcePath: configPath,
    });
    stores.push(store);
    const db = store.internal.db;
    const timestamp = "2026-01-01T00:00:00.000Z";
    insertContent(db, "fixture-hash", "# Synthetic document", timestamp);
    insertDocument(db, "notes", "note.md", "Synthetic", "fixture-hash", timestamp, timestamp);
    if (schema === "legacy") useLegacyVectorSchema(store);
    const sqlBefore = logicalState(store);
    let yamlBefore = readFileSync(configPath);
    const oldStatus = vi.spyOn(store.internal, "getStatus").mockImplementation(() => {
      throw new Error("MCP health must not use the migration-capable status reader");
    });
    const network = vi.fn(async () => { throw new Error("No network in the native MCP fixture"); });
    globalThis.fetch = network;
    const [client, transport] = InMemoryTransport.createLinkedPair();
    let server: McpServer | undefined;
    let id = 0;
    function request(method: string, params: JSONRPCRequest["params"] = {}): Promise<JSONRPCResultResponse["result"]> {
      const requestId = ++id;
      const { promise, resolve, reject } = Promise.withResolvers<JSONRPCResultResponse["result"]>();
      client.onerror = reject;
      client.onmessage = message => {
        if (!("id" in message) || message.id !== requestId) return;
        if ("error" in message) reject(new Error(message.error.message));
        else if ("result" in message) resolve(message.result);
      };
      void client.send({ jsonrpc: "2.0", id: requestId, method, params }).catch(reject);
      return promise;
    }

    db.exec("PRAGMA query_only = ON");
    try {
      server = await createMcpServer(store, undefined, "index");
      await server.connect(transport);
      await client.start();
      const initialized = await request("initialize", {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "qmd-health-fixture", version: "1.0.0" },
      }) as InitializeResult;
      expect(initialized.serverInfo.name).toBe("qmd");
      expect(initialized.capabilities.experimental?.["qmd/healthMetadata"]).toEqual({ schemaVersion: 1 });
      if (schema === "legacy") {
        expect(initialized.instructions).toContain("Index health metadata is unavailable");
        expect(initialized.instructions).not.toContain("markdown documents");
      } else {
        expect(initialized.instructions).toContain("QMD is your local search engine over 1 markdown documents.");
        expect(initialized.instructions).toContain("Context: Fixture global context");
        expect(initialized.instructions).toContain("Collections (scope with `collections` parameter): notes");
      }
      expect(logicalState(store)).toBe(sqlBefore);
      expect(readFileSync(configPath)).toEqual(yamlBefore);
      expect(modelCalls).toBe(0);
      await client.send({ jsonrpc: "2.0", method: "notifications/initialized" });

      const listed = await request("tools/list") as ListToolsResult;
      expect(listed.tools.map(tool => tool.name)).toEqual(["query", "get", "multi_get", "status", "metadata"]);
      expect(listed.tools.find(tool => tool.name === "status")?._meta?.["qmd/healthMetadata"]).toBe(1);
      const result = await request("tools/call", { name: "status", arguments: {} }) as CallToolResult;
      if (schema === "legacy") {
        expect(result.isError).toBe(true);
        expect(result.structuredContent).toBeUndefined();
        expect(logicalState(store)).toBe(sqlBefore);
        expect(readFileSync(configPath)).toEqual(yamlBefore);
        expect(oldStatus).not.toHaveBeenCalled();
        expect(network).not.toHaveBeenCalled();
        expect(modelCalls).toBe(0);
        return;
      }
      expect(result.isError).not.toBe(true);
      expect(result.content).toEqual([{ type: "text", text: [
        "QMD Index Status:",
        "  Total documents: 1",
        "  Needs embedding: 1",
        "  Vector index: no",
        "  Collections: 1",
        `    - notes: ${docs} (1 docs)`,
      ].join("\n") }]);
      const snapshot = result.structuredContent as HealthStatus;
      expect(snapshot).toMatchObject({
        totalDocuments: 1, needsEmbedding: 1, hasVectorIndex: false,
        collections: [{ name: "notes", path: docs, pattern: "**/*.md", documents: 1, lastUpdated: timestamp }],
      });
      const original = metadata(snapshot);
      expect(original.active).toEqual(original.current);
      expect(original.pendingEmbeddings).toBe(1);
      expect(logicalState(store)).toBe(sqlBefore);
      expect(readFileSync(configPath)).toEqual(yamlBefore);

      const edited = configuration();
      edited.global_context = "Changed global context";
      edited.collections.notes!.context = { "/": "Changed collection context" };
      writeFileSync(configPath, YAML.stringify(edited));
      yamlBefore = readFileSync(configPath);
      const changed = await request("tools/call", { name: "status", arguments: {} }) as CallToolResult;
      expect(changed.isError).not.toBe(true);
      expect(changed.content).toEqual(result.content);
      const stale = metadata(changed.structuredContent as HealthStatus);
      expect(stale.active).toEqual(original.active);
      expect(stale.current.configuration).not.toBe(stale.active.configuration);
      expect(logicalState(store)).toBe(sqlBefore);
      expect(readFileSync(configPath)).toEqual(yamlBefore);
      expect(oldStatus).not.toHaveBeenCalled();
      expect(network).not.toHaveBeenCalled();
      expect(modelCalls).toBe(0);
    } finally {
      oldStatus.mockRestore();
      await server?.close();
      await client.close();
      db.exec("PRAGMA query_only = OFF");
    }
  });

  test("native selection is pure and notices a newly selected project-local index", () => {
    const priorCwd = process.cwd();
    try {
      process.chdir(docs);
      const cache = join(root, "uncreated-cache");
      process.env.XDG_CACHE_HOME = cache;
      const global = resolveIndexContext("fixture");
      expect(global.dbPath).toContain("fixture.sqlite");
      mkdirSync(join(docs, ".qmd"));
      writeFileSync(join(docs, ".qmd", "index.yml"), "collections: {}\n");
      const local = resolveIndexContext();
      expect(local.local).toBe(true);
      expect(local.dbPath).toBe(join(docs, ".qmd", "index.sqlite"));
      expect(global.configPath).toBe(join(root, "fixture.yml"));
      expect(existsSync(cache)).toBe(false);
    } finally { process.chdir(priorCwd); }
  });
});
