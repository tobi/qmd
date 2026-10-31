/**
 * metadata-surfaces.test.ts - Metadata filter support across the public
 * surfaces: SDK, MCP tool, and HTTP REST endpoints. Uses lex-only searches
 * with reranking disabled so no models are needed.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import {
  createStore,
  MetadataBindingBudgetError,
  MetadataOptionError,
  type MetadataFilter,
  type MetadataMatch,
  type QMDStore,
} from "../src/index.js";
import {
  createStore as createInternalStore,
  insertContent,
  insertDocument,
  hashContent,
  syncConfigToDb,
  _resetProductionModeForTesting,
} from "../src/store.js";
import { replaceDocumentMetadata } from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.js";
import type { CollectionConfig } from "../src/collections.js";

let testDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-surfaces-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

function buildDoc(status: string, body: string): string {
  return `---\nqmd:\n  metadata:\n    status: ${status}\n    topics: [typescript]\n---\n\n${body}`;
}

/** A filter and a match that each pass the parser's limits and together exceed the SQL binding budget. */
function buildWidePredicates(): { filter: MetadataFilter; match: MetadataMatch } {
  const members = Array.from({ length: 64 }, (_, index) => `v${index}`);
  const groups = <Operand,>(operand: () => Operand) => Array.from({ length: 7 }, () => Array.from({ length: 31 }, operand));
  return {
    filter: { operator: "and", operands: groups(() => ({ field: "status", operator: "all" as const, value: members })).map(operands => ({ operator: "and" as const, operands })) },
    match: { operator: "or", operands: groups(() => ({ field: "value" as const, operator: "in" as const, value: members })).map(operands => ({ operator: "or" as const, operands })) },
  };
}

// =============================================================================
// SDK
// =============================================================================

describe("SDK metadata filter", () => {
  let store: QMDStore;
  let collectionDir: string;

  beforeAll(async () => {
    collectionDir = join(testDir, "sdk-collection");
    await mkdir(collectionDir, { recursive: true });
    await writeFile(join(collectionDir, "published.md"), buildDoc("published", "# Pub\n\nsdk keyword body"));
    await writeFile(join(collectionDir, "draft.md"), buildDoc("draft", "# Draft\n\nsdk keyword body"));

    store = await createStore({
      dbPath: join(testDir, "sdk.sqlite"),
      config: { collections: { docs: { path: collectionDir, pattern: "**/*.md" } } },
    });
    await store.update();
  });

  afterAll(async () => {
    await store.close();
  });

  test("searchLex applies the filter and returns metadata", async () => {
    const unfiltered = await store.searchLex("sdk keyword");
    expect(unfiltered.length).toBe(2);

    const filtered = await store.searchLex("sdk keyword", {
      filter: { field: "status", operator: "eq", value: "published" },
    });
    expect(filtered.map(r => r.displayPath)).toEqual(["docs/published.md"]);
    expect(filtered[0]!.metadata).toEqual({ status: "published", topics: ["typescript"] });
  });

  test("search with pre-expanded queries applies the filter", async () => {
    const results = await store.search({
      queries: [{ type: "lex", query: "sdk keyword" }],
      filter: { field: "status", operator: "ne", value: "draft" },
      rerank: false,
    });
    expect(results.map(r => r.displayPath)).toEqual(["docs/published.md"]);
    expect(results[0]!.metadata).toEqual({ status: "published", topics: ["typescript"] });
  });

  test("getStatus exposes the pending metadata count", async () => {
    const status = await store.getStatus();
    expect(status.pendingMetadata).toBe(0);
  });

  test("validates filters at the SDK runtime boundary", async () => {
    await expect(store.searchLex("sdk keyword", {
      filter: { operator: "and", operands: [] },
    })).rejects.toThrow(/non-empty 'operands'/);
  });

  test("listMetadata summarizes keys, types, and value counts", async () => {
    const result = await store.listMetadata();
    expect(result.documents).toBe(2);
    expect(result.filteredDocuments).toBeUndefined();
    expect(result.keys.map(summary => summary.key)).toEqual(["status", "topics"]);

    const topics = result.keys[1]!.types[0]!;
    expect(topics).toMatchObject({ type: "string", documents: 2, distinctValues: 1, remainingValues: 0, collections: ["docs"] });
    expect(result).toMatchObject({ totalKeys: 2, remainingKeys: 0 });
    expect(topics.values).toEqual([{ value: "typescript", documents: 2 }]);
  });

  test("listMetadata scopes, narrows by filter, and selects entries by match", async () => {
    const filtered = await store.listMetadata({
      collection: "docs",
      match: { field: "key", operator: "eq", value: "status" },
      filter: { field: "status", operator: "eq", value: "published" },
    });
    expect(filtered.filteredDocuments).toBe(1);
    expect(filtered.keys[0]!.types[0]!.values).toEqual([{ value: "published", documents: 1 }]);

    const reverse = await store.listMetadata({ match: { field: "value", operator: "prefix", value: "dra" } });
    expect(reverse.keys.map(summary => summary.key)).toEqual(["status"]);

    const scoped = await store.listMetadata({ collection: "missing" });
    expect(scoped).toEqual({ documents: 0, totalKeys: 0, keys: [], remainingKeys: 0 });
  });

  test("listMetadata windows keys and values and rejects options outside their domain", async () => {
    const page = await store.listMetadata({ keyLimit: 1, keyOffset: 1 });
    expect(page.keys.map(summary => summary.key)).toEqual(["topics"]);
    expect(page).toMatchObject({ totalKeys: 2, remainingKeys: 0 });

    await expect(store.listMetadata({ keyLimit: 0 })).rejects.toThrow(/^Invalid keyLimit: expected a positive integer or Infinity, received 0$/);
    await expect(store.listMetadata({ valueOffset: -1 })).rejects.toThrow(MetadataOptionError);
    await expect(store.listMetadata({ keyLimit: 1e30 })).rejects.toThrow(MetadataOptionError);
    await expect(store.listMetadata(buildWidePredicates())).rejects.toThrow(MetadataBindingBudgetError);
  });

  test("listMetadata validates filters and matches at the SDK runtime boundary", async () => {
    // Plain-JS callers bypass the declarations, so the SDK must reject a bad AST at runtime.
    const untrustedFilter: import("../src/index.js").ListMetadataOptions = JSON.parse('{"filter":{"field":"status","operator":"equal","value":"x"}}');
    await expect(store.listMetadata(untrustedFilter)).rejects.toThrow(/unknown operator 'equal'/);

    const untrustedMatch: import("../src/index.js").ListMetadataOptions = JSON.parse('{"match":{"field":"status","operator":"eq","value":"x"}}');
    await expect(store.listMetadata(untrustedMatch)).rejects.toThrow(/Invalid metadata match at \$: 'status' is not a field of a metadata entry/);
  });

  test("negated text matches preserve extracted empty strings through the SDK", async () => {
    const document = store.internal.db.prepare("SELECT id FROM documents WHERE path = 'published.md'").get() as { id: number };
    replaceDocumentMetadata(store.internal.db, document.id, { metadata: { status: "", topics: ["typescript"] }, extractionVersion: METADATA_EXTRACTION_VERSION });
    try {
      for (const operator of ["prefix", "suffix"] as const) {
        const result = await store.listMetadata({ match: { operator: "and", operands: [
          { field: "key", operator: "eq", value: "status" },
          { operator: "not", operand: { field: "value", operator, value: "zzz" } },
        ] } });
        expect(result.keys[0]!.documents).toBe(2);
        expect(result.keys[0]!.types[0]!.values).toEqual([{ value: "", documents: 1 }, { value: "draft", documents: 1 }]);
      }
    } finally {
      replaceDocumentMetadata(store.internal.db, document.id, { metadata: { status: "published", topics: ["typescript"] }, extractionVersion: METADATA_EXTRACTION_VERSION });
    }
  });

  test("getStatus lists metadata keys per collection", async () => {
    const status = await store.getStatus();
    expect(status.collections[0]!.metadataKeyCount).toBe(2);
    expect(status.collections[0]!.metadataKeys).toEqual([
      { key: "status", documents: 2, types: ["string"] },
      { key: "topics", documents: 2, types: ["string"] },
    ]);
  });
});

// =============================================================================
// MCP tool + HTTP REST
// =============================================================================

describe("MCP and HTTP metadata filter", () => {
  let handle: HttpServerHandle;
  let baseUrl: string;
  let dbPath: string;
  let configDir: string;
  const origIndexPath = process.env.INDEX_PATH;
  const origConfigDir = process.env.QMD_CONFIG_DIR;

  async function seedDoc(db: import("../src/db.js").Database, path: string, body: string, metadata: DocumentMetadata): Promise<void> {
    const now = new Date().toISOString();
    const hash = await hashContent(body);
    insertContent(db, hash, body, now);
    const documentId = insertDocument(db, "docs", path, path, hash, now, now);
    replaceDocumentMetadata(db, documentId, { metadata, extractionVersion: METADATA_EXTRACTION_VERSION });
  }

  beforeAll(async () => {
    dbPath = join(testDir, `mcp-${Date.now()}.sqlite`);
    const internal = createInternalStore(dbPath);
    await seedDoc(internal.db, "published.md", "# Pub\n\nhttp keyword body", { status: "published" });
    await seedDoc(internal.db, "draft.md", "# Draft\n\nhttp keyword body", { status: "draft" });
    await seedDoc(internal.db, "tagged.md", "# Tagged\n\nhttp keyword body", { status: "archived", topics: ["sqlite", "search"], priority: 3 });

    const testConfig: CollectionConfig = {
      collections: { docs: { path: "/test/docs", pattern: "**/*.md" } },
    };
    syncConfigToDb(internal.db, testConfig);
    internal.close();

    configDir = await mkdtemp(join(tmpdir(), "qmd-metadata-surfaces-config-"));
    await writeFile(join(configDir, "index.yml"), YAML.stringify(testConfig));

    process.env.INDEX_PATH = dbPath;
    process.env.QMD_CONFIG_DIR = configDir;
    handle = await startMcpHttpServer(0, { quiet: true, dbPath });
    baseUrl = `http://localhost:${handle.port}`;
  });

  afterAll(async () => {
    if (handle) await handle.stop();
    _resetProductionModeForTesting();
    if (origIndexPath !== undefined) process.env.INDEX_PATH = origIndexPath;
    else delete process.env.INDEX_PATH;
    if (origConfigDir !== undefined) process.env.QMD_CONFIG_DIR = origConfigDir;
    else delete process.env.QMD_CONFIG_DIR;
    try { unlinkSync(dbPath); } catch {}
    await rm(configDir, { recursive: true, force: true });
  });

  type HttpRequestBody = Record<string, unknown>;

  async function postJson(path: string, body: HttpRequestBody): Promise<{ status: number; json: any }> {
    const res = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return { status: res.status, json: await res.json() };
  }

  async function callQueryTool(args: Record<string, unknown>): Promise<{ status: number; json: any }> {
    return callTool("query", args);
  }

  async function callTool(name: string, args: Record<string, unknown>): Promise<{ status: number; json: any }> {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "tools/call",
        "Mcp-Name": name,
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: {
          name,
          arguments: args,
          _meta: {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": { name: "metadata-test", version: "1.0.0" },
            "io.modelcontextprotocol/clientCapabilities": {},
          },
        },
      }),
    });
    return { status: res.status, json: await res.json() };
  }

  test("MCP initialization lists tools without reading metadata value rows", async () => {
    const internal = createInternalStore(dbPath);
    // Make value-table reads fail deterministically instead of testing a noisy
    // timing threshold. The running server already has its database connection.
    internal.db.exec("ALTER TABLE document_metadata_values RENAME TO hidden_metadata_values");
    try {
      const response = await fetch(`${baseUrl}/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: { _meta: {
          "io.modelcontextprotocol/protocolVersion": "2026-07-28",
          "io.modelcontextprotocol/clientInfo": { name: "metadata-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        } } }),
      });
      expect(response.status).toBe(200);
      const body = await response.json() as { result: { tools: { name: string }[] } };
      expect(body.result.tools.map(tool => tool.name)).toContain("metadata");
    } finally {
      internal.db.exec("ALTER TABLE hidden_metadata_values RENAME TO document_metadata_values");
      internal.close();
    }
  });

  test("HTTP and MCP negated text matches include empty strings", async () => {
    const internal = createInternalStore(dbPath);
    const document = internal.db.prepare("SELECT id FROM documents WHERE path = 'draft.md'").get() as { id: number };
    replaceDocumentMetadata(internal.db, document.id, { metadata: { status: "" }, extractionVersion: METADATA_EXTRACTION_VERSION });
    try {
      for (const operator of ["prefix", "suffix"] as const) {
        const args = { match: { operator: "and", operands: [
          { field: "key", operator: "eq", value: "status" },
          { operator: "not", operand: { field: "value", operator, value: "zzz" } },
        ] } };
        const http = await postJson("/metadata", args);
        const mcp = await callTool("metadata", args);
        expect(http.status).toBe(200);
        expect(mcp.status).toBe(200);
        expect(mcp.json.result.structuredContent).toEqual(http.json);
        expect(http.json.keys[0].documents).toBe(3);
        expect(http.json.keys[0].types[0].values).toContainEqual({ value: "", documents: 1 });
        expect(mcp.json.result.content[0].text).toContain('""');
      }
    } finally {
      replaceDocumentMetadata(internal.db, document.id, { metadata: { status: "draft" }, extractionVersion: METADATA_EXTRACTION_VERSION });
      internal.close();
    }
  });

  test("POST /query applies the filter and includes metadata", async () => {
    const { status, json } = await postJson("/query", {
      searches: [{ type: "lex", query: "http keyword" }],
      filter: { field: "status", operator: "eq", value: "published" },
      rerank: false,
    });
    expect(status).toBe(200);
    expect(json.results.length).toBe(1);
    expect(json.results[0].file).toBe("qmd://docs/published.md");
    expect(json.results[0].metadata).toEqual({ status: "published" });
  });

  test("POST /search alias accepts the same filter", async () => {
    const { status, json } = await postJson("/search", {
      searches: [{ type: "lex", query: "http keyword" }],
      filter: { field: "status", operator: "eq", value: "draft" },
      rerank: false,
    });
    expect(status).toBe(200);
    expect(json.results.length).toBe(1);
    expect(json.results[0].file).toBe("qmd://docs/draft.md");
  });

  test("POST /query rejects non-object and invalid filters with 400", async () => {
    const stringFilter = await postJson("/query", {
      searches: [{ type: "lex", query: "http keyword" }],
      filter: "status = published",
    });
    expect(stringFilter.status).toBe(400);
    expect(stringFilter.json.error).toMatch(/must be an object/);

    const invalidAst = await postJson("/query", {
      searches: [{ type: "lex", query: "http keyword" }],
      filter: { field: "status", operator: "equal", value: "published" },
    });
    expect(invalidAst.status).toBe(400);
    expect(invalidAst.json.error).toMatch(/unknown operator 'equal'/);
  });

  test("MCP query tool applies a nested filter and includes metadata", async () => {
    const { status, json } = await callQueryTool({
      searches: [{ type: "lex", query: "http keyword" }],
      filter: {
        operator: "and",
        operands: [
          { field: "status", operator: "eq", value: "published" },
          { operator: "not", operand: { field: "status", operator: "eq", value: "draft" } },
        ],
      },
      rerank: false,
    });
    expect(status).toBe(200);
    expect(json.result.isError).toBeFalsy();
    const items = json.result.structuredContent.results;
    expect(items.length).toBe(1);
    expect(items[0].file).toBe("docs/published.md");
    expect(items[0].metadata).toEqual({ status: "published" });
  });

  test("MCP query tool rejects invalid filters", async () => {
    const { status, json } = await callQueryTool({
      searches: [{ type: "lex", query: "http keyword" }],
      filter: { operator: "and", operands: [] },
      rerank: false,
    });
    expect(status).toBe(200);
    expect(json.result.isError).toBe(true);
    expect(json.result.content[0].text).toMatch(/non-empty 'operands'/);
  });

  test("MCP metadata tool returns key summaries as structured content and CLI-shaped text", async () => {
    const { status, json } = await callTool("metadata", { match: { field: "key", operator: "eq", value: "status" } });
    expect(status).toBe(200);
    expect(json.result.isError).toBeFalsy();

    const result = json.result.structuredContent;
    expect(result.documents).toBe(3);
    expect(result.keys).toHaveLength(1);
    expect(result.keys[0].types[0]).toMatchObject({ type: "string", documents: 3, distinctValues: 3, remainingValues: 0 });
    expect(result.keys[0].types[0].values).toEqual([
      { value: "archived", documents: 1 },
      { value: "draft", documents: 1 },
      { value: "published", documents: 1 },
    ]);
    expect(json.result.content[0].text).toBe("status  string  3 of 3 documents  3 distinct\n  archived   1\n  draft      1\n  published  1");
  });

  test("MCP metadata tool narrows by filter, windows values, and reports the remainder", async () => {
    const { json } = await callTool("metadata", {
      match: { field: "key", operator: "eq", value: "topics" },
      valueLimit: 1,
      filter: { field: "status", operator: "eq", value: "archived" },
    });
    const result = json.result.structuredContent;
    expect(result.filteredDocuments).toBe(1);
    const topics = result.keys[0].types[0];
    expect(topics.multiValued).toBe(true);
    expect(topics.values).toHaveLength(1);
    expect(topics.remainingValues).toBe(1);
    expect(json.result.content[0].text).toContain("filter: 1 of 3 documents\n\ntopics  string[]  1 of 1 documents  2 distinct");
    expect(json.result.content[0].text).toContain("1 more value, use a higher 'valueLimit' or a 'valueOffset'");
  });

  test("MCP metadata tool windows and pages keys", async () => {
    const firstPage = await callTool("metadata", { keyLimit: 2 });
    const first = firstPage.json.result.structuredContent;
    expect(first.keys.map((summary: { key: string }) => summary.key)).toEqual(["status", "priority"]);
    expect(first).toMatchObject({ totalKeys: 3, remainingKeys: 1 });
    expect(firstPage.json.result.content[0].text).toContain("1 more key, use a higher 'keyLimit' or a 'keyOffset'");

    const secondPage = await callTool("metadata", { keyLimit: 2, keyOffset: 2 });
    expect(secondPage.json.result.structuredContent.keys.map((summary: { key: string }) => summary.key)).toEqual(["topics"]);
    expect(secondPage.json.result.structuredContent.remainingKeys).toBe(0);

    const pastTheEnd = await callTool("metadata", { keyOffset: 3 });
    expect(pastTheEnd.json.result.isError).toBeFalsy();
    expect(pastTheEnd.json.result.content[0].text).toBe("No keys at keyOffset 3, 3 keys in total.");

    const badOffset = await callTool("metadata", { keyOffset: -1 });
    expect(badOffset.json.error ?? badOffset.json.result?.isError).toBeTruthy();

    const unsafeLimit = await callTool("metadata", { keyLimit: 1e30 });
    expect(unsafeLimit.json.error ?? unsafeLimit.json.result?.isError).toBeTruthy();
  });

  test("MCP metadata tool keeps the filter line over an empty result and reports the binding budget", async () => {
    // Every seeded document declares a status, so this filter admits none.
    const empty = await callTool("metadata", { filter: { field: "status", operator: "exists", value: false } });
    expect(empty.json.result.isError).toBeFalsy();
    expect(empty.json.result.structuredContent).toMatchObject({ documents: 3, filteredDocuments: 0, totalKeys: 0 });
    expect(empty.json.result.content[0].text).toBe(
      "filter: 0 of 3 documents\n\nNo metadata matches. Call without match/filter to see which keys exist, or check the status tool for collections with metadata.",
    );

    const overBudget = await callTool("metadata", buildWidePredicates());
    expect(overBudget.json.result.isError).toBe(true);
    expect(overBudget.json.result.content[0].text).toMatch(/^Error: filter and match together bind \d+ SQL parameters, over the budget of 30000\./);
  });

  test("MCP metadata tool rejects invalid filters and matches", async () => {
    const { json } = await callTool("metadata", { filter: { field: "status", operator: "equal", value: "x" } });
    expect(json.result.isError).toBe(true);
    expect(json.result.content[0].text).toMatch(/unknown operator 'equal'/);

    const badMatch = await callTool("metadata", { match: { field: "value", operator: "all", value: ["x"] } });
    expect(badMatch.json.result.isError).toBe(true);
    expect(badMatch.json.result.content[0].text).toMatch(/Invalid metadata match at \$: 'all' has no meaning for a single metadata entry/);
  });

  test("MCP status tool lists metadata keys per collection", async () => {
    const { json } = await callTool("status", {});
    expect(json.result.isError).toBeFalsy();
    expect(json.result.structuredContent.collections[0].metadataKeyCount).toBe(3);
    expect(json.result.structuredContent.collections[0].metadataKeys).toEqual([
      { key: "status", documents: 3, types: ["string"] },
      { key: "priority", documents: 1, types: ["number"] },
      { key: "topics", documents: 1, types: ["string"] },
    ]);
    expect(json.result.content[0].text).toContain("metadata keys: status (string), priority (number), topics (string)");
    expect(json.result.content[0].text).toContain("call the 'metadata' tool");
  });

  test("POST /metadata returns the same result as the tool", async () => {
    const { status, json } = await postJson("/metadata", { match: { field: "value", operator: "type", value: "number" } });
    expect(status).toBe(200);
    expect(json.documents).toBe(3);
    expect(json.keys.map((summary: { key: string }) => summary.key)).toEqual(["priority"]);
    expect(json.keys[0].types[0]).toMatchObject({ type: "number", range: { min: 3, median: 3, max: 3 } });

    const reverse = await postJson("/metadata", { match: { field: "value", operator: "eq", value: "search" }, valueLimit: 5 });
    expect(reverse.json.keys.map((summary: { key: string }) => summary.key)).toEqual(["topics"]);

    const page = await postJson("/metadata", { keyLimit: 1, keyOffset: 1 });
    expect(page.json.keys.map((summary: { key: string }) => summary.key)).toEqual(["priority"]);
    expect(page.json).toMatchObject({ totalKeys: 3, remainingKeys: 1 });
  });

  test("POST /metadata rejects invalid bodies, filters, matches, and sort with 400", async () => {
    const stringFilter = await postJson("/metadata", { filter: "status = published" });
    expect(stringFilter.status).toBe(400);
    expect(stringFilter.json.error).toMatch(/must be an object/);

    const stringMatch = await postJson("/metadata", { match: "key = topics" });
    expect(stringMatch.status).toBe(400);
    expect(stringMatch.json.error).toMatch(/Invalid field: match \(must be an object\)/);

    const invalidMatch = await postJson("/metadata", { match: { field: "topics", operator: "eq", value: "x" } });
    expect(invalidMatch.status).toBe(400);
    expect(invalidMatch.json.error).toMatch(/'topics' is not a field of a metadata entry/);

    const invalidAst = await postJson("/metadata", { filter: { field: "status", operator: "equal", value: "x" } });
    expect(invalidAst.status).toBe(400);
    expect(invalidAst.json.error).toMatch(/unknown operator 'equal'/);

    const badSort = await postJson("/metadata", { sort: "size" });
    expect(badSort.status).toBe(400);
    expect(badSort.json.error).toMatch(/sort/);

    const stringLimit = await postJson("/metadata", { keyLimit: "5" });
    expect(stringLimit.status).toBe(400);
    expect(stringLimit.json.error).toBe("Invalid field: keyLimit (must be a number)");

    const zeroLimit = await postJson("/metadata", { valueLimit: 0 });
    expect(zeroLimit.status).toBe(400);
    expect(zeroLimit.json.error).toBe("Invalid valueLimit: expected a positive integer or Infinity, received 0");

    const fractionalOffset = await postJson("/metadata", { keyOffset: 1.5 });
    expect(fractionalOffset.status).toBe(400);
    expect(fractionalOffset.json.error).toBe("Invalid keyOffset: expected a non-negative integer, received 1.5");

    const stringCollections = await postJson("/metadata", { collections: "docs" });
    expect(stringCollections.status).toBe(400);
    expect(stringCollections.json.error).toBe("Invalid field: collections (must be an array)");

    const unsafeLimit = await postJson("/metadata", { keyLimit: 1e30 });
    expect(unsafeLimit.status).toBe(400);
    expect(unsafeLimit.json.error).toBe("Invalid keyLimit: expected a positive integer or Infinity, received 1e+30");

    const overBudget = await postJson("/metadata", buildWidePredicates());
    expect(overBudget.status).toBe(400);
    expect(overBudget.json.error).toMatch(/^filter and match together bind \d+ SQL parameters, over the budget of 30000\./);

    const arrayBody = await fetch(`${baseUrl}/metadata`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "[]" });
    expect(arrayBody.status).toBe(400);
  });
});
