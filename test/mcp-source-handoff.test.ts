import { afterEach, beforeEach, describe, expect, test, vi, type MockInstance } from "vitest";
import { createMcpHandler, type McpHttpHandler } from "@modelcontextprotocol/server";
import { createStore, type QMDStore } from "../src/index.js";
import { createMcpServer, type McpServerOptions } from "../src/mcp/server.js";

const sourceRoot = "/tmp/qmd-source-handoff";
let store: QMDStore;
let handler: McpHttpHandler;
let searchSpy: MockInstance<QMDStore["search"]>;

function result(displayPath = "docs/_meta/source_path.md") {
  return {
    file: "qmd://" + displayPath, displayPath, title: "Source",
    body: "source handoff", bestChunk: "source handoff", bestChunkPos: 0,
    score: 0.8, context: null, docid: "abcdef", metadata: { status: "published" },
  };
}

beforeEach(async () => {
  store = await createStore({
    dbPath: ":memory:",
    config: { collections: {
      docs: { path: sourceRoot + "/notes", pattern: "**/*.md" },
      outside: { path: "/tmp/unrelated", pattern: "**/*.md" },
    } },
  });
  searchSpy = vi.spyOn(store, "search").mockResolvedValue([result()]);
  handler = createMcpHandler(() => createMcpServer(store, undefined, {
    queryPolicy: "split-rerank", sourceRoot,
  }));
});

afterEach(async () => {
  await handler.close();
  await store.close();
});

async function query(name = "query", target = handler) {
  const response = await target.fetch(new Request("http://localhost/mcp", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
      "MCP-Protocol-Version": "2026-07-28",
      "Mcp-Method": "tools/call",
      "Mcp-Name": name,
    },
    body: JSON.stringify({
      jsonrpc: "2.0", id: 1, method: "tools/call",
      params: {
        name, arguments: { searches: [{ type: "lex", query: "source" }] },
        _meta: {
          "io.modelcontextprotocol/protocolVersion": "2026-07-28",
          "io.modelcontextprotocol/clientInfo": { name: "source-handoff-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }),
  }));
  expect(response.status, await response.clone().text()).toBe(200);
  const payload = await response.json();
  expect(payload.error).toBeUndefined();
  expect(payload.result.isError).not.toBe(true);
  return payload.result;
}

describe("MCP configured source handoff", () => {
  test.each(["query", "query_reranked"])("preserves exact source spelling for %s", async (name) => {
    const output = await query(name);
    expect(output.structuredContent.results[0]).toMatchObject({
      file: "docs/_meta/source_path.md", collection: "docs",
      source_relative_path: "notes/_meta/source_path.md",
      metadata: { status: "published" },
    });
    expect(JSON.stringify(output)).not.toContain(sourceRoot);
  });

  test("preserves distinct original names instead of normalizing them", async () => {
    searchSpy.mockResolvedValue([result("docs/_meta/source_path.md"), result("docs/meta/source-path.md")]);
    const output = await query();
    expect(output.structuredContent.results.map((item: { source_relative_path: string }) => item.source_relative_path))
      .toEqual(["notes/_meta/source_path.md", "notes/meta/source-path.md"]);
  });

  test.each([
    "docs/../outside.md", "docs/..\\outside.md", "docs//absolute.md", "outside/file.md", "missing/file.md", "without-collection",
  ])("refuses an invalid or outside-root mapping for %s", async (path) => {
    searchSpy.mockResolvedValue([result(path)]);
    const output = await query();
    expect(output.structuredContent.results[0].source_relative_path).toBeNull();
    expect(JSON.stringify(output)).not.toContain(sourceRoot);
    expect(JSON.stringify(output)).not.toContain("/tmp/unrelated");
  });

  test("reads supported collection configuration changes on the next query", async () => {
    const initial = await query();
    expect(initial.structuredContent.results[0].source_relative_path).toBe("notes/_meta/source_path.md");
    await store.addCollection("docs", { path: sourceRoot + "/changed" });
    const changed = await query();
    expect(changed.structuredContent.results[0].source_relative_path).toBe("changed/_meta/source_path.md");
    await store.addCollection("docs", { path: "/tmp/outside-root" });
    const outside = await query();
    expect(outside.structuredContent.results[0].source_relative_path).toBeNull();
  });

  test("rejects a sibling whose name merely shares the root prefix", async () => {
    await store.addCollection("docs", { path: sourceRoot + "-other" });
    const output = await query();
    expect(output.structuredContent.results[0].source_relative_path).toBeNull();
  });

  test("keeps simultaneous source-root instances isolated", async () => {
    const other = createMcpHandler(() => createMcpServer(store, undefined, { sourceRoot: sourceRoot + "/notes" }));
    try {
      const [first, second] = await Promise.all([query(), query("query", other)]);
      expect(first.structuredContent.results[0].source_relative_path).toBe("notes/_meta/source_path.md");
      expect(second.structuredContent.results[0].source_relative_path).toBe("_meta/source_path.md");
    } finally {
      await other.close();
    }
  });

  test("captures sourceRoot at construction rather than reading mutated options", async () => {
    await handler.close();
    const options: McpServerOptions = { sourceRoot };
    handler = createMcpHandler(async () => {
      const server = await createMcpServer(store, undefined, options);
      options.sourceRoot = "/tmp/changed-after-construction";
      return server;
    });
    const output = await query();
    expect(options.sourceRoot).toBe("/tmp/changed-after-construction");
    expect(output.structuredContent.results[0].source_relative_path).toBe("notes/_meta/source_path.md");
  });

  test("keeps ordinary results unchanged without sourceRoot", async () => {
    await handler.close();
    handler = createMcpHandler(() => createMcpServer(store));
    const output = await query();
    expect(output.structuredContent.results[0]).not.toHaveProperty("collection");
    expect(output.structuredContent.results[0]).not.toHaveProperty("source_relative_path");
    expect(output.structuredContent.results[0].file).toBe("docs/_meta/source_path.md");
  });
});
