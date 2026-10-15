import { afterEach, beforeEach, describe, expect, test, vi, type MockInstance } from "vitest";
import { createMcpHandler, type McpHttpHandler } from "@modelcontextprotocol/server";
import { createStore, type QMDStore } from "../src/index.js";
import { createMcpServer, type McpServerOptions } from "../src/mcp/server.js";

let store: QMDStore;
let searchSpy: MockInstance<QMDStore["search"]>;
let handler: McpHttpHandler;
const filter = { field: "status", operator: "eq", value: "published" };

beforeEach(async () => {
  store = await createStore({
    dbPath: ":memory:",
    config: { collections: { docs: { path: "/tmp/query-policy-fixture", pattern: "**/*.md" } } },
  });
  searchSpy = vi.spyOn(store, "search").mockResolvedValue([]);
  handler = createMcpHandler(() => createMcpServer(store, undefined, { queryPolicy: "split-rerank" }));
});

afterEach(async () => {
  await handler.close();
  await store.close();
});

async function request(method: string, params: Record<string, unknown> = {}, target = handler) {
  const response = await target.fetch(new Request("http://localhost/mcp", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
      "MCP-Protocol-Version": "2026-07-28",
      "Mcp-Method": method,
      ...(typeof params.name === "string" ? { "Mcp-Name": params.name } : {}),
    },
    body: JSON.stringify({
      jsonrpc: "2.0", id: 1, method,
      params: {
        ...params,
        _meta: {
          "io.modelcontextprotocol/protocolVersion": "2026-07-28",
          "io.modelcontextprotocol/clientInfo": { name: "query-policy-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }),
  }));
  expect(response.status, await response.clone().text()).toBe(200);
  return response.json();
}

describe("MCP split rerank query policy", () => {
  test("publishes two distinct tools with bounded rerank schema", async () => {
    const { result } = await request("tools/list");
    const query = result.tools.find((tool: { name: string }) => tool.name === "query");
    const reranked = result.tools.find((tool: { name: string }) => tool.name === "query_reranked");
    expect(query.inputSchema.properties).not.toHaveProperty("rerank");
    expect(query.inputSchema.properties).not.toHaveProperty("candidateLimit");
    expect(reranked).toBeDefined();
    expect(reranked.inputSchema.properties).not.toHaveProperty("rerank");
    expect(reranked.inputSchema.properties.candidateLimit).toMatchObject({
      type: "integer", minimum: 1, maximum: 20, default: 10,
    });
    expect(query.annotations.readOnlyHint).toBe(true);
    expect(reranked.annotations.readOnlyHint).toBe(true);
  });

  test("routine plain query cannot opt into reranking through extra input", async () => {
    const { result } = await request("tools/call", {
      name: "query", arguments: { query: "pool timeout", rerank: true, candidateLimit: 100, filter, intent: "databases" },
    });
    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toEqual({ results: [] });
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith({
      query: "pool timeout", collections: ["docs"], filter, limit: 10, minScore: 0,
      candidateLimit: undefined, rerank: false, intent: "databases",
    });
  });

  test("reranked typed query forces reranking and defaults to ten candidates", async () => {
    const { result } = await request("tools/call", {
      name: "query_reranked",
      arguments: { searches: [{ type: "lex", query: "pool" }], rerank: false, collections: ["other"], filter },
    });
    expect(result.isError).not.toBe(true);
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith({
      queries: [{ type: "lex", query: "pool" }], collections: ["other"], filter,
      limit: 10, minScore: 0, candidateLimit: 10, rerank: true, intent: undefined,
    });
  });

  test.each([1, 20])("accepts candidate limit boundary %s", async (candidateLimit) => {
    const { result } = await request("tools/call", {
      name: "query_reranked", arguments: { query: "pool", candidateLimit },
    });
    expect(result.isError).not.toBe(true);
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ rerank: true, candidateLimit }));
  });

  test.each([0, 21, 1.5])("rejects invalid candidate limit %s before search", async (candidateLimit) => {
    const response = await request("tools/call", {
      name: "query_reranked", arguments: { query: "pool", candidateLimit },
    });
    expect(response.error || response.result?.isError).toBeTruthy();
    expect(store.search).not.toHaveBeenCalled();
  });

  test("preserves mutually exclusive query and metadata validation", async () => {
    for (const args of [
      { query: "pool", searches: [{ type: "lex", query: "pool" }] },
      { query: "pool", filter: { operator: "unknown" } },
      {},
    ]) {
      const response = await request("tools/call", { name: "query_reranked", arguments: args });
      expect(response.error || response.result?.isError).toBeTruthy();
    }
    expect(store.search).not.toHaveBeenCalled();
  });

  test("keeps simultaneous standard and split server policies isolated", async () => {
    const standard = createMcpHandler(() => createMcpServer(store));
    try {
      await Promise.all([
        request("tools/call", { name: "query", arguments: { query: "routine" } }),
        request("tools/call", { name: "query", arguments: { query: "standard" } }, standard),
      ]);
      expect(store.search).toHaveBeenCalledTimes(2);
      expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ query: "routine", rerank: false }));
      expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ query: "standard", rerank: true }));
    } finally {
      await standard.close();
    }
  });

  test("captures the policy when constructing each server", async () => {
    await handler.close();
    const options: McpServerOptions = { queryPolicy: "split-rerank" };
    handler = createMcpHandler(async () => {
      const server = await createMcpServer(store, undefined, options);
      options.queryPolicy = "standard";
      return server;
    });
    await request("tools/call", { name: "query", arguments: { query: "pool", rerank: true } });
    expect(options.queryPolicy).toBe("standard");
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ rerank: false }));
  });

  test("preserves nonempty response formatting and metadata across both split tools", async () => {
    searchSpy.mockResolvedValue([{
      file: "qmd://docs/pool.md", displayPath: "docs/pool.md", title: "Pool",
      body: "# Pool\n\npool timeout", bestChunk: "pool timeout", bestChunkPos: 8,
      score: 0.678, context: "databases", docid: "abcdef", metadata: { status: "published" },
    }]);
    const standard = createMcpHandler(() => createMcpServer(store));
    try {
      const params = { name: "query", arguments: { query: "pool", filter } };
      const baseline = await request("tools/call", params, standard);
      const routine = await request("tools/call", params);
      const reranked = await request("tools/call", { ...params, name: "query_reranked" });
      expect(routine.result).toEqual(baseline.result);
      expect(reranked.result).toEqual(baseline.result);
      expect(routine.result.structuredContent.results).toEqual([expect.objectContaining({
        docid: "#abcdef", file: "docs/pool.md", title: "Pool", score: 0.68,
        context: "databases", metadata: { status: "published" }, line: 1,
      })]);
    } finally {
      await standard.close();
    }
  });

  test("keeps existing factory defaults and caller rerank control", async () => {
    await handler.close();
    handler = createMcpHandler(() => createMcpServer(store));
    const { result } = await request("tools/list");
    expect(result.tools.some((tool: { name: string }) => tool.name === "query_reranked")).toBe(false);
    const query = result.tools.find((tool: { name: string }) => tool.name === "query");
    expect(query.inputSchema.properties.rerank.default).toBe(true);
    await request("tools/call", { name: "query", arguments: { query: "pool" } });
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ rerank: true, candidateLimit: undefined }));
    searchSpy.mockClear();
    await request("tools/call", { name: "query", arguments: { query: "pool", rerank: false, candidateLimit: 40 } });
    expect(store.search).toHaveBeenCalledTimes(1);
    expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ rerank: false, candidateLimit: 40 }));
  });
});
