import { afterEach, describe, expect, test } from "vitest";
import { createServer } from "node:net";
import { LlamaCpp } from "../src/llm.ts";
import { startMcpHttpServer, startMcpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import type { QMDStore } from "../src/index.ts";

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

function createFakeStore(
  calls: string[],
  block: () => Promise<void>,
): QMDStore {
  const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
  const origDispose = llm.dispose.bind(llm);
  llm.dispose = async () => {
    calls.push("llm-dispose");
    return origDispose();
  };

  return {
    internal: { llm, close() { calls.push("internal-close"); } } as QMDStore["internal"],
    dbPath: ":memory:",
    search: async () => {
      await block();
      return [];
    },
    searchLex: async () => [],
    searchVector: async () => [],
    expandQuery: async () => [],
    get: async () => ({ error: "not_found", similarFiles: [] } as never),
    getDocumentBody: async () => null,
    multiGet: async () => ({ docs: [], errors: [] }),
    addCollection: async () => undefined,
    removeCollection: async () => false,
    renameCollection: async () => false,
    listCollections: async () => [],
    getDefaultCollectionNames: async () => [],
    addContext: async () => false,
    removeContext: async () => false,
    setGlobalContext: async () => undefined,
    getGlobalContext: async () => undefined,
    listContexts: async () => [],
    update: async () => ({ collections: 0, indexed: 0, updated: 0, unchanged: 0, removed: 0, skipped: 0, needsEmbedding: 0 }),
    embed: async () => ({ embedded: 0, skipped: 0, failed: 0 } as never),
    getStatus: async () => {
      await block();
      return { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
    },
    getIndexHealth: async () => ({ needsEmbedding: 0, totalDocs: 0, daysStale: null } as never),
    close: async () => { calls.push("store-close"); },
  };
}

const MCP_VERSION = "2026-07-28";

async function postJson(url: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

describe("HTTP MCP/REST shutdown", () => {
  let handle: HttpServerHandle | undefined;

  afterEach(async () => {
    if (handle) {
      const stopping = handle.stop();
      handle = undefined;
      await Promise.race([
        stopping.catch(() => undefined),
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
    }
  });

  test("stop waits for one MCP and one REST request, counts each once, and shares one promise", async () => {
    const releaseAll = deferred();
    const bothStarted = deferred();
    const calls: string[] = [];
    let entered = 0;

    const store = createFakeStore(calls, async () => {
      entered += 1;
      if (entered === 2) bothStarted.resolve();
      await releaseAll.promise;
    });

    handle = await startMcpHttpServer(0, {
      quiet: true,
      host: "127.0.0.1",
      store,
      setExitCode: () => undefined,
      holdOpen: () => () => undefined,
    });
    const baseUrl = `http://127.0.0.1:${handle.port}`;

    const rest = postJson(`${baseUrl}/query`, {
      searches: [{ type: "lex", query: "rest" }],
    });
    const mcp = postJson(`${baseUrl}/mcp`, {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/list",
      params: {
        _meta: {
          "io.modelcontextprotocol/protocolVersion": MCP_VERSION,
          "io.modelcontextprotocol/clientInfo": { name: "lifecycle", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }, {
      Accept: "application/json, text/event-stream",
      "MCP-Protocol-Version": MCP_VERSION,
      "Mcp-Method": "tools/list",
    });

    await bothStarted.promise;
    expect(handle.inflight.getActiveCount()).toBe(2);

    const first = handle.stop();
    const second = handle.stop();
    expect(first).toBe(second);

    const thirdStatus = await postJson(`${baseUrl}/query`, {
      searches: [{ type: "lex", query: "late" }],
    }).then((res) => res.status).catch(() => "closed");
    expect(thirdStatus === 503 || thirdStatus === "closed").toBe(true);
    expect(handle.inflight.getActiveCount()).toBe(2);
    expect(calls).not.toContain("llm-dispose");
    expect(calls).not.toContain("store-close");

    releaseAll.resolve();
    await Promise.all([rest, mcp]);
    await first;

    expect(calls).toEqual(["llm-dispose", "store-close"]);

    await expect(new Promise<void>((resolve, reject) => {
      const socket = createServer();
      socket.once("error", reject);
      socket.listen(handle!.port, "127.0.0.1", () => {
        socket.close(() => resolve());
      });
    })).resolves.toBeUndefined();
  });
});

describe("MCP startup failure ownership", () => {
  test("listen failure does not close a caller-owned store or LLM", async () => {
    const calls: string[] = [];
    const store = createFakeStore(calls, async () => undefined);
    await expect(startMcpHttpServer(0, {
      quiet: true,
      store,
      host: "127.0.0.1",
      listen: async () => {
        throw Object.assign(new Error("listen EADDRINUSE"), { code: "EADDRINUSE" });
      },
      holdOpen: () => () => undefined,
    })).rejects.toMatchObject({ code: "EADDRINUSE" });
    expect(calls).not.toContain("store-close");
    expect(calls).not.toContain("llm-dispose");
  });

  test("stdio transport failure does not close a caller-owned store", async () => {
    const calls: string[] = [];
    const store = createFakeStore(calls, async () => undefined);
    const { EventEmitter } = await import("node:events");
    class FakeStdin extends EventEmitter {
      readableEnded = false;
      destroyed = false;
    }
    await expect(startMcpServer({
      store,
      stdin: new FakeStdin(),
      createTransport: () => {
        throw new Error("transport failed");
      },
    })).rejects.toThrow("transport failed");
    expect(calls).not.toContain("store-close");
    expect(calls).not.toContain("llm-dispose");
  });
});
