import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { createServer, type Server } from "http";
import {
  discoverDaemon,
  readPortFile,
  searchViaDaemon,
  type SearchViaDaemonOptions,
} from "../src/daemon.js";
import type { HybridQueryResult } from "../src/store.js";

describe("daemon discovery", () => {
  let cacheDir: string;
  let origCache: string | undefined;
  let origDefaultPort: string | undefined;
  let httpServer: Server | undefined;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), "qmd-daemon-test-"));
    origCache = process.env.XDG_CACHE_HOME;
    origDefaultPort = process.env.QMD_DEFAULT_PORT;
    process.env.XDG_CACHE_HOME = cacheDir;
    delete process.env.QMD_DEFAULT_PORT;
  });

  afterEach(async () => {
    if (httpServer) {
      await new Promise<void>((r) => httpServer!.close(() => r()));
      httpServer = undefined;
    }
    if (origCache === undefined) delete process.env.XDG_CACHE_HOME;
    else process.env.XDG_CACHE_HOME = origCache;
    if (origDefaultPort === undefined) delete process.env.QMD_DEFAULT_PORT;
    else process.env.QMD_DEFAULT_PORT = origDefaultPort;
    await rm(cacheDir, { recursive: true, force: true });
  });

  function startFakeDaemon(
    port: number,
    handler: (url: string) => { status: number; body: string },
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      httpServer = createServer((req, res) => {
        const { status, body } = handler(req.url || "/");
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(body);
      });
      httpServer.once("error", reject);
      httpServer.listen(port, "127.0.0.1", () => resolve());
    });
  }

  const healthBody = JSON.stringify({ status: "ok", uptime: 1, dbPath: "/tmp/fake.sqlite" });

  async function writeCacheFiles(pid: number, port: number): Promise<void> {
    const dir = join(cacheDir, "qmd");
    await writeFile(join(dir, "mcp.pid"), String(pid), { flag: "w" }).catch(async () => {
      const { mkdir } = await import("fs/promises");
      await mkdir(dir, { recursive: true });
      await writeFile(join(dir, "mcp.pid"), String(pid));
    });
    await writeFile(join(dir, "mcp.port"), String(port));
  }

  test("returns null when no PID file", async () => {
    const result = await discoverDaemon();
    expect(result).toBeNull();
  });

  test("returns null when PID points to dead process", async () => {
    await writeCacheFiles(999999999, 12345);
    const result = await discoverDaemon();
    expect(result).toBeNull();
  });

  test("returns baseUrl and dbPath when PID is live AND /health responds", async () => {
    // Use current process PID as the "daemon" since it's guaranteed alive
    const port = 17000 + Math.floor(Math.random() * 1000);
    await startFakeDaemon(port, (url) => {
      if (url === "/health") return { status: 200, body: healthBody };
      return { status: 404, body: "" };
    });
    await writeCacheFiles(process.pid, port);

    const result = await discoverDaemon();
    expect(result).not.toBeNull();
    expect(result!.baseUrl).toBe(`http://localhost:${port}`);
    expect(result!.pid).toBe(process.pid);
    expect(result!.port).toBe(port);
    expect(result!.dbPath).toBe("/tmp/fake.sqlite");
  });

  test("returns null when /health omits dbPath (old daemon version)", async () => {
    const port = 17000 + Math.floor(Math.random() * 1000);
    await startFakeDaemon(port, (url) => {
      if (url === "/health") return { status: 200, body: JSON.stringify({ status: "ok", uptime: 1 }) };
      return { status: 404, body: "" };
    });
    await writeCacheFiles(process.pid, port);

    const result = await discoverDaemon();
    expect(result).toBeNull();
  });

  test("returns null when /health fails within timeout", async () => {
    // Point at a port with nothing listening
    await writeCacheFiles(process.pid, 17999);
    const result = await discoverDaemon({ timeoutMs: 200 });
    expect(result).toBeNull();
  });

  test("falls back to DEFAULT_PORT when port file missing", async () => {
    // Use a random high port so tests don't collide with a real qmd
    // daemon possibly running on 8181.
    const fallbackPort = 17000 + Math.floor(Math.random() * 40000);
    process.env.QMD_DEFAULT_PORT = String(fallbackPort);

    // Only write PID file
    const { mkdir, writeFile: wf } = await import("fs/promises");
    await mkdir(join(cacheDir, "qmd"), { recursive: true });
    await wf(join(cacheDir, "qmd", "mcp.pid"), String(process.pid));

    await startFakeDaemon(fallbackPort, (url) =>
      url === "/health" ? { status: 200, body: healthBody } : { status: 404, body: "" }
    );

    try {
      const result = await discoverDaemon();
      expect(result?.port).toBe(fallbackPort);
      expect(result?.dbPath).toBe("/tmp/fake.sqlite");
    } finally {
      delete process.env.QMD_DEFAULT_PORT;
    }
  });

  test("readPortFile returns null when file missing", async () => {
    expect(await readPortFile()).toBeNull();
  });

  test("readPortFile parses integer", async () => {
    const { mkdir, writeFile: wf } = await import("fs/promises");
    await mkdir(join(cacheDir, "qmd"), { recursive: true });
    await wf(join(cacheDir, "qmd", "mcp.port"), "12345\n");
    expect(await readPortFile()).toBe(12345);
  });
});

describe("searchViaDaemon", () => {
  let cacheDir: string;
  let origCache: string | undefined;
  let httpServer: Server | undefined;
  let baseUrl: string;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), "qmd-daemon-client-test-"));
    origCache = process.env.XDG_CACHE_HOME;
    process.env.XDG_CACHE_HOME = cacheDir;
  });

  afterEach(async () => {
    if (httpServer) {
      await new Promise<void>((r) => httpServer!.close(() => r()));
      httpServer = undefined;
    }
    if (origCache === undefined) delete process.env.XDG_CACHE_HOME;
    else process.env.XDG_CACHE_HOME = origCache;
    await rm(cacheDir, { recursive: true, force: true });
  });

  function startFakeDaemon(
    handler: (url: string, body: string) => { status: number; body: string },
  ): Promise<number> {
    return new Promise((resolve, reject) => {
      httpServer = createServer(async (req, res) => {
        let body = "";
        for await (const chunk of req) body += chunk;
        const { status, body: respBody } = handler(req.url || "/", body);
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(respBody);
      });
      httpServer.once("error", reject);
      httpServer.listen(0, "127.0.0.1", () => {
        const port = (httpServer!.address() as import("net").AddressInfo).port;
        baseUrl = `http://localhost:${port}`;
        resolve(port);
      });
    });
  }

  const stubResult: HybridQueryResult = {
    file: "qmd://test/a.md",
    displayPath: "test/a.md",
    title: "a",
    body: "hello",
    bestChunk: "hello",
    bestChunkPos: 0,
    score: 0.9,
    context: null,
    docid: "abcdef",
  };

  test("returns parsed results on 200", async () => {
    await startFakeDaemon((url) => {
      if (url === "/v1/search") {
        return { status: 200, body: JSON.stringify({ results: [stubResult] }) };
      }
      return { status: 404, body: "" };
    });
    const res = await searchViaDaemon(baseUrl, { query: "hi", limit: 1 });
    expect(res).not.toBeNull();
    expect(res!).toHaveLength(1);
    expect(res![0]!.file).toBe("qmd://test/a.md");
  });

  test("returns null on 500", async () => {
    await startFakeDaemon(() => ({ status: 500, body: JSON.stringify({ error: "boom" }) }));
    const res = await searchViaDaemon(baseUrl, { query: "hi" });
    expect(res).toBeNull();
  });

  test("returns null on 400", async () => {
    await startFakeDaemon(() => ({ status: 400, body: JSON.stringify({ error: "bad" }) }));
    const res = await searchViaDaemon(baseUrl, { query: "hi" });
    expect(res).toBeNull();
  });

  test("returns null when server unreachable", async () => {
    const res = await searchViaDaemon("http://localhost:1", { query: "hi" });
    expect(res).toBeNull();
  });

  test("forwards all option fields", async () => {
    let received: any = null;
    await startFakeDaemon((url, body) => {
      if (url === "/v1/search") {
        received = JSON.parse(body);
        return { status: 200, body: JSON.stringify({ results: [] }) };
      }
      return { status: 404, body: "" };
    });

    const options: SearchViaDaemonOptions = {
      searches: [{ type: "lex", query: "foo" }],
      limit: 7,
      minScore: 0.25,
      candidateLimit: 21,
      skipRerank: true,
      explain: true,
      intent: "intent hint",
      collections: ["x"],
      chunkStrategy: "auto",
    };

    await searchViaDaemon(baseUrl, options);

    expect(received).toEqual({
      searches: [{ type: "lex", query: "foo" }],
      limit: 7,
      minScore: 0.25,
      candidateLimit: 21,
      skipRerank: true,
      explain: true,
      intent: "intent hint",
      collections: ["x"],
      chunkStrategy: "auto",
    });
    expect(received.query).toBeUndefined();
  });
});
