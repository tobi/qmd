/**
 * Integration tests for bin/qmd's daemon fast-path.
 *
 * These do NOT bootstrap the real MCP server. Instead they stand up a tiny
 * HTTP mock on a random port, export QMD_DAEMON_URL, and spawn bin/qmd to
 * verify the shell wrapper:
 *
 *   1. Sends the expected POST body shape (type, query, collections, limit).
 *   2. Accepts -n for limit (upstream CLI flag, not -l).
 *   3. Accumulates multiple -c / --collection flags.
 *   4. Falls through to cold-start with the ORIGINAL argv when /health or
 *      POST fails.
 */
import { describe, test, expect, afterAll } from "vitest";
import { spawn } from "node:child_process";
import { createServer, type Server, type IncomingMessage, type ServerResponse } from "node:http";
import { resolve } from "node:path";

interface Capture {
  method: string;
  path: string;
  body: string;
}

async function startMockServer(
  healthStatus: number,
  searchStatus: number,
  searchBody: string
): Promise<{ server: Server; url: string; captures: Capture[] }> {
  const captures: Capture[] = [];
  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const body = Buffer.concat(chunks).toString("utf8");
    captures.push({ method: req.method || "GET", path: req.url || "/", body });

    if (req.url === "/health") {
      res.writeHead(healthStatus, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
      return;
    }
    if (req.url === "/search") {
      res.writeHead(searchStatus, { "Content-Type": "application/json" });
      res.end(searchBody);
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
  const port = (server.address() as { port: number }).port;
  return { server, url: `http://127.0.0.1:${port}`, captures };
}

const BIN_QMD = resolve(__dirname, "..", "bin", "qmd");

/**
 * Async spawn so the event loop keeps running while bin/qmd's curl talks to
 * the mock HTTP server living in this same Node process. spawnSync would
 * block the loop and the mock could never respond.
 */
function runBin(args: string[], env: NodeJS.ProcessEnv): Promise<{ status: number; stdout: string; stderr: string }> {
  return new Promise((resolvePromise) => {
    const child = spawn("sh", [BIN_QMD, ...args], {
      env: { ...process.env, ...env },
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += String(d)));
    child.stderr.on("data", (d) => (stderr += String(d)));
    const killTimer = setTimeout(() => child.kill(), 10_000);
    child.on("close", (code) => {
      clearTimeout(killTimer);
      resolvePromise({ status: code ?? -1, stdout, stderr });
    });
  });
}

describe("bin/qmd daemon fast-path", () => {
  let mock: { server: Server; url: string; captures: Capture[] };

  afterAll(async () => {
    if (mock?.server) await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("POSTs correct shape for `qmd search <q> -c <c> -n <N>`", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [{ docid: "aa", file: "x.md", title: "X", score: 1 }] }));
    const run = await runBin(["search", "hello", "-c", "alpha", "-n", "3"], { QMD_DAEMON_URL: mock.url });
    if (run.status !== 0) {
      console.error("runBin stderr:", run.stderr);
      console.error("runBin stdout:", run.stdout);
      console.error("mock captures:", mock.captures);
    }
    expect(run.status).toBe(0);
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.searches).toEqual([{ type: "lex", query: "hello" }]);
    expect(payload.collections).toEqual(["alpha"]);
    expect(payload.limit).toBe(3);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("vsearch uses type='vec'", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["vsearch", "semantic query"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.searches).toEqual([{ type: "vec", query: "semantic query" }]);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("multiple -c flags are accumulated in collections", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo", "-c", "alpha", "-c", "beta", "-c", "gamma"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.collections).toEqual(["alpha", "beta", "gamma"]);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("--collection= form is accepted alongside -c", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "q", "-c", "one", "--collection=two"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.collections).toEqual(["one", "two"]);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("unset -c yields collections: null", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "untouched"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.collections).toBeNull();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("unset -n uses 5 to match the interactive CLI default", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.limit).toBe(5);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("-n 0 normalises to the default 5 (matches parseInt || default)", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo", "-n", "0"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.limit).toBe(5);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("-n <non-numeric> normalises to the default 5", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo", "-n", "1e2"], { QMD_DAEMON_URL: mock.url });
    const searchReq = mock.captures.find((c) => c.path === "/search");
    expect(searchReq).toBeDefined();
    const payload = JSON.parse(searchReq!.body);
    expect(payload.limit).toBe(5);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("--index bypasses daemon entirely (no /health call)", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["--index", "library", "search", "foo"], { QMD_DAEMON_URL: mock.url });
    expect(mock.captures.length).toBe(0);
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("non-search subcommands don't touch the daemon", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["status"], { QMD_DAEMON_URL: mock.url });
    expect(mock.captures.find((c) => c.path === "/search")).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("unknown flag falls through rather than eating its value into the query", async () => {
    // --min-score is a real upstream option that takes a value. If the
    // fast-path blindly skipped just the flag token, `0.8` would get
    // appended to `_qmd_query`. Correct behavior is to bail so the
    // cold-start CLI (which knows the full flag set) handles it.
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo", "--min-score", "0.8"], { QMD_DAEMON_URL: mock.url });
    expect(mock.captures.find((c) => c.path === "/search")).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("boolean flags also fall through (e.g. --json)", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    await runBin(["search", "foo", "--json"], { QMD_DAEMON_URL: mock.url });
    expect(mock.captures.find((c) => c.path === "/search")).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("dangling value flag (-n with no argument) falls through without aborting shell", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    const run = await runBin(["search", "foo", "-n"], { QMD_DAEMON_URL: mock.url });
    // The fast-path must return non-zero and let cold-start handle it.
    // Most importantly it must NOT abort /bin/sh with
    // "shift: can't shift that many" (exit 2 or higher from /bin/sh).
    expect(run.stderr).not.toMatch(/can't shift|shift.*many/);
    expect(mock.captures.find((c) => c.path === "/search")).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("dangling -c with no argument also falls through cleanly", async () => {
    mock = await startMockServer(200, 200, JSON.stringify({ results: [] }));
    const run = await runBin(["search", "foo", "-c"], { QMD_DAEMON_URL: mock.url });
    expect(run.stderr).not.toMatch(/can't shift|shift.*many/);
    expect(mock.captures.find((c) => c.path === "/search")).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });

  test("health-check failure silently falls through (no /search attempted)", async () => {
    mock = await startMockServer(500, 200, "{}");
    await runBin(["search", "foo"], { QMD_DAEMON_URL: mock.url });
    const searchCall = mock.captures.find((c) => c.path === "/search");
    expect(searchCall).toBeUndefined();
    await new Promise<void>((r) => mock.server.close(() => r()));
  });
});
