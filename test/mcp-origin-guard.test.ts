/**
 * DNS-rebinding protection for the HTTP MCP transport (#881).
 *
 * The predicate tests run everywhere; the live-server tests boot a real
 * `startMcpHttpServer` on an ephemeral port and drive it with spoofed headers.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import YAML from "yaml";
import {
  checkRequestOrigin,
  isLoopbackHostname,
  resolveOriginGuard,
} from "../src/mcp/origin-guard.js";
import { openDatabase } from "../src/db.js";
import type { CollectionConfig } from "../src/collections.js";

// =============================================================================
// Predicates
// =============================================================================

describe("isLoopbackHostname", () => {
  test.each([
    "localhost",
    "LOCALHOST",
    "app.localhost",
    "127.0.0.1",
    "127.1.2.3",
    "::1",
    "[::1]",
    "::ffff:127.0.0.1",
  ])("accepts %s", (hostname) => {
    expect(isLoopbackHostname(hostname)).toBe(true);
  });

  test.each([
    "evil.example",
    "notlocalhost",
    "localhost.evil.example",
    "127.0.0.1.evil.example",
    "10.0.0.1",
    "::2",
  ])("rejects %s", (hostname) => {
    expect(isLoopbackHostname(hostname)).toBe(false);
  });

  // 0.0.0.0 routes to loopback in browsers on Linux/macOS, which makes it a
  // rebinding-free path to a local server — so it is not a trusted origin.
  test("rejects 0.0.0.0", () => {
    expect(isLoopbackHostname("0.0.0.0")).toBe(false);
  });
});

describe("resolveOriginGuard", () => {
  const env = {} as NodeJS.ProcessEnv;

  test("loopback bind enforces the Host header", () => {
    const guard = resolveOriginGuard({ host: "localhost", env });
    expect(guard.disabled).toBe(false);
    expect(guard.enforceHost).toBe(true);
  });

  test("wildcard bind without an allowlist skips Host enforcement", () => {
    const guard = resolveOriginGuard({ host: "0.0.0.0", env });
    expect(guard.enforceHost).toBe(false);
  });

  test("wildcard bind with an explicit allowlist enforces Host", () => {
    const guard = resolveOriginGuard({ host: "0.0.0.0", allowedHosts: ["qmd.internal"], env });
    expect(guard.enforceHost).toBe(true);
    expect(guard.allowedHosts).toContain("qmd.internal");
  });

  test("a concrete non-loopback bind trusts its own address", () => {
    const guard = resolveOriginGuard({ host: "192.168.1.5", env });
    expect(guard.enforceHost).toBe(true);
    expect(guard.allowedHosts).toContain("192.168.1.5");
  });

  test("QMD_ALLOWED_ORIGINS=* disables every check", () => {
    const guard = resolveOriginGuard({ host: "localhost", env: { QMD_ALLOWED_ORIGINS: "*" } });
    expect(guard.disabled).toBe(true);
    expect(checkRequestOrigin({ origin: "https://evil.example", host: "evil.example" }, guard).ok).toBe(true);
  });

  test("reads comma-separated allowlists from the environment", () => {
    const guard = resolveOriginGuard({
      host: "localhost",
      env: {
        QMD_ALLOWED_ORIGINS: "https://notes.internal , https://app.internal",
        QMD_ALLOWED_HOSTS: "notes.internal",
      },
    });
    expect(guard.allowedOrigins).toEqual(["https://notes.internal", "https://app.internal"]);
    expect(guard.allowedHosts).toEqual(["notes.internal"]);
  });
});

describe("checkRequestOrigin", () => {
  const guard = resolveOriginGuard({ host: "localhost", env: {} as NodeJS.ProcessEnv });

  test("allows a request with no Origin header (curl, MCP clients)", () => {
    expect(checkRequestOrigin({ host: "localhost:8181" }, guard).ok).toBe(true);
  });

  test("allows loopback origins", () => {
    for (const origin of ["http://localhost:8181", "http://127.0.0.1:8181", "http://[::1]:8181"]) {
      expect(checkRequestOrigin({ origin, host: "localhost:8181" }, guard).ok).toBe(true);
    }
  });

  test("rejects a foreign Origin", () => {
    const verdict = checkRequestOrigin({ origin: "https://evil.example", host: "localhost:8181" }, guard);
    expect(verdict.ok).toBe(false);
    expect(verdict.ok === false && verdict.reason).toContain("Origin not allowed");
  });

  test("rejects a rebound Host even when Origin is absent", () => {
    const verdict = checkRequestOrigin({ host: "evil.example:8181" }, guard);
    expect(verdict.ok).toBe(false);
    expect(verdict.ok === false && verdict.reason).toContain("Host not allowed");
  });

  test("rejects an origin that merely embeds a loopback name", () => {
    expect(checkRequestOrigin({ origin: "https://localhost.evil.example" }, guard).ok).toBe(false);
    expect(checkRequestOrigin({ origin: "https://evil.example#localhost" }, guard).ok).toBe(false);
  });

  test("rejects the opaque null origin from sandboxed frames", () => {
    expect(checkRequestOrigin({ origin: "null", host: "localhost:8181" }, guard).ok).toBe(false);
  });

  test("rejects non-http schemes", () => {
    expect(checkRequestOrigin({ origin: "file://", host: "localhost:8181" }, guard).ok).toBe(false);
  });

  test("honours an explicit origin allowlist", () => {
    const allowing = resolveOriginGuard({
      host: "localhost",
      allowedOrigins: ["https://notes.internal"],
      allowedHosts: ["notes.internal"],
      env: {} as NodeJS.ProcessEnv,
    });
    expect(checkRequestOrigin({ origin: "https://notes.internal", host: "notes.internal" }, allowing).ok).toBe(true);
    expect(checkRequestOrigin({ origin: "https://other.internal", host: "notes.internal" }, allowing).ok).toBe(false);
  });

  test("skips Host validation on a wildcard bind but still screens Origin", () => {
    const docker = resolveOriginGuard({ host: "0.0.0.0", env: {} as NodeJS.ProcessEnv });
    expect(checkRequestOrigin({ host: "qmd-container:8181" }, docker).ok).toBe(true);
    expect(checkRequestOrigin({ origin: "https://evil.example", host: "qmd-container:8181" }, docker).ok).toBe(false);
  });
});

// =============================================================================
// Live server
// =============================================================================

describe.skipIf(!!process.env.CI)("MCP HTTP server rejects cross-origin requests", () => {
  let handle: import("../src/mcp/server.js").HttpServerHandle;
  let baseUrl: string;
  let workDir: string;
  let configDir: string;
  const origIndexPath = process.env.INDEX_PATH;
  const origConfigDir = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    workDir = mkdtempSync(join(tmpdir(), "qmd-origin-"));
    const dbPath = join(workDir, "index.sqlite");

    const { createStore } = await import("../src/store.js");
    createStore(dbPath).close();

    const db = openDatabase(dbPath);
    const now = new Date().toISOString();
    db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
      .run("hash-secret", "# Secrets\n\nThe recovery password is hunter2.\n", now);
    db.prepare(
      `INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES ('docs', ?, ?, ?, ?, ?, 1)`,
    ).run("secrets.md", "Secrets", "hash-secret", now, now);

    const config: CollectionConfig = {
      collections: { docs: { path: workDir, pattern: "**/*.md" } },
    };
    const { syncConfigToDb } = await import("../src/store.js");
    syncConfigToDb(db, config);
    db.close();

    configDir = mkdtempSync(join(tmpdir(), "qmd-origin-cfg-"));
    writeFileSync(join(configDir, "index.yml"), YAML.stringify(config));

    process.env.INDEX_PATH = dbPath;
    process.env.QMD_CONFIG_DIR = configDir;

    const { startMcpHttpServer } = await import("../src/mcp/server.js");
    handle = await startMcpHttpServer(0, { quiet: true });
    baseUrl = `http://localhost:${handle.port}`;
  });

  afterAll(async () => {
    await handle?.stop();
    if (origIndexPath !== undefined) process.env.INDEX_PATH = origIndexPath;
    else delete process.env.INDEX_PATH;
    if (origConfigDir !== undefined) process.env.QMD_CONFIG_DIR = origConfigDir;
    else delete process.env.QMD_CONFIG_DIR;
    for (const dir of [workDir, configDir]) {
      // Best effort: Windows keeps the SQLite WAL handles briefly after close.
      try { if (dir) rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
    }
  });

  const searchBody = JSON.stringify({ searches: [{ type: "lex", query: "password" }], rerank: false });

  test("POST /query leaks nothing to a foreign origin", async () => {
    const res = await fetch(`${baseUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://evil.example" },
      body: searchBody,
    });
    expect(res.status).toBe(403);
    const text = await res.text();
    expect(text).not.toContain("hunter2");
    expect(text).toContain("Origin not allowed");
  });

  test("POST /search is screened too", async () => {
    const res = await fetch(`${baseUrl}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://evil.example" },
      body: searchBody,
    });
    expect(res.status).toBe(403);
  });

  test("POST /mcp initialize is rejected from a foreign origin", async () => {
    const res = await fetch(`${baseUrl}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Origin: "https://evil.example",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "evil", version: "1" } },
      }),
    });
    expect(res.status).toBe(403);
  });

  test("a rebound Host header is rejected without an Origin", async () => {
    // fetch() forbids setting Host, so drive a raw socket for this one.
    const { request } = await import("node:http");
    const status = await new Promise<number>((resolve, reject) => {
      const req = request(
        {
          hostname: "localhost",
          port: handle.port,
          path: "/query",
          method: "POST",
          headers: { "Content-Type": "application/json", Host: "evil.example", "Content-Length": Buffer.byteLength(searchBody) },
        },
        (res) => {
          res.resume();
          resolve(res.statusCode ?? 0);
        },
      );
      req.on("error", reject);
      req.end(searchBody);
    });
    expect(status).toBe(403);
  });

  test("local clients still work", async () => {
    const noOrigin = await fetch(`${baseUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: searchBody,
    });
    expect(noOrigin.status).toBe(200);
    expect(await noOrigin.text()).toContain("hunter2");

    const loopbackOrigin = await fetch(`${baseUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: `http://localhost:${handle.port}` },
      body: searchBody,
    });
    expect(loopbackOrigin.status).toBe(200);

    const health = await fetch(`${baseUrl}/health`);
    expect(health.status).toBe(200);
  });
});
