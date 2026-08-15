/** Covers QMD API-v1 HTTP behavior and lifecycle boundaries. */
import { request } from "node:http";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  startMcpHttpServer,
  type HttpServerHandle,
} from "../src/mcp/server.js";
import { _resetProductionModeForTesting } from "../src/store.js";

const REQUIRED_FEATURES = [
  "admission",
  "operations",
  "coalesced-update",
  "daemon-embed",
  "collection-ensure",
];

type OperationBody = {
  operationId: string;
  state: "queued" | "running" | "completed" | "failed";
  error?: { reason: string };
};

describe("QMD daemon API v1", () => {
  let handle: HttpServerHandle;
  let baseUrl: string;
  let root: string;
  let docsPath: string;
  let baselineSigtermListeners: number;
  let baselineSigintListeners: number;
  const originalIndexPath = process.env.INDEX_PATH;
  const originalConfigDir = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "qmd-http-v1-"));
    const configDir = join(root, "config");
    docsPath = join(root, "docs");
    await mkdir(configDir);
    await mkdir(docsPath);
    await writeFile(join(configDir, "index.yml"), "collections: {}\n");
    process.env.INDEX_PATH = join(root, "index.sqlite");
    process.env.QMD_CONFIG_DIR = configDir;
    baselineSigtermListeners = process.listenerCount("SIGTERM");
    baselineSigintListeners = process.listenerCount("SIGINT");
    handle = await startMcpHttpServer(0, {
      host: "127.0.0.1",
      quiet: true,
    });
    baseUrl = `http://127.0.0.1:${handle.port}`;
  });

  afterAll(async () => {
    await handle?.stop();
    if (originalIndexPath === undefined) delete process.env.INDEX_PATH;
    else process.env.INDEX_PATH = originalIndexPath;
    if (originalConfigDir === undefined) delete process.env.QMD_CONFIG_DIR;
    else process.env.QMD_CONFIG_DIR = originalConfigDir;
    _resetProductionModeForTesting();
    await rm(root, { recursive: true, force: true });
  });

  async function waitForOperation(operationId: string): Promise<OperationBody> {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      const response = await fetch(`${baseUrl}/v1/operations/${operationId}`);
      expect(response.status).toBe(200);
      const operation = (await response.json()) as OperationBody;
      if (operation.state === "completed" || operation.state === "failed")
        return operation;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error("operation did not finish");
  }

  test("reports the exact API-v1 capability gate", async () => {
    const response = await fetch(`${baseUrl}/health`);
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body).toMatchObject({
      status: "ok",
      apiVersion: 1,
      features: REQUIRED_FEATURES,
      indexGeneration: 0,
      admission: {
        activeHeavy: 0,
        queuedInteractive: 0,
        maintenanceActive: false,
        queuedMaintenance: 0,
      },
    });
    expect(typeof body.version).toBe("string");
    expect(typeof body.uptime).toBe("number");
  });

  test("returns safe 400 responses for malformed and oversized bodies", async () => {
    const malformed = await fetch(`${baseUrl}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{",
    });
    expect(malformed.status).toBe(400);
    expect(await malformed.json()).toEqual({
      status: "unavailable",
      reason: "malformed",
      authoritativeEmpty: false,
    });

    const oversized = await fetch(`${baseUrl}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "x".repeat(1_048_577),
    });
    expect(oversized.status).toBe(400);
    expect(await oversized.json()).toEqual({
      status: "unavailable",
      reason: "malformed",
      authoritativeEmpty: false,
    });

    const streamed = await new Promise<{ status: number; body: string }>(
      (resolve, reject) => {
        const req = request(
          {
            hostname: "127.0.0.1",
            port: handle.port,
            path: "/v1/search",
            method: "POST",
            headers: { "Content-Type": "application/json" },
          },
          (response) => {
            let body = "";
            response.setEncoding("utf8");
            response.on("data", (chunk) => {
              body += chunk;
            });
            response.on("end", () =>
              resolve({ status: response.statusCode ?? 0, body }),
            );
          },
        );
        req.on("error", reject);
        req.write("x".repeat(700_000));
        req.end("x".repeat(400_000));
      },
    );
    expect(streamed.status).toBe(400);
    expect(JSON.parse(streamed.body)).toEqual({
      status: "unavailable",
      reason: "malformed",
      authoritativeEmpty: false,
    });

    const excessiveLimit = await fetch(`${baseUrl}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "bounded", limit: 501 }),
    });
    expect(excessiveLimit.status).toBe(400);
  });

  test("returns authoritative empty lexical results without local inference", async () => {
    const response = await fetch(`${baseUrl}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        searches: [{ type: "lex", query: "absent" }],
        collections: [],
        rerank: false,
      }),
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      status: "ok",
      mode: "lexical",
      authoritativeEmpty: true,
      results: [],
    });
  });

  test("normalizes collections and exposes privacy-safe operation failures", async () => {
    const accepted = await fetch(`${baseUrl}/v1/collections/ensure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        adds: [{ name: "notes", path: docsPath, pattern: "**/*.md" }],
        markDirty: false,
      }),
    });
    expect(accepted.status).toBe(202);
    const scheduled = (await accepted.json()) as OperationBody;
    expect(scheduled.operationId).toMatch(/^op_/);
    expect((await waitForOperation(scheduled.operationId)).state).toBe(
      "completed",
    );

    const listed = await fetch(`${baseUrl}/v1/collections`);
    expect(listed.status).toBe(200);
    expect(await listed.json()).toEqual({
      collections: [
        {
          name: "notes",
          path: docsPath,
          pattern: "**/*.md",
          documents: 0,
          activeDocuments: 0,
          lastModified: null,
          includeByDefault: true,
        },
      ],
    });

    const failing = await fetch(`${baseUrl}/v1/collections/ensure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        updates: [{ name: "missing", path: "/private/path" }],
        markDirty: false,
      }),
    });
    expect(failing.status).toBe(202);
    const failed = await waitForOperation(
      ((await failing.json()) as OperationBody).operationId,
    );
    expect(failed).toMatchObject({
      state: "failed",
      error: { reason: "maintenance_failed" },
    });
    expect(JSON.stringify(failed)).not.toContain("/private/path");
  });

  test("cleans up disconnected requests", async () => {
    await new Promise<void>((resolve) => {
      const req = request({
        hostname: "127.0.0.1",
        port: handle.port,
        path: "/v1/search",
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      req.on("error", () => resolve());
      req.write('{"searches":[');
      req.destroy();
      setTimeout(resolve, 50);
    });
    const response = await fetch(`${baseUrl}/health`);
    expect(response.status).toBe(200);
    await response.arrayBuffer();
  });

  test("stops partial requests and shares concurrent stop calls", async () => {
    expect(process.listenerCount("SIGTERM")).toBe(baselineSigtermListeners + 1);
    expect(process.listenerCount("SIGINT")).toBe(baselineSigintListeners + 1);
    const partial = request({
      hostname: "127.0.0.1",
      port: handle.port,
      path: "/v1/search",
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    partial.on("error", () => {});
    partial.write('{"query":"partial');
    await new Promise((resolve) => setTimeout(resolve, 10));

    const first = handle.stop();
    const second = handle.stop();
    expect(second).toBe(first);
    await Promise.all([first, second]);
    partial.destroy();
    expect(process.listenerCount("SIGTERM")).toBe(baselineSigtermListeners);
    expect(process.listenerCount("SIGINT")).toBe(baselineSigintListeners);
  }, 2000);
});
