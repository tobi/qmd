import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { request } from "node:http";
import { createServer, type AddressInfo, type Server } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server";
import { _resetProductionModeForTesting } from "../src/store";

function listen(server: Server, host: string, port = 0): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      server.removeListener("error", reject);
      resolve((server.address() as AddressInfo).port);
    });
  });
}

function close(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close(error => error ? reject(error) : resolve());
  });
}

function getMcp(hostname: string, port: number, hostHeader?: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const req = request({
      hostname, port, path: "/mcp", agent: false,
      headers: hostHeader ? { Host: hostHeader, Origin: `http://${hostHeader}` } : {},
    }, res => {
      res.resume();
      res.on("end", () => resolve(res.statusCode!));
    });
    req.on("error", reject);
    req.end();
  });
}

const ipv6Available = await (async () => {
  const probe = createServer();
  try {
    await listen(probe, "::1");
    return true;
  } catch (error) {
    if (!["EADDRNOTAVAIL", "EAFNOSUPPORT"].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
    return false;
  } finally {
    if (probe.listening) await close(probe);
  }
})();

describe("MCP HTTP loopback listeners", () => {
  let workDir: string;
  let handle: HttpServerHandle | undefined;
  const servers: Server[] = [];
  const envKeys = ["QMD_HOST", "QMD_CONFIG_DIR", "QMD_ALLOWED_HOSTS", "QMD_ALLOWED_ORIGINS"] as const;
  const originalEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));
  const signalListeners = {
    SIGTERM: process.listeners("SIGTERM"),
    SIGINT: process.listeners("SIGINT"),
  };
  beforeEach(async () => {
    workDir = await mkdtemp(join(tmpdir(), "qmd-http-loopback-"));
    for (const key of envKeys) delete process.env[key];
    process.env.QMD_CONFIG_DIR = workDir;
  });

  afterEach(async () => {
    await handle?.stop();
    handle = undefined;
    for (const server of servers.splice(0)) {
      if (server.listening) await close(server);
    }
    _resetProductionModeForTesting();
    for (const key of envKeys) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
    for (const signal of ["SIGTERM", "SIGINT"] as const) {
      for (const listener of process.listeners(signal)) {
        if (!signalListeners[signal].includes(listener)) process.removeListener(signal, listener);
      }
    }
    await rm(workDir, { recursive: true, force: true });
  });

  async function start(host?: string, port = 0): Promise<HttpServerHandle> {
    handle = await startMcpHttpServer(port, { quiet: true, dbPath: join(workDir, "index.sqlite"), host });
    return handle;
  }

  async function occupy(host: string, port = 0): Promise<number> {
    const server = createServer();
    servers.push(server);
    return listen(server, host, port);
  }

  test("default host serves IPv4 on an ephemeral port", async () => {
    const { port } = await start();
    expect(port).toBeGreaterThan(0);
    expect(await getMcp("127.0.0.1", port)).toBe(405);
  });

  test.skipIf(!ipv6Available)("default host serves both families on the same port (requires IPv6 loopback)", async () => {
    const { port } = await start();
    expect(await getMcp("127.0.0.1", port)).toBe(405);
    expect(await getMcp("::1", port)).toBe(405);
  });

  test("accepts all loopback Host and Origin headers over IPv4", async () => {
    const { port } = await start();
    for (const host of ["127.0.0.1", "[::1]", "localhost"]) {
      expect(await getMcp("127.0.0.1", port, `${host}:${port}`)).toBe(405);
    }
  });

  test.skipIf(!ipv6Available)("accepts all loopback Host and Origin headers over IPv6 (requires IPv6 loopback)", async () => {
    const { port } = await start();
    for (const host of ["127.0.0.1", "[::1]", "localhost"]) {
      expect(await getMcp("::1", port, `${host}:${port}`)).toBe(405);
    }
  });

  test.skipIf(!ipv6Available).each(["option", "environment"])("explicit IPv4 host via %s does not serve IPv6 (requires IPv6 loopback)", async (source) => {
    if (source === "environment") process.env.QMD_HOST = "127.0.0.1";
    const { port } = await start(source === "option" ? "127.0.0.1" : undefined);
    expect(await getMcp("127.0.0.1", port)).toBe(405);
    await expect(getMcp("::1", port)).rejects.toMatchObject({ code: "ECONNREFUSED" });
  });

  test("IPv4 collision rejects and leaves no IPv6 listener when IPv6 loopback is available", async () => {
    const port = await occupy("127.0.0.1");
    await expect(start(undefined, port)).rejects.toMatchObject({ code: "EADDRINUSE" });
    if (ipv6Available) await occupy("::1", port);
  });

  test.skipIf(!ipv6Available)("IPv6 collision rejects and releases the IPv4 listener (requires IPv6 loopback)", async () => {
    const port = await occupy("::1");
    await expect(start(undefined, port)).rejects.toMatchObject({ code: "EADDRINUSE" });
    await occupy("127.0.0.1", port);
  });

  test.each(["stop", "close"])("handle %s releases IPv4 and IPv6 when IPv6 loopback is available", async (method) => {
    const server = await start();
    expect(await getMcp("127.0.0.1", server.port)).toBe(405);
    if (ipv6Available) expect(await getMcp("::1", server.port)).toBe(405);
    if (method === "stop") await server.stop();
    else await close(server.httpServer);
    await occupy("127.0.0.1", server.port);
    if (ipv6Available) await occupy("::1", server.port);
  });
});
