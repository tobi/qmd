/**
 * StdoutWriteGuard: keeps stdio JSON-RPC responses on the real stdout while
 * node-llama-cpp's native-init redirect (withNativeStdoutRedirectedToStderr
 * in src/llm.ts) has process.stdout.write hijacked for a concurrent request.
 * See createStdoutWriteGuard in src/mcp/server.ts.
 */

import { describe, test, expect, vi } from "vitest";
import { EventEmitter } from "node:events";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { createStdoutWriteGuard } from "../src/mcp/server";
import { withNativeStdoutRedirectedToStderr } from "../src/llm";

describe("qmd mcp stdio server/discover over the guarded transport", () => {
  const repoRoot = fileURLToPath(new URL("..", import.meta.url));
  const cliPath = join(repoRoot, "src", "cli", "qmd.ts");

  test("answers a 2026-07-28 client's server/discover opening exchange", async () => {
    // Regression: serveStdio's dual-era negotiation (2026-07-28 vs. 2025-era)
    // must keep working now that its transport writes through StdoutWriteGuard
    // instead of process.stdout directly.
    const workDir = await mkdtemp(join(tmpdir(), "qmd-stdio-lifecycle-discover-"));
    let child: ReturnType<typeof spawn> | undefined;
    try {
      await writeFile(join(workDir, "index.yml"), "collections: {}\n");

      const runtimeArgs = process.versions.bun
        ? [cliPath, "mcp"]
        : ["--import", "tsx", cliPath, "mcp"];

      child = spawn(process.execPath, runtimeArgs, {
        cwd: repoRoot,
        env: {
          ...process.env,
          INDEX_PATH: join(workDir, "discover.sqlite"),
          QMD_CONFIG_DIR: workDir,
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      const stderrChunks: string[] = [];
      child.stderr.on("data", (chunk) => stderrChunks.push(String(chunk)));

      const response = await new Promise<string>((resolve, reject) => {
        let buffer = "";
        const onData = (chunk: Buffer) => {
          buffer += String(chunk);
          if (buffer.includes("\n")) {
            child.stdout.off("data", onData);
            resolve(buffer);
          }
        };
        child.stdout.on("data", onData);
        child.once("error", reject);
        child.once("exit", (code) =>
          reject(new Error(`server exited before responding (code ${code}): ${stderrChunks.join("")}`))
        );
        child.stdin.write(
          JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "server/discover",
            params: {
              _meta: {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
              },
            },
          }) + "\n"
        );
      });

      expect(response).toContain('"jsonrpc":"2.0"');
      expect(response).toContain('"supportedVersions"');
      expect(response).toContain("2026-07-28");
    } finally {
      if (child && child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
      await rm(workDir, { recursive: true, force: true });
    }
  }, 60_000);
});

describe("createStdoutWriteGuard vs the native stdout redirect", () => {
  test("a JSON-RPC response sent through the guarded transport still lands on stdout while a concurrent native init redirects stdout to stderr", async () => {
    const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      // Must be built before the redirect starts, exactly like startMcpServer
      // does — that's what pins the guard to the true stdout write.
      const transport = new StdioServerTransport(process.stdin, createStdoutWriteGuard());

      await withNativeStdoutRedirectedToStderr(async () => {
        process.stdout.write("cmake build spam\n");
        await transport.send({ jsonrpc: "2.0", id: 200, result: {} });
      });

      expect(stderrSpy.mock.calls.some((call) => String(call[0]).includes("cmake build spam"))).toBe(true);
      expect(stdoutSpy.mock.calls.some((call) => String(call[0]).includes('"id":200'))).toBe(true);
      // Proves the fix: the response must never leak onto stderr, where the
      // client would never look for it.
      expect(stderrSpy.mock.calls.some((call) => String(call[0]).includes('"id":200'))).toBe(false);
    } finally {
      stdoutSpy.mockRestore();
      stderrSpy.mockRestore();
    }
  });

  test("regression: without the guard, a bare stdout transport loses the same response to stderr", async () => {
    // The bug: a transport wired straight to process.stdout (the pre-fix
    // default) has its response rerouted to stderr by the concurrent redirect.
    const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      const unguardedTransport = new StdioServerTransport(process.stdin, process.stdout);

      await withNativeStdoutRedirectedToStderr(async () => {
        await unguardedTransport.send({ jsonrpc: "2.0", id: 200, result: {} });
      });

      expect(stdoutSpy.mock.calls.some((call) => String(call[0]).includes('"id":200'))).toBe(false);
      expect(stderrSpy.mock.calls.some((call) => String(call[0]).includes('"id":200'))).toBe(true);
    } finally {
      stdoutSpy.mockRestore();
      stderrSpy.mockRestore();
    }
  });
});

describe("StdoutWriteGuard error forwarding", () => {
  /** Minimal stand-in for `process.stdout`: an EventEmitter with a `write`. */
  class FakeStdoutStream extends EventEmitter {
    write(_chunk: unknown, encodingOrCallback?: unknown, callback?: unknown): boolean {
      const cb = typeof encodingOrCallback === "function" ? encodingOrCallback : callback;
      if (typeof cb === "function") (cb as () => void)();
      return true;
    }
  }

  test("does not throw when the real stdout errors while nobody is listening on the guard", () => {
    // Trap: Writable throws synchronously on emit("error", ...) with zero
    // listeners. This is the window before start() attaches its handler and
    // after close() removes it, where the guard itself has none.
    const fakeStdout = new FakeStdoutStream();
    createStdoutWriteGuard(fakeStdout);

    expect(() => fakeStdout.emit("error", new Error("EPIPE"))).not.toThrow();
  });

  test("still forwards stdout errors to a guard that has an 'error' listener", async () => {
    const fakeStdout = new FakeStdoutStream();
    const guard = createStdoutWriteGuard(fakeStdout);

    const seen = await new Promise<Error>((resolve) => {
      guard.on("error", resolve);
      fakeStdout.emit("error", new Error("EPIPE"));
    });

    expect(seen.message).toBe("EPIPE");
  });
});
