/**
 * Lifecycle regression tests for the MCP transports' explicit
 * `serving -> draining -> stopped` state machine and its bounded shutdown.
 *
 * The machine is the safety property both transports depend on, so these tests
 * pin the transitions themselves and the races they close:
 *
 * - HTTP stop() used to close the shared store while request handlers were
 *   still in flight (httpServer.close() does not wait for them), so a request
 *   that finished during shutdown found a closed SQLite database.
 * - HTTP stop() used to close the MCP transport before it drained, so an
 *   admitted /mcp request that resumed afterwards found a closed handler.
 * - HTTP stop() used to return early for a concurrent caller, so the second
 *   caller resolved before teardown finished and nothing awaited it.
 * - HTTP stop() used to have no deadline at all, so one stalled request held
 *   the process forever. It now has one budget (5 s by default) for the whole
 *   teardown; at the deadline it reports a typed timeout while the abandoned
 *   cleanup continues without closing the store under live work.
 * - The /mcp route used to buffer the SDK response with arrayBuffer(), so a
 *   subscriptions/listen stream never produced a byte and never ended.
 *
 * A request is held in flight by sending only half of its body: the handler is
 * admitted, parked in body collection, and cannot finish until the test sends
 * the rest. Admission and draining are observed through the lifecycle the
 * listener is given, so no test waits on a guessed duration. The deadline
 * tests use a small explicit `shutdownDrainTimeoutMs` because they are about
 * wall-clock bounds by definition; the 5 s default is pinned as a constant and
 * through the drain budget the default stop() asks for.
 */

import { describe, test, expect, afterEach } from "vitest";
import { connect } from "node:net";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  MCP_SHUTDOWN_TIMEOUT_MS,
  McpShutdownTimeoutError,
  createInflightGate,
  createMcpLifecycle,
  createShutdownDeadline,
  registerStdioEofShutdown,
  runMcpTeardown,
  startMcpHttpServer,
  terminateMcpServer,
  type HttpServerHandle,
  type McpLifecycle,
  type McpTeardownSteps,
} from "../src/mcp/server.js";
import { _resetProductionModeForTesting } from "../src/store.js";

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  const timer = setTimeout(resolve, ms);
  timer.unref?.();
  return promise;
}

class FakeStdin extends EventEmitter {
  readableEnded = false;
  destroyed = false;
}

/**
 * A lifecycle that delegates every transition to a real one and exposes the
 * moments the transport would otherwise only reveal through timing: the first
 * admitted request, the start of the drain, and the deadline each drain wait
 * asked for (`undefined` = the graceful default, no deadline at all).
 */
function observableLifecycle(): {
  lifecycle: McpLifecycle;
  admitted: Promise<void>;
  draining: Promise<void>;
  drainWaits: Array<number | undefined>;
} {
  const inner = createMcpLifecycle();
  const firstAdmit = Promise.withResolvers<void>();
  const drainStart = Promise.withResolvers<void>();
  const drainWaits: Array<number | undefined> = [];
  let admittedOnce = false;

  return {
    admitted: firstAdmit.promise,
    draining: drainStart.promise,
    drainWaits,
    lifecycle: {
      state: () => inner.state(),
      admit: () => {
        const release = inner.admit();
        if (release && !admittedOnce) {
          admittedOnce = true;
          firstAdmit.resolve();
        }
        return release;
      },
      beginDrain: () => {
        const began = inner.beginDrain();
        if (began) drainStart.resolve();
        return began;
      },
      finishTeardown: () => inner.finishTeardown(),
      waitForIdle: (timeoutMs?: number) => {
        drainWaits.push(timeoutMs);
        return inner.waitForIdle(timeoutMs);
      },
    },
  };
}

describe("createMcpLifecycle", () => {
  test("admits work while serving and counts it as in-flight", async () => {
    const lifecycle = createMcpLifecycle();
    expect(lifecycle.state()).toBe("serving");

    const release = lifecycle.admit();
    expect(release).not.toBeNull();
    await expect(lifecycle.waitForIdle(10)).resolves.toBe(false);

    release!();
    await expect(lifecycle.waitForIdle(1000)).resolves.toBe(true);
    expect(lifecycle.state()).toBe("serving");
  });

  test("a release function is idempotent", async () => {
    const lifecycle = createMcpLifecycle();
    const release = lifecycle.admit()!;
    const other = lifecycle.admit()!;

    // Double-release must not drive the counter negative and fool the drain.
    release();
    release();
    await expect(lifecycle.waitForIdle(10)).resolves.toBe(false);

    other();
    await expect(lifecycle.waitForIdle(1000)).resolves.toBe(true);
  });

  test("draining refuses admission and cannot be entered twice", async () => {
    const lifecycle = createMcpLifecycle();

    expect(lifecycle.beginDrain()).toBe(true);
    expect(lifecycle.state()).toBe("draining");
    // A second transition attempt is a no-op, not a second drain.
    expect(lifecycle.beginDrain()).toBe(false);
    expect(lifecycle.admit()).toBeNull();
    await expect(lifecycle.waitForIdle(1000)).resolves.toBe(true);
  });

  test("waitForIdle still resolves once an admitted unit settles mid-drain", async () => {
    const lifecycle = createMcpLifecycle();
    const release = lifecycle.admit()!;

    lifecycle.beginDrain();
    const idle = lifecycle.waitForIdle(5000);
    expect(lifecycle.state()).toBe("draining");

    release();
    await expect(idle).resolves.toBe(true);
  });

  test("waitForIdle with no timeout waits for the last admitted unit instead of expiring", async () => {
    const lifecycle = createMcpLifecycle();
    const release = lifecycle.admit()!;

    let settled: boolean | undefined;
    const idle = lifecycle.waitForIdle().then((value) => {
      settled = value;
      return value;
    });
    // No timer is armed, so a settled flag here would mean an implicit
    // deadline: delay(0) only flushes pending callbacks.
    await delay(0);
    expect(settled).toBeUndefined();

    release();
    await expect(idle).resolves.toBe(true);
  });

  test("stopped is terminal and only reachable from draining", async () => {
    const lifecycle = createMcpLifecycle();

    expect(lifecycle.finishTeardown()).toBe(false); // serving -> stopped is illegal
    expect(lifecycle.state()).toBe("serving");

    expect(lifecycle.beginDrain()).toBe(true);
    expect(lifecycle.finishTeardown()).toBe(true);
    expect(lifecycle.state()).toBe("stopped");

    expect(lifecycle.finishTeardown()).toBe(false);
    expect(lifecycle.beginDrain()).toBe(false);
    expect(lifecycle.admit()).toBeNull();
  });
});

describe("createInflightGate", () => {
  test("refuses a handler dispatched after admission closed", async () => {
    const gate = createInflightGate();
    const handler = gate.track(async () => "ok");

    await expect(handler()).resolves.toBe("ok");

    gate.beginDrain();
    // The store is about to close: a late dispatch must not run the handler.
    await expect(handler()).rejects.toThrow(/shutting down/);
    // A refused handler never counted as in-flight, so the drain is immediate.
    await expect(gate.waitForIdle(1000)).resolves.toBe(true);
  });
});

describe("registerStdioEofShutdown lifecycle", () => {
  function register(gate: McpLifecycle, overrides: {
    closeServer?: () => Promise<void>;
    closeStore?: () => void;
    waitForIdle?: (timeoutMs?: number) => Promise<boolean>;
    idleTimeoutMs?: number;
    stdin?: FakeStdin;
  } = {}) {
    const stdin = overrides.stdin ?? new FakeStdin();
    const warnings: string[] = [];
    const shutdown = registerStdioEofShutdown({
      stdin,
      lifecycle: gate,
      closeServer: overrides.closeServer ?? (async () => {}),
      closeStore: overrides.closeStore ?? (() => {}),
      ...(overrides.waitForIdle ? { waitForIdle: overrides.waitForIdle } : {}),
      ...(overrides.idleTimeoutMs !== undefined ? { idleTimeoutMs: overrides.idleTimeoutMs } : {}),
      setExitCode: () => {},
      getExitCode: () => undefined,
      stderr: { write: (chunk: string) => { warnings.push(chunk); return true; } },
    });
    return { stdin, shutdown, warnings };
  }

  test("closes admission before the transport and stops after the store", async () => {
    const gate = createInflightGate();
    const seen: string[] = [];

    const { stdin, shutdown } = register(gate, {
      closeServer: async () => { seen.push(`server-close:${gate.state()}`); },
      closeStore: () => { seen.push(`store-close:${gate.state()}`); },
    });

    stdin.emit("end");
    await shutdown();

    // A request the closing transport still dispatches finds admission shut,
    // and the store only goes while the machine is still draining.
    expect(seen).toEqual(["server-close:draining", "store-close:draining"]);
    expect(gate.state()).toBe("stopped");
  });

  test("drains through the lifecycle when no waitForIdle is injected", async () => {
    const gate = createInflightGate();
    const order: string[] = [];
    const handlerSettled = Promise.withResolvers<void>();
    const handler = gate.track(() => handlerSettled.promise);

    const running = handler();
    const serverClosed = Promise.withResolvers<void>();
    const { stdin, shutdown } = register(gate, {
      closeServer: async () => {
        order.push("server-close");
        serverClosed.resolve();
      },
      closeStore: () => { order.push("store-close"); },
      idleTimeoutMs: 5000,
    });

    let settled = false;
    const done = shutdown();
    void done.then(() => { settled = true; });
    stdin.emit("end");
    await serverClosed.promise;

    // The lifecycle's drain is what stdio waits on now: the store close cannot
    // happen while the tracked handler is still running. delay(0) only flushes
    // pending teardown steps; it is not a guessed duration.
    await delay(0);
    expect(settled).toBe(false);
    expect(order).toEqual(["server-close"]);

    handlerSettled.resolve();
    await running;
    await done;

    expect(order).toEqual(["server-close", "store-close"]);
    expect(gate.state()).toBe("stopped");
    // Late work is refused rather than run against the closed store.
    await expect(handler()).rejects.toThrow(/shutting down/);
  });

  test("an injected waitForIdle still wins over the lifecycle drain", async () => {
    const gate = createInflightGate();
    const calls: string[] = [];
    const { stdin, shutdown } = register(gate, {
      // The gate is serving, so its own drain would say "idle" immediately.
      waitForIdle: async () => { calls.push("injected-drain"); return true; },
    });

    stdin.emit("end");
    await shutdown();

    expect(calls).toEqual(["injected-drain"]);
    expect(gate.state()).toBe("stopped");
  });
});

describe("qmd mcp HTTP shutdown lifecycle", () => {
  const cleanups: Array<() => Promise<void>> = [];
  /**
   * Sockets parked mid-request. Destroying them releases the drain, so a
   * failing assertion cannot leave cleanup waiting on a request the test never
   * finished -- a graceful stop has no deadline to fall back on.
   */
  const heldSockets = new Set<() => void>();
  const envSnapshot = {
    configDir: process.env.QMD_CONFIG_DIR,
    indexPath: process.env.INDEX_PATH,
  };

  afterEach(async () => {
    while (cleanups.length > 0) await cleanups.pop()!();
    _resetProductionModeForTesting();
  });

  async function startHarness(
    options: { shutdownDrainTimeoutMs?: number } = {},
  ): Promise<{
    handle: HttpServerHandle;
    lifecycle: McpLifecycle;
    admitted: Promise<void>;
    draining: Promise<void>;
    drainWaits: Array<number | undefined>;
    port: number;
  }> {
    const dir = await mkdtemp(join(tmpdir(), "qmd-mcp-lifecycle-"));
    process.env.QMD_CONFIG_DIR = dir;
    delete process.env.INDEX_PATH;

    const observed = observableLifecycle();
    let handle: HttpServerHandle | undefined;
    cleanups.push(async () => {
      for (const destroy of heldSockets) destroy();
      heldSockets.clear();
      if (handle) await handle.stop().catch(() => {});
      _resetProductionModeForTesting();
      if (envSnapshot.configDir === undefined) delete process.env.QMD_CONFIG_DIR;
      else process.env.QMD_CONFIG_DIR = envSnapshot.configDir;
      if (envSnapshot.indexPath === undefined) delete process.env.INDEX_PATH;
      else process.env.INDEX_PATH = envSnapshot.indexPath;
      await rm(dir, { recursive: true, force: true });
    });

    handle = await startMcpHttpServer(0, {
      quiet: true,
      host: "127.0.0.1",
      dbPath: join(dir, "index.sqlite"),
      lifecycle: observed.lifecycle,
      ...(options.shutdownDrainTimeoutMs !== undefined
        ? { shutdownDrainTimeoutMs: options.shutdownDrainTimeoutMs }
        : {}),
    });
    return {
      handle,
      port: handle.port,
      lifecycle: observed.lifecycle,
      admitted: observed.admitted,
      draining: observed.draining,
      drainWaits: observed.drainWaits,
    };
  }

  /** A REST search body, and an MCP initialize body the handler serves 200 to. */
  const QUERY_BODY = JSON.stringify({ searches: [{ type: "lex", query: "anything" }] });
  const MCP_INITIALIZE_BODY = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "lifecycle-test", version: "1.0.0" },
    },
  });
  /** The 2026-07-28 envelope every modern (per-request) message carries. */
  const MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": { name: "lifecycle-test", version: "1.0.0" },
    "io.modelcontextprotocol/clientCapabilities": {},
  };

  /**
   * Open a request whose body is only half sent, so the handler is admitted and
   * parked in body collection: the server cannot finish it until sendRest().
   * Resolves once the listener has admitted the request. `waitForText` lets a
   * test observe a streamed response without ending the connection.
   */
  async function heldRequest(
    port: number,
    admitted: Promise<void>,
    request: { path: string; body: string; headers?: Record<string, string> },
  ) {
    const { path, body } = request;
    const cut = Math.floor(body.length / 2);
    const socket = connect(port, "127.0.0.1");
    socket.setEncoding("utf8");
    let text = "";
    const waiters: Array<{ needle: string; resolve: () => void }> = [];
    const closed = Promise.withResolvers<string>();
    const destroy = socket.destroy.bind(socket);
    // The cleanup destroys this socket if the test fails before the request
    // finishes; otherwise the drain it is holding would never go idle.
    heldSockets.add(destroy);
    const settle = () => {
      heldSockets.delete(destroy);
      closed.resolve(text);
    };
    socket.on("data", (chunk: string) => {
      text += chunk;
      for (const waiter of [...waiters]) {
        if (!text.includes(waiter.needle)) continue;
        waiters.splice(waiters.indexOf(waiter), 1);
        waiter.resolve();
      }
    });
    socket.on("close", settle);
    socket.on("error", settle);

    const connected = Promise.withResolvers<void>();
    socket.on("connect", () => connected.resolve());
    await connected.promise;

    const rawHeaders = Object.entries(request.headers ?? {})
      .map(([name, value]) => `${name}: ${value}\r\n`)
      .join("");
    socket.write(
      `POST ${path} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Type: application/json\r\n`
      + `Connection: close\r\n${rawHeaders}Content-Length: ${Buffer.byteLength(body)}\r\n\r\n`
      + body.slice(0, cut),
    );
    await admitted;

    return {
      response: closed.promise,
      sendRest: () => socket.write(body.slice(cut)),
      destroy,
      /** Watches the bytes already received (headers included) for `needle`. */
      async waitForText(needle: string): Promise<string> {
        while (!text.includes(needle)) {
          const { promise, resolve } = Promise.withResolvers<void>();
          waiters.push({ needle, resolve });
          await promise;
        }
        return text;
      },
    };
  }

  async function connectionOutcome(port: number): Promise<string> {
    const probe = connect(port, "127.0.0.1");
    const { promise, resolve } = Promise.withResolvers<string>();
    probe.on("connect", () => { probe.destroy(); resolve("connected"); });
    probe.on("error", (error: NodeJS.ErrnoException) => resolve(error.code ?? "error"));
    return promise;
  }

  /**
   * stop() closes the listener before it drains, so no new connection is
   * accepted while an admitted request still runs. Polled rather than timed:
   * the refusal appears once httpServer.close() ran, whenever that is.
   */
  async function waitForListenerClosed(port: number): Promise<void> {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      if (await connectionOutcome(port) === "ECONNREFUSED") return;
      await delay(10);
    }
    throw new Error(`listener on port ${port} still accepts connections`);
  }

  test("the shipped default is a five second budget for the whole teardown", () => {
    expect(MCP_SHUTDOWN_TIMEOUT_MS).toBe(5000);
  });

  test("an admitted request finishes inside the default grace and stop() resolves cleanly", async () => {
    const { handle, port, admitted, draining, drainWaits, lifecycle } = await startHarness();
    const held = await heldRequest(port, admitted, { path: "/query", body: QUERY_BODY });

    let firstSettled = false;
    let secondSettled = false;
    const first = handle.stop().then(() => { firstSettled = true; });
    const second = handle.stop();
    // A second stop() must wait on the teardown the first one started, not
    // resolve early with a store that is still being closed.
    void second.then(() => { secondSettled = true; });

    // Wait until the teardown has actually begun. The listener is already
    // closed for new connections while the admitted request still runs, so
    // nothing can enter behind the drain.
    await draining;
    await waitForListenerClosed(port);

    await delay(0);
    expect(firstSettled).toBe(false);
    expect(secondSettled).toBe(false);

    // Finishing the request now must still hit an open store: before the drain
    // existed, stop() had already closed it and this answered 500.
    held.sendRest();
    const response = await Promise.race([held.response, delay(10_000).then(() => "")]);
    expect(response).toContain("200 OK");
    expect(response).toContain('"results"');

    await Promise.all([first, second]);
    expect(firstSettled).toBe(true);
    expect(secondSettled).toBe(true);
    expect(lifecycle.state()).toBe("stopped");
    await expect(handle.closed).resolves.toBeUndefined();

    // One budget for the whole teardown, taken from the 5 s default. The
    // request was admitted, so the drain asked for the time that was left.
    expect(drainWaits).toHaveLength(1);
    expect(drainWaits[0]).toBeGreaterThan(MCP_SHUTDOWN_TIMEOUT_MS - 1000);
    expect(drainWaits[0]).toBeLessThanOrEqual(MCP_SHUTDOWN_TIMEOUT_MS);
  });

  test("an admitted /mcp request finishes after the listener closes: the transport is not torn down underneath it", async () => {
    const { handle, port, admitted, draining } = await startHarness();
    const held = await heldRequest(port, admitted, {
      path: "/mcp",
      body: MCP_INITIALIZE_BODY,
      headers: { Accept: "application/json, text/event-stream" },
    });

    const stopped = handle.stop();
    await draining;
    await waitForListenerClosed(port);

    // The request resumes only now, after stop() began. Closing the MCP
    // transport before the drain made mcpHandler.fetch throw ("handler has
    // been closed"), which surfaced as a 500 with no JSON-RPC result.
    held.sendRest();
    const response = await Promise.race([held.response, delay(10_000).then(() => "")]);
    expect(response).toContain("200 OK");
    expect(response).toContain("serverInfo");

    await stopped;
    await handle.closed;
  });

  // Real timers: the deadline is wall-clock behaviour the platform provides.
  test("a stalled request is reported as a typed timeout, and the store closes only after it settles", async () => {
    const drainMs = 50;
    const { handle, lifecycle, port, admitted, drainWaits } = await startHarness({
      shutdownDrainTimeoutMs: drainMs,
    });
    const held = await heldRequest(port, admitted, { path: "/query", body: QUERY_BODY });

    const started = Date.now();
    const failure = await handle.stop().catch((error: unknown) => error);
    const elapsed = Date.now() - started;

    expect(failure).toBeInstanceOf(McpShutdownTimeoutError);
    const timeout = failure as McpShutdownTimeoutError;
    expect(timeout.code).toBe("MCP_SHUTDOWN_TIMEOUT");
    expect(timeout.phase).toBe("drain");
    expect(timeout.timeoutMs).toBe(drainMs);
    expect(elapsed).toBeGreaterThanOrEqual(drainMs - 10);
    expect(drainWaits[0]).toBeGreaterThanOrEqual(drainMs - 5);
    expect(drainWaits[0]).toBeLessThanOrEqual(drainMs);

    // Reported, not torn down: the store is still open under the abandoned
    // request, and the lifecycle has not claimed "stopped".
    expect(lifecycle.state()).toBe("draining");
    expect(await connectionOutcome(port)).toBe("ECONNREFUSED");
    // Shared stop outcome: a second caller sees the same timeout.
    await expect(handle.stop()).rejects.toBeInstanceOf(McpShutdownTimeoutError);

    // Once the abandoned request releases (its socket died, as it would after
    // the deadline severed it), the background cleanup closes the store and
    // only then claims stopped.
    held.destroy();
    await handle.closed;
    expect(lifecycle.state()).toBe("stopped");
  });

  test("the /mcp route streams SSE bytes, and a client disconnect releases its admission", async () => {
    const { handle, lifecycle, port, admitted } = await startHarness({ shutdownDrainTimeoutMs: 2000 });
    const held = await heldRequest(port, admitted, {
      path: "/mcp",
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 7,
        method: "subscriptions/listen",
        params: { notifications: { toolsListChanged: true }, _meta: MODERN_META },
      }),
      headers: {
        Accept: "application/json, text/event-stream",
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "subscriptions/listen",
      },
    });
    held.sendRest();

    // The ack frame arrives while the subscription is still open. Buffering
    // the SDK response with arrayBuffer() meant no byte was ever written and
    // the handler never returned.
    const streaming = await held.waitForText("notifications/subscriptions/acknowledged");
    expect(streaming).toContain("200 OK");
    expect(streaming).toContain("text/event-stream");

    // The client goes away: the exchange is aborted through its web Request
    // signal, so the admission is released and stop() finishes inside its
    // grace instead of timing out on a stream nobody is reading.
    held.destroy();
    await expect(handle.stop()).resolves.toBeUndefined();
    await handle.closed;
    expect(lifecycle.state()).toBe("stopped");
  });

  test("stop() is idempotent once the lifecycle is stopped", async () => {
    const { handle, lifecycle } = await startHarness();

    await handle.stop();
    expect(lifecycle.state()).toBe("stopped");

    // A second await returns the shared, already-settled teardown.
    await expect(handle.stop()).resolves.toBeUndefined();
  });

  test("a pre-drained lifecycle answers 503 instead of reading the shared store", async () => {
    const { handle, lifecycle, port } = await startHarness();

    // An owner that knows the process is going away drains admission first;
    // stop() has not run yet, so the listener is still up and dispatching.
    expect(lifecycle.beginDrain()).toBe(true);

    const response = await fetch(`http://127.0.0.1:${port}/health`);
    expect(response.status).toBe(503);
    const body = await response.json() as { error?: { message?: string } };
    expect(body.error?.message).toContain("shutting down");

    // Teardown still runs for a lifecycle someone else drained.
    await handle.stop();
    expect(lifecycle.state()).toBe("stopped");
  });

  test("programmatic startup installs no signal handlers and a timeout never exits the caller", async () => {
    const termListeners = process.listenerCount("SIGTERM");
    const intListeners = process.listenerCount("SIGINT");
    const exitCode = process.exitCode;

    const { handle, port, admitted } = await startHarness({ shutdownDrainTimeoutMs: 40 });
    // A standalone entry owns termination; an embedded host must not have its
    // process exited by handlers it never installed.
    expect(process.listenerCount("SIGTERM")).toBe(termListeners);
    expect(process.listenerCount("SIGINT")).toBe(intListeners);

    const held = await heldRequest(port, admitted, { path: "/query", body: QUERY_BODY });
    await expect(handle.stop()).rejects.toBeInstanceOf(McpShutdownTimeoutError);

    // The timeout is a rejected promise, not an exit: this test is still
    // running and the host's own status is untouched.
    expect(process.exitCode).toBe(exitCode);

    held.destroy();
    await handle.closed;
  });
});

describe("createShutdownDeadline", () => {
  test("names the step that was in progress when it expired", async () => {
    const deadline = createShutdownDeadline(20);
    const failure = await deadline.at("store").catch((error: unknown) => error);
    deadline.dispose();

    expect(failure).toBeInstanceOf(McpShutdownTimeoutError);
    const timeout = failure as McpShutdownTimeoutError;
    expect(timeout.phase).toBe("store");
    expect(timeout.timeoutMs).toBe(20);
    expect(timeout.code).toBe("MCP_SHUTDOWN_TIMEOUT");
  });

  test("dispose() disarms it so a finished teardown leaves no rejection behind", async () => {
    const deadline = createShutdownDeadline(20);
    deadline.dispose();

    // Real timer on purpose: an armed deadline would reject at 20 ms.
    await expect(Promise.race([deadline.at("drain"), delay(80).then(() => "waited")])).resolves.toBe("waited");
  });

  test("the remaining budget shrinks and stops at zero", async () => {
    const deadline = createShutdownDeadline(20);
    expect(deadline.remainingMs()).toBeGreaterThan(0);
    expect(deadline.remainingMs()).toBeLessThanOrEqual(20);

    await delay(40);
    expect(deadline.remainingMs()).toBe(0);
    deadline.dispose();
  });
});

describe("runMcpTeardown", () => {
  /**
   * The shipped step order, recording each call into `order`. Tests that
   * override a step pass their own array so the base and the overrides write
   * to the same log.
   */
  function makeSteps(overrides: Partial<McpTeardownSteps> = {}, order: string[] = []) {
    const steps: McpTeardownSteps = {
      beginDrain: () => { order.push("beginDrain"); return true; },
      waitForIdle: async (timeoutMs?: number) => {
        order.push(timeoutMs === undefined ? "drain:unbounded" : "drain:bounded");
        return true;
      },
      abortWork: () => { order.push("abortWork"); },
      severConnections: () => { order.push("severConnections"); },
      beginListenerClose: () => { order.push("beginListenerClose"); },
      listenerClosed: async () => { order.push("listenerClosed"); },
      closeTransport: async () => { order.push("closeTransport"); },
      closeStore: async () => { order.push("closeStore"); },
      finishTeardown: () => { order.push("finishTeardown"); return true; },
      ...overrides,
    };
    return { steps, order };
  }

  test("starts nothing until stop() is called and runs the steps in lifecycle order", async () => {
    const { steps, order } = makeSteps();
    const teardown = runMcpTeardown(steps, { timeoutMs: 1000 });

    expect(order).toEqual([]);
    expect(teardown.stop()).toBe(teardown.stop());

    await expect(teardown.stop()).resolves.toBeUndefined();
    await expect(teardown.closed).resolves.toBeUndefined();
    expect(order).toEqual([
      "beginDrain",
      "beginListenerClose",
      "drain:bounded",
      "closeTransport",
      "listenerClosed",
      "closeStore",
      "finishTeardown",
    ]);
  });

  test("a drain that outlives the deadline is reported, and the store closes only after the work settles", async () => {
    const order: string[] = [];
    const work = Promise.withResolvers<boolean>();
    const drainWaits: Array<number | undefined> = [];
    const { steps } = makeSteps({
      waitForIdle: (timeoutMs?: number) => {
        drainWaits.push(timeoutMs);
        order.push(timeoutMs === undefined ? "drain:unbounded" : "drain:bounded");
        return work.promise;
      },
      closeStore: async () => { order.push("closeStore"); },
      finishTeardown: () => { order.push("finishTeardown"); return true; },
    }, order);
    const teardown = runMcpTeardown(steps, { timeoutMs: 40 });

    const failure = await teardown.stop().catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(McpShutdownTimeoutError);
    expect((failure as McpShutdownTimeoutError).phase).toBe("drain");
    expect(drainWaits[0]).toBeGreaterThan(0);
    expect(drainWaits[0]).toBeLessThanOrEqual(40);
    // Reported, then abandoned: the work was told to stop and the cleanup went
    // on to the unbounded wait instead of closing the store underneath it.
    expect(order).toEqual([
      "beginDrain",
      "beginListenerClose",
      "drain:bounded",
      "abortWork",
      "severConnections",
      "drain:unbounded",
    ]);

    let closedSettled = false;
    void teardown.closed.then(() => { closedSettled = true; });
    await delay(0);
    expect(closedSettled).toBe(false);
    expect(order).not.toContain("closeStore");
    expect(order).not.toContain("finishTeardown");

    work.resolve(true);
    await teardown.closed;
    expect(order.slice(-2)).toEqual(["closeStore", "finishTeardown"]);
  });

  test("a transport close that hangs is reported and the store waits for it", async () => {
    const order: string[] = [];
    const transport = Promise.withResolvers<void>();
    const { steps } = makeSteps({
      closeTransport: () => { order.push("closeTransport"); return transport.promise; },
      closeStore: async () => { order.push("closeStore"); },
      finishTeardown: () => { order.push("finishTeardown"); return true; },
    }, order);
    const teardown = runMcpTeardown(steps, { timeoutMs: 40 });

    const failure = await teardown.stop().catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(McpShutdownTimeoutError);
    expect((failure as McpShutdownTimeoutError).phase).toBe("transport");
    expect(order).not.toContain("closeStore");

    let closedSettled = false;
    void teardown.closed.then(() => { closedSettled = true; });
    await delay(0);
    expect(closedSettled).toBe(false);

    // The hung close finishing on its own is what lets cleanup continue; the
    // store closes only after it, never before it.
    transport.resolve();
    await teardown.closed;
    expect(order.slice(-3)).toEqual(["listenerClosed", "closeStore", "finishTeardown"]);
  });

  test("a store close that hangs reports a timeout and never claims the teardown finished", async () => {
    const order: string[] = [];
    const store = Promise.withResolvers<void>();
    const { steps } = makeSteps({
      closeStore: () => { order.push("closeStore"); return store.promise; },
      finishTeardown: () => { order.push("finishTeardown"); return true; },
    }, order);
    const teardown = runMcpTeardown(steps, { timeoutMs: 40 });

    const failure = await teardown.stop().catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(McpShutdownTimeoutError);
    expect((failure as McpShutdownTimeoutError).phase).toBe("store");
    // The store close was attempted (and is still pending), but the teardown
    // has not been claimed as finished.
    expect(order).toContain("closeStore");
    expect(order).not.toContain("finishTeardown");

    let closedSettled = false;
    void teardown.closed.then(() => { closedSettled = true; });
    await delay(0);
    // `closed` stays pending while the store is still closing: "stopped" is
    // never claimed before the store actually closed.
    expect(closedSettled).toBe(false);

    store.resolve();
    await teardown.closed;
    expect(order.slice(-1)).toEqual(["finishTeardown"]);
    // Ordering, not just presence: completion is only claimed after the store
    // close resolved, never before it.
    expect(order.indexOf("finishTeardown")).toBeGreaterThan(order.lastIndexOf("closeStore"));
  });

  test("a failed step is logged, the store still closes, and the failure is rethrown", async () => {
    const order: string[] = [];
    const logged: string[] = [];
    const { steps } = makeSteps({
      closeTransport: async () => {
        order.push("closeTransport");
        throw new Error("transport already gone");
      },
    }, order);
    const teardown = runMcpTeardown(steps, { timeoutMs: 500, log: (message) => logged.push(message) });

    await expect(teardown.stop()).rejects.toThrow("transport already gone");
    await expect(teardown.closed).resolves.toBeUndefined();
    expect(order).toContain("closeStore");
    expect(order).toContain("finishTeardown");
    expect(logged.join("")).toContain("transport close failed during shutdown");
  });
});

describe("terminateMcpServer", () => {
  function record() {
    const exits: number[] = [];
    const logs: string[] = [];
    return {
      exits,
      logs,
      termination: {
        exit: (code: number) => { exits.push(code); },
        log: (message: string) => { logs.push(message); },
      },
    };
  }

  test("a clean stop exits 0", async () => {
    const recorded = record();
    await terminateMcpServer(async () => {}, recorded.termination);
    expect(recorded.exits).toEqual([0]);
    expect(recorded.logs).toEqual([]);
  });

  test("a reported timeout exits 1 immediately instead of waiting for abandoned work", async () => {
    const recorded = record();
    await terminateMcpServer(async () => {
      throw new McpShutdownTimeoutError("drain", MCP_SHUTDOWN_TIMEOUT_MS);
    }, recorded.termination);

    expect(recorded.exits).toEqual([1]);
    expect(recorded.logs.join("")).toContain("work was still running");
  });

  test("a failed teardown exits 1", async () => {
    const recorded = record();
    await terminateMcpServer(async () => {
      throw new Error("store.close() exploded");
    }, recorded.termination);

    expect(recorded.exits).toEqual([1]);
    expect(recorded.logs.join("")).toContain("store.close() exploded");
  });
});
