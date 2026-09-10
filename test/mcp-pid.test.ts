/**
 * Unit tests for MCP pidfile identity helpers (#806).
 */

import { describe, test, expect } from "vitest";
import {
  createProcessIdentity,
  createProcessRecord,
  isProcessIdentityLive,
  mcpDaemonStateFiles,
  looksLikeQmdRoleCommand,
  parseProcessIdentity,
  parseProcessRecord,
  processIdentityStatus,
  processRecordStatus,
  sameProcessIdentity,
  sameProcessRecord,
  serializeProcessIdentity,
  type ProcessIdentityProbe,
  type QmdProcessIdentity,
} from "../src/cli/mcp-pid.ts";

describe("mcpDaemonStateFiles", () => {
  test("default index keeps mcp.pid / mcp.log", () => {
    expect(mcpDaemonStateFiles("index")).toEqual({ pidFile: "mcp.pid", logFile: "mcp.log" });
    expect(mcpDaemonStateFiles("")).toEqual({ pidFile: "mcp.pid", logFile: "mcp.log" });
    expect(mcpDaemonStateFiles()).toEqual({ pidFile: "mcp.pid", logFile: "mcp.log" });
  });

  test("named indexes get scoped pid/log files (#772)", () => {
    expect(mcpDaemonStateFiles("hsm-public-repro")).toEqual({
      pidFile: "mcp-hsm-public-repro.pid",
      logFile: "mcp-hsm-public-repro.log",
    });
  });
});

describe("structured process identity", () => {
  const identity: QmdProcessIdentity = {
    format: "qmd-process/v1",
    pid: 4242,
    role: "mcp-http",
    startToken: "win32:123456789",
    port: 8181,
  };

  test("round-trips valid identity and rejects legacy or malformed state", () => {
    expect(parseProcessIdentity(serializeProcessIdentity(identity))).toEqual(identity);
    expect(parseProcessIdentity("4242\n")).toBeNull();
    expect(parseProcessIdentity('{"format":"qmd-process/v1","pid":4242,"role":"embed","startToken":""}')).toBeNull();
    expect(parseProcessRecord("4242\n")).toEqual({ kind: "legacy", pid: 4242 });
    expect(parseProcessRecord("not-a-pid")).toEqual({ kind: "invalid" });
  });

  test("requires liveness, exact start token, and expected role", () => {
    const liveProbe: ProcessIdentityProbe = {
      isAlive: pid => pid === identity.pid,
      startToken: pid => pid === identity.pid ? identity.startToken : null,
    };
    expect(isProcessIdentityLive(identity, "mcp-http", liveProbe)).toBe(true);
    expect(isProcessIdentityLive(identity, "embed", liveProbe)).toBe(false);
    expect(isProcessIdentityLive({ ...identity, startToken: "win32:recycled" }, "mcp-http", liveProbe)).toBe(false);
    expect(isProcessIdentityLive(identity, "mcp-http", { ...liveProbe, isAlive: () => false })).toBe(false);
    expect(processIdentityStatus(identity, "mcp-http", { ...liveProbe, startToken: () => null })).toBe("unknown");
    expect(processRecordStatus({ kind: "legacy", pid: identity.pid }, "mcp-http", liveProbe)).toBe("unknown");
    expect(processRecordStatus({ kind: "legacy", pid: identity.pid }, "mcp-http", {
      ...liveProbe,
      cmdline: () => "node /opt/qmd/dist/cli/qmd.js mcp --http --daemon",
    })).toBe("live");
    expect(processRecordStatus({ kind: "legacy", pid: identity.pid }, "mcp-http", {
      ...liveProbe,
      cmdline: () => "node unrelated.js",
    })).toBe("unknown");
    expect(processRecordStatus({ kind: "legacy", pid: identity.pid }, "mcp-http", {
      ...liveProbe,
      isAlive: () => false,
    })).toBe("dead");
    expect(processRecordStatus({ kind: "invalid" }, "mcp-http", liveProbe)).toBe("unknown");
    expect(sameProcessIdentity(identity, { ...identity })).toBe(true);
    expect(sameProcessIdentity(identity, { ...identity, pid: identity.pid + 1 })).toBe(false);
    expect(sameProcessIdentity(identity, { ...identity, startToken: "win32:replacement" })).toBe(false);
    expect(sameProcessRecord({ kind: "legacy", pid: 42 }, { kind: "legacy", pid: 42 })).toBe(true);
    expect(sameProcessRecord({ kind: "invalid" }, { kind: "invalid" })).toBe(false);
  });

  test("recognizes role-specific legacy qmd commands", () => {
    expect(looksLikeQmdRoleCommand("node /opt/qmd/qmd.js embed", "embed")).toBe(true);
    expect(looksLikeQmdRoleCommand("node /opt/qmd/qmd.js mcp --http --daemon", "mcp-http")).toBe(true);
    expect(looksLikeQmdRoleCommand("node /opt/qmd/qmd.js search query", "mcp-http")).toBe(false);
  });

  test("creates a verifiable identity for the current process on this platform", () => {
    const current = createProcessIdentity("embed");
    expect(current.pid).toBe(process.pid);
    expect(current.role).toBe("embed");
    expect(current.startToken.length).toBeGreaterThan(0);
    expect(isProcessIdentityLive(current, "embed")).toBe(true);
  });

  test("falls back to fail-closed legacy state when start identity is unavailable", () => {
    const record = createProcessRecord("embed", { pid: 1234 }, () => {
      throw new Error("identity unavailable");
    });

    expect(record).toEqual({ kind: "legacy", pid: 1234 });
    expect(processRecordStatus(record, "embed", {
      isAlive: () => true,
      startToken: () => null,
    })).toBe("unknown");
  });
});
