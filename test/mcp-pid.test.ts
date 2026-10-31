/**
 * Unit tests for MCP pidfile identity helpers (#806).
 */

import { describe, test, expect } from "vitest";
import { looksLikeQmdMcpCommand, isQmdMcpPid, mcpDaemonStateFiles } from "../src/cli/mcp-pid.ts";

describe("looksLikeQmdMcpCommand", () => {
  test("matches bare qmd and common CLI script paths", () => {
    expect(looksLikeQmdMcpCommand("qmd mcp --http --port 8181")).toBe(true);
    expect(looksLikeQmdMcpCommand("/usr/local/bin/qmd mcp --http")).toBe(true);
    expect(looksLikeQmdMcpCommand("node /home/me/qmd/src/cli/qmd.ts mcp --http")).toBe(true);
    expect(looksLikeQmdMcpCommand("node /home/me/qmd/dist/cli/qmd.js mcp --http")).toBe(true);
    expect(looksLikeQmdMcpCommand("tsx src/cli/qmd.ts mcp --http --daemon")).toBe(true);
  });

  test("rejects empty / whitespace and unrelated processes", () => {
    expect(looksLikeQmdMcpCommand("")).toBe(false);
    expect(looksLikeQmdMcpCommand("   ")).toBe(false);
    expect(looksLikeQmdMcpCommand(
      "/System/Library/PrivateFrameworks/GenerativeExperiencesRuntime.framework/Versions/A/generativeexperiencesd",
    )).toBe(false);
    expect(looksLikeQmdMcpCommand("sleep 1000000")).toBe(false);
    expect(looksLikeQmdMcpCommand("node server.js")).toBe(false);
  });

  test("does not match qmd as a substring of another token", () => {
    expect(looksLikeQmdMcpCommand("myqmdtool serve")).toBe(false);
    expect(looksLikeQmdMcpCommand("qmdfoo")).toBe(false);
  });
});

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

describe("isQmdMcpPid", () => {
  test("returns false for invalid / dead PIDs", () => {
    expect(isQmdMcpPid(0)).toBe(false);
    expect(isQmdMcpPid(-1)).toBe(false);
    expect(isQmdMcpPid(1.5)).toBe(false);
    expect(isQmdMcpPid(999999999)).toBe(false);
  });

  test("returns true for the current process when it looks like qmd", () => {
    // Vitest/tsx argv typically includes the test file, not qmd — so this
    // process itself usually fails the cmdline check. Assert the live+match
    // path using our own PID only when argv happens to include qmd; otherwise
    // just confirm a clearly-alive non-qmd PID (self) returns false.
    const self = process.pid;
    const argvJoined = process.argv.join(" ");
    if (looksLikeQmdMcpCommand(argvJoined)) {
      expect(isQmdMcpPid(self)).toBe(true);
    } else {
      expect(isQmdMcpPid(self)).toBe(false);
    }
  });
});
