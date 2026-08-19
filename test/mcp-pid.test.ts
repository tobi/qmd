/**
 * Unit tests for MCP pidfile identity helpers (#806).
 */

import { describe, test, expect } from "vitest";
import { looksLikeQmdMcpCommand, isQmdMcpPid, mcpDaemonStateFiles, stopQmdMcpProcess } from "../src/cli/mcp-pid.ts";

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

describe("stopQmdMcpProcess", () => {
  test("keeps waiting while the PID remains owned and does not escalate before grace", async () => {
    let now = 0;
    const signals: NodeJS.Signals[] = [];
    let owned = true;
    const outcome = stopQmdMcpProcess(4242, {
      gracefulMs: 100,
      now: () => now,
      isOwnedPid: () => owned,
      kill: (_pid, signal) => { signals.push(signal); },
      sleep: async (ms) => { now += ms; },
    });
    await Promise.resolve();
    expect(signals).toEqual(["SIGTERM"]);
    owned = false;
    await expect(outcome).resolves.toBe("graceful");
    expect(signals).toEqual(["SIGTERM"]);
  });

  test("returns graceful without SIGKILL when the PID disappears during grace", async () => {
    let now = 0;
    let owned = true;
    const signals: NodeJS.Signals[] = [];
    const result = await stopQmdMcpProcess(7, {
      gracefulMs: 80,
      now: () => now,
      isOwnedPid: () => owned,
      kill: (_pid, signal) => { signals.push(signal); },
      sleep: async (ms) => {
        now += ms;
        if (now >= 50) owned = false;
      },
    });
    expect(result).toBe("graceful");
    expect(signals).toEqual(["SIGTERM"]);
  });

  test("rechecks identity before escalating and sends SIGKILL only after grace", async () => {
    let now = 0;
    const signals: NodeJS.Signals[] = [];
    let owned = true;
    const result = stopQmdMcpProcess(9, {
      gracefulMs: 50,
      killWaitMs: 50,
      now: () => now,
      isOwnedPid: () => owned,
      kill: (_pid, signal) => {
        signals.push(signal);
        if (signal === "SIGKILL") owned = false;
      },
      sleep: async (ms) => { now += ms; },
    });
    await expect(result).resolves.toBe("forced");
    expect(signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  test("does not SIGKILL a reused PID", async () => {
    let now = 0;
    const signals: NodeJS.Signals[] = [];
    let checks = 0;
    const result = await stopQmdMcpProcess(11, {
      gracefulMs: 50,
      now: () => now,
      isOwnedPid: () => {
        checks += 1;
        return checks === 1;
      },
      kill: (_pid, signal) => { signals.push(signal); },
      sleep: async (ms) => { now += ms; },
    });
    expect(result).toBe("graceful");
    expect(signals).toEqual(["SIGTERM"]);
  });

  test("throws and does not claim success when the process survives SIGKILL", async () => {
    let now = 0;
    await expect(stopQmdMcpProcess(13, {
      gracefulMs: 20,
      killWaitMs: 20,
      now: () => now,
      isOwnedPid: () => true,
      kill: () => undefined,
      sleep: async (ms) => { now += ms; },
    })).rejects.toThrow("survived SIGKILL");
  });

  test("treats ESRCH on SIGTERM as already-gone", async () => {
    const err = Object.assign(new Error("No such process"), { code: "ESRCH" });
    await expect(stopQmdMcpProcess(21, {
      isOwnedPid: () => true,
      kill: () => { throw err; },
    })).resolves.toBe("already-gone");
  });

  test("treats ESRCH on SIGKILL as forced after grace", async () => {
    let now = 0;
    const err = Object.assign(new Error("No such process"), { code: "ESRCH" });
    await expect(stopQmdMcpProcess(22, {
      gracefulMs: 20,
      now: () => now,
      isOwnedPid: () => true,
      kill: (_pid, signal) => {
        if (signal === "SIGKILL") throw err;
      },
      sleep: async (ms) => { now += ms; },
    })).resolves.toBe("forced");
  });

  test("does not hide EPERM from kill", async () => {
    const err = Object.assign(new Error("Operation not permitted"), { code: "EPERM" });
    await expect(stopQmdMcpProcess(23, {
      isOwnedPid: () => true,
      kill: () => { throw err; },
    })).rejects.toMatchObject({ code: "EPERM" });
  });
});
