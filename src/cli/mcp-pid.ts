/**
 * MCP daemon pidfile identity helpers.
 *
 * Pidfiles alone are unsafe after PID reuse (e.g. post-reboot). Callers must
 * confirm a recorded PID still belongs to a qmd process before signalling it
 * or treating it as "already running".
 */

import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

/**
 * Pid/log filenames for the MCP HTTP daemon.
 * The default index keeps `mcp.pid` / `mcp.log` for compatibility; named
 * indexes are scoped so a named daemon can run alongside the default (#772).
 */
export function mcpDaemonStateFiles(indexName: string = "index"): { pidFile: string; logFile: string } {
  const suffix = !indexName || indexName === "index" ? "" : `-${indexName}`;
  return {
    pidFile: `mcp${suffix}.pid`,
    logFile: `mcp${suffix}.log`,
  };
}

/** True if a process command line looks like a qmd CLI invocation. */
export function looksLikeQmdMcpCommand(cmdline: string): boolean {
  const s = cmdline.trim();
  if (!s) return false;
  // Match bare `qmd`, `qmd.ts`/`qmd.js`, or a path ending in /qmd(.ts|.js).
  // Windows quotes a path containing spaces, so a closing quote can end the token.
  return /(?:^|[\s/\\])qmd(?:\.(?:ts|js))?"?(?:\s|$)/i.test(s);
}

/** Read process cmdline (Linux /proc preferred; ps fallback for macOS; CIM on Windows). */
export function readProcessCmdline(pid: number): string | null {
  if (!Number.isInteger(pid) || pid <= 0) return null;

  if (process.platform === "win32") {
    // No /proc, and a ps on PATH (e.g. Git Bash) does not know Windows PIDs (#908).
    // Use the absolute path: by default Windows also looks for a bare name in the
    // working directory, which may be an untrusted checkout.
    const systemRoot = process.env.SystemRoot ?? process.env.WINDIR;
    if (!systemRoot) return null;
    try {
      const cmdline = execFileSync(
        join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
        ["-NoProfile", "-NonInteractive", "-Command", `(Get-CimInstance Win32_Process -Filter ProcessId=${pid}).CommandLine`],
        { encoding: "utf-8", timeout: 10000, windowsHide: true, stdio: ["ignore", "pipe", "ignore"] },
      );
      return cmdline.trim() || null;
    } catch {
      return null;
    }
  }

  const procPath = `/proc/${pid}/cmdline`;
  if (existsSync(procPath)) {
    try {
      const raw = readFileSync(procPath, "utf-8");
      const cmdline = raw.replace(/\0/g, " ").trim();
      if (cmdline) return cmdline;
    } catch {
      // fall through to ps
    }
  }

  try {
    let cmdline = "";
    try {
      cmdline = execFileSync("ps", ["-p", String(pid), "-o", "args="], {
        encoding: "utf-8",
        timeout: 2000,
        stdio: ["ignore", "pipe", "ignore"],
      });
    } catch {
      cmdline = execFileSync("ps", ["-p", String(pid), "-o", "command="], {
        encoding: "utf-8",
        timeout: 2000,
        stdio: ["ignore", "pipe", "ignore"],
      });
    }
    const trimmed = cmdline.trim();
    return trimmed || null;
  } catch {
    return null;
  }
}

/**
 * Returns true only if `pid` is alive AND its command line looks like qmd.
 * If cmdline cannot be read or does not match, returns false (treat as stale).
 */
export function isQmdMcpPid(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;

  try {
    process.kill(pid, 0);
  } catch {
    return false;
  }

  const cmdline = readProcessCmdline(pid);
  if (!cmdline) return false;
  return looksLikeQmdMcpCommand(cmdline);
}
