/**
 * MCP daemon pidfile identity helpers.
 *
 * Pidfiles alone are unsafe after PID reuse (e.g. post-reboot). Callers must
 * confirm a recorded PID still belongs to a qmd process before signalling it
 * or treating it as "already running".
 */

import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

export type QmdProcessRole = "mcp-http" | "embed";

export type QmdProcessIdentity = {
  format: "qmd-process/v1";
  pid: number;
  role: QmdProcessRole;
  startToken: string;
  port?: number;
};

export type ProcessIdentityProbe = {
  isAlive: (pid: number) => boolean;
  startToken: (pid: number) => string | null;
  cmdline?: (pid: number) => string | null;
};

export type ProcessIdentityStatus = "live" | "dead" | "unknown";

export type QmdProcessRecord =
  | { kind: "identity"; identity: QmdProcessIdentity }
  | { kind: "legacy"; pid: number }
  | { kind: "invalid" };

/** Parse a structured QMD process identity file. Legacy numeric pidfiles are intentionally untrusted. */
export function parseProcessIdentity(raw: string): QmdProcessIdentity | null {
  try {
    const value = JSON.parse(raw) as Partial<QmdProcessIdentity>;
    if (value.format !== "qmd-process/v1") return null;
    if (!Number.isInteger(value.pid) || (value.pid ?? 0) <= 0) return null;
    if (value.role !== "mcp-http" && value.role !== "embed") return null;
    if (typeof value.startToken !== "string" || value.startToken.length === 0) return null;
    if (value.port !== undefined && (!Number.isInteger(value.port) || value.port <= 0 || value.port > 65535)) return null;
    return value as QmdProcessIdentity;
  } catch {
    return null;
  }
}

export function readProcessIdentity(path: string): QmdProcessIdentity | null {
  try {
    return parseProcessIdentity(readFileSync(path, "utf-8"));
  } catch {
    return null;
  }
}

/** Parse either the current JSON format or the numeric format written by QMD <= 2.8.3. */
export function parseProcessRecord(raw: string): QmdProcessRecord {
  const identity = parseProcessIdentity(raw);
  if (identity) return { kind: "identity", identity };
  const trimmed = raw.trim();
  if (/^\d+$/.test(trimmed)) {
    const pid = Number(trimmed);
    if (Number.isSafeInteger(pid) && pid > 0) return { kind: "legacy", pid };
  }
  return { kind: "invalid" };
}

export function readProcessRecord(path: string): QmdProcessRecord {
  try {
    return parseProcessRecord(readFileSync(path, "utf-8"));
  } catch {
    return { kind: "invalid" };
  }
}

export function processRecordPid(record: QmdProcessRecord): number | null {
  if (record.kind === "identity") return record.identity.pid;
  if (record.kind === "legacy") return record.pid;
  return null;
}

export function serializeProcessIdentity(identity: QmdProcessIdentity): string {
  return `${JSON.stringify(identity)}\n`;
}

export function sameProcessIdentity(
  left: QmdProcessIdentity | null,
  right: QmdProcessIdentity | null,
): boolean {
  return left !== null && right !== null
    && left.pid === right.pid
    && left.role === right.role
    && left.startToken === right.startToken;
}

export function sameProcessRecord(left: QmdProcessRecord, right: QmdProcessRecord): boolean {
  if (left.kind === "identity" && right.kind === "identity") {
    return sameProcessIdentity(left.identity, right.identity);
  }
  if (left.kind === "legacy" && right.kind === "legacy") return left.pid === right.pid;
  // Invalid records are never authoritative enough to delete.
  return false;
}

let cachedCurrentProcessStartToken: string | undefined;

function cacheCurrentProcessStartToken(pid: number, token: string | null): string | null {
  if (pid === process.pid && token) cachedCurrentProcessStartToken = token;
  return token;
}

/**
 * Return an OS-issued process-start token. Combining this with PID prevents a
 * recycled PID from inheriting authority to a daemon state file or embed lock.
 */
export function readProcessStartToken(pid: number): string | null {
  if (!Number.isInteger(pid) || pid <= 0) return null;
  if (pid === process.pid && cachedCurrentProcessStartToken) return cachedCurrentProcessStartToken;

  if (process.platform === "linux") {
    try {
      const stat = readFileSync(`/proc/${pid}/stat`, "utf-8");
      const afterName = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
      const startTime = afterName[19]; // proc(5) field 22; array starts at field 3.
      return cacheCurrentProcessStartToken(pid, startTime ? `linux:${startTime}` : null);
    } catch {
      return null;
    }
  }

  if (process.platform === "win32") {
    try {
      const script = [
        "$ErrorActionPreference='Stop'",
        `$process = Get-CimInstance Win32_Process -Filter "ProcessId = ${pid}"`,
        "if ($null -eq $process) { exit 3 }",
        "$process.CreationDate.ToUniversalTime().Ticks",
      ].join("\n");
      const output = execFileSync("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], {
        encoding: "utf-8",
        // PowerShell 5.1 cold starts can be slow under Windows Defender. This
        // probe is a safety boundary, so tolerate startup contention instead
        // of turning a transient two-second delay into an unverifiable daemon.
        timeout: 10_000,
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
      }).trim();
      return cacheCurrentProcessStartToken(pid, /^\d+$/.test(output) ? `win32:${output}` : null);
    } catch {
      return null;
    }
  }

  try {
    const output = execFileSync("ps", ["-p", String(pid), "-o", "lstart="], {
      encoding: "utf-8",
      env: { ...process.env, LC_ALL: "C", TZ: "UTC" },
      timeout: 2000,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return cacheCurrentProcessStartToken(pid, output ? `${process.platform}:${output}` : null);
  } catch {
    return null;
  }
}

export function createProcessIdentity(
  role: QmdProcessRole,
  options: { pid?: number; port?: number } = {},
): QmdProcessIdentity {
  const pid = options.pid ?? process.pid;
  const startToken = readProcessStartToken(pid);
  if (!startToken) {
    throw new Error(`Cannot read process start identity for PID ${pid}`);
  }
  return {
    format: "qmd-process/v1",
    pid,
    role,
    startToken,
    ...(options.port === undefined ? {} : { port: options.port }),
  };
}

export function createProcessRecord(
  role: QmdProcessRole,
  options: { pid?: number; port?: number } = {},
  identityFactory: typeof createProcessIdentity = createProcessIdentity,
): QmdProcessRecord {
  const pid = options.pid ?? process.pid;
  try {
    return { kind: "identity", identity: identityFactory(role, options) };
  } catch {
    return { kind: "legacy", pid };
  }
}

export function serializeProcessRecord(record: QmdProcessRecord): string {
  return record.kind === "identity"
    ? serializeProcessIdentity(record.identity)
    : record.kind === "legacy"
      ? `${record.pid}\n`
      : "";
}

const defaultIdentityProbe: ProcessIdentityProbe = {
  isAlive(pid) {
    try {
      process.kill(pid, 0);
      return true;
    } catch (error: unknown) {
      // EPERM means the process exists but cannot be signalled from this token.
      return typeof error === "object" && error !== null && "code" in error
        && (error as NodeJS.ErrnoException).code === "EPERM";
    }
  },
  startToken: readProcessStartToken,
  cmdline: readProcessCmdline,
};

/** True if a command line belongs to the expected QMD process role. */
export function looksLikeQmdRoleCommand(cmdline: string, role: QmdProcessRole): boolean {
  const normalized = cmdline.trim();
  if (!/(?:^|[\s/\\])qmd(?:\.(?:ts|js))?(?:[\s]|$)/i.test(normalized)) return false;
  return role === "embed"
    ? /(?:^|\s)embed(?:\s|$)/i.test(normalized)
    : /(?:^|\s)mcp(?:\s|$)/i.test(normalized) && /(?:^|\s)--http(?:\s|$)/i.test(normalized);
}

/** Read a process command line from procfs or the POSIX `ps` interface. */
export function readProcessCmdline(pid: number): string | null {
  if (!Number.isInteger(pid) || pid <= 0 || process.platform === "win32") return null;
  const procPath = `/proc/${pid}/cmdline`;
  if (existsSync(procPath)) {
    try {
      const value = readFileSync(procPath, "utf-8").replace(/\0/g, " ").trim();
      if (value) return value;
    } catch {
      // Fall through to ps.
    }
  }
  try {
    const value = execFileSync("ps", ["-p", String(pid), "-o", "args="], {
      encoding: "utf-8",
      env: { ...process.env, LC_ALL: "C", TZ: "UTC" },
      timeout: 2000,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return value || null;
  } catch {
    return null;
  }
}

export function isProcessAlive(pid: number, probe: ProcessIdentityProbe = defaultIdentityProbe): boolean {
  return Number.isInteger(pid) && pid > 0 && probe.isAlive(pid);
}

export function processIdentityStatus(
  identity: QmdProcessIdentity | null,
  expectedRole: QmdProcessRole,
  probe: ProcessIdentityProbe = defaultIdentityProbe,
): ProcessIdentityStatus {
  if (!identity || identity.role !== expectedRole) return "dead";
  if (!probe.isAlive(identity.pid)) return "dead";
  const startToken = probe.startToken(identity.pid);
  if (!startToken) return "unknown";
  return startToken === identity.startToken ? "live" : "dead";
}

export function isProcessIdentityLive(
  identity: QmdProcessIdentity | null,
  expectedRole: QmdProcessRole,
  probe: ProcessIdentityProbe = defaultIdentityProbe,
): boolean {
  return processIdentityStatus(identity, expectedRole, probe) === "live";
}

/** Legacy live PIDs and malformed/in-flight records are deliberately unverifiable. */
export function processRecordStatus(
  record: QmdProcessRecord,
  expectedRole: QmdProcessRole,
  probe: ProcessIdentityProbe = defaultIdentityProbe,
): ProcessIdentityStatus {
  if (record.kind === "identity") return processIdentityStatus(record.identity, expectedRole, probe);
  if (record.kind === "legacy") {
    if (!isProcessAlive(record.pid, probe)) return "dead";
    const cmdline = probe.cmdline?.(record.pid);
    if (cmdline === undefined || cmdline === null) return "unknown";
    return looksLikeQmdRoleCommand(cmdline, expectedRole) ? "live" : "unknown";
  }
  return "unknown";
}

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
