import { isIP } from "net";

export type NormalizedMcpHost = {
  bindHost: string;
  displayHost: string;
};

/**
 * Validate an MCP HTTP host argument from CLI input.
 * Host must be a hostname/IP literal, not a URL.
 */
export function validateMcpHostInput(rawHost: string): void {
  const value = rawHost.trim();
  if (value.includes("://") || value.includes("/")) {
    throw new Error(`Invalid --host value: "${rawHost}". Provide a hostname or IP address, not a URL.`);
  }
  if (value.startsWith("-")) {
    throw new Error(`Invalid --host value: "${rawHost}". "--host" requires a hostname or IP address argument.`);
  }
  // Reject host:port patterns (e.g. "localhost:8181", "[::1]:8080")
  // but allow bare IPv6 (e.g. "::1", "2001:db8::1") and bracketed IPv6 ("[::1]")
  const inner = value.startsWith("[") && value.endsWith("]")
    ? value.slice(1, -1).trim()
    : value;
  if (inner.includes(":") && isIP(inner) === 0) {
    throw new Error(
      `Invalid --host value: "${rawHost}". Use --host and --port separately (e.g., --host localhost --port 8181).`,
    );
  }
}

/**
 * Normalize host for socket binding and URL display.
 * - "[::1]" -> bind "::1", display "[::1]"
 * - empty/whitespace -> localhost
 */
export function normalizeMcpHost(rawHost?: string | null): NormalizedMcpHost {
  let host = typeof rawHost === "string" ? rawHost.trim() : "";

  if (host.startsWith("[") && host.endsWith("]")) {
    host = host.slice(1, -1).trim();
  }

  if (!host) host = "localhost";

  return {
    bindHost: host,
    displayHost: host.includes(":") ? `[${host}]` : host,
  };
}
