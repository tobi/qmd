/**
 * origin-guard.ts — DNS-rebinding protection for the HTTP MCP transport.
 *
 * Binding to localhost keeps *remote* clients out, but it does not keep a web
 * page out. A page served from `attacker.example` can re-point that hostname at
 * `127.0.0.1` after load (DNS rebinding); the browser then treats
 * `http://attacker.example:8181` as same-origin with the page and hands the
 * response body to attacker JavaScript. The only server-side signal that
 * distinguishes such a request from a legitimate local client is the `Origin`
 * and `Host` headers, so the MCP spec requires local HTTP servers to validate
 * them.
 *
 * The predicates here are pure so they can be unit-tested without a socket;
 * `startMcpHttpServer` applies them to every request before routing.
 */

/** Wildcard bind addresses — the request's legitimate `Host` is unknowable. */
const WILDCARD_BIND_HOSTS = new Set(["", "0.0.0.0", "::", "[::]", "*"]);

/**
 * Hostnames that resolve to this machine's loopback interface.
 *
 * `0.0.0.0` is deliberately excluded: browsers happily route it to loopback on
 * Linux and macOS, which makes it a rebinding-free path to local servers, so an
 * `Origin: http://0.0.0.0:8181` is treated as untrusted like any other.
 */
export function isLoopbackHostname(hostname: string): boolean {
  const h = hostname.trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (h === "localhost" || h.endsWith(".localhost")) return true;
  if (h === "::1" || h === "0:0:0:0:0:0:0:1") return true;
  // IPv4-mapped IPv6 (::ffff:127.0.0.1) and the whole 127.0.0.0/8 range.
  const v4 = h.startsWith("::ffff:") ? h.slice(7) : h;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(v4);
}

/** Parse an `Origin` header into its hostname, or undefined if unusable. */
function originHostname(origin: string): string | undefined {
  try {
    const url = new URL(origin.trim());
    if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
    return url.hostname;
  } catch {
    return undefined;
  }
}

/** Normalize an origin for comparison against the allowlist. */
function normalizeOrigin(origin: string): string | undefined {
  try {
    return new URL(origin.trim()).origin.toLowerCase();
  } catch {
    return undefined;
  }
}

/** Parse a `Host` header (`name`, `name:port`, `[::1]:port`) into a hostname. */
function hostHeaderHostname(host: string): string | undefined {
  try {
    return new URL(`http://${host.trim()}`).hostname;
  } catch {
    return undefined;
  }
}

export type OriginGuard = {
  /** All checks disabled (`QMD_ALLOWED_ORIGINS=*`). */
  disabled: boolean;
  /** Extra origins accepted beyond the loopback defaults, normalized. */
  allowedOrigins: string[];
  /** Extra `Host` values accepted beyond the loopback defaults, lowercased. */
  allowedHosts: string[];
  /** Whether the `Host` header is checked at all — see `resolveOriginGuard`. */
  enforceHost: boolean;
};

function splitList(value: string | undefined): string[] {
  if (!value) return [];
  return value.split(",").map(v => v.trim()).filter(v => v.length > 0);
}

/**
 * Build the guard for a server bound to `host`.
 *
 * `Host` validation is enforced whenever the bind address names a concrete
 * interface, since the legitimate `Host` header is then known (loopback, or the
 * bind address itself). For a wildcard bind — `--host 0.0.0.0`, e.g. Docker —
 * any of the container's names may legitimately appear, so the check is only
 * enforced when the operator supplies an explicit allowlist. `Origin`
 * validation always applies; browsers are the threat, and they always send it.
 */
export function resolveOriginGuard(options: {
  host: string;
  allowedOrigins?: string[];
  allowedHosts?: string[];
  env?: NodeJS.ProcessEnv;
}): OriginGuard {
  const env = options.env ?? process.env;
  const rawOrigins = options.allowedOrigins ?? splitList(env.QMD_ALLOWED_ORIGINS);
  const rawHosts = options.allowedHosts ?? splitList(env.QMD_ALLOWED_HOSTS);

  if (rawOrigins.includes("*")) {
    return { disabled: true, allowedOrigins: [], allowedHosts: [], enforceHost: false };
  }

  const allowedOrigins = rawOrigins
    .map(o => normalizeOrigin(o))
    .filter((o): o is string => o !== undefined);
  const allowedHosts = rawHosts.map(h => h.toLowerCase());

  const bindHost = options.host.trim().toLowerCase();
  const isWildcardBind = WILDCARD_BIND_HOSTS.has(bindHost);
  if (!isWildcardBind && !isLoopbackHostname(bindHost)) {
    // Explicit non-loopback interface: its own address is a legitimate Host.
    allowedHosts.push(bindHost);
  }

  return {
    disabled: false,
    allowedOrigins,
    allowedHosts,
    enforceHost: !isWildcardBind || allowedHosts.length > 0,
  };
}

export type GuardVerdict = { ok: true } | { ok: false; reason: string };

/**
 * Validate a request's `Origin` and `Host` headers against the guard.
 *
 * A missing `Origin` is allowed: non-browser clients (curl, the MCP SDK's HTTP
 * client, editors) omit it, and browsers cannot. The header is the signal that
 * a request came from page JavaScript, so its *absence* is not suspicious while
 * a foreign *value* is conclusive.
 */
export function checkRequestOrigin(
  headers: { origin?: string | undefined; host?: string | undefined },
  guard: OriginGuard,
): GuardVerdict {
  if (guard.disabled) return { ok: true };

  const origin = headers.origin?.trim();
  if (origin) {
    const normalized = normalizeOrigin(origin);
    const hostname = originHostname(origin);
    const allowed =
      (hostname !== undefined && isLoopbackHostname(hostname)) ||
      (normalized !== undefined && guard.allowedOrigins.includes(normalized));
    if (!allowed) {
      return { ok: false, reason: `Origin not allowed: ${origin}` };
    }
  }

  const host = headers.host?.trim();
  if (guard.enforceHost && host) {
    const lowered = host.toLowerCase();
    const hostname = hostHeaderHostname(host);
    const allowed =
      (hostname !== undefined && isLoopbackHostname(hostname)) ||
      guard.allowedHosts.includes(lowered) ||
      (hostname !== undefined && guard.allowedHosts.includes(hostname.toLowerCase()));
    if (!allowed) {
      return { ok: false, reason: `Host not allowed: ${host}` };
    }
  }

  return { ok: true };
}
