/**
 * Daemon discovery and HTTP client for the qmd MCP HTTP daemon.
 *
 * The CLI uses this to short-circuit local model loading when a daemon
 * is already running — cutting `qmd query` from ~15s cold to ~3s.
 *
 * Falls back silently to null on any failure — the caller should use
 * the in-process path in that case.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import type { HybridQueryResult, ExpandedQuery, ChunkStrategy } from "./store.js";

/**
 * Default port the daemon listens on. Used only when mcp.port file is
 * missing (legacy fallback). Can be overridden via QMD_DEFAULT_PORT env
 * var — primarily useful for tests that can't use the real 8181.
 */
export function getDefaultPort(): number {
  const env = process.env.QMD_DEFAULT_PORT;
  if (env) {
    const n = parseInt(env, 10);
    if (Number.isInteger(n) && n > 0 && n <= 65535) return n;
  }
  return 8181;
}

const DEFAULT_TIMEOUT_MS = 300;

export interface DaemonHandle {
  baseUrl: string;  // http://localhost:<port>
  pid: number;
  port: number;
  dbPath: string;   // absolute path to the DB the daemon is serving (from /health)
}

export interface DiscoverOptions {
  timeoutMs?: number;
}

function getCacheDir(): string {
  return process.env.XDG_CACHE_HOME
    ? resolve(process.env.XDG_CACHE_HOME, "qmd")
    : resolve(homedir(), ".cache", "qmd");
}

export function getPidFilePath(): string {
  return resolve(getCacheDir(), "mcp.pid");
}

export function getPortFilePath(): string {
  return resolve(getCacheDir(), "mcp.port");
}

/** Read the port file. Returns null if missing, malformed, or 0. */
export async function readPortFile(): Promise<number | null> {
  const path = getPortFilePath();
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf-8").trim();
    const n = parseInt(raw, 10);
    if (!Number.isInteger(n) || n <= 0 || n > 65535) return null;
    return n;
  } catch {
    return null;
  }
}

/** Read the PID file. Returns null if missing or malformed. */
function readPidFile(): number | null {
  const path = getPidFilePath();
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf-8").trim();
    const n = parseInt(raw, 10);
    if (!Number.isInteger(n) || n <= 0) return null;
    return n;
  } catch {
    return null;
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

interface HealthBody {
  status: string;
  dbPath?: string;
}

async function fetchHealth(url: string, timeoutMs: number): Promise<HealthBody | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) return null;
    const body = await res.json().catch(() => null) as HealthBody | null;
    if (!body || body.status !== "ok" || typeof body.dbPath !== "string") return null;
    return body;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function probeHealth(baseUrl: string, timeoutMs: number): Promise<HealthBody | null> {
  const primary = await fetchHealth(`${baseUrl}/health`, timeoutMs);
  if (primary) return primary;

  const url = new URL(baseUrl);
  if (url.hostname !== "localhost") return null;

  return fetchHealth(`http://127.0.0.1:${url.port}/health`, timeoutMs);
}

/**
 * Discover a running daemon. Returns null if:
 *   - No PID file
 *   - Process is dead
 *   - /health fails within timeoutMs
 *   - /health response omits dbPath (old daemon version pre-dating this
 *     plan). Old daemons can still be used via the MCP protocol; they
 *     just can't satisfy the DB-match safety check, so the CLI falls
 *     back to in-process.
 *
 * Port resolution order:
 *   1. `mcp.port` file (written by `qmd mcp --http [--daemon]`)
 *   2. `getDefaultPort()` (defaults to 8181)
 *
 * NOTE: when `mcp.port` is missing and something else is listening on
 * the default port, `discoverDaemon` may return a handle pointing at
 * the wrong server. The DB-match check in `maybeDiscoverDaemon`
 * (Task 4) protects against this — the unrelated server won't match
 * the CLI's dbPath.
 */
export async function discoverDaemon(options: DiscoverOptions = {}): Promise<DaemonHandle | null> {
  if (process.env.QMD_NO_DAEMON === "1") return null;

  const pid = readPidFile();
  if (pid === null) return null;
  if (!isProcessAlive(pid)) return null;

  const port = (await readPortFile()) ?? getDefaultPort();
  const baseUrl = `http://localhost:${port}`;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const health = await probeHealth(baseUrl, timeoutMs);
  if (!health || typeof health.dbPath !== "string") return null;

  return { baseUrl, pid, port, dbPath: health.dbPath };
}

const DEFAULT_CALL_TIMEOUT_MS = 60_000;

export interface SearchViaDaemonOptions {
  /** Hybrid search — server performs expansion + search + rerank. */
  query?: string;
  /** Structured search — caller provides pre-expanded sub-queries. */
  searches?: ExpandedQuery[];
  limit?: number;
  minScore?: number;
  candidateLimit?: number;
  skipRerank?: boolean;
  explain?: boolean;
  intent?: string;
  collections?: string[];
  chunkStrategy?: ChunkStrategy;
  /** Override call timeout (default 60s). */
  timeoutMs?: number;
}

function debugDaemonSearch(message: string): void {
  if (process.env.QMD_DEBUG === "1") {
    process.stderr.write(`[qmd daemon] ${message}\n`);
  }
}

/**
 * POST /v1/search on the daemon.
 *
 * Returns HybridQueryResult[] on success, or null on any error (including
 * network failure, timeout, 4xx, 5xx, or malformed JSON). Callers should
 * fall back to the in-process pipeline when null is returned.
 *
 * This function never throws.
 */
export async function searchViaDaemon(
  baseUrl: string,
  options: SearchViaDaemonOptions,
): Promise<HybridQueryResult[] | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_CALL_TIMEOUT_MS);

  const body: Record<string, unknown> = {};
  if (options.query !== undefined) body.query = options.query;
  if (options.searches !== undefined) body.searches = options.searches;
  if (options.limit !== undefined) body.limit = options.limit;
  if (options.minScore !== undefined) body.minScore = options.minScore;
  if (options.candidateLimit !== undefined) body.candidateLimit = options.candidateLimit;
  if (options.skipRerank !== undefined) body.skipRerank = options.skipRerank;
  if (options.explain !== undefined) body.explain = options.explain;
  if (options.intent !== undefined) body.intent = options.intent;
  if (options.collections !== undefined) body.collections = options.collections;
  if (options.chunkStrategy !== undefined) body.chunkStrategy = options.chunkStrategy;

  try {
    const res = await fetch(`${baseUrl}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      debugDaemonSearch(`/v1/search -> ${res.status}`);
      return null;
    }

    const json = await res.json() as { results?: unknown };
    if (!Array.isArray(json.results)) {
      debugDaemonSearch("/v1/search returned malformed JSON payload");
      return null;
    }

    return json.results as HybridQueryResult[];
  } catch (err) {
    debugDaemonSearch(
      `/v1/search failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    return null;
  } finally {
    clearTimeout(timer);
  }
}
