/**
 * transport.ts — Low-level HTTP primitives for remote API calls.
 *
 * Uses Node's built-in http/https modules instead of fetch()/undici to
 * avoid the Node v24 undici ByteString bug that crashes on Unicode response
 * bodies (e.g. U+2026 HORIZONTAL ELLIPSIS from OpenRouter).
 *
 * Both POST and GET cap response bodies at 10MB to prevent memory exhaustion.
 *
 * @module remote/transport
 */

import * as http from 'http';
import * as https from 'https';
import { URL } from 'url';

// =============================================================================
// Constants
// =============================================================================

/** Max response body size (10 MB) to prevent memory exhaustion from large payloads. */
export const MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024;
const HTTP_KEEP_ALIVE_AGENT = new http.Agent({ keepAlive: true });
const HTTPS_KEEP_ALIVE_AGENT = new https.Agent({ keepAlive: true });

type JsonRequestMethod = 'GET' | 'POST';

function getKeepAliveAgent(url: URL): http.Agent | https.Agent {
  return url.protocol === 'https:' ? HTTPS_KEEP_ALIVE_AGENT : HTTP_KEEP_ALIVE_AGENT;
}

export function buildBearerHeaders(
  apiKey?: string,
  headers: Record<string, string> = {},
): Record<string, string> {
  const trimmedApiKey = apiKey?.trim();
  if (!trimmedApiKey) return { ...headers };
  return {
    ...headers,
    Authorization: `Bearer ${trimmedApiKey}`,
  };
}

async function requestJson(
  method: JsonRequestMethod,
  urlStr: string,
  headers: Record<string, string>,
  timeoutMs: number,
  body?: unknown,
): Promise<any> {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const bodyBuf = body === undefined
      ? null
      : Buffer.from(JSON.stringify(body), 'utf-8');
    const requestHeaders: Record<string, string> = bodyBuf === null
      ? { ...headers }
      : {
          ...headers,
          'Content-Type': 'application/json',
          'Content-Length': String(bodyBuf.length),
        };

    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request(
      {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method,
        headers: requestHeaders,
        agent: getKeepAliveAgent(url),
      },
      (res) => {
        const chunks: Buffer[] = [];
        let totalBytes = 0;

        res.on('data', (chunk: Buffer) => {
          const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          totalBytes += buf.length;
          if (totalBytes > MAX_RESPONSE_BODY_BYTES) {
            req.destroy(new Error(
              `Response body exceeded ${MAX_RESPONSE_BODY_BYTES} byte limit (got ${totalBytes}+)`
            ));
            return;
          }
          chunks.push(buf);
        });

        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          const status = res.statusCode ?? 0;
          if (status < 200 || status >= 300) {
            reject(new Error(`HTTP ${status}: ${raw.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (e) {
            reject(new Error(
              `JSON parse failed: ${(e as Error).message} — body: ${raw.slice(0, 200)}`
            ));
          }
        });

        res.on('error', reject);
      },
    );

    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    if (bodyBuf !== null) {
      req.write(bodyBuf);
    }
    req.end();
  });
}

// =============================================================================
// nodePost
// =============================================================================

/**
 * Low-level HTTP/HTTPS POST using Node's built-in http/https modules.
 *
 * ## Why not fetch()?
 *
 * Node v24's undici-based fetch() has a ByteString bug that crashes when
 * response bodies contain Unicode code points > U+00FF (e.g. U+2026
 * HORIZONTAL ELLIPSIS, commonly returned by OpenRouter). Using raw
 * http/https with Buffer concatenation avoids this entirely.
 *
 * ## Behavior
 *
 * - Sends JSON body with Content-Type: application/json
 * - Parses response as JSON and returns the parsed object
 * - Caps response body at 10MB (throws if exceeded)
 * - Throws on non-2xx status codes or JSON parse failures
 *
 * @param urlStr    - Full URL to POST to
 * @param headers   - Request headers (Authorization, etc.)
 * @param body      - JSON-serializable request body
 * @param timeoutMs - Request timeout in ms (default 30000)
 * @returns Parsed JSON response body
 * @throws Error on HTTP errors, timeouts, JSON parse failures, or oversized responses
 */
export async function nodePost(
  urlStr: string,
  headers: Record<string, string>,
  body: unknown,
  timeoutMs = 30000,
): Promise<any> {
  return requestJson('POST', urlStr, headers, timeoutMs, body);
}

// =============================================================================
// nodeGet
// =============================================================================

/**
 * Low-level HTTP/HTTPS GET using Node's built-in http/https modules.
 *
 * Used for /models endpoint health checks. Same safety properties as
 * nodePost(): Buffer-based response handling, 10MB body cap, timeout.
 *
 * @param urlStr    - Full URL to GET
 * @param headers   - Request headers
 * @param timeoutMs - Request timeout in ms (default 5000)
 * @returns Parsed JSON response body
 */
export async function nodeGet(
  urlStr: string,
  headers: Record<string, string>,
  timeoutMs = 5000,
): Promise<any> {
  return requestJson('GET', urlStr, headers, timeoutMs);
}
