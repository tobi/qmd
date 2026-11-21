/**
 * Bounded JSON POST shared by remote model clients. Callers validate their URL
 * and response schema. One attempt, no redirects, no response/error text escapes.
 * Limits apply to encoded request bytes and streamed response bytes, not tokens.
 */
const MAX_REQUEST_BYTES = 2 * 1024 * 1024;
const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
export const REMOTE_MODEL_TIMEOUT_MS = 30_000;

export async function postModelJson(
  url: string,
  payload: unknown,
  apiKeyEnv?: string,
  callerSignal?: AbortSignal,
): Promise<unknown> {
  const body = JSON.stringify(payload);
  if (Buffer.byteLength(body, "utf8") > MAX_REQUEST_BYTES) {
    throw new Error("Remote model request exceeds byte limit");
  }
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKeyEnv !== undefined) {
    const key = process.env[apiKeyEnv];
    if (!key || !/^[\x21-\x7e]+$/.test(key)) {
      throw new Error("Remote model credential unavailable");
    }
    headers.Authorization = `Bearer ${key}`;
  }
  const controller = new AbortController();
  const signal = callerSignal
    ? AbortSignal.any([callerSignal, controller.signal])
    : controller.signal;
  const timer = setTimeout(() => controller.abort(), REMOTE_MODEL_TIMEOUT_MS);
  timer.unref();
  let cancelBody: (() => Promise<void>) | undefined;
  let status: number | undefined;
  try {
    signal.throwIfAborted();
    const response = await fetch(url, { method: "POST", headers, body, signal, redirect: "error" });
    signal.throwIfAborted();
    if (!response.ok) {
      if (Number.isInteger(response.status) && response.status >= 100 && response.status <= 599) status = response.status;
      throw new Error();
    }
    const length = response.headers.get("content-length");
    if (length !== null && (!/^\d+$/.test(length) || Number(length) > MAX_RESPONSE_BYTES)) throw new Error();
    if (!response.body) throw new Error();
    const reader = response.body.getReader();
    cancelBody = () => reader.cancel();
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MAX_RESPONSE_BYTES) throw new Error();
      chunks.push(value);
    }
    const result: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks, bytes)));
    signal.throwIfAborted();
    return result;
  } catch {
    if (signal.aborted) throw new Error("Remote model request aborted");
    throw new Error(status === undefined ? "Remote model request failed" : `Remote model request failed (HTTP ${status})`);
  } finally {
    clearTimeout(timer);
    // Cancel unread data on failure, including rejected Content-Length headers.
    if (cancelBody) await cancelBody().catch(() => {});
    controller.abort();
  }
}
