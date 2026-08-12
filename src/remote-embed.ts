/**
 * remote-embed.ts - Remote OpenAI-compatible embedding backend
 *
 * Ports the LM Studio/vLLM/TEI/OpenAI `/v1/embeddings` client used by the
 * sibling project `localcrab` (opencrab/stores/openai_embedding.py +
 * resilient_embedding.py) so qmd can route embedding generation to a
 * remote server instead of a local GGUF model.
 *
 * No node-llama-cpp import here: this module only uses the global `fetch`,
 * so it lazy-loads cleanly and is trivial to unit test without a GPU.
 */

/** A single remote embedding endpoint, parsed from a `http[s]://host/v1#model-id` URI. */
export interface RemoteEmbedEndpoint {
  /** The original URI as configured, e.g. `http://localhost:1234/v1#text-embedding-3-small`. */
  raw: string;
  /** Everything before the `#` fragment, e.g. `http://localhost:1234/v1`. `/embeddings` and `/models` are appended to this. */
  apiBase: string;
  /** The percent-decoded URL fragment, used as the `model` field in request bodies. */
  modelId: string;
}

/** Detect whether a model URI string should be routed to the remote embedding backend. */
export function isRemoteEmbedModel(uri: string): boolean {
  return uri.startsWith("http://") || uri.startsWith("https://");
}

/**
 * Parse a remote embedding URI into its API base and model id.
 * The model id is carried in the URL fragment (`#model-id`) since fragments
 * are never sent over the wire, keeping the base URL and model cleanly separated
 * while flowing through the existing single-string config field.
 * Throws if the URI has no fragment, since there is no way to infer the model id otherwise.
 */
export function parseRemoteEmbedUri(uri: string): RemoteEmbedEndpoint {
  const hashIndex = uri.indexOf("#");
  const fragment = hashIndex === -1 ? "" : uri.slice(hashIndex + 1);
  if (!fragment) {
    throw new Error(`remote embed URI requires a \`#model-id\` fragment: ${uri}`);
  }
  const apiBase = uri.slice(0, hashIndex);
  const modelId = decodeURIComponent(fragment);
  return { raw: uri, apiBase, modelId };
}

export const DEFAULT_REMOTE_EMBED_TIMEOUT_MS = 30000;
export const DEFAULT_REMOTE_EMBED_HEALTH_TTL_MS = 15000;
// Soft guardrail only: remote servers have no shared tokenizer for qmd to
// truncate against locally, so this is a generous char cap (chunks are
// already ~900 tokens / ~3600 chars) that just prevents pathological inputs
// from being sent, rather than an accurate token-aware limit.
const DEFAULT_MAX_CHARS = 12000;

export type RemoteEmbedderOptions = {
  /** Sent as `Authorization: Bearer <apiKey>` when set; omitted entirely otherwise (e.g. unauthenticated LM Studio). */
  apiKey?: string;
  /** Timeout in ms for `/embeddings` requests. Default 30000. */
  timeoutMs?: number;
  /** How long (ms) an endpoint is skipped after a failure, independently per endpoint. Default 15000. */
  healthTtlMs?: number;
  /** Soft per-text char cap applied before sending (guardrail only, see DEFAULT_MAX_CHARS). */
  maxChars?: number;
  /** L2-normalize returned vectors. Default true. */
  normalize?: boolean;
};

/** Thrown when every configured endpoint is unhealthy or failed for a given call. */
export class RemoteEmbedAllFailedError extends Error {}

type EmbeddingsResponse = {
  data: { embedding: number[] }[];
};

function l2Normalize(vector: number[]): number[] {
  let sumSquares = 0;
  for (const v of vector) sumSquares += v * v;
  const norm = Math.sqrt(sumSquares);
  if (norm === 0) return vector;
  return vector.map((v) => v / norm);
}

/**
 * Client for one or more OpenAI-compatible `/v1/embeddings` endpoints, with
 * sequential fallback and independent per-endpoint unhealthy TTLs (mirrors
 * localcrab's ResilientEmbeddingFunction).
 */
export class RemoteEmbedder {
  private readonly endpoints: RemoteEmbedEndpoint[];
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly healthTtlMs: number;
  private readonly maxChars: number;
  private readonly normalize: boolean;
  private readonly unhealthyUntil: number[];

  constructor(endpoints: RemoteEmbedEndpoint[], opts: RemoteEmbedderOptions = {}) {
    this.endpoints = endpoints;
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_REMOTE_EMBED_TIMEOUT_MS;
    this.healthTtlMs = opts.healthTtlMs ?? DEFAULT_REMOTE_EMBED_HEALTH_TTL_MS;
    this.maxChars = opts.maxChars ?? DEFAULT_MAX_CHARS;
    this.normalize = opts.normalize ?? true;
    this.unhealthyUntil = endpoints.map(() => 0);
  }

  private isHealthy(index: number): boolean {
    return Date.now() >= (this.unhealthyUntil[index] ?? 0);
  }

  private markUnhealthy(index: number): void {
    this.unhealthyUntil[index] = Date.now() + this.healthTtlMs;
  }

  private headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
    };
  }

  /**
   * Embed a batch of texts by trying each healthy endpoint in order.
   * The first endpoint to respond successfully wins; its response order is
   * expected to match the input order (one embedding object per input text).
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const inputs = texts.map((t) => (t.length > this.maxChars ? t.slice(0, this.maxChars) : t));

    for (let i = 0; i < this.endpoints.length; i++) {
      if (!this.isHealthy(i)) continue;
      const endpoint = this.endpoints[i]!;
      try {
        const resp = await fetch(`${endpoint.apiBase}/embeddings`, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify({ model: endpoint.modelId, input: inputs }),
          signal: AbortSignal.timeout(this.timeoutMs),
        });
        if (!resp.ok) {
          throw new Error(`remote embed endpoint ${endpoint.apiBase} returned HTTP ${resp.status}`);
        }
        const body = (await resp.json()) as EmbeddingsResponse;
        const vectors = body.data.map((item) => item.embedding);
        return this.normalize ? vectors.map(l2Normalize) : vectors;
      } catch (error) {
        console.warn(
          `QMD Warning: remote embed endpoint ${endpoint.apiBase} failed (${error instanceof Error ? error.message : String(error)}), marking unhealthy for ${this.healthTtlMs}ms.`
        );
        this.markUnhealthy(i);
      }
    }

    throw new RemoteEmbedAllFailedError("all remote embedding endpoints failed or unhealthy");
  }

  /** Convenience wrapper around embedBatch for a single text. */
  async embed(text: string): Promise<number[]> {
    const [result] = await this.embedBatch([text]);
    return result ?? [];
  }

  /** Health probe: GET `{apiBase}/models`, true only on HTTP 200, 5s timeout. */
  async ping(endpoint: RemoteEmbedEndpoint): Promise<boolean> {
    try {
      const resp = await fetch(`${endpoint.apiBase}/models`, {
        signal: AbortSignal.timeout(5000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }
}
