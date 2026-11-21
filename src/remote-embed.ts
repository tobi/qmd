/** OpenAI-compatible embeddings selected explicitly by an http[s] API-base#model URI. */
import { createHash } from "node:crypto";
import { postModelJson, REMOTE_MODEL_TIMEOUT_MS } from "./remote-http.js";

export type RemoteEmbedOptions = {
  dimensions?: number;
  revision?: string;
  /** Environment variable NAME, never a credential value. Absent means keyless. */
  apiKeyEnv?: string;
  /** Maximum number of inputs per HTTP request, from 1 to 64. */
  maxBatchSize?: number;
  /** Maximum sum of UTF-8 input text bytes per HTTP request. */
  maxBatchInputBytes?: number;
};

export function isRemoteEmbedModel(uri: string): boolean {
  return /^https?:/i.test(uri);
}

export function copyRemoteEmbedOptions(options: RemoteEmbedOptions): Readonly<RemoteEmbedOptions> {
  if (!options || typeof options !== "object" || Array.isArray(options)
    || Object.keys(options).some(key => !["dimensions", "revision", "apiKeyEnv", "maxBatchSize", "maxBatchInputBytes"].includes(key))
    || (options.maxBatchSize !== undefined && (!Number.isSafeInteger(options.maxBatchSize) || options.maxBatchSize < 1 || options.maxBatchSize > 64))
    || (options.maxBatchInputBytes !== undefined && (!Number.isSafeInteger(options.maxBatchInputBytes) || options.maxBatchInputBytes < 1))
    || (options.dimensions !== undefined && (!Number.isInteger(options.dimensions) || options.dimensions < 1 || options.dimensions > 4096))
    || (options.revision !== undefined && (typeof options.revision !== "string" || !options.revision.trim() || options.revision.length > 256 || /[\x00-\x1f\x7f]/.test(options.revision)))
    || (options.apiKeyEnv !== undefined && (typeof options.apiKeyEnv !== "string" || !/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(options.apiKeyEnv)))) {
    throw new Error("Invalid remote embedding options");
  }
  return Object.freeze({
    dimensions: options.dimensions,
    revision: options.revision,
    apiKeyEnv: options.apiKeyEnv,
    maxBatchSize: options.maxBatchSize,
    maxBatchInputBytes: options.maxBatchInputBytes,
  });
}

export function parseRemoteEmbedUri(uri: string): Readonly<{ apiBase: string; modelId: string }> {
  try {
    if (!/^https?:\/\//i.test(uri) || uri.length > 4096 || /[\s\x00-\x1f\x7f-\x9f\\]/.test(uri) || /%(?:0[0-9a-f]|1[0-9a-f]|7f)/i.test(uri)) throw new Error();
    if (uri.slice(uri.indexOf("://") + 3).split(/[/?#]/, 1)[0]!.includes("@")) throw new Error();
    const url = new URL(uri);
    if (!isRemoteEmbedModel(uri) || !url.hostname || url.username || url.password || url.search || uri.includes("?")) throw new Error();
    const modelId = decodeURIComponent(url.hash.slice(1));
    if (!modelId.trim() || modelId.length > 256 || /[\x00-\x1f\x7f]/.test(modelId)) throw new Error();
    url.hash = "";
    return Object.freeze({ apiBase: url.href.replace(/\/+$/, ""), modelId });
  } catch {
    throw new Error("Invalid remote embedding URI (expected http[s] API-base#model-id without credentials or query)");
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** No tokenizer, model discovery, normalization, fallback, or implicit auth. */
export class RemoteEmbedder {
  private readonly endpoint: ReturnType<typeof parseRemoteEmbedUri>;
  private readonly options: Readonly<RemoteEmbedOptions>;
  readonly modelName: string;

  constructor(uri: string, options: RemoteEmbedOptions = {}) {
    this.endpoint = parseRemoteEmbedUri(uri);
    this.options = copyRemoteEmbedOptions(options);
    // Full opaque space id: same model name on two deployments is NOT the same
    // vector space. Credentials never enter it; revision pins deployment changes.
    const identity = JSON.stringify([
      this.endpoint.apiBase, this.endpoint.modelId,
      this.options.dimensions ?? "server-default", this.options.revision ?? "",
      "raw-query/title-newline-document/char3600-overlap540/v1",
    ]);
    this.modelName = `remote-openai:${createHash("sha256").update(identity).digest("hex")}`;
  }

  inputBudgetError(text: string, index = 0): string | undefined {
    const limit = this.options.maxBatchInputBytes;
    if (limit === undefined) return;
    const size = Buffer.byteLength(text, "utf8");
    if (size > limit) {
      return `Input ${index + 1} uses ${size} UTF-8 bytes, exceeding maxBatchInputBytes (${limit})`;
    }
  }

  async embedBatch(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    signal?.throwIfAborted();
    if (!Array.isArray(texts) || texts.length > 64 || texts.some(text => typeof text !== "string" || !text.length || !text.isWellFormed() || Buffer.byteLength(text, "utf8") > 64 * 1024)) {
      throw new Error("Invalid remote embedding inputs (max 64 inputs, 64 KiB each)");
    }
    const maxInputs = this.options.maxBatchSize ?? 64;
    const maxBytes = this.options.maxBatchInputBytes ?? Infinity;
    const batches: string[][] = [];
    let batch: string[] = [];
    let bytes = 0;
    // Validate and snapshot the entire logical batch before sending any input.
    for (const [index, text] of texts.entries()) {
      const size = Buffer.byteLength(text, "utf8");
      if (size > maxBytes) throw new Error(this.inputBudgetError(text, index));
      if (batch.length === maxInputs || bytes + size > maxBytes) {
        batches.push(batch);
        batch = [];
        bytes = 0;
      }
      batch.push(text);
      bytes += size;
    }
    if (batch.length) batches.push(batch);

    if (batches.length > 1) {
      const deadline = AbortSignal.timeout(REMOTE_MODEL_TIMEOUT_MS);
      signal = signal ? AbortSignal.any([signal, deadline]) : deadline;
    }
    const vectors: number[][] = [];
    let dimension = this.options.dimensions;
    for (const inputs of batches) {
      signal?.throwIfAborted();
      const result = await this.requestEmbeddings(inputs, dimension, signal);
      dimension ??= result[0]!.length;
      vectors.push(...result);
    }
    return vectors;
  }

  private async requestEmbeddings(texts: string[], dimension: number | undefined, signal?: AbortSignal): Promise<number[][]> {
    const count = texts.length;
    const body = await postModelJson(`${this.endpoint.apiBase}/embeddings`, {
      model: this.endpoint.modelId,
      input: texts,
      encoding_format: "float",
      ...(this.options.dimensions !== undefined ? { dimensions: this.options.dimensions } : {}),
    }, this.options.apiKeyEnv, signal);
    const invalid = () => new Error("Invalid remote embedding response");
    if (!record(body) || body.model !== this.endpoint.modelId || !Array.isArray(body.data) || body.data.length !== count) throw invalid();
    const vectors: number[][] = new Array(count);
    for (const item of body.data) {
      if (!record(item) || typeof item.index !== "number" || !Number.isInteger(item.index) || item.index < 0 || item.index >= count || vectors[item.index] !== undefined || !Array.isArray(item.embedding)) throw invalid();
      const vector: unknown[] = item.embedding;
      dimension ??= vector.length;
      if (!dimension || dimension > 4096 || vector.length !== dimension
        || vector.some(value => typeof value !== "number" || !Number.isFinite(value) || !Number.isFinite(Math.fround(value)))
        || !vector.some(value => Math.fround(value as number) !== 0)) throw invalid();
      vectors[item.index] = [...vector] as number[];
    }
    signal?.throwIfAborted();
    return vectors;
  }
}
