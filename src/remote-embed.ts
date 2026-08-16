/** Strict OpenAI-compatible remote embedding transport. */

export const REMOTE_EMBEDDING_NATIVE_DIMENSIONS = 2560;
export const REMOTE_EMBEDDING_OUTPUT_DIMENSIONS = 1024;
export const REMOTE_EMBEDDING_REDUCTION = "mrl-prefix" as const;
export const REMOTE_EMBEDDING_FORMAT_VERSION = "qmd-remote-embedding-v1";

export type RemoteEmbeddingNormalization = "l2";
export type RemoteEmbeddingReduction = typeof REMOTE_EMBEDDING_REDUCTION;

export interface OpenAIEmbeddingConfig {
  provider: "openai";
  endpoint: string;
  model: string;
  nativeDimensions: number;
  dimensions: number;
  reduction: RemoteEmbeddingReduction;
  apiKey?: string;
  modelAliases?: string[];
  normalization: RemoteEmbeddingNormalization;
  formatVersion?: string;
  timeoutMs?: number;
}

export type EmbeddingConfig = string | OpenAIEmbeddingConfig;
export type ResolvedEmbeddingConfig =
  | { kind: "local"; model: string }
  | ({ kind: "remote" } & Omit<OpenAIEmbeddingConfig, "provider"> & { provider: "openai"; formatVersion: string });

export class RemoteEmbeddingError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}
export class RemoteEmbeddingTransportError extends RemoteEmbeddingError {}
export class RemoteEmbeddingProtocolError extends RemoteEmbeddingError {}

function embeddingsEndpoint(endpoint: string): string {
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new RemoteEmbeddingProtocolError("remote embedding endpoint must be a valid HTTP(S) /v1 URL");
  }
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password || url.search || url.hash) {
    throw new RemoteEmbeddingProtocolError("remote embedding endpoint must be a credential-free HTTP(S) /v1 URL");
  }
  const pathname = url.pathname.replace(/\/+$/, "");
  if (pathname === "/v1") url.pathname = "/v1/embeddings";
  else if (pathname === "/v1/embeddings") url.pathname = pathname;
  else throw new RemoteEmbeddingProtocolError("remote embedding endpoint path must be /v1 or /v1/embeddings");
  return url.toString();
}

export function resolveEmbeddingConfig(config: EmbeddingConfig): ResolvedEmbeddingConfig {
  if (typeof config === "string") return { kind: "local", model: config };
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new RemoteEmbeddingProtocolError("remote embedding configuration must be an object");
  }
  if (config.provider !== "openai") throw new RemoteEmbeddingProtocolError("unsupported remote embedding provider");
  if (typeof config.endpoint !== "string" || !config.endpoint.trim() || typeof config.model !== "string" || !config.model.trim()) {
    throw new RemoteEmbeddingProtocolError("remote embedding endpoint and model must be non-empty strings");
  }
  if (config.nativeDimensions !== REMOTE_EMBEDDING_NATIVE_DIMENSIONS) {
    throw new RemoteEmbeddingProtocolError(`remote embedding nativeDimensions must be exactly ${REMOTE_EMBEDDING_NATIVE_DIMENSIONS}`);
  }
  if (config.dimensions !== REMOTE_EMBEDDING_OUTPUT_DIMENSIONS || config.dimensions >= config.nativeDimensions) {
    throw new RemoteEmbeddingProtocolError(`remote embedding dimensions must be exactly ${REMOTE_EMBEDDING_OUTPUT_DIMENSIONS} and less than nativeDimensions`);
  }
  if (config.reduction !== REMOTE_EMBEDDING_REDUCTION) {
    throw new RemoteEmbeddingProtocolError(`remote embedding reduction must be ${REMOTE_EMBEDDING_REDUCTION}`);
  }
  if (config.normalization !== "l2") {
    throw new RemoteEmbeddingProtocolError("remote embedding normalization must be l2");
  }
  if (config.formatVersion !== undefined && (typeof config.formatVersion !== "string" || !config.formatVersion.trim())) {
    throw new RemoteEmbeddingProtocolError("remote embedding formatVersion must be a non-empty string");
  }
  if (config.timeoutMs !== undefined && (!Number.isInteger(config.timeoutMs) || config.timeoutMs <= 0)) {
    throw new RemoteEmbeddingProtocolError("remote embedding timeoutMs must be a positive integer");
  }
  if (config.apiKey !== undefined && (typeof config.apiKey !== "string" || config.apiKey.length === 0)) {
    throw new RemoteEmbeddingProtocolError("remote embedding apiKey must be a non-empty string");
  }
  if (config.modelAliases !== undefined) {
    if (!Array.isArray(config.modelAliases) || config.modelAliases.some(alias => typeof alias !== "string" || !alias.trim())) {
      throw new RemoteEmbeddingProtocolError("remote embedding modelAliases must contain only non-empty strings");
    }
    if (new Set(config.modelAliases).size !== config.modelAliases.length) {
      throw new RemoteEmbeddingProtocolError("remote embedding modelAliases must be unique");
    }
  }
  return {
    kind: "remote",
    ...config,
    endpoint: embeddingsEndpoint(config.endpoint),
    formatVersion: config.formatVersion ?? REMOTE_EMBEDDING_FORMAT_VERSION,
  };
}

function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!Number.isFinite(norm) || norm === 0) throw new RemoteEmbeddingProtocolError("remote embedding vector has zero or invalid norm");
  const normalized = vector.map(value => value / norm);
  if (!normalized.every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))) {
    throw new RemoteEmbeddingProtocolError("remote embedding normalized vector must contain only finite Float32 values");
  }
  return normalized;
}

export class OpenAIEmbeddingClient {
  private readonly config: ReturnType<typeof resolveEmbeddingConfig> & { kind: "remote" };

  constructor(config: OpenAIEmbeddingConfig) {
    const resolved = resolveEmbeddingConfig(config);
    if (resolved.kind !== "remote") throw new RemoteEmbeddingProtocolError("remote embedding configuration required");
    this.config = resolved;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    let response: Response;
    try {
      response = await fetch(this.config.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
        },
        body: JSON.stringify({ model: this.config.model, input: texts }),
        signal: AbortSignal.timeout(this.config.timeoutMs ?? 30_000),
      });
    } catch {
      throw new RemoteEmbeddingTransportError("remote embedding request failed");
    }
    if (!response.ok) {
      throw new RemoteEmbeddingTransportError(`remote embedding request returned HTTP ${response.status}`);
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new RemoteEmbeddingProtocolError("remote embedding response was not valid JSON");
    }
    return this.validate(body, texts.length);
  }

  async embed(text: string): Promise<number[]> {
    return (await this.embedBatch([text]))[0]!;
  }

  private validate(body: unknown, count: number): number[][] {
    if (!body || typeof body !== "object") throw new RemoteEmbeddingProtocolError("remote embedding response must be an object");
    const record = body as Record<string, unknown>;
    const allowed = new Set([this.config.model, ...(this.config.modelAliases ?? [])]);
    if (typeof record.model !== "string" || !allowed.has(record.model)) {
      throw new RemoteEmbeddingProtocolError("remote embedding response model does not match configured model identity");
    }
    if (!Array.isArray(record.data) || record.data.length !== count) {
      throw new RemoteEmbeddingProtocolError("remote embedding response cardinality does not match input");
    }

    const output: number[][] = new Array(count);
    const seen = new Set<number>();
    for (const rawItem of record.data) {
      if (!rawItem || typeof rawItem !== "object") throw new RemoteEmbeddingProtocolError("remote embedding item must be an object");
      const item = rawItem as Record<string, unknown>;
      const index = item.index;
      if (!Number.isInteger(index) || (index as number) < 0 || (index as number) >= count || seen.has(index as number)) {
        throw new RemoteEmbeddingProtocolError("remote embedding response has invalid index coverage");
      }
      const embedding = item.embedding;
      if (!Array.isArray(embedding) || embedding.length === 0 || embedding.length !== this.config.nativeDimensions) {
        throw new RemoteEmbeddingProtocolError("remote embedding vector has invalid dimensions");
      }
      if (!embedding.every(value => typeof value === "number" && Number.isFinite(value) && Number.isFinite(Math.fround(value)))) {
        throw new RemoteEmbeddingProtocolError("remote embedding vector must contain only finite Float32 values");
      }
      seen.add(index as number);
      const prefix = (embedding as number[]).slice(0, this.config.dimensions);
      output[index as number] = normalize(prefix);
    }
    if (seen.size !== count || output.some(value => value === undefined)) {
      throw new RemoteEmbeddingProtocolError("remote embedding response does not cover every input index exactly once");
    }
    return output;
  }
}
