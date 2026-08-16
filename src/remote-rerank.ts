import type { RerankDocument, RerankResult } from "./llm.js";

export interface OpenAIRerankConfig {
  provider: "openai";
  endpoint: string;
  model: string;
  apiKey?: string;
  modelAliases?: string[];
  timeoutMs?: number;
  failurePolicy?: "fail-closed";
}

export type RerankConfig = string | OpenAIRerankConfig;
export type ResolvedRerankConfig =
  | { kind: "local"; model: string }
  | ({ kind: "remote" } & OpenAIRerankConfig);

export class RemoteRerankError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}
export class RemoteRerankTransportError extends RemoteRerankError {}
export class RemoteRerankProtocolError extends RemoteRerankError {}

function rerankEndpoint(endpoint: string): string {
  if (typeof endpoint !== "string" || endpoint.length === 0) throw new RemoteRerankProtocolError("remote rerank endpoint must be a non-empty string");
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new RemoteRerankProtocolError("remote rerank endpoint must be a valid URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new RemoteRerankProtocolError("remote rerank endpoint must use HTTP or HTTPS");
  if (url.username || url.password || url.search || url.hash) throw new RemoteRerankProtocolError("remote rerank endpoint must not contain credentials, query, or fragment");
  const pathname = url.pathname.replace(/\/+$/, "");
  if (pathname === "/v1") url.pathname = "/v1/rerank";
  else if (pathname !== "/v1/rerank") throw new RemoteRerankProtocolError("remote rerank endpoint path must be /v1 or /v1/rerank");
  return url.toString();
}

export function resolveRerankConfig(config: RerankConfig): ResolvedRerankConfig {
  if (typeof config === "string") return { kind: "local", model: config };
  if (!config || typeof config !== "object" || Array.isArray(config)) throw new RemoteRerankProtocolError("remote rerank configuration must be an object");
  const raw = config as unknown as Record<string, unknown>;
  if (raw.provider !== "openai") throw new RemoteRerankProtocolError("unsupported remote rerank provider");
  if (typeof raw.model !== "string" || raw.model.trim().length === 0 || raw.model !== raw.model.trim()) throw new RemoteRerankProtocolError("remote rerank model must be a canonical non-empty string");
  if (raw.apiKey !== undefined && (typeof raw.apiKey !== "string" || raw.apiKey.trim().length === 0 || raw.apiKey !== raw.apiKey.trim())) throw new RemoteRerankProtocolError("remote rerank apiKey must be a canonical non-empty string");
  if (raw.timeoutMs !== undefined && (typeof raw.timeoutMs !== "number" || !Number.isFinite(raw.timeoutMs) || !Number.isInteger(raw.timeoutMs) || raw.timeoutMs <= 0)) {
    throw new RemoteRerankProtocolError("remote rerank timeoutMs must be a positive integer");
  }
  if (raw.modelAliases !== undefined) {
    if (!Array.isArray(raw.modelAliases) || raw.modelAliases.some(alias => typeof alias !== "string")) {
      throw new RemoteRerankProtocolError("remote rerank modelAliases must contain only canonical non-empty strings");
    }
    const aliases = raw.modelAliases as string[];
    const trimmed = aliases.map(alias => alias.trim());
    if (aliases.some((alias, index) => trimmed[index]!.length === 0 || alias !== trimmed[index]) || new Set(trimmed).size !== trimmed.length || trimmed.includes(raw.model as string)) {
      throw new RemoteRerankProtocolError("remote rerank modelAliases must be unique canonical aliases distinct from the primary model");
    }
  }
  if (raw.failurePolicy !== undefined && raw.failurePolicy !== "fail-closed") {
    throw new RemoteRerankProtocolError("remote rerank failurePolicy must be fail-closed");
  }
  return { ...config, kind: "remote", endpoint: rerankEndpoint(config.endpoint) };
}

export function getRerankCacheIdentity(config: ResolvedRerankConfig): string {
  if (config.kind === "local") return config.model;
  return JSON.stringify({
    provider: config.provider,
    endpoint: config.endpoint,
    model: config.model,
    modelAliases: [...(config.modelAliases ?? [])].sort(),
    scoreContract: "normalized-relevance-score-v1",
    failurePolicy: config.failurePolicy ?? "fail-closed",
  });
}

export class OpenAIRerankClient {
  readonly failurePolicy = "fail-closed" as const;
  private readonly config: OpenAIRerankConfig & { endpoint: string };

  constructor(config: OpenAIRerankConfig) {
    const resolved = resolveRerankConfig(config);
    if (resolved.kind !== "remote") throw new RemoteRerankProtocolError("remote rerank configuration required");
    this.config = resolved;
  }

  async rerank(query: string, documents: RerankDocument[]): Promise<RerankResult> {
    if (documents.length === 0) return { model: this.config.model, results: [] };
    let response: Response;
    try {
      response = await fetch(this.config.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
        },
        body: JSON.stringify({ model: this.config.model, query, documents: documents.map(document => document.text) }),
        signal: AbortSignal.timeout(this.config.timeoutMs ?? 60_000),
      });
    } catch {
      throw new RemoteRerankTransportError("remote rerank transport failed");
    }
    if (!response.ok) throw new RemoteRerankTransportError(`remote rerank request returned HTTP ${response.status}`);
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new RemoteRerankProtocolError("remote rerank response was not valid JSON");
    }
    if (!body || typeof body !== "object") throw new RemoteRerankProtocolError("remote rerank response must be an object");
    const record = body as Record<string, unknown>;
    const allowedModels = new Set([this.config.model, ...(this.config.modelAliases ?? [])]);
    if (typeof record.model !== "string" || !allowedModels.has(record.model)) {
      throw new RemoteRerankProtocolError("remote rerank response model does not match configured model identity");
    }
    if (!Array.isArray(record.results) || record.results.length !== documents.length) {
      throw new RemoteRerankProtocolError("remote rerank response cardinality does not match documents");
    }
    const seen = new Set<number>();
    const results = record.results.map((rawResult) => {
      if (!rawResult || typeof rawResult !== "object") throw new RemoteRerankProtocolError("remote rerank result must be an object");
      const result = rawResult as Record<string, unknown>;
      const index = result.index;
      if (!Number.isInteger(index) || (index as number) < 0 || (index as number) >= documents.length || seen.has(index as number)) {
        throw new RemoteRerankProtocolError("remote rerank response has invalid index coverage");
      }
      const score = result.relevance_score;
      if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1) {
        throw new RemoteRerankProtocolError("remote rerank relevance_score must be finite and normalized to [0,1]");
      }
      seen.add(index as number);
      return { file: documents[index as number]!.file, index: index as number, score };
    });
    if (seen.size !== documents.length) throw new RemoteRerankProtocolError("remote rerank response does not cover every document exactly once");
    return { model: record.model, results };
  }
}
