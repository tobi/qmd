import {
  isQwen3EmbeddingModel,
  type EmbedOptions,
  type EmbeddingResult,
  type GenerateOptions,
  type GenerateResult,
  type LLM,
  type ModelInfo,
  type Queryable,
  type RerankDocument,
  type RerankOptions,
  type RerankResult,
} from "./llm.js";

const DEFAULT_CONNECT_TIMEOUT_MS = 500;
const DEFAULT_READ_TIMEOUT_MS = 10_000;
const DEFAULT_BREAKER_THRESHOLD = 2;
const DEFAULT_BREAKER_COOLDOWN_MS = 10 * 60 * 1000;
const QWEN_QUERY_PREFIX = "Instruct: Retrieve relevant documents for the given query\nQuery: ";
const LEGACY_QUERY_PREFIX = "task: search result | query: ";
const LEGACY_DOC_PREFIX = /^title:\s*(.*?)\s*\|\s*text:\s*([\s\S]*)$/;
const COUNTED_FAILURE = Symbol("remoteFailureCounted");

type EndpointName = "embed" | "rerank";
type BreakerStatus = "closed" | "open" | "half-open";

type BreakerState = {
  failures: number;
  openUntil: number;
  state: BreakerStatus;
};

export type RemoteLLMConfig = {
  embedUrl?: string;
  rerankUrl?: string;
  embedModel?: string;
  rerankModel?: string;
  apiKey?: string;
  connectTimeoutMs?: number;
  readTimeoutMs?: number;
  breakerThreshold?: number;
  breakerCooldownMs?: number;
};

function nowIso(): string {
  return new Date().toISOString();
}

function logStderr(message: string): void {
  process.stderr.write(`[${nowIso()}] ${message}\n`);
}

function parseTimeout(value: number | string | undefined, fallback: number, label: string): number {
  if (value === undefined || value === "") {
    return fallback;
  }

  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
  return parsed;
}

function parsePositiveInt(value: number | undefined, fallback: number, label: string): number {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
  return value;
}

function joinEndpoint(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/+$/, "")}${path}`;
}

function clipForLog(value: string, max = 300): string {
  return value.length <= max ? value : `${value.slice(0, max)}...`;
}

function markFailureCounted(error: unknown): unknown {
  if (error && typeof error === "object") {
    Reflect.set(error, COUNTED_FAILURE, true);
  }
  return error;
}

function wasFailureCounted(error: unknown): boolean {
  return !!(error && typeof error === "object" && Reflect.get(error, COUNTED_FAILURE));
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timer: NodeJS.Timeout | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => {
          reject(new Error(`${label} timeout after ${timeoutMs}ms`));
        }, timeoutMs);
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function extractOriginalEmbeddingText(text: string): { kind: "query" | "document" | "unknown"; text: string } {
  if (text.startsWith(QWEN_QUERY_PREFIX)) {
    return { kind: "query", text: text.slice(QWEN_QUERY_PREFIX.length) };
  }
  if (text.startsWith(LEGACY_QUERY_PREFIX)) {
    return { kind: "query", text: text.slice(LEGACY_QUERY_PREFIX.length) };
  }

  const docMatch = LEGACY_DOC_PREFIX.exec(text);
  if (docMatch) {
    const [, title, body] = docMatch;
    return { kind: "document", text: title ? `${title}\n${body ?? ""}` : (body ?? "") };
  }

  return { kind: "unknown", text };
}

export class RemoteLLM implements LLM {
  readonly isRemote = true;
  private readonly embedUrl: string;
  private readonly rerankUrl: string;
  private readonly embedModel: string;
  private readonly rerankModel: string;
  private readonly apiKey: string;
  private readonly connectTimeoutMs: number;
  private readonly readTimeoutMs: number;
  private readonly breakerThreshold: number;
  private readonly breakerCooldownMs: number;
  private readonly breakerState: Record<EndpointName, BreakerState>;
  private expectedEmbedDim: number | null = null;

  constructor(config: RemoteLLMConfig = {}) {
    const embedUrl = config.embedUrl ?? process.env.QMD_REMOTE_EMBED_URL;
    const rerankUrl = config.rerankUrl ?? process.env.QMD_REMOTE_RERANK_URL;

    if (!embedUrl) throw new Error("QMD_REMOTE_EMBED_URL is required for RemoteLLM");
    if (!rerankUrl) throw new Error("QMD_REMOTE_RERANK_URL is required for RemoteLLM");

    this.embedUrl = String(embedUrl).replace(/\/+$/, "");
    this.rerankUrl = String(rerankUrl).replace(/\/+$/, "");
    this.embedModel =
      config.embedModel ??
      process.env.QMD_REMOTE_EMBED_MODEL ??
      process.env.QMD_EMBED_MODEL ??
      "remote-embedding";
    this.rerankModel =
      config.rerankModel ??
      process.env.QMD_REMOTE_RERANK_MODEL ??
      process.env.QMD_RERANK_MODEL ??
      "remote-reranker";
    this.apiKey = config.apiKey ?? process.env.QMD_REMOTE_API_KEY ?? "";
    this.connectTimeoutMs = parseTimeout(
      config.connectTimeoutMs ?? process.env.QMD_REMOTE_CONNECT_TIMEOUT,
      DEFAULT_CONNECT_TIMEOUT_MS,
      "QMD_REMOTE_CONNECT_TIMEOUT",
    );
    this.readTimeoutMs = parseTimeout(
      config.readTimeoutMs ?? process.env.QMD_REMOTE_READ_TIMEOUT,
      DEFAULT_READ_TIMEOUT_MS,
      "QMD_REMOTE_READ_TIMEOUT",
    );
    this.breakerThreshold = parsePositiveInt(
      config.breakerThreshold,
      DEFAULT_BREAKER_THRESHOLD,
      "breakerThreshold",
    );
    this.breakerCooldownMs = parsePositiveInt(
      config.breakerCooldownMs,
      DEFAULT_BREAKER_COOLDOWN_MS,
      "breakerCooldownMs",
    );
    this.breakerState = {
      embed: { failures: 0, openUntil: 0, state: "closed" },
      rerank: { failures: 0, openUntil: 0, state: "closed" },
    };
  }

  private normalizeEmbeddingInput(text: string, options: EmbedOptions = {}): string {
    const extracted = extractOriginalEmbeddingText(text);
    const kind = options.isQuery ? "query" : extracted.kind;
    const plainText = extracted.text;
    const embedModelHint = options.model ?? this.embedModel;

    if (kind === "query" && isQwen3EmbeddingModel(embedModelHint)) {
      return `${QWEN_QUERY_PREFIX}${plainText}`;
    }

    return plainText;
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      accept: "application/json",
    };
    if (this.apiKey) {
      headers.authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private beforeRequest(endpoint: EndpointName): void {
    const state = this.breakerState[endpoint];
    const now = Date.now();

    if (state.openUntil > now) {
      const retryAt = new Date(state.openUntil).toISOString();
      logStderr(`RemoteLLM ${endpoint} circuit open until ${retryAt}; skipping request`);
      throw markFailureCounted(new Error(`Remote ${endpoint} endpoint is in circuit-breaker cooldown until ${retryAt}`));
    }

    if (state.state === "open" && state.openUntil !== 0 && state.openUntil <= now) {
      state.state = "half-open";
      state.openUntil = 0;
      logStderr(`RemoteLLM ${endpoint} circuit cooldown elapsed; retrying endpoint`);
    }
  }

  private onSuccess(endpoint: EndpointName): void {
    const state = this.breakerState[endpoint];
    const previousState = state.state;
    const hadFailures = state.failures > 0;
    state.failures = 0;
    state.openUntil = 0;
    state.state = "closed";

    if (previousState !== "closed" || hadFailures) {
      logStderr(`RemoteLLM ${endpoint} circuit closed after successful request`);
    }
  }

  private onFailure(endpoint: EndpointName, error: unknown): void {
    const state = this.breakerState[endpoint];
    state.failures += 1;

    if (state.failures >= this.breakerThreshold) {
      state.state = "open";
      state.openUntil = Date.now() + this.breakerCooldownMs;
      logStderr(
        `RemoteLLM ${endpoint} circuit opened after ${state.failures} consecutive failures; cooldown ${Math.round(this.breakerCooldownMs / 60000)}m`,
      );
    } else {
      state.state = "closed";
    }

    logStderr(`RemoteLLM ${endpoint} error: ${error instanceof Error ? error.message : String(error)}`);
  }

  private async postJson(endpoint: EndpointName, url: string, payload: unknown): Promise<any> {
    this.beforeRequest(endpoint);

    const connectController = new AbortController();
    const connectTimer = setTimeout(() => {
      connectController.abort(new Error(`connect timeout after ${this.connectTimeoutMs}ms`));
    }, this.connectTimeoutMs);
    connectTimer.unref?.();

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: this.buildHeaders(),
        body: JSON.stringify(payload),
        signal: connectController.signal,
      });
    } catch (error) {
      clearTimeout(connectTimer);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }
    clearTimeout(connectTimer);

    let responseText: string;
    try {
      responseText = await withTimeout(response.text(), this.readTimeoutMs, "read");
    } catch (error) {
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }

    if (!response.ok) {
      const error = new Error(`HTTP ${response.status} from ${url}: ${clipForLog(responseText)}`);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }

    try {
      return responseText ? JSON.parse(responseText) : {};
    } catch {
      const error = new Error(`Invalid JSON from ${url}: ${clipForLog(responseText)}`);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }
  }

  private async fetchModels(endpoint: EndpointName, baseUrl: string): Promise<string[]> {
    this.beforeRequest(endpoint);

    const connectController = new AbortController();
    const connectTimer = setTimeout(() => {
      connectController.abort(new Error(`connect timeout after ${this.connectTimeoutMs}ms`));
    }, this.connectTimeoutMs);
    connectTimer.unref?.();

    let response: Response;
    try {
      response = await fetch(joinEndpoint(baseUrl, "/v1/models"), {
        method: "GET",
        headers: this.buildHeaders(),
        signal: connectController.signal,
      });
    } catch (error) {
      clearTimeout(connectTimer);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }
    clearTimeout(connectTimer);

    let body: string;
    try {
      body = await withTimeout(response.text(), this.readTimeoutMs, "read");
    } catch (error) {
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }

    if (!response.ok) {
      const error = new Error(`HTTP ${response.status} from ${baseUrl}/v1/models: ${clipForLog(body)}`);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }

    try {
      const parsed = body ? JSON.parse(body) : {};
      const models = Array.isArray(parsed?.data) ? parsed.data : [];
      this.onSuccess(endpoint);
      return models
        .map((model) => (typeof model?.id === "string" ? model.id : null))
        .filter((value): value is string => value !== null);
    } catch {
      const error = new Error(`Invalid JSON from ${baseUrl}/v1/models: ${clipForLog(body)}`);
      this.onFailure(endpoint, error);
      throw markFailureCounted(error);
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const [result] = await this.embedBatch([text], options);
    return result ?? null;
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const input = texts.map((text) => this.normalizeEmbeddingInput(text, options));
    const url = joinEndpoint(this.embedUrl, "/v1/embeddings");

    try {
      const data = await this.postJson("embed", url, {
        input,
        model: options.model ?? this.embedModel,
      });

      if (!Array.isArray(data?.data)) {
        throw new Error(`Invalid embedding response from ${url}: missing data array`);
      }

      const sorted = [...data.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      if (sorted.length !== texts.length) {
        throw new Error(`Invalid embedding response from ${url}: expected ${texts.length} vectors, got ${sorted.length}`);
      }

      const results = sorted.map((item) => {
        if (!Array.isArray(item?.embedding)) {
          throw new Error(`Invalid embedding response item from ${url}: missing embedding array`);
        }

        const dim = item.embedding.length;
        if (this.expectedEmbedDim === null) {
          this.expectedEmbedDim = dim;
          logStderr(`RemoteLLM embed dimension locked: ${dim}`);
        } else if (dim !== this.expectedEmbedDim) {
          throw new Error(
            `Embedding dimension mismatch: expected ${this.expectedEmbedDim}, got ${dim}. Model or server config may have changed.`,
          );
        }

        return {
          embedding: item.embedding,
          model: data.model ?? options.model ?? this.embedModel,
        } satisfies EmbeddingResult;
      });

      this.onSuccess("embed");
      return results;
    } catch (error) {
      if (!wasFailureCounted(error)) {
        this.onFailure("embed", error);
      }
      throw error;
    }
  }

  async generate(_prompt: string, _options: GenerateOptions = {}): Promise<GenerateResult | null> {
    throw new Error("RemoteLLM does not implement generate(); use HybridLLM with a local LlamaCpp");
  }

  async expandQuery(_query: string, _options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    throw new Error("RemoteLLM does not implement expandQuery(); use HybridLLM with a local LlamaCpp");
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: options.model ?? this.rerankModel };
    }

    const url = joinEndpoint(this.rerankUrl, "/v1/rerank");
    try {
      const data = await this.postJson("rerank", url, {
        query,
        documents: documents.map((document) => document.text),
        model: options.model ?? this.rerankModel,
        return_documents: false,
      });

      const results = Array.isArray(data?.results) ? data.results : Array.isArray(data?.data) ? data.data : null;
      if (!results) {
        throw new Error(`Invalid rerank response from ${url}: missing results array`);
      }

      const normalized = results.map((item) => {
        const index = item.index ?? item.document_index;
        const score = item.relevance_score ?? item.score;
        if (!Number.isInteger(index) || index < 0 || index >= documents.length) {
          throw new Error(`Invalid rerank response item from ${url}: bad index ${String(index)}`);
        }
        if (typeof score !== "number" || Number.isNaN(score)) {
          throw new Error(`Invalid rerank response item from ${url}: bad score ${String(score)}`);
        }
        const document = documents[index];
        if (!document) {
          throw new Error(`Invalid rerank response item from ${url}: missing document for index ${String(index)}`);
        }
        return {
          file: document.file,
          index,
          score,
        };
      });

      normalized.sort((a, b) => b.score - a.score);
      this.onSuccess("rerank");
      return {
        results: normalized,
        model: data.model ?? options.model ?? this.rerankModel,
      };
    } catch (error) {
      if (!wasFailureCounted(error)) {
        this.onFailure("rerank", error);
      }
      throw error;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const [embedModels, rerankModels] = await Promise.allSettled([
        this.fetchModels("embed", this.embedUrl),
        this.fetchModels("rerank", this.rerankUrl),
      ]);

      const available = new Set<string>();
      if (embedModels.status === "fulfilled") {
        for (const name of embedModels.value) available.add(name);
      }
      if (rerankModels.status === "fulfilled") {
        for (const name of rerankModels.value) available.add(name);
      }

      if (available.size > 0) {
        return { name: model, exists: available.has(model) };
      }
    } catch {
      // Fall through to optimistic default below.
    }

    return { name: model, exists: true };
  }

  async dispose(): Promise<void> {
    // No local resources to release.
  }
}
