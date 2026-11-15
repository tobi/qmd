/**
 * remote-rerank.ts — Cohere-shaped remote /v1/rerank client.
 *
 * Rerank-only remote offload: embedding and query expansion stay local. The
 * endpoint comes from the trusted config path (global config, or a
 * project-local .qmd/index.yml explicitly approved via `qmd trust`) or from
 * env vars — never from an untrusted project-local config (see trust.ts; this
 * closes #705's exfiltration vector).
 */

import type { ModelsConfig } from "./collections.js";
import type { RerankDocument, RerankResult } from "./llm.js";

export type RerankScoreMode = "logits" | "probability" | "auto";

export type RemoteRerankConfig = {
  /** Base endpoint, e.g. http://host:8082/v1 — POSTs to <url>/rerank. */
  url: string;
  model: string;
  apiKey?: string;
  scoreMode: RerankScoreMode;
  timeoutMs: number;
  /** Stable cache identity of the remote reranker: remote:<model>@<url>@<scoreMode>.
   *  Score mode participates because cached scores are already normalized. */
  identity: string;
};

const DEFAULT_RERANK_TIMEOUT_MS = 10000;

function parseScoreMode(raw: string | undefined): RerankScoreMode {
  if (raw === undefined || raw === "") return "auto";
  if (raw === "logits" || raw === "probability" || raw === "auto") return raw;
  throw new Error(`invalid rerank_score "${raw}" (expected logits|probability|auto)`);
}

function parseTimeoutMs(raw: number | undefined): number {
  if (raw === undefined) return DEFAULT_RERANK_TIMEOUT_MS;
  if (!Number.isFinite(raw) || raw <= 0) {
    throw new Error(`invalid rerank_api_timeout ${String(raw)} (expected positive milliseconds)`);
  }
  return raw;
}

function envTimeoutMs(raw: string | undefined): number {
  if (raw === undefined || raw === "") return DEFAULT_RERANK_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`invalid QMD_RERANK_API_TIMEOUT "${raw}" (expected positive milliseconds)`);
  }
  return parseTimeoutMs(parsed);
}

/**
 * Resolve the remote rerank endpoint. YAML `rerank_api_*` fields are honored
 * only when `trusted` (the global config path is always trusted; a
 * project-local config must be approved via `qmd trust`). Env vars are the
 * user's own environment and stay trusted. Remote rerank activates only when
 * both url and model are present.
 */
export function resolveRemoteRerankConfig(
  models?: ModelsConfig,
  opts?: { trusted?: boolean },
): RemoteRerankConfig | null {
  const trusted = opts?.trusted ?? false;
  const url = (trusted ? models?.rerank_api_url : undefined) || process.env.QMD_RERANK_API_URL;
  const model = (trusted ? models?.rerank_api_model : undefined) || process.env.QMD_RERANK_API_MODEL;
  if (!url || !model) return null;
  const apiKey = (trusted ? models?.rerank_api_key : undefined) || process.env.QMD_RERANK_API_KEY || undefined;
  const scoreModeRaw = trusted ? models?.rerank_score : undefined;
  const scoreMode = parseScoreMode(scoreModeRaw ?? process.env.QMD_RERANK_SCORE);
  const timeoutRaw = trusted ? models?.rerank_api_timeout : undefined;
  const timeoutMs = timeoutRaw !== undefined ? parseTimeoutMs(timeoutRaw) : envTimeoutMs(process.env.QMD_RERANK_API_TIMEOUT);
  return { url, model, apiKey, scoreMode, timeoutMs, identity: `remote:${model}@${url}@${scoreMode}` };
}

function sigmoid(x: number): number {
  // Two-branch form so extreme logits don't overflow Math.exp.
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Normalize raw rerank scores to [0,1] per the rerank_score knob.
 * - "logits": always sigmoid (llama.cpp cross-encoders emit log-odds).
 * - "probability": pass through (backends already emitting 0..1).
 * - "auto": sigmoid only when the batch contains a score outside [0,1].
 */
export function normalizeRerankScores(scores: number[], mode: RerankScoreMode): number[] {
  if (mode === "logits") return scores.map(sigmoid);
  if (mode === "probability") return [...scores];
  const needsSigmoid = scores.some((s) => s < 0 || s > 1);
  return needsSigmoid ? scores.map(sigmoid) : [...scores];
}

/**
 * POST the batch to <url>/rerank (Cohere shape) and map results back onto the
 * original documents, sorted by normalized score. Any transport, HTTP, or
 * response-shape failure throws so the caller can fall back to local rerank.
 */
export async function remoteRerank(
  query: string,
  documents: RerankDocument[],
  cfg: RemoteRerankConfig,
): Promise<RerankResult> {
  if (documents.length === 0) return { results: [], model: cfg.identity };

  const base = cfg.url.replace(/\/+$/, "");
  const endpoint = base.endsWith("/rerank") ? base : `${base}/rerank`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: cfg.model,
      query,
      documents: documents.map((d) => d.text),
      top_n: documents.length,
    }),
    signal: AbortSignal.timeout(cfg.timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`remote rerank endpoint returned HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`);
  }
  const payload: unknown = await response.json();
  const scores = parseRerankResponse(payload, documents.length);
  const normalized = normalizeRerankScores(scores, cfg.scoreMode);
  const results = documents
    .map((doc, index) => ({ ...doc, score: normalized[index] ?? 0, index }))
    .sort((a, b) => b.score - a.score);
  return { results, model: cfg.identity };
}

/** Validate a Cohere-shaped results array and return scores by document index. */
function parseRerankResponse(payload: unknown, expected: number): number[] {
  if (typeof payload !== "object" || payload === null || !("results" in payload) || !Array.isArray(payload.results)) {
    throw new Error("remote rerank response missing results array");
  }
  const rawResults: unknown[] = payload.results;
  const byIndex = new Map<number, number>();
  for (const item of rawResults) {
    if (typeof item !== "object" || item === null || !("index" in item) || !("relevance_score" in item)) {
      throw new Error("remote rerank result missing index or relevance_score");
    }
    const index = item.index;
    const score = item.relevance_score;
    if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= expected) {
      throw new Error(`remote rerank result index ${String(index)} out of range for ${expected} documents`);
    }
    if (typeof score !== "number" || !Number.isFinite(score)) {
      throw new Error(`remote rerank result ${index} has a non-finite relevance_score`);
    }
    if (byIndex.has(index)) throw new Error(`remote rerank response has duplicate index ${index}`);
    byIndex.set(index, score);
  }
  if (byIndex.size !== expected) {
    throw new Error(`remote rerank response missing ${expected - byIndex.size} of ${expected} results`);
  }
  return Array.from({ length: expected }, (_, i) => byIndex.get(i) ?? 0);
}
