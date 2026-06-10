/**
 * tokenizer.ts — Remote tokenizer client for vLLM-style `/tokenize` and
 * `/detokenize` endpoints.
 *
 * Used by store.ts to enforce token limits in remote mode without relying on
 * character-based estimates.
 */

import { buildBearerHeaders, nodePost } from './transport.js';

export type RemoteTokenizerMode = 'auto' | 'force' | 'off';

export type RemoteTokenizerConfig = {
  baseUrl: string;
  model: string;
  apiKey?: string;
  timeoutMs?: number;
};

const tokenizePathCache = new Map<string, string>();
const detokenizePathCache = new Map<string, string>();
const availabilityCache = new Map<string, boolean>();

function cacheKey(baseUrl: string, model: string): string {
  return `${baseUrl.replace(/\/+$/, '')}|${model}`;
}

function sanitizeBase(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function buildTokenizeUrlCandidates(baseUrl: string): string[] {
  const trimmed = sanitizeBase(baseUrl);

  if (trimmed.endsWith('/tokenize')) return [trimmed];
  if (trimmed.endsWith('/detokenize')) {
    const sibling = trimmed.replace(/\/detokenize$/, '/tokenize');
    return [sibling];
  }

  let root = trimmed;
  root = root.replace(/\/v1\/embed$/, '');
  root = root.replace(/\/v2\/embed$/, '');
  root = root.replace(/\/pooling$/, '');
  root = root.replace(/\/v1\/pooling$/, '');
  root = root.replace(/\/v1$/, '');

  const candidates = [
    `${root}/tokenize`,
    `${trimmed}/tokenize`,
    `${trimmed}/v1/tokenize`,
  ];
  return Array.from(new Set(candidates));
}

function buildDetokenizeUrlCandidates(baseUrl: string): string[] {
  const trimmed = sanitizeBase(baseUrl);

  if (trimmed.endsWith('/detokenize')) return [trimmed];
  if (trimmed.endsWith('/tokenize')) {
    const sibling = trimmed.replace(/\/tokenize$/, '/detokenize');
    return [sibling];
  }

  let root = trimmed;
  root = root.replace(/\/v1\/embed$/, '');
  root = root.replace(/\/v2\/embed$/, '');
  root = root.replace(/\/pooling$/, '');
  root = root.replace(/\/v1\/pooling$/, '');
  root = root.replace(/\/v1$/, '');

  const candidates = [
    `${root}/detokenize`,
    `${trimmed}/detokenize`,
    `${trimmed}/v1/detokenize`,
  ];
  return Array.from(new Set(candidates));
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
}

function parseTokenizeResponse(raw: unknown): number[] {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Tokenizer response is not an object');
  }
  const data = raw as Record<string, unknown>;
  const tokens = data['tokens'];
  if (!Array.isArray(tokens)) {
    throw new Error('Tokenizer response missing tokens[]');
  }
  if (!tokens.every((t) => typeof t === 'number' && Number.isFinite(t))) {
    throw new Error('Tokenizer response contains non-numeric tokens');
  }
  return tokens as number[];
}

function parseDetokenizeResponse(raw: unknown): string {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Detokenizer response is not an object');
  }
  const data = raw as Record<string, unknown>;
  const prompt = data['prompt'];
  if (typeof prompt !== 'string') {
    throw new Error('Detokenizer response missing prompt');
  }
  return prompt;
}

function resolveTimeoutMs(cfg: RemoteTokenizerConfig): number {
  if (typeof cfg.timeoutMs === 'number' && Number.isFinite(cfg.timeoutMs) && cfg.timeoutMs > 0) {
    return cfg.timeoutMs;
  }
  const envTimeout = parseInt(process.env.QMD_REMOTE_TOKENIZER_TIMEOUT_MS || '', 10);
  if (Number.isFinite(envTimeout) && envTimeout > 0) return envTimeout;
  return 30000;
}

export function resolveRemoteTokenizerMode(): RemoteTokenizerMode {
  const raw = (process.env.QMD_REMOTE_TOKENIZER || 'auto').trim().toLowerCase();
  if (raw === 'off' || raw === 'disabled' || raw === 'false' || raw === '0') return 'off';
  if (raw === 'force' || raw === 'required') return 'force';
  return 'auto';
}

export async function remoteTokenize(
  cfg: RemoteTokenizerConfig,
  prompt: string,
): Promise<number[]> {
  const key = cacheKey(cfg.baseUrl, cfg.model);
  const headers = buildBearerHeaders(cfg.apiKey);
  const candidates = orderByCachedFirst(
    buildTokenizeUrlCandidates(cfg.baseUrl),
    tokenizePathCache.get(key),
  );

  let lastRecoverableError: Error | null = null;
  for (const endpoint of candidates) {
    try {
      const response = await nodePost(
        endpoint,
        headers,
        {
          model: cfg.model,
          prompt,
          add_special_tokens: false,
        },
        resolveTimeoutMs(cfg),
      );
      const tokens = parseTokenizeResponse(response);
      tokenizePathCache.set(key, endpoint);
      availabilityCache.set(key, true);
      return tokens;
    } catch (err) {
      if (shouldTryNextEndpoint(err)) {
        lastRecoverableError = err instanceof Error ? err : new Error(String(err));
        continue;
      }
      throw err;
    }
  }

  availabilityCache.set(key, false);
  if (lastRecoverableError) throw lastRecoverableError;
  throw new Error('Remote tokenize request failed for all endpoint candidates');
}

export async function remoteDetokenize(
  cfg: RemoteTokenizerConfig,
  tokens: readonly number[],
): Promise<string> {
  const key = cacheKey(cfg.baseUrl, cfg.model);
  const headers = buildBearerHeaders(cfg.apiKey);
  const candidates = orderByCachedFirst(
    buildDetokenizeUrlCandidates(cfg.baseUrl),
    detokenizePathCache.get(key),
  );

  let lastRecoverableError: Error | null = null;
  for (const endpoint of candidates) {
    try {
      const response = await nodePost(
        endpoint,
        headers,
        {
          model: cfg.model,
          tokens,
        },
        resolveTimeoutMs(cfg),
      );
      const text = parseDetokenizeResponse(response);
      detokenizePathCache.set(key, endpoint);
      availabilityCache.set(key, true);
      return text;
    } catch (err) {
      if (shouldTryNextEndpoint(err)) {
        lastRecoverableError = err instanceof Error ? err : new Error(String(err));
        continue;
      }
      throw err;
    }
  }

  availabilityCache.set(key, false);
  if (lastRecoverableError) throw lastRecoverableError;
  throw new Error('Remote detokenize request failed for all endpoint candidates');
}

export async function remoteTokenizerAvailable(cfg: RemoteTokenizerConfig): Promise<boolean> {
  const key = cacheKey(cfg.baseUrl, cfg.model);
  const cached = availabilityCache.get(key);
  if (cached !== undefined) return cached;
  try {
    await remoteTokenize(cfg, 'qmd-tokenizer-probe');
    availabilityCache.set(key, true);
    return true;
  } catch {
    availabilityCache.set(key, false);
    return false;
  }
}
