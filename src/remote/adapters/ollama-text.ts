/**
 * ollama-text.ts — Ollama-native adapters for `/api/chat` and `/api/generate`.
 *
 * - Expand adapters:
 *   - ollama_chat     -> POST /api/chat with system+user messages
 *   - ollama_generate -> POST /api/generate with system + prompt fields
 * - Generate adapters:
 *   - ollama_chat     -> POST /api/chat with user message
 *   - ollama_generate -> POST /api/generate with prompt
 *
 * All calls force `stream: false` so transport receives one JSON payload.
 */

import type {
  ExpandAdapter,
  ExpandAdapterContext,
  GenerateAdapter,
  GenerateAdapterContext,
} from './types.js';
import type { GenerateOptions } from '../../llm.js';
import { buildBearerHeaders, nodePost } from '../transport.js';
import { expandFallback, parseExpandResponse } from '../expand.js';
import {
  EXPAND_SYSTEM_PROMPT,
  buildExpandUserPrompt,
  checkGate,
  handleGenerateError,
  normalizeModelName,
} from './normalization.js';

const chatPathCache = new Map<string, string>();
const generatePathCache = new Map<string, string>();

function cacheKey(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function orderByCachedFirst(values: string[], cached?: string): string[] {
  if (!cached || !values.includes(cached)) return values;
  return [cached, ...values.filter((v) => v !== cached)];
}

function shouldTryNextEndpoint(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return /HTTP (400|404|405|415|422|501)\b/.test(err.message);
}

function buildChatUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/api/chat')) return [trimmed];
  if (trimmed.endsWith('/chat')) {
    const apiSibling = trimmed.replace(/\/chat$/, '/api/chat');
    return Array.from(new Set([trimmed, apiSibling]));
  }
  if (trimmed.endsWith('/api')) return [`${trimmed}/chat`];
  if (trimmed.endsWith('/v1')) {
    const root = trimmed.replace(/\/v1$/, '');
    return Array.from(new Set([`${root}/api/chat`, `${trimmed}/chat/completions`]));
  }
  return Array.from(new Set([
    `${trimmed}/api/chat`,
    `${trimmed}/chat`,
    `${trimmed}/v1/chat/completions`,
  ]));
}

function buildGenerateUrlCandidates(baseUrl: string): string[] {
  const trimmed = baseUrl.replace(/\/+$/, '');

  if (trimmed.endsWith('/api/generate')) return [trimmed];
  if (trimmed.endsWith('/generate')) {
    const apiSibling = trimmed.replace(/\/generate$/, '/api/generate');
    return Array.from(new Set([trimmed, apiSibling]));
  }
  if (trimmed.endsWith('/api')) return [`${trimmed}/generate`];
  if (trimmed.endsWith('/v1')) {
    const root = trimmed.replace(/\/v1$/, '');
    return Array.from(new Set([`${root}/api/generate`, `${trimmed}/completions`]));
  }
  return Array.from(new Set([
    `${trimmed}/api/generate`,
    `${trimmed}/generate`,
    `${trimmed}/v1/completions`,
  ]));
}

function normalizeOllamaChatText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;
  const message = d['message'];
  if (!message || typeof message !== 'object') return '';
  const m = message as Record<string, unknown>;
  return typeof m['content'] === 'string' ? m['content'] : '';
}

function normalizeOllamaGenerateText(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const d = data as Record<string, unknown>;
  if (typeof d['response'] === 'string') return d['response'];
  const message = d['message'];
  if (!message || typeof message !== 'object') return '';
  const m = message as Record<string, unknown>;
  return typeof m['content'] === 'string' ? m['content'] : '';
}

function normalizeDone(data: unknown): boolean {
  if (!data || typeof data !== 'object') return true;
  const d = data as Record<string, unknown>;
  return typeof d['done'] === 'boolean' ? d['done'] : true;
}

function buildOllamaOptions(options?: GenerateOptions): Record<string, unknown> | undefined {
  const runtime: Record<string, unknown> = {};
  if (options?.maxTokens !== undefined) {
    runtime['num_predict'] = options.maxTokens;
  }
  if (options?.temperature !== undefined) {
    runtime['temperature'] = options.temperature;
  }
  return Object.keys(runtime).length > 0 ? runtime : undefined;
}

async function postWithFallback(
  baseUrl: string,
  candidates: (url: string) => string[],
  pathCache: Map<string, string>,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  const key = cacheKey(baseUrl);
  const cached = pathCache.get(key);
  const endpoints = orderByCachedFirst(candidates(baseUrl), cached);

  let lastRecoverableError: Error | null = null;
  for (const endpoint of endpoints) {
    try {
      const response = await nodePost(endpoint, headers, body, timeoutMs);
      pathCache.set(key, endpoint);
      return response;
    } catch (err) {
      if (shouldTryNextEndpoint(err)) {
        lastRecoverableError = err instanceof Error ? err : new Error(String(err));
        continue;
      }
      throw err;
    }
  }

  if (lastRecoverableError) throw lastRecoverableError;
  throw new Error('Request failed for all endpoint candidates');
}

async function postChatWithFallback(
  cfg: ExpandAdapterContext['cfg'] | GenerateAdapterContext['cfg'],
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  return postWithFallback(
    cfg.baseUrl,
    buildChatUrlCandidates,
    chatPathCache,
    buildBearerHeaders(cfg.apiKey),
    body,
    timeoutMs,
  );
}

async function postGenerateWithFallback(
  cfg: ExpandAdapterContext['cfg'] | GenerateAdapterContext['cfg'],
  body: Record<string, unknown>,
  timeoutMs: number,
): Promise<unknown> {
  return postWithFallback(
    cfg.baseUrl,
    buildGenerateUrlCandidates,
    generatePathCache,
    buildBearerHeaders(cfg.apiKey),
    body,
    timeoutMs,
  );
}

export const ollamaChatExpandAdapter: ExpandAdapter = {
  id: 'ollama/chat-expand',

  async expandQuery(ctx, query, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    const gate = checkGate(breaker);
    if (!gate.allowed) {
      log.warn(`RemoteLLM: expand ${gate.reason}`);
      return expandFallback(query, options?.includeLexical ?? true);
    }

    const includeLexical = options?.includeLexical ?? true;
    try {
      const data = await postChatWithFallback(
        cfg,
        {
          model: cfg.model,
          messages: [
            { role: 'system', content: EXPAND_SYSTEM_PROMPT },
            { role: 'user', content: buildExpandUserPrompt(query, options?.intent) },
          ],
          stream: false,
        },
        readTimeoutMs,
      );

      const content = normalizeOllamaChatText(data);
      if (content.trim().length > 0) {
        const queryables = parseExpandResponse(content, query, includeLexical);
        if (queryables.length === 0) {
          log.warn(
            'RemoteLLM: expandQuery received response but could not parse any ' +
            `valid query variants. Raw response (first 200 chars): "${content.slice(0, 200)}"`,
          );
        }
        breaker.onSuccess();
        return queryables.length > 0 ? queryables : expandFallback(query, includeLexical);
      }

      log.warn('RemoteLLM: expandQuery returned empty response, using fallback');
      breaker.onSuccess();
      return expandFallback(query, includeLexical);
    } catch (err) {
      breaker.onFailure();
      log.error(
        'RemoteLLM: expandQuery failed:',
        err instanceof Error ? err.message : String(err),
      );
      return expandFallback(query, includeLexical);
    }
  },
};

export const ollamaGenerateExpandAdapter: ExpandAdapter = {
  id: 'ollama/generate-expand',

  async expandQuery(ctx, query, options) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    const gate = checkGate(breaker);
    if (!gate.allowed) {
      log.warn(`RemoteLLM: expand ${gate.reason}`);
      return expandFallback(query, options?.includeLexical ?? true);
    }

    const includeLexical = options?.includeLexical ?? true;
    try {
      const data = await postGenerateWithFallback(
        cfg,
        {
          model: cfg.model,
          system: EXPAND_SYSTEM_PROMPT,
          prompt: buildExpandUserPrompt(query, options?.intent),
          stream: false,
        },
        readTimeoutMs,
      );

      const content = normalizeOllamaGenerateText(data);
      if (content.trim().length > 0) {
        const queryables = parseExpandResponse(content, query, includeLexical);
        if (queryables.length === 0) {
          log.warn(
            'RemoteLLM: expandQuery received response but could not parse any ' +
            `valid query variants. Raw response (first 200 chars): "${content.slice(0, 200)}"`,
          );
        }
        breaker.onSuccess();
        return queryables.length > 0 ? queryables : expandFallback(query, includeLexical);
      }

      log.warn('RemoteLLM: expandQuery returned empty response, using fallback');
      breaker.onSuccess();
      return expandFallback(query, includeLexical);
    } catch (err) {
      breaker.onFailure();
      log.error(
        'RemoteLLM: expandQuery failed:',
        err instanceof Error ? err.message : String(err),
      );
      return expandFallback(query, includeLexical);
    }
  },
};

export const ollamaChatGenerateAdapter: GenerateAdapter = {
  id: 'ollama/chat-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
      return null;
    }

    try {
      const runtimeOptions = buildOllamaOptions(options);
      const data = await postChatWithFallback(
        cfg,
        {
          model: cfg.model,
          messages: [{ role: 'user', content: prompt }],
          stream: false,
          ...(runtimeOptions ? { options: runtimeOptions } : {}),
        },
        readTimeoutMs,
      );

      breaker.onSuccess();
      return {
        text: normalizeOllamaChatText(data),
        model: normalizeModelName(data, cfg.model),
        done: normalizeDone(data),
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (ollama chat)');
    }
  },
};

export const ollamaGenerateGenerateAdapter: GenerateAdapter = {
  id: 'ollama/generate-generate',

  async generate(ctx, prompt, options?: GenerateOptions) {
    const { cfg, breaker, log, readTimeoutMs } = ctx;
    if (!breaker.canAttempt()) {
      log.warn('RemoteLLM: generate circuit breaker is open, returning null');
      return null;
    }

    try {
      const runtimeOptions = buildOllamaOptions(options);
      const data = await postGenerateWithFallback(
        cfg,
        {
          model: cfg.model,
          prompt,
          stream: false,
          ...(runtimeOptions ? { options: runtimeOptions } : {}),
        },
        readTimeoutMs,
      );

      breaker.onSuccess();
      return {
        text: normalizeOllamaGenerateText(data),
        model: normalizeModelName(data, cfg.model),
        done: normalizeDone(data),
      };
    } catch (err) {
      return handleGenerateError(err, breaker, log, 'generate (ollama generate)');
    }
  },
};

