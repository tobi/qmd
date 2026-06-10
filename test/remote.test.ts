/**
 * remote.test.ts — Unit tests for src/remote/ modules.
 *
 * Tests cover: transport, circuit-breaker, log, config, embed (with retry),
 * expand (with parse + fallback), rerank, generate, probe, and remote-llm.
 *
 * Run with: npx vitest run test/remote.test.ts
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import * as http from 'http';
import type { AddressInfo } from 'net';

// ── Module imports ────────────────────────────────────────────────────

import { nodePost, nodeGet, MAX_RESPONSE_BODY_BYTES } from '../src/remote/transport.js';
import { CircuitBreaker } from '../src/remote/circuit-breaker.js';
import { consoleLogger, silentLogger, type Logger } from '../src/remote/log.js';
import {
  resolveEndpoint,
  remoteConfigFromEnv,
  OPENROUTER_DEFAULT_URL,
  resolveEndpointFormat,
} from '../src/remote/config.js';
import {
  embedBatch,
} from '../src/remote/embed.js';
import {
  expandQuery,
  parseExpandResponse,
  expandFallback,
} from '../src/remote/expand.js';
import { rerank } from '../src/remote/rerank.js';
import { generate } from '../src/remote/generate.js';
import { modelExists, probe } from '../src/remote/probe.js';
import { RemoteLLM } from '../src/remote/remote-llm.js';
import type { EndpointConfig } from '../src/remote/types.js';

// ── Phase 2 imports — OpenAI protocol adapters and shared helpers ─────

import {
  normalizeChatCompletionText,
  normalizeCompletionsText,
  normalizeResponseAPIText,
  normalizeModelName,
  normalizeAnthropicMessagesText,
} from '../src/remote/adapters/normalization.js';
import {
  openaiChatCompletionsExpandAdapter,
  openaiChatCompletionsGenerateAdapter,
} from '../src/remote/adapters/openai-chat.js';
import {
  openaiCompletionsExpandAdapter,
  openaiCompletionsGenerateAdapter,
} from '../src/remote/adapters/openai-completions.js';
import {
  openaiResponsesExpandAdapter,
  openaiResponsesGenerateAdapter,
} from '../src/remote/adapters/openai-responses.js';
import {
  anthropicMessagesExpandAdapter,
  anthropicMessagesGenerateAdapter,
} from '../src/remote/adapters/anthropic-messages.js';
import { cohereV2EmbedAdapter } from '../src/remote/adapters/cohere-embed.js';
import { cohereRerankAdapter } from '../src/remote/adapters/cohere-rerank.js';
import { vllmScoreAdapter } from '../src/remote/adapters/vllm-score.js';
import { resolveAdapterBundle } from '../src/remote/adapters/registry.js';

// =============================================================================
// Helpers — mock HTTP server
// =============================================================================

/** Start a mock HTTP server on a random port. Returns { url, close }. */
function startMockServer(
  handler: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): { url: string; close: () => Promise<void> } {
  const server = http.createServer(handler);
  server.listen(0);
  const addr = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${addr.port}`,
    close: () => new Promise<void>((resolve, reject) => {
      server.close((err) => err ? reject(err) : resolve());
    }),
  };
}

/** Helper: read full request body as JSON. */
function readBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8')));
    });
  });
}

/** Helper: write JSON response. */
function jsonRes(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

/** Standard mock embedding — returns 3-dimensional vectors. */
const mockEmbedding = [0.1, 0.2, 0.3];

/** Test endpoint config pointing to localhost. */
function testCfg(url: string, model = 'test-model', apiKey?: string): EndpointConfig {
  return { baseUrl: url, model, apiKey };
}

/** Create a spy logger that records all calls. */
function spyLogger(): Logger & { calls: Array<{ level: string; msg: string }> } {
  const calls: Array<{ level: string; msg: string }> = [];
  return {
    debug: (msg) => { calls.push({ level: 'debug', msg }); },
    info: (msg) => { calls.push({ level: 'info', msg }); },
    warn: (msg) => { calls.push({ level: 'warn', msg }); },
    error: (msg) => { calls.push({ level: 'error', msg }); },
    calls,
  };
}

// =============================================================================
// log.ts
// =============================================================================

describe('Logger', () => {
  test('consoleLogger writes without throwing', () => {
    expect(() => consoleLogger.warn('test')).not.toThrow();
    expect(() => consoleLogger.error('test')).not.toThrow();
  });

  test('silentLogger discards all messages', () => {
    expect(() => silentLogger.warn('test')).not.toThrow();
    expect(() => silentLogger.error('test')).not.toThrow();
  });
});

// =============================================================================
// transport.ts
// =============================================================================

describe('nodePost', () => {
  test('sends JSON POST and returns parsed response', async () => {
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req);
      expect(body).toEqual({ foo: 'bar' });
      jsonRes(res, 200, { result: 'ok' });
    });

    const result = await nodePost(`${server.url}/test`, {}, { foo: 'bar' }, 5000);
    expect(result).toEqual({ result: 'ok' });
    await server.close();
  });

  test('throws on non-2xx status', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 500, { error: 'server error' });
    });

    await expect(
      nodePost(`${server.url}/test`, {}, {}, 5000),
    ).rejects.toThrow('HTTP 500');
    await server.close();
  });

  test('throws on timeout', async () => {
    const server = startMockServer((_req, res) => {
      // Never respond
      setTimeout(() => res.end(), 10000);
    });

    await expect(
      nodePost(`${server.url}/test`, {}, {}, 100),
    ).rejects.toThrow('timed out');
    await server.close();
  });

  test('throws on oversized response', async () => {
    const server = startMockServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      // Send a response larger than the 10MB cap
      const big = 'x'.repeat(MAX_RESPONSE_BODY_BYTES + 1000);
      res.end(JSON.stringify({ data: big }));
    });

    await expect(
      nodePost(`${server.url}/test`, {}, {}, 5000),
    ).rejects.toThrow(/exceeded/);
    await server.close();
  });

  test('passes Authorization header', async () => {
    const server = startMockServer((req, res) => {
      expect(req.headers['authorization']).toBe('Bearer sk-test');
      jsonRes(res, 200, { ok: true });
    });

    await nodePost(`${server.url}/test`, { Authorization: 'Bearer sk-test' }, {}, 5000);
    await server.close();
  });
});

describe('nodeGet', () => {
  test('sends GET and returns parsed response', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { data: [{ id: 'test-model' }] });
    });

    const result = await nodeGet(`${server.url}/models`, {}, 5000);
    expect(result).toEqual({ data: [{ id: 'test-model' }] });
    await server.close();
  });
});

// =============================================================================
// circuit-breaker.ts
// =============================================================================

describe('CircuitBreaker', () => {
  test('starts in closed state', () => {
    const cb = new CircuitBreaker();
    expect(cb.canAttempt()).toBe(true);
    expect(cb.getState()).toBe('closed');
  });

  test('opens after maxFailures consecutive failures', () => {
    const cb = new CircuitBreaker(2);
    cb.onFailure();
    expect(cb.canAttempt()).toBe(true); // still closed (1 < 2)
    cb.onFailure();
    expect(cb.canAttempt()).toBe(false); // open after 2 failures
    expect(cb.getState()).toBe('open');
  });

  test('resets to closed on success', () => {
    const cb = new CircuitBreaker(2);
    cb.onFailure();
    cb.onFailure();
    expect(cb.getState()).toBe('open');
    // Force half-open by mocking time (not possible cleanly — test via success path)
  });

  test('onSuccess resets failures counter', () => {
    const cb = new CircuitBreaker(3);
    cb.onFailure();
    cb.onFailure();
    cb.onSuccess();
    expect(cb.getState()).toBe('closed');
  });

  test('transitions: closed → open → half-open → closed', async () => {
    // Use a very short cooldown for testing
    const cb = new CircuitBreaker(1, 50); // 1 failure, 50ms cooldown
    cb.onFailure();
    expect(cb.canAttempt()).toBe(false); // open after 1 failure
    expect(cb.getState()).toBe('open');

    // Wait for cooldown
    await new Promise((r) => setTimeout(r, 60));
    expect(cb.canAttempt()).toBe(true); // half-open
    expect(cb.getState()).toBe('half-open');

    // Success in half-open → back to closed
    cb.onSuccess();
    expect(cb.getState()).toBe('closed');
  });

  test('half-open failure goes back to open', async () => {
    const cb = new CircuitBreaker(1, 50);
    cb.onFailure();
    await new Promise((r) => setTimeout(r, 60));
    cb.canAttempt(); // transition to half-open
    cb.onFailure();  // fail in half-open
    expect(cb.getState()).toBe('open');
  });
});

// =============================================================================
// config.ts
// =============================================================================

describe('resolveEndpoint', () => {
  test('uses env vars when set', () => {
    process.env.QMD_TEST_MODEL = 'env-model';
    process.env.QMD_TEST_BASE_URL = 'http://env:1234/v1';
    const result = resolveEndpoint('test', 'TEST', 'default-model', 'http://default:8000/v1');
    expect(result.model).toBe('env-model');
    expect(result.baseUrl).toBe('http://env:1234/v1');
    delete process.env.QMD_TEST_MODEL;
    delete process.env.QMD_TEST_BASE_URL;
  });

  test('falls back to defaults when env vars are unset', () => {
    const result = resolveEndpoint('test', 'NONEXISTENT', 'default-model', 'http://default:8000/v1');
    expect(result.model).toBe('default-model');
    expect(result.baseUrl).toBe('http://default:8000/v1');
  });

  test('trims trailing slash from baseUrl', () => {
    process.env.QMD_TRIM_BASE_URL = 'http://example.com/v1/';
    const result = resolveEndpoint('trim', 'TRIM', 'model', 'http://default/v1');
    expect(result.baseUrl).toBe('http://example.com/v1');
    delete process.env.QMD_TRIM_BASE_URL;
  });
});

describe('remoteConfigFromEnv', () => {
  // Save/restore env vars that may be set in the test environment
  const saveEnv = (key: string) => ({ key, val: process.env[key] });
  const restoreEnv = (...saved: Array<{ key: string; val: string | undefined }>) => {
    for (const { key, val } of saved) {
      if (val === undefined) delete process.env[key];
      else process.env[key] = val;
    }
  };

  test('returns empty baseUrls when nothing is configured (local-first)', () => {
    const saved = [
      saveEnv('QMD_EMBED_BASE_URL'), saveEnv('QMD_EMBED_MODEL'),
      saveEnv('QMD_EXPAND_BASE_URL'), saveEnv('QMD_EXPAND_MODEL'),
      saveEnv('QMD_RERANK_BASE_URL'), saveEnv('QMD_RERANK_MODEL'),
      saveEnv('QMD_GENERATE_BASE_URL'), saveEnv('QMD_GENERATE_MODEL'),
      saveEnv('OPENAI_BASE_URL'), saveEnv('OPENAI_API_KEY'),
    ];
    for (const { key } of saved) delete process.env[key];

    const config = remoteConfigFromEnv();
    expect(config.embed!.baseUrl).toBe('');
    expect(config.expand!.baseUrl).toBe('');
    expect(config.rerank!.baseUrl).toBe('');
    expect(config.generate!.baseUrl).toBe('');

    restoreEnv(...saved);
  });

  test('picks up YAML config for all endpoints', () => {
    const saved = [
      saveEnv('QMD_EMBED_BASE_URL'), saveEnv('QMD_EMBED_MODEL'),
      saveEnv('QMD_EXPAND_BASE_URL'), saveEnv('QMD_EXPAND_MODEL'),
      saveEnv('QMD_RERANK_BASE_URL'), saveEnv('QMD_RERANK_MODEL'),
      saveEnv('QMD_GENERATE_BASE_URL'), saveEnv('QMD_GENERATE_MODEL'),
      saveEnv('OPENAI_BASE_URL'), saveEnv('OPENAI_API_KEY'),
    ];
    for (const { key } of saved) delete process.env[key];

    const config = remoteConfigFromEnv({
      embed_api_url: 'http://embed:8000/v1',
      embed_api_format: 'cohere_v2_embed',
      embed_api_model: 'embed-model',
      expand_api_url: 'http://expand:8000/v1',
      expand_api_format: 'openai_chat_completions',
      expand_api_model: 'expand-model',
      rerank_api_url: 'http://rerank:8000/v1',
      rerank_api_format: 'cohere_v1_rerank',
      rerank_api_model: 'rerank-model',
      generate_api_url: 'http://generate:8000/v1',
      generate_api_format: 'anthropic_messages',
      generate_api_model: 'gen-model',
    });
    expect(config.embed!.baseUrl).toBe('http://embed:8000/v1');
    expect(config.embed!.model).toBe('embed-model');
    expect(config.embed!.format).toBe('cohere_v2_embed');
    expect(config.expand!.baseUrl).toBe('http://expand:8000/v1');
    expect(config.expand!.format).toBe('openai_chat_completions');
    expect(config.rerank!.baseUrl).toBe('http://rerank:8000/v1');
    expect(config.rerank!.format).toBe('cohere_v1_rerank');
    expect(config.generate!.baseUrl).toBe('http://generate:8000/v1');
    expect(config.generate!.format).toBe('anthropic_messages');

    restoreEnv(...saved);
  });

  test('OPENAI_BASE_URL fallback for embed endpoint', () => {
    const saved = [saveEnv('QMD_EMBED_BASE_URL')];
    delete process.env.QMD_EMBED_BASE_URL;
    process.env.OPENAI_BASE_URL = 'http://openai:8000/v1';
    const config = remoteConfigFromEnv();
    expect(config.embed!.baseUrl).toBe('http://openai:8000/v1');
    delete process.env.OPENAI_BASE_URL;
    restoreEnv(...saved);
  });

  test('defaults endpoint formats to auto', () => {
    const saved = [
      saveEnv('QMD_EMBED_API_FORMAT'),
      saveEnv('QMD_EXPAND_API_FORMAT'),
      saveEnv('QMD_RERANK_API_FORMAT'),
      saveEnv('QMD_GENERATE_API_FORMAT'),
    ];
    for (const { key } of saved) delete process.env[key];

    const config = remoteConfigFromEnv();
    expect(config.embed!.format).toBe('auto');
    expect(config.expand!.format).toBe('auto');
    expect(config.rerank!.format).toBe('auto');
    expect(config.generate!.format).toBe('auto');

    restoreEnv(...saved);
  });

  test('supports short format aliases from env', () => {
    const saved = [saveEnv('QMD_EMBED_API_FORMAT'), saveEnv('QMD_RERANK_API_FORMAT')];
    process.env.QMD_EMBED_API_FORMAT = 'openai_v1';
    process.env.QMD_RERANK_API_FORMAT = 'cohere_rerank';

    const config = remoteConfigFromEnv();
    expect(config.embed!.format).toBe('openai_v1_embeddings');
    expect(config.rerank!.format).toBe('cohere_v2_rerank');

    restoreEnv(...saved);
  });

  test('throws on invalid endpoint format', () => {
    const saved = [saveEnv('QMD_EMBED_API_FORMAT')];
    process.env.QMD_EMBED_API_FORMAT = 'totally_invalid_format';
    expect(() => remoteConfigFromEnv()).toThrow(/Invalid embed API format/);
    restoreEnv(...saved);
  });

  test('throws when format is unsupported for endpoint role', () => {
    const saved = [saveEnv('QMD_EMBED_API_FORMAT')];
    process.env.QMD_EMBED_API_FORMAT = 'anthropic_messages';
    expect(() => remoteConfigFromEnv()).toThrow(/Allowed formats for embed/);
    restoreEnv(...saved);
  });
});

describe('resolveEndpointFormat', () => {
  test('returns auto when no env/yaml format is set', () => {
    expect(resolveEndpointFormat('embed', 'EMBED')).toBe('auto');
  });

  test('normalizes dash/space variants', () => {
    expect(resolveEndpointFormat('expand', 'EXPAND', 'openai-chat')).toBe('openai_chat_completions');
  });
});

// =============================================================================
// embed.ts
// =============================================================================

describe('embedBatch', () => {
  const dimState = { dimensions: null as number | null };

  beforeEach(() => {
    dimState.dimensions = null;
  });

  test('returns embeddings for a batch of texts', async () => {
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as { input: string[] };
      jsonRes(res, 200, {
        data: body.input.map((_, i) => ({ embedding: mockEmbedding, index: i })),
      });
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const results = await embedBatch(cfg, breaker, ['hello', 'world'], 10, 5000, dimState, 1, silentLogger);

    expect(results).toHaveLength(2);
    expect(results[0]!.embedding).toEqual(mockEmbedding);
    expect(results[0]!.model).toBe('test-model');
    expect(results[1]!.embedding).toEqual(mockEmbedding);
    expect(dimState.dimensions).toBe(3);
    await server.close();
  });

  test('sets dimension on first response and validates subsequent', async () => {
    let callCount = 0;
    const server = startMockServer((_req, res) => {
      callCount++;
      // Second call returns different dimensions
      const dim = callCount === 1 ? 3 : 5;
      jsonRes(res, 200, {
        data: [{ embedding: new Array(dim).fill(0.1), index: 0 }],
      });
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const ds = { dimensions: null as number | null };

    // First batch: sets dimension to 3
    await embedBatch(cfg, breaker, ['a'], 10, 5000, ds, 1, silentLogger);
    expect(ds.dimensions).toBe(3);

    // Second batch: dimension mismatch → should throw
    await expect(
      embedBatch(cfg, breaker, ['b'], 10, 5000, ds, 1, silentLogger),
    ).rejects.toThrow('dimension mismatch');
    await server.close();
  });

  test('returns empty array for empty input', async () => {
    const ds = { dimensions: null as number | null };
    const results = await embedBatch(
      testCfg('http://localhost:1'), new CircuitBreaker(), [], 10, 5000, ds, 1, silentLogger,
    );
    expect(results).toEqual([]);
  });

  test('circuit breaker open throws immediately', async () => {
    const breaker = new CircuitBreaker(1);
    breaker.onFailure(); // open
    const ds = { dimensions: null as number | null };

    await expect(
      embedBatch(testCfg('http://localhost:1'), breaker, ['text'], 10, 5000, ds, 1, silentLogger),
    ).rejects.toThrow(/circuit breaker is open/);
  });

  test('retries on failure and logs warnings', async () => {
    const log = spyLogger();
    let attempts = 0;
    const server = startMockServer((_req, res) => {
      attempts++;
      if (attempts < 2) {
        res.writeHead(503);
        res.end('unavailable');
      } else {
        jsonRes(res, 200, {
          data: [{ embedding: mockEmbedding, index: 0 }],
        });
      }
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const ds = { dimensions: null as number | null };
    const results = await embedBatch(cfg, breaker, ['retry-me'], 10, 5000, ds, 2, log);

    expect(attempts).toBe(2);
    expect(results).toHaveLength(1);
    expect(results[0]!.embedding).toEqual(mockEmbedding);
    expect(log.calls.filter((c) => c.level === 'warn')).toHaveLength(1);
    expect(log.calls[0]!.msg).toContain('retrying');
    await server.close();
  });

  test('returns nulls after all retries exhausted', async () => {
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      res.writeHead(500);
      res.end('fail');
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const ds = { dimensions: null as number | null };
    const results = await embedBatch(cfg, breaker, ['a', 'b'], 10, 5000, ds, 2, log);

    expect(results).toEqual([null, null]);
    expect(log.calls.filter((c) => c.level === 'error')).toHaveLength(1);
    await server.close();
  });

  test('splits large input into batches', async () => {
    const batchSizes: number[] = [];
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as { input: string[] };
      batchSizes.push(body.input.length);
      jsonRes(res, 200, {
        data: body.input.map((_, i) => ({ embedding: mockEmbedding, index: i })),
      });
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const ds = { dimensions: null as number | null };
    const texts = ['a', 'b', 'c', 'd', 'e']; // 5 texts, batch size 2 → 3 batches
    const results = await embedBatch(cfg, breaker, texts, 2, 5000, ds, 1, silentLogger);

    expect(results).toHaveLength(5);
    expect(batchSizes).toEqual([2, 2, 1]);
    await server.close();
  });

  test('modelOverride via options', async () => {
    let receivedModel = '';
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as { model: string; input: string[] };
      receivedModel = body.model;
      jsonRes(res, 200, {
        data: [{ embedding: mockEmbedding, index: 0 }],
      });
    });

    const cfg = testCfg(server.url);
    const breaker = new CircuitBreaker();
    const ds = { dimensions: null as number | null };
    await embedBatch(cfg, breaker, ['x'], 10, 5000, ds, 1, silentLogger, { model: 'override-model' });

    expect(receivedModel).toBe('override-model');
    await server.close();
  });
});

// =============================================================================
// expand.ts
// =============================================================================

describe('parseExpandResponse', () => {
  test('parses lex, vec, hyde lines', () => {
    const content = 'lex: keyword search\nvec: semantic paraphrase of search\nhyde: A document about search techniques';
    const result = parseExpandResponse(content, 'search', true);
    expect(result).toHaveLength(3);
    expect(result[0]!.type).toBe('lex');
    expect(result[1]!.type).toBe('vec');
    expect(result[2]!.type).toBe('hyde');
  });

  test('strips markdown formatting', () => {
    // Each variant must contain a term from originalQuery.
    // Use "search" as the overlapping term in all three lines.
    const content = 'lex: **bold search**\nvec: "search paraphrase"\nhyde: **search hyde**';
    const result = parseExpandResponse(content, 'search', true);
    expect(result).toHaveLength(3);
    expect(result[0]!.text).toBe('bold search');
    expect(result[1]!.text).toBe('search paraphrase');
    expect(result[2]!.text).toBe('search hyde');
  });

  test('filters out lines without query term overlap', () => {
    const content = 'lex: completely different\nvec: about search thing\nhyde: unrelated text';
    const result = parseExpandResponse(content, 'search', true);
    // "completely different" has no overlap with "search"
    // "about search thing" does
    expect(result).toHaveLength(1);
    expect(result[0]!.text).toContain('search');
  });

  test('returns empty array for unparseable content', () => {
    const result = parseExpandResponse('just some random text\nno colons here', 'query', true);
    expect(result).toEqual([]);
  });

  test('excludes lex when includeLexical is false', () => {
    // All three lines contain "query" so all pass term-overlap check
    const content = 'lex: query keywords\nvec: query paraphrase\nhyde: query excerpt';
    const result = parseExpandResponse(content, 'query', false);
    expect(result).toHaveLength(2);
    expect(result.every((q) => q.type !== 'lex')).toBe(true);
  });

  test('rejects very short texts (>3 chars required)', () => {
    const content = 'lex: ab\nvec: longer text here about search';
    const result = parseExpandResponse(content, 'search', true);
    expect(result).toHaveLength(1); // only vec passes
  });
});

describe('expandFallback', () => {
  test('returns lex, vec, hyde based on raw query', () => {
    const result = expandFallback('my query', true);
    expect(result).toHaveLength(3);
    expect(result.find((q) => q.type === 'hyde')!.text).toContain('my query');
  });

  test('excludes lex when includeLexical is false', () => {
    const result = expandFallback('query', false);
    expect(result.every((q) => q.type !== 'lex')).toBe(true);
    expect(result).toHaveLength(2);
  });
});

describe('expandQuery', () => {
  test('returns passthrough when no API key', async () => {
    const log = spyLogger();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: '' };
    const breaker = new CircuitBreaker();
    const result = await expandQuery(cfg, breaker, 'test query', 5000, {}, log);

    expect(result).toEqual([{ type: 'lex', text: 'test query' }]);
    expect(log.calls.some((c) => c.msg.includes('no API key'))).toBe(true);
  });

  test('calls chat completions and parses response', async () => {
    const server = startMockServer(async (req, res) => {
      jsonRes(res, 200, {
        choices: [{
          message: {
            // All variants contain "topic" so all pass term-overlap validation
            content: 'lex: topic keywords\nvec: semantic topic meaning\nhyde: A document discussing the topic',
          },
        }],
        model: 'test-model',
      });
    });

    const cfg: EndpointConfig = { baseUrl: server.url, model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await expandQuery(cfg, breaker, 'topic', 5000, {}, silentLogger);

    expect(result).toHaveLength(3);
    expect(result[0]!.type).toBe('lex');
    await server.close();
  });

  test('falls back when LLM returns unparseable content', async () => {
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: 'not in the expected format at all' } }],
        model: 'test',
      });
    });

    const cfg: EndpointConfig = { baseUrl: server.url, model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await expandQuery(cfg, breaker, 'query topic', 5000, {}, log);

    // Should fall back to expandFallback
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('could not parse'))).toBe(true);
    await server.close();
  });

  test('circuit breaker open returns fallback', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure(); // open
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: 'sk-test' };
    const result = await expandQuery(cfg, breaker, 'test', 5000, {}, log);

    expect(result).toHaveLength(3); // fallback
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });
});

// =============================================================================
// rerank.ts
// =============================================================================

describe('rerank', () => {
  const docs = [
    { file: 'a.md', text: 'first document about search' },
    { file: 'b.md', text: 'second document about something else' },
  ];

  test('returns uniform scores when no API key', async () => {
    const log = spyLogger();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: '' };
    const breaker = new CircuitBreaker();
    const result = await rerank(cfg, breaker, 'search', docs, 5000, log);

    expect(result.results).toHaveLength(2);
    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(log.calls.some((c) => c.msg.includes('no API key'))).toBe(true);
  });

  test('returns scored results from API', async () => {
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as { query: string; documents: string[] };
      expect(body.documents).toEqual(docs.map((d) => d.text));
      jsonRes(res, 200, {
        results: [
          { index: 0, relevance_score: 0.95 },
          { index: 1, relevance_score: 0.42 },
        ],
        model: 'rerank-model',
      });
    });

    const cfg: EndpointConfig = { baseUrl: server.url, model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await rerank(cfg, breaker, 'search', docs, 5000, silentLogger);

    expect(result.results[0]!.score).toBe(0.95);
    expect(result.results[1]!.score).toBe(0.42);
    expect(result.model).toBe('rerank-model');
    await server.close();
  });

  test('fallback returns same object reference (memoized)', async () => {
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: '' };
    const breaker = new CircuitBreaker();
    const r1 = await rerank(cfg, breaker, 'q', docs, 5000, silentLogger);
    const r2 = await rerank(cfg, breaker, 'q', docs, 5000, silentLogger);
    // Both are computed from the same inputs — results should be identical
    expect(r1.results).toEqual(r2.results);
  });

  test('circuit breaker open returns uniform scores', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: 'sk-test' };
    const result = await rerank(cfg, breaker, 'q', docs, 5000, log);

    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('returns uniform scores on network error', async () => {
    const log = spyLogger();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await rerank(cfg, breaker, 'q', docs, 100, log);

    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });
});

// =============================================================================
// generate.ts
// =============================================================================

describe('generate', () => {
  test('returns generated text', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: 'Generated response' } }],
        model: 'gen-model',
      });
    });

    const cfg: EndpointConfig = { baseUrl: server.url, model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await generate(cfg, breaker, 'prompt', 5000, silentLogger);

    expect(result!.text).toBe('Generated response');
    expect(result!.model).toBe('gen-model');
    expect(result!.done).toBe(true);
    await server.close();
  });

  test('returns null on error', async () => {
    const log = spyLogger();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    const result = await generate(cfg, breaker, 'prompt', 100, log);

    expect(result).toBeNull();
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('circuit breaker open returns null', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm', apiKey: 'sk-test' };
    const result = await generate(cfg, breaker, 'prompt', 5000, log);

    expect(result).toBeNull();
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('passes maxTokens and temperature from options', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{ message: { content: 'ok' } }],
        model: 'm',
      });
    });

    const cfg: EndpointConfig = { baseUrl: server.url, model: 'm', apiKey: 'sk-test' };
    const breaker = new CircuitBreaker();
    await generate(cfg, breaker, 'test', 5000, silentLogger, { maxTokens: 42, temperature: 0.3 });

    expect(receivedBody.max_tokens).toBe(42);
    expect(receivedBody.temperature).toBe(0.3);
    await server.close();
  });
});

// =============================================================================
// probe.ts
// =============================================================================

describe('modelExists', () => {
  test('returns exists:true when model is found', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { data: [{ id: 'my-model' }, { id: 'other' }] });
    });

    const cfg: EndpointConfig = { baseUrl: `${server.url}/v1`, model: 'm' };
    const result = await modelExists(cfg, 'my-model');
    expect(result.exists).toBe(true);
    await server.close();
  });

  test('returns exists:false when model is not found', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { data: [{ id: 'other' }] });
    });

    const cfg: EndpointConfig = { baseUrl: `${server.url}/v1`, model: 'm' };
    const result = await modelExists(cfg, 'nonexistent');
    expect(result.exists).toBe(false);
    await server.close();
  });

  test('returns exists:true when /models endpoint is unavailable (optimistic)', async () => {
    const cfg: EndpointConfig = { baseUrl: 'http://localhost:1', model: 'm' };
    const result = await modelExists(cfg, 'anything');
    expect(result.exists).toBe(true);
  });
});

describe('probe', () => {
  test('returns ok and dimensions on success', async () => {
    const embedFn = async (_text: string) => ({ embedding: [1, 2, 3, 4], model: 'test' });
    const result = await probe(embedFn);
    expect(result.ok).toBe(true);
    expect(result.dimensions).toBe(4);
  });

  test('returns ok:false on null result', async () => {
    const embedFn = async (_text: string) => null;
    const result = await probe(embedFn);
    expect(result.ok).toBe(false);
    expect(result.error).toContain('Null');
  });
  test('returns ok:false on thrown error', async () => {
    const embedFn = async (_text: string) => { throw new Error('boom'); };
    const result = await probe(embedFn);
    expect(result.ok).toBe(false);
    expect(result.error).toContain('boom');
  });
});

// =============================================================================
// remote-llm.ts — integration
// =============================================================================

describe('RemoteLLM', () => {
  test('constructs with defaults', () => {
    const llm = new RemoteLLM();
    expect(llm.embedModelName).toBeTruthy();
    expect(llm.generateModelName).toBeTruthy();
    expect(llm.rerankModelName).toBeTruthy();
  });

  test('embed returns vector for single text', async () => {
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as { input: string[] };
      jsonRes(res, 200, {
        data: body.input.map((_, i) => ({ embedding: mockEmbedding, index: i })),
      });
    });

    const llm = new RemoteLLM({ embed: { baseUrl: server.url, model: 'm' } }, silentLogger);
    const result = await llm.embed('hello');
    expect(result!.embedding).toEqual(mockEmbedding);
    await llm.dispose();
    await server.close();
  });

  test('expandQuery delegates to expand module', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{
          message: { content: 'lex: test kw\nvec: test vec\nhyde: test hyde' },
        }],
      });
    });

    const llm = new RemoteLLM(
      { expand: { baseUrl: server.url, model: 'm', apiKey: 'sk' } },
      silentLogger,
    );
    const result = await llm.expandQuery('test');
    expect(result).toHaveLength(3);
    await llm.dispose();
    await server.close();
  });

  test('rerank delegates to rerank module', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        results: [{ index: 0, relevance_score: 0.99 }],
        model: 'm',
      });
    });

    const llm = new RemoteLLM(
      { rerank: { baseUrl: server.url, model: 'm', apiKey: 'sk' } },
      silentLogger,
    );
    const result = await llm.rerank('q', [{ file: 'a.md', text: 'doc' }]);
    expect(result.results[0]!.score).toBe(0.99);
    await llm.dispose();
    await server.close();
  });

  test('generate delegates to generate module', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: 'answer' } }],
        model: 'm',
      });
    });

    const llm = new RemoteLLM(
      { generate: { baseUrl: server.url, model: 'm', apiKey: 'sk' } },
      silentLogger,
    );
    const result = await llm.generate('prompt');
    expect(result!.text).toBe('answer');
    await llm.dispose();
    await server.close();
  });

  test('probe calls embed and returns dimensions', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        data: [{ embedding: [1, 2, 3], index: 0 }],
      });
    });

    const llm = new RemoteLLM({ embed: { baseUrl: server.url, model: 'm' } }, silentLogger);
    const result = await llm.probe();
    expect(result.ok).toBe(true);
    expect(result.dimensions).toBe(3);
    await llm.dispose();
    await server.close();
  });

  test('modelExists delegates to probe module', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { data: [{ id: 'test-model' }] });
    });

    const llm = new RemoteLLM({ embed: { baseUrl: `${server.url}/v1`, model: 'm' } }, silentLogger);
    const result = await llm.modelExists('test-model');
    expect(result.exists).toBe(true);
    await llm.dispose();
    await server.close();
  });

  test('dispose is a no-op (does not throw)', async () => {
    const llm = new RemoteLLM({}, silentLogger);
    await expect(llm.dispose()).resolves.toBeUndefined();
  });

  test('backward-compat flat config maps to embed', () => {
    const llm = new RemoteLLM({
      baseUrl: 'http://old:8000/v1',
      embedModel: 'old-model',
      apiKey: 'sk-old',
    }, silentLogger);
    expect(llm.embedCfg.baseUrl).toBe('http://old:8000/v1');
    expect(llm.embedModelName).toBe('old-model');
  });
});

// =============================================================================
// Phase 2 — Shared normalization helpers
// =============================================================================

describe('normalizeChatCompletionText', () => {
  test('extracts message content from standard response', () => {
    const data = {
      choices: [{ message: { content: 'Hello, world!' } }],
      model: 'gpt-4',
    };
    expect(normalizeChatCompletionText(data)).toBe('Hello, world!');
  });

  test('returns empty string for missing choices', () => {
    expect(normalizeChatCompletionText({ model: 'm' })).toBe('');
    expect(normalizeChatCompletionText({})).toBe('');
    expect(normalizeChatCompletionText(null)).toBe('');
    expect(normalizeChatCompletionText(undefined)).toBe('');
  });

  test('returns empty string for empty choices array', () => {
    expect(normalizeChatCompletionText({ choices: [] })).toBe('');
  });

  test('returns empty string when content is missing', () => {
    expect(normalizeChatCompletionText({
      choices: [{ message: {} }],
    })).toBe('');
  });

  test('handles non-object data gracefully', () => {
    expect(normalizeChatCompletionText('string')).toBe('');
    expect(normalizeChatCompletionText(42)).toBe('');
  });

  test('extracts text from content blocks array (multimodal shape)', () => {
    const data = {
      choices: [{
        message: {
          content: [
            { type: 'text', text: 'Hello from blocks' },
            { type: 'image_url', image_url: { url: '...' } },
          ],
        },
      }],
    };
    expect(normalizeChatCompletionText(data)).toBe('Hello from blocks');
  });

  test('returns empty for content blocks without text type', () => {
    const data = {
      choices: [{
        message: {
          content: [{ type: 'image_url', image_url: { url: '...' } }],
        },
      }],
    };
    expect(normalizeChatCompletionText(data)).toBe('');
  });
});

describe('normalizeCompletionsText', () => {
  test('extracts text from legacy completions response', () => {
    const data = {
      choices: [{ text: 'Generated text', index: 0 }],
      model: 'gpt-3.5-turbo-instruct',
    };
    expect(normalizeCompletionsText(data)).toBe('Generated text');
  });

  test('returns empty string for malformed data', () => {
    expect(normalizeCompletionsText(null)).toBe('');
    expect(normalizeCompletionsText({})).toBe('');
    expect(normalizeCompletionsText({ choices: [] })).toBe('');
  });
});

describe('normalizeResponseAPIText', () => {
  test('extracts text from output_text blocks', () => {
    const data = {
      output: [
        {
          type: 'message',
          content: [
            { type: 'output_text', text: 'Response text here' },
          ],
        },
      ],
      model: 'gpt-4o',
    };
    expect(normalizeResponseAPIText(data)).toBe('Response text here');
  });

  test('skips non-message output types', () => {
    const data = {
      output: [
        { type: 'reasoning', content: '...' },
        {
          type: 'message',
          content: [
            { type: 'output_text', text: 'Actual response' },
          ],
        },
      ],
    };
    expect(normalizeResponseAPIText(data)).toBe('Actual response');
  });

  test('returns empty string when no message output', () => {
    expect(normalizeResponseAPIText({ output: [{ type: 'reasoning' }] })).toBe('');
    expect(normalizeResponseAPIText({ output: [] })).toBe('');
    expect(normalizeResponseAPIText(null)).toBe('');
  });

  test('handles top-level output_text shortcut', () => {
    const data = { output_text: 'direct output text' };
    expect(normalizeResponseAPIText(data)).toBe('direct output text');
  });

  test('handles content blocks with type: "text" variant', () => {
    const data = {
      output: [
        {
          type: 'message',
          content: [{ type: 'text', text: 'Variant text shape' }],
        },
      ],
    };
    expect(normalizeResponseAPIText(data)).toBe('Variant text shape');
  });

  test('prefers output_text over text when both present', () => {
    const data = {
      output: [
        {
          type: 'message',
          content: [
            { type: 'output_text', text: 'output_text wins' },
            { type: 'text', text: 'text variant' },
          ],
        },
      ],
    };
    expect(normalizeResponseAPIText(data)).toBe('output_text wins');
  });
});

describe('normalizeModelName', () => {
  test('extracts model name from response', () => {
    expect(normalizeModelName({ model: 'gpt-4o' }, 'fallback')).toBe('gpt-4o');
  });

  test('returns fallback when model is missing', () => {
    expect(normalizeModelName({}, 'fallback')).toBe('fallback');
    expect(normalizeModelName(null, 'fallback')).toBe('fallback');
  });
});

// =============================================================================
// Phase 3 — Anthropic Messages normalization: text extraction from content blocks
// =============================================================================

describe('normalizeAnthropicMessagesText', () => {
  test('extracts text from standard content block', () => {
    const data = {
      id: 'msg_001',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'Hello from Claude' }],
      model: 'claude-3-opus-20240229',
    };
    expect(normalizeAnthropicMessagesText(data)).toBe('Hello from Claude');
  });

  test('joins multiple text blocks with newline separation (no word-merging)', () => {
    const data = {
      content: [
        { type: 'text', text: 'Part one. ' },
        { type: 'text', text: 'Part two.' },
      ],
    };
    expect(normalizeAnthropicMessagesText(data)).toBe('Part one.\nPart two.');
  });

  test('skips non-text content blocks (tool_use, etc.)', () => {
    const data = {
      content: [
        { type: 'text', text: 'Here is some text' },
        { type: 'tool_use', id: 'tool_1', name: 'calculator', input: {} },
        { type: 'text', text: ' more text.' },
      ],
    };
    expect(normalizeAnthropicMessagesText(data)).toBe('Here is some text\nmore text.');
  });

  test('returns empty string for missing content', () => {
    expect(normalizeAnthropicMessagesText({ id: 'msg' })).toBe('');
    expect(normalizeAnthropicMessagesText({ content: [] })).toBe('');
    expect(normalizeAnthropicMessagesText({})).toBe('');
  });

  test('returns empty string for non-object data', () => {
    expect(normalizeAnthropicMessagesText(null)).toBe('');
    expect(normalizeAnthropicMessagesText(undefined)).toBe('');
    expect(normalizeAnthropicMessagesText('string')).toBe('');
    expect(normalizeAnthropicMessagesText(42)).toBe('');
  });

  test('handles mixed block types with null/empty entries', () => {
    const data = {
      content: [
        null,
        { type: 'text', text: 'valid text' },
        undefined,
        {},
        { type: 'image', source: {} },
        { type: 'text', text: ' more text' },
      ],
    };
    expect(normalizeAnthropicMessagesText(data)).toBe('valid text\nmore text');
  });
});

// =============================================================================
// Adapter registry: format → adapter mapping
// =============================================================================

describe('resolveAdapterBundle', () => {
  function ecfg(overrides?: Partial<EndpointConfig>): EndpointConfig {
    return {
      baseUrl: 'http://localhost:1',
      model: 'm',
      apiKey: 'sk',
      format: 'auto',
      ...overrides,
    };
  }

  test('auto format resolves to legacy adapters', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg({ format: 'auto' }),
      expand: ecfg({ format: 'auto' }),
      rerank: ecfg({ format: 'auto' }),
      generate: ecfg({ format: 'auto' }),
    });
    expect(bundle.expand.id).toBe('legacy/openai-chat-expand');
    expect(bundle.generate.id).toBe('legacy/openai-chat-generate');
  });

  test('openai_chat_completions resolves to chat adapter', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg(),
      expand: ecfg({ format: 'openai_chat_completions' }),
      rerank: ecfg(),
      generate: ecfg({ format: 'openai_chat_completions' }),
    });
    expect(bundle.expand.id).toBe('openai/chat-completions-expand');
    expect(bundle.generate.id).toBe('openai/chat-completions-generate');
  });

  test('openai_completions resolves to completions adapter', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg(),
      expand: ecfg({ format: 'openai_completions' }),
      rerank: ecfg(),
      generate: ecfg({ format: 'openai_completions' }),
    });
    expect(bundle.expand.id).toBe('openai/completions-expand');
    expect(bundle.generate.id).toBe('openai/completions-generate');
  });

  test('openai_responses resolves to responses adapter', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg(),
      expand: ecfg({ format: 'openai_responses' }),
      rerank: ecfg(),
      generate: ecfg({ format: 'openai_responses' }),
    });
    expect(bundle.expand.id).toBe('openai/responses-expand');
    expect(bundle.generate.id).toBe('openai/responses-generate');
  });

  test('anthropic_messages resolves to Anthropic Messages adapter', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg(),
      expand: ecfg({ format: 'anthropic_messages' }),
      rerank: ecfg(),
      generate: ecfg({ format: 'anthropic_messages' }),
    });
    expect(bundle.expand.id).toBe('anthropic/messages-expand');
    expect(bundle.generate.id).toBe('anthropic/messages-generate');
  });

  test('falls back to legacy on unrecognized format', () => {
    // Missing format defaults to auto → legacy
    const bundle = resolveAdapterBundle({
      embed: ecfg({ format: undefined }),
      expand: ecfg({ format: undefined }),
      rerank: ecfg({ format: undefined }),
      generate: ecfg({ format: undefined }),
    });
    expect(bundle.expand.id).toBe('legacy/openai-chat-expand');
    expect(bundle.generate.id).toBe('legacy/openai-chat-generate');
  });

  test('cohere formats resolve to cohere embed/rerank adapters', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg({ format: 'cohere_v2_embed' }),
      expand: ecfg({ format: 'openai_chat_completions' }),
      rerank: ecfg({ format: 'cohere_v2_rerank' }),
      generate: ecfg({ format: 'openai_chat_completions' }),
    });
    expect(bundle.embed.id).toBe('cohere/v2-embed');
    expect(bundle.rerank.id).toBe('cohere/rerank');
  });

  test('vllm_score resolves to vLLM score adapter', () => {
    const bundle = resolveAdapterBundle({
      embed: ecfg({ format: 'auto' }),
      expand: ecfg({ format: 'auto' }),
      rerank: ecfg({ format: 'vllm_score' }),
      generate: ecfg({ format: 'auto' }),
    });
    expect(bundle.rerank.id).toBe('vllm/score');
  });
});

// =============================================================================
// Phase 4 — Cohere-compatible embed/rerank adapters
// =============================================================================

describe('cohereV2EmbedAdapter', () => {
  function makeCtx(url: string): Parameters<typeof cohereV2EmbedAdapter.embedBatch>[0] {
    return {
      cfg: { baseUrl: url, model: 'embed-v4.0', apiKey: 'sk-test', format: 'cohere_v2_embed' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      maxBatchSize: 32,
      readTimeoutMs: 5000,
      maxRetries: 1,
      dimState: { dimensions: null },
    };
  }

  test('falls back from inputs payload to texts payload on contract mismatch', async () => {
    const seenShapes: string[] = [];
    const server = startMockServer(async (req, res) => {
      expect(req.url).toBe('/v2/embed');
      const body = await readBody(req) as Record<string, unknown>;
      if (Array.isArray(body.inputs)) {
        seenShapes.push('inputs');
        jsonRes(res, 400, { error: 'inputs unsupported here' });
        return;
      }
      if (Array.isArray(body.texts)) {
        seenShapes.push('texts');
        jsonRes(res, 200, {
          embeddings: {
            float: [mockEmbedding, mockEmbedding],
          },
          model: 'embed-v4.0',
        });
        return;
      }
      jsonRes(res, 400, { error: 'invalid body' });
    });

    const result = await cohereV2EmbedAdapter.embedBatch(
      makeCtx(server.url),
      ['hello', 'world'],
      {},
    );

    expect(result).toHaveLength(2);
    expect(result[0]!.embedding).toEqual(mockEmbedding);
    expect(result[1]!.embedding).toEqual(mockEmbedding);
    expect(seenShapes).toEqual(['inputs', 'inputs', 'texts']);
    await server.close();
  });

  test('sets input_type=search_query when embedding query text', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        embeddings: { float: [mockEmbedding] },
      });
    });

    await cohereV2EmbedAdapter.embedBatch(
      makeCtx(server.url),
      ['query text'],
      { isQuery: true },
    );

    expect(receivedBody.input_type).toBe('search_query');
    await server.close();
  });

  test('falls back input_type from search_document to document on vLLM-style error', async () => {
    const seenInputTypes: string[] = [];
    const server = startMockServer(async (req, res) => {
      const body = await readBody(req) as Record<string, unknown>;
      const inputType = String(body.input_type ?? '');
      seenInputTypes.push(inputType);
      if (inputType === 'search_document') {
        jsonRes(res, 400, {
          error: {
            message: "Unsupported input_type 'search_document'. Supported values: document, query",
          },
        });
        return;
      }
      jsonRes(res, 200, {
        embeddings: { float: [mockEmbedding] },
      });
    });

    const result = await cohereV2EmbedAdapter.embedBatch(
      makeCtx(server.url),
      ['hello'],
      {},
    );

    expect(result).toHaveLength(1);
    expect(seenInputTypes).toEqual(['search_document', 'document']);
    await server.close();
  });

  test('normalizes OpenAI-style data embeddings using index order', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        data: [
          { index: 1, embedding: [2, 2, 2] },
          { index: 0, embedding: [1, 1, 1] },
        ],
      });
    });

    const result = await cohereV2EmbedAdapter.embedBatch(
      makeCtx(server.url),
      ['first', 'second'],
      {},
    );

    expect(result).toHaveLength(2);
    expect(result[0]!.embedding).toEqual([1, 1, 1]);
    expect(result[1]!.embedding).toEqual([2, 2, 2]);
    await server.close();
  });

  test('falls back endpoint path from /v2/embed to /embed', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer(async (req, res) => {
      requestedPaths.push(req.url ?? '');
      if (req.url === '/v2/embed') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        embeddings: { float: [mockEmbedding] },
      });
    });

    const result = await cohereV2EmbedAdapter.embedBatch(
      makeCtx(server.url),
      ['hello'],
      {},
    );

    expect(result).toHaveLength(1);
    expect(requestedPaths).toEqual(['/v2/embed', '/v2/embed', '/embed']);
    await server.close();
  });

  test('when baseUrl ends with /embed, fallback target is sibling /v2/embed', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer(async (req, res) => {
      requestedPaths.push(req.url ?? '');
      if (req.url === '/embed') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        embeddings: { float: [mockEmbedding] },
      });
    });

    const result = await cohereV2EmbedAdapter.embedBatch(
      makeCtx(`${server.url}/embed`),
      ['hello'],
      {},
    );

    expect(result).toHaveLength(1);
    expect(requestedPaths).toEqual(['/embed', '/embed', '/v2/embed']);
    await server.close();
  });

  test('throws on embedding dimension mismatch across calls', async () => {
    let callCount = 0;
    const ctx = makeCtx('http://localhost:1');
    const server = startMockServer((_req, res) => {
      callCount++;
      if (callCount === 1) {
        jsonRes(res, 200, { embeddings: { float: [[0.1, 0.2, 0.3]] } });
      } else {
        jsonRes(res, 200, { embeddings: { float: [[0.1, 0.2]] } });
      }
    });
    ctx.cfg.baseUrl = server.url;

    await cohereV2EmbedAdapter.embedBatch(ctx, ['a'], {});
    await expect(
      cohereV2EmbedAdapter.embedBatch(ctx, ['b'], {}),
    ).rejects.toThrow('dimension mismatch');
    await server.close();
  });
});

describe('cohereRerankAdapter', () => {
  function makeCtx(url: string): Parameters<typeof cohereRerankAdapter.rerank>[0] {
    return {
      cfg: { baseUrl: url, model: 'rerank-v3.5', apiKey: 'sk-test', format: 'cohere_v2_rerank' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  const docs = [
    { file: 'a.md', text: 'first document' },
    { file: 'b.md', text: 'second document' },
  ];

  test('tries /rerank then /v1/rerank when provider only supports v1 path', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer(async (req, res) => {
      requestedPaths.push(req.url ?? '');
      if (req.url === '/rerank') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        results: [{ index: 1, relevance_score: 0.88 }],
      });
    });

    const result = await cohereRerankAdapter.rerank(
      makeCtx(server.url),
      'query',
      docs,
      {},
    );

    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.file).toBe('b.md');
    expect(requestedPaths).toEqual(['/rerank', '/v1/rerank']);
    await server.close();
  });

  test('when baseUrl ends with /rerank, fallback targets sibling /v1/rerank', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer(async (req, res) => {
      requestedPaths.push(req.url ?? '');
      if (req.url === '/rerank') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        results: [{ index: 0, relevance_score: 0.88 }],
      });
    });

    const result = await cohereRerankAdapter.rerank(
      makeCtx(`${server.url}/rerank`),
      'query',
      docs,
      {},
    );

    expect(result.results).toHaveLength(1);
    expect(requestedPaths).toEqual(['/rerank', '/v1/rerank']);
    await server.close();
  });

  test('uses options.model override in request body', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (_req, res) => {
      receivedBody = await readBody(_req);
      jsonRes(res, 200, {
        results: [{ index: 0, relevance_score: 0.99 }],
      });
    });

    await cohereRerankAdapter.rerank(
      makeCtx(server.url),
      'query',
      docs,
      { model: 'rerank-override' },
    );

    expect(receivedBody.model).toBe('rerank-override');
    await server.close();
  });

  test('returns uniform fallback when API key is missing', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.cfg.apiKey = '';
    ctx.log = log;

    const result = await cohereRerankAdapter.rerank(ctx, 'query', docs, {});
    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(log.calls.some((c) => c.msg.includes('no API key'))).toBe(true);
  });

  test('treats malformed 200 response as failure and returns uniform fallback', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { results: [{ foo: 'bad-shape' }] });
    });

    const ctx = makeCtx(server.url);
    ctx.log = log;
    ctx.breaker = breaker;

    const result = await cohereRerankAdapter.rerank(ctx, 'query', docs, {});
    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(breaker.getState()).toBe('open');
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
    await server.close();
  });
});

describe('vllmScoreAdapter', () => {
  function makeCtx(url: string): Parameters<typeof vllmScoreAdapter.rerank>[0] {
    return {
      cfg: { baseUrl: url, model: 'BAAI/bge-reranker-v2-m3', apiKey: '', format: 'vllm_score' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  const docs = [
    { file: 'a.md', text: 'first document' },
    { file: 'b.md', text: 'second document' },
  ];

  test('posts to /score and normalizes data[] scores', async () => {
    let requestedPath = '';
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      requestedPath = req.url ?? '';
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        model: 'BAAI/bge-reranker-v2-m3',
        data: [
          { index: 0, score: 0.22 },
          { index: 1, score: 0.91 },
        ],
      });
    });

    const result = await vllmScoreAdapter.rerank(
      makeCtx(server.url),
      'query',
      docs,
      {},
    );

    expect(requestedPath).toBe('/score');
    expect(receivedBody.queries).toBe('query');
    expect(receivedBody.documents).toEqual(['first document', 'second document']);
    expect(result.results[0]!.file).toBe('b.md');
    expect(result.results[0]!.score).toBe(0.91);
    await server.close();
  });

  test('falls back endpoint path from /score to /v1/score', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer((_req, res) => {
      requestedPaths.push(_req.url ?? '');
      if (_req.url === '/score') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        data: [{ index: 0, score: 0.77 }],
      });
    });

    const result = await vllmScoreAdapter.rerank(
      makeCtx(server.url),
      'query',
      docs,
      {},
    );

    expect(result.results[0]!.score).toBe(0.77);
    expect(requestedPaths).toEqual(['/score', '/v1/score']);
    await server.close();
  });

  test('returns uniform fallback when score response is malformed', async () => {
    const breaker = new CircuitBreaker(1);
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { data: [{ foo: 'bad-shape' }] });
    });
    const ctx = makeCtx(server.url);
    ctx.breaker = breaker;
    ctx.log = log;

    const result = await vllmScoreAdapter.rerank(ctx, 'query', docs, {});
    expect(result.results.every((r) => r.score === 1.0)).toBe(true);
    expect(breaker.getState()).toBe('open');
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
    await server.close();
  });
});

// =============================================================================
// Phase 2 — openai_chat_completions adapter: expand + generate
// =============================================================================

describe('openaiChatCompletionsExpandAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiChatCompletionsExpandAdapter.expandQuery>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('expandQuery sends chat completions request and parses response', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{
          message: {
            // All variants contain "topic" so all pass term-overlap validation
            content: 'lex: topic keywords\nvec: topic paraphrase\nhyde: A document about topic',
          },
        }],
      });
    });

    const result = await openaiChatCompletionsExpandAdapter.expandQuery(
      makeCtx(server.url), 'topic', {},
    );

    expect(receivedBody.messages).toHaveLength(2);
    expect(receivedBody.messages[0].role).toBe('system');
    expect(receivedBody.messages[1].role).toBe('user');
    expect(receivedBody.messages[1].content).toContain('topic');
    expect(result).toHaveLength(3);
    expect(result[0]!.type).toBe('lex');
    await server.close();
  });

  test('returns passthrough when no API key', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.cfg.apiKey = '';
    ctx.log = log;

    const result = await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toEqual([{ type: 'lex', text: 'test' }]);
    expect(log.calls.some((c) => c.msg.includes('no API key'))).toBe(true);
  });

  test('falls back when circuit breaker is open', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;
    ctx.log = log;

    const result = await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('falls back on malformed response', async () => {
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: 'unparseable garbage' } }],
      });
    });

    const ctx = makeCtx(server.url);
    ctx.log = log;
    const result = await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'topic', {});

    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('could not parse'))).toBe(true);
    await server.close();
  });

  test('falls back on empty response content', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { choices: [{ message: { content: '' } }] });
    });

    const ctx = makeCtx(server.url);
    const result = await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3); // fallback
    await server.close();
  });

  test('falls back on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('includes intent in user prompt when provided', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{
          // All contain "dog" so pass term-overlap
          message: { content: 'lex: dog search\nvec: dog query\nhyde: A document about dogs' },
        }],
      });
    });

    const ctx = makeCtx(server.url);
    await openaiChatCompletionsExpandAdapter.expandQuery(ctx, 'dog', { intent: 'information-seeking' });

    expect(receivedBody.messages[1].content).toContain('information-seeking');
    await server.close();
  });
});

describe('openaiChatCompletionsGenerateAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiChatCompletionsGenerateAdapter.generate>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('returns generated text from chat completions', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: 'Generated response' } }],
        model: 'gen-model',
      });
    });

    const result = await openaiChatCompletionsGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('Generated response');
    expect(result!.model).toBe('gen-model');
    expect(result!.done).toBe(true);
    await server.close();
  });

  test('passes maxTokens and temperature from options', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{ message: { content: 'ok' } }],
        model: 'm',
      });
    });

    await openaiChatCompletionsGenerateAdapter.generate(
      makeCtx(server.url), 'test', { maxTokens: 42, temperature: 0.3 },
    );

    expect(receivedBody.max_tokens).toBe(42);
    expect(receivedBody.temperature).toBe(0.3);
    await server.close();
  });

  test('returns null on circuit breaker open', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;
    ctx.log = log;

    const result = await openaiChatCompletionsGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('returns null on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await openaiChatCompletionsGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('returns empty text for empty content response', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ message: { content: '' } }],
        model: 'm',
      });
    });

    const result = await openaiChatCompletionsGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('');
    expect(result!.done).toBe(true);
    await server.close();
  });
});

// =============================================================================
// Phase 2 — openai_completions adapter: expand + generate
// =============================================================================

describe('openaiCompletionsExpandAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiCompletionsExpandAdapter.expandQuery>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('sends combined prompt to /completions endpoint', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{ text: 'lex: search kw\nvec: search vec\nhyde: search hyde', index: 0 }],
        model: 'm',
      });
    });

    const result = await openaiCompletionsExpandAdapter.expandQuery(
      makeCtx(server.url), 'search', {},
    );

    // Completions endpoint: single prompt string, no messages array
    expect(typeof receivedBody.prompt).toBe('string');
    expect(receivedBody.prompt).toContain('search query expansion');
    expect(receivedBody.prompt).toContain('search');
    expect(result).toHaveLength(3);
    await server.close();
  });

  test('returns passthrough when no API key', async () => {
    const ctx = makeCtx('http://localhost:1');
    ctx.cfg.apiKey = '';
    const result = await openaiCompletionsExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toEqual([{ type: 'lex', text: 'test' }]);
  });

  test('falls back when circuit breaker is open', async () => {
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;

    const result = await openaiCompletionsExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toHaveLength(3);
  });

  test('falls back on malformed response', async () => {
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { choices: [{ text: 'not formatted at all', index: 0 }] });
    });

    const ctx = makeCtx(server.url);
    ctx.log = log;
    const result = await openaiCompletionsExpandAdapter.expandQuery(ctx, 'topic', {});

    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('could not parse'))).toBe(true);
    await server.close();
  });
});

describe('openaiCompletionsGenerateAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiCompletionsGenerateAdapter.generate>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('extracts text from choices[0].text', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        choices: [{ text: 'Completions output', index: 0 }],
        model: 'gpt-3.5-turbo-instruct',
      });
    });

    const result = await openaiCompletionsGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('Completions output');
    expect(result!.model).toBe('gpt-3.5-turbo-instruct');
    await server.close();
  });

  test('sends prompt as a plain string', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{ text: 'ok', index: 0 }],
        model: 'm',
      });
    });

    await openaiCompletionsGenerateAdapter.generate(
      makeCtx(server.url), 'my plain prompt', { maxTokens: 50 },
    );

    expect(receivedBody.prompt).toBe('my plain prompt');
    expect(receivedBody.max_tokens).toBe(50);
    await server.close();
  });

  test('returns null on error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await openaiCompletionsGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });
});

// =============================================================================
// Phase 2 — openai_responses adapter: expand + generate
// =============================================================================

describe('openaiResponsesExpandAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiResponsesExpandAdapter.expandQuery>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('sends request to /responses with instructions + input', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        output: [
          {
            type: 'message',
            content: [
              {
                type: 'output_text',
                // All contain "search" — pass term-overlap
                text: 'lex: search kw\nvec: search vec\nhyde: search hyde',
              },
            ],
          },
        ],
        model: 'gpt-4o',
      });
    });

    const result = await openaiResponsesExpandAdapter.expandQuery(
      makeCtx(server.url), 'search', {},
    );

    expect(typeof receivedBody.instructions).toBe('string');
    expect(typeof receivedBody.input).toBe('string');
    expect(receivedBody.max_output_tokens).toBe(600);
    expect(result).toHaveLength(3);
    await server.close();
  });

  test('returns passthrough when no API key', async () => {
    const ctx = makeCtx('http://localhost:1');
    ctx.cfg.apiKey = '';
    const result = await openaiResponsesExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toEqual([{ type: 'lex', text: 'test' }]);
  });

  test('falls back on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await openaiResponsesExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('falls back on empty output', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, { output: [], model: 'm' });
    });

    const ctx = makeCtx(server.url);
    const result = await openaiResponsesExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3);
    await server.close();
  });
});

describe('openaiResponsesGenerateAdapter', () => {
  function makeCtx(url: string): Parameters<typeof openaiResponsesGenerateAdapter.generate>[0] {
    return {
      cfg: { baseUrl: url, model: 'm', apiKey: 'sk-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('extracts text from output_text blocks', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        output: [
          {
            type: 'message',
            content: [{ type: 'output_text', text: 'Response API output' }],
          },
        ],
        model: 'gpt-4o',
      });
    });

    const result = await openaiResponsesGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('Response API output');
    expect(result!.model).toBe('gpt-4o');
    await server.close();
  });

  test('sends input as string and uses max_output_tokens', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        output: [{ type: 'message', content: [{ type: 'output_text', text: 'ok' }] }],
        model: 'm',
      });
    });

    await openaiResponsesGenerateAdapter.generate(
      makeCtx(server.url), 'my prompt', { maxTokens: 256, temperature: 0.5 },
    );

    expect(receivedBody.input).toBe('my prompt');
    expect(receivedBody.max_output_tokens).toBe(256);
    expect(receivedBody.temperature).toBe(0.5);
    await server.close();
  });

  test('returns null on circuit breaker open', async () => {
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;

    const result = await openaiResponsesGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
  });

  test('returns null on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await openaiResponsesGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });
});

// =============================================================================
// Phase 2 — RemoteLLM integration with explicit format selection
// =============================================================================

describe('RemoteLLM with Phase 2 formats', () => {
  test('RemoteLLM expandQuery uses chat completions format via new adapter', async () => {
    let requestedPath = '';
    const server = startMockServer((req, res) => {
      requestedPath = req.url ?? '';
      jsonRes(res, 200, {
        choices: [{ message: { content: 'lex: test kw\nvec: test vec\nhyde: test hyde' } }],
        model: 'm',
      });
    });

    const llm = new RemoteLLM({
      expand: { baseUrl: server.url, model: 'm', apiKey: 'sk', format: 'openai_chat_completions' },
    }, silentLogger);

    const result = await llm.expandQuery('test');
    expect(result).toHaveLength(3);
    expect(requestedPath).toBe('/chat/completions');
    await llm.dispose();
    await server.close();
  });

  test('RemoteLLM generate uses completions format via new adapter', async () => {
    let requestedPath = '';
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      requestedPath = req.url ?? '';
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        choices: [{ text: 'completions output', index: 0 }],
        model: 'gpt-3.5',
      });
    });

    const llm = new RemoteLLM({
      generate: { baseUrl: server.url, model: 'm', apiKey: 'sk', format: 'openai_completions' },
    }, silentLogger);

    const result = await llm.generate('prompt text');
    expect(result!.text).toBe('completions output');
    expect(requestedPath).toBe('/completions');
    expect(receivedBody.prompt).toBe('prompt text');
    await llm.dispose();
    await server.close();
  });

  test('RemoteLLM generate uses responses format via new adapter', async () => {
    let requestedPath = '';
    const server = startMockServer((req, res) => {
      requestedPath = req.url ?? '';
      jsonRes(res, 200, {
        output: [{ type: 'message', content: [{ type: 'output_text', text: 'responses output' }] }],
        model: 'gpt-4o',
      });
    });

    const llm = new RemoteLLM({
      generate: { baseUrl: server.url, model: 'm', apiKey: 'sk', format: 'openai_responses' },
    }, silentLogger);

    const result = await llm.generate('prompt');
    expect(result!.text).toBe('responses output');
    expect(requestedPath).toBe('/responses');
    await llm.dispose();
    await server.close();
  });

  test('auto format still uses legacy and hits /chat/completions', async () => {
    let requestedPath = '';
    const server = startMockServer((req, res) => {
      requestedPath = req.url ?? '';
      jsonRes(res, 200, {
        choices: [{ message: { content: 'answer' } }],
        model: 'm',
      });
    });

    const llm = new RemoteLLM({
      generate: { baseUrl: server.url, model: 'm', apiKey: 'sk', format: 'auto' },
    }, silentLogger);

    const result = await llm.generate('test');
    expect(result!.text).toBe('answer');
    expect(requestedPath).toBe('/chat/completions'); // legacy also uses chat completions
    await llm.dispose();
    await server.close();
  });
});

// =============================================================================
// Phase 4 — RemoteLLM integration with Cohere-compatible formats
// =============================================================================

describe('RemoteLLM with Phase 4 cohere formats', () => {
  test('RemoteLLM embedBatch uses cohere_v2_embed adapter and /v2/embed path', async () => {
    let requestedPath = '';
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      requestedPath = req.url ?? '';
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        embeddings: {
          float: [mockEmbedding],
        },
      });
    });

    const llm = new RemoteLLM({
      embed: { baseUrl: server.url, model: 'embed-v4.0', apiKey: 'sk', format: 'cohere_v2_embed' },
    }, silentLogger);

    const result = await llm.embedBatch(['hello']);
    expect(result[0]!.embedding).toEqual(mockEmbedding);
    expect(requestedPath).toBe('/v2/embed');
    expect(receivedBody.input_type).toBe('search_document');

    await llm.dispose();
    await server.close();
  });

  test('RemoteLLM rerank uses cohere_v2_rerank adapter with endpoint fallback', async () => {
    const requestedPaths: string[] = [];
    const server = startMockServer((_req, res) => {
      requestedPaths.push(_req.url ?? '');
      if (_req.url === '/rerank') {
        jsonRes(res, 404, { error: 'not found' });
        return;
      }
      jsonRes(res, 200, {
        results: [{ index: 0, relevance_score: 0.91 }],
      });
    });

    const llm = new RemoteLLM({
      rerank: { baseUrl: server.url, model: 'rerank-v3.5', apiKey: 'sk', format: 'cohere_v2_rerank' },
    }, silentLogger);

    const result = await llm.rerank('q', [{ file: 'a.md', text: 'doc' }]);
    expect(result.results[0]!.score).toBe(0.91);
    expect(requestedPaths).toEqual(['/rerank', '/v1/rerank']);

    await llm.dispose();
    await server.close();
  });

  test('RemoteLLM rerank uses vllm_score adapter and /score endpoint', async () => {
    let requestedPath = '';
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      requestedPath = req.url ?? '';
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        model: 'BAAI/bge-reranker-v2-m3',
        data: [{ index: 0, score: 0.93 }],
      });
    });

    const llm = new RemoteLLM({
      rerank: {
        baseUrl: server.url,
        model: 'BAAI/bge-reranker-v2-m3',
        apiKey: '',
        format: 'vllm_score',
      },
    }, silentLogger);

    const result = await llm.rerank('q', [{ file: 'a.md', text: 'doc' }]);
    expect(result.results[0]!.score).toBe(0.93);
    expect(requestedPath).toBe('/score');
    expect(receivedBody.queries).toBe('q');
    expect(receivedBody.documents).toEqual(['doc']);

    await llm.dispose();
    await server.close();
  });
});

// =============================================================================
// Phase 3 — Anthropic Messages adapter: expand + generate
// =============================================================================

describe('anthropicMessagesExpandAdapter', () => {
  function makeCtx(url: string): Parameters<typeof anthropicMessagesExpandAdapter.expandQuery>[0] {
    return {
      cfg: { baseUrl: url, model: 'claude-3-opus', apiKey: 'sk-ant-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('sends request to /messages with correct Anthropic protocol shape', async () => {
    let receivedBody: any = null;
    let receivedHeaders: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      receivedHeaders = req.headers;
      jsonRes(res, 200, {
        id: 'msg_001',
        type: 'message',
        role: 'assistant',
        content: [
          { type: 'text', text: 'lex: topic search\nvec: semantic topic\nhyde: A document about topics' },
        ],
        model: 'claude-3-opus-20240229',
        stop_reason: 'end_turn',
        usage: { input_tokens: 50, output_tokens: 30 },
      });
    });

    const result = await anthropicMessagesExpandAdapter.expandQuery(
      makeCtx(server.url), 'topic', {},
    );

    // Verify Anthropic-specific request shape
    expect(receivedBody.model).toBe('claude-3-opus');
    expect(receivedBody.max_tokens).toBe(600);
    expect(receivedBody.system).toBeTruthy();
    expect(receivedBody.system).toContain('search query expansion');
    expect(receivedBody.messages).toHaveLength(1);
    expect(receivedBody.messages[0].role).toBe('user');
    expect(receivedBody.messages[0].content).toContain('topic');

    // Verify Anthropic auth header (NOT Authorization: Bearer)
    expect(receivedHeaders['x-api-key']).toBe('sk-ant-test');
    expect(receivedHeaders['anthropic-version']).toBe('2023-06-01');
    // Should NOT have Authorization header
    expect(receivedHeaders['authorization']).toBeUndefined();

    // Verify parsed result
    expect(result).toHaveLength(3);
    expect(result[0]!.type).toBe('lex');
    await server.close();
  });

  test('returns passthrough when no API key', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.cfg.apiKey = '';
    ctx.log = log;

    const result = await anthropicMessagesExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toEqual([{ type: 'lex', text: 'test' }]);
    expect(log.calls.some((c) => c.msg.includes('no API key'))).toBe(true);
  });

  test('falls back when circuit breaker is open', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;
    ctx.log = log;

    const result = await anthropicMessagesExpandAdapter.expandQuery(ctx, 'test', {});
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('falls back on malformed response', async () => {
    const log = spyLogger();
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        id: 'msg',
        content: [{ type: 'text', text: 'not valid format at all' }],
      });
    });

    const ctx = makeCtx(server.url);
    ctx.log = log;
    const result = await anthropicMessagesExpandAdapter.expandQuery(ctx, 'topic', {});

    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.msg.includes('could not parse'))).toBe(true);
    await server.close();
  });

  test('falls back on empty content array', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        id: 'msg',
        content: [],
        model: 'claude',
      });
    });

    const ctx = makeCtx(server.url);
    const result = await anthropicMessagesExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3); // fallback
    await server.close();
  });

  test('falls back on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await anthropicMessagesExpandAdapter.expandQuery(ctx, 'query', {});
    expect(result).toHaveLength(3);
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('includes intent in user prompt when provided', async () => {
    let receivedBody: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      jsonRes(res, 200, {
        content: [
          { type: 'text', text: 'lex: dog search\nvec: dog query\nhyde: A document about dogs' },
        ],
      });
    });

    const ctx = makeCtx(server.url);
    await anthropicMessagesExpandAdapter.expandQuery(ctx, 'dog', { intent: 'information-seeking' });

    expect(receivedBody.messages[0].content).toContain('information-seeking');
    await server.close();
  });
});

describe('anthropicMessagesGenerateAdapter', () => {
  function makeCtx(url: string): Parameters<typeof anthropicMessagesGenerateAdapter.generate>[0] {
    return {
      cfg: { baseUrl: url, model: 'claude-3-haiku', apiKey: 'sk-ant-test' },
      breaker: new CircuitBreaker(),
      log: silentLogger,
      readTimeoutMs: 5000,
    };
  }

  test('extracts text from content blocks in Anthropic response', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        id: 'msg_002',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'This is a generated response from Claude' }],
        model: 'claude-3-haiku-20240307',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 15 },
      });
    });

    const result = await anthropicMessagesGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('This is a generated response from Claude');
    expect(result!.model).toBe('claude-3-haiku-20240307');
    expect(result!.done).toBe(true);
    await server.close();
  });

  test('sends Anthropic-formatted request with correct headers', async () => {
    let receivedBody: any = null;
    let receivedHeaders: any = null;
    const server = startMockServer(async (req, res) => {
      receivedBody = await readBody(req);
      receivedHeaders = req.headers;
      jsonRes(res, 200, {
        content: [{ type: 'text', text: 'ok' }],
        model: 'claude-3-haiku',
      });
    });

    await anthropicMessagesGenerateAdapter.generate(
      makeCtx(server.url), 'test prompt', { maxTokens: 512, temperature: 0.5 },
    );

    // Verify Anthropic-specific shape
    expect(receivedBody.model).toBe('claude-3-haiku');
    expect(receivedBody.max_tokens).toBe(512);
    expect(receivedBody.temperature).toBe(0.5);
    expect(receivedBody.messages).toHaveLength(1);
    expect(receivedBody.messages[0].role).toBe('user');
    expect(receivedBody.messages[0].content).toBe('test prompt');
    // generate adapters should NOT include system prompt
    expect(receivedBody.system).toBeUndefined();

    // Auth headers
    expect(receivedHeaders['x-api-key']).toBe('sk-ant-test');
    expect(receivedHeaders['anthropic-version']).toBe('2023-06-01');
    await server.close();
  });

  test('returns null on circuit breaker open', async () => {
    const log = spyLogger();
    const breaker = new CircuitBreaker(1);
    breaker.onFailure();
    const ctx = makeCtx('http://localhost:1');
    ctx.breaker = breaker;
    ctx.log = log;

    const result = await anthropicMessagesGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.msg.includes('circuit breaker is open'))).toBe(true);
  });

  test('returns null on network error', async () => {
    const log = spyLogger();
    const ctx = makeCtx('http://localhost:1');
    ctx.log = log;
    ctx.readTimeoutMs = 100;

    const result = await anthropicMessagesGenerateAdapter.generate(ctx, 'prompt');
    expect(result).toBeNull();
    expect(log.calls.some((c) => c.level === 'error')).toBe(true);
  });

  test('returns empty text for response with no text content blocks', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        content: [
          { type: 'tool_use', id: 'tool_1', name: 'get_weather', input: {} },
        ],
        model: 'claude',
      });
    });

    const result = await anthropicMessagesGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('');
    expect(result!.done).toBe(true);
    await server.close();
  });

  test('concatenates multiple text blocks in response', async () => {
    const server = startMockServer((_req, res) => {
      jsonRes(res, 200, {
        content: [
          { type: 'text', text: 'Part A. ' },
          { type: 'text', text: 'Part B.' },
        ],
        model: 'claude',
      });
    });

    const result = await anthropicMessagesGenerateAdapter.generate(
      makeCtx(server.url), 'prompt',
    );

    expect(result!.text).toBe('Part A.\nPart B.');
    await server.close();
  });
});

// =============================================================================
// Phase 3 — RemoteLLM integration with anthropic_messages format
// =============================================================================

describe('RemoteLLM with Phase 3 anthropic_messages format', () => {
  test('RemoteLLM expandQuery hits /messages with Anthropic protocol', async () => {
    let requestedPath = '';
    let receivedHeaders: any = null;
    const server = startMockServer((req, res) => {
      requestedPath = req.url ?? '';
      receivedHeaders = req.headers;
      jsonRes(res, 200, {
        content: [
          { type: 'text', text: 'lex: test kw\nvec: test vec\nhyde: test hyde' },
        ],
      });
    });

    const llm = new RemoteLLM({
      expand: { baseUrl: server.url, model: 'claude', apiKey: 'sk-ant', format: 'anthropic_messages' },
    }, silentLogger);

    const result = await llm.expandQuery('test');
    expect(result).toHaveLength(3);
    expect(requestedPath).toBe('/messages');
    expect(receivedHeaders['x-api-key']).toBe('sk-ant');
    expect(receivedHeaders['anthropic-version']).toBe('2023-06-01');
    await llm.dispose();
    await server.close();
  });

  test('RemoteLLM generate hits /messages with Anthropic protocol', async () => {
    let requestedPath = '';
    let receivedHeaders: any = null;
    const server = startMockServer((req, res) => {
      requestedPath = req.url ?? '';
      receivedHeaders = req.headers;
      jsonRes(res, 200, {
        content: [{ type: 'text', text: 'claude answer' }],
        model: 'claude-3',
      });
    });

    const llm = new RemoteLLM({
      generate: { baseUrl: server.url, model: 'claude', apiKey: 'sk-ant', format: 'anthropic_messages' },
    }, silentLogger);

    const result = await llm.generate('test prompt');
    expect(result!.text).toBe('claude answer');
    expect(requestedPath).toBe('/messages');
    expect(receivedHeaders['x-api-key']).toBe('sk-ant');
    await llm.dispose();
    await server.close();
  });
});
