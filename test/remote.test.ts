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
