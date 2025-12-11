/**
 * Tests for Ollama API service
 * Target coverage: 85%+
 */

import { describe, test, expect, beforeEach, mock } from 'bun:test';
import { ensureModelAvailable, getEmbedding, generateCompletion } from './ollama.ts';
import { mockOllamaEmbed, mockOllamaGenerate, mockOllamaModelCheck, mockOllamaPull } from '../../tests/fixtures/helpers/mock-ollama.ts';

describe('ensureModelAvailable', () => {
  beforeEach(() => {
    // Reset global fetch before each test
    global.fetch = fetch;
  });

  test('does nothing when model exists', async () => {
    global.fetch = mockOllamaModelCheck(true);

    await expect(ensureModelAvailable('test-model')).resolves.toBeUndefined();
  });

  test('pulls model when not found', async () => {
    let showCalled = false;
    let pullCalled = false;

    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/show')) {
        showCalled = true;
        return Promise.resolve({
          ok: false,
          status: 404,
          text: () => Promise.resolve('not found'),
        } as Response);
      }
      if (typeof url === 'string' && url.includes('/api/pull')) {
        pullCalled = true;
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'success' }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await ensureModelAvailable('new-model');

    expect(showCalled).toBe(true);
    expect(pullCalled).toBe(true);
  });

  test('throws error when pull fails', async () => {
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/show')) {
        return Promise.resolve({ ok: false, status: 404 } as Response);
      }
      if (typeof url === 'string' && url.includes('/api/pull')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          text: () => Promise.resolve('pull failed'),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await expect(ensureModelAvailable('bad-model')).rejects.toThrow('Failed to pull model');
  });
});

describe('getEmbedding', () => {
  beforeEach(() => {
    global.fetch = fetch;
  });

  test('returns embedding for query', async () => {
    const mockEmbedding = [0.1, 0.2, 0.3];
    global.fetch = mockOllamaEmbed([mockEmbedding]);

    const result = await getEmbedding('test query', 'test-model', true);

    expect(result).toEqual(mockEmbedding);
  });

  test('returns embedding for document', async () => {
    const mockEmbedding = [0.4, 0.5, 0.6];
    global.fetch = mockOllamaEmbed([mockEmbedding]);

    const result = await getEmbedding('test content', 'test-model', false, 'Test Title');

    expect(result).toEqual(mockEmbedding);
  });

  test('formats query with search_query prefix', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/embed')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ embeddings: [[0.1]] }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await getEmbedding('my query', 'test-model', true);

    expect(requestBody.input).toBe('search_query: my query');
  });

  test('formats document with title', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/embed')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ embeddings: [[0.1]] }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await getEmbedding('content', 'test-model', false, 'My Title');

    expect(requestBody.input).toBe('search_document: My Title\n\ncontent');
  });

  test('formats document without title', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/embed')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ embeddings: [[0.1]] }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await getEmbedding('content', 'test-model', false);

    expect(requestBody.input).toBe('search_document: content');
  });

  test('retries once when model not found', async () => {
    let embedAttempts = 0;
    let showCalled = false;

    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/embed')) {
        embedAttempts++;
        if (embedAttempts === 1) {
          return Promise.resolve({
            ok: false,
            status: 404,
            text: () => Promise.resolve('model not found'),
          } as Response);
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ embeddings: [[0.1]] }),
        } as Response);
      }
      if (typeof url === 'string' && url.includes('/api/show')) {
        showCalled = true;
        return Promise.resolve({ ok: true, status: 200 } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    const result = await getEmbedding('test', 'test-model');

    expect(embedAttempts).toBe(2);
    expect(showCalled).toBe(true);
    expect(result).toEqual([0.1]);
  });

  test('throws error on API failure', async () => {
    global.fetch = mock((url: string) => {
      if (typeof url === 'string' && url.includes('/api/embed')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          text: () => Promise.resolve('server error'),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await expect(getEmbedding('test', 'test-model', false, undefined, true)).rejects.toThrow('Ollama API error');
  });
});

describe('generateCompletion', () => {
  beforeEach(() => {
    global.fetch = fetch;
  });

  test('generates completion with default options', async () => {
    global.fetch = mockOllamaGenerate('Generated response');

    const result = await generateCompletion('test-model', 'test prompt');

    expect(result.response).toBe('Generated response');
  });

  test('includes logprobs when requested', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/generate')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ response: 'test', logprobs: [] }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await generateCompletion('test-model', 'prompt', false, true);

    expect(requestBody.logprobs).toBe(true);
  });

  test('includes num_predict when provided', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/generate')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ response: 'test' }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await generateCompletion('test-model', 'prompt', false, false, 100);

    expect(requestBody.options.num_predict).toBe(100);
  });

  test('uses raw mode when specified', async () => {
    let requestBody: any;
    global.fetch = mock((url: string, options?: any) => {
      if (typeof url === 'string' && url.includes('/api/generate')) {
        requestBody = JSON.parse(options?.body || '{}');
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ response: 'test' }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await generateCompletion('test-model', 'prompt', true);

    expect(requestBody.raw).toBe(true);
  });

  test('throws error on API failure', async () => {
    global.fetch = mock((url: string) => {
      if (typeof url === 'string' && url.includes('/api/generate')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          text: () => Promise.resolve('server error'),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404 } as Response);
    });

    await expect(generateCompletion('test-model', 'prompt')).rejects.toThrow('Ollama API error');
  });
});
