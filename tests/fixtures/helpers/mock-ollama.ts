/**
 * Ollama API mocking utilities
 * Provides mock functions for Ollama API endpoints used in tests
 */

import { mock } from 'bun:test';

/**
 * Mock Ollama embedding API
 * @param embeddings - Array of embedding vectors to return
 * @returns Mocked fetch function
 */
export function mockOllamaEmbed(embeddings: number[][]) {
  return mock((url: string, options?: any) => {
    if (typeof url === 'string' && url.includes('/api/embed')) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ embeddings }),
      });
    }

    // For other URLs, return 404
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    });
  });
}

/**
 * Mock Ollama generation API (for reranking)
 * @param response - Response text to return
 * @param logprobs - Array of log probabilities (optional)
 * @returns Mocked fetch function
 */
export function mockOllamaGenerate(response: string, logprobs?: any[]) {
  return mock((url: string, options?: any) => {
    if (typeof url === 'string' && url.includes('/api/generate')) {
      const body = options?.body ? JSON.parse(options.body) : {};

      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          response,
          logprobs: logprobs || [],
          done: true,
          model: body.model || 'test-model',
          created_at: new Date().toISOString(),
        }),
      });
    }

    // For other URLs, return 404
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    });
  });
}

/**
 * Mock Ollama model check (show endpoint)
 * @param exists - Whether the model exists
 * @returns Mocked fetch function
 */
export function mockOllamaModelCheck(exists: boolean) {
  return mock((url: string, options?: any) => {
    if (typeof url === 'string' && url.includes('/api/show')) {
      return Promise.resolve({
        ok: exists,
        status: exists ? 200 : 404,
        json: () => exists
          ? Promise.resolve({
              modelfile: 'test-modelfile',
              parameters: 'test-parameters',
              template: 'test-template',
            })
          : Promise.resolve({ error: 'model not found' }),
      });
    }

    // For other URLs, return 404
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    });
  });
}

/**
 * Mock Ollama pull endpoint (for model downloading)
 * @param success - Whether the pull succeeds
 * @returns Mocked fetch function
 */
export function mockOllamaPull(success: boolean = true) {
  return mock((url: string, options?: any) => {
    if (typeof url === 'string' && url.includes('/api/pull')) {
      return Promise.resolve({
        ok: success,
        status: success ? 200 : 500,
        json: () => success
          ? Promise.resolve({ status: 'success' })
          : Promise.resolve({ error: 'pull failed' }),
      });
    }

    // For other URLs, return 404
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    });
  });
}

/**
 * Create a comprehensive mock for all Ollama endpoints
 * @param config - Configuration object for each endpoint
 * @returns Mocked fetch function
 */
export function mockOllamaComplete(config: {
  embeddings?: number[][];
  generateResponse?: string;
  generateLogprobs?: any[];
  modelExists?: boolean;
  pullSuccess?: boolean;
}) {
  return mock((url: string, options?: any) => {
    if (typeof url !== 'string') {
      return Promise.resolve({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: 'Invalid URL' }),
      });
    }

    // Embedding endpoint
    if (url.includes('/api/embed')) {
      const embeddings = config.embeddings || [[0.1, 0.2, 0.3]];
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ embeddings }),
      });
    }

    // Generation endpoint
    if (url.includes('/api/generate')) {
      const response = config.generateResponse || 'yes';
      const logprobs = config.generateLogprobs || [];
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ response, logprobs, done: true }),
      });
    }

    // Model check endpoint
    if (url.includes('/api/show')) {
      const exists = config.modelExists !== undefined ? config.modelExists : true;
      return Promise.resolve({
        ok: exists,
        status: exists ? 200 : 404,
        json: () => exists
          ? Promise.resolve({ modelfile: 'test' })
          : Promise.resolve({ error: 'model not found' }),
      });
    }

    // Pull endpoint
    if (url.includes('/api/pull')) {
      const success = config.pullSuccess !== undefined ? config.pullSuccess : true;
      return Promise.resolve({
        ok: success,
        status: success ? 200 : 500,
        json: () => success
          ? Promise.resolve({ status: 'success' })
          : Promise.resolve({ error: 'pull failed' }),
      });
    }

    // Default 404 for unknown endpoints
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    });
  });
}

/**
 * Restore global fetch after mocking
 */
export function restoreFetch(): void {
  // Bun's mock() should auto-restore, but this is a safety measure
  if ((global.fetch as any).mockRestore) {
    (global.fetch as any).mockRestore();
  }
}
