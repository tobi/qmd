/**
 * Tests for Reranking Service
 * Target coverage: 85%+ (core algorithms)
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { rerank } from './reranking.ts';
import { createTestDb, cleanupDb } from '../../tests/fixtures/helpers/test-db.ts';
import { mockOllamaComplete } from '../../tests/fixtures/helpers/mock-ollama.ts';

describe('rerank', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('reranks documents and returns sorted results', async () => {
    global.fetch = mockOllamaComplete({
      generateResponse: 'yes',
      generateLogprobs: [
        { token: 'yes', logprob: -0.1 },
      ],
    });

    const documents = [
      { file: '/doc1.md', text: 'Content 1' },
      { file: '/doc2.md', text: 'Content 2' },
    ];

    const results = await rerank('test query', documents, 'test-model', db);

    expect(results).toHaveLength(2);
    expect(results[0]).toHaveProperty('file');
    expect(results[0]).toHaveProperty('score');
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
  });

  test('handles empty document list', async () => {
    const results = await rerank('query', [], 'test-model', db);
    expect(results).toHaveLength(0);
  });

  test('caches reranking results', async () => {
    let callCount = 0;
    global.fetch = mockOllamaComplete({
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.1 }],
    });

    // Override to count calls
    const originalFetch = global.fetch;
    global.fetch = async (...args: any[]) => {
      callCount++;
      return originalFetch(...args);
    };

    const documents = [{ file: '/doc.md', text: 'Content' }];

    // First call
    await rerank('query', documents, 'test-model', db);
    const firstCallCount = callCount;

    // Second call with same params - should use cache (no new calls)
    await rerank('query', documents, 'test-model', db);

    // Should not have made additional calls (fully cached)
    expect(callCount).toBe(firstCallCount);
  });

  test('handles yes response correctly', async () => {
    global.fetch = mockOllamaComplete({
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.5 }],
    });

    const documents = [{ file: '/doc.md', text: 'Relevant content' }];
    const results = await rerank('query', documents, 'test-model');

    expect(results[0].score).toBeGreaterThan(0);
    expect(results[0].score).toBeLessThanOrEqual(1);
  });

  test('handles no response correctly', async () => {
    global.fetch = mockOllamaComplete({
      generateResponse: 'no',
      generateLogprobs: [{ token: 'no', logprob: -0.5 }],
    });

    const documents = [{ file: '/doc.md', text: 'Irrelevant content' }];
    const results = await rerank('query', documents, 'test-model');

    // 'no' scores should be lower (scaled by 0.3)
    expect(results[0].score).toBeLessThan(0.5);
  });
});
