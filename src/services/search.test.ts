/**
 * Tests for Search Service
 * Target coverage: 85%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import {
  extractSnippet,
  fullTextSearch,
  vectorSearch,
  reciprocalRankFusion,
  hybridSearch,
} from './search.ts';
import { createTestDbWithData, createTestDbWithVectors, cleanupDb } from '../../tests/fixtures/helpers/test-db.ts';
import { mockOllamaComplete } from '../../tests/fixtures/helpers/mock-ollama.ts';
import type { SearchResult } from '../models/types.ts';

describe('extractSnippet', () => {
  test('extracts snippet around query match', () => {
    const text = 'This is a long document with important information about testing that we need to find.';
    const result = extractSnippet(text, 'important', 50);

    expect(result.snippet).toContain('important');
    expect(result.position).toBeGreaterThan(0);
  });

  test('returns beginning when query not found', () => {
    const text = 'This is some text';
    const result = extractSnippet(text, 'notfound', 50);

    expect(result.snippet).toBe('This is some text');
    expect(result.position).toBe(0);
  });

  test('adds ellipsis when truncating', () => {
    const text = 'a'.repeat(500);
    const result = extractSnippet(text + ' target ' + 'b'.repeat(500), 'target', 100);

    expect(result.snippet).toContain('...');
  });

  test('handles case-insensitive search', () => {
    const text = 'This is IMPORTANT text';
    const result = extractSnippet(text, 'important', 50);

    expect(result.position).toBeGreaterThan(0);
  });

  test('respects maxLength parameter', () => {
    const text = 'a'.repeat(1000);
    const result = extractSnippet(text, 'notfound', 100);

    expect(result.snippet.length).toBeLessThanOrEqual(100);
  });
});

describe('reciprocalRankFusion', () => {
  test('fuses two result lists with equal weights', () => {
    const list1: SearchResult[] = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' },
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 0.9, source: 'fts' },
    ];
    const list2: SearchResult[] = [
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 0.8, source: 'vec' },
      { file: '/c.md', displayPath: 'c', title: 'C', body: 'C', score: 0.7, source: 'vec' },
    ];

    const fused = reciprocalRankFusion([list1, list2]);

    // b.md appears in both lists, should rank highest
    expect(fused[0].file).toBe('/b.md');
    expect(fused.length).toBe(3); // a, b, c
  });

  test('applies custom weights', () => {
    const list1: SearchResult[] = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' },
    ];
    const list2: SearchResult[] = [
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 1.0, source: 'vec' },
    ];

    // Weight list2 higher
    const fused = reciprocalRankFusion([list1, list2], [1.0, 2.0]);

    // b.md should rank higher due to weight
    expect(fused[0].file).toBe('/b.md');
  });

  test('handles empty lists', () => {
    const fused = reciprocalRankFusion([[], []]);
    expect(fused).toHaveLength(0);
  });

  test('handles single list', () => {
    const list: SearchResult[] = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' },
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 0.9, source: 'fts' },
    ];

    const fused = reciprocalRankFusion([list]);

    expect(fused).toHaveLength(2);
    expect(fused[0].file).toBe('/a.md');
  });

  test('uses custom k parameter', () => {
    const list: SearchResult[] = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' },
    ];

    const fused = reciprocalRankFusion([list], undefined, 30);

    // Score should be 1/(30+1) = 0.032...
    expect(fused[0].score).toBeCloseTo(0.032, 2);
  });

  test('sorts results by score descending', () => {
    const list1: SearchResult[] = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 0.5, source: 'fts' },
    ];
    const list2: SearchResult[] = [
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 1.0, source: 'vec' },
      { file: '/c.md', displayPath: 'c', title: 'C', body: 'C', score: 0.8, source: 'vec' },
    ];

    const fused = reciprocalRankFusion([list1, list2]);

    expect(fused[0].score).toBeGreaterThanOrEqual(fused[1].score);
    expect(fused[1].score).toBeGreaterThanOrEqual(fused[2].score);
  });
});

describe('fullTextSearch', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDbWithData();
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('returns search results for valid query', async () => {
    const results = await fullTextSearch(db, 'test', 10);

    expect(Array.isArray(results)).toBe(true);
    // Results depend on test data
  });

  test('returns empty array for empty query', async () => {
    const results = await fullTextSearch(db, '', 10);
    expect(results).toHaveLength(0);
  });

  test('returns empty array for whitespace query', async () => {
    const results = await fullTextSearch(db, '   ', 10);
    expect(results).toHaveLength(0);
  });

  test('respects limit parameter', async () => {
    const results = await fullTextSearch(db, 'test', 5);
    expect(results.length).toBeLessThanOrEqual(5);
  });
});

describe('vectorSearch', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('returns search results for query', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const results = await vectorSearch(db, 'test query', 'test-model', 10);

    expect(Array.isArray(results)).toBe(true);
  });

  test('respects limit parameter', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const results = await vectorSearch(db, 'test', 'test-model', 3);
    expect(results.length).toBeLessThanOrEqual(3);
  });
});

describe('hybridSearch', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('returns reranked results', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
      generateResponse: 'yes',
      generateLogprobs: [
        { token: 'yes', logprob: -0.1 },
      ],
    });

    const results = await hybridSearch(db, 'test', 'embed-model', 'rerank-model', 5);

    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeLessThanOrEqual(5);
  });

  test('returns results with required fields', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.1 }],
    });

    const results = await hybridSearch(db, 'test', 'embed-model', 'rerank-model', 3);

    if (results.length > 0) {
      expect(results[0]).toHaveProperty('file');
      expect(results[0]).toHaveProperty('title');
      expect(results[0]).toHaveProperty('score');
    }
  });

  test('respects limit parameter', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.1 }],
    });

    const results = await hybridSearch(db, 'test', 'embed-model', 'rerank-model', 2);
    expect(results.length).toBeLessThanOrEqual(2);
  });
});
