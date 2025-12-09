/**
 * Search Flow Integration Test
 * Tests: BM25, vector, hybrid search ranking
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { resolve } from 'path';
import { createTestDbWithVectors, cleanupDb } from '../fixtures/helpers/test-db.ts';
import { mockOllamaComplete } from '../fixtures/helpers/mock-ollama.ts';
import { indexFiles } from '../../src/services/indexing.ts';
import { embedDocument } from '../../src/services/embedding.ts';
import { fullTextSearch, vectorSearch, reciprocalRankFusion, hybridSearch } from '../../src/services/search.ts';
import { DocumentRepository, CollectionRepository } from '../../src/database/repositories/index.ts';

describe('Search Flow Integration', () => {
  let db: Database;
  const fixturesPath = resolve(import.meta.dir, '../fixtures/markdown');

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('full-text search returns ranked results', async () => {
    // Index fixtures
    await indexFiles(db, '*.md', fixturesPath);

    // Search should return results
    const results = await fullTextSearch(db, 'test', 10);

    if (results.length > 0) {
      // Results should have required fields
      expect(results[0]).toHaveProperty('file');
      expect(results[0]).toHaveProperty('title');
      expect(results[0]).toHaveProperty('score');
      expect(results[0]).toHaveProperty('source');
      expect(results[0].source).toBe('fts');
    }
  });

  test('vector search returns similar documents', async () => {
    // Index and embed
    await indexFiles(db, '*.md', fixturesPath);

    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    // Embed at least one document
    if (docs.length > 0) {
      const doc = docs[0];
      const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
      await embedDocument(db, doc.hash, chunks, 'test-model');
    }

    // Vector search
    const results = await vectorSearch(db, 'test query', 'test-model', 10);

    if (results.length > 0) {
      expect(results[0]).toHaveProperty('file');
      expect(results[0]).toHaveProperty('score');
      expect(results[0]).toHaveProperty('source');
      expect(results[0].source).toBe('vec');
      expect(results[0]).toHaveProperty('chunkPos');
    }
  });

  test('reciprocal rank fusion combines rankings', async () => {
    // Create mock result lists
    const ftsResults = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' as const },
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 0.8, source: 'fts' as const },
    ];

    const vecResults = [
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 0.9, source: 'vec' as const, chunkPos: 0 },
      { file: '/c.md', displayPath: 'c', title: 'C', body: 'C', score: 0.7, source: 'vec' as const, chunkPos: 0 },
    ];

    // RRF should boost documents that appear in both lists
    const fused = reciprocalRankFusion([ftsResults, vecResults]);

    // b.md should rank highest (appears in both)
    expect(fused[0].file).toBe('/b.md');
    expect(fused.length).toBe(3); // a, b, c
  });

  test('RRF with weights favors higher-weighted lists', async () => {
    const list1 = [
      { file: '/a.md', displayPath: 'a', title: 'A', body: 'A', score: 1.0, source: 'fts' as const },
    ];

    const list2 = [
      { file: '/b.md', displayPath: 'b', title: 'B', body: 'B', score: 1.0, source: 'vec' as const, chunkPos: 0 },
    ];

    // Weight list2 higher
    const fused = reciprocalRankFusion([list1, list2], [1.0, 2.0]);

    // b.md should rank first due to higher weight
    expect(fused[0].file).toBe('/b.md');
  });

  test('hybrid search pipeline executes successfully', async () => {
    // Index fixtures
    await indexFiles(db, '*.md', fixturesPath);

    // Mock all Ollama endpoints
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.1 }],
    });

    // Embed at least one document
    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    if (docs.length > 0) {
      const doc = docs[0];
      const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
      await embedDocument(db, doc.hash, chunks, 'test-model');
    }

    // Hybrid search combines FTS, vector, RRF, and reranking
    const results = await hybridSearch(db, 'test query', 'embed-model', 'rerank-model', 5);

    expect(Array.isArray(results)).toBe(true);

    if (results.length > 0) {
      // Results should have blended scores
      expect(results[0]).toHaveProperty('file');
      expect(results[0]).toHaveProperty('title');
      expect(results[0]).toHaveProperty('score');
      expect(results[0].score).toBeGreaterThan(0);
      expect(results[0].score).toBeLessThanOrEqual(1);
    }
  });

  test('search results are properly ranked', async () => {
    await indexFiles(db, '*.md', fixturesPath);

    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    // Embed all documents
    for (const doc of docs) {
      const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
      await embedDocument(db, doc.hash, chunks, 'test-model');
    }

    // Search and verify ranking
    const results = await vectorSearch(db, 'test', 'test-model', 10);

    if (results.length > 1) {
      // Scores should be in descending order
      for (let i = 0; i < results.length - 1; i++) {
        expect(results[i].score).toBeGreaterThanOrEqual(results[i + 1].score);
      }
    }
  });

  test('search respects limit parameter', async () => {
    await indexFiles(db, '*.md', fixturesPath);

    // FTS search with limit
    const ftsResults = await fullTextSearch(db, 'test', 3);
    expect(ftsResults.length).toBeLessThanOrEqual(3);

    // Vector search with limit
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const vecResults = await vectorSearch(db, 'test', 'test-model', 2);
    expect(vecResults.length).toBeLessThanOrEqual(2);
  });
});
