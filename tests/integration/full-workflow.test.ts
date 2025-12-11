/**
 * Full Workflow Integration Test
 * Tests: add → embed → search pipeline
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { resolve } from 'path';
import { createTestDb, cleanupDb } from '../fixtures/helpers/test-db.ts';
import { mockOllamaComplete } from '../fixtures/helpers/mock-ollama.ts';
import { indexFiles } from '../../src/services/indexing.ts';
import { embedDocument } from '../../src/services/embedding.ts';
import { fullTextSearch, vectorSearch, hybridSearch } from '../../src/services/search.ts';
import { DocumentRepository, VectorRepository, CollectionRepository } from '../../src/database/repositories/index.ts';

describe('Full Workflow Integration', () => {
  let db: Database;
  const fixturesPath = resolve(import.meta.dir, '../fixtures/markdown');

  beforeEach(() => {
    db = createTestDb();
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('complete workflow: index → embed → search', async () => {
    // Step 1: Index markdown files
    const indexResult = await indexFiles(db, '*.md', fixturesPath);

    expect(indexResult.indexed).toBeGreaterThan(0);
    expect(indexResult.needsEmbedding).toBeGreaterThan(0);

    // Step 2: Get documents and embed them
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    const docRepo = new DocumentRepository(db);
    const vecRepo = new VectorRepository(db);

    // Get collection and its documents
    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    expect(collections.length).toBeGreaterThan(0);

    const docs = docRepo.findByCollection(collections[0].id);
    expect(docs.length).toBeGreaterThan(0);

    // Embed first document
    const doc = docs[0];
    const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
    await embedDocument(db, doc.hash, chunks, 'test-model');

    // Verify embedding was created
    const vectors = vecRepo.findByHash(doc.hash);
    expect(vectors.length).toBeGreaterThan(0);

    // Step 3: Full-text search
    const ftsResults = await fullTextSearch(db, 'test', 10);
    expect(Array.isArray(ftsResults)).toBe(true);

    // Step 4: Vector search
    const vecResults = await vectorSearch(db, 'test query', 'test-model', 10);
    expect(Array.isArray(vecResults)).toBe(true);
  });

  test('workflow handles multiple documents', async () => {
    // Index all fixtures
    const indexResult = await indexFiles(db, '*.md', fixturesPath);
    expect(indexResult.indexed).toBeGreaterThan(1);

    // Mock embeddings for multiple documents
    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
    });

    // Embed all documents
    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    for (const doc of docs) {
      const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
      await embedDocument(db, doc.hash, chunks, 'test-model');
    }

    // Verify all have embeddings
    const vecRepo = new VectorRepository(db);
    const totalDocs = vecRepo.countDocumentsWithEmbeddings();
    expect(totalDocs).toBe(docs.length);

    // Search should return results
    const results = await vectorSearch(db, 'test', 'test-model', 10);
    expect(results.length).toBeGreaterThan(0);
  });

  test('hybrid search integrates FTS and vector results', async () => {
    // Index and embed
    await indexFiles(db, '*.md', fixturesPath);

    global.fetch = mockOllamaComplete({
      embeddings: [Array(128).fill(0.1)],
      generateResponse: 'yes',
      generateLogprobs: [{ token: 'yes', logprob: -0.1 }],
    });

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    if (docs.length > 0) {
      const doc = docs[0];
      const chunks = [{ text: doc.body, pos: 0, title: doc.title }];
      await embedDocument(db, doc.hash, chunks, 'test-model');
    }

    // Hybrid search combines FTS + vector + reranking
    const results = await hybridSearch(db, 'test', 'embed-model', 'rerank-model', 5);
    expect(Array.isArray(results)).toBe(true);
  });
});
