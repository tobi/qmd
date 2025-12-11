/**
 * Indexing Flow Integration Test
 * Tests: add, update, remove documents
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { resolve } from 'path';
import { createTestDb, cleanupDb } from '../fixtures/helpers/test-db.ts';
import { indexFiles } from '../../src/services/indexing.ts';
import { DocumentRepository, CollectionRepository } from '../../src/database/repositories/index.ts';

describe('Indexing Flow Integration', () => {
  let db: Database;
  const fixturesPath = resolve(import.meta.dir, '../fixtures/markdown');

  beforeEach(() => {
    db = createTestDb();
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('indexes new files and creates collection', async () => {
    const result = await indexFiles(db, '*.md', fixturesPath);

    // Should index fixture files
    expect(result.indexed).toBeGreaterThan(0);
    expect(result.updated).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.removed).toBe(0);

    // Collection should be created
    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    expect(collections.length).toBeGreaterThan(0);

    // Documents should be active
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);
    expect(docs.length).toBe(result.indexed);
  });

  test('detects unchanged files on re-index', async () => {
    // First index
    const first = await indexFiles(db, '*.md', fixturesPath);
    expect(first.indexed).toBeGreaterThan(0);

    // Second index - no changes
    const second = await indexFiles(db, '*.md', fixturesPath);
    expect(second.indexed).toBe(0);
    expect(second.unchanged).toBe(first.indexed);
    expect(second.updated).toBe(0);
    expect(second.removed).toBe(0);
  });

  test('creates unique display paths for documents', async () => {
    await indexFiles(db, '*.md', fixturesPath);

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);

    // All active documents should have display_path
    const displayPaths = docs
      .map(d => d.display_path)
      .filter(p => p && p !== '');

    expect(displayPaths.length).toBeGreaterThan(0);

    // Display paths should be unique
    const uniquePaths = new Set(displayPaths);
    expect(uniquePaths.size).toBe(displayPaths.length);
  });

  test('handles multiple glob patterns', async () => {
    // Index with wildcard
    const result = await indexFiles(db, '**/*.md', fixturesPath);
    expect(result.indexed).toBeGreaterThan(0);

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAll();
    const docRepo = new DocumentRepository(db);
    const docs = docRepo.findByCollection(collections[0].id);
    expect(docs.length).toBe(result.indexed);
  });

  test('reports documents needing embeddings', async () => {
    const result = await indexFiles(db, '*.md', fixturesPath);

    // New documents should need embeddings
    expect(result.needsEmbedding).toBe(result.indexed);
  });

  test('maintains collection statistics', async () => {
    await indexFiles(db, '*.md', fixturesPath);

    const collectionRepo = new CollectionRepository(db);
    const collections = collectionRepo.findAllWithCounts();

    expect(collections.length).toBeGreaterThan(0);

    const collection = collections[0];
    expect(collection.active_count).toBeGreaterThan(0);
    expect(collection.created_at).toBeDefined();
  });

  test('handles empty glob pattern results', async () => {
    const result = await indexFiles(db, 'nonexistent-*.xyz', fixturesPath);

    expect(result.indexed).toBe(0);
    expect(result.updated).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.removed).toBe(0);
  });
});
