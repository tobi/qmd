/**
 * Tests for Indexing Service
 * Target coverage: 70%+ (core functionality, integration tested separately)
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { indexFiles } from './indexing.ts';
import { createTestDb, cleanupDb } from '../../tests/fixtures/helpers/test-db.ts';
import { resolve } from 'path';

describe('indexFiles', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('indexes markdown files from fixtures', async () => {
    const fixturesPath = resolve(import.meta.dir, '../../tests/fixtures/markdown');

    const result = await indexFiles(db, '*.md', fixturesPath);

    expect(result).toHaveProperty('indexed');
    expect(result).toHaveProperty('updated');
    expect(result).toHaveProperty('unchanged');
    expect(result).toHaveProperty('removed');
    expect(result).toHaveProperty('needsEmbedding');

    // Should find at least some fixtures
    const total = result.indexed + result.updated + result.unchanged;
    expect(total).toBeGreaterThan(0);
  });

  test('handles re-indexing with unchanged files', async () => {
    const fixturesPath = resolve(import.meta.dir, '../../tests/fixtures/markdown');

    // First index
    const first = await indexFiles(db, '*.md', fixturesPath);
    expect(first.indexed).toBeGreaterThan(0);

    // Second index - should find unchanged files
    const second = await indexFiles(db, '*.md', fixturesPath);
    expect(second.unchanged).toBeGreaterThan(0);
    expect(second.indexed).toBe(0);
  });

  test('returns zero counts when no files match pattern', async () => {
    const fixturesPath = resolve(import.meta.dir, '../../tests/fixtures/markdown');

    const result = await indexFiles(db, 'nonexistent-*.xyz', fixturesPath);

    expect(result.indexed).toBe(0);
    expect(result.updated).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.removed).toBe(0);
  });

  test('creates collection for pwd and glob pattern', async () => {
    const fixturesPath = resolve(import.meta.dir, '../../tests/fixtures/markdown');

    await indexFiles(db, '*.md', fixturesPath);

    // Verify collection was created
    const collections = db.prepare(`SELECT * FROM collections`).all();
    expect(collections.length).toBeGreaterThan(0);
  });

  test('reports files needing embedding', async () => {
    const fixturesPath = resolve(import.meta.dir, '../../tests/fixtures/markdown');

    const result = await indexFiles(db, '*.md', fixturesPath);

    // New files should need embedding
    if (result.indexed > 0) {
      expect(result.needsEmbedding).toBeGreaterThan(0);
    }
  });
});
